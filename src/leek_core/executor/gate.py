#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gate.io 交易执行模块
"""

import time
import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional

from cachetools import TTLCache, cached
from leek_core.adapts import GateAdapter

from leek_core.models import Order, OrderStatus, OrderUpdateMessage
from leek_core.models import PositionSide as PS, OrderType as OT, TradeMode, TradeInsType, Field, FieldType
from leek_core.utils import get_logger, DateTimeUtils, retry
from .base import Executor

logger = get_logger(__name__)


class GateRestExecutor(Executor):
    """
    Gate.io 交易所REST API执行器，通过REST API进行下单、撤单等操作。
    使用内部轮询机制查询订单状态并触发回调。
    """
    display_name = "Gate.io REST"
    init_params: List['Field'] = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, default="", required=True),
        Field(name="secret_key", label="API Secret Key", type=FieldType.PASSWORD, default="", required=True),
        Field(name="testnet", label="测试网", type=FieldType.BOOLEAN, default=False, required=False, description="是否使用测试网"),
        Field(name="slippage_level", label="允许滑档", type=FieldType.INT, default=5, required=True, description="限价交易允许滑档数"),
    ]

    # 映射表
    __Side_Map = {
        PS.LONG: "buy",
        PS.SHORT: "sell",
    }
    __Order_Type_Map = {
        OT.LimitOrder: "limit",
        OT.MarketOrder: "market",
    }
    __Status_Map = {
        "open": OrderStatus.SUBMITTED,
        "closed": OrderStatus.FILLED,
        "cancelled": OrderStatus.CANCELED,
        "expired": OrderStatus.EXPIRED,
    }

    def __init__(self, api_key, secret_key, testnet=False, slippage_level=5, **kwargs):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = bool(testnet)
        self.slippage_level = int(slippage_level)
        self.adapter = GateAdapter(api_key=api_key, secret_key=secret_key, testnet=testnet)
        
        self._pending_orders: Dict[str, Dict] = {}  # key: text (clientOrderId), value: 订单信息
        self._orders_lock = threading.RLock()
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pull_interval = 1.0

    def send_order(self, orders: Order | List[Order]):
        """
        组装参数并通过REST API下单，支持现货和合约
        """
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            # Gate.io 现货不支持做空
            if order.ins_type == TradeInsType.SPOT and order.side == PS.SHORT:
                raise ValueError("Gate.io 现货不支持做空操作")
            
            # 构建交易对符号
            currency_pair = self.adapter.build_symbol(order.symbol, order.quote_currency)
            
            # 判断是现货还是合约
            is_futures = order.ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES]
            
            # 数量精度校验
            quantity = self._check_sz(order, is_futures=is_futures)
            
            # 订单类型
            order_type = GateRestExecutor.__Order_Type_Map.get(order.order_type, "limit")
            
            # 价格处理
            price = None
            if order_type == "limit":
                if order.order_price is not None:
                    # 限价单需要价格精度校验
                    price = self._format_price(currency_pair, order.order_price, is_futures=is_futures)
                else:
                    # 限价单如果没有价格，从订单簿获取
                    orderbook = self._get_book(currency_pair, is_futures=is_futures)
                    if order.side == PS.LONG:
                        price = orderbook["asks"][-1][0]
                    else:
                        price = orderbook["bids"][-1][0]
                    # 格式化价格
                    price = self._format_price(currency_pair, Decimal(price), is_futures=is_futures)
            
            # 调用REST API下单
            try:
                if is_futures:
                    # 合约下单
                    result = self._send_futures_order(order, currency_pair, quantity, price, order_type)
                else:
                    # 现货下单
                    result = self._send_spot_order(order, currency_pair, quantity, price, order_type)
                
                if not result or result.get("code") != "0":
                    error_msg = result.get("msg", "未知错误") if result else "无响应"
                    logger.error(f"下单失败: {error_msg}, 订单: {order.order_id}, {result}")
                    raise RuntimeError(f"下单失败: {error_msg}")
                
                # 下单成功，添加到待查询列表
                order_data = result.get("data", [{}])[0]
                ord_id = order_data.get("id")
                logger.info(f"下单完成: {order.order_id}, 交易所订单ID: {ord_id}")
                with self._orders_lock:
                    self._pending_orders[order.order_id] = {
                        "currency_pair": currency_pair,
                        "order_id": ord_id,
                        "order": order,
                        "is_futures": is_futures,
                    }
                logger.info(f"下单成功: {order.order_id}, 交易所订单ID: {ord_id}")
            except Exception as e:
                logger.error(f"下单异常: {e}, 订单: {order.order_id}", exc_info=True)
                raise
    
    def _send_spot_order(self, order: Order, currency_pair: str, quantity: Decimal, price: str, order_type: str) -> Dict:
        """
        发送现货订单
        """
        return self.adapter.place_order(
            currency_pair=currency_pair,
            side=GateRestExecutor.__Side_Map[order.side],
            amount=str(quantity),
            price=price,
            text=str(order.order_id),
            order_type=order_type
        )
    
    def _send_futures_order(self, order: Order, contract: str, quantity: Decimal, price: str, order_type: str) -> Dict:
        """
        发送合约订单
        
        Gate.io 合约下单规则：
        - size: 正数为买入（做多），负数为卖出（做空）
        - 平仓时设置 close=true 和 size=0，或者使用 reduce_only=true
        - 市价单：price=0 且 tif=ioc
        """
        # 获取结算货币（usdt 或 btc）
        settle = order.quote_currency.lower() if order.quote_currency else "usdt"
        
        # 设置杠杆倍数（开仓时设置）
        if order.is_open and order.leverage:
            leverage = int(order.leverage)
            if leverage > 0:
                self._set_futures_leverage(settle, contract, leverage, order.trade_mode == TradeMode.CROSS)
        
        # 计算 size（正数买入/负数卖出）
        # side 只表示买卖方向：LONG=买入，SHORT=卖出
        # 开多：is_open=True, side=LONG（买入）→ size 正数
        # 开空：is_open=True, side=SHORT（卖出）→ size 负数
        # 平多：is_open=False, side=SHORT（卖出）→ size 负数 + reduce_only
        # 平空：is_open=False, side=LONG（买入）→ size 正数 + reduce_only
        size = int(quantity) if order.side == PS.LONG else -int(quantity)
        reduce_only = not order.is_open
        
        logger.debug(f"合约{'开仓' if order.is_open else '平仓'}: side={order.side}, quantity={quantity}, size={size}, reduce_only={reduce_only}")
        
        # 构建合约下单参数
        params = {
            "contract": contract,
            "size": size,
            "text": str(order.order_id),
            "reduce_only": reduce_only,
        }
        
        # 价格处理
        if order_type == "market":
            # 市价单：price=0 且 tif=ioc
            params["price"] = "0"
            params["tif"] = "ioc"
        else:
            # 限价单
            params["price"] = price if price else "0"
            params["tif"] = "gtc"
        
        # 如果是平仓且 size 为 0，设置 close=true
        if not order.is_open and size == 0:
            params["close"] = True
            params["size"] = 0
        
        return self.adapter.place_futures_order(settle=settle, **params)
    
    @cached(cache=TTLCache(maxsize=1000, ttl=36000))
    def _set_futures_leverage(self, settle: str, contract: str, leverage: int, cross: bool = False):
        """
        设置合约杠杆倍数（带缓存，36000秒内相同合约+模式不重复设置）
        Gate.io: leverage=0 表示全仓模式，此时用 cross_leverage_limit 指定倍数；
                 leverage>0 表示逐仓模式。
        """
        if cross:
            result = self.adapter.update_futures_leverage(
                settle=settle, contract=contract, leverage=0, cross_leverage_limit=leverage
            )
        else:
            result = self.adapter.update_futures_leverage(settle=settle, contract=contract, leverage=leverage)
        if not result or result.get("code") != "0":
            error_msg = result.get("msg", "未知错误") if result else "无响应"
            logger.warning(f"设置杠杆失败: {error_msg}, contract: {contract}, leverage: {leverage}, cross: {cross}")
        else:
            logger.info(f"设置杠杆成功: contract={contract}, leverage={leverage}, cross={cross}")

    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        通过REST API撤单，支持现货和合约
        """
        try:
            # 从待查询列表中获取订单信息
            with self._orders_lock:
                order_info = self._pending_orders.get(order_id)
            
            if not order_info:
                logger.warning(f"订单 {order_id} 不在待查询列表中，可能已完成或不存在")
                return
            
            currency_pair = order_info["currency_pair"]
            is_futures = order_info.get("is_futures", False)
            exchange_order_id = order_info.get("order_id")
            
            if is_futures:
                # 合约撤单
                settle = currency_pair.split("_")[-1].lower() if "_" in currency_pair else "usdt"
                result = self.adapter.cancel_futures_order(settle=settle, order_id=exchange_order_id)
            else:
                # 现货撤单
                result = self.adapter.cancel_order(currency_pair=currency_pair, text=order_id)
            
            if not result or result.get("code") != "0":
                error_msg = result.get("msg", "未知错误") if result else "无响应"
                logger.error(f"撤单失败: {error_msg}, 订单: {order_id}")
                raise RuntimeError(f"撤单失败: {error_msg}")
            
            logger.info(f"撤单成功: {order_id}")
        except Exception as e:
            logger.error(f"撤单异常: {e}, 订单: {order_id}", exc_info=True)
            raise

    @retry(max_retries=3, retry_interval=0.2)
    def _get_book(self, currency_pair, is_futures=False):
        """获取订单簿"""
        if is_futures:
            # 合约订单簿
            settle = currency_pair.split("_")[-1].lower() if "_" in currency_pair else "usdt"
            orderbook = self.adapter.get_futures_orderbook(settle=settle, contract=currency_pair, limit=self.slippage_level)
        else:
            # 现货订单簿
            orderbook = self.adapter.get_orderbook(currency_pair=currency_pair, limit=self.slippage_level)
        
        if not orderbook or orderbook["code"] != "0":
            raise RuntimeError(f"获取深度失败:{orderbook.get('msg') if orderbook else '无响应'}")
        return orderbook["data"][0]

    def _check_sz(self, order: Order, is_futures=False):
        """
        数量精度校验
        """
        currency_pair = self.adapter.build_symbol(order.symbol, order.quote_currency)
        instrument = self._get_instrument(currency_pair, is_futures=is_futures)
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        
        if is_futures:
            # 合约：使用张数，quanto_multiplier 是每张合约对应的币数量
            quanto_multiplier = Decimal(instrument.get("quanto_multiplier", "1"))
            order.sz_value = quanto_multiplier
            
            # 平仓时：order.sz 是币的数量，需要转换为张数
            if not order.is_open and order.sz:
                # 币数量 / 面值 = 张数（向下取整）
                size = int(Decimal(order.sz) / quanto_multiplier) if quanto_multiplier > 0 else int(order.sz)
                logger.info(f"Gate.io 平仓张数计算: sz={order.sz}, quanto_multiplier={quanto_multiplier}, size={size}")
            else:
                # 开仓：根据金额计算张数
                # 计算开仓价值（考虑杠杆）
                # 开仓价值 = 保证金 × 杠杆
                leverage = Decimal(order.leverage) if order.leverage else Decimal("1")
                position_value = order.order_amount * leverage
                
                if not order.order_price:
                    # 没有指定价格，从订单簿获取最新价格
                    orderbook = self._get_book(currency_pair, is_futures=True)
                    if order.side.is_long:
                        latest_price = Decimal(orderbook["asks"][0]["p"])  # 买入用卖一价
                    else:
                        latest_price = Decimal(orderbook["bids"][0]["p"])  # 卖出用买一价
                    order.order_price = latest_price

                # 计算币数量
                quantity = position_value / order.order_price
                
                # 转换为张数（向下取整）
                # 张数 = 币数量 / 每张合约对应的币数量
                size = int(quantity / quanto_multiplier) if quanto_multiplier > 0 else int(quantity)
                logger.info(f"Gate.io 开仓张数计算: amount={order.order_amount}*{leverage}={position_value}, quantity={quantity}, quanto_multiplier={quanto_multiplier}, size={size}")
            
            # 检查最小张数
            order_size_min = int(instrument.get("order_size_min", 1))
            if size < order_size_min:
                raise RuntimeError(f"{currency_pair}下单张数 {size} 小于最低限制 {order_size_min}")
            
            return Decimal(size)
        else:
            # 现货：使用数量
            min_base_amount = Decimal(instrument.get("min_base_amount", "0"))
            amount_precision = int(instrument.get("amount_precision", 8))
            
            # 计算数量
            if order.order_type == OT.MarketOrder:
                if not order.order_price:
                    # 市价单没有指定价格，从订单簿获取最新价格
                    orderbook = self._get_book(currency_pair, is_futures=False)
                    if order.side.is_long:
                        latest_price = Decimal(orderbook["asks"][0]["p"])  # 买入用卖一价
                    else:
                        latest_price = Decimal(orderbook["bids"][0]["p"])  # 卖出用买一价
                    order.order_price = latest_price
                quantity = order.order_amount / order.order_price
            else:
                quantity = order.sz if order.sz else (order.order_amount / order.order_price if order.order_price else order.order_amount)
            
            # 精度调整
            quantity = Decimal(str(quantity)).quantize(Decimal('0.1') ** amount_precision)
            
            if quantity < min_base_amount:
                raise RuntimeError(f"{currency_pair}下单数量 {quantity} 小于最低限制 {min_base_amount}")
            
            order.sz_value = Decimal("1")
            return quantity

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def _get_instrument(self, currency_pair, is_futures=False):
        """
        获取交易对/合约信息
        """
        if is_futures:
            settle = currency_pair.split("_")[-1].lower() if "_" in currency_pair else "usdt"
            contract_info = self.adapter.get_futures_contract(settle=settle, contract=currency_pair)
            if not contract_info or contract_info.get("code") != "0":
                return None
            return contract_info.get("data")
        else:
            currency_pairs_info = self.adapter.get_currency_pairs(currency_pair=currency_pair)
            if not currency_pairs_info or currency_pairs_info.get("code") != "0":
                return None
            return currency_pairs_info.get("data")
    
    def _format_price(self, currency_pair: str, price: Decimal, is_futures=False) -> str:
        """
        格式化价格，根据交易对的精度要求
        """
        instrument = self._get_instrument(currency_pair, is_futures=is_futures)
        if not instrument:
            return str(price)
        
        if is_futures:
            # 合约价格精度
            # Gate.io 合约 API 返回格式：{"order_price_round": "0.01", ...}
            order_price_round = Decimal(instrument.get("order_price_round", "0.01"))
            # 调整价格到 order_price_round 的倍数
            if order_price_round > 0:
                price = (price // order_price_round) * order_price_round
        else:
            # 现货价格精度
            price_precision = int(instrument.get("price_precision", instrument.get("precision", 8)))
            price = Decimal(str(price)).quantize(Decimal('0.1') ** price_precision)
        
        return str(price)

    @staticmethod
    def _safe_decimal(value, default="0"):
        """
        安全地将值转换为 Decimal
        """
        if value is None:
            return Decimal(default)
        try:
            value_str = str(value).strip()
            if not value_str or value_str == "":
                return Decimal(default)
            return Decimal(value_str)
        except (ValueError, TypeError, InvalidOperation):
            return Decimal(default)

    def _query_order_fee_and_pnl(self, order_info: Dict, order_data: Dict, order: Order) -> tuple:
        """
        查询订单的成交手续费和已实现盈亏
        
        fee 总是从 get_futures_my_trades 获取
        pnl 总是从 get_futures_position_close 获取
        
        Args:
            order_info: 订单信息
            order_data: API 返回的订单数据
            order: 原始订单对象
            
        Returns:
            tuple: (fee, pnl) - fee为负数（支出），pnl可能为正或负，无数据时返回None
        """
        contract = order_info.get("currency_pair")
        if not contract:
            return None, None
        
        # 获取结算货币
        settle = contract.split("_")[-1].lower() if "_" in contract else "usdt"
        
        # 从成交记录获取手续费
        fee = Decimal(0)
        try:
            exchange_order_id = str(order_data.get("id", ""))
            if exchange_order_id:
                result = self.adapter.get_futures_my_trades(
                    settle=settle,
                    contract=contract,
                    order_id=exchange_order_id,
                    limit=100
                )
                
                if result and result.get("code") == "0":
                    trades = result.get("data", [])
                    if trades:
                        total_fee = Decimal(0)
                        for trade in trades:
                            trade_fee = self._safe_decimal(trade.get("fee", "0"))
                            # 确保为负数（支出）
                            if trade_fee > 0:
                                trade_fee = -trade_fee
                            total_fee += trade_fee
                        fee = total_fee
                        logger.info(f"从成交记录获取手续费 - order_id: {exchange_order_id}, fee: {fee}, 成交笔数: {len(trades)}")
                else:
                    logger.warning(f"查询成交记录失败: {result.get('msg') if result else '无响应'}, order_id: {exchange_order_id}")
        except Exception as e:
            logger.warning(f"从成交记录查询手续费异常: {e}", exc_info=True)
        
        # 从平仓历史获取pnl，最多重试3次，每次间隔2秒
        pnl = None
        try:
            text_id = str(order_data.get("text", ""))
            if text_id:
                from_time = None
                if order.order_time:
                    from_time = int(order.order_time.timestamp()) - 1
                
                max_retries = 3
                for retry_count in range(max_retries):
                    result = self.adapter.get_futures_position_close(settle=settle, contract=contract, limit=100, from_time=from_time)
                    logger.info(f"从平仓历史获取数据 - {text_id}, settle={settle}, contract={contract}, from_time={from_time}, retry={retry_count+1}, result: {result}")
                    if result and result.get("code") == "0":
                        position_closes = result.get("data", [])
                        for close_record in position_closes:
                            if close_record.get("text") == text_id:
                                pnl = self._safe_decimal(close_record.get("pnl", "0"))
                                logger.info(f"从平仓历史获取pnl - text_id: {text_id}, pnl: {pnl}")
                                break
                    
                    if pnl is not None:
                        break
                    
                    # 没有查询到结果，等待2秒后重试
                    if retry_count < max_retries - 1:
                        logger.info(f"未找到平仓记录，等待2秒后重试 - text_id: {text_id}, retry={retry_count+1}/{max_retries}")
                        time.sleep(2)
        except Exception as e:
            logger.warning(f"从平仓历史查询pnl异常: {e}", exc_info=True)
        
        return fee, pnl if pnl is not None else Decimal(0)

    def _polling_loop(self):
        """
        订单状态轮询循环，支持现货和合约
        """
        logger.info("[Gate.io REST] 订单状态轮询线程已启动")
        while not self._stop_event.is_set():
            try:
                # 快速获取待查询的订单列表（最小化锁持有时间）
                with self._orders_lock:
                    if not self._pending_orders:
                        text_ids = []
                        orders_info = {}
                    else:
                        text_ids = list(self._pending_orders.keys())
                        orders_info = dict(self._pending_orders)
                
                # 在锁外检查，避免持有锁时等待
                if not text_ids:
                    self._stop_event.wait(self._pull_interval)  
                    continue
                
                # 按 currency_pair 和 is_futures 分组查询订单状态
                spot_groups: Dict[str, List[str]] = {}
                futures_groups: Dict[str, List[str]] = {}
                
                for text_id in text_ids:
                    order_info = orders_info[text_id]
                    currency_pair = order_info["currency_pair"]
                    is_futures = order_info.get("is_futures", False)
                    
                    if is_futures:
                        if currency_pair not in futures_groups:
                            futures_groups[currency_pair] = []
                        futures_groups[currency_pair].append(text_id)
                    else:
                        if currency_pair not in spot_groups:
                            spot_groups[currency_pair] = []
                        spot_groups[currency_pair].append(text_id)
                
                # 处理现货订单
                for currency_pair, group_text_ids in spot_groups.items():
                    self._poll_spot_orders(currency_pair, group_text_ids, orders_info)
                
                # 处理合约订单
                for contract, group_text_ids in futures_groups.items():
                    self._poll_futures_orders(contract, group_text_ids, orders_info)
                
                # 等待2秒后继续下一轮查询
                self._stop_event.wait(self._pull_interval)
            
            except Exception as e:
                logger.error(f"订单状态轮询异常: {e}", exc_info=True)
                self._stop_event.wait(self._pull_interval)
        
        logger.info("[Gate.io REST] 订单状态轮询线程已停止")
    
    def _poll_spot_orders(self, currency_pair: str, group_text_ids: List[str], orders_info: Dict):
        """轮询现货订单状态"""
        try:
            # 查询该交易对的所有待成交订单
            result = self.adapter.get_open_orders(currency_pair=currency_pair)
            
            if not result or result.get("code") != "0":
                logger.warning(f"查询现货订单状态失败: {result.get('msg') if result else '无响应'}, currency_pair: {currency_pair}")
                return
            
            # 处理查询结果，只处理我们关心的订单
            orders_data = result.get("data", [])
            found_text_ids = set()
            
            for order_data in orders_data:
                text = order_data.get("text")
                if not text or text not in orders_info:
                    continue
                
                found_text_ids.add(text)
                # 待成交订单状态为 open，部分成交也是 open 状态
                filled_total = self._safe_decimal(order_data.get("filled_total", "0"))
                if filled_total > 0:
                    # 部分成交，触发回调但不移除
                    self._process_order_update(text, orders_info[text], order_data, remove_from_pending=False)
            
            # 对于不在待成交列表中的订单，使用 get_order 单独查询
            for text_id in group_text_ids:
                if text_id in found_text_ids:
                    continue
                
                order_info = orders_info.get(text_id)
                if not order_info:
                    continue
                
                try:
                    order_result = self.adapter.get_order(currency_pair=currency_pair, text=text_id)
                    if order_result and order_result.get("code") == "0":
                        order_data_list = order_result.get("data", [])
                        if order_data_list:
                            order_data = order_data_list[0]
                            status = order_data.get("status")
                            if status in ["closed", "cancelled", "expired"]:
                                self._process_order_update(text_id, order_info, order_data, remove_from_pending=True)
                            elif status == "open":
                                filled_total = self._safe_decimal(order_data.get("filled_total", "0"))
                                if filled_total > 0:
                                    self._process_order_update(text_id, order_info, order_data, remove_from_pending=False)
                except Exception as e:
                    logger.warning(f"查询现货订单状态失败: {e}, text_id: {text_id}", exc_info=True)
        
        except Exception as e:
            logger.error(f"批量查询现货订单状态异常: {e}", exc_info=True)
    
    def _poll_futures_orders(self, contract: str, group_text_ids: List[str], orders_info: Dict):
        """轮询合约订单状态"""
        try:
            settle = contract.split("_")[-1].lower() if "_" in contract else "usdt"
            
            # 查询该合约的所有待成交订单
            result = self.adapter.get_futures_open_orders(settle=settle, contract=contract)
            
            if not result or result.get("code") != "0":
                logger.warning(f"查询合约订单状态失败: {result.get('msg') if result else '无响应'}, contract: {contract}")
                return
            
            # 处理查询结果
            orders_data = result.get("data", [])
            found_text_ids = set()
            
            for order_data in orders_data:
                text = order_data.get("text")
                if not text or text not in orders_info:
                    continue
                
                found_text_ids.add(text)
                # 检查是否部分成交
                size = abs(int(order_data.get("size", 0)))
                left = abs(int(order_data.get("left", size)))
                if size > left:
                    # 部分成交
                    self._process_futures_order_update(text, orders_info[text], order_data, remove_from_pending=False)
            
            # 对于不在待成交列表中的订单，使用 get_order 单独查询
            for text_id in group_text_ids:
                if text_id in found_text_ids:
                    continue
                
                order_info = orders_info.get(text_id)
                if not order_info:
                    continue
                
                exchange_order_id = order_info.get("order_id")
                if not exchange_order_id:
                    continue
                
                try:
                    order_result = self.adapter.get_futures_order(settle=settle, order_id=exchange_order_id)
                    if order_result and order_result.get("code") == "0":
                        order_data_list = order_result.get("data", [])
                        if order_data_list:
                            order_data = order_data_list[0]
                            status = order_data.get("status")
                            if status == "finished":
                                self._process_futures_order_update(text_id, order_info, order_data, remove_from_pending=True)
                            elif status == "open":
                                size = abs(int(order_data.get("size", 0)))
                                left = abs(int(order_data.get("left", size)))
                                if size > left:
                                    self._process_futures_order_update(text_id, order_info, order_data, remove_from_pending=False)
                except Exception as e:
                    logger.warning(f"查询合约订单状态失败: {e}, text_id: {text_id}", exc_info=True)
        
        except Exception as e:
            logger.error(f"批量查询合约订单状态异常: {e}", exc_info=True)
    
    def _process_order_update(self, text_id: str, order_info: Dict, order_data: Dict, remove_from_pending: bool = True):
        """
        处理订单更新并触发回调
        
        Args:
            text_id: 客户端订单ID (text)
            order_info: 订单信息
            order_data: API 返回的订单数据
            remove_from_pending: 是否从待查询列表中移除
        """
        try:
            order = order_info["order"]
            status = order_data.get("status")
            
            # 映射订单状态
            order_status = GateRestExecutor.__Status_Map.get(status, OrderStatus.SUBMITTED)
            
            # 获取成交价格和数量
            # Gate.io API: price 是订单价格，filled_total 是已成交金额
            price = self._safe_decimal(order_data.get("price", "0"))
            filled_total = self._safe_decimal(order_data.get("filled_total", "0"))
            filled_amount = self._safe_decimal(order_data.get("filled_amount", "0"))
            
            # 计算平均成交价格
            if filled_amount > 0:
                avg_price = filled_total / filled_amount
            else:
                avg_price = price
            
            # 构造 OrderUpdateMessage
            oum = OrderUpdateMessage(
                order_id=text_id,
                market_order_id=str(order_data.get("id", "")),
                execution_price=avg_price,
                sz=filled_amount,
                fee=self._safe_decimal(order_data.get("fee", "0")),
                pnl=Decimal(0),  # Gate.io 现货没有 pnl
                unrealized_pnl=Decimal(0),
                friction=Decimal(0),
                sz_value=Decimal(1),
            )
            
            # 计算结算金额（Gate.io 现货：结算金额 = 成交金额）
            oum.settle_amount = filled_total
            
            # 设置订单状态
            oum.order_status = order_status
            
            # 设置完成时间
            update_time = order_data.get("update_time")
            if update_time:
                oum.finish_time = DateTimeUtils.to_datetime(int(update_time) * 1000)
            
            logger.info(f"Gate.io REST现货订单更新: {text_id} -> {oum}")
            self._trade_callback(oum)
            
            # 如果订单已完成，从待查询列表中移除
            if remove_from_pending and order_status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                with self._orders_lock:
                    self._pending_orders.pop(text_id, None)
        except Exception as e:
            logger.error(f"处理现货订单更新异常: {e}, text_id: {text_id}", exc_info=True)
    
    def _process_futures_order_update(self, text_id: str, order_info: Dict, order_data: Dict, remove_from_pending: bool = True):
        """
        处理合约订单更新并触发回调
        
        Args:
            text_id: 客户端订单ID (text)
            order_info: 订单信息
            order_data: API 返回的订单数据
            remove_from_pending: 是否从待查询列表中移除
        """
        try:
            order = order_info["order"]
            status = order_data.get("status")
            
            # 合约订单状态映射
            # Gate.io 合约状态：open, finished
            if status == "finished":
                finish_as = order_data.get("finish_as", "")
                if finish_as == "filled":
                    order_status = OrderStatus.FILLED
                elif finish_as == "cancelled":
                    order_status = OrderStatus.CANCELED
                elif finish_as == "liquidated":
                    order_status = OrderStatus.FILLED  # 强平也算成交
                elif finish_as == "ioc":
                    order_status = OrderStatus.FILLED  # IOC 部分成交
                elif finish_as == "auto_deleveraged":
                    order_status = OrderStatus.FILLED  # 自动减仓
                elif finish_as == "reduce_only":
                    order_status = OrderStatus.CANCELED  # reduce_only 失败
                else:
                    order_status = OrderStatus.FILLED
            else:
                order_status = OrderStatus.SUBMITTED
            
            # 获取成交信息
            # Gate.io 合约 API: size 是总张数，left 是剩余张数，fill_price 是成交价格
            size = abs(int(order_data.get("size", 0)))
            left = abs(int(order_data.get("left", size)))
            filled_size = size - left  # 已成交张数
            
            fill_price = self._safe_decimal(order_data.get("fill_price", "0"))
            price = self._safe_decimal(order_data.get("price", "0"))
            avg_price = fill_price if fill_price > 0 else price
            
            # 获取面值 (sz_value)
            sz_value = order.sz_value if order.sz_value else Decimal(1)
            
            # 获取杠杆倍数
            leverage = order.leverage if order.leverage else Decimal("1")
            
            # 查询成交记录获取手续费和盈亏
            fee, pnl = self._query_order_fee_and_pnl(order_info, order_data, order)
            
            # 构造 OrderUpdateMessage
            # sz 应该是币的数量，不是张数：sz = 张数 * 面值
            sz = Decimal(filled_size) * sz_value  # 币的数量 = 张数 * 面值
            oum = OrderUpdateMessage(
                order_id=text_id,
                market_order_id=str(order_data.get("id", "")),
                execution_price=avg_price,
                sz=sz,
                fee=fee,  # 手续费（负数）
                pnl=pnl,  # 已实现盈亏
                unrealized_pnl=Decimal(0),
                friction=Decimal(0),
                sz_value=sz_value,
            )
            
            # 计算结算金额（参考OKX的逻辑）
            # 开仓：settle_amount = 名义价值 / 杠杆 = sz * price / leverage（保证金）
            # 平仓：需要考虑盈亏
            oum.settle_amount = sz * avg_price / leverage
            
            if not order.is_open:
                # 平仓时需要考虑盈亏调整结算金额
                # 平空（买入平空）：side=LONG
                # 平多（卖出平多）：side=SHORT
                if order.side == PS.LONG:  # 平空（买入平空）
                    oum.settle_amount = (avg_price * sz + pnl) / leverage + pnl
                else:  # 平多（卖出平多）
                    oum.settle_amount = (avg_price * sz - pnl) / leverage + pnl
            
            # 设置订单状态
            oum.order_status = order_status
            
            # 设置完成时间
            finish_time = order_data.get("finish_time")
            if finish_time:
                oum.finish_time = DateTimeUtils.to_datetime(int(finish_time) * 1000)
            
            logger.info(f"Gate.io REST合约订单更新: {text_id} -> {oum}")
            self._trade_callback(oum)
            
            # 如果订单已完成，从待查询列表中移除
            if remove_from_pending and order_status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                with self._orders_lock:
                    self._pending_orders.pop(text_id, None)
        except Exception as e:
            logger.error(f"处理合约订单更新异常: {e}, text_id: {text_id}", exc_info=True)

    def on_start(self):
        """
        启动执行器，初始化订单列表并启动轮询线程
        """
        logger.info("[Gate.io REST] 启动执行器")
        self._pending_orders.clear()
        self._stop_event.clear()
        
        # 启动轮询线程
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()
        logger.info("[Gate.io REST] 执行器已启动")

    def on_stop(self):
        """
        停止执行器，停止轮询线程并清理资源
        """
        logger.info("[Gate.io REST] 停止执行器")
        
        # 设置停止标志
        self._stop_event.set()
        
        # 等待轮询线程结束（最多等待5秒）
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)
            if self._polling_thread.is_alive():
                logger.warning("[Gate.io REST] 轮询线程未能在5秒内停止")
        
        # 清理订单列表
        with self._orders_lock:
            self._pending_orders.clear()
        
        logger.info("[Gate.io REST] 执行器已停止")
