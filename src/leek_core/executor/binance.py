#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
币安交易执行模块
"""

import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional

from cachetools import TTLCache, cached
from leek_core.adapts import BinanceAdapter

from leek_core.models import Order, OrderStatus, OrderUpdateMessage
from leek_core.models import PositionSide as PS, OrderType as OT, TradeMode, TradeInsType, Field, FieldType
from leek_core.utils import get_logger, DateTimeUtils, retry
from .base import Executor

logger = get_logger(__name__)


class BinanceRestExecutor(Executor):
    """
    币安交易所REST API执行器，通过REST API进行下单、撤单等操作。
    使用内部轮询机制查询订单状态并触发回调。
    """
    display_name = "Binance REST"
    init_params: List['Field'] = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, default="", required=True),
        Field(name="secret_key", label="API Secret Key", type=FieldType.PASSWORD, default="", required=True),
        Field(name="testnet", label="测试网", type=FieldType.BOOLEAN, default=False, required=False, description="是否使用测试网"),
        Field(name="slippage_level", label="允许滑档", type=FieldType.INT, default=5, required=True, description="限价交易允许滑档数"),
    ]

    # 映射表
    __Side_Map = {
        PS.LONG: "BUY",
        PS.SHORT: "SELL",  # 币安现货不支持做空，但保留映射用于错误提示
    }
    __Order_Type_Map = {
        OT.LimitOrder: "LIMIT",
        OT.MarketOrder: "MARKET",
    }
    __Status_Map = {
        "NEW": OrderStatus.SUBMITTED,
        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        "FILLED": OrderStatus.FILLED,
        "CANCELED": OrderStatus.CANCELED,
        "REJECTED": OrderStatus.REJECTED,
        "EXPIRED": OrderStatus.EXPIRED,
    }

    def __init__(self, api_key, secret_key, testnet=False, slippage_level=5, **kwargs):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = bool(testnet)
        self.slippage_level = int(slippage_level)
        self.adapter = BinanceAdapter(api_key=api_key, secret_key=secret_key, testnet=testnet)
        
        self._pending_orders: Dict[str, Dict] = {}  # key: clientOrderId, value: 订单信息
        self._orders_lock = threading.RLock()
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def send_order(self, orders: Order | List[Order]):
        """
        组装参数并通过REST API下单
        """
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            # 币安现货不支持做空
            if order.ins_type == TradeInsType.SPOT and order.side == PS.SHORT:
                raise ValueError("币安现货不支持做空操作")
            
            # 构建交易对符号
            symbol = self.adapter.build_symbol(order.symbol, order.quote_currency)
            
            # 数量精度校验
            quantity = self._check_sz(order)
            
            # 订单类型
            order_type = BinanceRestExecutor.__Order_Type_Map.get(order.order_type, "LIMIT")
            
            # 价格处理
            price = None
            if order_type == "LIMIT":
                if order.order_price is not None:
                    # 限价单需要价格精度校验
                    price = self._format_price(symbol, order.order_price)
                else:
                    # 限价单如果没有价格，从订单簿获取
                    orderbook = self._get_book(symbol)
                    if order.side == PS.LONG:
                        price = orderbook["asks"][-1][0]
                    else:
                        price = orderbook["bids"][-1][0]
                    # 格式化价格
                    price = self._format_price(symbol, Decimal(price))
            
            # 调用REST API下单
            try:
                result = self.adapter.place_order(
                    symbol=symbol,
                    side=BinanceRestExecutor.__Side_Map[order.side],
                    order_type=order_type,
                    quantity=str(quantity),
                    price=price,
                    client_order_id=str(order.order_id)
                )
                
                if not result or result.get("code") != "0":
                    error_msg = result.get("msg", "未知错误") if result else "无响应"
                    logger.error(f"下单失败: {error_msg}, 订单: {order.order_id}, {result}")
                    raise RuntimeError(f"下单失败: {error_msg}")
                
                # 下单成功，添加到待查询列表
                order_data = result.get("data", [{}])[0]
                ord_id = order_data.get("orderId")
                logger.info(f"下单完成: {order.order_id}, 交易所订单ID: {ord_id}")
                with self._orders_lock:
                    self._pending_orders[order.order_id] = {
                        "symbol": symbol,
                        "orderId": ord_id,
                        "order": order,
                    }
                logger.info(f"下单成功: {order.order_id}, 交易所订单ID: {ord_id}")
            except Exception as e:
                logger.error(f"下单异常: {e}, 订单: {order.order_id}", exc_info=True)
                raise

    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        通过REST API撤单
        """
        try:
            # 从待查询列表中获取订单信息
            with self._orders_lock:
                order_info = self._pending_orders.get(order_id)
            
            if not order_info:
                logger.warning(f"订单 {order_id} 不在待查询列表中，可能已完成或不存在")
                return
            
            symbol = order_info["symbol"]
            result = self.adapter.cancel_order(symbol=symbol, client_order_id=order_id)
            
            if not result or result.get("code") != "0":
                error_msg = result.get("msg", "未知错误") if result else "无响应"
                logger.error(f"撤单失败: {error_msg}, 订单: {order_id}")
                raise RuntimeError(f"撤单失败: {error_msg}")
            
            logger.info(f"撤单成功: {order_id}")
        except Exception as e:
            logger.error(f"撤单异常: {e}, 订单: {order_id}", exc_info=True)
            raise

    @retry(max_retries=3, retry_interval=0.2)
    def _get_book(self, symbol):
        orderbook = self.adapter.get_orderbook(symbol=symbol, limit=self.slippage_level)
        if not orderbook or orderbook["code"] != "0":
            raise RuntimeError(f"获取深度失败:{orderbook.get('msg') if orderbook else '无响应'}")
        return orderbook["data"][0]

    def _check_sz(self, order: Order):
        """
        数量精度校验（使用币安的 exchange_info）
        """
        symbol = self.adapter.build_symbol(order.symbol, order.quote_currency)
        instrument = self._get_instrument(symbol)
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        
        # 获取精度信息
        filters = instrument.get("filters", [])
        lot_size_filter = next((f for f in filters if f.get("filterType") == "LOT_SIZE"), None)
        if not lot_size_filter:
            raise RuntimeError("无法获取数量精度信息")
        
        step_size = Decimal(lot_size_filter.get("stepSize", "1"))
        min_qty = Decimal(lot_size_filter.get("minQty", "0"))
        max_qty = Decimal(lot_size_filter.get("maxQty", "0"))
        
        # 计算数量
        if order.order_type == OT.MarketOrder:
            # 市价单：使用订单金额除以价格
            if not order.order_price:
                # 如果没有价格，使用订单金额
                quantity = order.order_amount
            else:
                quantity = order.order_amount / order.order_price
        else:
            # 限价单：使用订单数量
            quantity = order.sz if order.sz else (order.order_amount / order.order_price if order.order_price else order.order_amount)
        
        # 精度调整
        quantity = (quantity // step_size) * step_size
        
        # 检查最小和最大数量
        if quantity < min_qty:
            raise RuntimeError(f"{symbol}下单数量 {quantity}小于最低限制{min_qty}")
        if max_qty > 0 and quantity > max_qty:
            quantity = max_qty
        
        order.sz_value = Decimal("1")  # 币安现货 sz_value 为 1
        return quantity

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def _get_instrument(self, symbol):
        """
        获取交易对信息
        """
        exchange_info = self.adapter.get_exchange_info(symbol=symbol)
        if not exchange_info or exchange_info.get("code") != "0":
            return None
        return exchange_info.get("data")
    
    def _format_price(self, symbol: str, price: Decimal) -> str:
        """
        格式化价格，根据交易对的精度要求
        
        Args:
            symbol: 交易对符号
            price: 价格
            
        Returns:
            str: 格式化后的价格字符串
        """
        instrument = self._get_instrument(symbol)
        if not instrument:
            return str(price)
        
        # 获取价格精度
        filters = instrument.get("filters", [])
        price_filter = next((f for f in filters if f.get("filterType") == "PRICE_FILTER"), None)
        if price_filter:
            tick_size = Decimal(price_filter.get("tickSize", "1"))
            # 调整价格到 tick_size 的倍数
            price = (price // tick_size) * tick_size
        
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

    def _query_order_fee(self, order_info: Dict, order_data: Dict) -> Decimal:
        """
        查询订单的成交手续费
        
        Args:
            order_info: 订单信息
            order_data: API 返回的订单数据
            
        Returns:
            Decimal: 手续费总额（负数）
        """
        try:
            symbol = order_info.get("symbol")
            exchange_order_id = order_data.get("orderId")
            
            if not symbol or not exchange_order_id:
                return Decimal(0)
            
            # 查询该订单的成交记录
            result = self.adapter.get_my_trades(
                symbol=symbol,
                order_id=exchange_order_id,
                limit=100
            )
            
            if not result or result.get("code") != "0":
                logger.warning(f"查询成交记录失败: {result.get('msg') if result else '无响应'}, order_id: {exchange_order_id}")
                return Decimal(0)
            
            # 汇总所有成交的手续费
            trades = result.get("data", [])
            total_fee = Decimal(0)
            for trade in trades:
                # 币安成交记录中的 commission 字段是手续费
                commission = self._safe_decimal(trade.get("commission", "0"))
                # 手续费为负数（支出）
                total_fee -= commission
            
            logger.debug(f"订单 {exchange_order_id} 手续费汇总: {total_fee}, 成交笔数: {len(trades)}")
            return total_fee
        except Exception as e:
            logger.warning(f"查询订单手续费异常: {e}", exc_info=True)
            return Decimal(0)

    def _polling_loop(self):
        """
        订单状态轮询循环
        """
        logger.info("[Binance REST] 订单状态轮询线程已启动")
        while not self._stop_event.is_set():
            try:
                # 快速获取待查询的订单列表（最小化锁持有时间）
                with self._orders_lock:
                    if not self._pending_orders:
                        cl_ord_ids = []
                        orders_info = {}
                    else:
                        cl_ord_ids = list(self._pending_orders.keys())
                        orders_info = dict(self._pending_orders)
                
                # 在锁外检查，避免持有锁时等待
                if not cl_ord_ids:
                    self._stop_event.wait(2.0)
                    continue
                
                # 按 symbol 分组查询订单状态，减少 API 调用次数
                # 按 symbol 分组
                symbol_groups: Dict[str, List[str]] = {}
                for cl_ord_id in cl_ord_ids:
                    order_info = orders_info[cl_ord_id]
                    symbol = order_info["symbol"]
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(cl_ord_id)
                
                # 对每个 symbol 查询订单状态
                for symbol, group_cl_ord_ids in symbol_groups.items():
                    try:
                        # 查询该交易对的所有待成交订单
                        result = self.adapter.get_open_orders(symbol=symbol)
                        
                        if not result or result.get("code") != "0":
                            logger.warning(f"查询订单状态失败: {result.get('msg') if result else '无响应'}, symbol: {symbol}")
                            continue
                        
                        # 处理查询结果，只处理我们关心的订单
                        orders_data = result.get("data", [])
                        found_cl_ord_ids = set()
                        
                        for order_data in orders_data:
                            cl_ord_id = order_data.get("clientOrderId")
                            if not cl_ord_id or cl_ord_id not in orders_info:
                                continue
                            
                            found_cl_ord_ids.add(cl_ord_id)
                            # 待成交订单状态为 NEW 或 PARTIALLY_FILLED
                            status = order_data.get("status")
                            if status == "PARTIALLY_FILLED":
                                # 部分成交，触发回调但不移除
                                self._process_order_update(cl_ord_id, orders_info[cl_ord_id], order_data, remove_from_pending=False)
                        
                        # 对于不在待成交列表中的订单，使用 get_order 单独查询
                        for cl_ord_id in group_cl_ord_ids:
                            if cl_ord_id in found_cl_ord_ids:
                                continue  # 已经在待成交列表中
                            
                            order_info = orders_info.get(cl_ord_id)
                            if not order_info:
                                continue
                            
                            # 使用 clientOrderId 单独查询订单状态
                            try:
                                order_result = self.adapter.get_order(symbol=symbol, client_order_id=cl_ord_id)
                                if order_result and order_result.get("code") == "0":
                                    order_data_list = order_result.get("data", [])
                                    if order_data_list:
                                        order_data = order_data_list[0]
                                        status = order_data.get("status")
                                        if status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                                            # 处理已完成的订单（会自动从待查询列表移除）
                                            self._process_order_update(cl_ord_id, order_info, order_data, remove_from_pending=True)
                                        elif status == "PARTIALLY_FILLED":
                                            # 部分成交，触发回调但不移除
                                            self._process_order_update(cl_ord_id, order_info, order_data, remove_from_pending=False)
                            except Exception as e:
                                logger.warning(f"单独查询订单状态失败: {e}, cl_ord_id: {cl_ord_id}", exc_info=True)
                    
                    except Exception as e:
                        logger.error(f"批量查询订单状态异常: {e}", exc_info=True)
                        continue
                
                # 已完成的订单已在 _process_order_update 中移除，这里不需要再次移除
                
                # 等待2秒后继续下一轮查询
                self._stop_event.wait(2.0)
            
            except Exception as e:
                logger.error(f"订单状态轮询异常: {e}", exc_info=True)
                self._stop_event.wait(2.0)
        
        logger.info("[Binance REST] 订单状态轮询线程已停止")
    
    def _process_order_update(self, cl_ord_id: str, order_info: Dict, order_data: Dict, remove_from_pending: bool = True):
        """
        处理订单更新并触发回调
        
        Args:
            cl_ord_id: 客户端订单ID
            order_info: 订单信息
            order_data: API 返回的订单数据
            remove_from_pending: 是否从待查询列表中移除
        """
        try:
            order = order_info["order"]
            status = order_data.get("status")
            
            # 映射订单状态
            order_status = BinanceRestExecutor.__Status_Map.get(status, OrderStatus.SUBMITTED)
            
            # 获取成交价格和数量
            # 币安 API: avgPrice 是平均成交价格（字符串），price 是订单价格
            avg_price = order_data.get("avgPrice")
            if not avg_price or avg_price == "0.00000000":
                # 如果没有平均成交价格，使用订单价格
                avg_price = order_data.get("price") or "0"
            executed_qty = order_data.get("executedQty") or "0"
            
            # 查询成交记录获取手续费
            fee = Decimal(0)
            sz_decimal = self._safe_decimal(executed_qty, "0")
            if sz_decimal > 0 and order_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                fee = self._query_order_fee(order_info, order_data)
            
            # 构造 OrderUpdateMessage
            oum = OrderUpdateMessage(
                order_id=cl_ord_id,
                market_order_id=str(order_data.get("orderId", "")),
                execution_price=self._safe_decimal(avg_price, "0"),
                sz=sz_decimal,
                fee=fee,
                pnl=Decimal(0),  # pnl 由 position_tracker 根据开仓价和平仓价计算
                unrealized_pnl=Decimal(0),
                friction=Decimal(0),
                sz_value=Decimal(1),
            )
            
            # 计算结算金额
            if oum.execution_price > 0 and oum.sz > 0:
                # 根据交易类型判断是否需要除以杠杆
                if order.ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES]:
                    # 合约：保证金 = 名义价值 / 杠杆
                    leverage = order.leverage if order.leverage else Decimal("1")
                    oum.settle_amount = oum.sz * oum.execution_price / leverage
                else:
                    # 现货：成交金额
                    oum.settle_amount = oum.execution_price * oum.sz
            else:
                # 如果没有成交，使用订单价格和数量
                oum.settle_amount = self._safe_decimal(order_data.get("price", "0"), "0") * self._safe_decimal(order_data.get("origQty", "0"), "0")
            
            # 设置订单状态
            oum.order_status = order_status
            
            # 设置完成时间
            update_time = order_data.get("updateTime")
            if update_time:
                oum.finish_time = DateTimeUtils.to_datetime(update_time)
            
            logger.info(f"Binance REST订单更新: {cl_ord_id} -> {oum}")
            self._trade_callback(oum)
            
            # 如果订单已完成，从待查询列表中移除
            if remove_from_pending and order_status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                with self._orders_lock:
                    self._pending_orders.pop(cl_ord_id, None)
        except Exception as e:
            logger.error(f"处理订单更新异常: {e}, cl_ord_id: {cl_ord_id}", exc_info=True)

    def on_start(self):
        """
        启动执行器，初始化订单列表并启动轮询线程
        """
        logger.info("[Binance REST] 启动执行器")
        self._pending_orders.clear()
        self._stop_event.clear()
        
        # 启动轮询线程
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()
        logger.info("[Binance REST] 执行器已启动")

    def on_stop(self):
        """
        停止执行器，停止轮询线程并清理资源
        """
        logger.info("[Binance REST] 停止执行器")
        
        # 设置停止标志
        self._stop_event.set()
        
        # 等待轮询线程结束（最多等待5秒）
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)
            if self._polling_thread.is_alive():
                logger.warning("[Binance REST] 轮询线程未能在5秒内停止")
        
        # 清理订单列表
        with self._orders_lock:
            self._pending_orders.clear()
        
        logger.info("[Binance REST] 执行器已停止")
