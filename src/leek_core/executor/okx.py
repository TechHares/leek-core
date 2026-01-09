#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX交易执行模块
"""

import asyncio
import base64
import hmac
import json
import time
import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional

import websockets

from cachetools import TTLCache, cached
from leek_core.adapts import OkxAdapter

from leek_core.models import Order, PosMode, OrderStatus, OrderUpdateMessage
from leek_core.models import PositionSide as PS, OrderType as OT, TradeMode, TradeInsType, Field, FieldType, ChoiceType
from leek_core.utils import get_logger, generate_str, DateTimeUtils, retry
from .base import WebSocketExecutor, Executor

logger = get_logger(__name__)
LOCK = asyncio.Lock()


class OkxWebSocketExecutor(WebSocketExecutor):
    """
    OKX交易所WebSocket异步执行器，支持自动重连、心跳、鉴权、频道订阅、订单推送、下单、撤单等。
    集成原OkxTrader的业务参数组装、映射和风控逻辑。
    """
    display_name = "OKX"
    init_params: List['Field'] = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, default="", required=True),
        Field(name="secret_key", label="API Secret Key", type=FieldType.PASSWORD, default="", required=True),
        Field(name="passphrase", label="Passphrase", type=FieldType.PASSWORD, default="", required=True),

        Field(name="slippage_level", label="允许滑档", type=FieldType.INT, default=4, required=True, description="限价交易允许滑档数"),
        Field(name="ccy", label="默认保证金币种", type=FieldType.STRING, default="USDT", required=True, description="交易币种"),
    ] + [WebSocketExecutor.init_params[2], WebSocketExecutor.init_params[3]]

    __Side_Map = {
        PS.LONG: "buy",
        PS.SHORT: "sell",
    }
    __Pos_Side_Map = {
        PS.LONG: "long",
        PS.SHORT: "short",
    }
    __Trade_Mode_Map = {
        TradeMode.ISOLATED: "isolated",
        TradeMode.CROSS: "cross",
        TradeMode.CASH: "cash",
        TradeMode.SPOT_ISOLATED: "spot_isolated",
    }
    __Inst_Type_Map = {
        TradeInsType.SPOT: "SPOT",
        TradeInsType.MARGIN: "MARGIN",
        TradeInsType.SWAP: "SWAP",
        TradeInsType.FUTURES: "FUTURES",
        TradeInsType.OPTION: "OPTION",
    }

    def __init__(self, api_key, secret_key, passphrase, slippage_level=4, ccy="", **kwargs):
        super().__init__(ws_url="wss://ws.okx.com:8443/ws/v5/private", heartbeat_interval=22, **kwargs)
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self._login_ok = False
        self.slippage_level = int(slippage_level)
        self.ccy = ccy
        self.adapter = OkxAdapter(api_key=api_key, secret_key=secret_key, passphrase=passphrase)

        self.pos_mode = None

    def sign(self, message, secretKey):
        mac = hmac.new(bytes(secretKey, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        return base64.b64encode(d)

    async def on_open(self):
        # 登录鉴权，失败时自动重试，重试间隔和次数与连接一致
        retry = 0
        while retry < getattr(self, 'max_retries', 5):
                        # 使用adapter进行鉴权，这里暂时保留原有逻辑
            timestamp = str(int(time.time()))
            s = self.sign(timestamp + "GET" + "/users/self/verify", self.secret_key).decode()
            data = {
                "op": "login",
                "args": [
                    {
                        "apiKey": self.api_key,
                        "passphrase": self.passphrase,
                        "timestamp": timestamp,
                        "sign": s,
                    }
                ]
            }
            await self.send(json.dumps(data))
            logger.info(f"[OKX] 已发送鉴权请求, timestamp={timestamp}, sign={s}, 第{retry+1}次")
            # 等待登录结果，超时或失败则重试
            try:
                # 等待 _login_ok 被设置（如on_message收到login成功事件），最多10秒
                for _ in range(10):
                    await asyncio.sleep(1)
                    if self._login_ok:
                        await self._subscribe_channels()
                        logger.info("[OKX] 登录成功并已订阅频道")
                        return
                logger.warning(f"[OKX] 登录超时/失败, {getattr(self, 'reconnect_interval', 5)}秒后重试")
            except Exception as e:
                logger.error(f"[OKX] 登录等待异常: {e}")
            retry += 1
            await asyncio.sleep(getattr(self, 'reconnect_interval', 5))
        logger.error(f"[OKX] 登录失败，超过最大重试次数({getattr(self, 'max_retries', 5)})，断开连接")
        await self.on_close()

    async def on_message(self, msg):
        # 处理消息
        try:
            if msg == "pong":
                return
            logger.info(f"OKX推送: {msg}")
            msg = json.loads(msg)
            if "event" in msg:
                if msg["event"] == "login" and msg.get("code") == "0":
                    self._login_ok = True
                if msg["event"] == "error":
                    logger.error(f"OKX错误: {msg}")
                if msg["event"] in ("subscribe", "channel-conn-count"):
                    ... # 订阅成功回调  连接数回调
                return
            if "arg" in msg and "channel" in msg["arg"] and msg["arg"]["channel"] == "orders":  # 订单推送
                if "data" in msg:
                    for data in msg["data"]:
                        order_id = data["clOrdId"]
                        if data["state"] == "live":
                            continue
                        instrument = self._get_instrument(data["instId"], data["instType"])
                        oum = OrderUpdateMessage(
                            order_id=order_id,
                            market_order_id=data["ordId"],
                            execution_price=Decimal(data["avgPx"]),
                            sz = Decimal(data["accFillSz"]) * Decimal(instrument["ctVal"]),
                            fee=Decimal(data["fee"]),
                            pnl=Decimal(data["pnl"]),
                            unrealized_pnl=Decimal(0),
                            friction=Decimal(0),
                            sz_value=Decimal(instrument["ctVal"]),
                        )
                        oum.settle_amount = oum.execution_price * oum.sz / Decimal(data["lever"])
                        if data['side'] == 'buy' and data['posSide'] == 'short':
                            oum.settle_amount = (oum.execution_price * oum.sz + oum.pnl)/Decimal(data["lever"]) + oum.pnl
                        if data['side'] == 'sell' and data['posSide'] == 'long':
                            oum.settle_amount = (oum.execution_price * oum.sz - oum.pnl)/Decimal(data["lever"]) + oum.pnl

                        oum.order_status = OrderStatus("canceled" if data["state"] == "mmp_canceled" else data["state"])
                        if oum.order_status == OrderStatus.FILLED or oum.order_status == OrderStatus.PARTIALLY_FILLED:
                            oum.finish_time = DateTimeUtils.to_datetime(int(data["fillTime"]))
                        else:
                            oum.finish_time = DateTimeUtils.to_datetime(int(data["uTime"]))
                        logger.info(f"OKX订单更新: {msg} -> {oum}")
                        self._trade_callback(oum)
            if "code" in msg and msg["code"] == "1" and "data" in msg and len(msg["data"]) > 0:
                for data in msg["data"]:
                    if "sCode" in data and data["sCode"] != "0":
                        logger.error(f"OKX推送异常信息: {data['sCode']} - {data['sMsg']}: {data}")
        except Exception as e:
            logger.error(f"OKX消息处理异常: {e}", exc_info=True)

    async def _subscribe_channels(self):
        # 订阅订单、持仓等频道
        channels = [
            {"channel": "orders", "instType": "SWAP"},
            {"channel": "orders", "instType": "SPOT"},
            {"channel": "orders", "instType": "FUTURES"},
            {"channel": "positions", "instType": "MARGIN"},
            {"channel": "positions", "instType": "OPTION"},
        ]
        data = {"op": "subscribe", "args": channels}
        await self.send(json.dumps(data))
        logger.info(f"[OKX] 已发送频道订阅请求 {channels}")

    async def _handle_push(self, msg):
        # 这里只做简单转发，实际可按频道细分业务
        if "data" in msg and "arg" in msg:
            # 订单/持仓/成交等
            await self.callback(msg)

    def send_order(self, orders: Order | List[Order]):
        """
        组装参数并通过WebSocket异步下单
        """
        if isinstance(orders, Order):
            orders = [orders]
        
        orders_args = []
        for order in orders:
            params = {
                "tdMode": OkxWebSocketExecutor.__Trade_Mode_Map[order.trade_mode],
                "instId": self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency),
                "clOrdId": "%s" % order.order_id,
                "side": OkxWebSocketExecutor.__Side_Map[order.side],
                "ordType": "limit",
            }
            if order.trade_mode == TradeMode.CROSS:
                params["ccy"] = self.ccy

             # 合约类型特殊处理
            if order.ins_type == TradeInsType.SWAP:
                # 省略账户模式
                self.init_account_mode()
                if self.pos_mode == PosMode.LONG_SHORT_MODE:
                    params["posSide"] = OkxWebSocketExecutor.__Pos_Side_Map[order.side if order.is_open else order.side.switch()]

                # 数量精度校验
                sz = self._check_sz(order)
                params["sz"] = "%s" % sz
                if order.order_price is not None:
                    params["px"] = "%s" % order.order_price
                if order.order_type == OT.MarketOrder:
                    params["ordType"] = "market" if order.ins_type not in [TradeInsType.SWAP, TradeInsType.FUTURES] else "optimal_limit_ioc"
                else:
                    orderbook = self._get_book(self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency))
                    if order.side == PS.LONG:
                        params["px"] = orderbook["asks"][-1][0]
                    else:
                        params["px"] = orderbook["bids"][-1][0]
            orders_args.append(params)
            if order.ins_type != TradeInsType.SPOT:
                self.set_leverage(self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency), params["posSide"] if "posSide" in params else "", order.trade_mode.value,
                                  order.leverage)
        # 发送WebSocket下单消息（示例，需根据OKX ws协议格式封装）
        logger.info(f"下单：{orders_args}")
        self.send_ws_order(orders_args)

    def init_account_mode(self):
        if self.pos_mode:
            return
        account_config = self.adapter.get_account_config()
        if not account_config or account_config["code"] != "0":
            logger.error(f"账户配置获取失败:{account_config['msg'] if account_config else account_config}")
        mode = account_config["data"][0]["posMode"]
        pos_mode = PosMode(mode)
        if pos_mode != PosMode.NET_MODE:
            self.pos_mode = pos_mode
        logger.info(f"当前账户为「{pos_mode.value}」模式")

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def set_leverage(self, symbol, posSide, td_mode, lever):
        res = self.adapter.set_leverage(lever="%s" % lever, mgn_mode=td_mode, inst_id=symbol, pos_side=posSide)
        if not res or res["code"] != "0":
            raise RuntimeError(f"设置杠杆失败:{res['msg'] if res else res}")
        logger.info(f"设置杠杆为{lever}成功")

    async def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        通过WebSocket异步撤单
        """
        args = {
            "instId": symbol,
            "clOrdId": str(order_id)
        }
        await self.send_ws_cancel(args)
        return args

    @retry(max_retries=3, retry_interval=0.2)
    def _get_book(self, symbol):
        orderbook = self.adapter.get_orderbook(inst_id=symbol, sz=self.slippage_level)
        if not orderbook or orderbook["code"] != "0":
            raise RuntimeError(f"获取深度失败:{orderbook['msg'] if orderbook else orderbook}")
        return orderbook["data"][0]
    


    def _get_order_extra(self, order: Order, key: str):
        extra = order.extra or {}
        info = extra.get(self.instance_id)
        if not info or key not in info:
            return None
        return info[key]

    def _check_sz(self, order: Order):
        ins_id = self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency)
        instrument = self._get_instrument(ins_id, OkxWebSocketExecutor.__Inst_Type_Map[order.ins_type])
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        if not order.order_price:
            res = self.adapter.get_mark_price(
                inst_type=OkxWebSocketExecutor.__Inst_Type_Map[order.ins_type if order.ins_type != TradeInsType.SPOT else TradeInsType.MARGIN],
                inst_id=ins_id
            )
            order.order_price = Decimal(res["data"][0]["markPx"])
        lot_sz = instrument["lotSz"]
        ct_val = "1"
        if order.ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES, TradeInsType.OPTION]:
            ct_val = instrument["ctVal"]
        order.sz_value = Decimal(ct_val)
        if not order.is_open:
            sz = Decimal(order.sz) / Decimal(ct_val)
            return sz - (sz % Decimal(lot_sz))
        
        if ins_id.upper().endswith("-USD-SWAP"):
            num = order.order_amount / Decimal(ct_val)
        else:
            num = order.order_amount * order.leverage / (order.order_price * Decimal(ct_val))
        logger.info(f"okx size计算: amount: {order.order_amount}, num: {num}, lot_sz: {lot_sz}, ct_val: {ct_val}, tail: {num % Decimal(lot_sz)}")
        sz = num - (num % Decimal(lot_sz))
        min_sz = instrument["minSz"]
        if sz < Decimal(min_sz):
            raise RuntimeError(f"{ins_id}下单数量 sz {sz}小于最低限制{min_sz}")
        if order.order_type == OT.MarketOrder:
            max_mkt_sz = instrument["maxMktSz"]
            sz = min(sz, Decimal(max_mkt_sz))
        else:
            max_lmt_sz = instrument["maxLmtSz"]
            sz = min(sz, Decimal(max_lmt_sz))
        return Decimal(sz)

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def _get_instrument(self, symbol, ins_type):
        instruments = self.adapter.get_instruments(inst_type=ins_type, inst_id=symbol)
        if not instruments:
            return None
        if instruments['code'] != '0':
            return None
        if len(instruments['data']) == 0:
            return None
        instrument = instruments['data'][0]
        return instrument

    # 发送WebSocket下单消息（需根据OKX协议实现）
    def send_ws_order(self, args):
        msg = {
            "id": generate_str(),
            "op": "batch-orders",
            "args": args
        }
        self.async_send(json.dumps(msg))

    # 发送WebSocket撤单消息（需根据OKX协议实现）
    async def send_ws_cancel(self, args):
        msg = {
            "op": "cancel-order",
            "args": [args]
        }
        await self.send(json.dumps(msg))

    async def on_close(self):
        logger.info("[OKX] WebSocket连接已关闭")
        self._login_ok = False

    async def on_error(self, error):
        logger.warning(f"[OKX] WebSocket异常: {error}", exc_info=True)


class OkxRestExecutor(Executor):
    """
    OKX交易所REST API执行器，通过REST API进行下单、撤单等操作。
    使用内部轮询机制查询订单状态并触发回调。
    """
    display_name = "OKX REST"
    init_params: List['Field'] = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, default="", required=True),
        Field(name="secret_key", label="API Secret Key", type=FieldType.PASSWORD, default="", required=True),
        Field(name="passphrase", label="Passphrase", type=FieldType.PASSWORD, default="", required=True),
        Field(name="slippage_level", label="允许滑档", type=FieldType.INT, default=4, required=True, description="限价交易允许滑档数"),
        Field(name="ccy", label="默认保证金币种", type=FieldType.STRING, default="USDT", required=True, description="交易币种"),
    ]

    # 复用 WebSocket 版本的映射表
    __Side_Map = {
        PS.LONG: "buy",
        PS.SHORT: "sell",
    }
    __Pos_Side_Map = {
        PS.LONG: "long",
        PS.SHORT: "short",
    }
    __Trade_Mode_Map = {
        TradeMode.ISOLATED: "isolated",
        TradeMode.CROSS: "cross",
        TradeMode.CASH: "cash",
        TradeMode.SPOT_ISOLATED: "spot_isolated",
    }
    __Inst_Type_Map = {
        TradeInsType.SPOT: "SPOT",
        TradeInsType.MARGIN: "MARGIN",
        TradeInsType.SWAP: "SWAP",
        TradeInsType.FUTURES: "FUTURES",
        TradeInsType.OPTION: "OPTION",
    }

    def __init__(self, api_key, secret_key, passphrase, slippage_level=4, ccy="USDT", **kwargs):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.slippage_level = int(slippage_level)
        self.ccy = ccy
        self.adapter = OkxAdapter(api_key=api_key, secret_key=secret_key, passphrase=passphrase)
        
        self.pos_mode = None
        self._pending_orders: Dict[str, Dict] = {}  # key: clOrdId, value: 订单信息
        self._orders_lock = threading.RLock()
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pull_interval = 2.0
    
    def send_order(self, orders: Order | List[Order]):
        """
        组装参数并通过REST API下单
        """
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            inst_id = self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency)
            params = {
                "inst_id": inst_id,
                "td_mode": OkxRestExecutor.__Trade_Mode_Map[order.trade_mode],
                "side": OkxRestExecutor.__Side_Map[order.side],
                "ord_type": "limit",
                "cl_ord_id": "%s" % order.order_id,
            }
            params["ccy"] = order.quote_currency or self.ccy

            # 合约类型特殊处理
            if order.ins_type == TradeInsType.SWAP:
                self.init_account_mode()
                if self.pos_mode == PosMode.LONG_SHORT_MODE:
                    params["pos_side"] = OkxRestExecutor.__Pos_Side_Map[order.side if order.is_open else order.side.switch()]

            # 数量精度校验（所有类型都需要）
            sz = self._check_sz(order)
            params["sz"] = "%s" % sz
            
            # 价格处理
            if order.order_price is not None:
                params["px"] = "%s" % order.order_price
            
            # 订单类型处理
            if order.order_type == OT.MarketOrder:
                params["ord_type"] = "market" if order.ins_type not in [TradeInsType.SWAP, TradeInsType.FUTURES] else "optimal_limit_ioc"
            else:
                # 限价单如果没有价格，从订单簿获取
                if "px" not in params:
                    orderbook = self._get_book(inst_id)
                    if order.side == PS.LONG:
                        params["px"] = orderbook["asks"][-1][0]
                    else:
                        params["px"] = orderbook["bids"][-1][0]
            
            if order.ins_type != TradeInsType.SPOT:
                self.set_leverage(
                    inst_id,
                    params.get("pos_side", ""),
                    order.trade_mode.value,
                    order.leverage
                )
            
            # 调用REST API下单
            try:
                result = self.adapter.place_order(**params)
                if not result or result.get("code") != "0":
                    error_msg = result.get("msg", "未知错误") if result else "无响应"
                    logger.error(f"下单失败: {error_msg}, 订单: {order.order_id}, {result}")
                    raise RuntimeError(f"下单失败: {error_msg}")
                
                # 下单成功，添加到待查询列表
                ord_id = result.get("data", [{}])[0].get("ordId") if result.get("data") else None
                logger.info(f"下单完成: {order.order_id}, 交易所订单ID: {ord_id}")
                with self._orders_lock:
                    self._pending_orders[order.order_id] = {
                        "instId": inst_id,
                        "ordId": ord_id,
                        "order": order,
                        "instType": OkxRestExecutor.__Inst_Type_Map[order.ins_type],
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
            
            inst_id = order_info["instId"]
            result = self.adapter.cancel_order(inst_id=inst_id, cl_ord_id=order_id)
            
            if not result or result.get("code") != "0":
                error_msg = result.get("msg", "未知错误") if result else "无响应"
                logger.error(f"撤单失败: {error_msg}, 订单: {order_id}")
                raise RuntimeError(f"撤单失败: {error_msg}")
            
            logger.info(f"撤单成功: {order_id}")
        except Exception as e:
            logger.error(f"撤单异常: {e}, 订单: {order_id}", exc_info=True)
            raise

    def init_account_mode(self):
        if self.pos_mode:
            return
        account_config = self.adapter.get_account_config()
        if not account_config or account_config["code"] != "0":
            logger.error(f"账户配置获取失败:{account_config.get('msg') if account_config else '无响应'}")
            return
        mode = account_config["data"][0]["posMode"]
        pos_mode = PosMode(mode)
        if pos_mode != PosMode.NET_MODE:
            self.pos_mode = pos_mode
        logger.info(f"当前账户为「{pos_mode.value}」模式")

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def set_leverage(self, symbol, posSide, td_mode, lever):
        res = self.adapter.set_leverage(lever="%s" % lever, mgn_mode=td_mode, inst_id=symbol, pos_side=posSide)
        if not res or res["code"] != "0":
            raise RuntimeError(f"设置杠杆失败:{res.get('msg') if res else '无响应'}")
        logger.info(f"设置杠杆为{lever}成功")

    @retry(max_retries=3, retry_interval=0.2)
    def _get_book(self, symbol):
        orderbook = self.adapter.get_orderbook(inst_id=symbol, sz=self.slippage_level)
        if not orderbook or orderbook["code"] != "0":
            raise RuntimeError(f"获取深度失败:{orderbook.get('msg') if orderbook else '无响应'}")
        return orderbook["data"][0]

    def _check_sz(self, order: Order):
        ins_id = self.adapter.build_inst_id(order.symbol, order.ins_type, order.quote_currency)
        instrument = self._get_instrument(ins_id, OkxRestExecutor.__Inst_Type_Map[order.ins_type])
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        if not order.order_price:
            res = self.adapter.get_mark_price(
                inst_type=OkxRestExecutor.__Inst_Type_Map[order.ins_type if order.ins_type != TradeInsType.SPOT else TradeInsType.MARGIN],
                inst_id=ins_id
            )
            order.order_price = Decimal(res["data"][0]["markPx"])
        lot_sz = instrument["lotSz"]
        ct_val = "1"
        if order.ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES, TradeInsType.OPTION]:
            ct_val = instrument["ctVal"]
        order.sz_value = Decimal(ct_val)
        if not order.is_open:
            sz = Decimal(order.sz) / Decimal(ct_val)
            return sz - (sz % Decimal(lot_sz))
        
        if ins_id.upper().endswith("-USD-SWAP"):
            num = order.order_amount / Decimal(ct_val)
        else:
            num = order.order_amount * order.leverage / (order.order_price * Decimal(ct_val))
        logger.info(f"okx size计算: amount: {order.order_amount}, num: {num}, lot_sz: {lot_sz}, ct_val: {ct_val}, tail: {num % Decimal(lot_sz)}")
        sz = num - (num % Decimal(lot_sz))
        min_sz = instrument["minSz"]
        if sz < Decimal(min_sz):
            raise RuntimeError(f"{ins_id}下单数量 sz {sz}小于最低限制{min_sz}")
        if order.order_type == OT.MarketOrder:
            max_mkt_sz = instrument["maxMktSz"]
            sz = min(sz, Decimal(max_mkt_sz))
        else:
            max_lmt_sz = instrument["maxLmtSz"]
            sz = min(sz, Decimal(max_lmt_sz))
        return Decimal(sz)

    @cached(cache=TTLCache(maxsize=20000, ttl=3600 * 24))
    @retry(max_retries=3, retry_interval=0.5)
    def _get_instrument(self, symbol, ins_type):
        instruments = self.adapter.get_instruments(inst_type=ins_type, inst_id=symbol)
        if not instruments:
            return None
        if instruments['code'] != '0':
            return None
        if len(instruments['data']) == 0:
            return None
        instrument = instruments['data'][0]
        return instrument
    
    @staticmethod
    def _safe_decimal(value, default="0"):
        """
        安全地将值转换为 Decimal
        
        Args:
            value: 要转换的值
            default: 默认值（字符串格式）
            
        Returns:
            Decimal: 转换后的 Decimal 值
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

    def _polling_loop(self):
        """
        订单状态轮询循环
        """
        logger.info("[OKX REST] 订单状态轮询线程已启动")
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
                    self._stop_event.wait(self._pull_interval)
                    continue
                
                # 按 inst_id 分组查询订单状态，减少 API 调用次数
                completed_orders = []
                
                # 按 inst_id 分组
                inst_groups: Dict[str, List[str]] = {}
                for cl_ord_id in cl_ord_ids:
                    order_info = orders_info[cl_ord_id]
                    inst_id = order_info["instId"]
                    if inst_id not in inst_groups:
                        inst_groups[inst_id] = []
                    inst_groups[inst_id].append(cl_ord_id)
                
                # 对每个 inst_id 查询订单状态
                for inst_id, group_cl_ord_ids in inst_groups.items():
                    try:
                        # 获取该组的 inst_type
                        first_order_info = orders_info[group_cl_ord_ids[0]]
                        inst_type = first_order_info["instType"]
                        
                        # 查询该交易对的所有订单状态
                        result = self.adapter.get_orders(
                            inst_type=inst_type,
                            inst_id=inst_id
                        )
                        
                        if not result or result.get("code") != "0":
                            logger.warning(f"查询订单状态失败: {result.get('msg') if result else '无响应'}, inst_id: {inst_id}")
                            continue
                        
                        # 处理查询结果，只处理我们关心的订单
                        orders_data = result.get("data", [])
                        found_cl_ord_ids = set()
                        
                        for order_data in orders_data:
                            cl_ord_id = order_data.get("clOrdId")
                            if not cl_ord_id or cl_ord_id not in orders_info:
                                continue
                            
                            found_cl_ord_ids.add(cl_ord_id)
                            order_info = orders_info[cl_ord_id]
                            order = order_info["order"]
                            state = order_data.get("state")
                            
                            # 如果订单已完成，处理并标记为完成
                            if state in ["filled", "canceled", "partially_filled"]:
                                instrument = self._get_instrument(order_data["instId"], order_data["instType"])
                                if not instrument:
                                    logger.warning(f"无法获取交易对信息: {order_data['instId']}")
                                    continue
                                
                                # 安全获取 ctVal，确保可以转换为 Decimal
                                ct_val_str = instrument.get("ctVal") or "1"
                                try:
                                    ct_val = Decimal(str(ct_val_str))
                                except (ValueError, TypeError, InvalidOperation):
                                    ct_val = Decimal("1")
                                    logger.warning(f"无法解析 ctVal: {ct_val_str}, 使用默认值 1")
                                
                                # 构造 OrderUpdateMessage（与 WebSocket 版本保持一致）
                                oum = OrderUpdateMessage(
                                    order_id=cl_ord_id,
                                    market_order_id=order_data.get("ordId", ""),
                                    execution_price=self._safe_decimal(order_data.get("avgPx"), "0"),
                                    sz=self._safe_decimal(order_data.get("accFillSz"), "0") * ct_val,
                                    fee=self._safe_decimal(order_data.get("fee"), "0"),
                                    pnl=self._safe_decimal(order_data.get("pnl"), "0"),
                                    unrealized_pnl=Decimal(0),
                                    friction=Decimal(0),
                                    sz_value=ct_val,
                                )
                                
                                # 计算结算金额（与 WebSocket 版本保持一致）
                                lever = self._safe_decimal(order_data.get("lever"), "1")
                                oum.settle_amount = oum.execution_price * oum.sz / lever
                                if order_data.get('side') == 'buy' and order_data.get('posSide') == 'short':
                                    oum.settle_amount = (oum.execution_price * oum.sz + oum.pnl) / lever + oum.pnl
                                if order_data.get('side') == 'sell' and order_data.get('posSide') == 'long':
                                    oum.settle_amount = (oum.execution_price * oum.sz - oum.pnl) / lever + oum.pnl
                                
                                # 设置订单状态（与 WebSocket 版本保持一致）
                                oum.order_status = OrderStatus("canceled" if state == "mmp_canceled" else state)
                                if oum.order_status == OrderStatus.FILLED or oum.order_status == OrderStatus.PARTIALLY_FILLED:
                                    fill_time = order_data.get("fillTime")
                                    if fill_time:
                                        oum.finish_time = DateTimeUtils.to_datetime(int(fill_time))
                                else:
                                    u_time = order_data.get("uTime")
                                    if u_time:
                                        oum.finish_time = DateTimeUtils.to_datetime(int(u_time))
                                
                                logger.info(f"OKX REST订单更新: {cl_ord_id} -> {oum}")
                                self._trade_callback(oum)
                                
                                # 标记为已完成
                                completed_orders.append(cl_ord_id)
                        
                        # 对于不在待成交列表中的订单，如果之前有 ordId，单独查询订单状态
                        # 这可能表示订单已完成或已取消
                        for cl_ord_id in group_cl_ord_ids:
                            if cl_ord_id in found_cl_ord_ids:
                                continue  # 已经在待成交列表中处理过了
                            
                            order_info = orders_info.get(cl_ord_id)
                            if not order_info:
                                continue
                            
                            ord_id = order_info.get("ordId")
                            if ord_id:
                                # 使用 ordId 单独查询订单状态
                                try:
                                    order_result = self.adapter.get_order(inst_id=inst_id, ord_id=ord_id)
                                    if order_result and order_result.get("code") == "0":
                                        order_data_list = order_result.get("data", [])
                                        if order_data_list:
                                            order_data = order_data_list[0]
                                            state = order_data.get("state")
                                            if state in ["filled", "canceled", "partially_filled"]:
                                                # 处理已完成的订单
                                                instrument = self._get_instrument(order_data["instId"], order_data["instType"])
                                                if instrument:
                                                    # 安全获取 ctVal，确保可以转换为 Decimal
                                                    ct_val_str = instrument.get("ctVal") or "1"
                                                    try:
                                                        ct_val = Decimal(str(ct_val_str))
                                                    except (ValueError, TypeError, InvalidOperation):
                                                        ct_val = Decimal("1")
                                                        logger.warning(f"无法解析 ctVal: {ct_val_str}, 使用默认值 1")
                                                    
                                                    oum = OrderUpdateMessage(
                                                        order_id=cl_ord_id,
                                                        market_order_id=order_data.get("ordId", ""),
                                                        execution_price=self._safe_decimal(order_data.get("avgPx"), "0"),
                                                        sz=self._safe_decimal(order_data.get("accFillSz"), "0") * ct_val,
                                                        fee=self._safe_decimal(order_data.get("fee"), "0"),
                                                        pnl=self._safe_decimal(order_data.get("pnl"), "0"),
                                                        unrealized_pnl=Decimal(0),
                                                        friction=Decimal(0),
                                                        sz_value=ct_val,
                                                    )
                                                    
                                                    # 计算结算金额（与 WebSocket 版本保持一致）
                                                    lever = self._safe_decimal(order_data.get("lever"), "1")
                                                    oum.settle_amount = oum.execution_price * oum.sz / lever
                                                    if order_data.get('side') == 'buy' and order_data.get('posSide') == 'short':
                                                        oum.settle_amount = (oum.execution_price * oum.sz + oum.pnl) / lever + oum.pnl
                                                    if order_data.get('side') == 'sell' and order_data.get('posSide') == 'long':
                                                        oum.settle_amount = (oum.execution_price * oum.sz - oum.pnl) / lever + oum.pnl
                                                    
                                                    # 设置订单状态（与 WebSocket 版本保持一致）
                                                    oum.order_status = OrderStatus("canceled" if state == "mmp_canceled" else state)
                                                    if oum.order_status == OrderStatus.FILLED or oum.order_status == OrderStatus.PARTIALLY_FILLED:
                                                        fill_time = order_data.get("fillTime")
                                                        if fill_time:
                                                            oum.finish_time = DateTimeUtils.to_datetime(int(fill_time))
                                                    else:
                                                        u_time = order_data.get("uTime")
                                                        if u_time:
                                                            oum.finish_time = DateTimeUtils.to_datetime(int(u_time))
                                                    
                                                    logger.info(f"OKX REST订单更新（单独查询）: {cl_ord_id} -> {oum}")
                                                    self._trade_callback(oum)
                                                    completed_orders.append(cl_ord_id)
                                except Exception as e:
                                    logger.warning(f"单独查询订单状态失败: {e}, cl_ord_id: {cl_ord_id}", exc_info=True)
                    
                    except Exception as e:
                        logger.error(f"批量查询订单状态异常: {e}", exc_info=True)
                        continue
                
                # 从待查询列表中删除已完成的订单
                if completed_orders:
                    with self._orders_lock:
                        for cl_ord_id in completed_orders:
                            self._pending_orders.pop(cl_ord_id, None)
                
                # 等待1秒后继续下一轮查询
                self._stop_event.wait(self._pull_interval)
            
            except Exception as e:
                logger.error(f"订单状态轮询异常: {e}", exc_info=True)
                self._stop_event.wait(self._pull_interval)
        
        logger.info("[OKX REST] 订单状态轮询线程已停止")

    def on_start(self):
        """
        启动执行器，初始化订单列表并启动轮询线程
        """
        logger.info("[OKX REST] 启动执行器")
        self._pending_orders.clear()
        self._stop_event.clear()
        
        # 启动轮询线程
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()
        logger.info("[OKX REST] 执行器已启动")

    def on_stop(self):
        """
        停止执行器，停止轮询线程并清理资源
        """
        logger.info("[OKX REST] 停止执行器")
        
        # 设置停止标志
        self._stop_event.set()
        
        # 等待轮询线程结束（最多等待5秒）
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)
            if self._polling_thread.is_alive():
                logger.warning("[OKX REST] 轮询线程未能在5秒内停止")
        
        # 清理订单列表
        with self._orders_lock:
            self._pending_orders.clear()
        
        logger.info("[OKX REST] 执行器已停止")
