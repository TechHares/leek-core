#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX交易执行模块
"""

import json
import asyncio
import time
import websocket
from cachetools import cached, TTLCache
from decimal import Decimal

from models import PositionSide as PS, OrderType as OT, TradeMode, TradeInsType, PosMode
from models import Order
from .base import Executor, WebSocketExecutor, WSStatus
from utils.event_bus import EventBus
from utils import get_logger
from utils import decimal_quantize
from okx.Account import AccountAPI
from okx.utils import sign
from okx.MarketData import MarketAPI
from okx.PublicData import PublicAPI
from okx.Trade import TradeAPI

logger = get_logger(__name__)
LOCK = asyncio.Lock()


class OkxWebSocketExecutor(WebSocketExecutor):
    """
    OKX交易所WebSocket异步执行器，支持自动重连、心跳、鉴权、频道订阅、订单推送、下单、撤单等。
    集成原OkxTrader的业务参数组装、映射和风控逻辑。
    """
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

    def __init__(self, callback, api_key, api_secret_key, passphrase, *, slippage_level=4, td_mode="isolated", ccy="", leverage=3, order_type="limit", trade_ins_type=3, **kwargs):
        super().__init__(callback, "wss://ws.okx.com:8443/ws/v5/private", **kwargs)
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.passphrase = passphrase
        self._login_ok = False
        self._order_cache = {}
        self.slippage_level = int(slippage_level)
        self.td_mode = TradeMode(td_mode)
        self.ccy = ccy
        self.lever = int(leverage)
        self.order_type = OT.LimitOrder if order_type == "limit" else OT.MarketOrder
        self.trade_ins_type = TradeInsType(int(trade_ins_type))
        self.market_client = MarketAPI(domain="https://www.okx.com", flag="0", debug=False, proxy=None)
        self.public_client = PublicAPI(domain="https://www.okx.com", flag="0", debug=False, proxy=None)

    async def on_open(self):
        # 登录鉴权
        timestamp = str(int(asyncio.get_event_loop().time()))
        s = sign(timestamp + "GET" + "/users/self/verify", self.api_secret_key).decode()
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
        logger.info("[OKX] 已发送鉴权请求")

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
                    await self._subscribe_channels()
                    logger.info("[OKX] 登录成功并已订阅频道")
                return
            # 订单/持仓推送等业务逻辑
            await self._handle_push(msg)
        except Exception as e:
            logger.error(f"OKX消息处理异常: {e}")

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
        logger.info("[OKX] 已发送频道订阅请求")

    async def _handle_push(self, msg):
        # 这里只做简单转发，实际可按频道细分业务
        if "data" in msg and "arg" in msg:
            # 订单/持仓/成交等
            await self.callback(msg)

    async def send_heartbeat(self):
        # OKX心跳采用ping
        await self.send(json.dumps({"op": "ping"}))
        logger.debug("[OKX] 发送心跳ping")

    async def send_order(self, order: Order):
        """
        组装参数并通过WebSocket异步下单
        """
        # 参数补全与映射
        if order.ccy is None:
            order.ccy = self.ccy
        if order.trade_mode is None:
            order.trade_mode = self.td_mode
        if order.lever is None:
            order.lever = self.lever
        if order.type is None:
            order.type = self.order_type
        if order.trade_ins_type is None:
            order.trade_ins_type = self.trade_ins_type
        if order.trade_ins_type == TradeInsType.SPOT:
            order.lever = 1
            order.trade_mode = TradeMode.CASH
        args = {
            "tdMode": OkxWebSocketExecutor.__Trade_Mode_Map[order.trade_mode],
            "instId": order.symbol,
            "clOrdId": "%s" % order.order_id,
            "side": OkxWebSocketExecutor.__Side_Map[order.side],
            "ordType": "limit",
        }
        if order.trade_mode == TradeMode.CROSS:
            args["ccy"] = self.ccy
        # 合约类型特殊处理
        if order.trade_ins_type == TradeInsType.SWAP:
            # 省略账户模式
            if hasattr(order, "pos_type") and order.pos_type:
                args["posSide"] = OkxWebSocketExecutor.__Pos_Side_Map[order.pos_type]
            else:
                args["posSide"] = OkxWebSocketExecutor.__Pos_Side_Map[order.side]
        # 数量精度校验
        sz = self._check_sz(order)
        args["sz"] = "%s" % sz
        if order.price is not None:
            args["px"] = "%s" % order.price
        if order.type == OT.MarketOrder:
            if order.trade_ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES]:
                args["ordType"] = "optimal_limit_ioc"
            elif self.slippage_level == 0:
                args["ordType"] = "market"
            else:
                orderbook = self._get_book(order.symbol)
                if order.side == PS.LONG:
                    args["px"] = orderbook["asks"][-1][0]
                else:
                    args["px"] = orderbook["bids"][-1][0]
        # 发送WebSocket下单消息（示例，需根据OKX ws协议格式封装）
        await self.send_ws_order(args)
        return args

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

    def _get_book(self, symbol):
        orderbook = self.market_client.get_orderbook(symbol, self.slippage_level)
        if not orderbook or orderbook["code"] != "0":
            logger.error(f"深度获取失败:{orderbook['msg'] if orderbook else orderbook}")
        return orderbook["data"][0]

    def _check_sz(self, order: Order):
        instrument = self._get_instrument(order.symbol, order.trade_ins_type)
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        if not order.price:
            res = self.public_client.get_mark_price(
                OkxWebSocketExecutor.__Inst_Type_Map[order.trade_ins_type if order.trade_ins_type != TradeInsType.SPOT else TradeInsType.MARGIN],
                instId=order.symbol
            )
            order.price = Decimal(res["data"][0]["markPx"])
        lot_sz = instrument["lotSz"]
        if order.sz:
            sz = Decimal(order.sz)
            return sz - (sz % Decimal(lot_sz))
        ct_val = "1"
        if order.trade_ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES, TradeInsType.OPTION]:
            ct_val = instrument["ctVal"]
        if order.symbol.upper().endswith("-USD-SWAP"):
            num = order.amount / Decimal(ct_val)
        else:
            num = order.amount * order.lever / (order.price * Decimal(ct_val))
        sz = num - (num % Decimal(lot_sz))
        min_sz = instrument["minSz"]
        if sz < Decimal(min_sz):
            raise RuntimeError(f"下单数量 sz {sz}小于最低限制{min_sz}")
        if order.type == OT.MarketOrder:
            max_mkt_sz = instrument["maxMktSz"]
            sz = min(sz, Decimal(max_mkt_sz))
        else:
            max_lmt_sz = instrument["maxLmtSz"]
            sz = min(sz, Decimal(max_lmt_sz))
        return Decimal(sz)

    def _get_instrument(self, symbol, ins_type=TradeInsType.SWAP):
        instruments = self.public_client.get_instruments(instType=OkxWebSocketExecutor.__Inst_Type_Map[ins_type], instId=symbol)
        if not instruments:
            return None
        if instruments['code'] != '0':
            return None
        if len(instruments['data']) == 0:
            return None
        instrument = instruments['data'][0]
        return instrument

    # 发送WebSocket下单消息（需根据OKX协议实现）
    async def send_ws_order(self, args):
        msg = {
            "op": "order",
            "args": [args]
        }
        await self.send(json.dumps(msg))

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
        logger.error(f"[OKX] WebSocket异常: {error}")
