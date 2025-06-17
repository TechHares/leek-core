#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX交易执行模块
"""

import asyncio
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import List

from cachetools import TTLCache, cached
from okx.MarketData import MarketAPI
from okx.PublicData import PublicAPI
from okx.Account import AccountAPI
from okx.utils import sign

from leek_core.models import Order, PosMode, OrderStatus, OrderUpdateMessage
from leek_core.models import PositionSide as PS, OrderType as OT, TradeMode, TradeInsType, Field, FieldType, ChoiceType
from leek_core.utils import get_logger, generate_str, DateTimeUtils
from .base import WebSocketExecutor

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
        super().__init__(ws_url="wss://ws.okx.com:8443/ws/v5/private", heartbeat_interval=25, **kwargs)
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self._login_ok = False
        self.slippage_level = int(slippage_level)
        self.ccy = ccy
        self.market_client = MarketAPI(domain="https://www.okx.com", flag="0", debug=False, proxy=None)
        self.public_client = PublicAPI(domain="https://www.okx.com", flag="0", debug=False, proxy=None)
        self.account = AccountAPI(api_key=api_key, api_secret_key=secret_key, passphrase=passphrase,
                                          domain="https://www.okx.com", flag="0", debug=False, proxy=None)

        self.pos_mode = None

    async def on_open(self):
        # 登录鉴权，失败时自动重试，重试间隔和次数与连接一致
        retry = 0
        while retry < getattr(self, 'max_retries', 5):
            timestamp = str(int(time.time()))
            s = sign(timestamp + "GET" + "/users/self/verify", self.secret_key).decode()
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
                        msg = OrderUpdateMessage(
                            order_id=order_id,
                            market_order_id=data["ordId"],
                            execution_price=Decimal(data["avgPx"]),
                            sz = Decimal(data["accFillSz"]) / Decimal(instrument["ctVal"]),
                            fee=Decimal(data["fee"]),
                            pnl=Decimal(data["pnl"]),
                            unrealized_pnl=Decimal(0),
                            friction=Decimal(0),
                            sz_value=Decimal(instrument["ctVal"]),
                        )
                        msg.settle_amount = msg.execution_price * msg.sz
                        msg.order_status = OrderStatus("canceled" if data["state"] == "mmp_canceled" else data["state"])
                        if msg.order_status == OrderStatus.FILLED or msg.order_status == OrderStatus.PARTIALLY_FILLED:
                            msg.finish_time = DateTimeUtils.to_datetime(int(data["fillTime"]))
                        else:
                            msg.finish_time = DateTimeUtils.to_datetime(int(data["uTime"]))
                        self._trade_callback(msg)
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

    async def send_heartbeat(self):
        # OKX心跳采用ping
        await self.send("ping")
        logger.debug("[OKX] 发送心跳ping")

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
                "instId": self._get_inst_id(order),
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
                    orderbook = self._get_book(self._get_inst_id(order))
                    if order.side == PS.LONG:
                        params["px"] = orderbook["asks"][-1][0]
                    else:
                        params["px"] = orderbook["bids"][-1][0]
            orders_args.append(params)
            if order.ins_type != TradeInsType.SPOT:
                self.set_leverage(self._get_inst_id(order), params["posSide"] if "posSide" in params else "", order.trade_mode.value,
                                  order.leverage)
        # 发送WebSocket下单消息（示例，需根据OKX ws协议格式封装）
        logger.info(f"下单：{orders_args}")
        self.send_ws_order(orders_args)

    def init_account_mode(self):
        if self.pos_mode:
            return
        account_config = self.account.get_account_config()
        if not account_config or account_config["code"] != "0":
            logger.error(f"账户配置获取失败:{account_config['msg'] if account_config else account_config}")
        mode = account_config["data"][0]["posMode"]
        pos_mode = PosMode(mode)
        if pos_mode != PosMode.NET_MODE:
            self.pos_mode = pos_mode
        logger.info(f"当前账户为「{pos_mode.value}」模式")

    @cached(cache=TTLCache(maxsize=200, ttl=3600 * 10))
    def set_leverage(self, symbol, posSide, td_mode, lever):
        res = self.account.set_leverage(lever="%s" % lever, mgnMode=td_mode, instId=symbol, posSide=posSide)
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

    def _get_book(self, symbol):
        orderbook = self.market_client.get_orderbook(symbol, self.slippage_level)
        if not orderbook or orderbook["code"] != "0":
            raise RuntimeError(f"获取深度失败:{orderbook['msg'] if orderbook else orderbook}")
        return orderbook["data"][0]
    
    def _get_inst_id(self, order: Order):
        ins_type = order.ins_type
        symbol = order.symbol
        quote_currency = order.quote_currency
        if ins_type == TradeInsType.SWAP or ins_type == TradeInsType.FUTURES:
            return f"{symbol}-{quote_currency}-SWAP"
        elif ins_type == TradeInsType.OPTION:
            return f"{symbol}-{quote_currency}-OPTION"
        elif ins_type == TradeInsType.SPOT or ins_type == TradeInsType.FUTURES:
            return f"{symbol}-{quote_currency}"
        raise ValueError(f"不支持的交易类型: {ins_type}")

    def _get_order_extra(self, order: Order, key: str):
        extra = order.extra or {}
        info = extra.get(self.instance_id)
        if not info or key not in info:
            return None
        return info[key]

    def _check_sz(self, order: Order):
        ins_id = self._get_inst_id(order)
        instrument = self._get_instrument(ins_id, OkxWebSocketExecutor.__Inst_Type_Map[order.ins_type])
        if not instrument:
            raise RuntimeError("交易信息获取失败")
        if not order.order_price:
            res = self.public_client.get_mark_price(
                OkxWebSocketExecutor.__Inst_Type_Map[order.ins_type if order.ins_type != TradeInsType.SPOT else TradeInsType.MARGIN],
                instId=ins_id
            )
            order.order_price = Decimal(res["data"][0]["markPx"])
        lot_sz = instrument["lotSz"]
        ct_val = "1"
        if order.ins_type in [TradeInsType.SWAP, TradeInsType.FUTURES, TradeInsType.OPTION]:
            ct_val = instrument["ctVal"]
        order.sz_value = Decimal(ct_val)
        if not order.is_open:
            sz = Decimal(order.sz) * Decimal(ct_val)
            return sz - (sz % Decimal(lot_sz))
        
        if ins_id.upper().endswith("-USD-SWAP"):
            num = order.order_amount / Decimal(ct_val)
        else:
            num = order.order_amount * order.leverage / (order.order_price * Decimal(ct_val))
        sz = num - (num % Decimal(lot_sz))
        min_sz = instrument["minSz"]
        if sz < Decimal(min_sz):
            raise RuntimeError(f"下单数量 sz {sz}小于最低限制{min_sz}")
        if order.order_type == OT.MarketOrder:
            max_mkt_sz = instrument["maxMktSz"]
            sz = min(sz, Decimal(max_mkt_sz))
        else:
            max_lmt_sz = instrument["maxLmtSz"]
            sz = min(sz, Decimal(max_lmt_sz))
        return Decimal(sz)

    @cached(cache=TTLCache(maxsize=200, ttl=3600 * 10))
    def _get_instrument(self, symbol, ins_type):
        instruments = self.public_client.get_instruments(instType=ins_type, instId=symbol)
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
        logger.error(f"[OKX] WebSocket异常: {error}", exc_info=True)
