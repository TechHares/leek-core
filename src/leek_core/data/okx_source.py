#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX WebSocket 数据源实现。
继承自 WebSocketDataSource，用于连接 OKX 并订阅 K 线数据。
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Iterator

import websockets
from leek_core.adapts import OkxAdapter

from leek_core.models import TimeFrame, DataType, Field, FieldType, ChoiceType, TradeInsType, AssetType, KLine
from leek_core.utils import get_logger, log_method, retry
from .websocket import WebSocketDataSource

logger = get_logger(__name__)

# OKX TimeFrame to channel string mapping (for WebSocket message parsing and UI options)
OKX_TIMEFRAME_MAP = {
    TimeFrame.M1: "1m", TimeFrame.M3: "3m", TimeFrame.M5: "5m",
    TimeFrame.M15: "15m", TimeFrame.M30: "30m", TimeFrame.H1: "1H",
    TimeFrame.H2: "2H", TimeFrame.H4: "4H", TimeFrame.H6: "6H",
    TimeFrame.H12: "12H", TimeFrame.D1: "1D", TimeFrame.W1: "1W",
    TimeFrame.MON1: "1M",
}
OKX_CHANNEL_MAP = {v: k for k, v in OKX_TIMEFRAME_MAP.items()}


class OkxDataSource(WebSocketDataSource):
    """
    OKX WebSocket K 线数据源。
    """
    display_name: str = "OKX行情"
    supported_data_type: DataType = DataType.KLINE
    supported_asset_type: DataType = AssetType.CRYPTO
    init_params: List[Field] = [
        Field(name="symbols", label="标的", type=FieldType.ARRAY, default=[], choice_type=ChoiceType.STRING),
    ]

    def __init__(self, symbols: List[str] = None):
        """
        初始化 OKX WebSocket 数据源。

        参数:
            name: 数据源名称
            instance_id: 数据源实例ID
        """
        self.ws_domain = "wss://ws.okx.com:8443/ws/v5/business"
        super().__init__(ws_url=self.ws_domain, ping_interval=25, ping_timeout=10)
        self.symbols = symbols or []
        self.subscribed_channels: Dict[str, int] = {}

        self.pre_time = {}
        self.adapter = OkxAdapter()

    def on_connect(self):
        """连接成功后调用"""
        logger.info(f"OKX数据源连接成功，使用websockets内置心跳")

    def on_disconnect(self):
        """断开连接前调用"""
        logger.info(f"OKX数据源断开连接")

    async def on_message(self, message: str):
        """
        处理收到的WebSocket消息
        
        参数:
            message: 收到的消息
        """
        # 处理心跳响应
        if message == "pong":
            return

        try:
            data = json.loads(message)
            if "event" in data:
                event = data.get("event")
                arg = data.get('arg', {})
                if event == "subscribe":
                    logger.info(f"OKX订阅确认: {arg}")
                elif event == "unsubscribe":
                    logger.info(f"OKX取消订阅确认: {arg}")
                elif event == "error":
                    logger.error(f"OKX WS错误: {data.get('msg')} (代码: {data.get('code')}) 参数: {arg}")
                return

            await self._process_kline_data(data)
        except json.JSONDecodeError:
            logger.warning(f"OKX非JSON消息: {message[:100] if isinstance(message, str) else str(message)[:100]}...")
        except Exception as e:
            logger.error(f"处理OKX消息时出错: {e}", exc_info=True)

    async def _process_kline_data(self, data: Dict):
        """处理K线数据"""
        try:
            # 确保是K线数据
            if "arg" not in data or "data" not in data:
                return

            arg = data.get('arg', {})

            channel = arg.get('channel', '')
            inst_id = arg.get('instId', '')
            if not channel or not inst_id:
                return

            # 检查是否是K线数据
            if not channel.startswith('candle'):
                return

            # 获取K线数据
            candle_data = data.get('data', [])
            if not candle_data:
                return

            # 从channel提取时间周期
            tf_str = channel[6:]  # 移除 'candle' 前缀
            tf = OKX_CHANNEL_MAP.get(tf_str)
            if not tf:
                logger.error(f"未知的OKX时间周期: {tf_str}")
                return

            # 处理每个K线数据并转换为标准格式
            for row in candle_data:
                if len(row) < 9:
                    logger.warning(f"不完整的OKX K线数据: {row}")
                    continue

                try:
                    # 解析交易对和计价币种
                    symbol_parts = inst_id.split('-')
                    base_currency = symbol_parts[0]
                    quote_currency = symbol_parts[1] if len(symbol_parts) > 1 else "USDT"

                    # 确定交易类型
                    ins_type = TradeInsType.SPOT
                    if len(symbol_parts) > 2:
                        if symbol_parts[2] == "SWAP":
                            ins_type = TradeInsType.SWAP
                        elif symbol_parts[2] == "FUTURES":
                            ins_type = TradeInsType.FUTURES

                    # 转换时间戳
                    timestamp = int(row[0])
                    end_time = timestamp + tf.milliseconds
                    # 创建KLine对象
                    kline = KLine(
                        symbol=base_currency,
                        market='okx',
                        open=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        volume=row[5],
                        amount=row[7],
                        start_time=timestamp,
                        end_time=end_time,
                        current_time=int(time.time() * 1000),
                        timeframe=tf,
                        quote_currency=quote_currency,
                        ins_type=ins_type,
                        is_finished=int(row[8]) == 1
                    )
                    kline.asset_type = AssetType.CRYPTO

                    if kline.start_time <= self.pre_time.get(kline.row_key, 0):
                        continue
                    if kline.is_finished:
                        self.pre_time[kline.row_key] = kline.start_time
                    # 调用订阅的回调函数
                    self.send_data(kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"解析OKX K线数据时出错: {e}, 原始数据: {row}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"处理OKX K线数据时出错: {e}", exc_info=True)

    def parse_row_key(self, symbols: List[str] = list, timeframes: List[Union[TimeFrame, str]] = list,
                  ins_types: List[TradeInsType] = list, quote_currencies: List[str] = list, **kwargs) -> List[tuple]:
        for s in symbols:
            for q in quote_currencies:
                for i in ins_types:
                    for t in timeframes:
                        yield KLine.pack_row_key(s, q, i, t)

    @log_method()
    def subscribe(self, row_key: str) -> bool:
        """
        订阅OKX K线数据
        
        参数:
            symbol: 交易对，如'BTC-USDT'
            timeframe: 时间周期，如TimeFrame.M1或'1m'
            ins_type: 交易类型，如TradeInsType.SWAP
            quote_currency: 计价币种，如'USDT'
            
        返回:
            bool: 订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        # 获取OKX格式的时间周期字符串
        tf_value = self.adapter.get_okx_timeframe(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和OKX订阅消息
        channel = f"candle{tf_value}"
        inst_id = self.adapter.build_inst_id(symbol, ins_type, quote_currency)

        # 发送订阅请求
        msg = {"op": "subscribe", "args": [{"channel": channel, "instId": inst_id}]}
        key = f"{channel}:{inst_id}"
        if key in self.subscribed_channels:
            self.subscribed_channels[key] += 1
            return True
        self.subscribed_channels[key] = 1
        return self.async_send(json.dumps(msg))



    @log_method()
    def unsubscribe(self, row_key: str) -> bool:
        """
        取消订阅OKX K线数据
        
        参数:
            symbol: 交易对，如'BTC-USDT'
            timeframe: 时间周期，如TimeFrame.M1或'1m'
            ins_type: 交易类型，如TradeInsType.SWAP
            quote_currency: 计价币种，如'USDT'
            
        返回:
            bool: 取消订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        # 获取OKX格式的时间周期字符串
        tf_value = self.adapter.get_okx_timeframe(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和OKX订阅消息
        channel = f"candle{tf_value}"
        inst_id = self.adapter.build_inst_id(symbol, ins_type, quote_currency)

        # 发送取消订阅请求
        msg = {"op": "unsubscribe", "args": [{"channel": channel, "instId": inst_id}]}
        key = f"{channel}:{inst_id}"
        if key in self.subscribed_channels and self.subscribed_channels[key] > 1:
            self.subscribed_channels[key] -= 1
            return True
        self.subscribed_channels.pop(key, None)
        return self.async_send(json.dumps(msg))

    def get_supported_parameters(self) -> List[Field]:
        if self.symbols is None or len(self.symbols) == 0:
            tickers = self.adapter.get_tickers(inst_type="SWAP")
            symbols = set([ticker["instId"].split("-")[0] for ticker in tickers["data"]])
            tickers = self.adapter.get_tickers(inst_type="SPOT")
            symbols |= set([ticker["instId"].split("-")[0] for ticker in tickers["data"]])
            self.symbols = list(symbols)
        ins_types = [(TradeInsType.SPOT.value, str(TradeInsType.SPOT)),
                     (TradeInsType.SWAP.value, str(TradeInsType.SWAP))]
        return [
            Field(name='symbols', label='交易标的', type=FieldType.SELECT, required=True, choices=list(self.symbols),
                  choice_type=ChoiceType.STRING),
            Field(name='timeframes', label='时间周期', type=FieldType.SELECT, required=True,
                  choices=list(OKX_TIMEFRAME_MAP.keys()), choice_type=ChoiceType.STRING),
            Field(name='quote_currencies', label='计价币种', type=FieldType.SELECT, required=True,
                  choices=["USDT"], choice_type=ChoiceType.STRING),
            Field(name='ins_types', label='交易标的类型', type=FieldType.SELECT, required=True,
                  choices=ins_types, choice_type=ChoiceType.INT),
        ]

    @log_method(log_result=False)
    @retry(max_retries=3, retry_interval=1)
    def get_history_data(
            self,
            row_key: str,
            start_time: datetime | int = None,
            end_time: datetime | int = None,
            limit: int = None
    ) -> Iterator[KLine]:
        """
        获取K线数据，WebSocket数据源通常不支持历史K线查询

        参数:
            symbol: 交易对符号
            quote_currency: 计价币种
            ins_type: 交易品种类型
            timeframe: K线时间周期
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制

        返回:
            Iterator[KLine]: K线数据迭代器（空）
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        if not isinstance(ins_type, TradeInsType):
            ins_type = TradeInsType(ins_type)
        if limit is None:
            limit = 100

        before = ""
        if start_time is not None and isinstance(start_time, datetime):
            before = int(start_time.timestamp() * 1000)
        if start_time is not None and isinstance(start_time, int):
            before = start_time
        after = int(time.time() * 1000)
        if end_time is not None and isinstance(end_time, datetime):
            after = int(end_time.timestamp() * 1000)

        inst_id = self.adapter.build_inst_id(symbol, ins_type, quote_currency)
        interval = self.adapter.get_okx_timeframe(timeframe)
        page_size = min(100, limit)
        res = []
        while len(res) < limit:
            candlesticks = self.adapter.get_history_candlesticks(inst_id=inst_id, bar=interval, limit=page_size, before=before, after=after)
            if len(candlesticks["data"]) == 0:
                break
            for row in candlesticks["data"]:
                # 创建KLine对象
                kline = KLine(
                    data_type=DataType.KLINE,
                    symbol=symbol,
                    market='okx',
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5],
                    amount=row[7],
                    start_time=int(row[0]),
                    end_time=int(row[0]) + timeframe.milliseconds,
                    current_time=int(time.time() * 1000),
                    timeframe=timeframe,
                    quote_currency=quote_currency,
                    ins_type=ins_type,
                    is_finished=int(row[8]) == 1
                )
                after = int(row[0]) - 1
                res.append(kline)
        res = res[:limit]
        res.reverse()
        return iter(res)


