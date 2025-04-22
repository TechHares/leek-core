#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX WebSocket 数据源实现。
继承自 WebSocketDataSource，用于连接 OKX 并订阅 K 线数据。
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable, Iterator
import pandas as pd
import websockets
from datetime import datetime

from .common import WebSocketDataSource
from models import TimeFrame, KLine, TradeInsType, DataType, Field, FieldType, ChoiceType
from utils import get_logger
from models.data import KLine
from models.constants import TradeInsType, AssetType

logger = get_logger(__name__)

# OKX TimeFrame to channel string mapping
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

    supported_data_type: DataType = DataType.KLINE
    supported_asset_type: DataType = AssetType.CRYPTO
    init_params: List[Field] = [
        Field(name="work_flag", label="环境", type=FieldType.RADIO, default="0", choices=[("0", "生产"), ("1", "模拟盘"), ("2", "生产(AWS)")], choice_type=ChoiceType.STR),
    ]
    verbose_name = "OKX K线"

    def __init__(self, name: str = "okx_ws",
                 instance_id: str = None):
        """
        初始化 OKX WebSocket 数据源。

        参数:
            name: 数据源名称
            instance_id: 数据源实例ID
        """
        self.ws_domain = "wss://ws.okx.com:8443/ws/v5/business"
        super().__init__(name=name, ws_url=self.ws_domain, instance_id=instance_id)
        self._loop = asyncio.get_event_loop()
        self._ping_interval = 25
        self._ping_task: Optional[asyncio.Task] = None

    def on_connect(self):
        """连接成功后启动心跳任务"""
        if self._ping_task:
            self._ping_task.cancel()

        if self._loop:
            self._ping_task = asyncio.run_coroutine_threadsafe(
                self._send_ping_loop(),
                self._loop
            )
            logger.info(f"OKX '{self.name}' 心跳任务已启动")

    def on_disconnect(self):
        """断开连接前取消心跳任务"""
        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None
            logger.info(f"OKX '{self.name}' 心跳任务已停止")

    async def _send_ping_loop(self):
        """OKX心跳任务"""
        while self.is_connected and self._connection:
            try:
                await asyncio.sleep(self._ping_interval)
                if self.is_connected and self._connection:
                    logger.debug(f"发送心跳到OKX")
                    await self._connection.send("ping")
            except asyncio.CancelledError:
                logger.info("OKX心跳循环已取消")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.warning("发送心跳时连接已关闭")
                break
            except Exception as e:
                logger.error(f"OKX心跳循环出错: {e}", exc_info=True)
                await asyncio.sleep(5)

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
                    current_time = timestamp
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
                        current_time=current_time,
                        timeframe=tf,
                        quote_currency=quote_currency,
                        ins_type=ins_type,
                        data_source_id=self.instance_id,
                        is_finished=int(row[8]) == 1
                    )

                    # 调用订阅的回调函数
                    self._callback(kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"解析OKX K线数据时出错: {e}, 原始数据: {row}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"处理OKX K线数据时出错: {e}", exc_info=True)

    def subscribe(self, symbol: str = "BTC", timeframe: Union[TimeFrame, str] = TimeFrame.M1,
                  ins_type: TradeInsType = TradeInsType.SWAP, quote_currency: str = "USDT", **kwargs) -> bool:
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
        # 检查WebSocket连接
        if not self.is_connected:
            logger.error("OKX未连接，无法订阅")
            return False
        # 获取OKX格式的时间周期字符串
        tf_value = self._get_okx_tf_value(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和OKX订阅消息
        channel = f"candle{tf_value}"
        inst_id = self.build_inst_id(symbol, ins_type, quote_currency)

        # 发送订阅请求
        msg = {"op": "subscribe", "args": [{"channel": channel, "instId": inst_id}]}
        return self.async_send(json.dumps(msg))

    @staticmethod
    def build_inst_id(symbol: str, ins_type: TradeInsType | int, quote_currency: str) -> str:
        """
        构建OKX订阅的instId

        参数:
            symbol: 交易对，如'BTC-USDT'
            ins_type: 交易类型，如TradeInsType.SWAP
            quote_currency: 计价币种，如'USDT'

        返回:
            str: 构建的instId
        """
        if isinstance(ins_type, int):
            ins_type = TradeInsType(ins_type)

        if ins_type == TradeInsType.SWAP or ins_type == TradeInsType.FUTURES:
            return f"{symbol}-{quote_currency}-SWAP"
        elif ins_type == TradeInsType.OPTION:
            return f"{symbol}-{quote_currency}-OPTION"
        elif ins_type == TradeInsType.SPOT or ins_type == TradeInsType.FUTURES:
            return f"{symbol}-{quote_currency}"
        raise ValueError(f"不支持的交易类型: {ins_type}")

    def unsubscribe(self, symbol: str = "BTC", timeframe: Union[TimeFrame, str] = TimeFrame.M1,
                    ins_type: TradeInsType = TradeInsType.SWAP, quote_currency: str = "USDT", **kwargs) -> bool:
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
        # 获取OKX格式的时间周期字符串
        tf_value = self._get_okx_tf_value(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和OKX订阅消息
        channel = f"candle{tf_value}"
        inst_id = self.build_inst_id(symbol, ins_type, quote_currency)

        # 发送取消订阅请求
        msg = {"op": "unsubscribe", "args": [{"channel": channel, "instId": inst_id}]}
        return self.async_send(json.dumps(msg))

    def get_supported_parameters(self) -> List[Field]:
        from okx import MarketData
        market_api = MarketData.MarketAPI(domain="https://www.okx.com", flag="0", debug=False)

        tickers = market_api.get_tickers(instType="SWAP")
        symbols = set([ticker["instId"].split("-")[0] for ticker in tickers["data"]])
        tickers = market_api.get_tickers(instType="SPOT")
        symbols |= set([ticker["instId"].split("-")[0] for ticker in tickers["data"]])
        ins_types = [(TradeInsType.SPOT.value, str(TradeInsType.SPOT)), (TradeInsType.SWAP.value, str(TradeInsType.SWAP))]
        return [
            Field(name='symbol', label='交易标的', type=FieldType.RADIO, required=True, choices=list(symbols),
                  choice_type=ChoiceType.STR),
            Field(name='timeframe', label='时间周期', type=FieldType.RADIO, required=True,
                  choices=list(OKX_TIMEFRAME_MAP.keys()), choice_type=ChoiceType.STR),
            Field(name='quote_currency', label='计价币种', type=FieldType.RADIO, required=True,
                  choices=["USDT"], choice_type=ChoiceType.STR),
            Field(name='ins_type', label='交易标的类型', type=FieldType.RADIO, required=True,
                  choices=ins_types, choice_type=ChoiceType.INT),
        ]

    def get_history_data(
            self,
            start_time: datetime | int = None,
            end_time: datetime | int = None,
            limit: int = None,
            symbol: str = None,
            timeframe: TimeFrame | str = None,
            quote_currency: str = 'USDT',
            ins_type: TradeInsType=None
    ) -> Iterator[KLine]:
        """
        获取K线数据，WebSocket数据源通常不支持历史K线查询

        参数:
            symbol: 交易对符号
            timeframe: K线时间周期
            quote_currency: 计价币种
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制
            ins_type: 交易品种类型

        返回:
            Iterator[KLine]: K线数据迭代器（空）
        """
        # WebSocket数据源通常不支持获取历史K线数据
        logger.warning(f"WebSocket数据源不支持获取历史K线数据，请使用REST API")
        return iter([])

    @staticmethod
    def _get_okx_tf_value(timeframe: Union[TimeFrame, str]) -> Optional[str]:
        """获取OKX格式的时间周期字符串"""
        if isinstance(timeframe, str):
            try:
                tf_enum = TimeFrame(timeframe)
                tf_val = OKX_TIMEFRAME_MAP.get(tf_enum)
            except ValueError:
                if timeframe in OKX_CHANNEL_MAP:
                    tf_val = timeframe
                else:
                    logger.error(f"无效的时间周期字符串: {timeframe}")
                    return None
        else:
            tf_val = OKX_TIMEFRAME_MAP.get(timeframe)

        if tf_val is None:
            logger.error(f"OKX不支持的时间周期: {timeframe}")

        return tf_val
