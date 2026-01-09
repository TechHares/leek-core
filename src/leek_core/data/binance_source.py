#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance WebSocket 数据源实现。
继承自 WebSocketDataSource，用于连接 Binance 并订阅 K 线数据。
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Union, Iterator

from leek_core.adapts import BinanceAdapter
from leek_core.models import TimeFrame, DataType, Field, FieldType, ChoiceType, TradeInsType, AssetType, KLine
from leek_core.utils import get_logger, log_method, retry
from .websocket import WebSocketDataSource

logger = get_logger(__name__)

# Binance TimeFrame to channel string mapping (for WebSocket message parsing and UI options)
# Binance supports: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M
BINANCE_TIMEFRAME_MAP = {
    TimeFrame.M1: "1m", TimeFrame.M3: "3m", TimeFrame.M5: "5m",
    TimeFrame.M15: "15m", TimeFrame.M30: "30m", TimeFrame.H1: "1h",
    TimeFrame.H2: "2h", TimeFrame.H4: "4h", TimeFrame.H6: "6h",
    TimeFrame.H12: "12h", TimeFrame.D1: "1d", TimeFrame.W1: "1w",
    TimeFrame.MON1: "1M",
}
BINANCE_CHANNEL_MAP = {v: k for k, v in BINANCE_TIMEFRAME_MAP.items()}


class BinanceDataSource(WebSocketDataSource):
    """
    Binance WebSocket K 线数据源。
    """
    display_name: str = "Binance行情"
    supported_data_type: DataType = DataType.KLINE
    supported_asset_type: DataType = AssetType.CRYPTO
    init_params: List[Field] = [
        Field(name="symbols", label="标的", type=FieldType.ARRAY, default=[], choice_type=ChoiceType.STRING),
    ]

    def __init__(self, symbols: List[str] = None):
        """
        初始化 Binance WebSocket 数据源。

        参数:
            symbols: 交易对列表
        """
        # Binance WebSocket base URL
        self.ws_domain = "wss://stream.binance.com:9443/ws"
        super().__init__(ws_url=self.ws_domain, ping_interval=25, ping_timeout=10)
        self.symbols = symbols or []
        self.subscribed_channels: Dict[str, int] = {}
        
        # 用于追踪已完成的K线，防止重复推送
        self.pre_time = {}
        # 订阅消息ID计数器
        self._sub_id = 0
        # 使用无密钥的adapter用于获取市场数据
        self.adapter = None

    def _get_adapter(self) -> BinanceAdapter:
        """延迟初始化adapter（避免无密钥时报错）"""
        if self.adapter is None:
            self.adapter = BinanceAdapter()
        return self.adapter

    def on_connect(self):
        """连接成功后调用"""
        logger.info(f"Binance数据源连接成功，使用websockets内置心跳")

    def on_disconnect(self):
        """断开连接前调用"""
        logger.info(f"Binance数据源断开连接")

    async def on_message(self, message: str):
        """
        处理收到的WebSocket消息
        
        参数:
            message: 收到的消息
        """
        try:
            data = json.loads(message)
            
            # 处理订阅响应消息
            if "result" in data and "id" in data:
                if data.get("result") is None:
                    logger.info(f"Binance订阅/取消订阅成功: id={data.get('id')}")
                else:
                    logger.warning(f"Binance订阅响应: {data}")
                return
            
            # 处理错误消息
            if "code" in data and "msg" in data:
                logger.error(f"Binance WS错误: code={data.get('code')}, msg={data.get('msg')}")
                return
            
            # 处理K线数据
            if data.get("e") == "kline":
                await self._process_kline_data(data)
                
        except json.JSONDecodeError:
            logger.warning(f"Binance非JSON消息: {message[:100] if isinstance(message, str) else str(message)[:100]}...")
        except Exception as e:
            logger.error(f"处理Binance消息时出错: {e}", exc_info=True)

    async def _process_kline_data(self, data: Dict):
        """处理K线数据
        
        Binance K线数据格式:
        {
            "e": "kline",       // Event type
            "E": 1672515782136, // Event time
            "s": "BTCUSDT",     // Symbol
            "k": {
                "t": 1672515780000,  // Kline start time
                "T": 1672515839999,  // Kline close time
                "s": "BTCUSDT",      // Symbol
                "i": "1m",           // Interval
                "f": 100,            // First trade ID
                "L": 200,            // Last trade ID
                "o": "0.0010",       // Open price
                "c": "0.0020",       // Close price
                "h": "0.0025",       // High price
                "l": "0.0015",       // Low price
                "v": "1000",         // Base asset volume
                "n": 100,            // Number of trades
                "x": false,          // Is this kline closed?
                "q": "1.0000",       // Quote asset volume
                "V": "500",          // Taker buy base asset volume
                "Q": "0.500",        // Taker buy quote asset volume
                "B": "123456"        // Ignore
            }
        }
        """
        try:
            kline_data = data.get("k")
            if not kline_data:
                return
            
            symbol_str = data.get("s", "")  # 如 "BTCUSDT"
            interval = kline_data.get("i")  # 如 "1m"
            
            if not symbol_str or not interval:
                return
            
            # 解析时间周期
            tf = BINANCE_CHANNEL_MAP.get(interval)
            if not tf:
                logger.error(f"未知的Binance时间周期: {interval}")
                return
            
            # 解析交易对 - Binance 格式为 BTCUSDT，需要拆分
            # 常见计价货币: USDT, BUSD, BTC, ETH, BNB
            quote_currencies = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "TUSD", "PAX", "FDUSD"]
            base_currency = None
            quote_currency = None
            
            for qc in quote_currencies:
                if symbol_str.endswith(qc):
                    base_currency = symbol_str[:-len(qc)]
                    quote_currency = qc
                    break
            
            if not base_currency or not quote_currency:
                logger.warning(f"无法解析Binance交易对: {symbol_str}")
                return
            
            # Binance 现货交易类型
            ins_type = TradeInsType.SPOT
            
            # 获取K线数据
            start_time = int(kline_data.get("t", 0))  # 毫秒时间戳
            end_time = int(kline_data.get("T", 0)) + 1  # Binance返回的是闭区间，需要+1
            is_finished = kline_data.get("x", False)
            
            # 创建KLine对象
            kline = KLine(
                symbol=base_currency,
                market='binance',
                open=kline_data.get("o"),
                high=kline_data.get("h"),
                low=kline_data.get("l"),
                close=kline_data.get("c"),
                volume=kline_data.get("v"),  # Base asset volume
                amount=kline_data.get("q"),  # Quote asset volume
                start_time=start_time,
                end_time=end_time,
                current_time=int(time.time() * 1000),
                timeframe=tf,
                quote_currency=quote_currency,
                ins_type=ins_type,
                is_finished=is_finished
            )
            kline.asset_type = AssetType.CRYPTO

            # 防止重复推送已完成的K线
            if kline.start_time <= self.pre_time.get(kline.row_key, 0):
                return
            if kline.is_finished:
                self.pre_time[kline.row_key] = kline.start_time
            
            # 调用订阅的回调函数
            self.send_data(kline)
            
        except Exception as e:
            logger.error(f"处理Binance K线数据时出错: {e}", exc_info=True)

    def parse_row_key(self, symbols: List[str] = list, timeframes: List[Union[TimeFrame, str]] = list,
                  ins_types: List[TradeInsType] = list, quote_currencies: List[str] = list, **kwargs) -> List[tuple]:
        for s in symbols:
            for q in quote_currencies:
                for i in ins_types:
                    for t in timeframes:
                        yield KLine.pack_row_key(s, q, i, t)

    def _get_binance_timeframe(self, timeframe: Union[TimeFrame, str]) -> str:
        """
        将TimeFrame转换为Binance的时间周期字符串
        
        参数:
            timeframe: TimeFrame枚举或字符串
            
        返回:
            str: Binance的时间周期字符串，如 "1m", "1h" 等
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        return BINANCE_TIMEFRAME_MAP.get(timeframe)

    def _build_stream_name(self, symbol: str, quote_currency: str, timeframe: Union[TimeFrame, str]) -> str:
        """
        构建Binance WebSocket流名称
        
        参数:
            symbol: 交易对符号，如 "BTC"
            quote_currency: 计价币种，如 "USDT"
            timeframe: 时间周期
            
        返回:
            str: Binance流名称，如 "btcusdt@kline_1m"
        """
        tf_value = self._get_binance_timeframe(timeframe)
        # Binance 要求流名称使用小写
        binance_symbol = f"{symbol}{quote_currency}".lower()
        return f"{binance_symbol}@kline_{tf_value}"

    @log_method()
    def subscribe(self, row_key: str) -> bool:
        """
        订阅Binance K线数据
        
        参数:
            row_key: 数据键，格式为 "symbol_quote_currency_ins_type_timeframe"
            
        返回:
            bool: 订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        
        # 获取Binance格式的时间周期字符串
        tf_value = self._get_binance_timeframe(timeframe)
        if tf_value is None:
            logger.error(f"不支持的时间周期: {timeframe}")
            return False

        # 构建订阅键和Binance WebSocket流名称
        stream_name = self._build_stream_name(symbol, quote_currency, timeframe)
        
        # 检查是否已订阅
        if stream_name in self.subscribed_channels:
            self.subscribed_channels[stream_name] += 1
            return True
        
        # 发送订阅请求
        self._sub_id += 1
        msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": self._sub_id
        }
        
        self.subscribed_channels[stream_name] = 1
        return self.async_send(json.dumps(msg))

    @log_method()
    def unsubscribe(self, row_key: str) -> bool:
        """
        取消订阅Binance K线数据
        
        参数:
            row_key: 数据键，格式为 "symbol_quote_currency_ins_type_timeframe"
            
        返回:
            bool: 取消订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        
        # 获取Binance格式的时间周期字符串
        tf_value = self._get_binance_timeframe(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和Binance WebSocket流名称
        stream_name = self._build_stream_name(symbol, quote_currency, timeframe)
        
        # 检查引用计数
        if stream_name in self.subscribed_channels and self.subscribed_channels[stream_name] > 1:
            self.subscribed_channels[stream_name] -= 1
            return True
        
        # 发送取消订阅请求
        self._sub_id += 1
        msg = {
            "method": "UNSUBSCRIBE",
            "params": [stream_name],
            "id": self._sub_id
        }
        
        self.subscribed_channels.pop(stream_name, None)
        return self.async_send(json.dumps(msg))

    def get_supported_parameters(self) -> List[Field]:
        """获取支持的参数列表"""
        if self.symbols is None or len(self.symbols) == 0:
            try:
                # 获取所有可用的交易对
                self.symbols = self._get_adapter().get_all_symbols(quote_currency="USDT")
            except Exception as e:
                logger.error(f"获取Binance交易对列表失败: {e}")
                self.symbols = []
        
        ins_types = [(TradeInsType.SPOT.value, str(TradeInsType.SPOT))]
        return [
            Field(name='symbols', label='交易标的', type=FieldType.SELECT, required=True, 
                  choices=list(self.symbols), choice_type=ChoiceType.STRING),
            Field(name='timeframes', label='时间周期', type=FieldType.SELECT, required=True,
                  choices=list(BINANCE_TIMEFRAME_MAP.keys()), choice_type=ChoiceType.STRING),
            Field(name='quote_currencies', label='计价币种', type=FieldType.SELECT, required=True,
                  choices=["USDT", "BUSD", "USDC", "BTC"], choice_type=ChoiceType.STRING),
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
        获取K线历史数据

        参数:
            row_key: 数据键
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制

        返回:
            Iterator[KLine]: K线数据迭代器
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        if not isinstance(ins_type, TradeInsType):
            ins_type = TradeInsType(ins_type)
        if limit is None:
            limit = 100

        # 处理时间参数（Binance 使用毫秒时间戳）
        start_ms = None
        if start_time is not None:
            if isinstance(start_time, datetime):
                start_ms = int(start_time.timestamp() * 1000)
            else:
                start_ms = start_time
        
        # 结束时间增加缓冲，确保获取最新K线
        end_ms = int(time.time() * 1000) + 60000  # 增加60秒缓冲
        if end_time is not None:
            if isinstance(end_time, datetime):
                end_ms = int(end_time.timestamp() * 1000) + 60000
            elif isinstance(end_time, int):
                end_ms = end_time + 60000

        # 构建Binance交易对符号
        binance_symbol = BinanceAdapter.build_symbol(symbol, quote_currency)
        interval = self._get_binance_timeframe(timeframe)
        if interval is None:
            logger.error(f"不支持的时间周期: {timeframe}")
            return iter([])
        
        page_size = min(1000, limit)  # Binance最大支持1000条
        res = []
        
        while len(res) < limit:
            klines = self._get_adapter().get_klines(
                symbol=binance_symbol,
                interval=interval,
                start_time=start_ms,
                end_time=end_ms,
                limit=page_size
            )
            
            if klines.get("code") != "0":
                logger.error(f"获取Binance K线数据失败: {klines.get('msg')}")
                break
            
            data = klines.get("data", [])
            if len(data) == 0:
                break
            
            # Binance REST API 返回升序数据（从旧到新）
            # 收集本批次数据
            batch = []
            for row in data:
                # Binance REST API 返回格式:
                # [open_time, open, high, low, close, volume, close_time, 
                #  quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
                if len(row) < 7:
                    continue
                
                open_time = int(row[0])  # 毫秒时间戳
                close_time = int(row[6]) + 1  # Binance返回的是闭区间
                
                kline = KLine(
                    data_type=DataType.KLINE,
                    symbol=symbol,
                    market='binance',
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5],  # Base asset volume
                    amount=row[7] if len(row) > 7 else "0",  # Quote asset volume
                    start_time=open_time,
                    end_time=close_time,
                    current_time=int(time.time() * 1000),
                    timeframe=timeframe,
                    quote_currency=quote_currency,
                    ins_type=ins_type,
                    is_finished=True  # 历史数据都是已完成的
                )
                kline.asset_type = AssetType.CRYPTO
                batch.append(kline)
            
            # 将本批次数据放到结果前面（因为是更早的数据）
            res = batch + res
            
            # 更新 end_ms 为本批次最早K线的时间-1，用于获取更早的数据
            if data:
                end_ms = int(data[0][0]) - 1
            
            # 如果返回的数据少于请求的数量，说明已经没有更多数据了
            if len(data) < page_size:
                break
        
        # 取最近的 limit 条数据（数据已是升序：从旧到新）
        res = res[-limit:] if len(res) > limit else res
        return iter(res)
