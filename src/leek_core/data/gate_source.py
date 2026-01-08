#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gate.io WebSocket 数据源实现。
继承自 WebSocketDataSource，用于连接 Gate.io 并订阅 K 线数据。
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Union, Iterator

from leek_core.adapts import GateAdapter
from leek_core.models import TimeFrame, DataType, Field, FieldType, ChoiceType, TradeInsType, AssetType, KLine
from leek_core.utils import get_logger, log_method, retry
from .websocket import WebSocketDataSource

logger = get_logger(__name__)

# Gate.io TimeFrame to channel string mapping (for WebSocket message parsing and UI options)
# Gate.io supports: 10s, 30s, 1m, 5m, 15m, 30m, 1h, 4h, 8h, 1d, 7d, 30d
GATE_TIMEFRAME_MAP = {
    TimeFrame.S30: "30s",
    TimeFrame.M1: "1m", TimeFrame.M5: "5m", TimeFrame.M15: "15m", TimeFrame.M30: "30m",
    TimeFrame.H1: "1h", TimeFrame.H4: "4h", TimeFrame.H8: "8h",
    TimeFrame.D1: "1d", TimeFrame.W1: "7d", TimeFrame.MON1: "30d",
}
GATE_CHANNEL_MAP = {v: k for k, v in GATE_TIMEFRAME_MAP.items()}


class GateDataSource(WebSocketDataSource):
    """
    Gate.io WebSocket K 线数据源。
    """
    display_name: str = "Gate K线"
    supported_data_type: DataType = DataType.KLINE
    supported_asset_type: DataType = AssetType.CRYPTO
    init_params: List[Field] = [
        Field(name="symbols", label="标的", type=FieldType.ARRAY, default=[], choice_type=ChoiceType.STRING),
    ]

    def __init__(self, symbols: List[str] = None, settle: str = "usdt"):
        """
        初始化 Gate.io WebSocket 数据源。

        参数:
            symbols: 交易对列表
            settle: 结算货币 (usdt/btc)，默认为 usdt
        """
        self.settle = settle.lower()
        self.ws_domain = f"wss://fx-ws.gateio.ws/v4/ws/{self.settle}"
        super().__init__(ws_url=self.ws_domain, ping_interval=25, ping_timeout=10)
        self.symbols = symbols or []
        self.subscribed_channels: Dict[str, int] = {}

        self.pre_time = {}
        self.adapter = GateAdapter()

    def on_connect(self):
        """连接成功后调用"""
        logger.info(f"Gate.io数据源连接成功，使用websockets内置心跳")

    def on_disconnect(self):
        """断开连接前调用"""
        logger.info(f"Gate.io数据源断开连接")

    async def on_message(self, message: str):
        """
        处理收到的WebSocket消息
        
        参数:
            message: 收到的消息
        """
        try:
            data = json.loads(message)
            
            # 处理系统消息
            event = data.get("event")
            channel = data.get("channel", "")
            
            if event == "subscribe":
                logger.info(f"Gate.io订阅确认: channel={channel}, result={data.get('result')}")
                return
            elif event == "unsubscribe":
                logger.info(f"Gate.io取消订阅确认: channel={channel}")
                return
            elif event == "error":
                logger.error(f"Gate.io WS错误: {data.get('error')} channel={channel}")
                return
            
            # 处理 ping/pong
            if channel == "futures.ping":
                return
            
            # 处理K线数据
            if channel == "futures.candlesticks":
                await self._process_kline_data(data)
                
        except json.JSONDecodeError:
            logger.warning(f"Gate.io非JSON消息: {message[:100] if isinstance(message, str) else str(message)[:100]}...")
        except Exception as e:
            logger.error(f"处理Gate.io消息时出错: {e}", exc_info=True)

    async def _process_kline_data(self, data: Dict):
        """处理K线数据
        
        Gate.io K线数据格式 (result 可能是列表或字典):
        {
            "time": 1545129600,
            "channel": "futures.candlesticks",
            "event": "update",
            "result": [
                {
                    "t": 1545129600,  # Unix timestamp in seconds
                    "v": 27270,       # Total volume
                    "c": "3815.1",    # Close price
                    "h": "3818.3",    # Highest price
                    "l": "3814.9",    # Lowest price
                    "o": "3817",      # Open price
                    "n": "1m_BTC_USDT",  # Interval_Contract
                    "a": "0"          # Total amount (quote currency)
                }
            ]
        }
        """
        try:
            result = data.get("result")
            if not result:
                return
            
            # result 可能是列表或字典
            items = result if isinstance(result, list) else [result]
            
            for item in items:
                await self._process_single_kline(item)
            
        except Exception as e:
            logger.error(f"处理Gate.io K线数据时出错: {e}", exc_info=True)

    async def _process_single_kline(self, item: Dict):
        """处理单条K线数据
        
        字段说明:
            t: Integer - 时间戳（秒）
            o: String - 开盘价格
            c: String - 收盘价格
            h: String - 最高价格
            l: String - 最低价格
            v: String/Integer - 成交量
            n: String - 合约名称，格式为 "interval_contract"，如 "1m_BTC_USDT"
            a: String - 成交原始币种数量
            w: Boolean - true 表示窗口已关闭（K线完成）
        """
        try:
            # 提取K线数据字段
            timestamp_sec = item.get("t")  # Unix timestamp in seconds
            interval = item.get("n")  # Interval string like "1m_BTC_USDT"
            
            if not timestamp_sec or not interval:
                return
            
            # Gate.io 的通知中 "n" 字段是 "interval_contract" 格式，如 "1m_BTC_USDT"
            parts = interval.split("_", 1)
            if len(parts) < 2:
                logger.warning(f"Gate.io K线数据缺少合约信息: {interval}")
                return
            
            tf_str = parts[0]  # e.g., "1m"
            contract = parts[1]  # e.g., "BTC_USDT"
            
            # 解析时间周期
            tf = GATE_CHANNEL_MAP.get(tf_str)
            if not tf:
                logger.error(f"未知的Gate.io时间周期: {tf_str}")
                return
            
            # 解析合约信息
            contract_parts = contract.split("_")
            if len(contract_parts) < 2:
                logger.warning(f"无效的Gate.io合约格式: {contract}")
                return
            
            base_currency = contract_parts[0]  # e.g., "BTC"
            quote_currency = contract_parts[1]  # e.g., "USDT"
            
            # Gate.io 期货合约类型
            ins_type = TradeInsType.SWAP
            
            # 转换时间戳（Gate.io 返回秒级时间戳，需要转为毫秒）
            timestamp_ms = int(timestamp_sec) * 1000
            end_time = timestamp_ms + tf.milliseconds
            
            # w 字段: true 表示窗口已关闭（K线完成），false 表示K线还在进行中
            # 注：可能会缺失 w=true 的消息，但不影响数据完整性
            is_finished = item.get("w", False)
            
            # 创建KLine对象
            kline = KLine(
                symbol=base_currency,
                market='gate',
                open=item.get("o"),
                high=item.get("h"),
                low=item.get("l"),
                close=item.get("c"),
                volume=item.get("v"),
                amount=item.get("a", "0"),
                start_time=timestamp_ms,
                end_time=end_time,
                current_time=int(time.time() * 1000),
                timeframe=tf,
                quote_currency=quote_currency,
                ins_type=ins_type,
                is_finished=is_finished
            )
            kline.asset_type = AssetType.CRYPTO

            # 防止重复推送已完成的K线
            # 只有当K线完成时才更新 pre_time，未完成的K线允许重复推送（实时更新）
            if is_finished:
                if kline.start_time <= self.pre_time.get(kline.row_key, 0):
                    return
                self.pre_time[kline.row_key] = kline.start_time
            
            # 调用订阅的回调函数
            self.send_data(kline)
            
        except Exception as e:
            logger.error(f"处理Gate.io单条K线数据时出错: {e}", exc_info=True)

    def parse_row_key(self, symbols: List[str] = list, timeframes: List[Union[TimeFrame, str]] = list,
                  ins_types: List[TradeInsType] = list, quote_currencies: List[str] = list, **kwargs) -> List[tuple]:
        for s in symbols:
            for q in quote_currencies:
                for i in ins_types:
                    for t in timeframes:
                        yield KLine.pack_row_key(s, q, i, t)

    def _get_gate_timeframe(self, timeframe: Union[TimeFrame, str]) -> str:
        """
        将TimeFrame转换为Gate.io的时间周期字符串
        
        参数:
            timeframe: TimeFrame枚举或字符串
            
        返回:
            str: Gate.io的时间周期字符串，如 "1m", "1h" 等
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        return GATE_TIMEFRAME_MAP.get(timeframe)

    def _build_contract(self, symbol: str, quote_currency: str) -> str:
        """
        构建Gate.io合约标识
        
        参数:
            symbol: 交易对符号，如 "BTC"
            quote_currency: 计价币种，如 "USDT"
            
        返回:
            str: Gate.io合约标识，如 "BTC_USDT"
        """
        return f"{symbol}_{quote_currency}"

    @log_method()
    def subscribe(self, row_key: str) -> bool:
        """
        订阅Gate.io K线数据
        
        参数:
            row_key: 数据键，格式为 "symbol_quote_currency_ins_type_timeframe"
            
        返回:
            bool: 订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        
        # 获取Gate.io格式的时间周期字符串
        tf_value = self._get_gate_timeframe(timeframe)
        if tf_value is None:
            logger.error(f"不支持的时间周期: {timeframe}")
            return False

        # 构建订阅键和Gate.io订阅消息
        contract = self._build_contract(symbol, quote_currency)
        
        # Gate.io WebSocket 订阅格式
        # payload: [interval, contract] 如 ["1m", "BTC_USDT"]
        msg = {
            "time": int(time.time()),
            "channel": "futures.candlesticks",
            "event": "subscribe",
            "payload": [tf_value, contract]
        }
        
        key = f"{tf_value}:{contract}"
        if key in self.subscribed_channels:
            self.subscribed_channels[key] += 1
            return True
        self.subscribed_channels[key] = 1
        return self.async_send(json.dumps(msg))

    @log_method()
    def unsubscribe(self, row_key: str) -> bool:
        """
        取消订阅Gate.io K线数据
        
        参数:
            row_key: 数据键，格式为 "symbol_quote_currency_ins_type_timeframe"
            
        返回:
            bool: 取消订阅成功返回True，否则返回False
        """
        symbol, quote_currency, ins_type, timeframe = KLine.parse_row_key(row_key)
        
        # 获取Gate.io格式的时间周期字符串
        tf_value = self._get_gate_timeframe(timeframe)
        if tf_value is None:
            return False

        # 构建订阅键和Gate.io取消订阅消息
        contract = self._build_contract(symbol, quote_currency)
        
        msg = {
            "time": int(time.time()),
            "channel": "futures.candlesticks",
            "event": "unsubscribe",
            "payload": [tf_value, contract]
        }
        
        key = f"{tf_value}:{contract}"
        if key in self.subscribed_channels and self.subscribed_channels[key] > 1:
            self.subscribed_channels[key] -= 1
            return True
        self.subscribed_channels.pop(key, None)
        return self.async_send(json.dumps(msg))

    def get_supported_parameters(self) -> List[Field]:
        if self.symbols is None or len(self.symbols) == 0:
            # 获取所有合约
            contracts = self.adapter.get_futures_contracts(settle=self.settle)
            if contracts.get("code") == "0":
                symbols = set()
                for contract in contracts.get("data", []):
                    # 合约名称格式如 "BTC_USDT"
                    name = contract.get("name", "")
                    if "_" in name:
                        symbols.add(name.split("_")[0])
                self.symbols = list(symbols)
            else:
                self.symbols = []
        
        ins_types = [(TradeInsType.SWAP.value, str(TradeInsType.SWAP))]
        return [
            Field(name='symbols', label='交易标的', type=FieldType.SELECT, required=True, choices=list(self.symbols),
                  choice_type=ChoiceType.STRING),
            Field(name='timeframes', label='时间周期', type=FieldType.SELECT, required=True,
                  choices=list(GATE_TIMEFRAME_MAP.keys()), choice_type=ChoiceType.STRING),
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

        # 处理时间参数（Gate.io 使用秒级时间戳）
        from_time = None
        if start_time is not None:
            if isinstance(start_time, datetime):
                from_time = int(start_time.timestamp())
            else:
                from_time = start_time // 1000  # 毫秒转秒
        
        to_time = int(time.time())
        if end_time is not None:
            if isinstance(end_time, datetime):
                to_time = int(end_time.timestamp())
            elif isinstance(end_time, int):
                to_time = end_time // 1000  # 毫秒转秒

        contract = self._build_contract(symbol, quote_currency)
        interval = self._get_gate_timeframe(timeframe)
        if interval is None:
            logger.error(f"不支持的时间周期: {timeframe}")
            return iter([])
        
        page_size = min(100, limit)
        res = []
        
        while len(res) < limit:
            candlesticks = self.adapter.get_futures_candlesticks(
                settle=self.settle,
                contract=contract,
                interval=interval,
                from_time=from_time,
                to_time=to_time,
                limit=page_size
            )
            
            if candlesticks.get("code") != "0":
                logger.error(f"获取Gate.io K线数据失败: {candlesticks.get('msg')}")
                break
            
            data = candlesticks.get("data", [])
            if len(data) == 0:
                break
            
            for row in data:
                # Gate.io REST API 返回格式: {"t": timestamp, "v": volume, "c": close, "h": high, "l": low, "o": open, "sum": amount}
                # 或者是列表格式: [t, v, c, h, l, o, sum]
                if isinstance(row, dict):
                    t = row.get("t")
                    v = row.get("v")
                    c = row.get("c")
                    h = row.get("h")
                    l = row.get("l")
                    o = row.get("o")
                    a = row.get("sum", "0")
                else:
                    # 列表格式
                    t, v, c, h, l, o = row[0], row[1], row[2], row[3], row[4], row[5]
                    a = row[6] if len(row) > 6 else "0"
                
                # 转换时间戳（秒转毫秒）
                timestamp_ms = int(t) * 1000
                
                kline = KLine(
                    data_type=DataType.KLINE,
                    symbol=symbol,
                    market='gate',
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    amount=a,
                    start_time=timestamp_ms,
                    end_time=timestamp_ms + timeframe.milliseconds,
                    current_time=int(time.time() * 1000),
                    timeframe=timeframe,
                    quote_currency=quote_currency,
                    ins_type=ins_type,
                    is_finished=True
                )
                
                # 更新 to_time 用于下一页查询
                to_time = int(t) - 1
                res.append(kline)
        
        res = res[:limit]
        res.reverse()
        return iter(res)
