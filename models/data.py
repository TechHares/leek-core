#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K线数据模型模块，定义K线数据传输对象。
"""
from abc import ABC, abstractmethod
from builtins import set
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Union, Dict, Any, Set, List, Tuple, ClassVar

from utils import DecimalEncoder
from .constants import TimeFrame, TradeInsType, DataType

# 设置Decimal精度
getcontext().prec = 28  # 设置足够高的精度以处理金融数据

# 定义数值类型，可以是Decimal或字符串（用于构造Decimal）
NumericType = Union[Decimal, str, int, float]


@dataclass
class Data(ABC):
    """
    数据源最终产出的数据结构
    """
    data_source_id: str = None  # 数据源ID
    data_type: DataType = None  # 数据类型

    # 保护的字段名称，这些字段不能被动态属性覆盖（类字段）
    _protected_fields: ClassVar[set[str]] = {}

    def __getattr__(self, name: str) -> Any:
        """
        获取动态属性值。当正常属性查找失败时调用。

        参数:
            name: 属性名称

        返回:
            属性值，如果不存在则返回None
        """
        if name in ["_dynamic_attrs", "metadata"]:
            if r := getattr(self, name, None):
                return r
            setattr(self, name, {})
            return {}

        if name in self._protected_fields or name in ["data_source_id", "data_type"]:
            return getattr(self, name)

        # 检查动态属性字典
        if name in self._dynamic_attrs:
            return self._dynamic_attrs[name]

        # 检查元数据字典（向后兼容）
        if name in self.metadata:
            return self.metadata[name]

        # 如果都不存在，返回None
        return None

    def __setattr__(self, name: str, value: Any) -> None:
        """
        设置属性值

        参数:
            name: 属性名称
            value: 属性值
        """
        # 如果是受保护字段或者是类的标准属性，使用标准方式设置
        if name in self._protected_fields or name.startswith('_') or name in self.__annotations__:
            object.__setattr__(self, name, value)
            return

        # 否则作为动态属性存储
        self._dynamic_attrs[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """
        获取属性值，支持默认值

        参数:
            name: 属性名称
            default: 如果属性不存在时返回的默认值

        返回:
            属性值或默认值
        """
        # 先检查标准属性
        if hasattr(self, name) and not name.startswith('_'):
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass

        # 检查动态属性
        if name in self._dynamic_attrs:
            return self._dynamic_attrs[name]

        # 检查元数据
        if name in self.metadata:
            return self.metadata[name]

        # 都不存在，返回默认值
        return default

    def set(self, name: str, value: Any) -> None:
        """
        设置属性值，更明确的API方法

        参数:
            name: 属性名称
            value: 属性值
        """
        self.__setattr__(name, value)

    @staticmethod
    def _to_decimal(value: NumericType) -> Decimal:
        """
        将各种数值类型转换为Decimal
        """
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))  # 通过字符串转换避免浮点精度问题

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Data":
        """
        从字典创建K线对象，自动处理字符串形式的Decimal值
        """
        ...

    def to_json(self) -> str:
        """
        将K线对象转换为JSON字符串
        """
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Data":
        """
        从JSON字符串创建K线对象
        """
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class KLine(Data):
    """
    K线数据传输对象
    
    属性:
        symbol: 交易对符号
        market: 市场标识（如"okx"）
        quote_currency: 计价币种（如"USDT"）
        ins_type: 交易品种类型（如现货、永续合约等）
        data_source_id: 数据来源ID，用于跟踪数据来源和过滤
        open: 开盘价
        close: 收盘价
        high: 最高价
        low: 最低价
        volume: 成交量
        amount: 成交额
        start_time: K线开始时间的时间戳(毫秒)
        end_time: K线结束时间的时间戳(毫秒)
        current_time: 当前时间的时间戳(毫秒)
        is_finished: 标识K线是否已完成
        timeframe: K线时间粒度
    """
    symbol: str = None
    market: str = None  # 市场标识，如"okx"
    open: Decimal = None
    close: Decimal = None
    high: Decimal = None
    low: Decimal = None
    volume: Decimal = None
    amount: Decimal = None
    start_time: int = None  # 毫秒时间戳
    end_time: int = None  # 毫秒时间戳
    current_time: int = None  # 毫秒时间戳
    timeframe: TimeFrame = None  # K线时间粒度
    quote_currency: str = "USDT"  # 计价币种，默认为USDT
    ins_type: TradeInsType = TradeInsType.SPOT  # 交易品种类型，默认为现货
    is_finished: bool = False
    # 元数据，可用于存储额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 保护的字段名称，这些字段不能被动态属性覆盖（类字段）
    _protected_fields: ClassVar[set[str]] = {
        'symbol', 'market', 'quote_currency', 'ins_type', 'data_source_id', 'open', 'close', 'high', 'low', 'volume',
        'amount',
        'start_time', 'end_time', 'current_time', 'timeframe', 'is_finished',
        'metadata', '_protected_fields', '_dynamic_attrs'
    }

    def __post_init__(self):
        """
        确保所有数值字段都是Decimal类型
        并将timeframe转换为TimeFrame枚举
        """
        self.open = self._to_decimal(self.open)
        self.close = self._to_decimal(self.close)
        self.high = self._to_decimal(self.high)
        self.low = self._to_decimal(self.low)
        self.volume = self._to_decimal(self.volume)
        self.amount = self._to_decimal(self.amount)

        # 处理timeframe
        if isinstance(self.timeframe, str):
            self.timeframe = TimeFrame.from_string(self.timeframe)

        # 处理元数据中的Decimal
        self.metadata = DecimalEncoder.decode(self.metadata)

    @property
    def duration(self) -> int:
        """
        获取K线持续时间(毫秒)
        """
        return self.end_time - self.start_time

    @property
    def expected_duration(self) -> Optional[int]:
        """
        获取基于timeframe的预期持续时间(毫秒)
        """
        return self.timeframe.milliseconds

    @property
    def start_datetime(self) -> datetime:
        """
        获取K线开始时间的datetime对象
        """
        return datetime.fromtimestamp(self.start_time / 1000)

    @property
    def end_datetime(self) -> datetime:
        """
        获取K线结束时间的datetime对象
        """
        return datetime.fromtimestamp(self.end_time / 1000)

    @property
    def current_datetime(self) -> datetime:
        """
        获取当前时间的datetime对象
        """
        return datetime.fromtimestamp(self.current_time / 1000)

    @property
    def row_key(self) -> tuple[str, str, str, Any, Any]:
        """
        获取K线的唯一标识符
        """
        return self.data_source_id, self.symbol, self.quote_currency, self.ins_type.value, self.timeframe.value

    def to_dict(self) -> dict:
        """
        将K线对象转换为字典，所有的Decimal类型都会被转换为字符串
        """
        # 基本字段
        result = {
            "symbol": self.symbol,
            "market": self.market,
            "open": str(self.open),
            "close": str(self.close),
            "high": str(self.high),
            "low": str(self.low),
            "volume": str(self.volume),
            "amount": str(self.amount),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_time": self.current_time,
            "timeframe": self.timeframe.value,
            "is_finished": self.is_finished,
            "metadata": DecimalEncoder.encode(self.metadata)  # 处理元数据中的Decimal
        }

        # 处理动态属性
        if self._dynamic_attrs:
            # 将动态属性中的Decimal也转换为字符串
            result["dynamic_attrs"] = DecimalEncoder.encode(self._dynamic_attrs)

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "KLine":
        """
        从字典创建K线对象，自动处理字符串形式的Decimal值
        """
        # 创建基本对象
        kline = cls(
            symbol=data["symbol"],
            market=data["market"],  # market是必需字段
            open=data["open"],  # 自动在__post_init__中转换为Decimal
            close=data["close"],
            high=data["high"],
            low=data["low"],
            volume=data["volume"],
            amount=data["amount"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            current_time=data["current_time"],
            timeframe=data["timeframe"],  # 自动在__post_init__中转换为TimeFrame
            is_finished=data.get("is_finished", False),
            metadata=data["metadata"]
        )

        # 处理动态属性
        if "dynamic_attrs" in data and isinstance(data["dynamic_attrs"], dict):
            # 将动态属性中的字符串数字转换为Decimal
            decoded_attrs = DecimalEncoder.decode(data["dynamic_attrs"])
            for key, value in decoded_attrs.items():
                kline.set(key, value)

        return kline
