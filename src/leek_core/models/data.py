#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K线数据模型模块，定义K线数据传输对象。
"""
from abc import abstractmethod, ABCMeta
from builtins import set
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Union, Dict, Any, Set, List, ClassVar

from leek_core.utils import DecimalEncoder
from .constants import TimeFrame, TradeInsType, DataType, AssetType

# 设置Decimal精度
getcontext().prec = 28  # 设置足够高的精度以处理金融数据

# 定义数值类型，可以是Decimal或字符串（用于构造Decimal）
NumericType = Union[Decimal, str, int, float]


@dataclass
class Data(metaclass=ABCMeta):
    """
    数据源最终产出的数据结构
    """
    data_source_id: str = None  # 数据源ID
    data_type: DataType = None  # 数据类型
    target_instance_id: Set[str] = field(default_factory=set)  # 目标实例ID

    dynamic_attrs: Dict[str, Any] = field(default_factory=dict)
    # 元数据，可用于存储额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 保护的字段名称，这些字段不能被动态属性覆盖（类字段）
    _protected_fields: ClassVar[set[str]] = {"dynamic_attrs", 'metadata', "data_source_id", "data_type", "target_instance_id"}

    def __getattr__(self, name: str) -> Any:
        """
        获取动态属性值。当正常属性查找失败时调用。

        参数:
            name: 属性名称

        返回:
            属性值，如果不存在则返回None
        """
        if name in self._protected_fields or name in Data._protected_fields:
            return object.__getattribute__(self, name)

        # 检查动态属性字典
        if name in self.dynamic_attrs:
            return self.dynamic_attrs[name]

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
        if (name in self._protected_fields or name.startswith('_') or name in self.__annotations__ or
                name in Data._protected_fields):
            object.__setattr__(self, name, value)
            return

        # 否则作为动态属性存储
        self.dynamic_attrs[name] = value

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
        if name in self.dynamic_attrs:
            return self.dynamic_attrs[name]

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

    @property
    @abstractmethod
    def row_key(self) -> tuple:
        ...

    @staticmethod
    def _to_decimal(value: NumericType) -> Decimal:
        """
        将各种数值类型转换为Decimal
        """
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))  # 通过字符串转换避免浮点精度问题


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
    asset_type: AssetType = AssetType.CRYPTO  # 资产类型，默认为加密货币
    is_finished: bool = False

    # 保护的字段名称，这些字段不能被动态属性覆盖（类字段）
    _protected_fields: ClassVar[set[str]] = {
        'symbol', 'market', 'quote_currency', 'ins_type', 'open', 'close', 'high', 'low', 'volume',
        'amount',
        'start_time', 'end_time', 'current_time', 'timeframe', 'is_finished',
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
    def row_key(self) -> str:
        """
        获取K线的唯一标识符
        """
        return "%s_%s_%s_%s" % (self.symbol, self.quote_currency, self.ins_type.value, self.timeframe.value)
    
    @staticmethod
    def pack_row_key(symbol: str, quote_currency: str, ins_type: TradeInsType|int, timeframe: TimeFrame|str) -> str:
        """
        打包行键。
        """
        return "%s_%s_%s_%s" % (symbol, quote_currency, ins_type.value if isinstance(ins_type, TradeInsType) else ins_type, 
        timeframe.value if isinstance(timeframe, TimeFrame) else timeframe)

    @staticmethod
    def parse_row_key(row_key: str) -> tuple:
        """
        解析行键。
        """
        symbol, quote_currency, ins_type, timeframe = row_key.split("_")
        return symbol, quote_currency, int(ins_type), timeframe

@dataclass
class InitDataPackage(Data):
    pack_row_key: str = None
    history_datas: List[Data] = None

    @property
    def row_key(self) -> str:
        return self.pack_row_key

