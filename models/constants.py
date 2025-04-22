#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
常量模块，定义系统中使用的常量和枚举类型。
"""

from enum import Enum
from typing import Optional


class TimeFrame(Enum):
    """
    K线时间粒度枚举
    """
    # 时间粒度与毫秒的映射表
    __milliseconds_map = {
        "tick": None,
        "1s": 1 * 1000,
        "5s": 5 * 1000,
        "15s": 15 * 1000,
        "30s": 30 * 1000,
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1H": 60 * 60 * 1000,
        "2H": 2 * 60 * 60 * 1000,
        "4H": 4 * 60 * 60 * 1000,
        "6H": 6 * 60 * 60 * 1000,
        "8H": 8 * 60 * 60 * 1000,
        "12H": 12 * 60 * 60 * 1000,
        "1D": 24 * 60 * 60 * 1000,
        "3D": 3 * 24 * 60 * 60 * 1000,
        "1W": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,  # 约30天
    }
    
    TICK = "tick"
    S1 = "1s"     # 1秒
    S5 = "5s"     # 5秒
    S15 = "15s"   # 15秒
    S30 = "30s"   # 30秒
    M1 = "1m"     # 1分钟
    M3 = "3m"     # 3分钟
    M5 = "5m"     # 5分钟
    M15 = "15m"   # 15分钟
    M30 = "30m"   # 30分钟
    H1 = "1H"     # 1小时
    H2 = "2H"     # 2小时
    H4 = "4H"     # 4小时
    H6 = "6H"     # 6小时
    H8 = "8H"     # 8小时
    H12 = "12H"   # 12小时
    D1 = "1D"     # 日线
    D3 = "3D"     # 3日线
    W1 = "1W"     # 周线
    MON1 = "1M"   # 月线
    
    @classmethod
    def from_string(cls, value: str) -> "TimeFrame":
        """从字符串创建TimeFrame对象"""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"无效的时间粒度: {value}")
    
    @property
    def milliseconds(self) -> Optional[int]:
        """返回时间粒度的毫秒表示"""
        return TimeFrame.__milliseconds_map[self.value]


class DataType(Enum):
    """处理的金融数据类型。"""
    KLINE = "kline"               # K线数据
    TICK = "tick"                 # 逐笔成交数据
    ORDER_BOOK = "order_book"     # 订单簿数据
    TRADE = "trade"               # 成交数据
    FUNDING_RATE = "funding_rate" # 资金费率（期货）
    OPEN_INTEREST = "open_interest" # 持仓量（期货）
    INDEX = "index"               # 指数数据
    FUNDAMENTAL = "fundamental"   # 基本面数据
    NEWS = "news"                 # 新闻数据


class AssetType(Enum):
    """金融资产类型。"""
    STOCK = "stock"           # 股票
    FUTURES = "futures"       # 期货
    CRYPTO = "crypto"         # 加密货币
    FOREX = "forex"           # 外汇
    INDEX = "index"           # 指数
    BOND = "bond"             # 债券
    COMMODITY = "commodity"   # 商品
    OPTION = "option"         # 期权


class MarketStatus(Enum):
    """市场交易状态。"""
    OPEN = "open"                 # 开盘
    CLOSED = "closed"             # 收盘
    PRE_MARKET = "pre_market"     # 盘前
    POST_MARKET = "post_market"   # 盘后
    LUNCH_BREAK = "lunch_break"   # 午休
    HOLIDAY = "holiday"           # 假期

class PositionSide(Enum):
    """持仓方向。"""
    LONG = 0    # 多头
    SHORT = 1   # 空头
    BOTH = 2    # 双向持仓


class StrategyInstanceState(Enum):
    """策略实例状态。"""
    CREATED = "created"      # 空仓
    READY = "ready"      # 空仓
    ENTERING = "entering"            # 入场中
    HOLDING = "holding"              # 持仓中
    EXITING = "exiting"              # 出场中
    STOPPING = "stopping"            # 停止中
    STOPPED = "stopped"              # 已停止

class StrategyState(Enum):
    """策略状态。"""
    CREATED = "created"      # 空仓
    PREPARING = "preparing"  # 准备中
    Running = "running"      # 运行中
    STOPPING = "stopping"    # 停止中
    STOPPED = "stopped"      # 已停止

class OrderType(Enum):
    """
    交易类型 OrderType
    """
    MarketOrder = 1  # 市价单
    LimitOrder = 2  # 限价单


class TradeMode(Enum):
    """
    交易模式
    """
    ISOLATED = "isolated"  # 保证金模式-逐仓
    CROSS = "cross"  # 保证金模式-全仓
    CASH = "cash"  # 非保证金模式-非保证金
    SPOT_ISOLATED = "spot_isolated" # 现货-带单


class PosMode(Enum):
    """
    持仓方式: 仅适用交割/永续
    """
    LONG_SHORT_MODE = "long_short_mode"
    NET_MODE = "net_mode"


class TradeInsType(Enum):
    """
    交易产品类型
    """
    SPOT = 1  # 现货
    MARGIN = 2  # 杠杆
    SWAP = 3  # 合约
    FUTURES = 4  # 期货
    OPTION = 5  # 期权

    def __str__(self):
        return self.name.upper()


class OrderStatus(Enum):
    """
    订单状态。
    """
    CREATED = "created"      # 已创建，待提交
    SUBMITTED = "submitted"  # 已提交到交易所/撮合系统
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"        # 全部成交
    CANCELED = "canceled"    # 已撤单
    REJECTED = "rejected"    # 被拒绝
    EXPIRED = "expired"      # 过期未成交
    ERROR = "error"          # 异常
