#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易信号数据结构定义。
本模块用于定义量化交易系统中策略产生的标准化信号结构，
信号用于在各组件（如策略、风控、交易执行等）之间传递交易意图。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from utils import generate_str
from .config import StrategyPositionConfig
from .constants import AssetType, TradeInsType
from .position import PositionSide


@dataclass
class Signal:
    """
    交易信号结构体。
    
    用于描述策略产生的标准化交易信号，
    作为策略与交易执行、风控等模块的数据桥梁。
    """
    data_source_instance_id: str  # 数据源实例ID，用于标识信号来源的数据源组件
    strategy_instance_id: str     # 策略实例ID，标识产生该信号的策略实例
    asset_type: AssetType         # 资产类型（如股票、期货、加密货币等），参见 AssetType 枚举
    ins_type: TradeInsType        # 交易品种类型（如现货、永续合约等），参见 TradeInsType 枚举
    symbol: str                   # 交易对或合约标识，例如 "BTC"、"AAPL"
    config: StrategyPositionConfig# 仓位配置
    quote_currency: str           # 计价币种，如 USDT、USD、CNY
    side: PositionSide            # 多空方向，PositionSide 枚举（LONG/SHORT/NETURAL等）
    ratio: Decimal                # 仓位比例，取值范围 0~1，表示本信号建议的持仓占比
    price: Decimal                # 交易价格，建议的交易价格
    signal_time: datetime         # 信号产生时间，用于标识信号的产生时间
    extra: Any = None             # 其他扩展信息（如信号置信度、触发条件、备注等，可选）

    def init_position(self):
        """
        初始化仓位
        :return: 仓位实例
        """
        from .position import Position
        return Position(
            position_id = generate_str(),
            strategy_id=self.strategy_instance_id,
            symbol=self.symbol,
            quote_currency=self.quote_currency,
            ins_type=self.ins_type,
            asset_type=self.asset_type,
            side=self.side,
            cost_price=Decimal(0),
            amount=Decimal(0),
            ratio=Decimal(0),
            signals=[
                self
            ]
        )