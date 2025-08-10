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

from .config import StrategyPositionConfig
from .constants import AssetType, TradeInsType
from .position import PositionSide


@dataclass
class Asset:
    """
    资产约束模型，用于描述信号中建议交易的具体资产及其配置。
    该模型包含资产类型、交易品种、标的、方向、建议价格、持仓比例等信息，
    用于风控、仓位管理和交易执行等环节的资产筛选与约束。
    """
    asset_type: AssetType      # 资产类型（如股票、期货、加密货币等），参见 AssetType 枚举
    ins_type: TradeInsType     # 交易品种类型（如现货、永续合约等），参见 TradeInsType 枚举
    symbol: str               # 交易对或合约标识，例如 "BTC"、"AAPL"

    side: PositionSide        # 多空方向，PositionSide 枚举（LONG/SHORT/NEUTRAL等）
    price: Decimal            # 交易价格，建议的交易价格
    ratio: Decimal            # 仓位比例，取值范围 0~1，表示本信号建议的持仓占比
    actual_ratio: Decimal = None # 实际仓位比例
    is_open: bool = False     # 是否为开仓信号，用于表示信号是否为开仓信号

    quote_currency: str = None# 计价币种，如 USDT、USD、CNY
    extra: Any = None         # 其他扩展信息（如信号置信度、触发条件、备注等，可选）

    @property
    def asset_key(self) -> str:
        return f"{self.symbol}_{self.quote_currency}_{self.ins_type.value}_{self.asset_type.value}_{self.side.value}"

@dataclass
class Signal:
    """
    交易信号模型，用于在量化交易系统中传递策略产生的信号。
    每个信号包含其来源、相关配置、产生时间、约束条件及扩展信息。
    该模型作为策略与执行、风控等组件之间的标准数据结构。
    """
    signal_id: str                # 信号ID
    data_source_instance_id: str  # 数据源实例ID，用于标识信号来源的数据源组件
    strategy_id: str              # 策略实例ID，标识产生该信号的策略实例
    strategy_instance_id: str     # 策略实例ID，标识产生该信号的策略实例
    signal_time: datetime         # 信号产生时间，用于标识信号的产生时间

    assets: list[Asset] = list            # 断言信息，用于描述信号的约束条件（如仓位限制、风险控制等）
    config: StrategyPositionConfig = None # 仓位配置
    extra: Any = None                     # 其他扩展信息（如信号置信度、触发条件、备注等，可选）
