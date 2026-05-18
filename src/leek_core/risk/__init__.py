#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控模块

- pre_trade: 信号级风控，对策略产出的开仓信号做准入检查
- post_trade: 仓位级风控，主动监测持仓触发止盈/止损/强平
"""

from .pre_trade import (
    StrategyPolicy,
    StrategyPolicyContext,
    RiskManager,
    StrategySignalLimit,
    StrategyProfitControl,
    StrategyTimeWindow,
)
from .post_trade import RiskPlugin, RiskContextContext

__all__ = [
    # pre-trade
    'StrategyPolicy',
    'StrategyPolicyContext',
    'RiskManager',
    'StrategySignalLimit',
    'StrategyProfitControl',
    'StrategyTimeWindow',
    # post-trade
    'RiskPlugin',
    'RiskContextContext',
]
