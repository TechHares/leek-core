#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风控模块，包含仓位和策略风险控制组件。
仓位风控负责对策略仓位进行控制，止盈止损等
策略风险控制负责对策略信号进行控制，决定对信号是否开仓。
"""


from .position import PositionPolicy
from .strategy import StrategyPolicy

__all__ = [
    'PositionPolicy',
    'StrategyPolicy',
]