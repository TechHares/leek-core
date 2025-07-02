#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
制定和实施交易策略的策略模块。
"""
from .base import Strategy, StrategyCommand
from .context import StrategyContext, StrategyWrapper
from .cta import CTAStrategy
from .strategy_mode import StrategyMode, KlineSimple, Single

from .strategy_dmi import DMIStrategy
# from.strategy_debug import DebugStrategy

__all__ = [
    'StrategyCommand',
    'CTAStrategy',
    'StrategyWrapper',
    'Strategy',
    'StrategyContext',
    'KlineSimple',
    'Single',
    'DMIStrategy',
    # 'DebugStrategy',
]