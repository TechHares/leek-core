#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
制定和实施交易策略的策略模块。
"""

from .base import Strategy
from .context import StrategyContext
from .cta import CTAStrategy
from .strategy_mode import StrategyMode, KlineSimple, Single

__all__ = [
    'CTAStrategy',
    'Strategy',
    'StrategyContext',
    'KlineSimple',
    'Single',
]