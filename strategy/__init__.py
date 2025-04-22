#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
制定和实施交易策略的策略模块。
"""

from .base import Strategy
from .context import StrategyContext
from .cta import STAStrategy
from .sidecar import CTAStrategySidecar
from .strategy_mode import StrategyMode, KlineSimple, Single

__all__ = [
    'STAStrategy',
    'Strategy',
    'StrategyContext',
    'CTAStrategySidecar',
    'KlineSimple',
    'Single',
]