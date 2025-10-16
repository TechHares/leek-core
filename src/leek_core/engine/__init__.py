#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易引擎模块，负责协调数据源、策略、风控和仓位管理
"""

from .base import Engine
from .process import ProcessEngine, ProcessEngineClient
from .indicator_view import IndicatorView
from .stategy_debug import StrategyDebugView
from .backtest.backtest import EnhancedBacktester
from .backtest.types import BacktestMode, BacktestResult, BacktestConfig, WalkForwardResult, WindowResult, OptimizationObjective, NormalBacktestResult
__all__ = [
    "Engine",
    "ProcessEngine",
    "ProcessEngineClient",
    "StrategyDebugView",
    "IndicatorView",
    "EnhancedBacktester",
    "BacktestConfig",
    "BacktestResult",
    "NormalBacktestResult",
    "BacktestMode",
    "WalkForwardResult",
    "WindowResult",
    "OptimizationObjective",
]
