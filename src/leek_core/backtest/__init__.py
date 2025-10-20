#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测模块，用于策略模拟和性能测试。
"""
from .backtest import EnhancedBacktester
from .runner import run_backtest
from .types import BacktestMode, BacktestResult, BacktestConfig, WalkForwardResult, WindowResult, OptimizationObjective, NormalBacktestResult
__all__ = [
    "run_backtest",
    "BacktestConfig",
    "EnhancedBacktester",
    "BacktestResult",
    "SimpleEngine",
    "NormalBacktestResult",
    "BacktestMode",
    "WalkForwardResult",
    "WindowResult",
    "OptimizationObjective",
]
