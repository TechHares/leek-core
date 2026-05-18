#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测模块，用于策略模拟和性能测试。
"""
from .backtest import EnhancedBacktester
from .runner import run_backtest
from .strategy_evaluation import StrategyEvaluator
from .types import (
    BacktestConfig,
    BacktestMode,
    NormalBacktestResult,
    OptimizationObjective,
    BacktestResult,
    StrategyEvaluationConfig,
    StrategyEvaluationResult,
    WalkForwardResult,
    WindowResult,
)
__all__ = [
    "run_backtest",
    "BacktestConfig",
    "EnhancedBacktester",
    "BacktestResult",
    "NormalBacktestResult",
    "BacktestMode",
    "StrategyEvaluator",
    "StrategyEvaluationConfig",
    "StrategyEvaluationResult",
    "WalkForwardResult",
    "WindowResult",
    "OptimizationObjective",
]
