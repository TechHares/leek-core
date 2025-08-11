#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析模块，用于性能评估和归因分析。
"""

from .performance import (
    PerformanceMetrics,
    calculate_performance_from_values,
    calculate_period_comparison_from_values
)
# Quantstats 后续考虑用改库计算策略性能
__all__ = [
    'PerformanceMetrics',
    'calculate_performance_from_values',
    'calculate_period_comparison_from_values'
] 