#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Optimizer for Backtest Engine

性能优化模块：
1. 数据预加载和缓存
2. 内存管理和优化
3. 计算并行化
4. 结果缓存
"""

from typing import Dict, List

import numpy as np

from leek_core.utils import get_logger

logger = get_logger(__name__)


def vectorized_operations(data: np.ndarray, operations: List[str], periods_per_year: int = 365) -> Dict[str, np.ndarray]:
    """向量化操作"""
    results = {}


    if 'returns' in operations:
        results['returns'] = np.diff(data) / data[:-1]

    if 'log_returns' in operations:
        results['log_returns'] = np.diff(np.log(data))

    if 'moving_average' in operations:
        # 简单移动平均
        window = 20
        if len(data) >= window:
            results['moving_average'] = np.convolve(
                data, np.ones(window) / window, mode='valid'
            )

    if 'volatility' in operations:
        returns = np.diff(data) / data[:-1]
        results['full_volatility'] = np.std(returns, ddof=1 if len(returns) > 1 else 0) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
        # 没用到滚动波动率， 先不计算
        # if len(returns) >= 20:
        #     # 滚动波动率
        #     window = 20
        #     volatility = []
        #     for i in range(window - 1, len(returns)):
        #         vol = np.std(returns[i - window + 1:i + 1]) * np.sqrt(252)
        #         volatility.append(vol)
        #     results['volatility'] = np.array(volatility)
        # elif len(returns) >= 2:
        #     # 样本不足20根时，使用全样本年化波动率，避免夏普恒为0
        #     results['volatility'] = np.array([full_vol])

    return results
