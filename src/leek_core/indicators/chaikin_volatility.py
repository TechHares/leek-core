#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : chaikin_volatility.py
# @Software: PyCharm

from .t import T
from .ma import EMA


class ChaikinVolatility(T):
    """
    Chaikin Volatility (CV) 指标
    衡量价格高低点之间的差异，用于识别市场波动性的变化
    
    计算公式：
    CV = ((EMA(High-Low, n) - EMA(High-Low, n)[m periods ago]) / EMA(High-Low, n)[m periods ago]) * 100
    
    参数：
    - ema_period: EMA 周期，默认 10
    - roc_period: 回看周期，默认 10
    """

    def __init__(self, ema_period=10, roc_period=10, max_cache=100):
        T.__init__(self, max_cache)
        self.ema_period = ema_period
        self.roc_period = roc_period
        # 使用 EMA 来计算 High-Low 的平滑值
        self.ema_hl = EMA(window=ema_period, vfunc=lambda x: x.high - x.low, max_cache=max_cache)

    def update(self, data):
        cv = None
        try:
            # 计算 High-Low 的 EMA
            ema_hl_value = self.ema_hl.update(data)
            
            if ema_hl_value is None:
                return cv
            
            # 获取历史 EMA 值（不包括当前值，因为当前值可能还没有被缓存）
            ema_history = self.ema_hl.last(self.roc_period)
            
            # 需要至少 roc_period 个历史数据点才能计算
            if len(ema_history) < self.roc_period:
                return cv
            
            # 获取 m 周期前的 EMA 值（最旧的值）
            ema_previous = ema_history[0]
            
            # 如果之前的值为 0，避免除零错误
            if ema_previous == 0:
                return cv
            
            # 计算百分比变化
            cv = ((ema_hl_value - ema_previous) / ema_previous) * 100
            
            return cv
        finally:
            if data.is_finished:
                self.cache.append(cv)

