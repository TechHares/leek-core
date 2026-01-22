#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价格尖峰检测指标 (Spike Detector)

基于论文 "Predicting Price Movements in High-Frequency Financial Data with Spiking Neural Networks"
(Ezinwoke & Rhodes, 2025) 的尖峰检测方法实现。

核心思想：
    - 计算窗口内的平均绝对收益率作为"尖峰强度"
    - 使用滚动中位数作为"尖峰阈值"
    - 当强度超过阈值时，判定为"尖峰"事件

输出:
    - spike_strength: 尖峰强度（窗口内平均绝对收益率）
    - spike_threshold: 尖峰阈值（滚动中位数）
    - is_spike: 是否为尖峰（1/0）
"""
from collections import deque

import numpy as np

from .t import T


class SpikeDetector(T):
    """
    价格尖峰检测指标
    
    检测价格的异常波动（尖峰事件），适用于：
    - 突破信号识别
    - 波动率异常检测
    - 风险事件预警
    
    参数:
        threshold_window: 阈值计算窗口，用于计算中位数，默认60
        strength_window: 强度计算窗口，用于计算平均绝对收益率，默认3
        max_cache: 最大缓存数量
    
    返回:
        tuple: (spike_strength, spike_threshold, is_spike)
    """

    def __init__(self, threshold_window=60, strength_window=3, max_cache=100):
        """
        参数:
        - threshold_window: 阈值计算窗口（滚动中位数），默认60
        - strength_window: 强度计算窗口（平均绝对收益率），默认3
        - max_cache: 最大缓存数量
        """
        T.__init__(self, max_cache)
        self.threshold_window = threshold_window
        self.strength_window = strength_window
        
        # 存储历史收益率
        self.returns_history = deque(maxlen=threshold_window)
        # 存储最近价格用于计算收益率
        self.last_price = None

    def update(self, data):
        """
        更新指标
        
        返回: (spike_strength, spike_threshold, is_spike)
        """
        result = (None, None, None)
        
        try:
            current_price = float(data.close)
            
            # 计算收益率
            if self.last_price is not None and self.last_price > 0:
                current_return = (current_price - self.last_price) / self.last_price
            else:
                current_return = None
            
            # 如果数据完成，更新历史
            if data.is_finished:
                self.last_price = current_price
                if current_return is not None:
                    self.returns_history.append(abs(current_return))
            
            # 需要足够的历史数据才能计算
            if len(self.returns_history) < self.strength_window:
                return result
            
            # 计算尖峰强度：最近 strength_window 个绝对收益率的平均值
            recent_returns = list(self.returns_history)[-self.strength_window:]
            spike_strength = np.mean(recent_returns)
            
            # 计算尖峰阈值：所有历史绝对收益率的中位数
            spike_threshold = np.median(list(self.returns_history))
            
            # 判定是否为尖峰
            is_spike = 1 if spike_strength > spike_threshold else 0
            
            result = (spike_strength, spike_threshold, is_spike)
            return result
            
        except Exception:
            return result
        finally:
            if data.is_finished and result[0] is not None:
                self.cache.append(result)


if __name__ == '__main__':
    pass
