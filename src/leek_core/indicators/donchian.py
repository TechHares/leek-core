#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : donchian.py
# @Software: PyCharm

from collections import deque
from decimal import Decimal

from .t import T
from leek_core.models import KLine


class DonchianChannel(T):
    """
    唐奇安通道（Donchian Channel）
    
    由Richard Donchian开发的技术分析指标，用于识别价格突破和趋势跟踪。
    
    通道组成：
    - 上轨（Upper Band）：过去N个周期的最高价
    - 下轨（Lower Band）：过去N个周期的最低价
    - 中轨（Middle Band）：上下轨的平均值
    
    参数：
    - period: 计算周期，默认20
    - max_cache: 最大缓存数量，默认100
    """
    
    def __init__(self, period=20, max_cache=100):
        T.__init__(self, max_cache)
        self.period = period
        # 使用deque存储过去N个周期的K线数据
        self.q = deque(maxlen=period - 1)
    
    def update(self, data: KLine):
        """
        更新指标并返回通道值
        
        参数:
            data: K线数据
            
        返回:
            tuple: (lower_band, middle, upper_band)
            - lower_band: 下轨（最低价）
            - middle: 中轨（上下轨平均值）
            - upper_band: 上轨（最高价）
            如果数据不足，返回 (None, None, None)
        """
        channel = (None, None, None)
        try:
            # 需要至少period个数据点才能计算
            if len(self.q) < self.period - 1:
                return channel
            
            # 获取历史K线数据
            history = list(self.q)
            history.append(data)
            
            # 计算上轨：过去N个周期的最高价
            upper_band = max(k.high for k in history)
            
            # 计算下轨：过去N个周期的最低价
            lower_band = min(k.low for k in history)
            
            # 计算中轨：上下轨的平均值
            middle = (upper_band + lower_band) / Decimal('2')
            
            channel = (lower_band, middle, upper_band)
            return channel
        finally:
            # 仅在K线完成时更新缓存
            if data.is_finished:
                self.q.append(data)
                self.cache.append(channel)


if __name__ == '__main__':
    pass

