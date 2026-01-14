#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : gann_hilo.py

"""
Gann Hi-Lo Trend (江恩高低趋势指标)
"""

from collections import deque
from leek_core.models import PositionSide, KLine
from .t import T


class GannHiLo(T):
    """
    Gann Hi-Lo Trend (江恩高低趋势指标)
    
    用于检测基于前N根K线高低点的潜在趋势。
    
    计算逻辑：
    1. Gann_High = 前shift根K线的最高价
    2. Gann_Low = 前shift根K线的最低价
    3. 趋势判断：
       - 上升趋势 (UpTrend): 收盘价 > Gann_Low
       - 下降趋势 (DownTrend): 收盘价 < Gann_High
    
    参数:
        shift: 回看K线数量，默认1（即前一根K线）
        max_cache: 最大缓存数量，默认100
    
    返回:
        (gann_high, gann_low, up_trend, down_trend) 元组
        gann_high: 江恩高点（前shift根K线的最高价）
        gann_low: 江恩低点（前shift根K线的最低价）
        up_trend: 是否上升趋势 (Close > Gann_Low)
        down_trend: 是否下降趋势 (Close < Gann_High)
    """

    def __init__(self, shift=1, max_cache=100):
        T.__init__(self, max_cache)
        self.shift = shift
        # 存储历史K线的high和low
        self.high_queue = deque(maxlen=shift)
        self.low_queue = deque(maxlen=shift)

    def update(self, data: KLine):
        """
        更新Gann Hi-Lo指标
        
        参数:
            data: KLine数据
            
        返回:
            (gann_high, gann_low, up_trend, down_trend) 元组
        """
        result = (None, None, None, None)
        
        try:
            # 需要至少shift根K线的历史数据
            if len(self.high_queue) < self.shift:
                return result
            
            # 计算Gann_High和Gann_Low（前shift根K线的高/低点）
            gann_high = max(self.high_queue)
            gann_low = min(self.low_queue)
            
            # 趋势判断
            up_trend = data.close > gann_low
            down_trend = data.close < gann_high
            
            result = (gann_high, gann_low, up_trend, down_trend)
            return result
            
        finally:
            if data.is_finished:
                # 添加当前K线的high和low到队列
                self.high_queue.append(data.high)
                self.low_queue.append(data.low)
                self.cache.append(result)
    
    def is_uptrend(self, data: KLine) -> bool:
        """
        判断是否上升趋势
        
        返回:
            True: 收盘价 > 江恩低点
            False: 否则
        """
        result = self.update(data)
        return result[2] if result[2] is not None else False
    
    def is_downtrend(self, data: KLine) -> bool:
        """
        判断是否下降趋势
        
        返回:
            True: 收盘价 < 江恩高点
            False: 否则
        """
        result = self.update(data)
        return result[3] if result[3] is not None else False
