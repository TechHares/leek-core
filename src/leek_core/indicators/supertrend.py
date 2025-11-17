#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : supertrend.py
# @Software: PyCharm

from leek_core.models import PositionSide, KLine
from decimal import Decimal
from .atr import ATR
from .t import T


class SuperTrend(T):
    """
    超级趋势指标 (SuperTrend)
    
    基于ATR的趋势跟踪指标，用于识别价格趋势方向。
    
    参数:
        period: ATR周期，默认14
        multiplier: ATR乘数，默认3.0
        max_cache: 最大缓存数量，默认100
    """

    def __init__(self, period=14, multiplier=3.0, max_cache=100):
        T.__init__(self, max_cache)
        self.period = period
        self.multiplier = Decimal(multiplier)
        self.atr = ATR(window=period, max_cache=max_cache)
        self.prev_final_upper = None
        self.prev_final_lower = None
        self.prev_supertrend = None
        self.prev_trend = None  # PositionSide.LONG表示上升趋势，PositionSide.SHORT表示下降趋势

    def update(self, data: KLine):
        """
        更新SuperTrend指标
        
        参数:
            data: KLine数据
            
        返回:
            (supertrend_value, trend) 元组
            supertrend_value: SuperTrend值
            trend: 趋势方向，PositionSide.LONG表示上升趋势，PositionSide.SHORT表示下降趋势，None表示未确定
        """
        supertrend = (None, None)
        final_upper = None
        final_lower = None
        try:
            # 计算ATR
            atr = self.atr.update(data)
            if atr is None:
                return supertrend

            # 计算基础带（Basic Band）
            basic_band = (data.high + data.low) / 2

            # 计算上带和下带
            upper_band = basic_band + (self.multiplier * atr)
            lower_band = basic_band - (self.multiplier * atr)

            # 计算最终带
            if self.prev_final_upper is None or self.prev_final_lower is None:
                # 初始化
                final_upper = upper_band
                final_lower = lower_band
            else:
                # 最终上带：如果收盘价 <= 前一个最终上带，取最小值；否则使用当前上带
                if data.close <= self.prev_final_upper:
                    final_upper = min(upper_band, self.prev_final_upper)
                else:
                    final_upper = upper_band
                # 最终下带：如果收盘价 >= 前一个最终下带，取最大值；否则使用当前下带
                if data.close >= self.prev_final_lower:
                    final_lower = max(lower_band, self.prev_final_lower)
                else:
                    final_lower = lower_band

            # 确定SuperTrend值和趋势方向
            # 标准SuperTrend算法：基于前一个趋势状态来判断趋势延续或反转
            if self.prev_trend is None:
                # 初始化：根据收盘价与基础带的关系确定初始趋势
                if data.close > basic_band:
                    supertrend_value = final_lower
                    trend = PositionSide.LONG  # 上升趋势
                else:
                    supertrend_value = final_upper
                    trend = PositionSide.SHORT  # 下降趋势
            else:
                # 基于前一个趋势状态判断
                if self.prev_trend == PositionSide.LONG:
                    # 前一个趋势是上升趋势
                    if data.close > self.prev_supertrend:
                        # 价格仍然高于前一个SuperTrend值，保持上升趋势
                        supertrend_value = final_lower
                        trend = PositionSide.LONG
                    else:
                        # 价格跌破前一个SuperTrend值，趋势反转
                        supertrend_value = final_upper
                        trend = PositionSide.SHORT
                else:
                    # 前一个趋势是下降趋势
                    if data.close <= self.prev_supertrend:
                        # 价格仍然低于或等于前一个SuperTrend值，保持下降趋势
                        supertrend_value = final_upper
                        trend = PositionSide.SHORT
                    else:
                        # 价格突破前一个SuperTrend值，趋势反转
                        supertrend_value = final_lower
                        trend = PositionSide.LONG

            supertrend = (supertrend_value, trend)
            return supertrend
        finally:
            if data.is_finished:
                # 更新前一个值
                if supertrend[0] is not None and final_upper is not None and final_lower is not None:
                    self.prev_supertrend = supertrend[0]
                    self.prev_trend = supertrend[1]
                    # 保存最终带
                    self.prev_final_upper = final_upper
                    self.prev_final_lower = final_lower
                self.cache.append(supertrend)

