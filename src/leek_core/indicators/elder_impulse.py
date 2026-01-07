#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : elder_impulse.py
# @Software: PyCharm

from .t import T
from .ma import EMA
from .macd import MACD
from leek_core.models import KLine


class ElderImpulse(T):
    """
    埃尔德脉冲系统（Elder Impulse System）
    
    由Alexander Elder开发的技术分析指标，结合EMA和MACD来判断市场趋势方向。
    
    信号规则：
    - 看涨脉冲（+1）：EMA上升 且 MACD柱状图（DIF-DEA）> 0
    - 看跌脉冲（-1）：EMA下降 且 MACD柱状图（DIF-DEA）< 0
    - 中性（0）：其他情况或数据不足
    
    参数：
    - ema_period: EMA周期，默认13
    - macd_fast: MACD快线周期，默认12
    - macd_slow: MACD慢线周期，默认26
    - macd_signal: MACD信号线周期，默认9
    - max_cache: 最大缓存数量，默认100
    """
    
    def __init__(self, ema_period=13, macd_fast=12, macd_slow=26, macd_signal=9, max_cache=100):
        T.__init__(self, max_cache)
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # 初始化EMA和MACD实例
        self.ema = EMA(window=ema_period, max_cache=max_cache)
        self.macd = MACD(fast_period=macd_fast, slow_period=macd_slow, moving_period=macd_signal, max_cache=max_cache)
        
        # 保存前一个EMA值用于趋势判断
        self.prev_ema = None
    
    def update(self, data: KLine):
        """
        更新指标并返回脉冲信号
        
        参数:
            data: K线数据
            
        返回:
            int: 1（看涨脉冲）、0（中性）、-1（看跌脉冲）
        """
        impulse = 0
        current_ema = None
        try:
            # 计算当前EMA
            current_ema = self.ema.update(data)
            if current_ema is None:
                return impulse
            
            # 计算MACD
            dif, dea = self.macd.update(data)
            if dif is None or dea is None:
                return impulse
            
            # 计算MACD柱状图（histogram）
            macd_histogram = dif - dea
            
            # 判断EMA趋势方向
            if self.prev_ema is not None:
                # EMA上升
                ema_rising = current_ema > self.prev_ema
                # EMA下降
                ema_falling = current_ema < self.prev_ema
                
                # 生成脉冲信号
                if ema_rising and macd_histogram > 0:
                    # 看涨脉冲：EMA上升 且 MACD柱状图为正
                    impulse = 1
                elif ema_falling and macd_histogram < 0:
                    # 看跌脉冲：EMA下降 且 MACD柱状图为负
                    impulse = -1
                else:
                    # 中性状态
                    impulse = 0
            else:
                # 数据不足，无法判断趋势
                impulse = 0
            
            return impulse
        finally:
            # 更新前一个EMA值（仅在K线完成时）
            if data.is_finished:
                if current_ema is not None:
                    self.prev_ema = current_ema
                self.cache.append(impulse)


if __name__ == '__main__':
    pass

