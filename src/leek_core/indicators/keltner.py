#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : keltner.py
# @Software: PyCharm

from decimal import Decimal

from .t import T
from .ma import EMA
from .atr import ATR


class KeltnerChannel(T):
    """
    肯特纳通道（Keltner Channel）
    
    由Chester Keltner在1960年代提出，后由Linda Raschke改进。
    是一个基于波动率的包络线指标，用于识别趋势方向和潜在的价格突破。
    
    通道组成：
    - 中轨（Middle Line）：收盘价的EMA
    - 上轨（Upper Band）：中轨 + ATR × multiplier
    - 下轨（Lower Band）：中轨 - ATR × multiplier
    
    参数：
    - ema_period: EMA周期，默认20
    - atr_period: ATR周期，默认10
    - multiplier: ATR乘数，默认2.0
    - max_cache: 最大缓存数量，默认100
    
    用法：
    - 价格突破上轨：可能的上升趋势信号
    - 价格跌破下轨：可能的下降趋势信号
    - 通道收窄：波动性降低，可能即将突破
    - 通道扩张：波动性增加，趋势可能加强
    """

    def __init__(self, ema_period=20, atr_period=10, multiplier=2.0, max_cache=100):
        T.__init__(self, max_cache)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = Decimal(str(multiplier))
        self.ema = EMA(ema_period, max_cache=ema_period + 1)
        self.atr = ATR(atr_period, max_cache=atr_period + 1)

    def update(self, data):
        """
        更新指标并返回通道值
        
        参数:
            data: K线数据
            
        返回:
            tuple: (lower_band, middle, upper_band)
            - lower_band: 下轨
            - middle: 中轨（EMA）
            - upper_band: 上轨
            如果数据不足，返回 (None, None, None)
        """
        channel = (None, None, None)
        try:
            # 计算EMA（中轨）
            middle = self.ema.update(data)
            if middle is None:
                return channel

            # 计算ATR
            atr = self.atr.update(data)
            if atr is None:
                return channel

            # 计算上下轨
            atr_band = Decimal(str(atr)) * self.multiplier
            upper_band = middle + atr_band
            lower_band = middle - atr_band

            channel = (lower_band, middle, upper_band)
            return channel
        finally:
            if data.is_finished:
                self.cache.append(channel)


if __name__ == '__main__':
    pass
