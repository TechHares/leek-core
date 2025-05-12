#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : rsrs.py
# @Software: PyCharm
from decimal import Decimal

from indicators.t import T
from models.constants import TimeFrame
from models.data import KLine


class RSRS(T):
    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window

    def update(self, data):
        if len(self.cache) < self.window:
            return None
            
        ls = list(self.cache)[-self.window:]
        ls.append(data)
        
        # 计算RSRS
        high = [d.high for d in ls]
        low = [d.low for d in ls]
        
        # 计算斜率
        slope = (max(high) - min(low)) / self.window
        
        if data.is_finished:
            self.cache.append(data)
            
        return slope


# 测试数据
test_data = [
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('11'),
          close=Decimal('11'),
          high=Decimal('21'),
          low=Decimal('1'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('12'),
          close=Decimal('12'),
          high=Decimal('22'),
          low=Decimal('2'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('13'),
          close=Decimal('13'),
          high=Decimal('23'),
          low=Decimal('3'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('14'),
          close=Decimal('14'),
          high=Decimal('24'),
          low=Decimal('4'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('15'),
          close=Decimal('15'),
          high=Decimal('25'),
          low=Decimal('5'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('16'),
          close=Decimal('16'),
          high=Decimal('26'),
          low=Decimal('6'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('17'),
          close=Decimal('17'),
          high=Decimal('27'),
          low=Decimal('7'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('18'),
          close=Decimal('18'),
          high=Decimal('28'),
          low=Decimal('8'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('19'),
          close=Decimal('19'),
          high=Decimal('29'),
          low=Decimal('9'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True),
    KLine(symbol="BTC-USDT",
          market="okx",
          open=Decimal('110'),
          close=Decimal('110'),
          high=Decimal('210'),
          low=Decimal('10'),
          volume=Decimal('0'),
          amount=Decimal('0'),
          start_time=0,
          end_time=0,
          current_time=0,
          timeframe=TimeFrame.MINUTE_1,
          is_finished=True)
]
