#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : rsrs.py
# @Software: PyCharm
from decimal import Decimal

from .t import T
from leek_core.models import TimeFrame, KLine


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
