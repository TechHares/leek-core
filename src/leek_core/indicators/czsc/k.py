#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅基础组件
"""
from collections import deque
from .base import Component

from leek_core.models import KLine, PositionSide


class CK(Component):
    """
    缠处理之后的K线
    """
    def __init__(self, k: KLine, side: PositionSide):
        super().__init__()
        self.eles.append(k)
        self.direction = side
        self.high = k.high
        self.low = k.low

    def update(self, k: KLine):
        if not self.include(k):
            self.is_finished = True
            return CK(k, PositionSide.LONG if self.high < k.high else PositionSide.SHORT)
        
        self.eles.append(k)
        if self.is_up:
            self.high = max(self.high, k.high)
            self.low = max(self.low, k.low)
        else:
            self.high = min(self.high, k.high)
            self.low = min(self.low, k.low)

    def include(self, k: KLine):
        return self.high <= k.high and self.low >= k.low or self.high >= k.high and self.low <= k.low
    
    def min_start_time(self):
        return min(self.eles, key=lambda k: k.low).start_time
    
    def max_start_time(self):
        return max(self.eles, key=lambda k: k.high).start_time
