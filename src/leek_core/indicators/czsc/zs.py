#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅组件
"""
from collections import deque
from dataclasses import dataclass

from leek_core.models.parameter import DateTimeUtils
from .base import Component
from .d import D

from leek_core.models import KLine, PositionSide

@dataclass
class ZSModel:
    high: float
    low: float
    start_time: int
    end_time: int

class ZS(Component):
    """
    缠中说禅 中枢
    该实现不关注中枢升级和中枢扩张/扩展，只处理中枢的成立、破坏， 用于判断价格支撑位置和阻力位置， 不适合用于递归表达
    初始化规定方向，中枢只关注该方向的中枢
    """
    def __init__(self):
        super().__init__(maxlen=30)
        self.high = None
        self.low = None

    def update(self, d: D):
        if d:
            self.eles.append(d)
        
        if len(self.eles) < 3:
            return
        
        if self.is_zs():
            if self.destroy_zs():
                self.update(None)
            return
        if self.create_zs():
            self.update(None)
    
    def destroy_zs(self):
        if len(self.eles) < 4:
            return
        for i in range(3, len(self.eles)):
            if not self.eles[i].is_finished:
                return
            if self.eles[i].low > self.high or self.eles[i].high < self.low:
                self.eles.popleft()
                self.high = None
                self.low = None
                return True

    def create_zs(self):
        self.high = min(list(self.eles)[0:3], key=lambda x: x.high).high
        self.low = max(list(self.eles)[0:3], key=lambda x: x.low).low
        if self.high > self.low:
            return True
        if self.eles[2].is_finished:
            self.eles.popleft()
        return False
    
    def is_zs(self) -> bool:
        return self.high is not None and self.low is not None and self.high > self.low
    
    @property
    def start_time(self):
        return self.eles[0].start_time
    
    @property
    def end_time(self):
        last = None
        for i in range(4, len(self.eles)):
            if self.low < self.eles[i].low < self.high or self.low < self.eles[i].high < self.high:
                last = self.eles[i-1].end_time
        return last if last is not None else self.eles[-1].end_time
    
    def to_model(self):
        return ZSModel(high=self.high, low=self.low, start_time=self.start_time, end_time=self.end_time)