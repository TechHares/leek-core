#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅基础组件
"""
import abc
from collections import deque
from leek_core.models import PositionSide


class Component(metaclass=abc.ABCMeta):
    """
    缠中说禅基础组件定义
    """
    def __init__(self, direction: PositionSide | None = None, maxlen: int = 2000):
        self.eles = deque(maxlen=2000)
        self.direction: PositionSide | None = direction
        self.high = None
        self.low = None
        self.is_finished = None

    @property
    def max(self):
        return max([ele.high for ele in self.eles])

    @property
    def min(self):
        return min([ele.low for ele in self.eles])

    @property
    def is_up(self):
        return self.direction == PositionSide.LONG
    
    @property
    def is_down(self):
        return self.direction == PositionSide.SHORT
    
    @property
    def start_time(self):
        return self.eles[0].start_time

    @property
    def end_time(self):
        return self.eles[-1].end_time

    @property
    def size(self):
        return len(self.eles)

