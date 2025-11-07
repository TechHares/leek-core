#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : extreme.py
# @Software: PyCharm
from collections import deque

from .t import T
from leek_core.models import KLine


class Extreme(T):
    """
    N日极值指标
    计算包含最新可能未完成K线的N日最高价和最低价
    """
    
    def __init__(self, window=20, max_cache=100):
        """
        初始化N日极值指标
        
        :param window: 计算窗口大小（天数）
        :param max_cache: 缓存大小
        """
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window - 1)
        self._current = None
    
    def update(self, data: KLine):
        """
        更新指标
        
        :param data: K线数据
        :return: Extreme对象，包含high和low属性
        """
        if self._current and data.is_finished:
            self.q.append(self._current)
        
        self._current = data
        return self.high, self.low
    
    @property
    def high(self):
        """
        获取N日最高价（包含最新可能未完成的K线）
        
        :return: Decimal，最高价，如果数据不足返回None
        """
        return max(self._current.high, [k.high for k in self.q]) if self.q else self._current.high
    
    @property
    def low(self):
        """
        获取N日最低价（包含最新可能未完成的K线）
        
        :return: Decimal，最低价，如果数据不足返回None
        """
        return min(self._current.low, [k.low for k in self.q]) if self.q else self._current.low

