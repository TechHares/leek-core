#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : bbi.py
# @Software: PyCharm

from .t import T
from .ma import MA


class BBI(T):
    """
    BBI（Bull and Bear Index）多空指标：
    使用四条简单移动平均（默认周期 3、6、12、24）的均值作为指标值。
    当最长周期尚未形成时返回 None。
    """

    def __init__(self, periods=(3, 6, 12, 24), max_cache=100):
        T.__init__(self, max_cache)
        if len(periods) != 4:
            raise ValueError("BBI requires 4 periods")
        self.periods = tuple(int(p) for p in periods)
        self.mas = [MA(p) for p in self.periods]
        self.bbi = None

    def update(self, data):
        try:
            values = [m.update(data) for m in self.mas]
            if any(v is None for v in values):
                return self.bbi
            self.bbi = sum(values) / 4
            return self.bbi
        finally:
            if data.is_finished:
                self.cache.append(self.bbi)


