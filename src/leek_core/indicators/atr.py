#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : atr.py
# @Software: PyCharm

from .t import T


class TR(T):
    """
    平均真实波动率
    """

    def __init__(self, max_cache=100):
        T.__init__(self, max_cache)
        self.pre_close = None

    def update(self, data):
        tr = data.high - data.low
        if self.pre_close is not None:
            tr = max(tr, abs(data.high - self.pre_close), abs(data.low - self.pre_close))
        if data.is_finished:
            self.cache.append(tr)
            self.pre_close = data.close
        return tr


class ATR(T):
    """
    真实波动率
    """

    def __init__(self, window=14, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.tr_cal = TR(window + 1)

    def update(self, data):
        atr = None
        try:
            tr = self.tr_cal.update(data)
            ls = self.tr_cal.last(self.window)
            if not data.is_finished:
                ls.append(tr)
            if len(ls) > self.window:
                ls = ls[-self.window:]
            atr = sum(ls) / len(ls)
            return atr
        finally:
            if data.is_finished:
                self.cache.append(atr)


if __name__ == '__main__':
    pass
