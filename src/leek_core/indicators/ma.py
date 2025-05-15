#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : ma.py
# @Software: PyCharm
import math
from collections import deque
from decimal import Decimal

import numpy as np

from indicators.t import T


class MA(T):
    """
    简单平均
    """

    def __init__(self, window=9, vfunc=lambda x: x.close, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window - 1)
        self.vfunc = vfunc

    def update(self, data, finish_v=None):
        ma = None
        try:
            if len(self.q) < self.window - 1:
                return ma
            ls = list(self.q)
            ma = sum([self.vfunc(d) for d in ls], self.vfunc(data)) / self.window
            return ma
        finally:
            if finish_v or (finish_v is None and data.is_finished == 1):
                self.q.append(data)
                self.cache.append(ma)


class EMA(T):
    """
    加权平均
    """

    def __init__(self, window=9, max_cache=100, vfunc=lambda x: x.close):
        T.__init__(self, max_cache)
        self.window = window
        self.vfunc = vfunc
        self.pre_ma = None
        self.alpha = Decimal(2 / (self.window + 1))

    def update(self, data, finish_v=None):
        ma = None
        try:
            if self.pre_ma is None:
                ma = self.vfunc(data)
                return ma
            ma = self.alpha * self.vfunc(data) + (1 - self.alpha) * self.pre_ma
            return ma
        finally:
            if finish_v or (finish_v is None and data.is_finished == 1):
                self.pre_ma = ma
                self.cache.append(ma)


class WMA(T):
    """
    加权移动平均
    """

    def __init__(self, window=9, max_cache=100, vfunc=lambda x: x.close):
        T.__init__(self, max_cache)
        self.window = window
        self.vfunc = vfunc
        self.weights = window * (window + 1) // 2
        self.q = deque(maxlen=window - 1)

    def update(self, data, finish_v=None):
        ma = None
        try:
            if len(self.q) < self.window - 1:
                return ma

            ls = list(self.q)
            ls.append(self.vfunc(data))
            w_sum = 0
            for i in range(len(ls)):
                w_sum += (i + 1) * ls[i]
            ma = w_sum / self.weights
            return ma
        finally:
            if finish_v or (finish_v is None and data.is_finished == 1):
                self.q.append(self.vfunc(data))
                self.cache.append(ma)


class HMA(T):
    """
    赫尔移动平均线（Hull Moving Average，简称HMA）
    """

    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.wma1 = WMA(int(window/2))
        self.wma2 = WMA(window)
        self.wma3 = WMA(int(math.sqrt(window)), vfunc=lambda x: x)

    def update(self, data, finish_v=None):
        ma = None
        try:
            wma1 = self.wma1.update(data)
            wma2 = self.wma2.update(data)
            if wma1 is None or wma2 is None:
                return ma
            hma_noo_smooth = (wma1 * 2 - wma2)
            ma = self.wma3.update(hma_noo_smooth, data.is_finished == 1)
            return ma
        finally:
            if data.is_finished == 1:
                self.cache.append(ma)


class KAMA(T):
    """
    卡夫曼自适应移动均线
    """

    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window - 1)

        self.fast_n = 2
        self.slow_n = 30
        self.pre_ama = None

        self.fast_factory = Decimal(2 / (self.fast_n + 1))
        self.slow_factory = Decimal(2 / (self.slow_n + 1))

    def update(self, data):
        ma = None
        try:
            if len(self.q) < self.window - 1:
                return ma
            ls = list(self.q)

            price_direction = data.close - ls[0].close
            volatility = sum([abs(d.close - ls[0].close) for d in ls]) + abs(price_direction)
            er = price_direction / volatility if volatility != 0 else 1
            c = er * self.fast_factory + (1 - er) * self.slow_factory
            c_squared = c ** 2
            ma = c_squared * data.close + (1 - c_squared) * self.pre_ama
            return ma
        finally:
            if data.is_finished == 1:
                self.q.append(data)
                self.cache.append(ma)
                self.pre_ama = data.close if ma is None else ma


class FRAMA(T):
    """
    分形自适应移动平均线(Fractal Adaptive Moving Average)
    """

    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window + window % 2
        self.q = deque(maxlen=self.window - 1)

        self.pre_ama = None

    def update(self, data):
        ma = None
        try:
            if len(self.q) < self.window - 1:
                return ma
            ls = list(self.q)
            ls.append(data)
            n1 = 2 * (max([d.high for d in ls[:self.window // 2]]) - min(
                [d.low for d in ls[:self.window // 2]])) / self.window
            n2 = 2 * (max([d.high for d in ls[self.window // 2:]]) - min(
                [d.low for d in ls[self.window // 2:]])) / self.window
            n3 = (max([d.high for d in ls]) - min([d.low for d in ls])) / self.window
            d = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
            a = Decimal(math.exp(-4.6 * (d - 1)))
            ma = a * (data.high + data.low) / 2 + (1 - a) * self.pre_ama
            return ma
        finally:
            if data.is_finished == 1:
                self.q.append(data)
                self.cache.append(ma)
                self.pre_ama = data.close if ma is None else ma


class LLT(T):
    """
    二阶滤波器
    """

    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.a = Decimal(2 / (self.window + 1))
        a = self.a
        self.m1 = a - (a ** 2) / 4
        self.m2 = (a ** 2) / 2
        self.m3 = a - (a ** 2) * 3 / 4
        self.n1 = 2 * (1 - a)
        self.n2 = (1 - a) ** 2

        self.pre_llt = None
        self.pre_pre_llt = None
        self.pre_close = None
        self.pre_pre_close = None
        self.k = 0

    def update(self, data):
        llt = None
        try:
            if self.pre_close is None:
                if data.is_finished != 1 or self.pre_pre_close is None:
                    llt = data.close
                    return llt
                self.pre_close = data.close
                llt = self.a * data.close + self.pre_pre_llt * (1 - self.a)
                self.pre_llt = llt
                return llt

            llt = self.m1 * data.close + self.m2 * self.pre_close - self.m3 * self.pre_pre_close + self.n1 * self.pre_llt - self.n2 * self.pre_pre_llt
            return llt
        finally:
            if data.is_finished == 1:
                self.cache.append(llt)
                if self.pre_pre_close is None:
                    self.pre_pre_close = data.close
                    self.pre_pre_llt = llt
                    return

                self.pre_pre_close = self.pre_close
                self.pre_pre_llt = self.pre_llt
                self.pre_close = data.close
                self.pre_llt = llt


class SuperSmoother(T):
    """
    Super Smoother（超级平滑器）算法实现
    """

    def __init__(self, window=10, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window

        # 系数
        a1 = np.exp(-1.414 * np.pi / self.window)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / self.window)
        self.c2 = Decimal(b1)
        self.c3 = Decimal(-a1 * a1)
        self.c1 = 1 - self.c2 - self.c3

        self.pre_close = None

    def update(self, data):
        ss = None
        try:
            lst = list(self.cache)
            if self.pre_close is None:
                ss = data.close
            elif len(self.cache) == 1:
                ss = self.c1 * (data.close + self.pre_close) / 2 + self.c2 * lst[-1] + self.c3 * lst[-1]
            else:
                ss = self.c1 * (data.close + self.pre_close) / 2 + self.c2 * lst[-1] + self.c3 * lst[-2]
            return ss
        finally:
            if data.is_finished == 1:
                self.cache.append(ss)
                self.pre_close = data.close


class UltimateOscillator(T):
    """
    Ultimate Oscillator（超级振荡器） 算法实现
    过于灵敏
    """

    def __init__(self, window=10, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window

        # 系数
        a1 = np.exp(-1.414 * np.pi / self.window)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / self.window)
        self.c2 = Decimal(b1)
        self.c3 = Decimal(-a1 * a1)
        self.c1 = (1 + self.c2 - self.c3) / 4

        self.q = deque(maxlen=self.window)

    def update(self, data):
        ss = None
        try:
            if len(self.cache) < 4:
                ss = data.close
            else:
                us = list(self.cache)
                prices = list(self.q)
                ss = ((1 - self.c1) * data.close + (2 * self.c1 - self.c2) * prices[-1] - (self.c1 + self.c3)
                      * prices[-2] + us[-1] * self.c2 + us[-2] * self.c3)
            return ss
        finally:
            if data.is_finished == 1:
                self.cache.append(ss)
                self.q.append(data.close)


if __name__ == '__main__':
    pass
