#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : r_squared.py
# @Software: PyCharm

from collections import deque
from decimal import Decimal

import numpy as np

from .t import T


class RSquared(T):
    """
    R²（决定系数）趋势强度指标

    用于测量价格趋势的强度，而非仅判断方向：
    - R² 接近 1.0（如 0.85 以上）：表示强劲、清晰的趋势
    - R² 接近 0.0：表示震荡、横盘、随机波动

    通过对窗口内价格序列做线性回归，计算决定系数 R² = 1 - (SS_res / SS_tot)。

    参数:
        window: 计算窗口大小，默认 20
        vfunc: 价格提取函数，默认 lambda x: x.close
        max_cache: 最大缓存数量，默认 100
    """

    def __init__(self, window=20, vfunc=lambda x: x.close, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window - 1)
        self.vfunc = vfunc

    def update(self, data, finish_v=None):
        """
        更新 R² 指标。

        参数:
            data: KLine 数据
            finish_v: 是否强制完成，默认 None（使用 data.is_finished）

        返回:
            R² 值（0~1），数据不足或无方差时返回 None/0
        """
        r_squared = None
        try:
            if len(self.q) < self.window - 1:
                return r_squared

            ls = list(self.q)
            values = [self.vfunc(d) for d in ls]
            values.append(self.vfunc(data))
            prices = np.array([float(v) for v in values], dtype=float)

            n = len(prices)
            x = np.arange(n, dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(prices)

            x_var = np.sum((x - x_mean) ** 2)
            if x_var == 0:
                return Decimal(0)

            slope = np.sum((x - x_mean) * (prices - y_mean)) / x_var
            intercept = y_mean - slope * x_mean
            y_pred = intercept + slope * x

            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - y_mean) ** 2)

            if ss_tot == 0:
                r_squared = Decimal(0)
            else:
                r2 = 1.0 - (ss_res / ss_tot)
                r2 = max(0.0, min(1.0, r2))
                r_squared = Decimal(str(r2))

            return r_squared
        finally:
            if finish_v or (finish_v is None and data.is_finished):
                self.q.append(data)
                self.cache.append(r_squared)


if __name__ == '__main__':
    pass
