#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : zscore.py
# @Software: PyCharm

from collections import deque
from decimal import Decimal

import numpy as np

from .t import T


class ZScore(T):
    """
    Z-Score指标（标准化分数）
    
    用于衡量当前值相对于历史均值的标准差倍数。
    Z-Score = (当前值 - 均值) / 标准差
    
    参数:
        window: 窗口大小，默认20
        vfunc: 取值函数，默认lambda x: x.close，可以自定义（如lambda x: x.high）
        max_cache: 最大缓存数量，默认100
    """

    def __init__(self, window=20, vfunc=lambda x: x.close, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window - 1)
        self.vfunc = vfunc

    def update(self, data, finish_v=None):
        """
        更新Z-Score指标
        
        参数:
            data: KLine数据
            finish_v: 是否强制完成，默认None（使用data.is_finished）
            
        返回:
            zscore值，如果数据不足则返回None
        """
        zscore = None
        try:
            if len(self.q) < self.window - 1:
                return zscore
            
            # 获取窗口内的所有值
            ls = list(self.q)
            values = [self.vfunc(d) for d in ls]
            current_value = self.vfunc(data)
            values.append(current_value)
            
            # 转换为numpy数组进行计算（统一转换为float）
            values_array = np.array([float(v) for v in values])
            current_value_float = float(current_value)
            
            # 计算均值和标准差
            mean = np.mean(values_array)
            std = np.std(values_array, ddof=0)  # 使用样本标准差（ddof=1）
            
            # 避免除零错误
            if std == 0:
                return Decimal(0)
            
            # 计算Z-Score
            zscore = Decimal(str((current_value_float - mean) / std))
            return zscore
        finally:
            if finish_v or (finish_v is None and data.is_finished):
                self.q.append(data)
                self.cache.append(zscore)


if __name__ == '__main__':
    pass

