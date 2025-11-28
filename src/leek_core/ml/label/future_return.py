#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
未来收益率标签（回归任务）

适用于：通用回归任务、趋势预测

计算未来 N 期的收益率，支持简单收益率和对数收益率两种方式。
"""
import numpy as np
import pandas as pd

from .base import LabelGenerator

class FutureReturnLabel(LabelGenerator):
    """
    未来收益率标签（回归任务）
    
    计算未来 N 期的收益率，用于回归任务训练。
    
    参数:
        periods: 未来几期，默认1
        use_log: 是否使用对数收益率，默认False（使用简单收益率）
        label_name: 标签列名，默认"label"
    
    示例:
        >>> label_gen = FutureReturnLabel({
        ...     "periods": 5,
        ...     "use_log": False,
        ...     "label_name": "future_return"
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.periods = int(params.get("periods", 1))
        self.use_log = params.get("use_log", False)
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成未来收益率标签
        
        计算公式:
        - 简单收益率: (close[t+n] - close[t]) / close[t]
        - 对数收益率: log(close[t+n] / close[t])
        
        :param df: 包含 close 列的 DataFrame
        :return: 增加了 label 列的 DataFrame
        """
        close = df['close']
        
        if self.use_log:
            future_close = close.shift(-self.periods)
            df[self.label_name] = np.log(future_close / close)
        else:
            future_close = close.shift(-self.periods)
            df[self.label_name] = (future_close - close) / close
        
        return df

