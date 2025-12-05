#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
未来收益率标签（回归任务）

适用于：通用回归任务、趋势预测

计算未来 N 期的收益率，支持简单收益率和对数收益率两种方式。
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

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
    display_name = "未来收益率标签"
    init_params = [
        Field(name="periods", label="未来期数", type=FieldType.INT, default=1, description="未来几期"),
        Field(name="use_log", label="使用对数收益率", type=FieldType.BOOLEAN, default=False, description="是否使用对数收益率"),
    ]
    
    def __init__(self, periods: int = 1, use_log: bool = False):
        super().__init__()
        self.periods = periods
        self.use_log = use_log
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成未来收益率标签
        
        计算公式:
        - 简单收益率: (close[t+n] - close[t]) / close[t]
        - 对数收益率: log(close[t+n] / close[t])
        
        :param df: 包含 close 列的 DataFrame
        :return: 标签 Series
        """
        close = df['close']
        
        if self.use_log:
            future_close = close.shift(-self.periods)
            label = np.log(future_close / close)
        else:
            future_close = close.shift(-self.periods)
            label = (future_close - close) / close
        
        return pd.Series(label, name=self.label_name)

