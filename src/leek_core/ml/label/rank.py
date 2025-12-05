#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分位数排名标签（排序任务）

适用于：多因子打分策略、跨资产排序

计算未来收益率的排名，用于学习排序任务。跨资产可比性强，适合多因子打分策略。
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator

class RankLabel(LabelGenerator):
    """
    分位数排名标签（排序任务）
    
    计算未来收益率的排名，用于排序学习（Learning to Rank）。
    特别适合多因子打分策略，因为排名具有跨资产可比性。
    
    参数:
        periods: 未来几期，默认1
        method: 排名方法，"percentile"=百分位排名(0-1)，"rank"=原始排名(1-N)，默认"percentile"
        label_name: 标签列名，默认"label"
    
    适用策略:
        - 多因子打分：跨资产可比性强，适合排序
    
    示例:
        >>> label_gen = RankLabel({
        ...     "periods": 5,
        ...     "method": "percentile"
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    display_name = "分位数排名标签"
    init_params = [
        Field(name="periods", label="未来期数", type=FieldType.INT, default=1, description="未来几期"),
        Field(name="method", label="排名方法", type=FieldType.RADIO, default="percentile", 
              choices=[("percentile", "百分位排名(0-1)"), ("rank", "原始排名(1-N)")],
              description="排名方法"),
    ]
    
    def __init__(self, periods: int = 1, method: str = "percentile"):
        super().__init__()
        self.periods = periods
        self.method = method
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成排名标签
        
        :param df: 包含 close 列的 DataFrame
        :return: 标签 Series
        """
        close = df['close']
        future_close = close.shift(-self.periods)
        return_pct = (future_close - close) / close
        
        if self.method == "percentile":
            label = return_pct.rank(pct=True)
        else:
            label = return_pct.rank()
        
        return pd.Series(label, name=self.label_name)

