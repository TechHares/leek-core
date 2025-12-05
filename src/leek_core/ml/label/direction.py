#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方向标签（分类任务）

适用于：涨跌预测、方向判断

判断未来是涨/跌/震荡，支持二分类和三分类。
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator

class DirectionLabel(LabelGenerator):
    """
    方向标签（分类任务）
    
    判断未来价格方向，用于分类任务训练。
    
    参数:
        periods: 未来几期，默认1
        threshold: 涨跌阈值（百分比），默认0.0
        num_classes: 分类数，2=二分类，3=三分类，默认2
        label_name: 标签列名，默认"label"
    
    标签含义:
        - 二分类: 0=跌, 1=涨
        - 三分类: 0=跌, 1=震荡, 2=涨
    
    示例:
        >>> label_gen = DirectionLabel({
        ...     "periods": 1,
        ...     "threshold": 0.01,  # 1%阈值
        ...     "num_classes": 2
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    display_name = "方向标签"
    init_params = [
        Field(name="periods", label="未来期数", type=FieldType.INT, default=1, description="未来几期"),
        Field(name="threshold", label="涨跌阈值", type=FieldType.FLOAT, default=0.0, description="涨跌阈值（百分比）"),
        Field(name="num_classes", label="分类数", type=FieldType.INT, default=2, description="分类数，2=二分类，3=三分类"),
    ]
    
    def __init__(self, periods: int = 1, threshold: float = 0.0, num_classes: int = 2):
        super().__init__()
        self.periods = periods
        self.threshold = threshold
        self.num_classes = num_classes
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成方向标签
        
        :param df: 包含 close 列的 DataFrame
        :return: 标签 Series
        """
        close = df['close']
        future_close = close.shift(-self.periods)
        return_pct = (future_close - close) / close
        
        if self.num_classes == 2:
            label = (return_pct > self.threshold).astype(int)
        else:
            label = pd.cut(
                return_pct,
                bins=[-np.inf, -self.threshold, self.threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
        
        return pd.Series(label, name=self.label_name)

