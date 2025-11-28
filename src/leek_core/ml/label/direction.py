#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方向标签（分类任务）

适用于：涨跌预测、方向判断

判断未来是涨/跌/震荡，支持二分类和三分类。
"""
import numpy as np
import pandas as pd

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
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.periods = int(params.get("periods", 1))
        self.threshold = float(params.get("threshold", 0.0))
        self.num_classes = int(params.get("num_classes", 2))
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成方向标签
        
        :param df: 包含 close 列的 DataFrame
        :return: 增加了 label 列的 DataFrame
        """
        close = df['close']
        future_close = close.shift(-self.periods)
        return_pct = (future_close - close) / close
        
        if self.num_classes == 2:
            df[self.label_name] = (return_pct > self.threshold).astype(int)
        else:
            df[self.label_name] = pd.cut(
                return_pct,
                bins=[-np.inf, -self.threshold, self.threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
        
        return df

