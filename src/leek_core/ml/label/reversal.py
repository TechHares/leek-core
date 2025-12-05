#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
反转强度标签（分类/回归任务）

适用于：均值回归策略

识别超买超卖状态，捕捉价格反转的强度和时机。
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator

class ReversalStrengthLabel(LabelGenerator):
    """
    反转强度标签（分类/回归任务）
    
    识别超买超卖状态，捕捉价格反转的强度和时机。
    适用于均值回归策略。
    
    参数:
        periods: 未来几期，默认3
        lookback_window: 回看窗口（用于计算偏离度），默认20
        method: 标签类型，"classification"=分类，"regression"=回归，默认"classification"
        threshold: 反转强度阈值（用于分类），默认0.02（2%）
        label_name: 标签列名，默认"label"
    
    标签含义（分类模式）:
        - 1: 强烈反转（超买后下跌或超卖后上涨）
        - 0: 无反转或反转较弱
    
    适用策略:
        - 均值回归：识别超买超卖
    
    示例:
        >>> label_gen = ReversalStrengthLabel({
        ...     "periods": 3,
        ...     "lookback_window": 20,
        ...     "method": "classification",
        ...     "threshold": 0.02
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    display_name = "反转强度标签"
    init_params = [
        Field(name="periods", label="未来期数", type=FieldType.INT, default=3, description="未来几期"),
        Field(name="lookback_window", label="回看窗口", type=FieldType.INT, default=20, description="回看窗口（用于计算偏离度）"),
        Field(name="method", label="标签类型", type=FieldType.RADIO, default="classification",
              choices=[("classification", "分类"), ("regression", "回归")],
              description="标签类型"),
        Field(name="threshold", label="反转强度阈值", type=FieldType.FLOAT, default=0.02, description="反转强度阈值（用于分类），默认0.02（2%）"),
    ]
    
    def __init__(self, periods: int = 3, lookback_window: int = 20, method: str = "classification",
                 threshold: float = 0.02):
        super().__init__()
        self.periods = periods
        self.lookback_window = lookback_window
        self.method = method
        self.threshold = threshold
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成反转强度标签
        
        逻辑:
        1. 计算当前价格相对于均线的偏离度
        2. 如果偏离度大（超买/超卖），且未来价格反转，则标记为强反转
        
        :param df: 包含 close 列的 DataFrame
        :return: 标签 Series
        """
        close = df['close']
        
        # 1. 计算均线和偏离度
        ma = close.rolling(window=self.lookback_window).mean()
        deviation = (close - ma) / ma  # 偏离度
        
        # 2. 计算未来收益率
        future_close = close.shift(-self.periods)
        future_return = (future_close - close) / close
        
        # 3. 判断反转：超买后下跌，或超卖后上涨
        # 超买：deviation > threshold，未来应该下跌（future_return < 0）
        # 超卖：deviation < -threshold，未来应该上涨（future_return > 0）
        oversold_reversal = (deviation < -self.threshold) & (future_return > self.threshold)
        overbought_reversal = (deviation > self.threshold) & (future_return < -self.threshold)
        
        if self.method == "classification":
            # 分类：是否有强反转
            labels = np.zeros(len(df))
            labels[oversold_reversal | overbought_reversal] = 1
            label = labels.astype(int)
        else:
            # 回归：反转强度（未来收益率 * 偏离度的绝对值）
            label = future_return * abs(deviation)
        
        return pd.Series(label, name=self.label_name)

