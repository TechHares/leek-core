#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险调整收益率标签（回归任务）

适用于：趋势跟踪策略

捕捉趋势同时控制风险，使用夏普比率、信息比率等风险调整指标。
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator

class RiskAdjustedReturnLabel(LabelGenerator):
    """
    风险调整收益率标签（回归任务）
    
    计算风险调整后的收益率，用于趋势跟踪策略。
    捕捉趋势同时控制风险，常用的风险调整指标包括：
    - 夏普比率 (Sharpe Ratio)
    - 信息比率 (Information Ratio)
    - 卡玛比率 (Calmar Ratio)
    
    参数:
        periods: 未来几期，默认5
        risk_free_rate: 无风险利率（年化），默认0.0
        method: 风险调整方法，"sharpe"=夏普比率，"calmar"=卡玛比率，"info"=信息比率，默认"sharpe"
        volatility_window: 波动率计算窗口，默认20
        label_name: 标签列名，默认"label"
    
    适用策略:
        - 趋势跟踪：捕捉趋势同时控制风险
    
    示例:
        >>> label_gen = RiskAdjustedReturnLabel({
        ...     "periods": 5,
        ...     "method": "sharpe",
        ...     "volatility_window": 20
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    display_name = "风险调整收益率标签"
    init_params = [
        Field(name="periods", label="未来期数", type=FieldType.INT, default=5, description="未来几期"),
        Field(name="risk_free_rate", label="无风险利率", type=FieldType.FLOAT, default=0.0, description="无风险利率（年化）"),
        Field(name="method", label="风险调整方法", type=FieldType.RADIO, default="sharpe",
              choices=[("sharpe", "夏普比率"), ("calmar", "卡玛比率"), ("info", "信息比率")],
              description="风险调整方法"),
        Field(name="volatility_window", label="波动率计算窗口", type=FieldType.INT, default=20, description="波动率计算窗口"),
    ]
    
    def __init__(self, periods: int = 5, risk_free_rate: float = 0.0, method: str = "sharpe",
                 volatility_window: int = 20):
        super().__init__()
        self.periods = periods
        self.risk_free_rate = risk_free_rate
        self.method = method
        self.volatility_window = volatility_window
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成风险调整收益率标签
        
        :param df: 包含 close 列的 DataFrame
        :return: 标签 Series
        """
        close = df['close']
        
        # 计算未来收益率
        future_close = close.shift(-self.periods)
        future_return = (future_close - close) / close
        
        # 计算波动率（使用历史收益率）
        returns = close.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        if self.method == "sharpe":
            # 夏普比率: (收益率 - 无风险利率) / 波动率
            # 简化版：直接用收益率/波动率
            label = future_return / (volatility + 1e-8)  # 避免除零
        elif self.method == "calmar":
            # 卡玛比率: 收益率 / 最大回撤
            # 需要计算最大回撤
            rolling_max = close.rolling(window=self.periods).max()
            drawdown = (close - rolling_max) / rolling_max
            max_dd = drawdown.rolling(window=self.periods).min()
            label = future_return / (abs(max_dd) + 1e-8)
        elif self.method == "info":
            # 信息比率: 超额收益 / 跟踪误差
            # 简化版：收益率 / 波动率（类似夏普）
            label = future_return / (volatility + 1e-8)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return pd.Series(label, name=self.label_name)

