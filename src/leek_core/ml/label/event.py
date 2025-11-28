#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件驱动标签（高级）

适用于：高频交易策略、需要严格风控的策略

定义"好"的交易事件：同时满足收益和风险条件。快速反应，严格风控。
"""
import numpy as np
import pandas as pd

from .base import LabelGenerator

class EventLabel(LabelGenerator):
    """
    事件驱动标签（高级）
    
    定义"好"的交易事件：同时满足收益和风险条件。
    例如：未来N天内收益率 > X%，且期间最大回撤 < Y%
    
    这种标签更贴近实际交易场景，考虑了风险控制，适合高频交易策略。
    
    参数:
        hold_periods: 持仓周期（天数），默认3
        min_return: 最小收益率阈值（小数形式，如0.05表示5%），默认0.05
        max_drawdown: 最大回撤阈值（小数形式，如0.03表示3%），默认0.03
        use_high_low: 是否使用 high/low 计算回撤（更精确），默认True
        label_name: 标签列名，默认"label"
    
    标签含义:
        - 1: "好"的交易事件（满足收益和风险条件）
        - 0: "坏"的交易事件（不满足条件）
    
    适用策略:
        - 高频交易：快速反应，严格风控
    
    示例:
        >>> label_gen = EventLabel({
        ...     "hold_periods": 3,
        ...     "min_return": 0.05,      # 5%收益率
        ...     "max_drawdown": 0.03,    # 3%回撤
        ...     "use_high_low": True
        ... })
        >>> df = label_gen.generate(df_raw)
    """
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.hold_periods = int(params.get("hold_periods", 3))
        self.min_return = float(params.get("min_return", 0.05))
        self.max_drawdown = float(params.get("max_drawdown", 0.03))
        self.use_high_low = params.get("use_high_low", True)
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成事件标签
        
        条件:
        1. 未来收益率 > min_return
        2. 最大回撤 < max_drawdown
        
        :param df: 包含 close 列（可选 high/low 列）的 DataFrame
        :return: 增加了 label 列的 DataFrame
        """
        close = df['close']
        
        # 1. 计算未来 hold_periods 期的收益率
        future_close = close.shift(-self.hold_periods)
        future_return = (future_close - close) / close
        
        # 2. 计算期间最大回撤
        if self.use_high_low and 'high' in df.columns and 'low' in df.columns:
            buy_price = close
            future_low = df['low'].rolling(window=self.hold_periods, min_periods=1).min()
            future_low = future_low.shift(-self.hold_periods)
            max_dd = (future_low - buy_price) / buy_price
        else:
            future_low = close.rolling(window=self.hold_periods, min_periods=1).min()
            future_low = future_low.shift(-self.hold_periods)
            max_dd = (future_low - close) / close
        
        # 3. 创建标签：同时满足收益和风险条件
        labels = np.zeros(len(df))
        labels[
            (future_return > self.min_return) & 
            (max_dd > -self.max_drawdown)
        ] = 1
        
        df[self.label_name] = labels.astype(int)
        
        return df

