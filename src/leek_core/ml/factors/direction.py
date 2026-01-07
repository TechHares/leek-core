#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor

class DirectionFactor(DualModeFactor):
    """
    近{window}个周期内涨跌K线之比
    
    计算公式：
    1. 在近 window 个周期内，识别上涨K线和下跌K线：
       - 上涨K线：close > open
       - 下跌K线：close < open
    
    2. 根据 compute_type 参数选择计算方式：
       - ratio: 涨跌幅之和之比
         上涨涨跌幅之和 / 下跌涨跌幅绝对值之和
       - count: 涨跌幅数量比
         上涨K线数量 / 下跌K线数量
       - average: 涨跌幅平均值之比
         上涨K线涨跌幅平均值 / 下跌K线涨跌幅绝对值平均值
    
    示例：
    - window=20, compute_type="ratio"
      表示：近20个周期内，上涨K线的涨跌幅之和 / 下跌K线的涨跌幅绝对值之和
    - window=20, compute_type="count"
      表示：近20个周期内，上涨K线数量 / 下跌K线数量
    - window=20, compute_type="average"
      表示：近20个周期内，上涨K线涨跌幅平均值 / 下跌K线涨跌幅绝对值平均值
    """
    display_name = "涨跌K线之比"
    _name = "DirectionFactor"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=20,
            required=True,
            description="统计窗口大小"
        ),
        Field(
            name="compute_type",
            label="计算方式",
            type=FieldType.RADIO,
            default="ratio",
            choices=[("ratio", "幅度"), ("count", "数量"), ("average", "平均幅度")],
            required=True,
            description="计算方式，涨跌幅之和之比，涨跌幅数量比，涨跌幅平均值"
        )
    ]
    
    def __init__(self, window=20, compute_type="ratio"):
        self.window = window
        self.compute_type = compute_type
        self._required_buffer_size = self.window + 10
        super().__init__()
        # 动态生成因子名称
        self._factor_name = f"Direction_{self.window}_{self.compute_type}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算涨跌K线之比因子
        
        对于每个时间点，计算近 window 个周期内：
        1. 识别上涨K线和下跌K线
        2. 根据 compute_type 参数计算比率：
           - ratio: 上涨涨跌幅之和 / 下跌涨跌幅绝对值之和
           - count: 上涨K线数量 / 下跌K线数量
           - average: 上涨K线涨跌幅平均值 / 下跌K线涨跌幅绝对值平均值
        """
        # 计算涨跌方向和涨跌幅
        close = df['close'].values
        open_price = df['open'].values
        n = len(df)
        
        # 计算涨跌幅（使用收盘价的变化率）
        pct_change = df['close'].pct_change().values
        
        # 计算涨跌方向（1=上涨，-1=下跌，0=平盘）
        direction = np.zeros(n, dtype=np.int32)
        direction[close > open_price] = 1  # 上涨
        direction[close < open_price] = -1  # 下跌
        
        # 初始化结果数组
        result = np.full(n, np.nan, dtype=np.float64)
        
        # 使用滑动窗口计算
        for i in range(self.window - 1, n):
            # 获取窗口内的数据
            start_idx = i - self.window + 1
            end_idx = i + 1
            
            pct_window = pct_change[start_idx:end_idx]
            dir_window = direction[start_idx:end_idx]
            
            # 过滤 NaN 值
            valid_mask = ~np.isnan(pct_window)
            pct_valid = pct_window[valid_mask]
            dir_valid = dir_window[valid_mask]
            
            # 需要至少2个有效数据点
            if len(pct_valid) < 2:
                continue
            
            # 分离上涨和下跌K线
            long_mask = dir_valid == 1
            short_mask = dir_valid == -1
            
            long_pct = pct_valid[long_mask]
            short_pct = pct_valid[short_mask]
            
            if self.compute_type == "ratio":
                # 使用涨跌幅之和之比
                long_sum = np.sum(long_pct) if len(long_pct) > 0 else 0.0
                short_sum = np.sum(np.abs(short_pct)) if len(short_pct) > 0 else 0.0
                
                # 避免除零
                if short_sum > 0:
                    result[i] = long_sum / short_sum
                elif long_sum > 0:
                    # 如果只有上涨没有下跌，返回一个较大的值
                    result[i] = long_sum / 1e-10
            elif self.compute_type == "count":
                # 使用涨跌幅数量比
                long_count = len(long_pct)
                short_count = len(short_pct)
                
                # 避免除零
                if short_count > 0:
                    result[i] = long_count / short_count
                elif long_count > 0:
                    # 如果只有上涨没有下跌，返回一个较大的值
                    result[i] = long_count / 1e-10
            elif self.compute_type == "average":
                # 使用涨跌幅平均值之比
                long_avg = np.mean(long_pct) if len(long_pct) > 0 else 0.0
                short_avg = np.mean(np.abs(short_pct)) if len(short_pct) > 0 else 0.0
                
                # 避免除零
                if short_avg > 0:
                    result[i] = long_avg / short_avg
                elif long_avg > 0:
                    # 如果只有上涨没有下跌，返回一个较大的值
                    result[i] = long_avg / 1e-10
        
        # 将结果赋值给 DataFrame
        return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    def get_output_names(self) -> list:
        return [self._factor_name]
