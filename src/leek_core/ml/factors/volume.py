#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal

import numpy as np
import pandas as pd

from leek_core.indicators import ATR, MA, RSI
from leek_core.models import ChoiceType, Field, FieldType

from .base import DualModeFactor, FeatureSpec

class LongShortVolumeRatioFactor(DualModeFactor):
    """
    近{window}个周期内涨跌{top}成交量比率
    
    计算公式：
    LongVolume = sum(近{window}个周期内涨跌{top}的成交量)
    ShortVolume = sum(近{window}个周期内涨跌{bottom}的成交量)
    LongShortVolumeRatio = LongVolume / (LongVolume + ShortVolume)
    """
    display_name = "涨跌成交量比率"
    _name = "LongShortVolumeRatioFactor"
    
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
            name="top",
            label="数量",
            type=FieldType.INT,
            default=20,
            required=True,
            description="涨跌幅最大的周期数量"
        ),
    ]
    
    def __init__(self, window=20, top=20):
        self.window = window
        self._required_buffer_size = self.window + 10
        super().__init__()
        self.top = top
        # 动态生成因子名称
        self._factor_name = f"LongShortVolumeRatio_{self.window}_{self.top}"


    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算长短期成交量比率因子（优化版：部分向量化 + 滑动窗口优化）
        
        对于每个时间点，计算近 window 个周期内：
        - LongVolume: 涨跌幅最大的 top 个周期的成交量之和
        - ShortVolume: 涨跌幅最小的 top 个周期的成交量之和
        - LongShortVolumeRatio: LongVolume / (LongVolume + ShortVolume)
        """
        # 计算涨跌幅（使用收盘价的变化率）
        pct_change = df['close'].pct_change().values
        volume = df['volume'].values
        n = len(df)
        
        # 确保 top 不超过 window
        top_count = min(self.top, self.window)
        
        # 初始化结果数组
        result = np.full(n, np.nan, dtype=np.float64)
        
        # 使用滑动窗口优化：直接从数组切片，避免 DataFrame 操作
        # 对于每个有效窗口位置
        for i in range(self.window - 1, n):
            # 获取窗口内的数据（直接数组切片，零拷贝）
            start_idx = i - self.window + 1
            end_idx = i + 1
            
            pct_window = pct_change[start_idx:end_idx]
            vol_window = volume[start_idx:end_idx]
            
            # 过滤 NaN 值（向量化操作）
            valid_mask = ~(np.isnan(pct_window) | np.isnan(vol_window))
            valid_count = np.sum(valid_mask)
            
            # 需要至少 top_count * 2 个有效数据点（top 个最大 + top 个最小）
            if valid_count < max(2, top_count * 2):
                continue
            
            # 提取有效数据
            pct_valid = pct_window[valid_mask]
            vol_valid = vol_window[valid_mask]
            
            # 计算实际要取的个数（不能超过有效数据个数的一半）
            actual_top = min(top_count, valid_count // 2)
            
            if actual_top == 0:
                continue
            
            # 使用 argsort 获取排序索引（优化：只排序需要的部分）
            # 获取涨跌幅最大的 actual_top 个索引
            top_indices = np.argsort(pct_valid)[-actual_top:]
            # 获取涨跌幅最小的 actual_top 个索引
            bottom_indices = np.argsort(pct_valid)[:actual_top]
            
            # 计算 LongVolume（涨跌幅最大的 top 个周期的成交量之和）
            long_volume = np.sum(vol_valid[top_indices])
            
            # 计算 ShortVolume（涨跌幅最小的 top 个周期的成交量之和）
            short_volume = np.sum(vol_valid[bottom_indices])
            
            # 计算比率
            total_volume = long_volume + short_volume
            if total_volume > 0:
                result[i] = long_volume / total_volume
        
        # 将结果赋值给 DataFrame
        return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    def get_output_specs(self) -> list:
        return [FeatureSpec(name=self._factor_name)]


class VolumeAverageFactor(DualModeFactor):
    """
    近{window}个周期内{side}方向{type}的成交量平均值 / 当前成交量
    
    计算公式：
    1. 在近 window 个周期内，根据 side 参数过滤K线：
       - FLAT: 统计所有K线
       - LONG: 只统计上涨K线（close > open）
       - SHORT: 只统计下跌K线（close < open）
    
    2. 在过滤后的K线中，根据 type 参数选择成交量：
       - min_volume: 取成交量最小的 top 个周期
       - max_volume: 取成交量最大的 top 个周期
    
    3. 计算这 top 个周期的成交量平均值
    
    4. 最终因子值 = 平均值 / 当前成交量
    
    示例：
    - window=20, side="LONG", type="max_volume", top=5
      表示：近20个周期内，上涨K线中成交量最大的5个周期的平均成交量 / 当前成交量
    """
    display_name = "成交量平均值"
    _name = "VolumeAverageFactor"
    
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
            name="side",
            label="方向",
            type=FieldType.RADIO,
            default="FLAT",
            choices=[("FLAT", "全部"), ("LONG", "上涨K线"), ("SHORT", "下跌K线")],
            choice_type=ChoiceType.STRING,
            required=True,
            description="统计方向，涨跌都统计、只统计上涨K线、只统计下跌K线"
        ),
        Field(
            name="type",
            label="类型",
            type=FieldType.RADIO,
            default="min_volume",
            choices=[("min_volume", "最小成交量"), ("max_volume", "最大成交量")],
            choice_type=ChoiceType.STRING,
            required=True,
            description="统计类型，最小成交量、最大成交量"
        ),
        Field(
            name="top",
            label="数量",
            type=FieldType.INT,
            default=20,
            required=True,
            description="取最小或最大成交量的周期数量"
        )
    ]
    
    def __init__(self, window=20, side="FLAT", type="min_volume", top=20, name=""):
        self.window = window
        self.side = side
        self.type = type
        self.top = top
        self._required_buffer_size = self.window + 10
        super().__init__()
        # 动态生成因子名称
        if name:
            self._factor_name = name
        else:
            self._factor_name = f"VolumeAverage_{self.window}_{self.side}_{self.type}_{self.top}"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量平均值因子
        
        对于每个时间点，计算近 window 个周期内：
        1. 根据 side 过滤K线（全部/上涨/下跌）
        2. 根据 type 取最小或最大成交量的 top 个周期
        3. 计算这 top 个周期的成交量平均值
        4. 平均值 / 当前成交量
        """
        # 计算涨跌方向（用于过滤）
        close = df['close'].values
        open_price = df['open'].values
        volume = df['volume'].values
        n = len(df)
        
        # 计算涨跌方向（1=上涨，-1=下跌，0=平盘）
        direction = np.zeros(n, dtype=np.int32)
        direction[close > open_price] = 1  # 上涨
        direction[close < open_price] = -1  # 下跌
        
        # 初始化结果数组
        result = np.full(n, np.nan, dtype=np.float64)
        
        # 确保 top 不超过 window
        top_count = min(self.top, self.window)
        
        # 使用滑动窗口计算
        for i in range(self.window - 1, n):
            # 获取窗口内的数据
            start_idx = i - self.window + 1
            end_idx = i + 1
            
            vol_window = volume[start_idx:end_idx]
            dir_window = direction[start_idx:end_idx]
            
            # 根据 side 参数过滤
            if self.side == "LONG":
                # 只统计上涨K线
                mask = dir_window == 1
            elif self.side == "SHORT":
                # 只统计下跌K线
                mask = dir_window == -1
            else:  # FLAT
                # 统计所有K线
                mask = np.ones(len(vol_window), dtype=bool)
            
            # 过滤 NaN 值
            valid_mask = ~np.isnan(vol_window) & mask
            valid_volumes = vol_window[valid_mask]
            
            # 需要至少 top_count 个有效数据点
            if len(valid_volumes) < top_count:
                continue
            
            # 根据 type 参数选择成交量
            if self.type == "min_volume":
                # 取成交量最小的 top_count 个
                selected_volumes = np.partition(valid_volumes, top_count - 1)[:top_count]
            else:  # max_volume
                # 取成交量最大的 top_count 个
                selected_volumes = np.partition(valid_volumes, -top_count)[-top_count:]
            
            # 计算平均值
            avg_volume = np.mean(selected_volumes)
            
            # 当前成交量
            current_volume = volume[i]
            
            # 避免除零
            if current_volume > 0 and avg_volume > 0:
                result[i] = avg_volume / current_volume
        
        # 将结果赋值给 DataFrame
        return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    def get_output_specs(self) -> list:
        return [FeatureSpec(name=self._factor_name)]