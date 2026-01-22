#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价格尖峰预测因子 (Price Spike Features Factor)

基于论文 "Predicting Price Movements in High-Frequency Financial Data with Spiking Neural Networks"
(Ezinwoke & Rhodes, 2025) 的特征工程方法实现。

核心思想：
    论文使用脉冲神经网络(SNN)预测高频价格尖峰，其数据预处理包括：
    1. VWAP聚合平滑价格
    2. 价格差分去除趋势
    3. 鲁棒归一化（IQR缩放 + 分位数截断 + 正负通道分离）
    4. 尖峰检测（超过中位数阈值的显著变动）

本因子将上述预处理步骤转化为可用于K线数据的因子特征，适用于：
    - 波动率预测
    - 突破信号检测
    - 风险事件识别
    - 作为ML模型的输入特征

输出因子：
    - price_diff: 价格差分（收益率）
    - price_diff_pos: 正通道（只保留正值）
    - price_diff_neg: 负通道（只保留负值的绝对值）
    - price_diff_pos_norm: 正通道鲁棒归一化后的值 [0,1]
    - price_diff_neg_norm: 负通道鲁棒归一化后的值 [0,1]
    - spike_strength: 尖峰强度（窗口内平均绝对收益率）
    - spike_threshold: 尖峰阈值（滚动窗口中位数收益率）
    - is_spike: 是否为尖峰（1/0）
"""
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor


class PriceSpikeFeaturesFactor(DualModeFactor):
    """
    价格尖峰预测因子
    
    基于SNN论文的特征工程方法，提取价格尖峰相关特征。
    
    特征工程流程：
    1. 价格差分计算：计算收益率 r_t = (P_t - P_{t-1}) / P_{t-1}
    2. 正负通道分离：正收益→正通道，负收益绝对值→负通道
    3. 鲁棒归一化：
       - IQR缩放：使用四分位距进行缩放，抵抗异常值
       - 分位数截断：将超出 [iqr_lower, iqr_upper] 分位数的值截断
       - Min-Max重缩放：各通道独立缩放到 [0,1]
    4. 尖峰检测：
       - 计算窗口内平均绝对收益率作为尖峰强度
       - 使用滚动中位数作为阈值
       - 强度超过阈值则判定为尖峰
    
    参数：
        window: 鲁棒归一化的计算窗口，默认60
        spike_window: 尖峰强度计算窗口，默认3（论文默认值）
        iqr_lower: IQR截断下分位数，默认0.1
        iqr_upper: IQR截断上分位数，默认0.9
        use_vwap: 是否使用VWAP计算差分，默认False使用close
    
    适用场景：
        - 高频/分钟级K线的波动检测
        - 突破信号识别
        - 风险事件预警
        - 作为SNN/ML模型的输入特征
    """
    display_name = "价格尖峰特征"
    _name = "PriceSpike"
    
    init_params = [
        Field(
            name="window",
            label="归一化窗口",
            type=FieldType.INT,
            default=60,
            required=True,
            description="鲁棒归一化的计算窗口大小"
        ),
        Field(
            name="spike_window",
            label="尖峰窗口",
            type=FieldType.INT,
            default=3,
            required=True,
            description="尖峰强度计算窗口（论文默认3）"
        ),
        Field(
            name="iqr_lower",
            label="IQR下分位数",
            type=FieldType.FLOAT,
            default=0.1,
            required=False,
            description="IQR截断下分位数，默认0.1"
        ),
        Field(
            name="iqr_upper",
            label="IQR上分位数",
            type=FieldType.FLOAT,
            default=0.9,
            required=False,
            description="IQR截断上分位数，默认0.9"
        ),
        Field(
            name="use_vwap",
            label="使用VWAP",
            type=FieldType.BOOLEAN,
            default=False,
            required=False,
            description="是否使用VWAP计算差分，默认使用close"
        ),
        Field(
            name="name",
            label="因子名称前缀",
            type=FieldType.STRING,
            default="",
            required=False,
            description="自定义因子名称前缀"
        )
    ]
    
    def __init__(self, **kwargs):
        """
        初始化价格尖峰因子
        
        Args:
            window: 鲁棒归一化窗口大小
            spike_window: 尖峰强度计算窗口
            iqr_lower: IQR截断下分位数
            iqr_upper: IQR截断上分位数
            use_vwap: 是否使用VWAP
            name: 自定义因子名称前缀
        """
        self.window = int(kwargs.get("window", 60))
        self.spike_window = int(kwargs.get("spike_window", 3))
        self.iqr_lower = float(kwargs.get("iqr_lower", 0.1))
        self.iqr_upper = float(kwargs.get("iqr_upper", 0.9))
        self.use_vwap = kwargs.get("use_vwap", False)
        
        # 设置缓冲区大小
        self._required_buffer_size = self.window + 50
        
        # 自定义因子名称
        name_prefix = kwargs.get("name", "")
        if name_prefix:
            self._name_prefix = name_prefix
        else:
            self._name_prefix = "PriceSpike"
        
        super().__init__()
        
        # 构建因子名称列表
        self.factor_names = self._build_factor_names()
    
    def _build_factor_names(self) -> List[str]:
        """构建因子名称列表"""
        prefix = self._name_prefix
        return [
            f"{prefix}_diff",           # 价格差分
            f"{prefix}_diff_pos",       # 正通道
            f"{prefix}_diff_neg",       # 负通道
            f"{prefix}_diff_pos_norm",  # 正通道归一化
            f"{prefix}_diff_neg_norm",  # 负通道归一化
            f"{prefix}_strength",       # 尖峰强度
            f"{prefix}_threshold",      # 尖峰阈值
            f"{prefix}_is_spike",       # 是否尖峰
        ]
    
    def _robust_normalize_rolling(
        self, 
        series: np.ndarray, 
        window: int,
        iqr_lower: float,
        iqr_upper: float
    ) -> np.ndarray:
        """
        滚动鲁棒归一化
        
        论文中的三步归一化法：
        1. IQR缩放：使用四分位距进行缩放
        2. 分位数截断：将超出分位数范围的值截断
        3. Min-Max重缩放：缩放到[0,1]
        
        Args:
            series: 输入序列
            window: 滚动窗口大小
            iqr_lower: 下分位数
            iqr_upper: 上分位数
        
        Returns:
            归一化后的序列 [0,1]
        """
        n = len(series)
        result = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(window - 1, n):
            # 获取窗口数据
            start_idx = i - window + 1
            window_data = series[start_idx:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            
            if len(valid_data) < 3:
                continue
            
            # 1. 计算IQR并缩放
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            
            if iqr < 1e-10:
                # IQR太小，使用简单的min-max
                iqr = np.std(valid_data) * 2 + 1e-10
            
            # 2. 计算截断边界
            lower_bound = np.percentile(valid_data, iqr_lower * 100)
            upper_bound = np.percentile(valid_data, iqr_upper * 100)
            
            # 当前值
            current_val = series[i]
            if np.isnan(current_val):
                continue
            
            # 3. 截断到边界
            clipped_val = np.clip(current_val, lower_bound, upper_bound)
            
            # 4. Min-Max重缩放到[0,1]
            if upper_bound > lower_bound:
                normalized = (clipped_val - lower_bound) / (upper_bound - lower_bound)
            else:
                normalized = 0.5  # 边界相等时返回中间值
            
            result[i] = normalized
        
        return result
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算价格尖峰因子
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
        
        Returns:
            包含因子列的 DataFrame
        """
        n = len(df)
        
        # 获取价格序列
        if self.use_vwap and 'amount' in df.columns and 'volume' in df.columns:
            # 计算VWAP
            volume = df['volume'].values.astype(np.float64)
            amount = df['amount'].values.astype(np.float64)
            # 防止除零
            price = np.where(volume > 0, amount / volume, df['close'].values.astype(np.float64))
        else:
            price = df['close'].values.astype(np.float64)
        
        # ===== 1. 计算价格差分（收益率）=====
        price_diff = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if price[i - 1] > 0 and not np.isnan(price[i]) and not np.isnan(price[i - 1]):
                price_diff[i] = (price[i] - price[i - 1]) / price[i - 1]
        
        # ===== 2. 正负通道分离 =====
        # 正通道：只保留正值，负值填0
        price_diff_pos = np.where(price_diff > 0, price_diff, 0.0)
        price_diff_pos = np.where(np.isnan(price_diff), np.nan, price_diff_pos)
        
        # 负通道：只保留负值的绝对值，正值填0
        price_diff_neg = np.where(price_diff < 0, np.abs(price_diff), 0.0)
        price_diff_neg = np.where(np.isnan(price_diff), np.nan, price_diff_neg)
        
        # ===== 3. 鲁棒归一化 =====
        price_diff_pos_norm = self._robust_normalize_rolling(
            price_diff_pos, self.window, self.iqr_lower, self.iqr_upper
        )
        price_diff_neg_norm = self._robust_normalize_rolling(
            price_diff_neg, self.window, self.iqr_lower, self.iqr_upper
        )
        
        # ===== 4. 尖峰强度计算 =====
        # 计算绝对收益率
        abs_return = np.abs(price_diff)
        
        # 滚动窗口平均绝对收益率作为尖峰强度
        spike_strength = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.spike_window - 1, n):
            window_data = abs_return[i - self.spike_window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                spike_strength[i] = np.mean(valid_data)
        
        # ===== 5. 尖峰阈值（滚动中位数）=====
        spike_threshold = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.window - 1, n):
            window_data = abs_return[i - self.window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                spike_threshold[i] = np.median(valid_data)
        
        # ===== 6. 尖峰判定 =====
        # 当尖峰强度超过阈值时，判定为尖峰
        is_spike = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if not np.isnan(spike_strength[i]) and not np.isnan(spike_threshold[i]):
                is_spike[i] = 1.0 if spike_strength[i] > spike_threshold[i] else 0.0
        
        # 构建结果DataFrame
        prefix = self._name_prefix
        results = {
            f"{prefix}_diff": price_diff,
            f"{prefix}_diff_pos": price_diff_pos,
            f"{prefix}_diff_neg": price_diff_neg,
            f"{prefix}_diff_pos_norm": price_diff_pos_norm,
            f"{prefix}_diff_neg_norm": price_diff_neg_norm,
            f"{prefix}_strength": spike_strength,
            f"{prefix}_threshold": spike_threshold,
            f"{prefix}_is_spike": is_spike,
        }
        
        return pd.DataFrame(results, index=df.index)
    
    def get_output_names(self) -> List[str]:
        """返回因子列名列表"""
        return self.factor_names


class SimplePriceSpikeDetector(DualModeFactor):
    """
    简化版价格尖峰检测因子
    
    只输出核心的尖峰检测信号，适用于实时计算场景。
    
    输出因子：
    - spike_strength: 尖峰强度
    - is_spike: 是否为尖峰
    
    参数：
        window: 阈值计算窗口，默认60
        spike_window: 尖峰强度计算窗口，默认3
    """
    display_name = "简化尖峰检测"
    _name = "SimpleSpikeDetector"
    
    init_params = [
        Field(
            name="window",
            label="阈值窗口",
            type=FieldType.INT,
            default=60,
            required=True,
            description="尖峰阈值计算窗口大小"
        ),
        Field(
            name="spike_window",
            label="强度窗口",
            type=FieldType.INT,
            default=3,
            required=True,
            description="尖峰强度计算窗口"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 60))
        self.spike_window = int(kwargs.get("spike_window", 3))
        
        self._required_buffer_size = self.window + 20
        
        name = kwargs.get("name", "")
        if name:
            self._factor_prefix = name
        else:
            self._factor_prefix = "SimpleSpike"
        
        super().__init__()
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算简化版尖峰检测因子"""
        price = df['close'].values.astype(np.float64)
        n = len(df)
        
        # 计算收益率
        returns = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if price[i - 1] > 0:
                returns[i] = (price[i] - price[i - 1]) / price[i - 1]
        
        abs_returns = np.abs(returns)
        
        # 尖峰强度
        spike_strength = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.spike_window - 1, n):
            window_data = abs_returns[i - self.spike_window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                spike_strength[i] = np.mean(valid_data)
        
        # 尖峰阈值
        spike_threshold = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.window - 1, n):
            window_data = abs_returns[i - self.window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                spike_threshold[i] = np.median(valid_data)
        
        # 尖峰判定
        is_spike = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if not np.isnan(spike_strength[i]) and not np.isnan(spike_threshold[i]):
                is_spike[i] = 1.0 if spike_strength[i] > spike_threshold[i] else 0.0
        
        return pd.DataFrame({
            f"{self._factor_prefix}_strength": spike_strength,
            f"{self._factor_prefix}_is_spike": is_spike
        }, index=df.index)
    
    def get_output_names(self) -> List[str]:
        return [
            f"{self._factor_prefix}_strength",
            f"{self._factor_prefix}_is_spike"
        ]
