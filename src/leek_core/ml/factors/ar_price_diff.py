#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AR价差因子 (Abnormal Return Price Difference Factor)

基于陈升锐研究报告的AR价差因子实现，适配加密货币市场。

核心公式：
    S² = -4E[(c_{t-1} - η_{t-1})(c_{t-1} - η_t)]
    AR = √max{-4E[(c_{t-1} - η_{t-1})(c_{t-1} - η_t)], 0}
    η_t = (h_t + l_t) / 2

其中：
    - c_t: t时刻收盘价的对数值 log(close)
    - η_t: t时刻中间价的对数值 log((high + low) / 2)
    - E[·]: 期望值（窗口内均值）

经济含义：
    - AR因子捕捉价格围绕中间价波动的异常程度
    - 较大的AR值表示价格波动剧烈，可能存在趋势或突破
    - 可用于波动率预测和风险管理

加密货币适配：
    - 24/7交易市场，无需考虑收盘时间
    - 支持不同K线周期（1m, 5m, 15m, 1h, 4h, 1d等）
    - 滚动窗口参数可根据交易频率调整
"""
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor, FeatureSpec


class ARPriceDiffFactor(DualModeFactor):
    """
    AR价差因子
    
    基于高频价格数据构建的波动率因子，通过计算收盘价与中间价之间的
    异常关系来捕捉市场波动特征。
    
    计算步骤：
    1. 计算对数收盘价: c_t = log(close_t)
    2. 计算对数中间价: η_t = log((high_t + low_t) / 2)
    3. 计算价差乘积: d_t = (c_{t-1} - η_{t-1}) * (c_{t-1} - η_t)
    4. 计算AR因子: AR = √max(-4 * mean(d), 0)
    
    输出因子：
    - AR: 全窗口期望值计算的AR因子
    - AR_{window}: 各滚动窗口的AR因子
    
    参数：
        base_window: 基础计算窗口，用于计算主AR因子
        rolling_windows: 滚动窗口列表，用于计算不同周期的AR因子
        use_log: 是否使用对数价格（推荐True，保证因子值与价格尺度无关）
    
    适用场景：
    - 波动率预测
    - 趋势强度判断
    - 风险管理信号
    """
    display_name = "AR价差"
    _name = "AR"
    
    init_params = [
        Field(
            name="base_window",
            label="基础窗口",
            type=FieldType.INT,
            default=240,
            required=True,
            description="主AR因子计算的基础窗口大小"
        ),
        Field(
            name="rolling_windows",
            label="滚动窗口",
            type=FieldType.STRING,
            default="5,15,30,60",
            required=False,
            description="滚动窗口列表，逗号分隔，用于计算不同周期的AR因子"
        ),
        Field(
            name="use_log",
            label="使用对数",
            type=FieldType.BOOLEAN,
            default=True,
            required=False,
            description="是否使用对数价格计算（推荐开启，使因子与价格尺度无关）"
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
        初始化AR价差因子
        
        Args:
            base_window: 基础窗口大小
            rolling_windows: 滚动窗口列表字符串（逗号分隔）
            use_log: 是否使用对数价格
            name: 自定义因子名称前缀
        """
        self.base_window = int(kwargs.get("base_window", 240))
        self.use_log = kwargs.get("use_log", True)
        
        # 解析滚动窗口参数
        windows_str = kwargs.get("rolling_windows", "5,15,30,60")
        self.rolling_windows = []
        if windows_str:
            try:
                for part in str(windows_str).split(","):
                    part = part.strip()
                    if part:
                        self.rolling_windows.append(int(part))
            except Exception:
                self.rolling_windows = [5, 15, 30, 60]
        
        # 设置缓冲区大小
        max_window = max([self.base_window] + self.rolling_windows)
        self._required_buffer_size = max_window + 50
        
        # 自定义因子名称
        name_prefix = kwargs.get("name", "")
        if name_prefix:
            self._name_prefix = name_prefix
        else:
            self._name_prefix = "AR"
        
        super().__init__()
        
        # 构建因子名称列表
        self.factor_names = self._build_factor_names()
    
    def _build_factor_names(self) -> List[str]:
        """构建因子名称列表"""
        names = [
            f"{self._name_prefix}",                      # 主AR因子（方法1）
            f"{self._name_prefix}_{self.base_window}",   # AR_{base_window}（方法2）
        ]
        
        # 添加滚动窗口因子名称
        for window in self.rolling_windows:
            names.append(f"{self._name_prefix}_{window}")
        
        return names
    
    def _compute_ar_component(self, log_close: np.ndarray, log_mid: np.ndarray) -> np.ndarray:
        """
        计算AR因子的核心组件 d_t = (c_{t-1} - η_{t-1}) * (c_{t-1} - η_t)
        
        Args:
            log_close: 对数收盘价序列
            log_mid: 对数中间价序列
        
        Returns:
            价差乘积序列 d_t
        """
        n = len(log_close)
        d = np.full(n, np.nan, dtype=np.float64)
        
        # d_t = (c_{t-1} - η_{t-1}) * (c_{t-1} - η_t)
        # 需要 t >= 2 才能计算
        for t in range(2, n):
            c_prev = log_close[t - 1]  # c_{t-1}
            mid_prev = log_mid[t - 1]  # η_{t-1}
            mid_curr = log_mid[t]      # η_t
            
            if np.isnan(c_prev) or np.isnan(mid_prev) or np.isnan(mid_curr):
                continue
            
            d[t] = (c_prev - mid_prev) * (c_prev - mid_curr)
        
        return d
    
    def _compute_ar_from_d(self, d: np.ndarray, start_idx: int, end_idx: int) -> float:
        """
        从价差乘积序列计算AR因子值
        AR = √max(-4 * mean(d), 0)
        
        Args:
            d: 价差乘积序列
            start_idx: 窗口起始索引
            end_idx: 窗口结束索引（不包含）
        
        Returns:
            AR因子值
        """
        window_d = d[start_idx:end_idx]
        valid_d = window_d[~np.isnan(window_d)]
        
        if len(valid_d) == 0:
            return np.nan
        
        mean_d = np.mean(valid_d)
        ar_squared = -4 * mean_d
        
        # AR = √max(ar_squared, 0)
        if ar_squared > 0:
            return np.sqrt(ar_squared)
        else:
            return 0.0
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算AR价差因子
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
        
        Returns:
            包含因子列的 DataFrame
        """
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        n = len(df)
        
        # 计算中间价
        mid = (high + low) / 2.0
        
        # 计算对数价格（如果启用）
        if self.use_log:
            # 防止对数计算中的无效值
            close_safe = np.where(close > 0, close, np.nan)
            mid_safe = np.where(mid > 0, mid, np.nan)
            
            log_close = np.log(close_safe)
            log_mid = np.log(mid_safe)
        else:
            log_close = close
            log_mid = mid
        
        # 计算价差乘积 d_t
        d = self._compute_ar_component(log_close, log_mid)
        
        # 初始化结果字典
        results = {}
        
        # ===== 计算主AR因子（方法1）: sqrt(mean(clip(-4*d, 0))) =====
        # 对每个位置，使用从开始到当前的所有数据计算AR
        ar_main = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.base_window, n):
            # 使用最近 base_window 个数据点
            window_d = d[i - self.base_window + 1:i + 1]
            valid_d = window_d[~np.isnan(window_d)]
            
            if len(valid_d) > 0:
                # factor1 = sqrt(mean(clip(-4*d, 0)))
                clipped = np.clip(-4 * valid_d, 0, None)
                ar_main[i] = np.sqrt(np.mean(clipped))
        
        results[f"{self._name_prefix}"] = ar_main
        
        # ===== 计算AR_{base_window}因子（方法2）: sqrt(clip(-4*mean(d), 0)) =====
        ar_base = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.base_window, n):
            ar_base[i] = self._compute_ar_from_d(d, i - self.base_window + 1, i + 1)
        
        results[f"{self._name_prefix}_{self.base_window}"] = ar_base
        
        # ===== 计算滚动窗口AR因子 =====
        for window in self.rolling_windows:
            ar_rolling = np.full(n, np.nan, dtype=np.float64)
            
            # 首先计算每个位置的滚动均值
            rolling_mean_d = np.full(n, np.nan, dtype=np.float64)
            for i in range(window - 1, n):
                window_d = d[i - window + 1:i + 1]
                valid_d = window_d[~np.isnan(window_d)]
                if len(valid_d) > 0:
                    rolling_mean_d[i] = np.mean(valid_d)
            
            # 然后在 base_window 上计算最终因子
            # factor = sqrt(mean(clip(-4*rolling_mean_d, 0)))
            for i in range(max(self.base_window, window), n):
                window_rolling = rolling_mean_d[i - self.base_window + 1:i + 1]
                valid_rolling = window_rolling[~np.isnan(window_rolling)]
                
                if len(valid_rolling) > 0:
                    clipped = np.clip(-4 * valid_rolling, 0, None)
                    ar_rolling[i] = np.sqrt(np.mean(clipped))
            
            results[f"{self._name_prefix}_{window}"] = ar_rolling
        
        return pd.DataFrame(results, index=df.index)
    
    def get_output_specs(self) -> List[FeatureSpec]:
        """返回因子元数据列表"""
        return [FeatureSpec(name=name) for name in self.factor_names]


class SimpleARFactor(DualModeFactor):
    """
    简化版AR价差因子
    
    只计算单一窗口的AR因子，适用于实时计算场景。
    
    公式：
        AR = √max(-4 * mean((c_{t-1} - η_{t-1}) * (c_{t-1} - η_t)), 0)
    
    参数：
        window: 滚动窗口大小
        use_log: 是否使用对数价格
    """
    display_name = "简化AR价差"
    _name = "SimpleAR"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=60,
            required=True,
            description="AR因子计算的滚动窗口大小"
        ),
        Field(
            name="use_log",
            label="使用对数",
            type=FieldType.BOOLEAN,
            default=True,
            required=False,
            description="是否使用对数价格计算"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 60))
        self.use_log = kwargs.get("use_log", True)
        
        self._required_buffer_size = self.window + 20
        
        name = kwargs.get("name", "")
        if name:
            self._factor_name = name
        else:
            self._factor_name = f"SimpleAR_{self.window}"
        
        super().__init__()
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算简化版AR因子"""
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        n = len(df)
        
        # 计算中间价
        mid = (high + low) / 2.0
        
        # 计算对数价格
        if self.use_log:
            close_safe = np.where(close > 0, close, np.nan)
            mid_safe = np.where(mid > 0, mid, np.nan)
            log_close = np.log(close_safe)
            log_mid = np.log(mid_safe)
        else:
            log_close = close
            log_mid = mid
        
        # 计算价差乘积 d_t = (c_{t-1} - η_{t-1}) * (c_{t-1} - η_t)
        d = np.full(n, np.nan, dtype=np.float64)
        for t in range(2, n):
            c_prev = log_close[t - 1]
            mid_prev = log_mid[t - 1]
            mid_curr = log_mid[t]
            
            if not (np.isnan(c_prev) or np.isnan(mid_prev) or np.isnan(mid_curr)):
                d[t] = (c_prev - mid_prev) * (c_prev - mid_curr)
        
        # 计算滚动AR因子
        ar_result = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(self.window, n):
            window_d = d[i - self.window + 1:i + 1]
            valid_d = window_d[~np.isnan(window_d)]
            
            if len(valid_d) > 0:
                mean_d = np.mean(valid_d)
                ar_squared = -4 * mean_d
                ar_result[i] = np.sqrt(max(ar_squared, 0))
        
        return pd.DataFrame({self._factor_name: ar_result}, index=df.index)
    
    def get_output_specs(self) -> List[FeatureSpec]:
        return [FeatureSpec(name=self._factor_name)]
