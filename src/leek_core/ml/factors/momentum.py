#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动量类因子

包括：
- AccelerationMomentumFactor: 加速度动量因子
  基于论文《Investor Attention, Visual Price Pattern, and Momentum Investing》
"""
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor


def compute_r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    计算R²（决定系数）
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        R²值
    """
    # 计算残差平方和
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    # 计算总平方和
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    # 避免除零
    ss_tot = np.where(ss_tot == 0, 1e-10, ss_tot)
    # 计算R²
    r2 = 1 - (ss_res / ss_tot)
    return r2


class AccelerationMomentumFactor(DualModeFactor):
    """
    加速度动量因子
    
    基于论文《Investor Attention, Visual Price Pattern, and Momentum Investing》
    (Chen & Lin, 2013)
    
    计算过程：
    1. 使用窗口期内的收盘价数据
    2. 构建回归模型: Price = alpha + beta * t + gamma * t² + epsilon
       - t: 时间序列 (1, 2, 3, ..., window)
       - t²: 时间序列的平方
    3. 通过最小二乘法估计参数
    
    输出因子：
    - alpha: 回归截距
    - beta: 价格涨跌强弱（一次项系数）
    - gamma: 加速度动量（二次项系数）
    - r2: 拟合优度（R²）
    
    经济含义：
    - gamma > 0: 价格加速上涨
    - gamma < 0: 价格加速下跌
    - |gamma| 越大，加速度越明显
    - 高gamma且呈上涨趋势的股票往往能吸引更多投资者注意力
    
    参数：
        window: 回归窗口大小，默认21（约一个月的交易日）
        min_periods: 最小数据点数量，默认10（至少需要10个数据点才能回归）
        name: 自定义因子名称前缀
    """
    display_name = "加速度动量"
    _name = "AccelerationMomentum"
    
    init_params = [
        Field(
            name="window",
            label="回归窗口",
            type=FieldType.INT,
            default=21,
            required=True,
            description="回归使用的历史窗口大小（交易日数量）"
        ),
        Field(
            name="min_periods",
            label="最小数据点",
            type=FieldType.INT,
            default=10,
            required=False,
            description="进行回归所需的最小数据点数量"
        )
    ]
    
    def __init__(self, **kwargs):
        """
        初始化加速度动量因子
        
        Args:
            window: 回归窗口大小
            min_periods: 最小数据点数量
            name: 自定义因子名称前缀
        """
        self.window = int(kwargs.get("window", 21))
        self.min_periods = int(kwargs.get("min_periods", 10))
        self._required_buffer_size = self.window + 10
        
        # 自定义因子名称
        name_prefix = kwargs.get("name", "")
        if name_prefix:
            self._name_prefix = name_prefix
        else:
            self._name_prefix = f"AccMom_{self.window}"
        
        super().__init__()
        
        # 预构建因子名称列表
        self.factor_names = [
            f"{self._name_prefix}_alpha",
            f"{self._name_prefix}_beta",
            f"{self._name_prefix}_gamma",
            f"{self._name_prefix}_r2"
        ]
    
    def _fit_quadratic_regression(self, prices: np.ndarray) -> tuple:
        """
        拟合二次回归模型: price = alpha + beta * t + gamma * t²
        
        Args:
            prices: 价格序列（一维数组）
        
        Returns:
            (alpha, beta, gamma, r2): 回归系数和拟合优度
        """
        n = len(prices)
        
        # 过滤NaN值
        valid_mask = ~np.isnan(prices)
        valid_prices = prices[valid_mask]
        
        # 数据点不足，返回NaN
        if len(valid_prices) < self.min_periods:
            return np.nan, np.nan, np.nan, np.nan
        
        # 构建时间序列（从有效数据的索引开始）
        valid_indices = np.where(valid_mask)[0]
        t = valid_indices + 1  # 时间从1开始
        t_square = t ** 2
        
        # 构建设计矩阵 X = [1, t, t²]
        X = np.column_stack([np.ones(len(valid_prices)), t, t_square])
        Y = valid_prices.reshape(-1, 1)
        
        try:
            # 最小二乘法求解: beta = (X'X)^(-1) X'Y
            XtX = X.T @ X
            XtY = X.T @ Y
            beta = np.linalg.pinv(XtX) @ XtY
            
            # 提取系数
            alpha = beta[0, 0]
            beta_coef = beta[1, 0]
            gamma = beta[2, 0]
            
            # 计算R²
            y_pred = X @ beta
            r2 = compute_r_squared(y_pred, Y)[0]
            
            return alpha, beta_coef, gamma, r2
        except Exception:
            return np.nan, np.nan, np.nan, np.nan
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算加速度动量因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
        
        Returns:
            包含四个因子列的DataFrame
        """
        close = df['close'].values
        n = len(df)
        
        # 初始化结果数组
        alpha_arr = np.full(n, np.nan)
        beta_arr = np.full(n, np.nan)
        gamma_arr = np.full(n, np.nan)
        r2_arr = np.full(n, np.nan)
        
        # 滚动窗口计算
        for i in range(self.window - 1, n):
            start_idx = i - self.window + 1
            end_idx = i + 1
            
            # 提取窗口内的价格
            window_prices = close[start_idx:end_idx]
            
            # 拟合二次回归
            alpha, beta, gamma, r2 = self._fit_quadratic_regression(window_prices)
            
            # 存储结果
            alpha_arr[i] = alpha
            beta_arr[i] = beta
            gamma_arr[i] = gamma
            r2_arr[i] = r2
        
        # 返回包含因子列的DataFrame
        result = pd.DataFrame({
            self.factor_names[0]: alpha_arr,
            self.factor_names[1]: beta_arr,
            self.factor_names[2]: gamma_arr,
            self.factor_names[3]: r2_arr
        }, index=df.index)
        
        return result
    
    def get_output_names(self) -> List[str]:
        """返回因子列名列表"""
        return self.factor_names
