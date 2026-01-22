#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强方向动量（D-MOM）因子

基于论文《Directional Information in Equity Returns》(Luca Del Viva, Carlo Sala, Andre B.M. Souza, 2023)

核心思路：
通过线性概率模型(LPM)预测收益率的"方向"而非"数值"，从而有效改进传统动量因子。

加密货币市场适配：
1. 特质波动率：使用收益率滚动标准差替代CAPM残差方差
2. 持续期计算：按K线周期计算正/负收益持续期
3. 24/7交易特性：调整窗口期适应不间断交易
"""
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor


class DirectionalMomentumFactor(DualModeFactor):
    """
    增强方向动量（D-MOM）因子
    
    通过线性概率模型(LPM)预测收益率方向，输出预测概率作为因子值。
    
    模型公式：
    D_{i,t} = α + β₁*σ²_{t-1} + β₂*Dur⁺_{t-1} + β₃*Dur⁻_{t-1} + β₄*R_{t-1} + ε
    
    其中：
    - D_{i,t}: 收益率方向哑变量（>0为1，否则为0）
    - σ²_{t-1}: 滞后特质波动率（收益率滚动标准差）
    - Dur⁺/⁻_{t-1}: 正/负收益持续期（连续正/负收益的周期数）
    - R_{t-1}: 滞后收益率
    
    输出：预测概率（0-1之间的值），即方向动量因子
    
    经济含义：
    - 因子值越高，预期未来收益为正的概率越大
    - 相比传统动量因子，能有效抵御"动量崩溃"风险
    - 收益分布呈右偏特征，降低左尾风险
    
    加密货币适配：
    - 无涨跌停限制，波动率更高
    - 24/7交易，持续期计算按K线周期
    - 高频交易环境下更适合短周期窗口
    """
    display_name = "方向动量"
    _name = "DMOM"
    
    init_params = [
        Field(
            name="train_window",
            label="训练窗口",
            type=FieldType.INT,
            default=120,
            required=True,
            description="线性概率模型训练所需的历史数据窗口大小"
        ),
        Field(
            name="vol_window",
            label="波动率窗口",
            type=FieldType.INT,
            default=20,
            required=True,
            description="计算特质波动率的滚动窗口大小"
        ),
        Field(
            name="max_duration",
            label="最大持续期",
            type=FieldType.INT,
            default=12,
            required=True,
            description="正/负收益持续期的最大值（超过此值则截断）"
        ),
        Field(
            name="expanding_window",
            label="扩展窗口",
            type=FieldType.BOOLEAN,
            default=True,
            required=False,
            description="是否使用扩展窗口（True）或滚动窗口（False）训练模型"
        )
    ]
    
    def __init__(self, **kwargs):
        """
        初始化增强方向动量因子
        
        Args:
            train_window: 训练窗口大小
            vol_window: 波动率计算窗口
            max_duration: 最大持续期
            expanding_window: 是否使用扩展窗口
            name: 自定义因子名称前缀
        """
        self.train_window = int(kwargs.get("train_window", 120))
        self.vol_window = int(kwargs.get("vol_window", 20))
        self.max_duration = int(kwargs.get("max_duration", 12))
        self.expanding_window = kwargs.get("expanding_window", True)
        
        # 设置缓冲区大小
        self._required_buffer_size = max(self.train_window + 50, 200)
        
        # 自定义因子名称
        name_prefix = kwargs.get("name", "")
        if name_prefix:
            self._name_prefix = name_prefix
        else:
            self._name_prefix = f"DMOM_{self.train_window}_{self.vol_window}"
        
        super().__init__()
        
        # 因子名称列表
        self.factor_names = [
            f"{self._name_prefix}_prob",      # 预测概率（主因子）
            f"{self._name_prefix}_dur_pos",   # 正收益持续期
            f"{self._name_prefix}_dur_neg",   # 负收益持续期
        ]
    
    def _compute_duration(self, returns: np.ndarray) -> tuple:
        """
        计算正/负收益持续期
        
        Args:
            returns: 收益率序列
        
        Returns:
            (dur_pos_arr, dur_neg_arr): 正收益持续期数组, 负收益持续期数组
        """
        n = len(returns)
        dur_pos = np.zeros(n, dtype=np.float64)
        dur_neg = np.zeros(n, dtype=np.float64)
        
        pos_count = 0
        neg_count = 0
        
        for i in range(n):
            if np.isnan(returns[i]):
                dur_pos[i] = np.nan
                dur_neg[i] = np.nan
                continue
            
            if returns[i] > 0:
                pos_count += 1
                neg_count = 0
            elif returns[i] < 0:
                neg_count += 1
                pos_count = 0
            else:
                # 收益率为0，持续期不变
                pass
            
            # 截断到最大持续期
            dur_pos[i] = min(pos_count, self.max_duration)
            dur_neg[i] = min(neg_count, self.max_duration)
        
        return dur_pos, dur_neg
    
    def _fit_lpm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        拟合线性概率模型
        
        Args:
            X: 自变量矩阵 (n_samples, n_features)
            y: 因变量（方向哑变量）
        
        Returns:
            回归系数 (n_features + 1,)，包括截距
        """
        # 过滤NaN值
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 10:  # 数据不足
            return None
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X_valid)), X_valid])
        
        try:
            # 最小二乘法求解: β = (X'X)^(-1) X'y
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y_valid
            coeffs = np.linalg.pinv(XtX) @ Xty
            return coeffs
        except Exception:
            return None
    
    def _predict_probability(self, X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        使用线性概率模型预测概率
        
        Args:
            X: 自变量矩阵
            coeffs: 回归系数
        
        Returns:
            预测概率数组
        """
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # 计算预测值
        prob = X_with_intercept @ coeffs
        
        # 将概率截断到 [0, 1] 范围（线性概率模型可能超出此范围）
        prob = np.clip(prob, 0, 1)
        
        return prob
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算增强方向动量因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
        
        Returns:
            包含因子列的DataFrame
        """
        # 计算收益率
        close = df['close'].values.astype(np.float64)
        returns = np.zeros(len(close), dtype=np.float64)
        returns[0] = np.nan
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-20)
        
        n = len(df)
        
        # 计算滚动波动率（特质波动率的代理）
        volatility = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.vol_window - 1, n):
            window_returns = returns[i - self.vol_window + 1:i + 1]
            valid_returns = window_returns[~np.isnan(window_returns)]
            if len(valid_returns) >= 5:
                volatility[i] = np.std(valid_returns)
        
        # 计算持续期
        dur_pos, dur_neg = self._compute_duration(returns)
        
        # 构建方向哑变量（因变量）
        direction = np.where(returns > 0, 1.0, 0.0)
        direction[np.isnan(returns)] = np.nan
        
        # 初始化结果数组
        prob_result = np.full(n, np.nan, dtype=np.float64)
        
        # 滚动/扩展窗口训练模型并预测
        min_train_size = max(30, self.vol_window + 10)
        
        for i in range(self.train_window, n):
            # 确定训练窗口
            if self.expanding_window:
                train_start = 0
            else:
                train_start = i - self.train_window
            train_end = i
            
            # 准备训练数据（使用滞后值）
            # 自变量：[波动率, 正持续期, 负持续期, 滞后收益率]
            # 注意：自变量使用 t-1 时刻的值，因变量使用 t 时刻的方向
            X_train = np.column_stack([
                volatility[train_start:train_end - 1],   # σ²_{t-1}
                dur_pos[train_start:train_end - 1],      # Dur⁺_{t-1}
                dur_neg[train_start:train_end - 1],      # Dur⁻_{t-1}
                returns[train_start:train_end - 1]       # R_{t-1}
            ])
            y_train = direction[train_start + 1:train_end]  # D_t
            
            # 拟合模型
            coeffs = self._fit_lpm(X_train, y_train)
            
            if coeffs is None:
                continue
            
            # 使用当前时刻的特征预测下一期的方向概率
            X_pred = np.array([[
                volatility[i - 1] if i > 0 else np.nan,
                dur_pos[i - 1] if i > 0 else 0,
                dur_neg[i - 1] if i > 0 else 0,
                returns[i - 1] if i > 0 else np.nan
            ]])
            
            if not np.isnan(X_pred).any():
                prob = self._predict_probability(X_pred, coeffs)
                prob_result[i] = prob[0]
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            self.factor_names[0]: prob_result,    # 预测概率
            self.factor_names[1]: dur_pos,        # 正持续期
            self.factor_names[2]: dur_neg,        # 负持续期
        }, index=df.index)
        
        return result
    
    def get_output_names(self) -> List[str]:
        """返回因子列名列表"""
        return self.factor_names


class SimplifiedDMOMFactor(DualModeFactor):
    """
    简化版方向动量因子
    
    不使用线性概率模型训练，而是直接使用固定权重组合特征。
    适用于数据量较少或需要快速计算的场景。
    
    计算公式：
    DMOM = w1 * norm(dur_pos - dur_neg) + w2 * norm(lag_return) + w3 * (1 - norm(volatility))
    
    其中 norm 表示归一化到 [0, 1] 范围
    
    经济含义：
    - 正收益持续期越长、负收益持续期越短 → 因子值越高
    - 滞后收益率越高 → 因子值越高
    - 波动率越低 → 因子值越高（低波动更可能延续趋势）
    """
    display_name = "简化方向动量"
    _name = "SimpleDMOM"
    
    init_params = [
        Field(
            name="vol_window",
            label="波动率窗口",
            type=FieldType.INT,
            default=20,
            required=True,
            description="计算波动率的滚动窗口大小"
        ),
        Field(
            name="max_duration",
            label="最大持续期",
            type=FieldType.INT,
            default=12,
            required=True,
            description="正/负收益持续期的最大值"
        ),
        Field(
            name="w_duration",
            label="持续期权重",
            type=FieldType.FLOAT,
            default=0.4,
            required=False,
            description="持续期差异的权重"
        ),
        Field(
            name="w_return",
            label="收益率权重",
            type=FieldType.FLOAT,
            default=0.4,
            required=False,
            description="滞后收益率的权重"
        ),
        Field(
            name="w_vol",
            label="波动率权重",
            type=FieldType.FLOAT,
            default=0.2,
            required=False,
            description="波动率的权重（负向影响）"
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
        self.vol_window = int(kwargs.get("vol_window", 20))
        self.max_duration = int(kwargs.get("max_duration", 12))
        self.w_duration = float(kwargs.get("w_duration", 0.4))
        self.w_return = float(kwargs.get("w_return", 0.4))
        self.w_vol = float(kwargs.get("w_vol", 0.2))
        
        self._required_buffer_size = self.vol_window + 50
        
        name_prefix = kwargs.get("name", "")
        if name_prefix:
            self._name_prefix = name_prefix
        else:
            self._name_prefix = f"SimpleDMOM_{self.vol_window}"
        
        super().__init__()
        
        self.factor_names = [f"{self._name_prefix}"]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算简化版方向动量因子"""
        close = df['close'].values.astype(np.float64)
        n = len(close)
        
        # 计算收益率
        returns = np.zeros(n, dtype=np.float64)
        returns[0] = np.nan
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-20)
        
        # 计算滚动波动率
        volatility = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.vol_window - 1, n):
            window_returns = returns[i - self.vol_window + 1:i + 1]
            valid_returns = window_returns[~np.isnan(window_returns)]
            if len(valid_returns) >= 5:
                volatility[i] = np.std(valid_returns)
        
        # 计算持续期
        dur_pos = np.zeros(n, dtype=np.float64)
        dur_neg = np.zeros(n, dtype=np.float64)
        pos_count = 0
        neg_count = 0
        
        for i in range(n):
            if np.isnan(returns[i]):
                dur_pos[i] = np.nan
                dur_neg[i] = np.nan
                continue
            
            if returns[i] > 0:
                pos_count += 1
                neg_count = 0
            elif returns[i] < 0:
                neg_count += 1
                pos_count = 0
            
            dur_pos[i] = min(pos_count, self.max_duration)
            dur_neg[i] = min(neg_count, self.max_duration)
        
        # 归一化并计算因子值
        result = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(self.vol_window, n):
            # 持续期差异归一化 (范围 [-max_dur, max_dur] → [0, 1])
            dur_diff = dur_pos[i - 1] - dur_neg[i - 1]
            dur_norm = (dur_diff + self.max_duration) / (2 * self.max_duration)
            
            # 收益率归一化（使用tanh压缩到 [0, 1]）
            ret_norm = (np.tanh(returns[i - 1] * 20) + 1) / 2
            
            # 波动率归一化（使用窗口内的排名）
            window_vol = volatility[max(0, i - self.vol_window):i + 1]
            valid_vol = window_vol[~np.isnan(window_vol)]
            if len(valid_vol) > 0 and not np.isnan(volatility[i - 1]):
                vol_rank = np.sum(valid_vol <= volatility[i - 1]) / len(valid_vol)
                vol_norm = 1 - vol_rank  # 低波动率 → 高分
            else:
                vol_norm = 0.5
            
            # 加权组合
            if not np.isnan(dur_norm) and not np.isnan(ret_norm):
                result[i] = (
                    self.w_duration * dur_norm +
                    self.w_return * ret_norm +
                    self.w_vol * vol_norm
                )
        
        return pd.DataFrame({self.factor_names[0]: result}, index=df.index)
    
    def get_output_names(self) -> List[str]:
        return self.factor_names
