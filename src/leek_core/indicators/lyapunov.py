#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : lyapunov.py
"""
Lyapunov 指数指标

衡量系统的不稳定性或对初始条件的敏感度（蝴蝶效应的量化）。
使用 Rosenstein 算法从时间序列估计最大 Lyapunov 指数。
"""
import math
from collections import deque

from .t import T
from leek_core.models import KLine


class LyapunovExponent(T):
    """
    Lyapunov 指数（Lyapunov Exponent）
    
    使用 Rosenstein 算法计算最大 Lyapunov 指数，衡量系统的混沌程度。
    
    算法步骤：
    1. 相空间重构：将一维时间序列嵌入到 m 维空间
    2. 寻找最近邻：对每个点找到最近的邻居（排除时间上相邻的点）
    3. 追踪距离演化：观察邻居对随时间的分离
    4. 通过线性回归计算 Lyapunov 指数
    
    返回值说明：
    - λ > 0: 系统混沌，轨迹指数级发散，市场不可预测
    - λ ≈ 0: 系统周期性或准周期
    - λ < 0: 系统稳定，轨迹收敛，市场可预测
    
    在金融市场中：
    - λ 下降 → 市场混沌程度降低，可预测性增加
    """
    
    def __init__(self, window=100, embedding_dim=10, time_delay=1, 
                 mean_period=10, max_cache=1000, vfunc=lambda x: x.close):
        """
        初始化 Lyapunov 指数指标
        
        :param window: 分析窗口长度
        :param embedding_dim: 嵌入维度 m，决定相空间的维度
        :param time_delay: 时间延迟 τ，用于相空间重构
        :param mean_period: 平均周期，用于排除时间相邻点
        :param max_cache: 缓存大小
        :param vfunc: 用于提取价格值的函数，默认为收盘价
        """
        T.__init__(self, max_cache)
        self.window = window
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.mean_period = mean_period
        self.vfunc = vfunc
        self.q = deque(maxlen=window)
    
    def _price_to_log_returns(self, prices):
        """
        将价格序列转换为对数收益率序列
        
        :param prices: 价格序列
        :return: 对数收益率序列
        """
        if len(prices) < 2:
            return []
        
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_ret = math.log(prices[i] / prices[i-1])
                log_returns.append(log_ret)
        
        return log_returns
    
    def _embed_time_series(self, series):
        """
        相空间重构：将一维时间序列嵌入到 m 维空间
        
        X(i) = [x(i), x(i+τ), x(i+2τ), ..., x(i+(m-1)τ)]
        
        :param series: 一维时间序列
        :return: 嵌入向量列表
        """
        n = len(series)
        # 嵌入后的向量数量
        m = self.embedding_dim
        tau = self.time_delay
        
        # 需要的最小长度
        min_length = (m - 1) * tau + 1
        if n < min_length:
            return []
        
        embedded = []
        for i in range(n - (m - 1) * tau):
            vector = [series[i + j * tau] for j in range(m)]
            embedded.append(vector)
        
        return embedded
    
    def _euclidean_distance(self, v1, v2):
        """计算欧几里得距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    def _find_nearest_neighbor(self, embedded, index):
        """
        找到指定点的最近邻居（排除时间上相邻的点）
        
        :param embedded: 嵌入向量列表
        :param index: 当前点的索引
        :return: (最近邻索引, 距离)
        """
        min_dist = float('inf')
        nearest_idx = -1
        
        for j in range(len(embedded)):
            # 排除时间上相邻的点（在 mean_period 范围内的点）
            if abs(j - index) <= self.mean_period:
                continue
            
            dist = self._euclidean_distance(embedded[index], embedded[j])
            if dist < min_dist and dist > 0:
                min_dist = dist
                nearest_idx = j
        
        return nearest_idx, min_dist
    
    def _calculate_lyapunov(self, prices):
        """
        计算 Lyapunov 指数
        
        使用 Rosenstein 算法：
        1. 对每个点找到最近邻
        2. 追踪邻居对的距离演化
        3. 通过线性回归计算指数
        
        :param prices: 价格序列
        :return: Lyapunov 指数
        """
        # 转换为对数收益率
        log_returns = self._price_to_log_returns(prices)
        
        if len(log_returns) < self.window // 2:
            return None
        
        # 相空间重构
        embedded = self._embed_time_series(log_returns)
        
        if len(embedded) < 2 * self.mean_period:
            return None
        
        n_points = len(embedded)
        # 追踪的最大步数
        max_steps = min(n_points // 4, 20)
        
        if max_steps < 2:
            return None
        
        # 存储每个步数的平均对数距离
        divergence = [[] for _ in range(max_steps)]
        
        # 对每个点找到最近邻并追踪距离演化
        for i in range(n_points - max_steps):
            nearest_idx, initial_dist = self._find_nearest_neighbor(embedded, i)
            
            if nearest_idx < 0 or initial_dist <= 0:
                continue
            
            # 确保邻居也有足够的演化空间
            if nearest_idx + max_steps >= n_points:
                continue
            
            # 追踪距离演化
            for step in range(max_steps):
                if i + step < n_points and nearest_idx + step < n_points:
                    dist = self._euclidean_distance(
                        embedded[i + step], 
                        embedded[nearest_idx + step]
                    )
                    if dist > 0:
                        divergence[step].append(math.log(dist))
        
        # 计算每个步数的平均对数距离
        mean_divergence = []
        valid_steps = []
        
        for step in range(max_steps):
            if len(divergence[step]) > 0:
                mean_div = sum(divergence[step]) / len(divergence[step])
                mean_divergence.append(mean_div)
                valid_steps.append(step)
        
        if len(valid_steps) < 2:
            return None
        
        # 线性回归：mean_divergence = λ * step + c
        # 使用最小二乘法计算斜率 λ
        n = len(valid_steps)
        sum_x = sum(valid_steps)
        sum_y = sum(mean_divergence)
        sum_xx = sum(x * x for x in valid_steps)
        sum_xy = sum(valid_steps[i] * mean_divergence[i] for i in range(n))
        
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return None
        
        # 计算 Lyapunov 指数（回归斜率）
        lyapunov = (n * sum_xy - sum_x * sum_y) / denominator
        
        return lyapunov
    
    def update(self, data: KLine):
        """
        更新指标
        
        :param data: K线数据
        :return: Lyapunov 指数值，如果数据不足则返回 None
        """
        lyapunov = None
        try:
            # 提取价格值
            price = self.vfunc(data)
            if price is None:
                return lyapunov
            
            # 构建包含当前K线的价格序列
            temp_prices = list(self.q)
            temp_prices.append(float(price))
            
            # 需要足够的数据
            min_required = (self.embedding_dim - 1) * self.time_delay + self.mean_period * 2 + 10
            if len(temp_prices) < min_required:
                return lyapunov
            
            # 计算 Lyapunov 指数
            lyapunov = self._calculate_lyapunov(temp_prices)
            
            return lyapunov
        finally:
            if data.is_finished:
                price = self.vfunc(data)
                if price is not None:
                    self.q.append(float(price))
                    if lyapunov is not None:
                        self.cache.append(lyapunov)
