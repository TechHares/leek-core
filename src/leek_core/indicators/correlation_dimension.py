#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : correlation_dimension.py
"""
相关维度指标 (Correlation Dimension)

反映吸引子的复杂性或主导自由度的数量。
使用 Grassberger-Procaccia 算法计算。
"""
import math
from collections import deque

from .t import T
from leek_core.models import KLine


class CorrelationDimension(T):
    """
    相关维度（Correlation Dimension）
    
    使用 Grassberger-Procaccia 算法计算相关维度，衡量吸引子的复杂性。
    
    算法步骤：
    1. 相空间重构：将一维时间序列嵌入到 m 维空间
    2. 计算相关积分 C(r)：统计距离小于 r 的点对比例
    3. 对不同的 r 值，通过 ln(C(r)) vs ln(r) 的斜率估计维度
    
    返回值说明：
    - D 值较低：系统自由度少，行为简单有规律，更容易预测
    - D 值较高：系统复杂，需要更多变量描述
    - 随机噪声的 D 值会随嵌入维度增加而无限增长
    - 确定性混沌的 D 值会在某个嵌入维度后趋于饱和
    
    在金融市场中：
    - 较低的相关维度 → 市场行为更有规律，更容易预测
    """
    
    def __init__(self, window=100, embedding_dim=10, time_delay=1, 
                 n_radius=10, max_cache=1000, vfunc=lambda x: x.close):
        """
        初始化相关维度指标
        
        :param window: 分析窗口长度
        :param embedding_dim: 嵌入维度 m
        :param time_delay: 时间延迟 τ
        :param n_radius: 用于回归的半径数量
        :param max_cache: 缓存大小
        :param vfunc: 用于提取价格值的函数，默认为收盘价
        """
        T.__init__(self, max_cache)
        self.window = window
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.n_radius = n_radius
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
        
        :param series: 一维时间序列
        :return: 嵌入向量列表
        """
        n = len(series)
        m = self.embedding_dim
        tau = self.time_delay
        
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
    
    def _compute_all_distances(self, embedded):
        """
        计算所有点对之间的距离（优化版本，只计算上三角）
        
        :param embedded: 嵌入向量列表
        :return: 所有距离的列表
        """
        n = len(embedded)
        distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(embedded[i], embedded[j])
                if dist > 0:
                    distances.append(dist)
        
        return distances
    
    def _correlation_integral(self, distances, r):
        """
        计算相关积分 C(r)
        
        C(r) = (2 / N(N-1)) * 点对距离小于 r 的数量
        
        :param distances: 所有点对距离列表
        :param r: 距离阈值
        :return: 相关积分值
        """
        if len(distances) == 0:
            return 0
        
        count = sum(1 for d in distances if d < r)
        return count / len(distances)
    
    def _calculate_correlation_dimension(self, prices):
        """
        计算相关维度
        
        使用 Grassberger-Procaccia 算法：
        1. 计算所有点对距离
        2. 对不同的 r 值计算 C(r)
        3. 通过 ln(C(r)) vs ln(r) 的斜率估计维度
        
        :param prices: 价格序列
        :return: 相关维度
        """
        # 转换为对数收益率
        log_returns = self._price_to_log_returns(prices)
        
        if len(log_returns) < self.window // 2:
            return None
        
        # 相空间重构
        embedded = self._embed_time_series(log_returns)
        
        if len(embedded) < 20:
            return None
        
        # 计算所有点对距离
        distances = self._compute_all_distances(embedded)
        
        if len(distances) < 10:
            return None
        
        # 确定 r 的范围
        distances_sorted = sorted(distances)
        # 使用距离分布的 5% 到 95% 范围
        r_min = distances_sorted[max(0, len(distances_sorted) // 20)]
        r_max = distances_sorted[min(len(distances_sorted) - 1, 
                                      len(distances_sorted) * 19 // 20)]
        
        if r_min <= 0 or r_max <= r_min:
            return None
        
        # 生成对数均匀分布的 r 值
        log_r_min = math.log(r_min)
        log_r_max = math.log(r_max)
        
        if log_r_max <= log_r_min:
            return None
        
        log_r_values = []
        log_c_values = []
        
        for i in range(self.n_radius):
            log_r = log_r_min + (log_r_max - log_r_min) * i / (self.n_radius - 1)
            r = math.exp(log_r)
            
            c_r = self._correlation_integral(distances, r)
            
            if c_r > 0:
                log_r_values.append(log_r)
                log_c_values.append(math.log(c_r))
        
        if len(log_r_values) < 3:
            return None
        
        # 线性回归：ln(C(r)) = D * ln(r) + c
        # 使用最小二乘法计算斜率 D
        n = len(log_r_values)
        sum_x = sum(log_r_values)
        sum_y = sum(log_c_values)
        sum_xx = sum(x * x for x in log_r_values)
        sum_xy = sum(log_r_values[i] * log_c_values[i] for i in range(n))
        
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return None
        
        # 计算相关维度（回归斜率）
        dimension = (n * sum_xy - sum_x * sum_y) / denominator
        
        # 相关维度应该是正值
        if dimension <= 0:
            return None
        
        return dimension
    
    def update(self, data: KLine):
        """
        更新指标
        
        :param data: K线数据
        :return: 相关维度值，如果数据不足则返回 None
        """
        dimension = None
        try:
            # 提取价格值
            price = self.vfunc(data)
            if price is None:
                return dimension
            
            # 构建包含当前K线的价格序列
            temp_prices = list(self.q)
            temp_prices.append(float(price))
            
            # 需要足够的数据
            min_required = (self.embedding_dim - 1) * self.time_delay + 20
            if len(temp_prices) < min_required:
                return dimension
            
            # 计算相关维度
            dimension = self._calculate_correlation_dimension(temp_prices)
            
            return dimension
        finally:
            if data.is_finished:
                price = self.vfunc(data)
                if price is not None:
                    self.q.append(float(price))
                    if dimension is not None:
                        self.cache.append(dimension)
