#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : hurst.py
# @Software: PyCharm
import math
from collections import deque

from .t import T
from leek_core.models import KLine


class HurstExponent(T):
    """
    赫斯特指数（Hurst Exponent）
    使用 R/S 分析方法计算时间序列的长期记忆性
    
    注意：本实现将对数收益率（log returns）作为输入，而非原始价格，
    以提高对非平稳序列的鲁棒性。
    
    返回值说明：
    - H = 0.5: 随机游走（无记忆性）
    - H > 0.5: 趋势性（长期记忆，正相关）
    - H < 0.5: 均值回归（反相关）
    """
    
    def __init__(self, max_window=100, min_window=10, max_cache=1000, vfunc=lambda x: x.close):
        """
        初始化赫斯特指数指标
        
        :param max_window: 最大分析窗口长度
        :param min_window: 最小分析窗口长度
        :param max_cache: 缓存大小
        :param vfunc: 用于提取价格值的函数，默认为收盘价
        """
        T.__init__(self, max_cache)
        self.max_window = max_window
        self.min_window = min_window
        self.vfunc = vfunc
        self.q = deque(maxlen=max_window)
    
    def _price_to_log_returns(self, prices):
        """
        将价格序列转换为对数收益率序列
        对数收益率 = log(prices[i] / prices[i-1]) = log(prices[i]) - log(prices[i-1])
        
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
            else:
                # 如果价格非正，跳过该点
                continue
        
        return log_returns
    
    def _calculate_rs(self, series, n):
        """
        计算给定窗口长度 n 的 R/S 值（标准方法）
        对于窗口长度 n，取最近 n 个点计算 R/S
        
        :param series: 时间序列（float列表），通常是对数收益率序列
        :param n: 窗口长度
        :return: R/S 值
        """
        if len(series) < n:
            return None
        
        # 取最近 n 个点
        window = series[-n:]
        
        # 计算均值
        mean = sum(window) / len(window)
        
        # 计算累积离差 Z(t) = Σ(x_i - mean)
        cumulative_deviations = []
        cumsum = 0.0
        for value in window:
            cumsum += value - mean
            cumulative_deviations.append(cumsum)
        
        # 计算范围 R = max(Z) - min(Z)
        R = max(cumulative_deviations) - min(cumulative_deviations)
        
        # 计算标准差 S
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        S = math.sqrt(variance) if variance > 0 else 0.0001
        
        # 计算 R/S
        if S <= 0:
            return None
        
        rs = R / S
        
        # 数值稳定性保护：防止 rs 过小导致 log(rs) 为很大的负数
        # 设置最小阈值，避免数值不稳定
        rs = max(rs, 1e-10)
        
        return rs
    
    def _calculate_hurst(self, prices):
        """
        通过线性回归计算 Hurst 指数
        
        :param prices: 价格序列
        :return: Hurst 指数
        """
        if len(prices) < self.min_window * 2:
            return None
        
        # 将价格序列转换为对数收益率序列以提高平稳性
        # 对数收益率序列比原始价格序列更平稳，更适合 R/S 分析
        log_returns = self._price_to_log_returns(prices)
        
        if len(log_returns) < self.min_window * 2:
            return None
        
        # 计算不同窗口长度的 R/S 值
        log_n = []
        log_rs = []
        
        # 使用多个窗口长度进行回归
        # 允许窗口最大到 len(log_returns)，但不超过 max_window
        # 确保至少生成2个不同的窗口大小
        max_window_size = min(len(log_returns), self.max_window)
        window_sizes = []
        
        # 如果数据较少，生成至少2个窗口（使用较小的步长）
        if len(log_returns) <= self.min_window * 2:
            # 数据刚好足够时，生成两个窗口：min_window 和 len(log_returns)
            # 但确保它们不同
            if self.min_window < max_window_size:
                window_sizes = [self.min_window, max_window_size]
            else:
                # 如果 min_window 已经等于 max_window_size，则无法生成两个不同窗口
                return None
        else:
            # 数据充足时，使用递增方式生成多个窗口
            current = self.min_window
            while current <= max_window_size:
                window_sizes.append(current)
                next_current = int(current * 1.5)
                # 如果下一个窗口会超过最大值，且当前窗口不是最大值，则添加最大值
                if next_current > max_window_size and current < max_window_size:
                    window_sizes.append(max_window_size)
                    break
                current = next_current
        
        # 去重并排序，确保至少有两个不同的窗口
        window_sizes = sorted(set(window_sizes))
        
        if len(window_sizes) < 2:
            return None
        
        for n in window_sizes:
            # 对对数收益率序列计算 R/S
            rs = self._calculate_rs(log_returns, n)
            if rs is not None and rs > 0:
                log_n.append(math.log(n))
                # rs 已经通过 max(rs, 1e-10) 保护，log(rs) 安全
                log_rs.append(math.log(rs))
        
        if len(log_n) < 2:
            return None
        
        # 线性回归：log(R/S) = H * log(n) + c
        # 使用最小二乘法计算斜率 H
        n_points = len(log_n)
        sum_log_n = sum(log_n)
        sum_log_rs = sum(log_rs)
        sum_log_n_squared = sum(n * n for n in log_n)
        sum_log_n_log_rs = sum(log_n[i] * log_rs[i] for i in range(n_points))
        
        denominator = n_points * sum_log_n_squared - sum_log_n * sum_log_n
        if abs(denominator) < 1e-10:
            return None
        
        # 计算 Hurst 指数（回归斜率）
        hurst = (n_points * sum_log_n_log_rs - sum_log_n * sum_log_rs) / denominator
        
        return hurst
    
    def update(self, data: KLine):
        """
        更新指标
        
        :param data: K线数据
        :return: Hurst 指数值，如果数据不足则返回 None
        """
        hurst = None
        try:
            # 提取价格值
            price = self.vfunc(data)
            if price is None:
                return hurst
            
            # 构建包含当前K线的价格序列（包含未完成的K线）
            temp_prices = list(self.q)
            temp_prices.append(float(price))
            
            # 如果数据不足，返回 None
            if len(temp_prices) < self.min_window * 2:
                return hurst
            
            # 计算 Hurst 指数
            hurst = self._calculate_hurst(temp_prices)
            
            return hurst
        finally:
            if data.is_finished:
                # 只在数据完成时添加到队列和缓存
                price = self.vfunc(data)
                if price is not None:
                    self.q.append(float(price))
                    if hurst is not None:
                        self.cache.append(hurst)

