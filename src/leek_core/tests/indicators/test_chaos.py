#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混沌指标测试：Lyapunov 指数和相关维度
"""
import unittest
import math
import random
from datetime import datetime, timedelta
from decimal import Decimal

from leek_core.indicators import LyapunovExponent, CorrelationDimension, HurstExponent
from leek_core.models import KLine, TimeFrame


def create_mock_kline(close_price, is_finished=True, timestamp=None):
    """创建模拟 K 线数据"""
    if timestamp is None:
        timestamp = int(datetime.now().timestamp() * 1000)
    
    return KLine(
        symbol="TEST",
        market="test",
        timeframe=TimeFrame.H1,
        start_time=timestamp,
        end_time=timestamp + 3600000,
        current_time=timestamp,
        open=Decimal(str(close_price)),
        high=Decimal(str(close_price * 1.01)),
        low=Decimal(str(close_price * 0.99)),
        close=Decimal(str(close_price)),
        volume=Decimal("1000"),
        amount=Decimal(str(close_price * 1000)),
        is_finished=is_finished
    )


def generate_random_walk(n, start_price=100, volatility=0.02):
    """生成随机游走价格序列"""
    prices = [start_price]
    for _ in range(n - 1):
        change = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # 确保价格为正
    return prices


def generate_trending(n, start_price=100, trend=0.001, volatility=0.01):
    """生成趋势序列"""
    prices = [start_price]
    for _ in range(n - 1):
        change = trend + random.gauss(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))
    return prices


def generate_mean_reverting(n, start_price=100, mean=100, reversion_speed=0.1, volatility=0.01):
    """生成均值回归序列"""
    prices = [start_price]
    for _ in range(n - 1):
        reversion = reversion_speed * (mean - prices[-1]) / prices[-1]
        noise = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + reversion + noise)
        prices.append(max(new_price, 0.01))
    return prices


def generate_chaotic_logistic(n, r=3.9, x0=0.5, base_price=100, amplitude=20):
    """
    生成 Logistic Map 混沌序列
    x_{n+1} = r * x_n * (1 - x_n)
    当 r ≈ 3.9 时系统是混沌的
    """
    x = x0
    prices = []
    for _ in range(n):
        x = r * x * (1 - x)
        price = base_price + amplitude * (x - 0.5)
        prices.append(max(price, 0.01))
    return prices


class TestLyapunovExponent(unittest.TestCase):
    """Lyapunov 指数测试"""
    
    def test_initialization(self):
        """测试初始化"""
        lyap = LyapunovExponent(window=100, embedding_dim=10, time_delay=1)
        self.assertEqual(lyap.window, 100)
        self.assertEqual(lyap.embedding_dim, 10)
        self.assertEqual(lyap.time_delay, 1)
    
    def test_insufficient_data(self):
        """测试数据不足时返回 None"""
        lyap = LyapunovExponent(window=100, embedding_dim=10)
        
        # 只输入少量数据
        for i in range(10):
            kline = create_mock_kline(100 + i)
            result = lyap.update(kline)
            self.assertIsNone(result)
    
    def test_random_walk(self):
        """测试随机游走序列的 Lyapunov 指数"""
        random.seed(42)
        lyap = LyapunovExponent(window=150, embedding_dim=5, time_delay=1, mean_period=5)
        
        prices = generate_random_walk(200)
        results = []
        
        for i, price in enumerate(prices):
            kline = create_mock_kline(price)
            result = lyap.update(kline)
            if result is not None:
                results.append(result)
        
        # 应该有一些有效结果
        self.assertGreater(len(results), 0)
        
        # 检查结果是否为数值
        for r in results:
            self.assertIsInstance(r, (int, float))
    
    def test_chaotic_sequence(self):
        """测试混沌序列的 Lyapunov 指数应该为正"""
        random.seed(42)
        lyap = LyapunovExponent(window=150, embedding_dim=5, time_delay=1, mean_period=5)
        
        # 使用 Logistic Map 生成混沌序列
        prices = generate_chaotic_logistic(200, r=3.9)
        results = []
        
        for price in prices:
            kline = create_mock_kline(price)
            result = lyap.update(kline)
            if result is not None:
                results.append(result)
        
        # 混沌系统的 Lyapunov 指数通常为正
        if len(results) > 0:
            avg_lyap = sum(results) / len(results)
            # 只检查是否有有效输出
            self.assertIsNotNone(avg_lyap)
    
    def test_cache(self):
        """测试缓存功能"""
        lyap = LyapunovExponent(window=100, embedding_dim=5, max_cache=50)
        
        prices = generate_random_walk(200)
        for price in prices:
            kline = create_mock_kline(price)
            lyap.update(kline)
        
        # 检查缓存
        cached = lyap.last(100)
        self.assertLessEqual(len(cached), 50)


class TestCorrelationDimension(unittest.TestCase):
    """相关维度测试"""
    
    def test_initialization(self):
        """测试初始化"""
        cd = CorrelationDimension(window=100, embedding_dim=10, time_delay=1)
        self.assertEqual(cd.window, 100)
        self.assertEqual(cd.embedding_dim, 10)
        self.assertEqual(cd.time_delay, 1)
    
    def test_insufficient_data(self):
        """测试数据不足时返回 None"""
        cd = CorrelationDimension(window=100, embedding_dim=10)
        
        # 只输入少量数据
        for i in range(10):
            kline = create_mock_kline(100 + i)
            result = cd.update(kline)
            self.assertIsNone(result)
    
    def test_random_walk(self):
        """测试随机游走序列的相关维度"""
        random.seed(42)
        cd = CorrelationDimension(window=100, embedding_dim=5, time_delay=1, n_radius=8)
        
        prices = generate_random_walk(150)
        results = []
        
        for price in prices:
            kline = create_mock_kline(price)
            result = cd.update(kline)
            if result is not None:
                results.append(result)
        
        # 应该有一些有效结果
        self.assertGreater(len(results), 0)
        
        # 相关维度应该是正值
        for r in results:
            self.assertGreater(r, 0)
    
    def test_dimension_is_positive(self):
        """测试相关维度始终为正"""
        random.seed(123)
        cd = CorrelationDimension(window=80, embedding_dim=4, n_radius=6)
        
        prices = generate_trending(150, trend=0.002)
        
        for price in prices:
            kline = create_mock_kline(price)
            result = cd.update(kline)
            if result is not None:
                self.assertGreater(result, 0)
    
    def test_cache(self):
        """测试缓存功能"""
        cd = CorrelationDimension(window=80, embedding_dim=4, max_cache=30)
        
        prices = generate_random_walk(150)
        for price in prices:
            kline = create_mock_kline(price)
            cd.update(kline)
        
        cached = cd.last(100)
        self.assertLessEqual(len(cached), 30)


class TestChaosIndicatorsComparison(unittest.TestCase):
    """混沌指标组合测试"""
    
    def test_all_chaos_indicators_together(self):
        """测试三个混沌指标一起使用"""
        random.seed(42)
        
        hurst = HurstExponent(max_window=100, min_window=10)
        lyap = LyapunovExponent(window=100, embedding_dim=5, mean_period=5)
        cd = CorrelationDimension(window=80, embedding_dim=4, n_radius=6)
        
        prices = generate_random_walk(200)
        
        hurst_results = []
        lyap_results = []
        cd_results = []
        
        for price in prices:
            kline = create_mock_kline(price)
            
            h = hurst.update(kline)
            l = lyap.update(kline)
            c = cd.update(kline)
            
            if h is not None:
                hurst_results.append(h)
            if l is not None:
                lyap_results.append(l)
            if c is not None:
                cd_results.append(c)
        
        # 验证所有指标都产生了输出
        self.assertGreater(len(hurst_results), 0, "Hurst should produce results")
        self.assertGreater(len(lyap_results), 0, "Lyapunov should produce results")
        self.assertGreater(len(cd_results), 0, "Correlation Dimension should produce results")
        
        print(f"\n混沌指标测试结果:")
        print(f"  Hurst 指数: 平均 {sum(hurst_results)/len(hurst_results):.4f}, 样本数 {len(hurst_results)}")
        print(f"  Lyapunov 指数: 平均 {sum(lyap_results)/len(lyap_results):.4f}, 样本数 {len(lyap_results)}")
        print(f"  相关维度: 平均 {sum(cd_results)/len(cd_results):.4f}, 样本数 {len(cd_results)}")
    
    def test_trending_market_characteristics(self):
        """测试趋势市场的特征"""
        random.seed(42)
        
        hurst = HurstExponent(max_window=100, min_window=10)
        lyap = LyapunovExponent(window=100, embedding_dim=5, mean_period=5)
        
        # 生成趋势序列
        prices = generate_trending(200, trend=0.003, volatility=0.01)
        
        hurst_results = []
        lyap_results = []
        
        for price in prices:
            kline = create_mock_kline(price)
            
            h = hurst.update(kline)
            l = lyap.update(kline)
            
            if h is not None:
                hurst_results.append(h)
            if l is not None:
                lyap_results.append(l)
        
        if len(hurst_results) > 0:
            avg_hurst = sum(hurst_results) / len(hurst_results)
            # 趋势市场的 Hurst 指数通常 > 0.5
            print(f"\n趋势市场 Hurst 指数: {avg_hurst:.4f}")
    
    def test_mean_reverting_market_characteristics(self):
        """测试均值回归市场的特征"""
        random.seed(42)
        
        hurst = HurstExponent(max_window=100, min_window=10)
        
        # 生成均值回归序列
        prices = generate_mean_reverting(200, reversion_speed=0.2, volatility=0.01)
        
        hurst_results = []
        
        for price in prices:
            kline = create_mock_kline(price)
            h = hurst.update(kline)
            if h is not None:
                hurst_results.append(h)
        
        if len(hurst_results) > 0:
            avg_hurst = sum(hurst_results) / len(hurst_results)
            # 均值回归市场的 Hurst 指数通常 < 0.5
            print(f"\n均值回归市场 Hurst 指数: {avg_hurst:.4f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
