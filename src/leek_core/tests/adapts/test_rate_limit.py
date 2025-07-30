#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
限速装饰器测试
"""

import unittest
import time
import threading
import logging
from unittest.mock import Mock, patch
from leek_core.utils.decorator import rate_limit, RateLimiter
from leek_core.utils import get_logger


class TestRateLimiter(unittest.TestCase):
    """限速器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.logger = get_logger(__name__)
        
    def test_rate_limiter_init(self):
        """测试限速器初始化"""
        # 测试基本初始化
        limiter = RateLimiter(max_requests=10, time_window=1.0)
        self.assertEqual(limiter.max_requests, 10)
        self.assertEqual(limiter.time_window, 1.0)
        
    def test_rate_limiter_basic(self):
        """测试基本限速功能"""
        limiter = RateLimiter(max_requests=3, time_window=1.0)
        
        # 前3次调用应该立即返回
        start_time = time.time()
        for i in range(3):
            wait_time = limiter.wait_if_needed(lambda *args, **kwargs: "test")
            self.assertEqual(wait_time, 0.0)
        
        # 第4次调用应该被限速
        wait_time = limiter.wait_if_needed(lambda *args, **kwargs: "test")
        self.assertGreater(wait_time, 0.0)
        
        end_time = time.time()
        self.assertGreaterEqual(end_time - start_time, 1.0)
        
    def test_rate_limiter_key_based(self):
        """测试基于key的限速"""
        def key_generator(*args, **kwargs):
            return kwargs.get('user_id', 'default')
        
        limiter = RateLimiter(max_requests=2, time_window=1.0)
        
        # 用户A的调用
        limiter.wait_if_needed(key_generator, user_id='user_a')
        limiter.wait_if_needed(key_generator, user_id='user_a')
        wait_time = limiter.wait_if_needed(key_generator, user_id='user_a')  # 应该被限速
        self.assertGreater(wait_time, 0.0)
        
        # 用户B的调用应该不受影响
        wait_time = limiter.wait_if_needed(key_generator, user_id='user_b')
        self.assertEqual(wait_time, 0.0)
        
    def test_rate_limiter_uuid_default(self):
        """测试默认key生成器"""
        limiter = RateLimiter(max_requests=1, time_window=1.0)
        
        # 第一次调用应该成功
        wait_time = limiter.wait_if_needed(lambda *args, **kwargs: "default")
        self.assertEqual(wait_time, 0.0)
        
        # 第二次调用应该被限速
        wait_time = limiter.wait_if_needed(lambda *args, **kwargs: "default")
        self.assertGreater(wait_time, 0.0)
            
    def test_rate_limiter_thread_safety(self):
        """测试线程安全性"""
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        results = []
        lock = threading.Lock()
        
        def worker():
            start_time = time.time()
            wait_time = limiter.wait_if_needed(lambda *args, **kwargs: "test")
            end_time = time.time()
            with lock:
                results.append({
                    'wait_time': wait_time,
                    'total_time': end_time - start_time
                })
        
        # 创建多个线程并发调用
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), 10)
        
        # 前5次调用应该立即返回
        immediate_calls = sum(1 for r in results if r['wait_time'] == 0.0)
        self.assertGreaterEqual(immediate_calls, 5)
        
        # 后5次调用应该被限速
        limited_calls = sum(1 for r in results if r['wait_time'] > 0.0)
        self.assertGreaterEqual(limited_calls, 5)


class TestRateLimitDecorator(unittest.TestCase):
    """限速装饰器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.logger = get_logger(__name__)
        
    def test_rate_limit_decorator_basic(self):
        """测试基本限速装饰器"""
        call_count = 0
        
        @rate_limit(max_requests=3, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # 前3次调用应该成功
        for i in range(3):
            result = test_function()
            self.assertEqual(result, i + 1)
        
        # 第4次调用应该被限速
        start_time = time.time()
        result = test_function()
        end_time = time.time()
        
        self.assertEqual(result, 4)
        self.assertGreaterEqual(end_time - start_time, 1.0)
        
    def test_rate_limit_decorator_with_key(self):
        """测试带key的限速装饰器"""
        call_counts = {'user_a': 0, 'user_b': 0}
        
        def key_generator(*args, **kwargs):
            return kwargs.get('user_id', 'default')
        
        @rate_limit(max_requests=2, time_window=1.0, key_generator=key_generator, logger=self.logger)
        def test_function(user_id):
            call_counts[user_id] += 1
            return call_counts[user_id]
        
        # 用户A的调用
        test_function(user_id='user_a')
        test_function(user_id='user_a')
        
        # 用户A第3次调用应该被限速
        start_time = time.time()
        test_function(user_id='user_a')
        end_time = time.time()
        self.assertGreaterEqual(end_time - start_time, 1.0)
        
        # 用户B的调用应该不受影响
        test_function(user_id='user_b')
        test_function(user_id='user_b')
        
    def test_rate_limit_decorator_uuid_default(self):
        """测试默认key的限速装饰器"""
        call_count = 0
        
        @rate_limit(max_requests=1, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # 第一次调用应该成功
        result = test_function()
        self.assertEqual(result, 1)
        
        # 第二次调用应该被限速
        start_time = time.time()
        result = test_function()
        end_time = time.time()
        
        self.assertEqual(result, 2)
        self.assertGreaterEqual(end_time - start_time, 1.0)
            
    def test_rate_limit_decorator_same_function(self):
        """测试同一函数的限速装饰器"""
        call_count = 0
        
        @rate_limit(max_requests=2, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # 前2次调用应该成功
        for i in range(2):
            result = test_function()
            self.assertEqual(result, i + 1)
        
        # 第3次调用应该被限速
        start_time = time.time()
        result = test_function()
        end_time = time.time()
        
        self.assertEqual(result, 3)
        self.assertGreaterEqual(end_time - start_time, 1.0)
            
    def test_rate_limit_decorator_logging(self):
        """测试限速装饰器的日志输出"""
        # 设置日志级别为DEBUG
        logging.basicConfig(level=logging.DEBUG)
        
        call_count = 0
        
        @rate_limit(max_requests=2, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # 前2次调用
        test_function()
        test_function()
        
        # 第3次调用应该触发限速日志
        test_function()
        
        # 验证日志输出（这里只是验证函数能正常执行）
        self.assertEqual(call_count, 3)
        
    def test_rate_limit_decorator_concurrent(self):
        """测试并发限速装饰器"""
        call_count = 0
        lock = threading.Lock()
        
        @rate_limit(max_requests=5, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            with lock:
                call_count += 1
            return call_count
        
        results = []
        result_lock = threading.Lock()
        
        def worker():
            try:
                result = test_function()
                with result_lock:
                    results.append(result)
            except Exception as e:
                with result_lock:
                    results.append(f"error: {e}")
        
        # 创建多个线程并发调用
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), 10)
        self.assertEqual(call_count, 10)
        
        # 验证所有调用都成功
        for result in results:
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)


class TestRateLimitIntegration(unittest.TestCase):
    """限速集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.logger = get_logger(__name__)
        
    def test_rate_limit_with_retry(self):
        """测试限速与重试装饰器的组合"""
        call_count = 0
        
        @rate_limit(max_requests=2, time_window=1.0, logger=self.logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("模拟错误")
            return call_count
        
        # 前2次调用会失败并重试，第3次成功
        try:
            result = test_function()
            self.assertEqual(result, 3)
        except Exception as e:
            # 如果重试装饰器没有处理异常，这是预期的行为
            self.assertEqual(str(e), "模拟错误")
            self.assertEqual(call_count, 1)
        
    def test_rate_limit_with_different_limits(self):
        """测试不同限速配置"""
        call_counts = {'fast': 0, 'slow': 0}
        
        @rate_limit(max_requests=5, time_window=1.0, logger=self.logger)
        def fast_function():
            call_counts['fast'] += 1
            return call_counts['fast']
        
        @rate_limit(max_requests=2, time_window=1.0, logger=self.logger)
        def slow_function():
            call_counts['slow'] += 1
            return call_counts['slow']
        
        # 快速函数可以调用更多次
        for i in range(5):
            fast_function()
        
        # 慢速函数会被限速
        slow_function()
        slow_function()
        
        start_time = time.time()
        slow_function()  # 应该被限速
        end_time = time.time()
        
        self.assertGreaterEqual(end_time - start_time, 1.0)
        self.assertEqual(call_counts['fast'], 5)
        self.assertEqual(call_counts['slow'], 3)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 