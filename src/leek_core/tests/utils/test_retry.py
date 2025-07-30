#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
retry装饰器测试用例
"""

import unittest
import logging
import time
from unittest.mock import patch, MagicMock
from io import StringIO

from leek_core.utils.decorator import retry


class TestRetryDecorator(unittest.TestCase):
    """retry装饰器测试类"""

    def setUp(self):
        """测试前准备"""
        # 设置日志捕获
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.logger = logging.getLogger('test_retry')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        
        # 重置日志捕获
        self.log_capture.truncate(0)
        self.log_capture.seek(0)

    def tearDown(self):
        """测试后清理"""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_basic_retry_success(self):
        """测试基本重试功能 - 成功情况"""
        call_count = 0
        
        @retry(max_retries=3, retry_interval=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("临时失败")
            return "成功"
        
        result = test_function()
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)

    def test_basic_retry_failure(self):
        """测试基本重试功能 - 失败情况"""
        @retry(max_retries=2, retry_interval=0.1)
        def test_function():
            raise ValueError("总是失败")
        
        with self.assertRaises(ValueError) as context:
            test_function()
        
        self.assertEqual(str(context.exception), "总是失败")

    def test_retry_with_parameters(self):
        """测试带参数的函数重试"""
        call_count = 0
        
        @retry(max_retries=2, retry_interval=0.1)
        def test_function(x, y, z=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"失败 {call_count}")
            return x + y + (z or 0)
        
        result = test_function(10, 20, z=30)
        self.assertEqual(result, 60)
        self.assertEqual(call_count, 3)

    def test_specific_exception_types(self):
        """测试特定异常类型重试"""
        call_count = 0
        
        @retry(max_retries=2, retry_interval=0.1, exceptions=(ValueError, RuntimeError))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("ValueError")
            elif call_count == 2:
                raise RuntimeError("RuntimeError")
            return "成功"
        
        result = test_function()
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)

    def test_non_retry_exception(self):
        """测试不重试的异常类型"""
        call_count = 0
        
        @retry(max_retries=3, retry_interval=0.1, exceptions=(ValueError,))
        def test_function():
            nonlocal call_count
            call_count += 1
            raise TypeError("TypeError不应该重试")
        
        with self.assertRaises(TypeError) as context:
            test_function()
        
        self.assertEqual(str(context.exception), "TypeError不应该重试")
        self.assertEqual(call_count, 1)  # 只调用一次，不重试

    def test_custom_logger(self):
        """测试自定义日志器"""
        # 创建新的日志捕获
        custom_log_capture = StringIO()
        custom_handler = logging.StreamHandler(custom_log_capture)
        custom_logger = logging.getLogger("custom_test")
        custom_logger.setLevel(logging.INFO)
        custom_logger.addHandler(custom_handler)
        
        call_count = 0
        
        @retry(max_retries=1, retry_interval=0.1, logger=custom_logger)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("测试日志")
        
        with self.assertRaises(ValueError):
            test_function()
        
        log_output = custom_log_capture.getvalue()
        # 检查日志输出包含测试信息
        self.assertIn("测试日志", log_output)
        self.assertIn("函数 test_function", log_output)
        
        custom_logger.removeHandler(custom_handler)
        custom_handler.close()
        custom_log_capture.close()

    def test_logger_by_name(self):
        """测试通过名称指定日志器"""
        call_count = 0
        
        @retry(max_retries=1, retry_interval=0.1, logger="named_logger")
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("命名日志器测试")
        
        with self.assertRaises(ValueError):
            test_function()

    def test_zero_retry_interval(self):
        """测试零重试间隔"""
        call_count = 0
        start_time = time.time()
        
        @retry(max_retries=2, retry_interval=0.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("快速失败")
            return "成功"
        
        result = test_function()
        end_time = time.time()
        
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)
        # 由于间隔为0，总时间应该很短
        self.assertLess(end_time - start_time, 0.1)

    def test_negative_retry_interval(self):
        """测试负重试间隔（应该等同于0）"""
        call_count = 0
        start_time = time.time()
        
        @retry(max_retries=2, retry_interval=-1.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("快速失败")
            return "成功"
        
        result = test_function()
        end_time = time.time()
        
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)
        # 由于间隔为负，应该等同于0
        self.assertLess(end_time - start_time, 0.1)

    def test_zero_max_retries(self):
        """测试零最大重试次数"""
        call_count = 0
        
        @retry(max_retries=0, retry_interval=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("失败")
        
        with self.assertRaises(ValueError):
            test_function()
        
        self.assertEqual(call_count, 1)  # 只调用一次，不重试

    def test_logging_output_format(self):
        """测试日志输出格式"""
        call_count = 0
        
        @retry(max_retries=1, retry_interval=0.1, logger=self.logger)
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            raise ValueError(f"参数测试: x={x}, y={y}")
        
        with self.assertRaises(ValueError):
            test_function(10, "test")
        
        log_output = self.log_capture.getvalue()
        
        # 检查警告日志格式
        self.assertIn("函数 test_function(10, 'test') 执行失败", log_output)
        self.assertIn("第1次尝试", log_output)
        self.assertIn("参数测试: x=10, y=test", log_output)
        self.assertIn("0.1秒后进行第2次尝试", log_output)
        
        # 检查错误日志格式
        self.assertIn("函数 test_function 执行失败，已达到最大重试次数(1)", log_output)

    def test_complex_parameters_logging(self):
        """测试复杂参数的日志记录"""
        call_count = 0
        
        @retry(max_retries=1, retry_interval=0.1, logger=self.logger)
        def test_function(dict_param, list_param, tuple_param, kwarg1=None, kwarg2="default"):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("复杂参数测试")
        
        test_dict = {"key": "value", "num": 123}
        test_list = [1, 2, 3]
        test_tuple = (True, False)
        
        with self.assertRaises(RuntimeError):
            test_function(test_dict, test_list, test_tuple, kwarg1="custom", kwarg2="override")
        
        log_output = self.log_capture.getvalue()
        
        # 检查复杂参数是否正确记录
        self.assertIn("函数 test_function(", log_output)
        self.assertIn("'key': 'value'", log_output)
        self.assertIn("[1, 2, 3]", log_output)
        self.assertIn("(True, False)", log_output)
        self.assertIn("kwarg1='custom'", log_output)
        self.assertIn("kwarg2='override'", log_output)

    def test_function_wraps(self):
        """测试函数包装器保持原函数信息"""
        @retry(max_retries=1, retry_interval=0.1)
        def test_function(x, y):
            """测试函数文档"""
            return x + y
        
        # 检查函数名和文档
        self.assertEqual(test_function.__name__, "test_function")
        self.assertEqual(test_function.__doc__, "测试函数文档")

    def test_method_retry(self):
        """测试类方法重试"""
        class TestClass:
            def __init__(self):
                self.call_count = 0
            
            @retry(max_retries=2, retry_interval=0.1)
            def test_method(self, value):
                self.call_count += 1
                if self.call_count < 3:
                    raise ValueError(f"方法失败: {value}")
                return f"成功: {value}"
        
        obj = TestClass()
        result = obj.test_method("test_value")
        
        self.assertEqual(result, "成功: test_value")
        self.assertEqual(obj.call_count, 3)

    def test_async_function_support(self):
        """测试异步函数支持（装饰器应该能正常工作）"""
        call_count = 0
        
        @retry(max_retries=1, retry_interval=0.1)
        async def async_test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("异步函数测试")
        
        # 注意：这里只是测试装饰器语法，不实际执行异步函数
        self.assertTrue(callable(async_test_function))
        self.assertEqual(async_test_function.__name__, "async_test_function")


class TestRetryDecoratorIntegration(unittest.TestCase):
    """retry装饰器集成测试"""

    def test_with_real_logging(self):
        """测试真实日志系统集成"""
        # 创建临时日志文件
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            # 创建专门的日志器
            file_logger = logging.getLogger("file_test")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_logger.addHandler(file_handler)
            file_logger.setLevel(logging.INFO)
            
            call_count = 0
            
            @retry(max_retries=1, retry_interval=0.1, logger=file_logger)
            def integration_test():
                nonlocal call_count
                call_count += 1
                raise ConnectionError("集成测试失败")
            
            with self.assertRaises(ConnectionError):
                integration_test()
            
            # 检查日志文件
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("集成测试失败", log_content)
            self.assertIn("已达到最大重试次数", log_content)
            
            file_logger.removeHandler(file_handler)
            file_handler.close()
            
        finally:
            # 清理临时文件
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_performance_impact(self):
        """测试性能影响"""
        import time
        
        # 基准测试：无装饰器
        def baseline_function():
            return "success"
        
        start_time = time.time()
        for _ in range(50):  # 进一步减少测试次数
            baseline_function()
        baseline_time = time.time() - start_time
        
        # 测试：有装饰器但无重试
        @retry(max_retries=3, retry_interval=0.1)
        def decorated_function():
            return "success"
        
        start_time = time.time()
        for _ in range(50):  # 进一步减少测试次数
            decorated_function()
        decorated_time = time.time() - start_time
        
        # 装饰器开销应该小于500%（考虑到日志记录等开销）
        overhead_ratio = (decorated_time - baseline_time) / baseline_time
        self.assertLess(overhead_ratio, 5.0, f"装饰器开销过大: {overhead_ratio:.2%}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 