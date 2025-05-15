#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块测试用例，演示各种日志功能的使用方法
"""

import json
import logging
import os
import time
import unittest
from datetime import datetime
from io import StringIO

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 导入模块
from leek_core.utils import (
    get_logger, log_function, log_method, log_trade
)
from leek_core.utils.logging.formatters import TextFormatter, JsonFormatter


# 配置临时日志目录
LOG_DIR = os.path.join(current_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# 创建一个内存日志处理器，用于捕获日志输出
class MemoryHandler(logging.Handler):
    """内存日志处理器，用于测试"""
    
    def __init__(self):
        super().__init__()
        self.stream = StringIO()
        self.records = []
        
    def emit(self, record):
        """记录日志条目"""
        self.records.append(record)
        self.stream.write(self.format(record) + "\n")
        
    def get_log_output(self):
        """获取日志输出内容"""
        return self.stream.getvalue()
        
    def clear(self):
        """清空日志内容"""
        self.stream = StringIO()
        self.records = []


class TestObject:
    """测试对象，用于测试方法日志装饰器"""
    
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"TestObject(name='{self.name}')"
    
    @log_method()
    def test_method(self, a, b=10):
        """测试方法"""
        return a + b
    
    @log_method(level=logging.DEBUG, log_execution_time=True)
    def slow_method(self, seconds=0.1):
        """模拟耗时操作"""
        time.sleep(seconds)
        return f"操作完成，耗时 {seconds} 秒"


@log_function()
def test_function(x, y):
    """测试函数"""
    return x * y


@log_trade()
def place_test_order(symbol, price, quantity, side="BUY"):
    """模拟下单函数"""
    # 模拟交易操作
    time.sleep(0.1)
    
    # 返回订单结果
    return {
        "order_id": "123456789",
        "symbol": symbol,
        "price": price,
        "quantity": quantity,
        "side": side,
        "status": "FILLED",
        "transaction_time": datetime.now().isoformat(),
    }


class LoggingTest(unittest.TestCase):
    """日志功能测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 初始化内存日志处理器
        print("SetUpClass")
        cls.memory_handler = MemoryHandler()
        cls.memory_handler.setLevel(logging.DEBUG)
        cls.memory_handler.setFormatter(TextFormatter(fmt="%(levelname)s - %(message)s", use_colors=False))
        
        # 清理现有日志处理器
        root_logger = logging.getLogger()
        # for handler in root_logger.handlers[:]:
        #     root_logger.removeHandler(handler)
        
        # 添加内存处理器
        root_logger.addHandler(cls.memory_handler)
        root_logger.setLevel(logging.DEBUG)

    def setUp(self):
        """每个测试前准备工作"""
        print("setUp")
        self.memory_handler.clear()

    def test_basic_logging(self):
        """测试基本日志功能"""
        # 获取测试日志器

        logger = get_logger("test.basic")
        
        # 记录不同级别的日志
        logger.debug("这是一条调试日志")
        logger.info("这是一条信息日志")
        logger.warning("这是一条警告日志")
        logger.error("这是一条错误日志")
        
        # 获取日志输出
        log_output = self.memory_handler.get_log_output()
        
        # 验证日志记录是否包含预期内容
        self.assertIn("DEBUG - 这是一条调试日志", log_output)
        self.assertIn("INFO - 这是一条信息日志", log_output)
        self.assertIn("WARNING - 这是一条警告日志", log_output)
        self.assertIn("ERROR - 这是一条错误日志", log_output)
        
        # 验证日志记录顺序
        lines = log_output.strip().split('\n')
        self.assertIn("DEBUG - 这是一条调试日志", lines[0])
        self.assertIn("INFO - 这是一条信息日志", lines[1])
        self.assertIn("WARNING - 这是一条警告日志", lines[2])
        self.assertIn("ERROR - 这是一条错误日志", lines[3])
    
    def test_json_formatter(self):
        """测试JSON格式化器"""
        # 创建JSON格式化器
        json_formatter = JsonFormatter()
        
        # 替换处理器格式化器
        self.memory_handler.setFormatter(json_formatter)
        
        # 获取测试日志器
        logger = get_logger("test.json")
        
        # 记录结构化日志
        transaction_id = "tx-12345"
        amount = "123.45"
        logger.info(
            "JSON日志测试", 
            extra={
                "transaction_id": transaction_id,
                "amount": amount,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # 获取日志输出
        log_output = self.memory_handler.get_log_output()
        
        # 解析JSON
        log_entry = json.loads(log_output.strip())
        
        # 验证基本内容
        self.assertEqual(log_entry["logger"], "test.json")
        self.assertEqual(log_entry["message"], "JSON日志测试")
        
        # 从调试输出可以看到，额外字段直接添加到了根级别
        self.assertEqual(log_entry["transaction_id"], transaction_id)
        self.assertEqual(log_entry["amount"], amount)
        
        # 还原文本格式化器
        self.memory_handler.setFormatter(TextFormatter(fmt="%(levelname)s - %(message)s"))
    
    def test_decorators(self):
        """测试日志装饰器"""
        # 测试函数装饰器
        result = test_function(5, 7)
        self.assertEqual(result, 35)
        
        # 测试方法装饰器
        test_obj = TestObject("测试实例")
        method_result = test_obj.test_method(5, b=8)
        self.assertEqual(method_result, 13)
        
        # 获取日志输出
        log_output = self.memory_handler.get_log_output()
        
        # 验证函数日志
        self.assertIn("调用函数 test_function", log_output)
        self.assertIn("函数 test_function 执行完成", log_output)
        
        # 验证方法日志
        self.assertIn("调用方法 TestObject.test_method", log_output)
        self.assertIn("方法 TestObject.test_method 执行完成", log_output)
    
    def test_trade_logging(self):
        """测试交易日志"""
        # 测试交易日志装饰器
        order = place_test_order("BTC/USDT", 50000, 0.1)
        self.assertEqual(order["status"], "FILLED")
        
        # 获取日志输出
        log_output = self.memory_handler.get_log_output()
        
        # 验证交易日志
        self.assertIn("交易操作开始: place_test_order", log_output)
        self.assertIn("交易操作完成: place_test_order", log_output)
    
    def test_exception_logging(self):
        """测试异常日志记录"""
        # 定义一个会抛出异常的函数
        @log_function()
        def failing_function():
            """故意抛出异常的函数"""
            x = 1 / 0
            return x
        
        # 测试异常记录
        try:
            failing_function()
        except ZeroDivisionError:
            pass  # 预期的异常
        
        # 获取日志输出
        log_output = self.memory_handler.get_log_output()
        
        # 验证异常信息是否被记录
        self.assertIn("ZeroDivisionError", log_output)
        self.assertIn("failing_function", log_output)
        self.assertIn("函数 failing_function 执行异常", log_output)


if __name__ == "__main__":
    # 运行测试
    unittest.main()