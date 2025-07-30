#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行适配器模块的所有测试
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from test_okx_adapter import TestOkxAdapter, TestOkxAdapterRateLimit
from test_rate_limit import TestRateLimiter, TestRateLimitDecorator, TestRateLimitIntegration


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加OKX适配器测试
    test_suite.addTest(unittest.makeSuite(TestOkxAdapter))
    test_suite.addTest(unittest.makeSuite(TestOkxAdapterRateLimit))
    
    # 添加限速装饰器测试
    test_suite.addTest(unittest.makeSuite(TestRateLimiter))
    test_suite.addTest(unittest.makeSuite(TestRateLimitDecorator))
    test_suite.addTest(unittest.makeSuite(TestRateLimitIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


def run_okx_adapter_tests():
    """只运行OKX适配器测试"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestOkxAdapter))
    test_suite.addTest(unittest.makeSuite(TestOkxAdapterRateLimit))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result.wasSuccessful()


def run_rate_limit_tests():
    """只运行限速装饰器测试"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestRateLimiter))
    test_suite.addTest(unittest.makeSuite(TestRateLimitDecorator))
    test_suite.addTest(unittest.makeSuite(TestRateLimitIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行适配器模块测试')
    parser.add_argument('--type', choices=['all', 'okx', 'rate_limit'], 
                       default='all', help='测试类型')
    
    args = parser.parse_args()
    
    if args.type == 'all':
        success = run_all_tests()
    elif args.type == 'okx':
        success = run_okx_adapter_tests()
    elif args.type == 'rate_limit':
        success = run_rate_limit_tests()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1) 