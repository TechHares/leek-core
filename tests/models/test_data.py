#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块测试用例，演示各种日志功能的使用方法
"""

import unittest

from models import KLine


class DataTest(unittest.TestCase):
    """日志功能测试用例"""
    
    def test_data(self):
        data = KLine()
        print(data.to_dict())



if __name__ == "__main__":
    # 运行测试
    unittest.main()