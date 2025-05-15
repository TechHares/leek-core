#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
彩色日志测试脚本
用于测试不同日志级别的颜色显示效果
"""

import os
import sys

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
root_dir = os.path.dirname(current_dir)
# 将项目根目录添加到系统路径
sys.path.insert(0, root_dir)

# 导入日志模块
from leek_core.utils import setup_logging, get_logger


def test_color_logging():
    """测试彩色日志输出"""
    # 获取测试日志器
    setup_logging(use_colors=True)
    logger = get_logger("test.color")
    
    # 输出不同级别的日志
    print("\n=== 彩色日志测试 ===")
    logger.debug("这是一条调试日志 (DEBUG) - 应该显示为白色")
    logger.info("这是一条信息日志 (INFO) - 应该显示为青色")
    logger.warning("这是一条警告日志 (WARNING) - 应该显示为黄色")
    logger.error("这是一条错误日志 (ERROR) - 应该显示为红色")
    print("=== 测试结束 ===\n")
    
    # 测试禁用颜色
    setup_logging(use_colors=False)
    print("\n=== 无颜色日志测试 ===")
    logger.debug("这是一条调试日志 (DEBUG) - 不应该有颜色")
    logger.info("这是一条信息日志 (INFO) - 不应该有颜色")
    logger.warning("这是一条警告日志 (WARNING) - 不应该有颜色")
    logger.error("这是一条错误日志 (ERROR) - 不应该有颜色")
    print("=== 测试结束 ===\n")


if __name__ == "__main__":
    test_color_logging()