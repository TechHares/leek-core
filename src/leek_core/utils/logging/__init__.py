#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块，提供Leek Core的日志功能，支持多种输出格式和目标。

主要功能:
- 统一日志接口
- 多种日志级别支持(DEBUG, INFO, WARNING, ERROR, CRITICAL)
- 多种输出格式(文本, JSON, 简单)
- 多种输出目标(控制台, 文件, 外部系统)
- 灵活的配置方式
"""

import logging
import os
from typing import List, Optional, Union

from .config import LogLevel, LogFormat, LogConfig, get_environment_info
from .decorators import log_function, log_method, log_trade
from .formatters import TextFormatter, JsonFormatter, create_formatter
from .handlers import create_handlers

# 导出公共API
__all__ = [
    # 日志级别相关
    'LogLevel',
    'LogFormat',

    # 主要功能
    'setup_logging',
    'get_logger',

    # 装饰器
    'log_function',
    'log_method',
    'log_trade'
]

# 初始化标志
_is_initialized = False


def setup_logging(
    level: Union[LogLevel, str, int] = LogLevel.INFO,
    console: bool = True,
    file: bool = False,
    use_colors=False,
    file_path: Optional[str] = None,
    format_type: Union[LogFormat, str] = LogFormat.TEXT,
    external_handlers: Optional[List[logging.Handler]] = None,
    **kwargs
) -> None:
    """
    配置全局日志设置
    
    Args:
        level: 日志级别，可以是LogLevel枚举、级别名称字符串或整数级别
        console: 是否输出到控制台
        file: 是否输出到文件
        use_colors: 输出到控制台是否使用彩色日志
        file_path: 日志文件路径，为None时使用默认路径
        format_type: 日志格式类型，可以是LogFormat枚举或格式名称字符串
        external_handlers: 额外的日志处理器列表
        **kwargs: 其他配置项
    """
    global _is_initialized

    # 处理日志级别
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    elif isinstance(level, int):
        level = logging.getLevelName(level)

    # 处理日志格式
    if isinstance(format_type, str):
        format_type = format_type.upper()
        if format_type == 'JSON':
            format_type = LogFormat.JSON
        elif format_type == 'SIMPLE':
            format_type = LogFormat.SIMPLE
        else:
            format_type = LogFormat.TEXT

    # 更新配置
    config = LogConfig.get_instance()
    config.update_config(
        level=level,
        format_type=format_type,
        console_output=console,
        file_output=file,
        use_colors=use_colors,
        file_path=file_path,
        external_handlers=external_handlers or [],
        **kwargs
    )

    # 获取根日志器
    root_logger = logging.getLogger()

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 设置日志级别
    root_logger.setLevel(level.value if isinstance(level, LogLevel) else level)

    # 创建处理器
    handlers = create_handlers(config)

    # 添加处理器到根日志器
    for handler in handlers:
        root_logger.addHandler(handler)

    # 标记为已初始化
    _is_initialized = True

    # 记录初始化完成日志
    logger = logging.getLogger('root')
    logger.info(
        f"日志系统已初始化 - 级别: {level.name if isinstance(level, LogLevel) else level}, "
        f"格式: {format_type.name if isinstance(format_type, LogFormat) else format_type}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器
    
    如果日志系统尚未初始化，会使用默认配置初始化
    
    Args:
        name: 日志器名称
        
    Returns:
        具有指定名称的日志器
    """
    global _is_initialized

    # 如果尚未初始化，使用默认配置初始化
    if not _is_initialized:
        setup_logging()

    return logging.getLogger(name)
