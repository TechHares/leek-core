#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志配置模块，提供统一的日志配置管理
"""

import getpass
import logging
import os
import socket
import sys
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any


# 日志级别枚举
class LogLevel(Enum):
    """日志级别"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """从字符串转换为日志级别"""
        level_map = {
            'debug': cls.DEBUG,
            'info': cls.INFO,
            'warning': cls.WARNING,
            'error': cls.ERROR,
            'critical': cls.CRITICAL
        }
        return level_map.get(level_str.lower(), cls.INFO)


# 日志格式枚举
class LogFormat(Enum):
    """日志格式"""
    TEXT = auto()    # 文本格式
    JSON = auto()    # JSON格式
    SIMPLE = auto()  # 简单格式


class LogConfig:
    """日志配置类，管理全局日志设置"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'LogConfig':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = LogConfig()
        return cls._instance
    
    def __init__(self):
        """初始化默认配置"""
        # 默认配置
        self._config = {
            'level': LogLevel.INFO,
            'format_type': LogFormat.TEXT,
            'console_output': True,
            'file_output': False,  # 默认不输出到文件
            'file_path': None,
            'use_colors': False,
            'max_bytes': 10 * 1024 * 1024,  # 默认10MB
            'backup_count': 5,
            'external_handlers': []
        }
        
        # 从环境变量加载配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 日志级别
        log_level = os.environ.get('LEEK_LOG_LEVEL')
        if log_level:
            self._config['level'] = LogLevel.from_string(log_level)
        
        # 日志格式
        log_format = os.environ.get('LEEK_LOG_FORMAT')
        if log_format:
            log_format = log_format.upper()
            if log_format == 'JSON':
                self._config['format_type'] = LogFormat.JSON
            elif log_format == 'SIMPLE':
                self._config['format_type'] = LogFormat.SIMPLE
        
        # 控制台输出
        console_output = os.environ.get('LEEK_LOG_CONSOLE')
        if console_output:
            self._config['console_output'] = console_output.lower() in ('1', 'true', 'yes', 'on')
        
        # 文件输出
        file_output = os.environ.get('LEEK_LOG_FILE')
        if file_output:
            self._config['file_output'] = file_output.lower() in ('1', 'true', 'yes', 'on')
        use_colors = os.environ.get('LEEK_LOG_USE_COLORS')
        if use_colors:
            self._config['use_colors'] = use_colors.lower() in ('1', 'true', 'yes', 'on')
        
        # 文件路径
        file_path = os.environ.get('LEEK_LOG_FILE_PATH')
        if file_path:
            self._config['file_path'] = file_path
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取指定配置项"""
        return self._config.get(key, default)



# 格式化字符串获取函数
def get_format_string(format_type: LogFormat = LogFormat.TEXT) -> str:
    """根据日志格式类型获取格式化字符串"""
    if format_type == LogFormat.SIMPLE:
        return "%(levelname).1s %(message)s"
    
    return (
        "%(asctime).1s [%(levelname)s] [%(name)s] [%(threadName)s:%(thread)d] "
        "[%(processName)s:%(process)d] %(module)s.%(funcName)s:%(lineno)d - "
        "%(message)s"
    )


# 环境信息获取函数
def get_environment_info() -> Dict[str, str]:
    """获取当前环境信息"""
    return {
        'hostname': socket.gethostname(),
        'username': getpass.getuser(),
        'python_version': sys.version.split()[0],
        'timestamp': datetime.now().isoformat()
    }