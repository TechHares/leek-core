#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志格式化器模块，提供文本和JSON两种格式的日志输出。
"""

from datetime import date, datetime, time
from decimal import Decimal
import json
import logging
import traceback

from .config import LogConfig, LogFormat, get_environment_info


# 解码器，用于JSON序列化特殊类型
class DecimalEncoder(json.JSONEncoder):
    """处理Decimal类型的JSON编码器"""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        return super().default(obj)


class TextFormatter(logging.Formatter):
    """文本格式化器，提供可读性好的日志输出，支持彩色日志"""
    
    # ANSI颜色代码
    COLORS = {
        'RED': '\033[31m',       # 错误 - 红色
        'YELLOW': '\033[33m',    # 警告 - 黄色
        'CYAN': '\033[36m',      # 信息 - 青色
        'WHITE': '\033[37m',     # 调试 - 白色
        'RESET': '\033[0m'       # 重置颜色
    }
    
    # 日志级别对应的颜色
    LEVEL_COLORS = {
        'ERROR': COLORS['RED'],
        'WARNING': COLORS['YELLOW'],
        'INFO': COLORS['CYAN'],
        'DEBUG': COLORS['WHITE']
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', use_colors=True):
        """初始化文本格式化器"""
        if fmt is None:
            fmt = (
                "%(asctime)s.f [%(levelname)s] [%(name)s] "
                "[%(threadName)s:%(thread)d] [%(processName)s:%(process)d] "
                "%(module)s.%(funcName)s:%(lineno)d - %(message)s"
            )
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
        self.use_colors = use_colors
        super().__init__(fmt, datefmt, style)
    
    def formatTime(self, record, datefmt=None):
        """重写时间格式化，增加毫秒显示"""
        result = super().formatTime(record, datefmt)
        return result
    
    def format(self, record):
        """重写格式化方法，添加颜色支持"""
        # 检查消息中是否包含颜色代码并移除它们（当禁用颜色时）
        if not self.use_colors and hasattr(record, 'msg'):
            # 移除所有ANSI颜色代码
            if isinstance(record.msg, str):
                for color_code in self.COLORS.values():
                    record.msg = record.msg.replace(color_code, '')
        
        # 检查是否启用颜色并且日志级别有对应的颜色
        if self.use_colors and record.levelname in self.LEVEL_COLORS:
            # 保存原始的levelname
            original_levelname = record.levelname
            # 添加颜色到levelname
            colored_levelname = f"{self.LEVEL_COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
            # 调用父类的format方法
            result = super().format(record)
            # 恢复原始的levelname
            record.levelname = original_levelname
            return result
        
        return super().format(record)
    
    def formatException(self, exc_info):
        """增强异常格式化，包含堆栈跟踪和时间戳"""
        # 获取标准异常格式
        result = super().formatException(exc_info)
        # 添加异常发生时间
        result += f"\n异常发生时间: {datetime.now().isoformat()}"
        # 如果启用颜色，为异常添加红色
        if self.use_colors:
            result = f"{self.COLORS['RED']}{result}{self.COLORS['RESET']}"
        return result


class JsonFormatter(logging.Formatter):
    """JSON格式化器，提供结构化的日志输出，便于机器处理"""
    
    def __init__(self, indent=None):
        """初始化JSON格式化器"""
        super().__init__()
        self.indent = indent
        
    def format(self, record):
        """将日志记录格式化为JSON字符串"""
        # 基本日志信息
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "env": {"hostname": get_environment_info()["hostname"]},
            "thread": {
                "name": record.threadName,
                "id": record.thread
            },
            "process": {
                "name": record.processName,
                "id": record.process
            },
            "taskName": getattr(record, "taskName", None)
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
                "timestamp": datetime.now().isoformat()
            }
        
        # 添加额外字段
        if hasattr(record, "stack_info") and record.stack_info:
            log_data["stack"] = self.formatStack(record.stack_info)
        
        # 添加自定义字段
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "args", "asctime", "created", "exc_info", "exc_text", "filename", 
                    "funcName", "id", "levelname", "levelno", "lineno", "module", 
                    "msecs", "message", "msg", "name", "pathname", "process", 
                    "processName", "relativeCreated", "stack_info", "thread", "threadName"
                ]:
                    log_data[key] = value
        
        # 转换为JSON字符串
        return json.dumps(log_data, cls=DecimalEncoder, indent=self.indent, ensure_ascii=False)
    
    def formatTime(self, record, datefmt=None):
        """格式化时间为ISO格式"""
        return datetime.fromtimestamp(record.created).isoformat()
    
    def formatException(self, exc_info):
        """格式化异常信息"""
        return "".join(traceback.format_exception(*exc_info))
    
    def formatStack(self, stack_info):
        """格式化堆栈信息"""
        return stack_info


def create_formatter(format_type=None, use_colors=True, **kwargs):
    """
    创建格式化器工厂函数
    
    Args:
        format_type: 格式类型，默认从配置获取
        use_colors: 是否启用彩色日志输出，默认为True
        **kwargs: 传递给格式化器的额外参数
        
    Returns:
        对应类型的格式化器
    """
    # 获取配置
    config = LogConfig.get_instance()
    
    # 如果未指定格式类型，从配置获取
    if format_type is None:
        format_type = config.get("format_type", LogFormat.TEXT)
    
    # 创建对应的格式化器
    if format_type == LogFormat.JSON:
        return JsonFormatter(**kwargs)
    elif format_type == LogFormat.SIMPLE:
        return TextFormatter(
            fmt="%(levelname).1s %(message)s",
            use_colors=use_colors,
            **kwargs
        )
    else:  # 默认使用文本格式
        return TextFormatter(use_colors=use_colors, **kwargs)