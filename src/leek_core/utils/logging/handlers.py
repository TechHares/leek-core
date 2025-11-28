#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志处理器模块，支持多种日志输出目标，包括控制台、文件和外部系统。
"""

from logging import FileHandler
from typing import List
import atexit
import logging
import os
import socket
import sys
import threading
import time

from .config import LogConfig, LogFormat
from .formatters import create_formatter


class SafeFileHandler(FileHandler):
    """安全的文件滚动处理器，处理文件锁和权限问题"""
    
    def __init__(self, filename, encoding=None):
        """初始化安全文件处理器"""
        # 确保目录存在
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"警告: 无法创建日志目录 {directory}: {e}", file=sys.stderr)
        
        # 调用父类初始化
        super().__init__(filename, encoding=encoding)


class HttpHandler(logging.Handler):
    """HTTP处理器，将日志发送到HTTP端点"""
    
    def __init__(self, url, method='POST', batch_size=10, 
                 flush_interval=5.0, headers=None, **kwargs):
        """
        初始化HTTP处理器
        
        Args:
            url: HTTP端点URL
            method: HTTP方法(GET/POST)
            batch_size: 批处理大小，达到此数量自动发送
            flush_interval: 刷新间隔(秒)
            headers: HTTP头信息
            **kwargs: 额外参数
        """
        super().__init__()
        self.url = url
        self.method = method
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.1, flush_interval)
        self.headers = headers or {'Content-Type': 'application/json'}
        self.buffer = []
        self.lock = threading.RLock()
        self.last_flush = time.time()
        
        # 尝试导入requests库
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None
            print("警告: 未安装requests库，HTTP日志处理器将不可用", file=sys.stderr)
        
        # 设置定时刷新和退出时刷新
        if self.flush_interval > 0:
            t = threading.Thread(
                target=self._flush_worker, 
                daemon=True,
                name="HttpHandler-Flusher"
            )
            t.start()
        
        # 注册退出处理
        atexit.register(self.flush)
    
    def emit(self, record):
        """处理日志记录"""
        if not self.requests:
            return
        
        try:
            # 格式化记录
            message = self.format(record)
            
            # 添加到缓冲区
            with self.lock:
                self.buffer.append(message)
                
                # 检查是否需要刷新
                if len(self.buffer) >= self.batch_size:
                    self.flush()
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """刷新缓冲区中的日志记录"""
        if not self.requests or not self.buffer:
            return
        
        with self.lock:
            if not self.buffer:
                return
            
            buffer_to_send = self.buffer.copy()
            self.buffer.clear()
        
        # 发送日志
        try:
            payload = buffer_to_send if len(buffer_to_send) > 1 else buffer_to_send[0]
            self.requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                json=payload,
                timeout=5
            )
            self.last_flush = time.time()
        except Exception as e:
            print(f"无法发送日志到 {self.url}: {e}", file=sys.stderr)
            # 重新添加到缓冲区
            with self.lock:
                self.buffer.extend(buffer_to_send)
                if len(self.buffer) > self.batch_size * 2:
                    # 防止内存不断增长
                    self.buffer = self.buffer[-self.batch_size:]
    
    def _flush_worker(self):
        """定时刷新工作线程"""
        while True:
            time.sleep(self.flush_interval)
            
            # 如果有数据且达到间隔时间，刷新
            if self.buffer and time.time() - self.last_flush >= self.flush_interval:
                self.flush()


class TcpSocketHandler(logging.Handler):
    """TCP Socket处理器，将日志发送到TCP服务器"""
    
    def __init__(self, host, port, reconnect_interval=5.0):
        """
        初始化TCP Socket处理器
        
        Args:
            host: 服务器主机名或IP
            port: 服务器端口
            reconnect_interval: 重连间隔(秒)
        """
        super().__init__()
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.socket = None
        self.last_connect_attempt = 0
        self.lock = threading.RLock()
        
        # 连接到服务器
        self._connect()
        
        # 注册退出处理
        atexit.register(self.close)
    
    def _connect(self):
        """连接到TCP服务器"""
        if self.socket:
            return True
        
        # 避免频繁重连
        now = time.time()
        if now - self.last_connect_attempt < self.reconnect_interval:
            return False
        
        self.last_connect_attempt = now
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            self.socket = None
            print(f"无法连接到日志服务器 {self.host}:{self.port}: {e}", file=sys.stderr)
            return False
    
    def emit(self, record):
        """处理日志记录"""
        try:
            message = self.format(record) + '\n'
            
            with self.lock:
                if not self._connect():
                    return
                
                try:
                    self.socket.sendall(message.encode('utf-8'))
                except Exception:
                    # 连接可能已断开，下次尝试重连
                    self.socket = None
        except Exception:
            self.handleError(record)
    
    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        
        super().close()


def create_handlers(config=None) -> List[logging.Handler]:
    """
    创建日志处理器
    
    Args:
        config: 日志配置，如果为None则使用全局配置
    
    Returns:
        处理器列表
    """
    if config is None:
        config = LogConfig.get_instance()
    
    handlers = []
    config_dict = config.get_config()
    
    # 获取格式化器
    format_type = config_dict.get('format_type', LogFormat.TEXT)
    formatter = create_formatter(format_type, use_colors=config_dict.get('use_colors', False))
    
    # 控制台处理器
    if config_dict.get('console_output', True):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # 文件处理器
    if config_dict.get('file_output', False):
        file_path = config_dict.get('file_path')
        if file_path:
            # 使用安全的滚动文件处理器
            file_handler = SafeFileHandler(
                filename=file_path,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
    
    # 添加外部处理器
    external_handlers = config_dict.get('external_handlers', [])
    for handler in external_handlers:
        if isinstance(handler, logging.Handler):
            handler.setFormatter(formatter)
            handlers.append(handler)
    
    return handlers