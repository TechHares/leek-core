#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import functools
import time
from .logging import get_logger
import uuid
from typing import Any, Callable, Optional, TypeVar, cast, Dict, Union, Type, Tuple, List
from collections import defaultdict, deque
logger = get_logger(__name__)

T = TypeVar('T')

# 使用模块级别的锁字典，确保所有实例共享同一个锁
_LOCKS: Dict[str, threading.Lock] = {}
_LOCK_DICT_LOCK = threading.Lock()

# 全局限速器字典，按限速参数分组
_GLOBAL_RATE_LIMITERS: Dict[str, 'RateLimiter'] = {}
_RATE_LIMITER_LOCK = threading.Lock()

class classproperty:
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner=None):
        return self.func(owner if owner is not None else type(instance))

def thread_lock(lock: Optional[threading.Lock] = None, try_lock: bool = False, on_trylock_fail: Optional[Callable[[], None]] = None):
    """
    线程锁装饰器，支持传入自定义锁实例。

    Args:
        lock (Optional[threading.Lock]): 传入的锁实例。如果不传，则为每个函数自动生成一个锁。
        try_lock (bool): 是否尝试获取锁（非阻塞），否则阻塞。
        on_trylock_fail (Optional[Callable[[], None]]): try_lock 模式下获取锁失败时的回调，默认为 None。

    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _lock = lock or threading.RLock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if try_lock:
                acquired = _lock.acquire(blocking=False)
                if not acquired:
                    if on_trylock_fail is not None:
                        on_trylock_fail()
                        return  # 不执行原函数
                    else:
                        return  # 什么都不做
            else:
                _lock.acquire()
            try:
                return func(*args, **kwargs)
            finally:
                if not try_lock or acquired:
                    _lock.release()
        return cast(Callable[..., T], wrapper)
    return decorator


class RateLimiter:
    """
    限速器类，使用滑动窗口算法实现限速
    """
    
    def __init__(self, max_requests: int, time_window: float):
        """
        初始化限速器
        
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口大小（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        
        # 使用线程安全的字典存储每个键的请求时间队列
        self._requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.RLock()
    
    def wait_if_needed(self, key_generator: Optional[Callable[..., str]] = None, *args, **kwargs) -> float:
        """
        如果需要限速则等待，返回等待时间
        
        Args:
            key_generator: 生成限速键的函数，用于区分不同的限速规则
            *args, **kwargs: 传递给key_generator的参数
            
        Returns:
            float: 等待的时间（秒）
        """
        # 使用传入的key_generator或默认的
        if key_generator is None:
            key_generator = lambda *args, **kwargs: "default"
        
        key = key_generator(*args, **kwargs)
        current_time = time.time()
        
        with self._lock:
            # 获取当前键的请求队列
            requests = self._requests[key]
            
            # 清理过期的请求记录
            while requests and current_time - requests[0] > self.time_window:
                requests.popleft()
            
            # 如果超过限制，计算需要等待的时间
            if len(requests) >= self.max_requests:
                wait_time = self.time_window - (current_time - requests[0])
                if wait_time > 0:
                    time.sleep(wait_time)
                    return wait_time
            
            # 记录当前请求时间
            requests.append(current_time)
            return 0.0


def rate_limit(max_requests: int, time_window: float, 
               key_generator: Optional[Callable[..., str]] = None,
               group: str = "default"):
    """
    限速装饰器，使用滑动窗口算法实现API限速
    
    Args:
        max_requests (int): 时间窗口内最大请求数
        time_window (float): 时间窗口大小（秒）
        key_generator (Optional[Callable[..., str]]): 生成限速键的函数，用于区分不同的限速规则
        group (str): 限速器分组，相同group的函数共享限速器，默认为"default"
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取函数信息
        func_name = func.__name__
        module_name = func.__module__
        
        # 使用全局日志器
        pass
        
        # 获取或创建全局限速器
        global _GLOBAL_RATE_LIMITERS
        with _RATE_LIMITER_LOCK:
            if group not in _GLOBAL_RATE_LIMITERS:
                _GLOBAL_RATE_LIMITERS[group] = RateLimiter(max_requests, time_window)
            limiter = _GLOBAL_RATE_LIMITERS[group]
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # 等待限速
            wait_time = limiter.wait_if_needed(key_generator, *args, **kwargs)
            
            if wait_time > 0:
                logger.debug(f"函数 {func_name} 触发限速，等待 {wait_time:.2f} 秒")
            
            # 执行原函数
            return func(*args, **kwargs)
        
        return cast(Callable[..., T], wrapper)
    return decorator


def retry(max_retries: int = 3, retry_interval: float = 1.0, 
          exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception):
    """
    重试装饰器，支持指定重试次数和重试间隔。
    
    Args:
        max_retries (int): 最大重试次数，默认为3
        retry_interval (float): 重试间隔（秒），默认为1.0
        exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]]): 需要重试的异常类型，默认为所有异常
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取函数信息
        func_name = func.__name__
        # 使用全局日志器
        pass
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 是因为第一次不算重试
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # 如果是最后一次尝试，抛出原始异常
                    if attempt == max_retries:
                        raise
                    
                    # 构建参数信息用于日志
                    args_str = ", ".join([repr(arg) for arg in args])
                    kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                    params_str = f"({args_str}{', ' + kwargs_str if kwargs_str else ''})"
                    
                    # 打印警告日志
                    logger.warning(
                        f"函数 {func_name}{params_str} 执行失败 (第{attempt + 1}次尝试): {type(e).__name__}: {str(e)}, "
                        f"{retry_interval}秒后进行第{attempt + 2}次尝试"
                    )
                    
                    # 等待重试间隔
                    if retry_interval > 0:
                        time.sleep(retry_interval)
            
            # 这里理论上不会执行到，因为最后一次尝试会抛出异常
            raise last_exception
            
        return cast(Callable[..., T], wrapper)
    return decorator

if __name__ == '__main__':
    pass
