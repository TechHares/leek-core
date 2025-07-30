#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import functools
import time
import logging
from typing import Any, Callable, Optional, TypeVar, cast, Dict, Union, Type, Tuple

T = TypeVar('T')

# 使用模块级别的锁字典，确保所有实例共享同一个锁
_LOCKS: Dict[str, threading.Lock] = {}
_LOCK_DICT_LOCK = threading.Lock()

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

def retry(max_retries: int = 3, retry_interval: float = 1.0, 
          logger: Optional[Union[str, logging.Logger]] = None,
          exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception):
    """
    重试装饰器，支持指定重试次数和重试间隔。
    
    Args:
        max_retries (int): 最大重试次数，默认为3
        retry_interval (float): 重试间隔（秒），默认为1.0
        logger (Optional[Union[str, logging.Logger]]): 日志器对象或名称，如果为None则使用函数模块名
        exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]]): 需要重试的异常类型，默认为所有异常
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取函数信息
        func_name = func.__name__
        module_name = func.__module__
        
        # 获取或创建日志器
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(module_name)
        elif isinstance(logger, str):
            logger = logging.getLogger(logger)
        
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
                        logger.error(f"函数 {func_name} 执行失败，已达到最大重试次数({max_retries})，抛出异常: {type(e).__name__}: {str(e)}")
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
