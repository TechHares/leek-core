#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import functools
from typing import Any, Callable, Optional, TypeVar, cast, Dict

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

if __name__ == '__main__':
    pass
