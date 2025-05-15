#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any


def run_func_timeout(func, args, kwargs, timeout) -> bool:
    """
    执行函数并设置超时
    :param func: 要执行的函数
    :param args: 函数参数
    :param kwargs: 函数关键字参数
    :param timeout: 超时时间（秒）
    :return: True if function executed within timeout, False otherwise
    """
    func_timeout = invoke_func_timeout(func, args, kwargs, timeout)
    return func_timeout[1]


def invoke_func_timeout(func, args, kwargs, timeout) -> (Any, bool):
    """
    执行函数并设置超时
    :param func: 要执行的函数
    :param args: 函数参数
    :param kwargs: 函数关键字参数
    :param timeout: 超时时间（秒）
    :return: True if function executed within timeout, False otherwise
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result, True
        except TimeoutError:
            # 取消任务避免资源泄露
            future.cancel()
            return None, False
