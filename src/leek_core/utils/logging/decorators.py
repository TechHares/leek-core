#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志装饰器模块，提供函数和方法的自动日志记录功能。
包括参数记录、异常捕获、性能跟踪等。
"""

from typing import Any, Callable, Optional, TypeVar, Union, cast
import functools
import inspect
import logging
import time

# 泛型类型变量，表示被装饰函数的类型
F = TypeVar('F', bound=Callable[..., Any])


def log_function(level: int = logging.INFO, logger: Optional[Union[str, logging.Logger]] = None,
                log_args: bool = True, log_result: bool = True, 
                log_execution_time: bool = True, exc_level: int = logging.ERROR) -> Callable[[F], F]:
    """
    函数调用日志装饰器，记录函数的调用、参数、结果和异常。
    
    参数:
        level: 日志级别，默认为INFO
        logger: 日志器对象或名称，如果为None则使用函数模块名
        log_args: 是否记录函数参数
        log_result: 是否记录函数结果
        log_execution_time: 是否记录执行时间
        exc_level: 异常日志级别，默认为ERROR
        
    返回:
        装饰过的函数
    """
    def decorator(func: F) -> F:
        """实际的装饰器函数"""
        
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
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """包装后的函数"""
            # 记录开始时间
            start_time = time.time()
            
            # 获取调用信息
            frame = inspect.currentframe()
            if frame:
                caller_frame = frame.f_back
                file_name = caller_frame.f_code.co_filename if caller_frame else "unknown"
                line_number = caller_frame.f_lineno if caller_frame else 0
                caller_info = f"{file_name}:{line_number}"
            else:
                caller_info = "unknown"
            
            # 记录函数调用
            call_msg = f"调用函数 {func_name} - 来自 {caller_info}"
            
            # 记录参数信息
            if log_args:
                # 获取函数签名
                sig = inspect.signature(func)
                # 绑定参数
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # 过滤和格式化参数值
                args_str = []
                for param_name, param_value in bound_args.arguments.items():
                    # 对于密码等敏感信息，不记录具体值
                    if param_name.lower() in ('password', 'passwd', 'secret', 'token', 'auth', 'key'):
                        param_value = '******'
                    # 对于大型对象，只记录类型和长度
                    elif isinstance(param_value, (list, tuple, dict)) and len(str(param_value)) > 100:
                        if isinstance(param_value, (list, tuple)):
                            param_value = f"{type(param_value).__name__}(len={len(param_value)})"
                        else:
                            param_value = f"{type(param_value).__name__}(keys={list(param_value.keys())})"
                            
                    args_str.append(f"{param_name}={param_value!r}")
                    
                call_msg += f" - 参数: {', '.join(args_str)}"
            
            # 记录函数调用日志
            logger.log(level, call_msg)
            
            try:
                # 执行原始函数
                result = func(*args, **kwargs)
                
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建结果消息
                result_msg = f"函数 {func_name} 执行完成"
                
                if log_execution_time:
                    result_msg += f" - 用时: {execution_time:.2f}ms"
                    
                if log_result:
                    # 过滤大型结果对象
                    result_str = str(result)
                    if len(result_str) > 200:
                        if isinstance(result, (list, tuple)):
                            result_info = f"{type(result).__name__}(len={len(result)})"
                        elif isinstance(result, dict):
                            result_info = f"{type(result).__name__}(keys={list(result.keys())})"
                        else:
                            result_info = f"{type(result).__name__}(长度={len(result_str)})"
                        result_msg += f" - 返回: {result_info}"
                    else:
                        result_msg += f" - 返回: {result!r}"
                        
                # 记录结果日志
                logger.log(level, result_msg)
                
                return result
            except Exception as e:
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建异常消息
                exc_msg = (f"函数 {func_name} 执行异常: {type(e).__name__}: {str(e)} "
                           f"- 用时: {execution_time:.2f}ms")
                
                # 记录异常日志
                logger.log(exc_level, exc_msg, exc_info=True)
                
                # 重新抛出异常
                raise
                
        return cast(F, wrapper)
    
    return decorator


def log_method(level: int = logging.INFO, logger: Optional[str] = None,
              log_args: bool = True, log_result: bool = True, 
              log_execution_time: bool = True, exc_level: int = logging.ERROR) -> Callable[[F], F]:
    """
    方法调用日志装饰器，类似log_function但特别适用于类方法。
    会自动记录类名和实例信息。
    
    参数:
        level: 日志级别，默认为INFO
        logger: 日志器名称，如果为None则使用类模块名
        log_args: 是否记录方法参数
        log_result: 是否记录方法结果
        log_execution_time: 是否记录执行时间
        exc_level: 异常日志级别，默认为ERROR
        
    返回:
        装饰过的方法
    """
    def decorator(method: F) -> F:
        """实际的装饰器函数"""
        
        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            """包装后的方法"""
            # 获取方法信息
            method_name = method.__name__
            class_name = self.__class__.__name__
            
            # 获取或创建日志器
            nonlocal logger
            log_instance = None
            if logger is None:
                log_instance = logging.getLogger(f"{self.__class__.__module__}.{class_name}")
            else:
                log_instance = logging.getLogger(logger)
            
            # 记录开始时间
            start_time = time.time()
            
            # 构建方法调用消息
            call_msg = f"调用方法 {class_name}.{method_name} [对象: {self!r}]"
            
            # 记录参数信息
            if log_args and (args or kwargs):
                # 过滤和格式化参数值
                args_str = []
                
                # 处理位置参数
                for i, arg in enumerate(args):
                    arg_str = repr(arg)
                    if len(arg_str) > 100:
                        if isinstance(arg, (list, tuple)):
                            arg_str = f"{type(arg).__name__}(len={len(arg)})"
                        elif isinstance(arg, dict):
                            arg_str = f"{type(arg).__name__}(keys={list(arg.keys())})"
                        else:
                            arg_str = f"{type(arg).__name__}(长度={len(str(arg))})"
                    args_str.append(f"arg{i+1}={arg_str}")
                
                # 处理关键字参数
                for key, value in kwargs.items():
                    # 对于密码等敏感信息，不记录具体值
                    if key.lower() in ('password', 'passwd', 'secret', 'token', 'auth', 'key'):
                        value_str = '******'
                    else:
                        value_str = repr(value)
                        if len(value_str) > 100:
                            if isinstance(value, (list, tuple)):
                                value_str = f"{type(value).__name__}(len={len(value)})"
                            elif isinstance(value, dict):
                                value_str = f"{type(value).__name__}(keys={list(value.keys())})"
                            else:
                                value_str = f"{type(value).__name__}(长度={len(str(value))})"
                    
                    args_str.append(f"{key}={value_str}")
                
                call_msg += f" - 参数: {', '.join(args_str)}"
            
            # 记录方法调用日志
            log_instance.log(level, call_msg)
            
            try:
                # 执行原始方法
                result = method(self, *args, **kwargs)
                
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建结果消息
                result_msg = f"方法 {class_name}.{method_name} 执行完成"
                
                if log_execution_time:
                    result_msg += f" - 用时: {execution_time:.2f}ms"
                    
                if log_result:
                    # 过滤大型结果对象
                    result_str = str(result)
                    if len(result_str) > 200:
                        if isinstance(result, (list, tuple)):
                            result_info = f"{type(result).__name__}(len={len(result)})"
                        elif isinstance(result, dict):
                            result_info = f"{type(result).__name__}(keys={list(result.keys())})"
                        else:
                            result_info = f"{type(result).__name__}(长度={len(result_str)})"
                        result_msg += f" - 返回: {result_info}"
                    else:
                        result_msg += f" - 返回: {result!r}"
                        
                # 记录结果日志
                log_instance.log(level, result_msg)
                
                return result
            except Exception as e:
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建异常消息
                exc_msg = (f"方法 {class_name}.{method_name} 执行异常: {type(e).__name__}: {str(e)} "
                          f"- 用时: {execution_time:.2f}ms")
                
                # 记录异常日志
                log_instance.log(exc_level, exc_msg, exc_info=True)
                
                # 重新抛出异常
                raise
                
        return cast(F, wrapper)
    
    return decorator


def log_trade(logger: Optional[Union[str, logging.Logger]] = None) -> Callable[[F], F]:
    """
    交易操作日志装饰器，专门用于记录交易操作信息。
    会记录交易相关的详细信息，并支持自定义日志格式。
    
    参数:
        logger: 日志器对象或名称，如果为None则使用"leek.trade"
        
    返回:
        装饰过的函数
    """
    def decorator(func: F) -> F:
        """实际的装饰器函数"""
        
        # 获取函数信息
        func_name = func.__name__
        
        # 获取或创建日志器
        nonlocal logger
        if logger is None:
            logger = logging.getLogger("leek.trade")
        elif isinstance(logger, str):
            logger = logging.getLogger(logger)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """包装后的函数"""
            # 记录开始时间
            start_time = time.time()
            
            # 交易操作前日志
            trade_context = {}
            
            # 提取常见的交易参数
            for key, value in kwargs.items():
                if key in ('symbol', 'side', 'type', 'quantity', 'price', 'order_id', 
                          'stop_price', 'client_order_id', 'leverage', 'margin_type'):
                    trade_context[key] = value
            
            # 如果使用位置参数，尝试从函数签名中获取参数名称
            if args:
                try:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    
                    # 如果是方法，第一个参数是self，跳过
                    if param_names[0] == 'self':
                        param_names = param_names[1:]
                        
                    # 将位置参数与名称对应
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            param_name = param_names[i]
                            if param_name in ('symbol', 'side', 'type', 'quantity', 'price', 
                                           'order_id', 'stop_price', 'client_order_id', 
                                           'leverage', 'margin_type'):
                                trade_context[param_name] = arg
                except Exception:
                    # 如果解析失败，简单记录位置参数
                    for i, arg in enumerate(args):
                        trade_context[f"arg{i+1}"] = arg
            
            # 记录交易开始
            logger.info(
                f"交易操作开始: {func_name}", 
                extra={"trade_context": trade_context, "trade_action": "start"}
            )
            
            try:
                # 执行原始函数
                result = func(*args, **kwargs)
                
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建结果信息
                result_context = trade_context.copy()
                result_context["execution_time_ms"] = f"{execution_time:.2f}"
                
                # 解析返回结果
                if result:
                    if isinstance(result, dict):
                        # 提取常见的订单返回字段
                        for key in ('order_id', 'client_order_id', 'status', 'filled_quantity', 
                                  'average_price', 'transaction_time', 'commission'):
                            if key in result:
                                result_context[f"result_{key}"] = result[key]
                    else:
                        # 简单记录结果类型
                        result_context["result_type"] = type(result).__name__
                
                # 记录交易完成
                logger.info(
                    f"交易操作完成: {func_name}", 
                    extra={"trade_context": result_context, "trade_action": "complete"}
                )
                
                return result
            except Exception as e:
                # 记录执行时间
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # 毫秒
                
                # 构建异常信息
                error_context = trade_context.copy()
                error_context["execution_time_ms"] = f"{execution_time:.2f}"
                error_context["error_type"] = type(e).__name__
                error_context["error_message"] = str(e)
                
                # 记录交易失败
                logger.error(
                    f"交易操作失败: {func_name} - {type(e).__name__}: {str(e)}", 
                    extra={"trade_context": error_context, "trade_action": "error"},
                    exc_info=True
                )
                
                # 重新抛出异常
                raise
                
        return cast(F, wrapper)
    
    return decorator 