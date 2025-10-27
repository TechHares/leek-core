#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
共享内存缓存模块：
1. 跨进程数据共享
2. 数据序列化和反序列化
3. 内存管理和清理
4. 线程安全访问
"""

import os
import pickle
import time
from datetime import datetime
from typing import Any, List, Iterator

from diskcache import Cache
from leek_core.models import Field
from leek_core.data import DataSource
from leek_core.utils import get_logger, DateTimeUtils

logger = get_logger(__name__)
DISKCACHE_CACHE = Cache(".cache")

class DataCache(DataSource):
    """共享内存缓存管理器"""
    
    def __init__(self, data_source: DataSource):
        super().__init__()
        self.data_source = data_source
        self.init = False

    def parse_row_key(self, **kwargs) -> List[tuple]:
        return self.data_source.parse_row_key(**kwargs)

    def subscribe(self, row_key: str):
        raise NotImplementedError("数据源不支持实时订阅")

    def unsubscribe(self, row_key: str):
        raise NotImplementedError("数据源不支持订阅")

    def get_history_data(self, row_key: str, start_time: datetime | int = None, end_time: datetime | int = None, limit: int = None,
            pre_load_start_time: datetime | int = None, pre_load_end_time: datetime | int = None, **kwargs) -> Iterator[Any]:
        # 转换datetime参数为int，确保可哈希
        assert start_time is not None and end_time is not None and row_key is not None
        normalized_start = DateTimeUtils.datetime_to_timestamp(start_time) if isinstance(start_time, datetime) else start_time
        normalized_end = DateTimeUtils.datetime_to_timestamp(end_time) if isinstance(end_time, datetime) else end_time
        normalized_pre_load_start = normalized_start
        if pre_load_start_time is not None:
            normalized_pre_load_start = DateTimeUtils.datetime_to_timestamp(pre_load_start_time) if isinstance(pre_load_start_time, datetime) else pre_load_start_time
        normalized_pre_load_end = normalized_end
        if pre_load_end_time is not None:
            normalized_pre_load_end = DateTimeUtils.datetime_to_timestamp(pre_load_end_time) if isinstance(pre_load_end_time, datetime) else pre_load_end_time
        
        logger.info(f"get_history_data: row_key={row_key}, start_time={start_time}, end_time={end_time}, limit={limit}, **kwargs={kwargs}")
        # 调用带缓存的方法
        result_list = self._get_history_data_from_memeory(row_key, normalized_pre_load_start, normalized_pre_load_end, limit, **kwargs)
        logger.info(f"get_history_data: row_key={row_key}, result_list=len(result_list)={len(result_list)}")
        for kline in result_list:
            if kline.start_time < normalized_start:
                continue
            if kline.end_time > normalized_end:
                break
            yield kline

    def _get_history_data_from_memeory(self, row_key: str, start_time: int | None = None,
                          end_time: int | None = None, limit: int = None, **kwargs) -> List[Any]:
        # 生成缓存 key
        cache_key = self._make_cache_key("get_history_data", row_key, start_time, end_time, limit, **kwargs)

        # 1. 先尝试读缓存
        result = DISKCACHE_CACHE.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit: {cache_key}")
            return result

        # 2. 缓存 miss：尝试抢占计算权（使用 add 设置一个临时占位符）
        placeholder = f"__COMPUTING___"
        lock_key = cache_key + "_lock"
        timeout = 300
        if DISKCACHE_CACHE.add(lock_key, placeholder, expire=timeout):  # 300秒超时防止死锁
            try:
                # 只有抢到锁的进程执行实际查询
                logger.info(f"Cache miss, computing: {cache_key}")
                result = self._get_history_data(row_key, start_time, end_time, limit, **kwargs)
                # 写入真实结果
                DISKCACHE_CACHE.set(cache_key, result, expire=3600 * 12)
                return result
            finally:
                DISKCACHE_CACHE.delete(lock_key)
        else:
            # 3. 没抢到：等待结果（简单轮询）
            logger.debug(f"Waiting for other process to compute: {cache_key}")
            for _ in range(timeout):  # 最多等 timeout 秒
                time.sleep(1)
                result = DISKCACHE_CACHE.get(cache_key)
                if result is not None:
                    return result
            logger.warning(f"Timeout waiting for cache, computing directly: {cache_key}")
            return self._get_history_data(row_key, start_time, end_time, limit, **kwargs)

    def _make_cache_key(self, func_name: str, *args, **kwargs) -> str:
        # 生成稳定、可哈希的 key
        key_tuple = (func_name, args, tuple(sorted(kwargs.items())))
        return pickle.dumps(key_tuple, protocol=0).hex()  # 转为 hex 字符串

    def _get_history_data(self, row_key: str, start_time: int | None = None,
                          end_time: int | None = None, limit: int = None, **kwargs) -> List[Any]:
        """实际查询方法，返回list用于缓存"""
        print(f"从数据库加载数据[{os.getpid()}]: row_key={row_key}, start_time={start_time}, end_time={end_time}, limit={limit}, **kwargs={kwargs}")
        logger.info(f"Cache miss for row_key={row_key}, start={start_time}, end={end_time}, limit={limit}")
        if not self.init:
            self.data_source.on_start()
            self.init = True
        # 直接传入int参数（如果数据源支持）
        raw_iterator = self.data_source.get_history_data(
            row_key=row_key, start_time=start_time, end_time=end_time, limit=limit, **kwargs
        )

        result_list = list(raw_iterator)
        logger.debug(f"Cached {len(result_list)} records for row_key={row_key}")

        return result_list

    def get_supported_parameters(self) -> List[Field]:
        return self.data_source.get_supported_parameters()

    def on_stop(self):
        if self.init:
            self.data_source.on_stop()