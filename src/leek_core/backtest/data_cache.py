#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
共享内存缓存模块：
1. 跨进程数据共享
2. 数据序列化和反序列化
3. 内存管理和清理
4. 线程安全访问
"""

from datetime import datetime
from typing import Any, List, Iterator

from diskcache import Cache

from leek_core.models import Field



from leek_core.data import DataSource
from leek_core.utils import get_logger, DateTimeUtils

logger = get_logger(__name__)


class DataCache(DataSource):
    """共享内存缓存管理器"""
    
    def __init__(self, data_source: DataSource, cache_dir: str=".cache"):
        super().__init__()
        self.data_source = data_source
        logger.info(f"DataCache: cache_dir={cache_dir}")
        self.cache = Cache(cache_dir)
        self.init = False
        self._cached_get_history_data = self.cache.memoize(expire=3600)(
            self._get_history_data
        )

    def parse_row_key(self, **kwargs) -> List[tuple]:
        return self.data_source.parse_row_key(**kwargs)

    def subscribe(self, row_key: str):
        raise NotImplementedError("数据源不支持实时订阅")

    def unsubscribe(self, row_key: str):
        raise NotImplementedError("数据源不支持订阅")

    def get_history_data(self, row_key: str, start_time: datetime | int = None, end_time: datetime | int = None, limit: int = None, **kwargs) -> Iterator[Any]:
        # 转换datetime参数为int，确保可哈希
        assert start_time is not None and end_time is not None and row_key is not None
        normalized_start = DateTimeUtils.datetime_to_timestamp(start_time) if isinstance(start_time, datetime) else start_time
        normalized_end = DateTimeUtils.datetime_to_timestamp(end_time) if isinstance(end_time, datetime) else end_time
        logger.info(f"get_history_data: row_key={row_key}, start_time={start_time}, end_time={end_time}, limit={limit}, **kwargs={kwargs}")
        # 调用带缓存的方法
        result_list = self._cached_get_history_data(row_key, normalized_start, normalized_end, limit, **kwargs)
        logger.info(f"get_history_data: row_key={row_key}, result_list=len(result_list)={len(result_list)}")
        # 返回新的iterator
        return iter(result_list)

    def _get_history_data(self, row_key: str, start_time: int | None = None,
                          end_time: int | None = None, limit: int = None, **kwargs) -> List[Any]:
        """实际查询方法，返回list用于缓存"""
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
        self.data_source.on_stop()
        self.cache.close()