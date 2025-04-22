#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日期时间工具模块，处理时间戳与日期时间格式转换。
"""

from datetime import datetime
from typing import Optional, Union


class DateTimeUtils:
    """日期时间工具类，提供各种时间格式转换方法"""
    
    # 标准时间格式
    PATTERN_MICROSECOND = '%Y-%m-%d %H:%M:%S.%f'
    PATTERN_SECOND = '%Y-%m-%d %H:%M:%S'
    PATTERN_MINUTE = '%Y-%m-%d %H:%M'
    PATTERN_DATE = '%Y-%m-%d'
    PATTERN_YYYYMMDD = '%Y%m%d'

    @staticmethod
    def to_timestamp(dt_str: str) -> int:
        """
        将日期时间字符串转换为毫秒时间戳
        
        参数:
            dt_str: 日期时间字符串，支持多种格式
            
        返回:
            毫秒时间戳(int)
        """
        if len(dt_str) == 8:
            return int(datetime.strptime(dt_str, DateTimeUtils.PATTERN_YYYYMMDD).timestamp() * 1000)
        if len(dt_str) == 10:
            return int(datetime.strptime(dt_str, DateTimeUtils.PATTERN_DATE).timestamp() * 1000)
        if len(dt_str) == 16:
            return int(datetime.strptime(dt_str, DateTimeUtils.PATTERN_MINUTE).timestamp() * 1000)
        return int(datetime.strptime(dt_str, DateTimeUtils.PATTERN_SECOND).timestamp() * 1000)

    @staticmethod
    def to_datetime(ts: Optional[Union[int, float]]) -> Optional[datetime]:
        """
        将毫秒时间戳转换为datetime对象
        
        参数:
            ts: 毫秒时间戳
            
        返回:
            datetime对象，如果输入为None则返回None
        """
        if ts is None:
            return None
        return datetime.fromtimestamp(ts / 1000)

    @staticmethod
    def to_date_str(ts: Optional[Union[int, float]], pattern: str = PATTERN_SECOND) -> str:
        """
        将毫秒时间戳转换为日期时间字符串
        
        参数:
            ts: 毫秒时间戳
            pattern: 日期时间格式，默认为年-月-日 时:分:秒
            
        返回:
            格式化的日期时间字符串，如果输入为None则返回空字符串
        """
        if ts is None:
            return ""
        return datetime.fromtimestamp(ts / 1000).strftime(pattern)
        
    @staticmethod
    def now_timestamp() -> int:
        """
        获取当前时间的毫秒时间戳
        
        返回:
            当前毫秒时间戳(int)
        """
        return int(datetime.now().timestamp() * 1000)
        
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """
        将datetime对象转换为毫秒时间戳
        
        参数:
            dt: datetime对象
            
        返回:
            毫秒时间戳(int)
        """
        return int(dt.timestamp() * 1000)
        
    @staticmethod
    def is_same_day(ts1: int, ts2: int) -> bool:
        """
        判断两个时间戳是否是同一天
        
        参数:
            ts1: 第一个毫秒时间戳
            ts2: 第二个毫秒时间戳
            
        返回:
            如果是同一天则返回True，否则返回False
        """
        dt1 = DateTimeUtils.to_datetime(ts1)
        dt2 = DateTimeUtils.to_datetime(ts2)
        return dt1.date() == dt2.date() if dt1 and dt2 else False