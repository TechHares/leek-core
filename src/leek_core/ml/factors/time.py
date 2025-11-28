#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType, KLine

from .base import DualModeFactor


class TimeFactor(DualModeFactor):
    """
    时间周期性因子
    
    提取时间相关的周期性特征，用于捕捉日内/周内/月内/季度/年度模式：
    
    基础时间特征：
    - hour: 小时 (0-23)
    - minute: 分钟 (0-59) - 适用于高频数据
    - day_of_week: 星期几 (0=周一, 6=周日)
    - day_of_month: 月内第几天 (1-31)
    - day_of_year: 年内第几天 (1-365/366)
    - week_of_year: 年内第几周 (1-52/53)
    - month: 月份 (1-12)
    - quarter: 季度 (1-4)
    
    周期性编码（sin/cos）：
    - hour_sin/cos: 小时的周期性编码
    - minute_sin/cos: 分钟的周期性编码
    - dow_sin/cos: 星期几的周期性编码
    - month_sin/cos: 月份的周期性编码
    - quarter_sin/cos: 季度的周期性编码
    
    布尔特征：
    - is_weekend: 是否周末
    - is_month_start: 是否月初
    - is_month_end: 是否月末
    - is_quarter_start: 是否季度初
    - is_quarter_end: 是否季度末
    - is_year_start: 是否年初
    - is_year_end: 是否年末
    
    距离特征：
    - days_since_month_start: 距离月初的天数
    - days_since_quarter_start: 距离季度初的天数
    - days_since_year_start: 距离年初的天数
    """
    
    display_name = "TimeFeatures"

    init_params = [
        Field(
            name="time_column",
            label="时间列名",
            type=FieldType.STRING,
            default="start_time",
            description="DataFrame中包含时间戳的列名（时间戳单位为毫秒）"
        ),
        Field(
            name="include_hour",
            label="包含小时特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含小时特征（0-23）及其周期性编码"
        ),
        Field(
            name="include_minute",
            label="包含分钟特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含分钟特征（0-59）及其周期性编码，适用于高频数据"
        ),
        Field(
            name="include_day_of_week",
            label="包含星期特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含星期几特征（0=周一, 6=周日）及其周期性编码"
        ),
        Field(
            name="include_day_of_month",
            label="包含月内天数特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含月内第几天特征（1-31）"
        ),
        Field(
            name="include_day_of_year",
            label="包含年内天数特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含年内第几天特征（1-365/366）"
        ),
        Field(
            name="include_week_of_year",
            label="包含年内周数特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含年内第几周特征（1-52/53）"
        ),
        Field(
            name="include_month",
            label="包含月份特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含月份特征（1-12）及其周期性编码"
        ),
        Field(
            name="include_quarter",
            label="包含季度特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含季度特征（1-4）及其周期性编码"
        ),
        Field(
            name="include_cyclic",
            label="包含周期性编码",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含sin/cos周期性编码（使周期首尾相邻）"
        ),
        Field(
            name="include_boolean",
            label="包含布尔特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含布尔特征（如是否周末、是否月初等）"
        ),
        Field(
            name="include_days_since",
            label="包含距离特征",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否包含距离特征（距离月初/季度初/年初的天数）"
        ),
    ]
    _name = "Time"  # 用于列名前缀
    def __init__(self, **kwargs):
        super().__init__()
        self.time_column = kwargs.get("time_column", "start_time")
        
        # 基础时间特征开关
        self.include_hour = kwargs.get("include_hour", True)
        self.include_minute = kwargs.get("include_minute", False)  # 默认关闭，适用于高频数据
        self.include_day_of_week = kwargs.get("include_day_of_week", True)
        self.include_day_of_month = kwargs.get("include_day_of_month", True)
        self.include_day_of_year = kwargs.get("include_day_of_year", False)
        self.include_week_of_year = kwargs.get("include_week_of_year", False)
        self.include_month = kwargs.get("include_month", False)
        self.include_quarter = kwargs.get("include_quarter", False)
        
        # 周期性编码开关
        self.include_cyclic = kwargs.get("include_cyclic", True)
        
        # 布尔特征开关
        self.include_boolean = kwargs.get("include_boolean", True)
        
        # 距离特征开关
        self.include_days_since = kwargs.get("include_days_since", False)
        
        # 构建因子名称列表
        self.factor_names = []
        self._build_factor_names()

    def update(self, kline: KLine) -> List[float]:
        """
        流式更新接口
        返回时间特征列表
        """
        if not hasattr(kline, 'start_time') and not hasattr(kline, self.time_column):
            return [np.nan] * len(self.factor_names)
        
        # 获取时间戳
        timestamp = getattr(kline, self.time_column, getattr(kline, 'start_time', None))
        if timestamp is None:
            return [np.nan] * len(self.factor_names)
        
        # 转换为datetime（时间戳单位为毫秒）
        dt = pd.to_datetime(timestamp, unit='ms')
        
        features = []
        
        # 基础时间特征
        if self.include_hour:
            hour = dt.hour
            features.append(float(hour))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * hour / 24)))
                features.append(float(np.cos(2 * np.pi * hour / 24)))
        
        if self.include_minute:
            minute = dt.minute
            features.append(float(minute))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * minute / 60)))
                features.append(float(np.cos(2 * np.pi * minute / 60)))
        
        if self.include_day_of_week:
            dow = dt.dayofweek
            features.append(float(dow))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * dow / 7)))
                features.append(float(np.cos(2 * np.pi * dow / 7)))
        
        if self.include_day_of_month:
            features.append(float(dt.day))
        
        if self.include_day_of_year:
            features.append(float(dt.dayofyear))
        
        if self.include_week_of_year:
            features.append(float(dt.isocalendar()[1]))  # ISO week number
        
        if self.include_month:
            month = dt.month
            features.append(float(month))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * month / 12)))
                features.append(float(np.cos(2 * np.pi * month / 12)))
        
        if self.include_quarter:
            quarter = (dt.month - 1) // 3 + 1
            features.append(float(quarter))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * quarter / 4)))
                features.append(float(np.cos(2 * np.pi * quarter / 4)))
        
        # 布尔特征
        if self.include_boolean:
            features.append(float(dt.weekday() >= 5))  # is_weekend
            features.append(float(dt.day == 1))  # is_month_start
            # 计算是否月末：下一天是下个月的第一天
            next_day = dt + pd.Timedelta(days=1)
            features.append(float(next_day.day == 1))  # is_month_end
            features.append(float(dt.month in [1, 4, 7, 10] and dt.day == 1))  # is_quarter_start
            # 季度末：下一天是下个季度的第一天
            is_quarter_end = (dt.month in [3, 6, 9, 12]) and (next_day.day == 1)
            features.append(float(is_quarter_end))  # is_quarter_end
            features.append(float(dt.month == 1 and dt.day == 1))  # is_year_start
            features.append(float(dt.month == 12 and dt.day == 31))  # is_year_end
        
        # 距离特征
        if self.include_days_since:
            month_start = dt.replace(day=1)
            features.append(float((dt - month_start).days))
            
            quarter_start_month = ((dt.month - 1) // 3) * 3 + 1
            quarter_start = dt.replace(month=quarter_start_month, day=1)
            features.append(float((dt - quarter_start).days))
            
            year_start = dt.replace(month=1, day=1)
            features.append(float((dt - year_start).days))
        
        return features

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算接口
        在 df 上增加时间特征列
        """
        if self.time_column not in df.columns:
            # 如果没有时间列，返回原始df（不添加任何列）
            return df
        
        # 将时间戳转换为datetime（时间戳单位为毫秒）
        df["_datetime"] = pd.to_datetime(df[self.time_column], unit='ms')
        dt = df["_datetime"].dt
        
        # 计算所有时间特征
        factor_results = {}
        
        # 基础时间特征
        if self.include_hour:
            hour = dt.hour
            factor_results[f"{self.name}_hour"] = hour
            if self.include_cyclic:
                factor_results[f"{self.name}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
                factor_results[f"{self.name}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        
        if self.include_minute:
            minute = dt.minute
            factor_results[f"{self.name}_minute"] = minute
            if self.include_cyclic:
                factor_results[f"{self.name}_minute_sin"] = np.sin(2 * np.pi * minute / 60)
                factor_results[f"{self.name}_minute_cos"] = np.cos(2 * np.pi * minute / 60)
        
        if self.include_day_of_week:
            dow = dt.dayofweek
            factor_results[f"{self.name}_day_of_week"] = dow
            if self.include_cyclic:
                factor_results[f"{self.name}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
                factor_results[f"{self.name}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        if self.include_day_of_month:
            factor_results[f"{self.name}_day_of_month"] = dt.day
        
        if self.include_day_of_year:
            factor_results[f"{self.name}_day_of_year"] = dt.dayofyear
        
        if self.include_week_of_year:
            factor_results[f"{self.name}_week_of_year"] = df["_datetime"].apply(lambda x: x.isocalendar()[1])
        
        if self.include_month:
            month = dt.month
            factor_results[f"{self.name}_month"] = month
            if self.include_cyclic:
                factor_results[f"{self.name}_month_sin"] = np.sin(2 * np.pi * month / 12)
                factor_results[f"{self.name}_month_cos"] = np.cos(2 * np.pi * month / 12)
        
        if self.include_quarter:
            quarter = ((dt.month - 1) // 3 + 1)
            factor_results[f"{self.name}_quarter"] = quarter
            if self.include_cyclic:
                factor_results[f"{self.name}_quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
                factor_results[f"{self.name}_quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
        
        # 布尔特征
        if self.include_boolean:
            factor_results[f"{self.name}_is_weekend"] = (dt.dayofweek >= 5).astype(float)
            factor_results[f"{self.name}_is_month_start"] = (dt.day == 1).astype(float)
            # 计算每月最后一天：下一天是下个月的第一天
            next_day = df["_datetime"] + pd.Timedelta(days=1)
            factor_results[f"{self.name}_is_month_end"] = (next_day.dt.day == 1).astype(float)
            # 季度初：1月1日、4月1日、7月1日、10月1日
            factor_results[f"{self.name}_is_quarter_start"] = ((dt.month.isin([1, 4, 7, 10])) & (dt.day == 1)).astype(float)
            # 季度末：下一天是下个季度的第一天（即下一天是1月、4月、7月或10月的第1天）
            factor_results[f"{self.name}_is_quarter_end"] = (
                (next_day.dt.day == 1) & (next_day.dt.month.isin([1, 4, 7, 10]))
            ).astype(float)
            factor_results[f"{self.name}_is_year_start"] = ((dt.month == 1) & (dt.day == 1)).astype(float)
            factor_results[f"{self.name}_is_year_end"] = ((dt.month == 12) & (dt.day == 31)).astype(float)
        
        # 距离特征
        if self.include_days_since:
            # 距离月初的天数
            month_start = df["_datetime"].apply(lambda x: x.replace(day=1))
            factor_results[f"{self.name}_days_since_month_start"] = (df["_datetime"] - month_start).dt.days
            
            # 距离季度初的天数
            quarter_start = df["_datetime"].apply(
                lambda x: x.replace(month=((x.month - 1) // 3) * 3 + 1, day=1)
            )
            factor_results[f"{self.name}_days_since_quarter_start"] = (df["_datetime"] - quarter_start).dt.days
            
            # 距离年初的天数
            year_start = df["_datetime"].apply(lambda x: x.replace(month=1, day=1))
            factor_results[f"{self.name}_days_since_year_start"] = (df["_datetime"] - year_start).dt.days
        
        return pd.DataFrame(factor_results, index=df.index)

    def _build_factor_names(self):
        """构建因子名称列表"""
        if self.include_hour:
            self.factor_names.append("hour")
            if self.include_cyclic:
                self.factor_names.append("hour_sin")
                self.factor_names.append("hour_cos")
        
        if self.include_minute:
            self.factor_names.append("minute")
            if self.include_cyclic:
                self.factor_names.append("minute_sin")
                self.factor_names.append("minute_cos")
        
        if self.include_day_of_week:
            self.factor_names.append("day_of_week")
            if self.include_cyclic:
                self.factor_names.append("dow_sin")
                self.factor_names.append("dow_cos")
        
        if self.include_day_of_month:
            self.factor_names.append("day_of_month")
        
        if self.include_day_of_year:
            self.factor_names.append("day_of_year")
        
        if self.include_week_of_year:
            self.factor_names.append("week_of_year")
        
        if self.include_month:
            self.factor_names.append("month")
            if self.include_cyclic:
                self.factor_names.append("month_sin")
                self.factor_names.append("month_cos")
        
        if self.include_quarter:
            self.factor_names.append("quarter")
            if self.include_cyclic:
                self.factor_names.append("quarter_sin")
                self.factor_names.append("quarter_cos")
        
        if self.include_boolean:
            self.factor_names.extend([
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "is_year_start",
                "is_year_end"
            ])
        
        if self.include_days_since:
            self.factor_names.extend([
                "days_since_month_start",
                "days_since_quarter_start",
                "days_since_year_start"
            ])

    def get_output_names(self) -> List[str]:
        """返回所有因子的输出名称"""
        return [f"{self.name}_{name}" for name in self.factor_names]

