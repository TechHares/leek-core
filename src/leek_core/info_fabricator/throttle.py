#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K线数据频率控制，限制K线数据的生成频率。避免后续策略无意义的计算过多
"""

from decimal import Decimal
from typing import List, Dict, Tuple

from leek_core.models import DataType, KLine, Field, FieldType
from leek_core.utils import get_logger
from .base import Fabricator

logger = get_logger(__name__)


class DataThrottleFabricator(Fabricator):
    """
    K线数据频率控制插件。

    该插件用于对K线（蜡烛图）数据流进行频率限制和过滤，避免因高频无效数据导致策略端无意义的计算压力。
    
    插件主要通过以下两种方式过滤K线：
    1. 价格变化率过滤：若相邻K线的价格变化率低于设定阈值，则该K线可被过滤。
    2. 时间间隔过滤：若相邻K线的时间间隔小于设定阈值，则该K线可被过滤。
    
    典型应用场景：
    - 高频行情推送时，减少对策略逻辑的无效触发
    - 控制回测或实盘时K线流的“降噪”
    
    参数说明：
    - price_change_ratio: 价格变化率阈值，低于该阈值的K线会被过滤
    - time_interval: 时间间隔阈值，低于该间隔的K线会被过滤
    """
    priority = -99
    process_data_type = {DataType.KLINE}
    display_name = "K线频率控制"
    init_params = [
        Field(name="price_change_ratio", type=FieldType.FLOAT, default=0.001, label="价格变化率",
              description="价格变化率小于阈值的K线可以过滤"),
        Field(name="time_interval", type=FieldType.INT, default=10, label="时间间隔(s)",
              description="时间间隔小于阈值的K线可以过滤"),
    ]

    def __init__(self, price_change_ratio:Decimal, time_interval:int):
        """
        初始化K线数据频率限制器
        参数:
            price_change_ratio: 价格变化率阈值
            time_interval: 时间间隔阈值
        """
        super().__init__()

        self.pre: Dict[Tuple, KLine] = {}
        self.time_interval = int(time_interval) * 1000
        self.price_change_ratio = price_change_ratio

    def process(self, kline: List[KLine]) -> List[KLine]:
        """
        时间变化和价格变化不超过阈值的K线丢弃
        """
        for k in kline:
            for nk in self._process(k):
                yield nk

    def _process(self, kline: KLine) -> List[KLine]:
        if kline.row_key not in self.pre or kline.is_finished:
            self.pre[kline.row_key] = kline
            return [kline]
        pre_kline = self.pre[kline.row_key]
        if (kline.is_finished or pre_kline.start_time != kline.start_time # 收k线或者k线时间发生变化 不能过滤
                or kline.current_time - pre_kline.current_time > self.time_interval # 时间间隔超过阈值 不能过滤
                or abs(1-kline.close/pre_kline.close) > self.price_change_ratio): # 价格变化率超过阈值 不能过滤
            self.pre[kline.row_key] = kline
            return [kline]
        return []
