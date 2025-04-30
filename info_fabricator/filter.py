#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Dict, Any

from data import DataSource
from info_fabricator import Fabricator
from models import DataType, Field, FieldType, Data, LeekComponentConfig


class FilterFabricator(Fabricator):
    priority = 100
    process_data_type = {DataType.KLINE, DataType.TICK, DataType.ORDER_BOOK, DataType.TRADE}
    display_name = "K线频率控制"
    init_params = [
        Field(name="data_source_configs", type=FieldType.ARRAY, default=0.001, label="数据源配置",
              description="根据数据源参数过滤数据")
    ]
    """
    通用数据源过滤插件。

    本插件用于根据配置的各类数据源参数，对多种类型行情数据（K线、Tick、盘口、逐笔成交等）进行过滤。
    
    主要功能：
    1. 支持多数据源、多数据类型的灵活过滤。
    2. 可根据每个数据源的配置参数（如字段阈值、有效区间等）过滤不符合要求的数据。
    3. 适用于行情清洗、策略前置过滤等场景，提升下游处理效率。
    
    参数说明：
    - data_source_configs: 每个数据源的过滤配置，支持自定义字段及过滤逻辑。
    """

    def __init__(self, data_source_configs: List[LeekComponentConfig[DataSource, Dict[str, Any]]]):
        """
        初始化K线数据频率限制器
        参数:
            data_source_configs: data_source_configs
        """
        super().__init__()
        self.data_source_configs = {cfg.instance_id:cfg for cfg in data_source_configs}

    def process(self, kline: List[Data]) -> List[Data]:
        """
        时间变化和价格变化不超过阈值的K线丢弃
        """
        res = []
        for k in kline:
            if self.match(k):
                res.append(k)
        return res

    def match(self, data: Data) -> bool:
        if data.data_source_id not in self.data_source_configs:
            return False
        cfg = self.data_source_configs[data.data_source_id]
        fields = cfg.cls.fields
        for field in fields:
            dv = data.get(field.name, None)
            cv = cfg.params.get(field.name, None)
            if field.covert(dv) != field.covert(cv):
                return False
        return True
