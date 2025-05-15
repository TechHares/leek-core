#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List

from leek_core.event import EventType
from leek_core.info_fabricator import Fabricator
from leek_core.models import DataType, Field, FieldType, Data, InitDataPackage


class KlineInitFabricator(Fabricator):
    priority = float("-inf")
    process_data_type = {DataType.KLINE, DataType.INIT_PACKAGE}
    display_name = "历史K线初始化"
    init_params = [
        Field(name="num", type=FieldType.INT, default=100, label="初始化K线数量",
              description="初始化K线数量"),
    ]
    """
    历史K线初始化插件。

    该插件用于在策略启动或回测环境初始化时，批量获取指定数量的历史K线数据，
    以满足策略对初始行情环境的依赖。

    主要功能：
    1. 支持通过参数灵活指定初始化K线数量（num），满足不同策略对历史数据长度的需求。
    2. 适用于多数据源、多品种的K线数据初始化场景。

    参数说明：
    - num: 初始化K线数量，默认为100。
    """

    def __init__(self, num: int=-1):
        """
        初始化K线数据频率限制器
        参数:
            data_source_configs: data_source_configs
        """
        super().__init__()
        self.num = num
        self.init = None if num > 0 else True

    def process(self, kline: List[Data]) -> List[Data]:
        if self.init:
            return kline

        if self.init is None:
            self.init = False
            self.send_event(EventType.DATA_REQUEST, {"limit": self.num, "row_key": kline[0].row_key, "data_source_id": kline[0].data_source_id})
            return []

        if kline[0].data_type == DataType.INIT_PACKAGE:
            self.init = True
            assert isinstance(kline[0], InitDataPackage)
            for d in kline[0].history_datas:
                d.data_source_id = kline[0].data_source_id
                d.history_data = True
            return kline[0].history_datas
        return []

