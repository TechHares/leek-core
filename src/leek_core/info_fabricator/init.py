#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List

from leek_core.event import EventType
from leek_core.info_fabricator import Fabricator
from leek_core.models import DataType, Field, FieldType, Data, InitDataPackage
from leek_core.utils import get_logger

logger = get_logger(__name__)


class KlineInitFabricator(Fabricator):
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
    priority = float("-inf")
    process_data_type = {DataType.KLINE, DataType.INIT_PACKAGE}
    display_name = "历史K线初始化"
    init_params = [
        Field(name="num", type=FieldType.INT, default=100, label="初始化K线数量",
              description="初始化K线数量"),
    ]

    def __init__(self, num: int=-1):
        """
        初始化K线数据频率限制器
        参数:
            data_source_configs: data_source_configs
        """
        super().__init__()
        self.num = num
        self.init = None if num > 0 else True
        self.init_data = {}

    def process(self, kline: List[Data]) -> List[Data]:
        if len(kline) == 0:
            return kline
        init = self.init_data.get(kline[0].row_key, self.init)
        if init:
            return kline

        if init is None:
            self.init_data[kline[0].row_key] = False
            data_request = {"limit": self.num, "row_key": kline[0].row_key, "data_source_id": kline[0].data_source_id}
            logger.info(f"准备初始化K线数据: {data_request}")
            self.send_event(EventType.DATA_REQUEST, data_request)
            return []

        if kline[0].data_type == DataType.INIT_PACKAGE:
            self.init_data[kline[0].row_key] = True
            assert isinstance(kline[0], InitDataPackage)
            logger.info(f"初始化K线数据: {kline[0].row_key} - {len(kline[0].history_datas)}")
            for d in kline[0].history_datas:
                d.data_source_id = kline[0].data_source_id
                d.history_data = True
            return kline[0].history_datas
        return []

