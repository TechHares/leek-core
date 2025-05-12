#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理器基类模块，定义处理器的基本接口。
"""

from abc import ABC, abstractmethod
from typing import List, Callable

from base.plugin import Plugin
from models import DataType, Data


class Fabricator(Plugin, ABC):
    """数据处理器基类，定义处理器的基本接口"""
    process_data_type = {DataType.KLINE}  # 处理器支持的数据类型

    def __init__(self):
        self.send_event: Callable = None

    @abstractmethod
    def process(self, data: List[Data]) -> List[Data]:
        """
        处理数据
        
        Args:
            data: 输入数据

        Returns:
            处理后的数据列表
        """
        ...


