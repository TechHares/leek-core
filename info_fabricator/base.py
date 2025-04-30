#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理器基类模块，定义处理器的基本接口。
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict

from base import LeekContext
from base.context import PluginContext
from base.plugin import Plugin
from event import EventBus, Event, EventType
from models import DataType, Data


class Fabricator(Plugin, ABC):
    """数据处理器基类，定义处理器的基本接口"""
    process_data_type = {DataType.KLINE}  # 处理器支持的数据类型

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


class FabricatorContext(PluginContext):
    """
    数据处理器上下文类，用于管理数据处理器实例和事件处理，伴随策略存在
    """
    def __init__(self, data_callback, event_bus: EventBus, instance_id: str, name: str, plugins: Dict[type[Plugin], Dict[str, Any]]):
        super().__init__(event_bus, instance_id, name, plugins)
        assert all(isinstance(fabricator, Fabricator) for fabricator in self._plugin_instances), "所有插件必须是Fabricator类型"
        self.fabricators: List[Fabricator] = self._plugin_instances
        self.data_callback = data_callback

    def on_data(self, data: Data):
        datas = [data]
        for fabricator in self.fabricators:
            if data.data_type not in fabricator.process_data_type:
                continue
            datas = fabricator.process(datas)
        for d in datas:
            self.data_callback(d)
