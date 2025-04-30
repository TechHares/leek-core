#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎组件数据源管理子模块，给引擎提供组件管理相关功能
"""

from typing import Dict

from data import DataSource
from info_fabricator import Processor
from models import Data, Component
from event.bus import EventType, Event, EventSource, EventBus


class DataManager(Component):
    """
    管理多个数据源并提供统一接口。

    该类允许应用程序以一致的方式与多个数据源交互，
    处理连接管理、数据源选择，以及潜在的数据源之间的数据对齐。
    """

    def __init__(self, instance_id: str, name: str, event_bus: EventBus):
        """
        初始化数据管理器。

        参数:
            event_bus: 事件总线
        """
        super().__init__(instance_id, name)
        self.data_sources: Dict[str, DataSource] = {}
        self.processors: Dict[str, Processor] = {}  # 数据处理器列表

        self.event_bus = event_bus

    def data_process(self, source: EventSource, data: Data):
        """
        数据处理
        参数:
            data_type: 数据类型
            data: 数据
        """

        datas = [data]
        for processor in self.processors.values():
            if processor.process_data_type != data.data_type:
                continue
            datas = list(processor.process(datas))
            self.event_bus.publish_event(Event(EventType.DATA_PROCESSING,
                                               datas,
                                               EventSource(instance_id=processor.instance_id, name=processor.name, cls=processor.__class__.__name__)))

        for d in datas:


    def add_data_source(self, source: DataSource):
        """
        向管理器添加数据源。

        参数:
            source: 要添加的数据源
        """
        if source.instance_id in self.data_sources:
            return
        self.data_sources[source.instance_id] = source
        source.set_callback(self.data_process)
        source.on_start()

    def remove_data_source(self, instance_id: str):
        """
        从管理器中移除数据源。

        参数:
            name: 要移除的实例ID
        """
        if instance_id not in self.data_sources:
            return
        source = self.data_sources[instance_id]
        source.on_stop()
        self.data_sources.pop(instance_id)

    def add_data_processor(self, processor: Processor):
        """
        向管理器添加数据源。

        参数:
            source: 要添加的数据源
        """
        if processor.instance_id in self.processors:
            return
        self.processors[processor.instance_id] = processor
        processor.on_start()

    def remove_data_processor(self, instance_id: str):
        """
        从管理器中移除数据源。

        参数:
            name: 要移除的实例ID
        """
        if instance_id not in self.processors:
            return
        processor = self.processors[instance_id]
        processor.on_stop()
        del self.processors[instance_id]

    def destroy(self):
        """
        停止服务
        """
        for source in self.data_sources.values():
            source.on_stop()
        self.data_sources.clear()

        for processor in self.processors.values():
            processor.on_stop()
        self.processors.clear()


if __name__ == '__main__':
    pass
