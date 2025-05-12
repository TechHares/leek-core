#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from base import LeekContext
from event import EventBus, Event, EventType, EventSource
from models import DataType, Data, KLine, LeekComponentConfig
from .base import DataSource


class DataSourceContext(LeekContext):
    """
    数据源上下文。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        super().__init__(event_bus, config)
        self._data_source = self.create_component()
        self._data_source.callback = self.send_data
        self.is_connected = False
        self.params_list = None
        self.subscribe_info: Dict[tuple, set[str]] = {}

    def send_data(self, data: Data):
        if self.params_list is None:
            self.params_list = self._data_source.get_supported_parameters()

        data.data_source_id = self.instance_id
        data.target_instance_id = self.subscribe_info.get(data.row_key, set())
        if isinstance(data, KLine):
            data.data_type = DataType.KLINE
        self.event_bus.publish_event(Event(EventType.DATA_RECEIVED, data, EventSource(
            instance_id=self.instance_id,
            name=self.name,
            cls=self.config.cls.__name__,
            extra={"params": [p.name for p in self.params_list]}
        )))

    def subscribe(self, instance_id, **kwargs):
        for row_key in self._data_source.parse_row_key(**kwargs):
            if row_key not in self.subscribe_info:
                self.subscribe_info[row_key] = set()
                self._data_source.subscribe(*row_key)
            self.subscribe_info[row_key].add(instance_id)

    def unsubscribe(self, instance_id, **kwargs):
        for row_key in self._data_source.parse_row_key(**kwargs):
            if row_key not in self.subscribe_info:
                self._data_source.unsubscribe(*row_key)
                continue
            self.subscribe_info[row_key].remove(instance_id)
            if len(self.subscribe_info[row_key]) == 0:
                self._data_source.unsubscribe(*row_key)
                del self.subscribe_info[row_key]

    def get_history_data(self,  *row_key,start_time: int = None, end_time: int = None, limit: int = None) -> List[Data]:
        return self._data_source.get_history_data(*row_key, start_time=start_time, end_time=end_time, limit=limit)

    def on_start(self):
        """
        启动数据源。
        """
        if self.is_connected:
            return
        self._data_source.on_start()
        self.is_connected = True

    def on_stop(self):
        """
        停止数据源。
        """
        if not self.is_connected:
            return
        self._data_source.on_stop()
        self.is_connected = False

    def get_state(self) -> Dict[str, Any]:
        """
        获取数据源状态。
        返回:
            Dict[str, Any]: 数据源状态
        """
        raise NotImplementedError("数据源暂不支持序列化状态")

    def load_state(self, state: Dict[str, Any]):
        """
        设置数据源状态。
        参数:
            state: 数据源状态
        """
        raise NotImplementedError("数据源暂不支持序列化状态")

