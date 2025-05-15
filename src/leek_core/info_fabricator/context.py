#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Any, Dict

from leek_core.base import LeekContext, create_component
from leek_core.event import EventBus, EventType, Event, EventSource
from leek_core.models import Data, LeekComponentConfig
from leek_core.utils import get_logger
from .base import Fabricator

logger = get_logger(__name__)

class FabricatorContext(LeekContext):
    """
    数据处理器上下文类，用于管理数据处理器实例和事件处理，伴随策略存在
    """
    def __init__(self, data_callback, event_bus: EventBus, config: LeekComponentConfig[None, List[LeekComponentConfig[Fabricator, Dict[str, Any]]]]):
        super().__init__(event_bus, config)
        self.fabricators: List[Fabricator] = []
        self.data_callback = data_callback

        for pla in config.config:
            plugin = create_component(pla.cls, **pla.config)
            plugin.send_event = self.send_event
            isinstance(plugin, Fabricator)
            self.fabricators.append(plugin)

        self.fabricators.sort(key=lambda p: p.priority)

    def send_event(self, tp: EventType, data: Dict[str, Any]):
        if isinstance(tp, str):
            tp = EventType(tp)
        self.event_bus.publish_event(Event(
            event_type=tp,
            data=data,
            source=EventSource(
                instance_id=self.instance_id,
                name=self.name,
                cls=self.__class__.__name__,
                extra={
                    "data_source_id": data.get("data_source_id", None)
                }
            ),
        ))


    def on_data(self, data: Data):
        datas = [data]
        for fabricator in self.fabricators:
            if len(datas) == 0:
                break
            if datas[0].data_type not in fabricator.process_data_type:
                continue
            try:
                datas = list(fabricator.process(datas))
            except Exception as e:
                logger.error(f"fabricator {fabricator.display_name} process error: {e}", exc_info=True)
                continue

        for d in datas:
            self.data_callback(d)


    def on_start(self):
        """
        启动组件
        """
        for plugin in self.fabricators:
            plugin.on_start()

    def on_stop(self):
        """
        停止组件
        """
        for plugin in self.fabricators:
            plugin.on_stop()