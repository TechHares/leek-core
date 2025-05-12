#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据源管理器模块，为引擎提供统一的数据源组件管理能力
"""
import logging
from typing import Dict, Any

from data import DataSource, DataSourceContext
from event import EventBus, Event, EventType, EventSource
from models import LeekComponentConfig, InitDataPackage, DataType
from utils import log_method, get_logger
from .base import ComponentManager
logger = get_logger(__name__)


class DataManager(ComponentManager[DataSourceContext, DataSource, Dict[str, Any]]):
    """
    数据源管理器。

    负责统一管理所有数据源上下文（DataSourceContext），提供数据源的添加、获取、删除等操作接口。
    支持数据请求转发、数据源生命周期管理。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[DataSourceContext, None]):
        """
        初始化数据源管理器。

        参数：
            event_bus: 事件总线
            config: 数据源管理器配置
        """
        super().__init__(event_bus, config)


    @log_method(level=logging.DEBUG, log_execution_time=True)
    def handle_data_request(self, event: Event) -> None:
        """
        处理数据请求事件，将请求参数转发给对应的数据源实例。

        参数：
            event: Event 事件对象，包含请求信息
        """
        data_source_id = event.source.extra['data_source_id']
        data_source = self.get(data_source_id)
        if data_source is None:
            raise ValueError(f"未找到对应数据源: {data_source_id}")
        assert isinstance(event.data, dict)
        data = data_source.get_history_data(*event.data.get('row_key', ()),
                                            start_time=event.data.get('start_time', None),
                                            end_time=event.data.get('end_time', None),
                                            limit=event.data.get('limit', None))
        pack = InitDataPackage(
            history_datas=list(data),
            data_source_id=data_source_id,
            pack_row_key=event.data.get('row_key', ()),
            data_type = DataType.INIT_PACKAGE,
            target_instance_id={event.source.instance_id}
        )
        self.event_bus.publish_event(Event(EventType.DATA_RESPONSE, pack,
                                           EventSource(
                                               instance_id=data_source_id,
                                               name=data_source.name
                                           )))

    def handle_data_subscribe(self, event: Event) -> None:
        """
        处理数据订阅，将订阅请求转发给对应的数据源。
        :param event: 事件对象，包含订阅信息
        """
        data_source_id = event.source.extra['data_source_id']
        data_source = self.get(data_source_id)
        if data_source is None:
            raise ValueError(f"未找到对应数据源: {data_source_id}")

        data_source.subscribe(event.source.instance_id, **event.data)

    def handle_data_unsubscribe(self, event: Event) -> None:
        """
        处理数据取消订阅，将取消订阅请求转发给对应的数据源。
        :param event: 事件对象，包含取消订阅信息
        """
        data_source_id = event.source.extra['data_source_id']
        data_source = self.get(data_source_id)
        if data_source is None:
            raise ValueError(f"未找到对应数据源: {data_source_id}")
        data_source.unsubscribe(event.source.instance_id, **event.data)

    def on_start(self):
        """
        订阅相关事件
        """
        self.event_bus.subscribe_event(EventType.DATA_REQUEST, self.handle_data_request)
        self.event_bus.subscribe_event(EventType.DATA_SOURCE_SUBSCRIBE, self.handle_data_subscribe)
        self.event_bus.subscribe_event(EventType.DATA_SOURCE_UNSUBSCRIBE, self.handle_data_unsubscribe)
        logger.info(f"事件订阅: 数据源管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.DATA_REQUEST, EventType.DATA_SOURCE_UNSUBSCRIBE, EventType.DATA_SOURCE_SUBSCRIBE]]}")

