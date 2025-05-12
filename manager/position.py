#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any

from data import DataSource, DataSourceContext
from event import EventBus, Event, EventType
from models import LeekComponentConfig, PositionConfig, Signal, SubOrder
from position import PositionContext
from utils import get_logger
logger = get_logger(__name__)

from .base import ComponentManager


class PositionManager(ComponentManager[None, None, PositionConfig]):
    """
    仓位管理器。

    负责统一管理所有仓位上下文（PositionContext），提供仓位的添加、获取、删除等操作接口。
    支持仓位生命周期管理，便于与引擎、风控等模块协同。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[None, PositionConfig]):
        """
        初始化仓位管理器。

        参数：
            event_bus: 事件总线
            config: 仓位配置
        """
        """
        初始化数据管理器。

        参数:
            event_bus: 事件总线
            config: 配置信息
        """
        super().__init__(event_bus, config)
        self.position_context: PositionContext = PositionContext(event_bus, config)

    def process_signal_event(self, event: Event):
        assert isinstance(event.data, Signal)
        self.position_context.process_signal(event.data)

    def process_order_update(self, event: Event):
        assert isinstance(event.data, SubOrder)
        self.position_context.order_update(event.data)

    def on_start(self):
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self.process_signal_event)
        self.event_bus.subscribe_event(EventType.ORDER_UPDATE, self.process_order_update)
        logger.info(f"事件订阅: 仓位管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.ORDER_UPDATE, EventType.STRATEGY_SIGNAL]]}")

    def get(self, instance_id: str) -> PositionContext:
        ...

    def add(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        ...

    def remove(self, instance_id: str):
        ...
