#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import Dict, Any

from leek_core.data import DataSource
from leek_core.event import EventBus, Event, EventType
from leek_core.models import LeekComponentConfig, PositionConfig, Signal, Order, ExecutionContext
from leek_core.models import Position, Data
from leek_core.position import PositionContext
from leek_core.policy import StrategyPolicy
from leek_core.utils import get_logger, LeekJSONEncoder
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
        self.position_context.process_signal(event.data, event.source.extra.get("class_name") if event.source and event.source.extra else None)

    def process_order_update(self, event: Event):
        assert isinstance(event.data, Order)
        self.position_context.order_update(event.data)

    def process_data_update(self, event: Event):
        assert isinstance(event.data, Data)
        self.position_context.on_data(event.data)

    def process_exec_update(self, event: Event):
        assert isinstance(event.data, ExecutionContext)
        self.position_context.exec_update(event.data)

    def on_start(self):
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self.process_signal_event)
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, self.process_order_update)
        self.event_bus.subscribe_event(EventType.DATA_RECEIVED, self.process_data_update)
        self.event_bus.subscribe_event(EventType.EXEC_ORDER_UPDATED, self.process_exec_update)
        logger.info(f"事件订阅: 仓位管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.ORDER_UPDATED, EventType.STRATEGY_SIGNAL, EventType.DATA_RECEIVED]]}")
    
    def update(self, config: LeekComponentConfig[None, PositionConfig]):
        self.position_context.update_config(config)
        if config.data:
            self.position_context.load_state(config.data)

    def get(self, instance_id: str) -> PositionContext:
        ...

    def add(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        ...

    def remove(self, instance_id: str):
        ...
    
    def add_policy(self, config: LeekComponentConfig[StrategyPolicy, Dict[str, Any]]):
        self.position_context.add_policy(config)
    
    def remove_policy(self, instance_id: str):
        self.position_context.remove_policy(instance_id)

    def get_state(self) -> dict:
        return json.loads(json.dumps(self.position_context.get_state(), cls=LeekJSONEncoder))
    
    def load_state(self, state: dict):
        self.position_context.load_state(state)

    def get_position(self, position_id: str) -> Position:
        return self.position_context.get_position(str(position_id))
    
    def reset_position_state(self):
        self.position_context.load_state({"reset_position_state": True})
        return self.get_state()
