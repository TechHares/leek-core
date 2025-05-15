#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎策略管理子模块，给引擎提供策略组件管理相关功能
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, Any

from leek_core.manager import ComponentManager
from leek_core.event import Event, EventType, EventBus
from leek_core.models import StrategyState, LeekComponentConfig, Position
from leek_core.strategy import StrategyContext, Strategy
from leek_core.utils import get_logger
logger = get_logger(__name__)

class StrategyManager(ComponentManager[StrategyContext, Strategy, Dict[str, Any]]):
    """
    管理多个策略上下文（StrategyContext）并提供统一接口。

    该类允许应用程序以一致的方式与多个策略上下文交互，
    处理策略的生命周期、信号分发。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[StrategyContext, None], max_workers: int=5):
        """
        初始化策略管理器。

        参数:
            event_bus: 事件总线
        """
        super().__init__(event_bus, config)
        self.event_bus = event_bus
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="StrategyManager-")

    def on_data_event(self, event: Event):
        """
        处理数据
        参数:
            data: 数据
        """
        del_list = [instance_id for instance_id, context in self.components.items() if context.state == StrategyState.STOPPED]
        for instance_id in del_list:
            del self.components[instance_id]
        data = event.data
        tasks = []
        for instance_id in data.target_instance_id:
            if instance_id in self.components:
                strategy_context = self.components[instance_id]
                tasks.append(strategy_context.on_data)
        if len(tasks) == 0:
            return
        if len(tasks) == 1:
            self.executor.submit(tasks[0](data))
            return
        for task in tasks:
            self.executor.submit(task, deepcopy(data))

    def on_position_update_event(self, event: Event):
        assert isinstance(event.data, Position)
        strategy_context = self.get(event.data.strategy_id)
        if strategy_context is None:
            return
        strategy_context.on_position_update(event.data)

    def on_start(self):
        self.event_bus.subscribe_event(EventType.DATA_RESPONSE, self.on_data_event)
        self.event_bus.subscribe_event(EventType.DATA_RECEIVED, self.on_data_event)
        logger.info(
            f"事件订阅: 策略管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.DATA_RESPONSE, EventType.DATA_RECEIVED]]}")


if __name__ == '__main__':
    pass
