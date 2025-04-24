#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎策略管理子模块，给引擎提供策略组件管理相关功能
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict

from utils import Event, EventType, EventBus
from models import StrategyState, Component
from strategy import StrategyContext



class StrategyManager(Component):
    """
    管理多个策略上下文（StrategyContext）并提供统一接口。

    该类允许应用程序以一致的方式与多个策略上下文交互，
    处理策略的生命周期、信号分发。
    """

    def __init__(self, instance_id, name, event_bus: EventBus, max_workers=5):
        """
        初始化策略管理器。

        参数:
            event_bus: 事件总线
        """
        super().__init__(instance_id, name)
        self.strategy_contexts: Dict[str, StrategyContext] = {}
        self.event_bus = event_bus
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="StrategyManager-")

    def on_data_event(self, data: Event):
        """
        处理数据
        参数:
            data: 数据
        """
        del_list = [instance_id for instance_id, context in self.strategy_contexts.items() if context.state == StrategyState.STOPPED]
        for instance_id in del_list:
            del self.strategy_contexts[instance_id]

        for context in self.strategy_contexts.values():
            self.executor.submit(context.on_data_event, deepcopy(data))

    def add_strategy_context(self, context: StrategyContext):
        """
        向管理器添加策略上下文。

        参数:
            context: 要添加的策略上下文实例
        """
        if context.instance_id in self.strategy_contexts:
            return
        self.strategy_contexts[context.instance_id] = context
        context.on_start()

    def remove_strategy_context(self, instance_id: str):
        """
        从管理器中移除策略上下文。

        参数:
            instance_id: 要移除的策略上下文实例ID
        """
        if instance_id not in self.strategy_contexts:
            return
        context = self.strategy_contexts[instance_id]
        context.on_stop()
        del self.strategy_contexts[instance_id]

    def destroy(self):
        """
        停止服务
        """
        for context in self.strategy_contexts.values():
            context.on_stop()
        self.strategy_contexts.clear()

if __name__ == '__main__':
    pass
