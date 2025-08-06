#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎策略管理子模块，给引擎提供策略组件管理相关功能
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

from leek_core.event.types import EventSource
from leek_core.manager import ComponentManager
from leek_core.event import Event, EventType, EventBus
from leek_core.models import StrategyState, LeekComponentConfig, Position, StrategyConfig, StrategyPositionConfig, OrderType, Asset, Signal
from leek_core.strategy import StrategyContext, Strategy
from leek_core.utils import get_logger
from leek_core.utils.id_generator import generate_str
logger = get_logger(__name__)

class StrategyManager(ComponentManager[StrategyContext, Strategy, StrategyConfig]):
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

    def update(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        """
        更新指定实例。
        """
        ins_id = str(config.instance_id)
        strategy_ctx = self.components.pop(ins_id, None)
        if strategy_ctx:
            state = strategy_ctx.get_state()
            config.config.runtime_data = state
            strategy_ctx.on_stop()
        self.add(config)
    
    def update_state(self, instance_id: str, state: Dict):
        """
        更新指定实例。
        """
        strategy_ctx = self.get(instance_id)
        if strategy_ctx:
            strategy_ctx.load_state(state)
    
    def clear_state(self, strategy_id: str, instance_id: str):
        strategy_ctx = self.get(strategy_id)
        if strategy_ctx:
            instance = strategy_ctx.strategies.pop(instance_id)
            if instance:
                instance.on_stop()

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

    def close_position(self, position: Position):
        strategy_context = self.get(position.strategy_id)
        if strategy_context is None:
            logger.info(f"策略上下文不存在: {position.strategy_id}， 直接平仓{position.position_id}")
            self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(position.strategy_id, self.name, self.__class__.__name__, {
                        "class_name": f"{self.__class__.__module__}|{self.__class__.__name__}",
                    }),
                    data=Signal(
                        signal_id=generate_str(),
                        data_source_instance_id=0,
                        strategy_id=self.instance_id,
                        strategy_instance_id=position.strategy_instance_id,
                        config=StrategyPositionConfig(order_type=OrderType.MarketOrder),
                        signal_time=datetime.now(),
                        assets=[Asset(
                            asset_type=position.asset_type,
                            ins_type=position.ins_type,
                            symbol=position.symbol,
                            quote_currency=position.quote_currency,
                            side=position.side.switch(),
                            ratio=Decimal("1"),
                            price=position.current_price,
                        )]
                    )
                )
            )
            return
        strategy_context.close_position(position)

    def on_position_update_event(self, event: Event):
        assert isinstance(event.data, Position)
        strategy_context = self.get(event.data.strategy_id)
        if strategy_context is None:
            return
        strategy_context.on_position_update(event.data)
    
    def on_signal_rollback_event(self, event: Event):
        assert isinstance(event.data, Signal)
        strategy_context = self.get(event.data.strategy_id)
        if strategy_context is None:
            return
        logger.info(f"策略回滚: {event.data.strategy_id}@{event.data.strategy_instance_id} 回滚信号: {event.data.signal_id}")
        strategy_context.on_signal_rollback(event.data)

    def on_start(self):
        self.event_bus.subscribe_event(EventType.DATA_RESPONSE, self.on_data_event)
        self.event_bus.subscribe_event(EventType.DATA_RECEIVED, self.on_data_event)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.on_position_update_event)
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL_ROLLBACK, self.on_signal_rollback_event)
        logger.info(
            f"事件订阅: 策略管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.DATA_RESPONSE, EventType.DATA_RECEIVED]]}")
        
    def on_stop(self):
        self.event_bus.unsubscribe_event(EventType.DATA_RESPONSE, self.on_data_event)
        self.event_bus.unsubscribe_event(EventType.DATA_RECEIVED, self.on_data_event)
        self.event_bus.unsubscribe_event(EventType.DATA_RECEIVED, self.on_position_update_event)
        self.event_bus.unsubscribe_event(EventType.STRATEGY_SIGNAL_ROLLBACK, self.on_signal_rollback_event)
        self.executor.shutdown(wait=True)
        super().on_stop()


if __name__ == '__main__':
    pass
