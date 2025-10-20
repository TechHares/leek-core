#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎策略管理子模块，给引擎提供策略组件管理相关功能
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Generator

from leek_core.event.types import EventSource
from leek_core.manager import ComponentManager
from leek_core.event import Event, EventType, EventBus
from leek_core.models import StrategyState, LeekComponentConfig, Position, StrategyConfig, StrategyPositionConfig, \
    OrderType, Asset, Signal, Data, ExecutionContext
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

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[StrategyContext, None], max_workers: int=0):
        """
        初始化策略管理器。

        参数:
            event_bus: 事件总线
        """
        super().__init__(event_bus, config)
        self.event_bus = event_bus
        self.executor = None
        if max_workers > 0:
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

    def exec_update(self, strategy_id: str, execution_context: ExecutionContext | str):
        strategy_ctx = self.get(strategy_id)
        if strategy_ctx:
            strategy_ctx.exec_update(execution_context)

    def process_data(self, data: Data) -> Generator[Signal, None, None]:
        """
        处理数据
        参数:
            data: 数据
        """
        del_list = [instance_id for instance_id, context in self.components.items() if
                    context.state == StrategyState.STOPPED]
        for instance_id in del_list:
            self.remove(instance_id)

        futures = []
        if self.executor: # 使用线程池执行
            for instance_id in data.target_instance_id:
                if instance_id in self.components:
                    strategy_context = self.components[instance_id]
                    future = self.executor.submit(strategy_context.on_data, deepcopy(data))
                    futures.append(future)
            for future in as_completed(futures):
                yield future.result()
        else:  # 使用单线程执行 用于debug backtest等场景
            for instance_id in data.target_instance_id:
                if instance_id in self.components:
                    strategy_context = self.components[instance_id]
                    yield strategy_context.on_data(data)

    def close_position(self, position: Position):
        strategy_context = self.get(position.strategy_id)
        if strategy_context is None:
            logger.info(f"策略上下文不存在: {position.strategy_id}， 直接平仓{position.position_id}")
            return Signal(
                        signal_id=generate_str(),
                        data_source_instance_id="0",
                        strategy_id=self.instance_id,
                        strategy_instance_id=position.strategy_instance_id,
                        strategy_cls=f"",
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
        return strategy_context.close_position(position)
    
    def check_component(self, instance_id: str = None):
        del_list = [inst_id for inst_id, context in self.components.items() if
                    context.state == StrategyState.STOPPED]
        for inst_id in del_list:
            self.remove(inst_id)
        return super().check_component(instance_id)


if __name__ == '__main__':
    pass
