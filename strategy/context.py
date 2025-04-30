#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略上下文，用于管理策略生命周期。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Type, Tuple, List, Callable

from base import LeekContext, create_component, LeekComponent
from models import DataType, PositionSide, StrategyState, Signal, StrategyConfig, LeekComponentConfig, Data, \
    StrategyInstanceState, Position
from policy import PositionPolicy
from . import STAStrategy
from .base import Strategy, StrategyCommand
from sub_strategy import EnterStrategy, ExitStrategy
from utils import get_logger
from event import EventBus, EventType, Event, EventSource

logger = get_logger(__name__)


class StrategyContext(LeekContext):
    """
    策略上下文抽象基类，管理策略生命周期。

    职责:
    1. 管理策略的配置（标的、时间周期、数据源等）
    2. 管理策略调用逻辑
    3. 管理策略的状态
    4. 管理进出场策略
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[Strategy, StrategyConfig]):
        """
        初始化策略上下文

        参数:
            config: 策略配置
        """
        super().__init__(event_bus, config)
        self.state = StrategyState.CREATED
        self.config = config
        self.strategy_mode = config.cls.strategy_mode

        self.strategies: Dict[tuple, "StrategyWrapper"] = {}

    def on_data_event(self, event: Event):
        """
        处理数据，更新策略状态
        """
        if self.state in [StrategyState.STOPPED, StrategyState.CREATED, StrategyState.PREPARING]:
            return
        data_source_id = event.source.instance_id
        data_type = event.data.data_type
        data = event.data
        key = self.strategy_mode.build_instance_key(data)

        if key not in self.strategies:
            self.strategies[key] = self.create_component()
        r = self.strategies[key].on_data(data)
        if r:
            self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(self.instance_id, self.name, self.config.strategy_cls.__name__, {}),
                    data=self.build_signal(r[0], r[1], data)
                )
            )
        if self.strategies[key].state == StrategyState.STOPPED:
            del self.strategies[key]

    def build_signal(self, side: PositionSide, rate: Decimal, data: Data) -> Signal:
        """
        构建信号
        """
        return Signal(
            data_source_instance_id=data.data_source_instance_id,
            strategy_instance_id=self.instance_id,
            asset_type=data.asset_type,
            ins_type=data.ins_type,
            symbol=data.symbol,
            config=self.config.config.strategy_position_config,
            quote_currency=data.quote_currency,
            side=side,
            ratio=rate,
            price=data.close,
            signal_time=data.timestamp,
        )

    def create_component(self) -> "StrategyWrapper":
        wrapper = StrategyWrapper(create_component(self.config.cls, **self.config.config.strategy_config),
                               create_component(self.config.config.enter_strategy_cls,
                                                **self.config.config.enter_strategy_config),
                               create_component(self.config.config.exit_strategy_cls,
                                                **self.config.config.exit_strategy_config),
                               [create_component(c.cls, **c.config) for c in self.config.config.risk_policies]
                               )
        wrapper.on_start()
        return wrapper

    def on_start(self):
        """
        启动策略
        """
        self.state = StrategyState.PREPARING
        for s in self.strategies.values():
            s.on_start()
        self.state = StrategyState.RUNNING

    def on_stop(self):
        """
        停止策略
        """
        for s in self.strategies.values():
            s.on_stop()

        self.state = StrategyState.STOPPED

    def get_state(self) -> Dict[str, Any]:
        """
        序列化策略状态
        """
        return {
            "strategies": {k:v.get_state() for k,v in self.strategies.items()},
        }

    def load_state(self, state: Dict[str, Any]):
        """
        加载策略状态
        """
        if "strategies" not in state:
            return
        for k, v in state["strategies"].items():
            if k not in self.strategies:
                self.strategies[k] = self.create_component()
            self.strategies[k].load_state(v)


class StrategyWrapper(LeekComponent):
    """
    管理择时策略实例的生命周期和状态。

    职责:
    1. 管理策略的状态
    2. 管理进出场策略
    """

    def __init__(self, strategy: Strategy, enter_strategy: EnterStrategy, exit_strategy: ExitStrategy,
                 policies: List[PositionPolicy]):
        """
        初始化策略上下文

        参数:
            strategy: 策略实例
            enter_strategy: 进场策略
            exit_strategy: 出场策略
        """
        self.strategy = strategy
        # 进出场策略
        self.enter_strategy = enter_strategy
        self.exit_strategy = exit_strategy
        # 风控策略
        self.policies = policies

        # 策略状态
        self.state = StrategyInstanceState.CREATED
        self.position_rate: Decimal = Decimal("0")  # 仓位比例，0-1之间的小数
        self.current_position_rate: Decimal = Decimal("0")  # 进/出场状态中需要变化的比例
        self.current_position_side: PositionSide | None = None

        self.position: Position | None = None

    def on_data(self, data: Data = None) -> (PositionSide, Decimal):
        """
        处理数据，更新进出场策略状态

        参数:
            data: 接收到的数据
            data_type: 数据类型
        """
        if self.state in [StrategyInstanceState.STOPPED, StrategyInstanceState.CREATED]:
            return
        # 更新计算信息
        self.strategy.on_data(data)
        self.enter_strategy.on_data(data)
        self.exit_strategy.on_data(data)
        if self.position:
            res = [p.evaluate(data, self.position) for p in self.policies]
            if not all(res):
                self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
                return self.current_position_side.switch(), Decimal("1")

        if self.state == StrategyInstanceState.READY:  # 无仓位
            return self.ready_handler()
        elif self.state == StrategyInstanceState.ENTERING:
            return self.entering_handler()
        elif self.state == StrategyInstanceState.HOLDING:
            return self.holding_handler()
        elif self.state == StrategyInstanceState.EXITING:
            return self.exiting_handler()
        elif self.state == StrategyInstanceState.STOPPING:
            return self.stopping_handler()

    def ready_handler(self):
        res = self.strategy.should_open()
        if res is None:
            return
        if isinstance(res, PositionSide):
            res = StrategyCommand(res, Decimal("1"))

        if not isinstance(res, StrategyCommand):
            raise ValueError("should_open return value must be StrategyCommand or PositionSide")

        self.state = StrategyInstanceState.ENTERING
        self.current_position_side = res.side
        self.current_position_rate = res.ratio
        return self.entering_handler()

    def entering_handler(self):
        rt = self.enter_strategy.ratio(self.current_position_side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate += position_change
            if self.enter_strategy.is_finished:
                self.state = StrategyInstanceState.HOLDING
                self.enter_strategy.reset()
            return self.current_position_side, position_change

    def holding_handler(self):
        res = self.strategy.should_close(self.current_position_side)
        if res is not None and res is not False:
            if isinstance(res, bool):
                res = Decimal("1")
            if not isinstance(res, Decimal):
                raise ValueError("should_close return value must be Decimal or bool")
            self.state = StrategyInstanceState.EXITING
            self.current_position_rate = res
            return self.entering_handler()

        # 策略要求重复开仓
        if self.strategy.open_just_no_pos is False:
            return self.ready_handler()

    def exiting_handler(self):
        rt = self.exit_strategy.ratio(self.current_position_side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate -= position_change
            if self.enter_strategy.is_finished:
                self.state = StrategyInstanceState.HOLDING if self.position_rate > 0 else StrategyInstanceState.READY
                self.enter_strategy.reset()
            return self.current_position_side.switch(), position_change

    def stopping_handler(self):
        if self.state == StrategyInstanceState.READY or self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED
            return

        res = self.exiting_handler()
        if res is not None and self.state == StrategyInstanceState.READY:
            self.state = StrategyInstanceState.STOPPED
        return res

    def get_state(self) -> Dict[str, Any]:
        """
        获取策略上下文状态

        返回:
            状态字典
        """
        return {
            "strategy": self.strategy.get_state(),
            "state": self.state,
            "position_rate": str(self.position_rate),
            "current_position_rate": str(self.current_position_rate),
            "current_position_side": self.current_position_side,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载策略上下文状态

        参数:
            state: 状态字典
        """
        self.strategy.load_state(state.get("strategy", {}))
        self.state = state.get("state", self.state)
        self.position_rate = Decimal(state.get("position_rate", str(self.position_rate)))
        self.current_position_rate = Decimal(state.get("current_position_rate", str(self.current_position_rate)))
        self.current_position_side = state.get("current_position_side", self.current_position_side)

    def start(self):
        """
        启动组件
        """
        self.strategy.on_start()

    def stop(self):
        """
        停止组件
        """
        self.strategy.on_stop()
