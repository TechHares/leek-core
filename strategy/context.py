#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略上下文，用于管理策略生命周期。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Type, Tuple, List

from models import Component, Signal
from models.constants import DataType, PositionSide, StrategyState
from strategy import Strategy
from strategy.sidecar import CTAStrategySidecar
from sub_strategy import EnterStrategy, ExitStrategy
from utils import get_logger
from utils import EventBus, EventType, Event, EventSource

logger = get_logger(__name__)


@dataclass
class StrategyConfig:
    """
    策略配置信息
    """
    instance_id: str
    name: str
    strategy_cls: Type[Strategy]
    strategy_config: Dict[str, Any] = None
    data_config: Dict[str, Any] = None
    enter_strategy_cls: Type[EnterStrategy] = None
    enter_strategy_config: Dict[str, Any] = None
    exit_strategy_cls: Type[ExitStrategy] = None
    exit_strategy_config: Dict[str, Any] = None
    risk_policies: List[Tuple[Type[ExitStrategy], Dict[str, Any]]] = field(default_factory=list)


class StrategyContext(Component, ABC):
    """
    策略上下文抽象基类，管理策略生命周期。

    职责:
    1. 管理策略的配置（标的、时间周期、数据源等）
    2. 过滤进入策略的数据
    3. 管理策略的状态
    4. 管理进出场策略
    """

    def __init__(self, event_bus: EventBus, config: StrategyConfig):
        """
        初始化策略上下文

        参数:
            config: 策略配置
        """
        super().__init__(config.instance_id, config.name)
        self.state = StrategyState.CREATED
        self.config = config
        self.event_bus = event_bus
        self.strategy_mode = config.strategy_cls.strategy_mode

        self.strategies: Dict[str, CTAStrategySidecar] = {}

    def on_data_event(self, event: Event):
        """
        处理数据，更新策略状态
        """
        if self.state in [StrategyState.STOPPED, StrategyState.CREATED, StrategyState.PREPARING]:
            return
        data_source_id = event.source.instance_id
        if data_source_id not in self.config.data_config.get("data_source_ids", []):
            return

        data_type = event.data.data_type
        data = event.data
        if data_type not in self.config.strategy_cls.accepted_data_types:
            return
        if ext_data := event.source.extra is None:
            return
        if params := ext_data.get("params", []):
            for param in params:  # 检查参数是否符合要求
                if data.get(param) is None or data.get(param) not in self.config.data_config.get(param, []):
                    return

        key = self.strategy_mode.build_instance_key(data)

        if key not in self.strategies:
            self.strategies[key] = self._create_strategy_instance()
        r = self.strategies[key].on_data(data)
        if r:
            self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(self.instance_id, self.name, self.config.strategy_cls.__name__, {}),
                    data=self.build_signal(r[0], r[1])
                )
            )
        if self.strategies[key].state == StrategyState.STOPPED:
            del self.strategies[key]

    def build_signal(self, side: PositionSide, rate: Decimal) -> Signal:
        """
        构建信号
        """
        # todo
        return None

    def _create_strategy_instance(self) -> CTAStrategySidecar:
        strategy = self.config.strategy_cls(**self.config.strategy_config)
        Strategy.__init__(strategy, self.instance_id, self.name)

        enter_strategy = self.config.enter_strategy_cls(**self.config.enter_strategy_config)
        EnterStrategy.__init__(enter_strategy)

        exit_strategy = self.config.exit_strategy_cls(**self.config.exit_strategy_config)
        ExitStrategy.__init__(exit_strategy)

        return CTAStrategySidecar(self.event_bus, strategy, enter_strategy, exit_strategy)

    def start(self):
        """
        启动策略
        """
        ...

    def stop(self):
        """
        停止策略
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        序列化策略状态
        """
        ...

    def load_state(self, state: Dict[str, Any]):
        """
        加载策略状态
        """
        ...
