#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略上下文，用于管理策略生命周期。
"""

from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional

from leek_core.base import LeekContext, create_component, LeekComponent
from leek_core.event import EventBus, EventType, Event, EventSource
from leek_core.info_fabricator import FabricatorContext
from leek_core.models import PositionSide, StrategyState, Signal, StrategyConfig, LeekComponentConfig, Data, \
    StrategyInstanceState, Position, Asset
from leek_core.policy import PositionPolicy
from leek_core.sub_strategy import EnterStrategy, ExitStrategy
from leek_core.utils import get_logger
from .base import Strategy, StrategyCommand
from .cta import CTAStrategy

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
        self.info_fabricators = FabricatorContext(self.process_data, event_bus,
                                                       LeekComponentConfig(
                                                           instance_id=self.instance_id,
                                                           name=self.name,
                                                           cls=None,
                                                           config=config.config.info_fabricator_configs
                                                       ))
        self.strategies: Dict[tuple, "StrategyWrapper"] = {}

    def on_data(self, data: Data):
        try:
            self.info_fabricators.on_data(data)
        except Exception as e:
            logger.error(f"info_fabricator {self.instance_id} process error: {e}", exc_info=True)

    def process_data(self, data: Data):
        """
        处理数据，更新策略状态
        """
        if self.state in [StrategyState.STOPPED, StrategyState.CREATED, StrategyState.PREPARING]:
            return
        key = self.strategy_mode.build_instance_key(data)

        if key not in self.strategies:
            self.strategies[key] = self.create_component()
        r = self.strategies[key].on_data(data)
        if r:
            self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(self.instance_id, self.name, self.config.strategy_cls.__name__, {}),
                    data=self.build_signal(assets=r, data=data)
                )
            )
        if self.strategies[key].state == StrategyState.STOPPED:
            del self.strategies[key]

    def build_signal(self, assets: List[Asset], data: Data) -> Signal:
        """
        构建信号
        """
        return Signal(
            data_source_instance_id=data.data_source_instance_id,
            strategy_instance_id=self.instance_id,
            config=self.config.config.strategy_position_config,
            signal_time=data.timestamp,
            assets=assets
        )

    def  on_position_update(self, position: Position):
        """
        处理仓位更新
        """
        ...



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
        self.load_state(self.config.config.runtime_data)
        for s in self.strategies.values():
            s.on_start()
        self.state = StrategyState.RUNNING
        for ds in self.config.config.data_source_configs:
            self.event_bus.publish_event(Event(
                event_type=EventType.DATA_SOURCE_SUBSCRIBE,
                data=ds.config,
                source=EventSource(
                    instance_id=self.instance_id,
                    name=self.name,
                    cls=self.config.cls.__name__,
                    extra={"data_source_id": ds.instance_id}
                )
            ))

    def on_stop(self):
        """
        停止策略
        """
        for s in self.strategies.values():
            s.on_stop()

        for ds in self.config.config.data_source_configs:
            self.event_bus.publish_event(Event(
                event_type=EventType.DATA_SOURCE_UNSUBSCRIBE,
                data=ds.config,
                source=EventSource(
                    instance_id=self.instance_id,
                    name=self.name,
                    cls=self.config.cls.__name__,
                    extra={"data_source_id": ds.instance_id}
                )
            ))
        self.state = StrategyState.STOPPED

    def get_state(self) -> Dict[Tuple, Dict[str, Any]]:
        """
        序列化策略状态
        """
        return {k:v.get_state() for k,v in self.strategies.items()}

    def load_state(self, state: Dict[Tuple, Dict[str, Any]]):
        """
        加载策略状态
        """
        data = self.config.config.runtime_data
        # 加载运行时数据
        if data is None:
            return
        for k, v in data.items():
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

        self.position: List[Position] = []

    def on_data(self, data: Data = None) -> Optional[List[Asset]]:
        if isinstance(self.strategy, CTAStrategy):
            r = self.on_cta_data(data)
            if r:
                return [Asset(
                    asset_type=data.asset_type,
                    ins_type=data.ins_type,
                    symbol=data.symbol,
                    quote_currency=data.quote_currency,
                    side=r[0],
                    ratio=r[1],
                    price=data.close,
                )]
            return None

        raise ValueError("strategy must be process")

    def on_cta_data(self, data: Data = None) -> (PositionSide, Decimal):
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
        if data.get("history_data", False):
            return None
        if len(self.position) > 0:
            res = [p.evaluate(data, self.position[0]) for p in self.policies]
            if not all(res):
                self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
                return self.position[0].side.switch(), Decimal("1")

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
        self.current_position_rate = res.ratio
        return self.entering_handler()

    def entering_handler(self):
        rt = self.enter_strategy.ratio(self.position[0].side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate += position_change
            if self.enter_strategy.is_finished:
                self.state = StrategyInstanceState.HOLDING
                self.enter_strategy.reset()
            return self.position[0].side, position_change

    def holding_handler(self):
        res = self.strategy.should_close(self.position[0].side)
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
        rt = self.exit_strategy.ratio(self.position[0].side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate -= position_change
            if self.enter_strategy.is_finished:
                self.state = StrategyInstanceState.HOLDING if self.position_rate > 0 else StrategyInstanceState.READY
                self.enter_strategy.reset()
            return self.position[0].side.switch(), position_change

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
            "strategy_state": self.strategy.get_state(),
            "state": self.state.value,
            "position_rate": str(self.position_rate),
            "current_position_rate": str(self.current_position_rate),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载策略上下文状态

        参数:
            state: 状态字典
        """
        self.strategy.load_state(state.get("strategy_state", {}))
        self.state = StrategyInstanceState(state.get("state", self.state))
        self.position_rate = Decimal(state.get("position_rate", str(self.position_rate)))
        self.current_position_rate = Decimal(state.get("current_position_rate", str(self.current_position_rate)))
        position = state.get("position", [])
        assert len(position) < 1 or all(isinstance(p, Position) for p in position), "position must be Position"
        self.position = position

    def on_start(self):
        """
        启动组件
        """
        self.strategy.on_start()
        self.state = StrategyInstanceState.READY

    def on_stop(self):
        """
        停止组件
        """
        self.strategy.on_stop()
