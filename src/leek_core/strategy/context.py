#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略上下文，用于管理策略生命周期。
"""

from decimal import Decimal
from datetime import datetime
import json
from threading import Lock
from typing import Callable, Dict, Any, List, Tuple, Optional

from leek_core.base import LeekContext, create_component, LeekComponent
from leek_core.event import EventBus, EventType, Event, EventSource
from leek_core.info_fabricator import FabricatorContext
from leek_core.models import ExecutionContext, PositionSide, StrategyState, Signal, StrategyConfig, LeekComponentConfig, Data, \
    StrategyInstanceState, Position, Asset, OrderType, StrategyPositionConfig, RiskEventType, RiskEvent
from leek_core.models.ctx import leek_context
from leek_core.sub_strategy import SubStrategy
from leek_core.utils import get_logger
from leek_core.utils import generate_str, thread_lock
from .base import Strategy, StrategyCommand
from .cta import CTAStrategy
 
from leek_core.utils import LeekJSONEncoder

logger = get_logger(__name__)


class StrategyContext(LeekContext):
    """
    策略上下文抽象基类，管理策略生命周期。

    职责:
    1. 管理策略的配置（标的、时间周期、数据源等）
    2. 管理策略调用逻辑
    3. 管理策略的状态
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
        self.strategies: Dict[str, "StrategyWrapper"] = {}

        self.event_bus.subscribe_event(None, self.on_event)

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
            self.strategies[key] = self.create_component(key)
        try:
            r = self.strategies[key].on_data(data)
        except Exception as e:
            logger.error(f"策略{self.name}|{self.instance_id}|{key}:  数据{data}处理异常: {e}", exc_info=True)
            return
        if r:
            self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(self.instance_id, self.name, self.config.cls.__name__, {
                        "class_name": f"{self.config.cls.__module__}|{self.config.cls.__name__}",
                    }),
                    data=self.build_signal(assets=r, data=data, key=key)
                )
            )
        if self.strategies[key].state == StrategyState.STOPPED:
            del self.strategies[key]

    def build_signal(self, assets: List[Asset], data: Data, key) -> Signal:
        """
        构建信号
        """
        return Signal(
            signal_id=generate_str(),
            data_source_instance_id=data.data_source_id,
            strategy_id=self.instance_id,
            strategy_instance_id=key,
            config=self.config.config.strategy_position_config,
            signal_time=datetime.now(),
            assets=assets
        )

    def close_position(self, position: Position):
        """
        处理仓位关闭
        """
        cfg = self.config.config.strategy_position_config or StrategyPositionConfig()
        cfg.order_type = OrderType.MarketOrder
        self.event_bus.publish_event(
                Event(
                    event_type=EventType.STRATEGY_SIGNAL,
                    source=EventSource(self.instance_id, self.name, self.config.cls.__name__, {
                        "class_name": f"{self.config.cls.__module__}|{self.config.cls.__name__}",
                    }),
                    data=Signal(
                        signal_id=generate_str(),
                        data_source_instance_id=0,
                        strategy_id=self.instance_id,
                        strategy_instance_id=position.strategy_instance_id,
                        config=cfg,
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

    def create_component(self, key: str) -> "StrategyWrapper":
        wrapper = StrategyWrapper(
            self.event_bus,
            create_component(self.config.cls, **self.config.config.strategy_config),
            [create_component(c.cls, **c.config) for c in self.config.config.risk_policies or []]
        )
        wrapper.positon_getter = lambda:leek_context.position_tracker.find_position(strategy_id=self.instance_id, strategy_instance_id=key)
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
        logger.info(f"策略{self.instance_id}启动完成, 实例数: {len(self.strategies)} 数据源: {len(self.config.config.data_source_configs)}")
    
    def on_event(self, event: Event):
        """
        接收所有事件，按策略ID和实例ID分发到对应 StrategyWrapper
        """
        # 仅处理与本策略上下文相关的事件
        data = event.data
        strategy_id = getattr(data, 'strategy_id', None)
        if strategy_id is not None and strategy_id != self.instance_id:
            return

        target_instance_id = getattr(data, 'strategy_instance_id', None)
        if target_instance_id is None:
            # 无实例ID的事件，广播给所有实例
            for wrapper in self.strategies.values():
                wrapper.on_event(event)
            return

        wrapper = self.strategies.get(str(target_instance_id))
        if wrapper:
            wrapper.on_event(event)

    def on_stop(self):
        """
        停止策略
        """
        for s in self.strategies.values():
            s.on_stop()
        logger.info(f"策略{self.instance_id}停止, 实例数: {len(self.strategies)}")
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
        self.strategies.clear()
        self.state = StrategyState.STOPPED

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        序列化策略状态
        """
        return {k: json.loads(json.dumps(v.get_state(), cls=LeekJSONEncoder)) for k, v in self.strategies.items()}

    def load_state(self, state: Dict[Tuple, Dict[str, Any]]):
        """
        加载策略状态
        """
        if state is not None and len(state) == 0:
            for cp in self.strategies.values():
                cp.on_stop()
            self.strategies.clear()
            return
        data = state if state else self.config.config.runtime_data
        # 加载运行时数据
        if data is None:
            return
        for k, v in data.items():
            if k not in self.strategies:
                self.strategies[k] = self.create_component(k)
            self.strategies[k].load_state(v)


class StrategyWrapper(LeekComponent):
    """
    管理择时策略实例的生命周期和状态。

    职责:
    1. 管理策略的状态
    2. 管理进出场策略
    """

    def __init__(self, event_bus: EventBus, strategy: Strategy, policies: List[SubStrategy]):
        """
        初始化策略上下文

        参数:
            strategy: 策略实例
        """
        self.event_bus = event_bus
        self.strategy = strategy
        # 已移除进出场子策略
        # 风控策略
        self.policies = policies

        # 策略状态
        self.state = StrategyInstanceState.CREATED
        self.current_command: StrategyCommand = None  # 当前命令
        self.position_rate: Decimal = Decimal("0")

        # self.position: Dict[str, Position] = {}
        self.lock = Lock()
        self.positon_getter = None
    
    # @property
    # def position_rate(self) -> Decimal:
    #     if len(self.position) == 0:
    #         return Decimal("0")
    #     return sum(p.ratio for p in self.position.values())
    
    @property
    def position(self) -> Dict[str, Position]:
        ps = self.positon_getter()
        return {p.position_id: p for p in ps}

    def on_data(self, data: Data = None) -> Optional[List[Asset]]:
        if not self.lock.acquire(blocking=False):
            return None
        try:
            if isinstance(self.strategy, CTAStrategy):
                p_rate = self.position_rate
                old_command = self.current_command
                r = self.on_cta_data(data)
                if r:
                    logger.info(f"策略{self.strategy.display_name} 当前仓位比例: {p_rate} -> {self.position_rate}, 当前命令: {old_command} -> {self.current_command}, 信号: {r}")
                    return [Asset(
                        asset_type=data.asset_type,
                        ins_type=data.ins_type,
                        symbol=data.symbol,
                        quote_currency=data.quote_currency,
                        side=r[0],
                        ratio=min(r[1], Decimal("1")),
                        is_open=r[2],
                        price=data.close,
                    )]
                return None

            raise ValueError("strategy must be process")
        finally:
            self.lock.release()
    
    def on_signal_finish(self, signal: Signal):
        """
        处理信号完成
        """
        self.current_command = None
        for asset in signal.assets:
            if asset.is_open:
                self.position_rate += asset.actual_ratio
            else:
                self.position_rate -= asset.actual_ratio
        # 进出场状态转换：将 ENTERING/EXITING 归一到正常态
        if self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
        elif self.position_rate > 0:
            self.state = StrategyInstanceState.HOLDING
        
        logger.info(f"策略信号处理完成: {signal.signal_id}, {self.state}, 当前仓位比例: {self.position_rate}")


    def on_event(self, event: Event):
        # 先将常见事件进行内置处理，再转交策略
        if event.event_type == EventType.STRATEGY_SIGNAL_FINISH and isinstance(event.data, Signal):
            self.on_signal_finish(event.data)

        # 将事件转交给策略
        try:
            self.strategy.on_event(event)
        except Exception as e:
            logger.error(f"策略事件处理异常: {e}", exc_info=True)

    def on_cta_data(self, data: Data = None) -> (PositionSide, Decimal, bool):
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
        if data.get("history_data", False):
            return None
        logger.debug(f"{self.strategy.display_name} 当前状态: {self.state}, 仓位[{self.position_rate}]: "
                    f"{['%s:%s-%s:%s, %s, %s' % (p.position_id, p.symbol, p.quote_currency, p.side, p.ratio, p.sz) for p in list(self.position.values())]}")
        for pos in list(self.position.values()):
            for p in self.policies:
                if p.evaluate(data, pos):
                    continue
                try:
                    logger.info(f"仓位风控策略执行完成: {p.display_name}, 触发策略{self.strategy.display_name}仓位清理, 当前仓位比例: {self.position_rate}， 仓位: {pos}")
                    self.current_command = StrategyCommand(pos.side.switch(), Decimal("1"))
                    self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
                    self._publish_embedded_risk_event(pos, p, data)
                    return self.current_command.side, min(self.current_command.ratio, self.position_rate), False
                finally:
                    try:
                        logger.info(f"仓位风控策略执行完成, 触发策略{self.strategy.display_name}仓位清理")
                        self.strategy.after_risk_control()
                    except Exception as e:
                        logger.error(f"仓位风控策略执行完成, 触发策略{self.strategy.display_name}仓位清理失败: {e}", exc_info=True)
        if self.state == StrategyInstanceState.READY:  # 无仓位
            return self.ready_handler()
        elif self.state == StrategyInstanceState.ENTERING:
            return
        elif self.state == StrategyInstanceState.HOLDING:
            return self.holding_handler()
        elif self.state == StrategyInstanceState.EXITING:
            return
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
        self.current_command = res
        ratio = min(self.current_command.ratio, 1 - self.position_rate)
        if ratio <= 0:
            return
        return self.current_command.side, ratio, True

    def holding_handler(self):
        if len(self.position) == 0:
            return
        res = self.strategy.close(list(self.position.values())[0])
        if res is not None and res is not False:
            if isinstance(res, bool):
                res = Decimal("1")
            if not isinstance(res, Decimal):
                raise ValueError("should_close return value must be Decimal or bool")
            self.state = StrategyInstanceState.EXITING
            self.current_command = StrategyCommand(list(self.position.values())[0].side.switch(), res)
            return self.current_command.side, min(self.current_command.ratio, self.position_rate), False

        # 策略要求重复开仓
        if self.strategy.open_just_no_pos is False:
            return self.ready_handler()

    def stopping_handler(self):
        if self.state == StrategyInstanceState.READY or self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED
            return

        # 直接按剩余仓位一次性退出
        if self.position_rate > 0:
            position_change = self.position_rate
            self.state = StrategyInstanceState.STOPPED
            return self.current_command.side.switch(), position_change, False
        self.state = StrategyInstanceState.STOPPED
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        获取策略上下文状态

        返回:
            状态字典
        """
        state = {
            "strategy_state": self.strategy.get_state(),
            "state": self.state,
            "position_rate": self.position_rate,
            "current_command": {
                "side": self.current_command.side,
                "ratio": self.current_command.ratio,
            } if self.current_command else None,
            "position": [json.loads(json.dumps(p, cls=LeekJSONEncoder)) for p in self.position.values()]
        }
        return json.loads(json.dumps(state, cls=LeekJSONEncoder))

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载策略上下文状态

        参数:
            state: 状态字典
        """
        self.state = StrategyInstanceState(state.get("state", self.state))
        self.position_rate = Decimal(state.get("position_rate", self.position_rate))
        current_command = state.get("current_command", {})
        if current_command and "side" in state.get("current_command", {}) and "ratio" in state.get("current_command", {}):
            self.current_command = StrategyCommand(
                side=PositionSide(state["current_command"]["side"]),
                ratio=Decimal(state["current_command"]["ratio"])
            )
        logger.info(f"加载策略{self.strategy.display_name}状态: {state.get('strategy_state')}")
        self.strategy.load_state(state.get("strategy_state", {}))
    
    def on_start(self):
        """
        启动组件
        """
        self.strategy.on_start()
        if self.state == StrategyInstanceState.CREATED:
            self.state = StrategyInstanceState.READY

    def on_stop(self):
        """
        停止组件
        """
        self.strategy.on_stop()

    def _publish_embedded_risk_event(self, position: Position, policy: SubStrategy, data: Data):
        """
        发布内嵌风控事件
        
        Args:
            position: 触发风控的仓位
            policy_results: 风控策略评估结果
            data: 触发的数据
        """
        try:
            # 创建风控事件数据
            data = RiskEvent(
                risk_type=RiskEventType.EMBEDDED,
                strategy_id=position.strategy_id,
                strategy_instance_id=position.strategy_instance_id,
                strategy_class_name=f"{self.strategy.__class__.__module__}|{self.strategy.__class__.__name__}",
                risk_policy_id=0,
                risk_policy_class_name=f"{policy.display_name}",
                trigger_time=datetime.now(),
                trigger_reason=f"「{policy.display_name}」触发平仓",
                signal_id=None,
                execution_order_id=None,
                position_id=position.position_id,
                original_amount=position.amount,
                pnl=None,
                extra_info={
                    "position_symbol": position.symbol,
                    "position_quote_currency": position.quote_currency,
                    "position_ins_type": position.ins_type,
                    "position_side": position.side.value,
                    "position_ratio": str(self.position_rate),
                    "position_pnl": str(position.pnl) if position.pnl else "0",
                },
            )
            # 发布风控触发事件
            event = Event(
                event_type=EventType.RISK_TRIGGERED,
                data=data,
                source=EventSource(
                    instance_id=position.strategy_id,
                    name=self.strategy.display_name,
                    cls=f"{policy.__class__.__module__}|{policy.__class__.__name__}"
                )
            )
            self.event_bus.publish_event(event)
        except Exception as e:
            logger.error(f"发布内嵌风控事件失败: {e}", exc_info=True)
