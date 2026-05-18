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
    StrategyInstanceState, Position, Asset, Order, OrderStatus, OrderType, StrategyPositionConfig, RiskEventType, RiskEvent
from leek_core.models.ctx import leek_context
from leek_core.sub_strategy import SubStrategy
from leek_core.utils import get_logger
from leek_core.utils import generate_str, thread_lock
from .base import Strategy, StrategyCommand, CancelCommand, StrategyAction
from .cta import CTAStrategy
from .intent import PendingOrderIndex, diff_intents, intent_key, IntentPatch

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
        self.info_fabricators = FabricatorContext(event_bus,
                                                       LeekComponentConfig(
                                                           instance_id=self.instance_id,
                                                           name=self.name,
                                                           cls=None,
                                                           config=config.config.info_fabricator_configs
                                                       ))
        self.strategies: Dict[str, "StrategyWrapper"] = {}
        self.signals: Dict[str, Signal] = {}

    def on_data(self, data: Data) -> Signal|None:
        try:
            ds = self.info_fabricators.on_data(data)
            s = None
            for d in ds:
                s = self._process_data(d)
            return s
        except Exception as e:
            logger.error(f"info_fabricator {self.instance_id} process error: {e}", exc_info=True)

    def exec_update(self, execution_context: ExecutionContext | str):
        if isinstance(execution_context, str):
            signal = self.signals.pop(execution_context, None)
            logger.error(f"信号处理失败: {signal}")
            if not signal:
                return
            for signal_asset in signal.assets:
                signal_asset.actual_ratio = 0
        else:
            logger.info(f"执行订单更新: {execution_context.is_finish}, {execution_context}")
            if not execution_context.is_finish:
                return
            signal = self.signals.pop(execution_context.signal_id, None)
            if not signal:
                return
            for signal_asset in signal.assets:
                signal_asset.actual_ratio = signal_asset.actual_ratio or 0
                for asset in execution_context.execution_assets:
                    if asset.asset_key == signal_asset.asset_key and asset.is_open == signal_asset.is_open:
                        signal_asset.actual_ratio += asset.ratio
        self.strategies[signal.strategy_instance_id].on_signal_finish(signal=signal)
        self.event_bus.publish_event(Event(
            event_type=EventType.STRATEGY_SIGNAL_FINISH,
            data=signal
        ))

    def dispatch_event(self, event_type: EventType, payload: Any):
        """
        把事件路由到对应的 StrategyWrapper。
        payload 类型按 event_type 决定:
            ORDER_UPDATED       → Order
            POSITION_UPDATE     → Position
            其他                → 透传(走 wrapper.on_event 兜底)

        路由规则: 通过 payload.strategy_instance_id 找到对应 wrapper。
        """
        if not payload:
            return
        target_instance_id = getattr(payload, 'strategy_instance_id', None)
        if target_instance_id is None:
            # 无实例 ID 的事件: 广播给所有 wrapper
            for wrapper in self.strategies.values():
                self._safe_dispatch_to_wrapper(wrapper, event_type, payload)
            return
        wrapper = self.strategies.get(str(target_instance_id))
        if wrapper is None:
            logger.debug(f"dispatch_event 找不到 wrapper instance_id={target_instance_id}")
            return
        self._safe_dispatch_to_wrapper(wrapper, event_type, payload)

    def _safe_dispatch_to_wrapper(self, wrapper: "StrategyWrapper", event_type: EventType, payload: Any):
        try:
            new_signal = wrapper.dispatch_event(event_type, payload)
            if new_signal is not None:
                # Tier 2/3 回调返回的意图: 包成 Signal, 通过 STRATEGY_SIGNAL_MANUAL 触发 Engine._on_signal
                # (Engine._on_signal 内部会再发布 STRATEGY_SIGNAL,因此这里只发一次 MANUAL)
                self.signals[new_signal.signal_id] = new_signal
                self.event_bus.publish_event(Event(
                    event_type=EventType.STRATEGY_SIGNAL_MANUAL,
                    data=new_signal
                ))
        except Exception as e:
            logger.error(f"wrapper dispatch_event 异常: {e}", exc_info=True)

    def _process_data(self, data: Data) -> None|Signal:
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
        finally:
            if self.strategies[key].state == StrategyState.STOPPED:
                del self.strategies[key]
        if r:
            s =self.build_signal(assets=r, data=data, key=key)
            self.signals[s.signal_id] = s
            return s

    def build_signal(self, assets: List[Asset], data: Data, key) -> Signal:
        """
        构建信号
        """
        return Signal(
            signal_id=generate_str(),
            data_source_instance_id=data.data_source_id,
            strategy_id=self.instance_id,
            strategy_instance_id=key,
            strategy_cls=f"{self.config.cls.__module__}|{self.config.cls.__name__}",
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
        s = Signal(
            signal_id=generate_str(),
            data_source_instance_id="0",
            strategy_id=self.instance_id,
            strategy_cls=f"{self.config.cls.__module__}|{self.config.cls.__name__}",
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
        self.signals[s.signal_id] = s
        return s

    def create_component(self, key: str=None) -> "StrategyWrapper":
        wrapper = StrategyWrapper(
            self.event_bus,
            create_component(self.config.cls, **(self.config.config.strategy_config or {})),
            [create_component(c.cls, **(c.config or {})) for c in self.config.config.risk_policies or []]
        )
        wrapper.positon_getter = lambda: leek_context.position_tracker.find_position(strategy_id=self.instance_id, strategy_instance_id=key)
        # 注入 signal_builder: wrapper 在 Tier 2/3 回调里生成的 assets 由 context 包成 Signal
        wrapper.signal_builder = lambda assets: Signal(
            signal_id=generate_str(),
            data_source_instance_id="0",
            strategy_id=self.instance_id,
            strategy_instance_id=key,
            strategy_cls=f"{self.config.cls.__module__}|{self.config.cls.__name__}",
            config=self.config.config.strategy_position_config,
            signal_time=datetime.now(),
            assets=assets,
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
        logger.info(f"策略{self.instance_id}启动完成, 实例数: {len(self.strategies)} 数据源: {len(self.config.config.data_source_configs)}")
    
    # def on_event(self, event: Event):
    #     """
    #     接收所有事件，按策略ID和实例ID分发到对应 StrategyWrapper
    #     """
    #     # 仅处理与本策略上下文相关的事件
    #     data = event.data
    #     strategy_id = getattr(data, 'strategy_id', None)
    #     if strategy_id is not None and strategy_id != self.instance_id:
    #         return
    #
    #     target_instance_id = getattr(data, 'strategy_instance_id', None)
    #     if target_instance_id is None:
    #         # 无实例ID的事件，广播给所有实例
    #         for wrapper in self.strategies.values():
    #             wrapper.on_event(event)
    #         return
    #
    #     wrapper = self.strategies.get(str(target_instance_id))
    #     if wrapper:
    #         wrapper.on_event(event)

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
        d = {k: json.loads(json.dumps(v.get_state(), cls=LeekJSONEncoder)) for k, v in self.strategies.items()}
        for k, v in d.items():
            v["signals"] = [s for s in self.signals.values() if s.strategy_instance_id == k]
        return d

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
            if "signals" in v:
                for s in v["signals"]:
                    signal = Signal(**s)
                    self.signals[signal.signal_id] = signal
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
        # 真实意图由 pending_index + position 推断, 不再保留"当前意图"快照字段
        # self.position_rate: Decimal = Decimal("0")

        # self.position: Dict[str, Position] = {}
        self.lock = Lock()
        self.positon_getter = None

        # 新方案: 挂单状态机
        self.pending_index = PendingOrderIndex()         # 该实例所有未成交挂单
        self.bar_counter = 0                              # 用于 expire_bars 计数
        self.signal_builder: Optional[Callable[[List[Asset]], Signal]] = None  # 由 context 注入
    
    @property
    def position_rate(self) -> Decimal:
        if len(self.position) == 0:
            return Decimal("0")
        return Decimal(sum(p.ratio + sum(v.ratio for v in p.virtual_positions) for p in self.position.values()))
    
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
                assets = self.on_cta_data(data)
                if assets:
                    logger.info(f"策略{self.strategy.display_name} 仓位: {p_rate} -> {self.position_rate}, 状态: {self.state}, 信号: {assets}")
                return assets

            raise ValueError("strategy must be process")
        finally:
            self.lock.release()

    def dispatch_event(self, event_type: EventType, payload: Any) -> Optional[Signal]:
        """
        Tier 2 / 3 入口: 把订单/仓位事件路由到对应 callback, 收集意图列表, 翻译为 Signal。

        撤单意图(CancelCommand)直接通过 ORDER_CANCEL_REQUEST 事件发出, 不返回。
        新挂单意图(StrategyCommand)包成 Signal 返回, 由 StrategyContext 进一步推到 portfolio。

        ⚠️ 注意: 与 on_data 共用 self.lock, 同时只有一个 callback 运行。
        """
        if not self.lock.acquire(blocking=False):
            logger.warning("dispatch_event 抢锁失败, 跳过事件")
            return None
        try:
            if event_type == EventType.ORDER_UPDATED:
                return self._on_order_event(payload)
            elif event_type == EventType.POSITION_UPDATE:
                return self._on_position_event(payload)
            else:
                return self._on_other_event(event_type, payload)
        finally:
            self.lock.release()

    # ------------------------ Tier 2 / 3 内部实现 ------------------------

    def _on_order_event(self, order: Order) -> Optional[Signal]:
        if order is None:
            return None
        # 1. 维护 pending_index: 限价单中间态入索引, 终态出索引
        if order.order_status.is_finished:
            self.pending_index.remove(order.order_id)
        elif order.order_type == OrderType.LimitOrder:
            self.pending_index.add(order)

        # 2. 终态触发 on_order_update; 中间态走 on_event 兜底
        actions = None
        try:
            if order.order_status.is_finished:
                actions = self.strategy.on_order_update(order)
            else:
                actions = self.strategy.on_event(Event(event_type=EventType.ORDER_UPDATED, data=order))
        except Exception as e:
            logger.error(f"on_order_update/on_event 异常: {e}", exc_info=True)
            return None
        return self._materialize_actions(actions, ref=order)

    def _on_position_event(self, position: Position) -> Optional[Signal]:
        try:
            actions = self.strategy.on_position_update(position)
        except Exception as e:
            logger.error(f"on_position_update 异常: {e}", exc_info=True)
            return None
        return self._materialize_actions(actions, ref=position)

    def _on_other_event(self, event_type, payload) -> Optional[Signal]:
        try:
            actions = self.strategy.on_event(Event(event_type=event_type, data=payload))
        except Exception as e:
            logger.error(f"on_event 异常: {e}", exc_info=True)
            return None
        return self._materialize_actions(actions, ref=payload)

    def _materialize_actions(self, actions, ref) -> Optional[Signal]:
        """
        Tier 2/3 actions → Signal。
        - StrategyCommand: 包成 Asset 加入 Signal
        - CancelCommand:   立即发 ORDER_CANCEL_REQUEST 事件
        """
        if not actions:
            return None
        submits = [a for a in actions if isinstance(a, StrategyCommand)]
        cancels = [a for a in actions if isinstance(a, CancelCommand)]

        for c in cancels:
            self._enqueue_cancels(c)

        if not submits or self.signal_builder is None:
            return None
        if ref is None or not hasattr(ref, 'symbol'):
            logger.warning("Tier 2 意图无法推断 symbol 上下文, 跳过")
            return None

        assets = []
        for cmd in submits:
            extra = dict(cmd.extra) if cmd.extra else {}
            extra["_submit_bar"] = self.bar_counter
            if cmd.expire_bars is not None:
                extra["_expire_bars"] = cmd.expire_bars
            is_open = self._infer_is_open(cmd)
            # 价格推断: 限价单用 cmd.price; 其他场景用 ref 中最合适的价
            if cmd.order_type == OrderType.LimitOrder and cmd.price is not None:
                price = cmd.price
            else:
                price = (getattr(ref, 'execution_price', None)
                         or getattr(ref, 'current_price', None)
                         or getattr(ref, 'order_price', None)
                         or Decimal("0"))
            assets.append(Asset(
                asset_type=ref.asset_type,
                ins_type=ref.ins_type,
                symbol=ref.symbol,
                quote_currency=ref.quote_currency,
                side=cmd.side,
                ratio=min(cmd.ratio, Decimal("1")),
                is_open=is_open,
                price=price,
                order_type=cmd.order_type,
                expire_bars=cmd.expire_bars,
                extra=extra,
            ))
        return self.signal_builder(assets)

    def _infer_is_open(self, cmd: StrategyCommand) -> bool:
        """根据当前持仓推断 cmd 是开仓还是平仓"""
        positions = list(self.position.values())
        if not positions:
            return True
        pos = positions[0]
        # cmd.side 与持仓方向相反 → 平仓
        return cmd.side != pos.side

    def _enqueue_cancels(self, cancel: CancelCommand):
        """解析 CancelCommand → 发 ORDER_CANCEL_REQUEST 事件"""
        targets = []
        if cancel.order_id:
            o = self.pending_index.get(cancel.order_id)
            if o:
                targets.append(o)
        elif cancel.cancel_all:
            targets = self.pending_index.all()
        elif cancel.tag:
            targets = self.pending_index.match_tag(cancel.tag)
        for o in targets:
            self.event_bus.publish_event(Event(event_type=EventType.ORDER_CANCEL_REQUEST, data=o))

    def _check_expired(self):
        """扫描 pending_index, 撤掉超过 expire_bars 的挂单"""
        for order in self.pending_index.all():
            if not order.extra:
                continue
            eb = order.extra.get("_expire_bars")
            sb = order.extra.get("_submit_bar")
            if eb is None or sb is None:
                continue
            if self.bar_counter - sb >= eb:
                if order.extra is None:
                    order.extra = {}
                order.extra["_expired"] = True
                self.event_bus.publish_event(Event(event_type=EventType.ORDER_CANCEL_REQUEST, data=order))

    # ------------------------ 意图规范化与 Asset 构造 ------------------------

    def _normalize_open(self, res) -> Optional[List[StrategyCommand]]:
        """should_open 返回值统一为 List[StrategyCommand], None 表示撤回所有开仓意图。"""
        if res is None:
            return None
        if isinstance(res, PositionSide):
            return [StrategyCommand(res, Decimal("1"))]
        if isinstance(res, StrategyCommand):
            return [res]
        if isinstance(res, list):
            return [c for c in res if isinstance(c, StrategyCommand)]
        raise ValueError(f"should_open 返回类型非法: {type(res)}")

    def _normalize_close(self, res, position: Position) -> Optional[List[StrategyCommand]]:
        """close 返回值统一为 List[StrategyCommand], None/False 表示不动。"""
        if res is None or res is False:
            return None
        if res is True:
            return [StrategyCommand(position.side.switch(), Decimal("1"))]
        if isinstance(res, Decimal):
            return [StrategyCommand(position.side.switch(), res)]
        if isinstance(res, StrategyCommand):
            return [res]
        if isinstance(res, list):
            return [c for c in res if isinstance(c, StrategyCommand)]
        raise ValueError(f"close 返回类型非法: {type(res)}")

    def _make_asset(self, cmd: StrategyCommand, data: Data, is_open: bool,
                    ratio_override: Optional[Decimal] = None) -> Asset:
        """统一构造 Asset 并写入挂单追踪元数据"""
        extra = dict(cmd.extra) if cmd.extra else {}
        extra["_submit_bar"] = self.bar_counter
        if cmd.expire_bars is not None:
            extra["_expire_bars"] = cmd.expire_bars
        ratio = ratio_override if ratio_override is not None else cmd.ratio
        return Asset(
            asset_type=data.asset_type,
            ins_type=data.ins_type,
            symbol=data.symbol,
            quote_currency=data.quote_currency,
            side=cmd.side,
            ratio=min(ratio, Decimal("1")),
            is_open=is_open,
            price=cmd.price if (cmd.order_type == OrderType.LimitOrder and cmd.price is not None) else data.close,
            order_type=cmd.order_type,
            expire_bars=cmd.expire_bars,
            extra=extra,
        )
    
    def on_signal_finish(self, signal: Signal):
        """
        处理信号完成
        """
        # for asset in signal.assets:
        #     if asset.is_open:
        #         self.position_rate += asset.actual_ratio
        #     else:
        #         self.position_rate -= asset.actual_ratio
        # 进出场状态转换：将 ENTERING/EXITING 归一到正常态
        if self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
        elif self.position_rate > 0:
            self.state = StrategyInstanceState.HOLDING
        
        logger.info(f"策略信号处理完成: {signal.signal_id}, {self.state}, 当前仓位比例: {self.position_rate}")


    def on_event(self, event: Event):
        """
        ⚠️ 旧入口,保留作为外部直接调用 wrapper.on_event 时的兜底。
        新方案下事件流应走 dispatch_event(event_type, payload), 由 wrapper 内部分发到
        on_order_update / on_position_update / on_event。
        """
        try:
            self.strategy.on_event(event)
        except Exception as e:
            logger.error(f"策略事件处理异常: {e}", exc_info=True)

    def on_cta_data(self, data: Data = None) -> Optional[List[Asset]]:
        """
        处理数据, 走状态机调度。返回 List[Asset] (Signal 由调用方构造)。

        状态机:
            READY      → ready_handler (首次提交, 进入 ENTERING)
            ENTERING   → 若有未成交限价开仓单, ready_handler(from_entering=True) 走 diff;否则阻塞
            HOLDING    → holding_handler (首次平仓提交, 进入 EXITING; 或重复开仓)
            EXITING    → 若有未成交限价平仓单, holding_handler(from_exiting=True) 走 diff;否则阻塞
            STOPPING   → stopping_handler
        """
        if self.state in [StrategyInstanceState.STOPPED, StrategyInstanceState.CREATED]:
            return None
        self.bar_counter += 1
        # 更新计算信息
        self.strategy.on_data(data)
        if data.get("history_data", False):
            return None

        # 扫描过期挂单
        self._check_expired()

        logger.debug(f"{self.strategy.display_name} 当前状态: {self.state}, 仓位[{self.position_rate}]: "
                    f"{['%s:%s-%s:%s, %s, %s' % (p.position_id, p.symbol, p.quote_currency, p.side, p.ratio, p.sz) for p in list(self.position.values())]}")

        # 风控触发: 直接生成强平意图
        for pos in list(self.position.values()):
            for p in self.policies:
                if p.evaluate(data, pos):
                    continue
                try:
                    logger.info(f"风控触发: {p.display_name}, 策略{self.strategy.display_name}清仓, 仓位比例: {self.position_rate}, 仓位: {pos}")
                    risk_cmd = StrategyCommand(pos.side.switch(), Decimal("1"))
                    self.state = StrategyInstanceState.STOPPED if self.state == StrategyInstanceState.STOPPING else StrategyInstanceState.READY
                    self._publish_embedded_risk_event(pos, p, data)
                    ratio = min(risk_cmd.ratio, self.position_rate)
                    return [self._make_asset(risk_cmd, data, is_open=False, ratio_override=ratio)]
                finally:
                    try:
                        self.strategy.after_risk_control()
                    except Exception as e:
                        logger.error(f"after_risk_control 失败: {e}", exc_info=True)

        # 状态分发
        if self.state == StrategyInstanceState.READY:
            return self.ready_handler(data)
        elif self.state == StrategyInstanceState.ENTERING:
            # 仅当存在未成交限价开仓单时, 继续调 should_open 以走 diff (支持改价/撤回)
            # 纯市价策略: 限价开仓单数为 0, 此分支返回 None, 行为与改造前完全一致
            if self.pending_index.limit_open_orders():
                return self.ready_handler(data, from_entering=True)
            return None
        elif self.state == StrategyInstanceState.HOLDING:
            return self.holding_handler(data)
        elif self.state == StrategyInstanceState.EXITING:
            if self.pending_index.limit_close_orders():
                return self.holding_handler(data, from_exiting=True)
            return None
        elif self.state == StrategyInstanceState.STOPPING:
            return self.stopping_handler(data)
        return None

    def ready_handler(self, data: Data, from_entering: bool = False) -> Optional[List[Asset]]:
        """
        处理 should_open 意图。

        from_entering=True 表示已经进入 ENTERING 状态再次调用, 走 diff 引擎;
        否则是首次提交 (READY → ENTERING)。
        """
        res = self.strategy.should_open()
        cmds = self._normalize_open(res)

        # cmds = None 表达"撤回意图": 撤所有未成交开仓挂单
        if cmds is None:
            for o in self.pending_index.open_orders():
                self.event_bus.publish_event(Event(event_type=EventType.ORDER_CANCEL_REQUEST, data=o))
            return None
        if not cmds:
            return None

        if from_entering:
            # diff 引擎: 只比较限价单
            existing = self.pending_index.limit_open_orders()
            desired = [(intent_key(c.extra), c) for c in cmds if c.order_type == OrderType.LimitOrder]
            patch = diff_intents(existing, desired)
            for o in patch.to_cancel:
                self.event_bus.publish_event(Event(event_type=EventType.ORDER_CANCEL_REQUEST, data=o))
            if not patch.to_submit:
                return None
            return [self._make_asset(c, data, is_open=True) for c in patch.to_submit]

        # 首次提交: READY → ENTERING
        total_ratio = sum(c.ratio for c in cmds)
        max_ratio = Decimal("1") - self.position_rate
        if total_ratio <= 0 or max_ratio <= 0:
            return None
        # 单笔意图保留旧行为(ratio 截断); 多笔意图按比例缩放以保证总和不超 max_ratio
        if len(cmds) == 1:
            cmds[0].ratio = min(cmds[0].ratio, max_ratio)
        elif total_ratio > max_ratio:
            factor = max_ratio / total_ratio
            for c in cmds:
                c.ratio = c.ratio * factor

        self.state = StrategyInstanceState.ENTERING
        return [self._make_asset(c, data, is_open=True) for c in cmds]

    def holding_handler(self, data: Data, from_exiting: bool = False) -> Optional[List[Asset]]:
        """处理 close 意图。from_exiting=True 走 diff 引擎。"""
        positions = list(self.position.values())
        if not positions:
            return None
        position = positions[0]
        res = self.strategy.close(position)
        cmds = self._normalize_close(res, position)

        if cmds is None:
            # close 返回 None/False: 兼容旧"重复开仓"逻辑
            if not from_exiting and not self.strategy.open_just_no_pos:
                return self.ready_handler(data)
            return None
        if not cmds:
            return None

        if from_exiting:
            existing = self.pending_index.limit_close_orders()
            desired = [(intent_key(c.extra), c) for c in cmds if c.order_type == OrderType.LimitOrder]
            patch = diff_intents(existing, desired)
            for o in patch.to_cancel:
                self.event_bus.publish_event(Event(event_type=EventType.ORDER_CANCEL_REQUEST, data=o))
            if not patch.to_submit:
                return None
            return [self._make_asset(c, data, is_open=False,
                                     ratio_override=min(c.ratio * self.position_rate, self.position_rate))
                    for c in patch.to_submit]

        # 首次平仓: HOLDING → EXITING
        self.state = StrategyInstanceState.EXITING
        return [self._make_asset(c, data, is_open=False,
                                 ratio_override=min(c.ratio * self.position_rate, self.position_rate))
                for c in cmds]

    def stopping_handler(self, data: Data) -> Optional[List[Asset]]:
        if self.state == StrategyInstanceState.READY or self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED
            return None

        # 直接按剩余仓位一次性退出: 从当前持仓推断方向(权威来源)
        if self.position_rate > 0:
            positions = list(self.position.values())
            if not positions:
                self.state = StrategyInstanceState.STOPPED
                return None
            position_change = self.position_rate
            self.state = StrategyInstanceState.STOPPED
            cmd = StrategyCommand(positions[0].side.switch(), position_change)
            return [self._make_asset(cmd, data, is_open=False, ratio_override=position_change)]
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
            "position": [json.loads(json.dumps(p, cls=LeekJSONEncoder)) for p in self.position.values()],
            # 新方案: 持久化挂单索引与 bar 计数器, 替代旧 current_command 字段
            "bar_counter": self.bar_counter,
            "pending_index": [
                json.loads(json.dumps(o, cls=LeekJSONEncoder))
                for o in self.pending_index.all()
            ],
        }
        return json.loads(json.dumps(state, cls=LeekJSONEncoder))

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载策略上下文状态

        参数:
            state: 状态字典
        """
        self.state = StrategyInstanceState(state.get("state", self.state))
        # self.position_rate = Decimal(state.get("position_rate", self.position_rate))
        # 旧版状态可能含 current_command 字段, 直接忽略(新方案已删除)
        # 恢复 bar_counter
        self.bar_counter = int(state.get("bar_counter", 0))
        # 恢复 pending_index (重启后会通过 PENDING_ORDER_RECONCILE 与交易所对账校正)
        self.pending_index = PendingOrderIndex()
        for o_dict in state.get("pending_index", []) or []:
            try:
                # 简化恢复: 仅恢复关键字段, 完整 Order 由 reconcile 重建
                # 这里依赖 Order 是 dataclass, dict 反序列化需要类型转换
                # v1 实现: 只保留 order_id/extra/order_status/is_open/order_type/order_price/ratio/side
                from leek_core.models import Order as OrderCls
                order = OrderCls(
                    order_id=o_dict.get("order_id"),
                    position_id=o_dict.get("position_id"),
                    strategy_id=o_dict.get("strategy_id"),
                    strategy_instance_id=o_dict.get("strategy_instance_id"),
                    signal_id=o_dict.get("signal_id"),
                    exec_order_id=o_dict.get("exec_order_id"),
                    order_status=OrderStatus(o_dict.get("order_status")) if o_dict.get("order_status") else OrderStatus.SUBMITTED,
                    signal_time=None,
                    order_time=None,
                    symbol=o_dict.get("symbol"),
                    quote_currency=o_dict.get("quote_currency"),
                    ins_type=o_dict.get("ins_type"),
                    asset_type=o_dict.get("asset_type"),
                    side=PositionSide(o_dict["side"]) if "side" in o_dict else PositionSide.LONG,
                    is_open=bool(o_dict.get("is_open", True)),
                    is_fake=bool(o_dict.get("is_fake", False)),
                    order_amount=Decimal(o_dict.get("order_amount") or 0),
                    order_price=Decimal(o_dict.get("order_price") or 0),
                    ratio=Decimal(o_dict.get("ratio") or 0),
                    order_type=OrderType(o_dict.get("order_type")) if o_dict.get("order_type") else None,
                    extra=o_dict.get("extra"),
                    executor_id=o_dict.get("executor_id"),
                    market_order_id=o_dict.get("market_order_id"),
                )
                self.pending_index.add(order)
            except Exception as e:
                logger.warning(f"恢复 pending_order 失败: {e}")
        logger.info(f"加载策略{self.strategy.display_name}状态: pending_count={len(self.pending_index)}, bar_counter={self.bar_counter}")
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
