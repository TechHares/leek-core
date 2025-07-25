#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
from decimal import Decimal
from threading import Lock, RLock
from typing import Dict, List

from leek_core.base import LeekContext, create_component
from leek_core.event import EventBus, Event, EventType
from leek_core.models import (TradeInsType, Position, Signal, Order, PositionConfig, AssetType,
                              OrderStatus, PositionInfo, LeekComponentConfig, ExecutionAsset, OrderExecutionState,
                              DataType, KLine, Data)
from leek_core.models import ExecutionContext
from leek_core.policy import StrategyPolicy
from leek_core.utils import get_logger, generate_str, decimal_quantize
from leek_core.utils import thread_lock
_position_lock = RLock()
logger = get_logger(__name__)

class PositionContext(LeekContext):

    """
    仓位管理器抽象基类，负责管理交易仓位。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[None, PositionConfig]):
        """初始化仓位管理器"""
        super().__init__(event_bus, config)
        self.policies: List[StrategyPolicy] = []

        self.position_config = config.config
        if self.position_config is not None:
            for policy in self.position_config.risk_policies:
                self.policies.append(create_component(policy.cls, **policy.config))

        self.pnl = Decimal(0)
        self.friction = Decimal(0)
        self.fee = Decimal(0)
        self.virtual_pnl = Decimal(0)


        self.activate_amount = self.position_config.init_amount

        self.positions: Dict[str, Position] = {}  # 仓位字典，键为仓位ID
        self.strategy_positions: Dict[str, List[Position]] = {}  # 策略仓位字典，键为策略ID
        self.execution_positions: Dict[str, List[Position]] = {}  # 执行仓位字典，键为执行器ID
        self.asset_positions: Dict[tuple, List[Position]] = {}

        self.load_state(config.data)

    @property
    def total_amount(self) -> Decimal:
        return self.position_config.init_amount + self.pnl + self.friction + self.fee
    
    @property
    def profit(self) -> Decimal:
        return self.pnl + self.friction + self.fee

    @property
    def value(self) -> Decimal: # 实时仓位价值
        return sum(p.value for p in self.positions.values()) + self.activate_amount
    
    def evaluate_amount(self, signal: Signal) -> List[ExecutionAsset]:
        if not signal.assets:
            return []
        
        if self.activate_amount <= 0:
            return []

        strategy_used_principal, strategy_used_ratio = self._get_strategy_current_principal(signal.strategy_id)
        strategy_ratio = min(self.position_config.max_strategy_ratio - strategy_used_ratio, self.position_config.max_ratio)
        # 单次开仓最大金额 单次开仓最大比例 策略最大金额 策略最大比例 策略本金 取最小
        principal = decimal_quantize(self.total_amount * strategy_ratio, 8)
        principal = min(self.position_config.max_amount, principal, self.position_config.max_strategy_amount - strategy_used_principal, self.activate_amount)
        if signal.config and signal.config.principal:
            principal = min(signal.config.principal, principal)
        
        # 计算每个资产的最大可投入金额
        ratios = []
        max_amounts = []
        fake_flag = False
        for asset in signal.assets:
            # 获取当前标的总仓位
            current_position = self._get_position(
                signal.strategy_id,
                signal.strategy_instance_id,
                asset.symbol,
                asset.quote_currency,
                asset.ins_type,
                asset.asset_type
            )
            if current_position and current_position.side != asset.side: # 减仓
                continue    

            ratios.append(asset.ratio)
             # 计算单个标的最大可投入金额
            symbol_used_principal, symbol_used_ratio = self._get_symbol_current_principal(asset.symbol, asset.quote_currency)
            symbol_amount = self.position_config.max_symbol_amount - symbol_used_principal

            symbol_amount = min(symbol_amount, decimal_quantize(self.total_amount * (self.position_config.max_symbol_ratio - symbol_used_ratio), 8))
            if symbol_amount <= 0:
                fake_flag = True
            elif principal * asset.ratio > symbol_amount:
                principal = symbol_amount / asset.ratio
            max_amounts.append(symbol_amount)
        
        if sum(ratios) > 1:
            logger.error(f"信号 {signal.signal_id} 的资产比例之和大于1，无法计算单位金额")
            return None
        execution_assets = []
        for asset in signal.assets:
            current_position = self._get_position(
                signal.strategy_id,
                signal.strategy_instance_id,
                asset.symbol,
                asset.quote_currency,
                asset.ins_type,
                asset.asset_type
            )
            execution_asset = ExecutionAsset(
                asset_type=asset.asset_type,
                    ins_type=asset.ins_type,
                    symbol=asset.symbol,
                    side=asset.side,
                    price=asset.price,
                    ratio=asset.ratio,
                    amount=0,
                    is_open=False,
                    is_fake=fake_flag,
                    quote_currency=asset.quote_currency,
                    extra=asset.extra,
            )

            if current_position:
                execution_asset.executor_sz = current_position.executor_sz
                execution_asset.position_id = current_position.position_id

            execution_assets.append(execution_asset)
            if current_position and current_position.side != asset.side: # 减仓
                execution_asset.is_fake = current_position.is_fake
                execution_asset.position_id = current_position.position_id
                close_ratio = min(1, asset.ratio / current_position.ratio)
                execution_asset.sz = current_position.sz * close_ratio
                execution_asset.amount = decimal_quantize(current_position.amount * close_ratio, 8)
                continue
            
            execution_asset.is_open = True
            execution_asset.amount = decimal_quantize(principal * asset.ratio, 8)
            execution_asset.ratio = decimal_quantize(execution_asset.amount / self.total_amount, 8)

        return execution_assets

    @thread_lock(_position_lock)
    def process_signal(self, signal: Signal):
        """
        处理信号 PositionManager 核心职责

        参数:
            signal: 策略信号

        返回:
            订单
        """
        execution_assets = self.evaluate_amount(signal)
        logger.info(f"处理策略信号: {signal} -> {execution_assets}")
        if not execution_assets:
            return None
        execution_context = ExecutionContext(
            context_id=generate_str(),
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            strategy_instant_id=signal.strategy_instance_id,
            target_executor_id=signal.config.executor_id if signal.config else None,
            execution_assets=execution_assets,
            created_time=signal.signal_time,
            leverage=signal.config.leverage if signal.config and signal.config.leverage else self.position_config.default_leverage,
            order_type=signal.config.order_type if signal.config and signal.config.order_type else self.position_config.order_type,
            trade_type=self.position_config.trade_type,
            trade_mode=self.position_config.trade_mode,
        )
        if self.do_risk_policy(execution_context):
            for asset in execution_context.execution_assets:
                if asset.is_open:
                    asset.is_fake = True
        else:
            self.activate_amount -= execution_context.open_amount
        
        self.event_bus.publish_event(Event(
            event_type=EventType.EXEC_ORDER_CREATED,
            data=execution_context
        ))

    def _get_position(self, strategy_id: str, strategy_instance_id, symbol: str, quote_currency: str, ins_type: TradeInsType,
                      asset_type: AssetType) -> Position | None:
        positions = self.strategy_positions.get(strategy_id, None)
        if not positions:
            return None
        for position in positions:
            if (strategy_instance_id == position.strategy_instance_id and position.symbol == symbol and
                    position.quote_currency == quote_currency and position.ins_type == ins_type and position.asset_type == asset_type):
                return position
        return None
    
    def get_position(self, position_id: str) -> Position:
        return self.positions.get(position_id, None)

    def _get_symbol_current_principal(self, symbol: str, quote_currency: str) -> (Decimal, Decimal):
        matching_positions = [p for p in self.positions.values() if p.symbol == symbol and p.quote_currency == quote_currency]
        if not matching_positions:
            return Decimal('0'), Decimal('0')
            
        principal = sum(p.amount + p.pnl for p in matching_positions)
        ratio = sum(p.ratio for p in matching_positions)
        return principal, ratio

    def _get_strategy_current_principal(self, strategy_id: str) -> (Decimal, Decimal):
        """
        获取策略当前已投入本金和比例
        参数:
            strategy_id: 策略ID
        返回:
            已投入资金, 比例
        """
        if strategy_id not in self.strategy_positions:
            return Decimal(0), Decimal(0)
        positions = self.strategy_positions[strategy_id]
        principal = sum(p.amount + p.pnl for p in positions)
        ratio = sum(p.ratio for p in positions)
        return principal, ratio

    def _add_position(self, position: Position):
        """
        添加仓位到管理器。
        会同步更新 positions、strategy_positions、execution_positions 字典。
        :param position: Position 实例
        """
        self.positions[position.position_id] = position
        # 更新策略仓位字典
        self.strategy_positions.setdefault(position.strategy_id, []).append(position)
        # 更新执行器仓位字典
        if position.executor_id:
            self.execution_positions.setdefault(position.executor_id, []).append(position)

        self.asset_positions.setdefault((position.symbol, position.quote_currency, position.ins_type, position.asset_type), []).append(position)
        

    def _remove_position(self, position_id: str):
        """
        根据 position_id 移除仓位。
        会同步从 positions、strategy_positions、execution_positions 字典中删除。
        :param position_id: 仓位ID
        """
        position = self.positions.pop(position_id, None)
        if position:
            # 从策略仓位字典移除
            if position.strategy_id in self.strategy_positions:
                self.strategy_positions[position.strategy_id] = [p for p in
                                                                 self.strategy_positions[position.strategy_id] if
                                                                 p.position_id != position_id]
                if not self.strategy_positions[position.strategy_id]:
                    del self.strategy_positions[position.strategy_id]
            # 从执行器仓位字典移除
            if position.executor_id and position.executor_id in self.execution_positions:
                self.execution_positions[position.executor_id] = [p for p in
                                                                  self.execution_positions[position.executor_id] if
                                                                  p.position_id != position_id]
                if not self.execution_positions[position.executor_id]:
                    del self.execution_positions[position.executor_id]
            # 从资产仓位字典移除
            assert_key = (position.symbol, position.quote_currency, position.ins_type, position.asset_type)
            if assert_key in self.asset_positions:
                self.asset_positions[assert_key] = [p for p in self.asset_positions[assert_key] if p.position_id != position_id]
                if not self.asset_positions[assert_key]:
                    del self.asset_positions[(position.symbol, position.quote_currency, position.ins_type, position.asset_type)]

    def get_state(self) -> dict:
        """
        获取当前仓位管理器的状态信息。
        包括资金、可用比例、仓位、策略、风控策略等。
        :return: dict
        """
        return {
            'activate_amount': self.activate_amount,
            'pnl': self.pnl,
            'friction': self.friction,
            'fee': self.fee,
            'total_amount': self.total_amount,
            'virtual_pnl': self.virtual_pnl,
            'positions': list(self.positions.values()),
        }

    def on_data(self, data: Data):
        if data.data_type == DataType.KLINE and isinstance(data, KLine):
            for position in self.asset_positions.get((data.symbol, data.quote_currency, data.ins_type, data.asset_type), []):
                position.current_price = data.close
    
    def load_state(self, state: dict):
        """
        根据外部传入的状态字典恢复仓位管理器的状态。
        :param state: 状态字典
        """
        if not state:
            return

        if state.get("reset_position_state", False):
            self.pnl = Decimal(0)
            self.friction = Decimal(0)
            self.fee = Decimal(0)
            self.activate_amount = self.position_config.init_amount
            self.positions = {}
            self.strategy_positions = {}
            self.execution_positions = {} 
            logger.info("重置仓位状态完成")
            return

        self.pnl = Decimal(state.get('pnl', self.pnl))
        self.friction = Decimal(state.get('friction', self.friction))
        self.fee = Decimal(state.get('fee', self.fee))
        self.activate_amount = Decimal(state.get('activate_amount', self.activate_amount))
        for p in state.get('positions', []):
            self._add_position(Position(**p))

    def do_risk_policy(self, execution_context: ExecutionContext) -> bool:
        pos_ctx = PositionInfo(
            positions=list(self.positions.values())
        )
        for policy in self.policies:
            if not policy.evaluate(execution_context, pos_ctx):
                logger.warning(
                    f"Risk policy {policy.display_name} rejected signal {execution_context}")
                return True
        return False

    def add_policy(self, policy: StrategyPolicy):
        """
        添加风控策略到仓位管理器。
        :param policy: Policy 实例
        """
        policy.on_start()
        self.policies.append(policy)
        self.event_bus.publish_event(Event(
            event_type=EventType.POSITION_POLICY_ADD,
            data={
                "instance_id": self.instance_id,
                "name": policy.display_name,
            }
        ))

    def remove_policy(self, cls: type[StrategyPolicy]):
        """
        根据策略ID（policy_id）删除风控策略。
        :param cls: 策略class
        """
        del_policies = [p for p in self.policies if isinstance(p, cls)]
        self.policies = [p for p in self.policies if not isinstance(p, cls)]
        for policy in del_policies:
            policy.on_stop()
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_POLICY_DEL,
                data={
                    "instance_id": self.instance_id,
                    "name": policy.display_name,
                }
            ))

    def update_config(self, config: PositionConfig):
        """
        更新仓位管理器配置。
        :param config: PositionConfig 实例
        """
        delta_amount = config.init_amount - self.config.init_amount
        self.activate_amount += delta_amount
        self.config = config

    @thread_lock(_position_lock)
    def order_update(self, order: Order):
        """
        根据订单更新更新仓位信息
        
        Args:
            order: 订单信息
        """
        logger.info(f"仓位处理收到订单更新: {order}")
        if order.order_status == OrderStatus.SUBMITTED or order.order_status == OrderStatus.CREATED:
            return
        
        if order.order_status.is_failed:
            self.activate_amount += order.order_amount
            if not order.position_id:
                return

        # 如果是开仓订单且没有position_id，创建新仓位
        if order.is_open and not order.position_id:
            position = Position(
                position_id=generate_str(),
                strategy_id=order.strategy_id,
                strategy_instance_id=order.strategy_instant_id,
                symbol=order.symbol,
                quote_currency=order.quote_currency,
                ins_type=order.ins_type,
                asset_type=order.asset_type,
                side=order.side,
                cost_price=0,
                amount=0,
                ratio=0,
                current_price=order.execution_price,
                executor_id=order.executor_id,
                is_fake=order.is_fake,
                fee=0,
                friction=0,
                leverage=order.leverage,
                open_time=order.order_time
            )
            order.position_id = position.position_id
            self._add_position(position)
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_INIT,
                data=order
            ))


        # 获取现有仓位
        position = self.positions.get(order.position_id)
        logger.info(f"仓位处理订单更新原仓位: {position}")
        if not position:
            logger.warning(f"Position {order.position_id} not found for order {order.order_id}")
            return
        try:
            position.current_price = order.execution_price
            # 获取或创建订单执行状态
            order_state = position.order_states.get(order.order_id)
            if not order_state:
                order_state = OrderExecutionState(order_id=order.order_id, is_open=order.is_open)
                position.order_states[order.order_id] = order_state

            # 计算本次更新的变化量
            delta_amount = (order.settle_amount or Decimal('0')) - order_state.settle_amount
            delta_fee = (order.fee or Decimal('0')) - order_state.fee
            delta_friction = (order.friction or Decimal('0')) - order_state.friction
            delta_pnl = (order.pnl or Decimal('0')) - order_state.pnl
            delta_sz = (order.sz or Decimal('0')) - order_state.sz

            # 更新订单执行状态
            order_state.settle_amount = order.settle_amount or Decimal('0')
            order_state.fee = order.fee or Decimal('0')
            order_state.friction = order.friction or Decimal('0')
            order_state.pnl = order.pnl or Decimal('0')
            order_state.sz = order.sz or Decimal('0')

            position.friction += delta_friction
            position.fee += delta_fee

            position.pnl += delta_friction + delta_fee
            if position.is_fake:
                self.virtual_pnl += delta_pnl
            else:
                self.pnl += delta_pnl
                self.activate_amount += (delta_friction + delta_fee)
                self.friction += delta_friction
                self.fee += delta_fee
            if order.is_open: # 如果是开仓订单
                position.executor_sz[order.executor_id] = position.executor_sz.get(order.executor_id, Decimal('0')) + delta_sz
                position.total_amount += delta_amount
                position.total_sz += delta_sz
                if position.total_sz > 0:
                    position.cost_price = order.leverage * position.total_amount / position.total_sz
                if order.order_status == OrderStatus.FILLED and not position.is_fake:
                    self.activate_amount += (order.order_amount - order.settle_amount)
                    position.ratio += order.ratio
            else:
                position.total_back_amount += delta_amount
                position.executor_sz[order.executor_id] = position.executor_sz.get(order.executor_id, Decimal('0')) - delta_sz
                position.pnl += (1 if position.side.is_long else -1) * (order.execution_price - position.cost_price) * delta_sz
                closed_sz = (position.total_sz - position.sz)
                if closed_sz > 0:
                    position.close_price = position.total_back_amount / closed_sz
                self.activate_amount += delta_amount
                if order.order_status == OrderStatus.FILLED and not position.is_fake:
                    position.ratio -= order.ratio
            
            position.amount = position.sz * position.cost_price
            if position.sz <= 0:
                self._remove_position(position.position_id)
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_UPDATE,
                data=position
            ))
            logger.info(f"仓位更新完成，当前可用余额: {self.activate_amount}, 资产仓位: {len(self.asset_positions)}, "
                        f"总价值: {self.total_amount}, 总仓位: {len(self.positions)}")
        except Exception as e:
            logger.error(f"更新仓位信息失败: {e}", exc_info=True)
            
            