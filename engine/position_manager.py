#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
仓位管理模块，提供仓位管理的基础功能。
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from models import TradeInsType, Position, Component, Signal, Order, PositionConfig, AssetType, \
    OrderStatus, PositionContext
from position.base import Policy
from utils import get_logger, EventBus, generate_str, decimal_quantize

logger = get_logger(__name__)


class PositionManager(Component):
    """
    仓位管理器抽象基类，负责管理交易仓位。
    """

    def __init__(self, instance_id: str, name: str, event_bus: EventBus, config: PositionConfig = None):
        """初始化仓位管理器"""
        super().__init__(instance_id, name)
        self.event_bus = event_bus
        self.config = config
        self.policies = []

        self.amount = config.init_amount if config is not None else Decimal(0)
        self.activate_ratio = Decimal(1)  # 可用比例
        self.activate_amount = Decimal(1)  # 可用金额
        self.virtual_amount = self.amount

        self.positions: Dict[str, Position] = {}  # 仓位字典，键为仓位ID
        self.strategy_positions: Dict[str, List[Position]] = {}  # 策略仓位字典，键为策略ID
        self.execution_positions: Dict[str, List[Position]] = {}  # 执行仓位字典，键为执行器ID

    def process_signal(self, signal: Signal) -> Order:
        """
        处理信号 PositionManager 核心职责

        参数:
            signal: 策略信号

        返回:
            订单
        """
        position = self._get_position(signal.strategy_instance_id, signal.symbol, signal.quote_currency,
                                      signal.ins_type, signal.asset_type)
        if position is None:
            position = signal.init_position()
            self._add_position(position)

        if position.side == signal.side:  # 相同方向，直接开仓
            return self.open_position(position, signal)
        return self.close_position(position, signal)

    def _get_position(self, strategy_id: str, symbol: str, quote_currency: str, ins_type: TradeInsType, asset_type: AssetType) -> Position|None:
        if strategy_id not in self.strategy_positions:
            return None
        positions = self.strategy_positions[strategy_id]
        for position in positions:
            if position.symbol == symbol and position.quote_currency == quote_currency and position.ins_type == ins_type and position.asset_type == asset_type:
                return position
        return None

    def _get_symbol_current_principal(self, symbol: str, quote_currency: str) -> (Decimal, Decimal):
        principal = sum(p.amount + p.pnl for p in self.positions.values() if p.symbol == symbol and p.quote_currency == quote_currency)
        ratio = sum(p.ratio for p in self.positions.values() if p.symbol == symbol and p.quote_currency == quote_currency)
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

    def open_position(self, position: Position, signal: Signal) -> Order:
        """
        开仓
        
        参数:
            position: 仓位
            signal: 信号

        返回:
            订单
        """
        policy_res = self.do_risk_policy(signal)
        order = Order(
            order_id=generate_str(),
            position_id=position.position_id,
            order_status=OrderStatus.CREATED,
            signal_time=signal.signal_time,
            order_time=datetime.now(),
            symbol=signal.symbol,
            quote_currency=signal.quote_currency,
            ins_type=signal.ins_type,
            asset_type=signal.asset_type,
            side=signal.side,
            is_open=True,
            is_fake=policy_res,
            order_amount=Decimal(0),
            order_price=signal.price,
            leverage=Decimal(1)
        )

        cfg = signal.config
        if cfg.leverage:
            order.leverage = cfg.leverage
        if cfg.order_type:
            order.order_type = cfg.order_type
        if cfg.executor_id:
            order.executor_id = cfg.executor_id

        strategy_principal, strategy_ratio = self._get_strategy_current_principal(
            strategy_id=signal.strategy_instance_id)
        symbol_principal, symbol_ratio = self._get_symbol_current_principal(symbol=signal.symbol,
                                                                           quote_currency=signal.quote_currency)
        # 该策略还可以投入的资金
        strategy_available_amount = self.config.max_strategy_amount - strategy_principal
        strategy_available_ratio = self.config.max_strategy_ratio - strategy_ratio

        # 该标的还可以投入的资金
        symbol_available_amount = self.config.max_symbol_amount - symbol_principal
        symbol_available_ratio = self.config.max_symbol_ratio - symbol_principal

        # 当次比例
        ratio = min(symbol_available_ratio, strategy_available_ratio, self.config.max_ratio) * signal.ratio
        # 当次金额
        order_amount = min(cfg.principal * signal.ratio, strategy_available_amount, symbol_available_amount, self.amount*ratio)

        order.order_amount = decimal_quantize(order_amount, 4)
        order.ratio=decimal_quantize(order_amount / self.amount, 4)

        position.executing_ratio += order.ratio
        position.executing_amount += order.order_amount
        position.orders.append(order)
        self.activate_amount -= order.order_amount
        self.activate_ratio -= order.ratio
        return order

    def close_position(self, position: Position, signal: Signal) -> Order:
        """
        平仓
        
        参数:
            position: 仓位
            signal: 信号
            
        返回:
            订单
        """
        amount = position.amount * signal.ratio

        order = Order(
            order_id=generate_str(),
            position_id=position.position_id,
            order_status=OrderStatus.CREATED,
            signal_time=signal.signal_time,
            order_time=datetime.now(),
            symbol=signal.symbol,
            quote_currency=signal.quote_currency,
            ins_type=signal.ins_type,
            asset_type=signal.asset_type,
            side=signal.side,
            is_open=True,
            is_fake=position.is_fake,
            order_amount=amount,
            order_price=signal.price
        )
        position.executing_ratio += order.ratio
        position.executing_amount += order.order_amount
        position.orders.append(order)
        return order

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
                self.strategy_positions[position.strategy_id] = [p for p in self.strategy_positions[position.strategy_id] if p.position_id != position_id]
                if not self.strategy_positions[position.strategy_id]:
                    del self.strategy_positions[position.strategy_id]
            # 从执行器仓位字典移除
            if position.executor_id and position.executor_id in self.execution_positions:
                self.execution_positions[position.executor_id] = [p for p in self.execution_positions[position.executor_id] if p.position_id != position_id]
                if not self.execution_positions[position.executor_id]:
                    del self.execution_positions[position.executor_id]


    def update_position(self, position_id: str, current_price: Decimal) -> Optional[Position]:
        """
        更新仓位状态
        
        参数:
            position_id: 仓位ID
            current_price: 当前价格
            
        返回:
            更新后的仓位实例，如果仓位不存在则返回None
        """
        pass

    def get_state(self) -> dict:
        """
        获取当前仓位管理器的状态信息。
        包括资金、可用比例、仓位、策略、风控策略等。
        :return: dict
        """
        return {
            'amount': str(self.amount),
            'activate_ratio': str(self.activate_ratio),
            'activate_amount': str(self.activate_amount),
            'virtual_amount': str(self.virtual_amount),
            'positions': {pid: p.__dict__ for pid, p in self.positions.items()},
            'strategy_positions': {k: [p.position_id for p in v] for k, v in self.strategy_positions.items()},
            'execution_positions': {k: [p.position_id for p in v] for k, v in self.execution_positions.items()},
            'policies': [getattr(policy, 'policy_id', None) for policy in self.policies],
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
        }

    def load_state(self, state: dict):
        """
        根据外部传入的状态字典恢复仓位管理器的状态。
        :param state: 状态字典
        """
        self.amount = Decimal(state.get('amount', self.amount))
        self.activate_ratio = Decimal(state.get('activate_ratio', self.activate_ratio))
        self.activate_amount = Decimal(state.get('activate_amount', self.activate_amount))
        self.virtual_amount = Decimal(state.get('virtual_amount', self.virtual_amount))
        # positions、strategy_positions、execution_positions 的恢复需结合具体序列化方案
        # 这里只做简单占位，实际使用时建议结合模型反序列化
        # self.positions = ...
        # self.strategy_positions = ...
        # self.execution_positions = ...
        # self.policies = ...
        # self.config = ...

    def do_risk_policy(self, signal: Signal) -> bool:
        pos_ctx = PositionContext(
            active_amount=self.activate_amount,
            active_ratio=self.activate_ratio,
            positions=list(self.positions.values())
        )
        for policy in self.policies:
            reason = policy.check(signal, pos_ctx)
            if reason:
                logger.warning(f"Risk policy {policy.name} rejected signal {signal} with reason: {policy.reject_reason}")
                return True
        return False

    def add_policy(self, policy: Policy):
        """
        添加风控策略到仓位管理器。
        :param policy: Policy 实例
        """
        self.policies.append(policy)

    def remove_policy(self, instance_id: str):
        """
        根据策略ID（policy_id）删除风控策略。
        :param instance_id: 策略实例ID
        """
        self.policies = [p for p in self.policies if getattr(p, 'policy_id', None) != instance_id]

    def update_config(self, config: PositionConfig):
        """
        更新仓位管理器配置。
        :param config: PositionConfig 实例
        """
        self.config = config
