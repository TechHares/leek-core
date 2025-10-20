#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal
from threading import RLock
from typing import Dict, List

from leek_core.base import LeekComponent
from leek_core.event import EventBus, Event, EventType, EventSource
from leek_core.models import Transaction, TransactionType, ExecutionContext, Order
from leek_core.utils import get_logger, thread_lock

logger = get_logger(__name__)
_capital_lock = RLock()


class CapitalAccount(LeekComponent):
    """
    资金账户组件 - 纯资金管理
    
    职责：
    - 资金余额管理
    - 资金冻结/解冻
    - 交易流水记录
    - 资金查询统计
    """
    
    def __init__(self, event_bus: EventBus, initial_balance: Decimal = Decimal('0')):
        """
        初始化资金账户
        
        参数:
            event_bus: 事件总线
            initial_balance: 初始资金
        """
        super().__init__()
        self.event_bus = event_bus
        self.init_balance = initial_balance  # 初始资金
        self.available_balance = initial_balance  # 可用余额
        
        # 按策略分组的冻结资金
        self.frozen_by_strategy: Dict[str, Decimal] = {}
        # 按信号分组的冻结资金明细
        self.frozen_by_signal: Dict[str, List[Transaction]] = {}
    
    @property
    def total_balance(self) -> Decimal:
        return self.available_balance + self.frozen_balance
    
    @property
    def frozen_balance(self) -> Decimal:
        if len(self.frozen_by_signal) == 0:
            return Decimal(0)
        return sum(self.frozen_by_strategy.values())

    @thread_lock(_capital_lock)
    def freeze_amount(self, execution_context: ExecutionContext) -> bool:
        """
        冻结资金
        
        参数:
            amount: 冻结金额
            strategy_id: 策略ID
            signal_id: 信号ID
            asset_key: 资产键
            desc: 描述
            
        返回:
            bool: 是否成功
        """
        if execution_context.extra and "policy_id" in execution_context.extra:
            return True
        amount = 0
        for asset in execution_context.execution_assets:
            if asset.is_open and asset.amount > 0:
                amount += asset.amount
        if amount == 0:
            return True
        
        if amount > self.available_balance:
            return False
        
        for asset in execution_context.execution_assets:
            if asset.is_open and asset.amount > 0:
                balance_before = self.available_balance
                self.available_balance -= asset.amount
                transaction = Transaction(
                        strategy_id = execution_context.strategy_id,
                        strategy_instance_id = execution_context.strategy_instance_id,
                        position_id = asset.position_id,
                        exec_order_id = execution_context.context_id,
                        order_id = None,
                        signal_id = execution_context.signal_id,
                        asset_key = asset.asset_key,
                        type = TransactionType.FROZEN,
                        amount = -asset.amount,
                        balance_before = balance_before,
                        balance_after = self.available_balance,
                        desc = f"冻结资金[{asset.asset_key}]: {asset.amount}"
                    )
                self.frozen_by_signal.setdefault(execution_context.signal_id, []).append(transaction)
                self._publish_transaction_event(transaction)
        logger.info(f"冻结资金{amount}完成, 当前剩余可用资金: {self.available_balance}, 信号下数据: {self.frozen_by_signal.get(execution_context.signal_id, [])}")
        return True
    
    @thread_lock(_capital_lock)
    def change_amount(self, amount: Decimal, desc: str):
        """
        变更资金
        
        参数:
            amount: 变更金额（正数为增加，负数为减少）
            desc: 变更描述
        """
        balance_before = self.available_balance
        self.available_balance += amount
        self.init_balance += amount
        self._publish_transaction_event(Transaction(
            type=TransactionType.DEPOSIT if amount > 0 else TransactionType.WITHDRAW,
            amount=amount,
            balance_before=balance_before,
            balance_after=self.available_balance,
            desc=desc
        ))
    
    @thread_lock(_capital_lock)
    def unfreeze_amount(self, order: Order):
        """
        解冻资金
        
        参数:
            order: 订单
        """
        try:
            if order.is_fake:
                return
            if order.is_open:
                # 解冻
                logger.debug(f"解冻资金[开仓], 信号下数据: {self.frozen_by_signal.get(order.signal_id, [])}")
                frozen_transactions = [t for t in self.frozen_by_signal.get(order.signal_id, []) if t.asset_key == order.asset_key]
                logger.info(f"解冻资金[开仓], 信号下数据, 过滤后[{order.asset_key}]: {frozen_transactions}")
                if len(frozen_transactions) != 1:
                    logger.error(f"冻结交易[{order.order_id}-{order.asset_key}]数量不正确: {order}")
                    return
                frozen_transaction = frozen_transactions[0]
                self.frozen_by_signal[order.signal_id].remove(frozen_transaction)
                self._publish_transaction_event(frozen_transaction.unfozen(order.exec_order_id, order.order_id))
                self.available_balance -= frozen_transaction.amount
                if order.order_status.is_failed:
                    return
                # 扣款
                transaction = order.settle(self.available_balance, -1)
                logger.info(f"解冻资金[开仓], 扣款: {transaction}")
                if transaction:
                    self.available_balance = transaction.balance_after
                    self._publish_transaction_event(transaction)
                fee_transaction = order.settle_fee(self.available_balance)
                logger.info(f"解冻资金[开仓], 手续费: {fee_transaction}")
                if fee_transaction:
                    self.available_balance = fee_transaction.balance_after
                    self._publish_transaction_event(fee_transaction)
                return
            # 回款
            transaction = order.settle(self.available_balance)
            logger.info(f"解冻资金[平仓], 回款: {transaction}")
            if transaction:
                self.available_balance = transaction.balance_after
                self._publish_transaction_event(transaction)
            fee_transaction = order.settle_fee(self.available_balance)
            logger.info(f"解冻资金[平仓], 手续费: {fee_transaction}")
            if fee_transaction:
                self.available_balance = fee_transaction.balance_after
                self._publish_transaction_event(fee_transaction)
        finally:
            if len(self.frozen_by_signal.get(order.signal_id, [])) == 0:
                self.frozen_by_signal.pop(order.signal_id, None)
    
    def get_balance_summary(self) -> dict:
        """获取余额摘要"""
        return {
            'available_balance': str(self.available_balance),
            'frozen_balance': str(self.frozen_balance),
            'total_balance': str(self.total_balance),
            'frozen_by_strategy': {k: str(v) for k, v in self.frozen_by_strategy.items()}
        }
    
    def load_state(self, state: dict):
        """
        加载资金账户状态
        
        参数:
            state: 状态数据
        """
        if not state:
            return
        
        # 加载余额数据
        self.available_balance = Decimal(state.get('available_balance', self.available_balance))
        
        # 加载策略冻结金额
        frozen_by_strategy_data = state.get('frozen_by_strategy', {})
        self.frozen_by_strategy.clear()
        for strategy_id, amount_str in frozen_by_strategy_data.items():
            self.frozen_by_strategy[strategy_id] = Decimal(amount_str)
        
        logger.info(f"CapitalAccount 状态加载完成，可用余额: {self.available_balance}")
    
    def get_state(self) -> dict:
        """获取资金账户状态"""
        return {
            'principal': str(self.init_balance),
            'available_balance': str(self.available_balance),
            'frozen_by_strategy': {k: str(v) for k, v in self.frozen_by_strategy.items()}
        }
    
    
    def _publish_transaction_event(self, transaction: Transaction):
        """发布交易事件"""
        self.event_bus.publish_event(Event(
            event_type=EventType.TRANSACTION,
            data=transaction,
            source=EventSource(
                instance_id=id(self),
                name="CapitalAccount",
                cls=self.__class__.__name__
            )
        ))

    def reset(self):
        """重置资金账户"""
        self.available_balance = self.init_balance
        self.frozen_by_strategy.clear()
        self.frozen_by_signal.clear()
        logger.info("CapitalAccount 重置完成")
