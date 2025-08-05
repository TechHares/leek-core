from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union

from .constants import TradeInsType, AssetType

class TransactionType(Enum):
    """
    流水类型
    """
    FROZEN = 0  # 冻结
    UNFROZEN = 1  # 解冻
    DEPOSIT = 2  # 充值
    WITHDRAW = 3  # 提现
    TRADE = 4  # 交易
    FEE = 5  # 手续费
    # PNL = 6  # 盈亏
    # FUNDING = 7  # 资金费
    # SETTLE = 8  # 结算
    OTHER = 9  # 其他

    def __str__(self):
        return self.name.upper()

@dataclass
class Transaction:
    """流水"""
    type: TransactionType
    amount: Decimal
    balance_before: Decimal
    balance_after: Decimal

    strategy_id: str = None
    strategy_instance_id: str = None
    position_id: str = None
    exec_order_id: str = None
    order_id: str = None
    signal_id: str = None
    asset_key: str = None


    desc: str = None

    def __post_init__(self):
        if self.strategy_id and not isinstance(self.strategy_id, str):
            self.strategy_id = str(self.strategy_id)
        if self.strategy_instance_id and not isinstance(self.strategy_instance_id, str):
            self.strategy_instance_id = str(self.strategy_instance_id)
        if self.position_id and not isinstance(self.position_id, str):
            self.position_id = str(self.position_id)
        if self.exec_order_id and not isinstance(self.exec_order_id, str):
            self.exec_order_id = str(self.exec_order_id)
        if self.order_id and not isinstance(self.order_id, str):
            self.order_id = str(self.order_id)
        if self.signal_id and not isinstance(self.signal_id, str):
            self.signal_id = str(self.signal_id)
        if self.asset_key and not isinstance(self.asset_key, str):
            self.asset_key = str(self.asset_key)
        if isinstance(self.type, int):
            self.type = TransactionType(self.type)
        elif isinstance(self.type, str):
            self.type = TransactionType(int(self.type))
        
        self.amount = Decimal(self.amount)
        self.balance_before = Decimal(self.balance_before)
        self.balance_after = Decimal(self.balance_after)

    def unfozen(self, exec_order_id: str, order_id: str):
        return Transaction(
            strategy_id=self.strategy_id,
            strategy_instance_id=self.strategy_instance_id,
            position_id=self.position_id,
            exec_order_id=exec_order_id,
            order_id=order_id,
            signal_id=self.signal_id,
            asset_key=self.asset_key,
            type=TransactionType.UNFROZEN,
            amount=self.amount,
            balance_before=self.balance_after,
            balance_after=self.balance_before,
            desc=f"解冻({self.desc})",
        )