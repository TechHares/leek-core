#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union

from .constants import TradeInsType, AssetType
    
class PositionSide(Enum):
    """
    头寸方向 PositionSide
    """
    LONG = 1  # 多头
    SHORT = 2  # 空头
    FLAT = 4  # 平仓

    @staticmethod
    def switch_side(side):
        """
        切换头寸方向
        :param side: 当前头寸方向
        :return: 切换后的头寸方向
        """
        if side == PositionSide.LONG:
            return PositionSide.SHORT
        if side == PositionSide.SHORT:
            return PositionSide.LONG
        raise RuntimeError("Invalid position side for switching")

    def switch(self):
        """
        切换当前头寸方向
        :return: 切换后的头寸方向
        """
        return self.switch_side(self)

    @property
    def is_long(self):
        """
        是否是多头
        :return: bool
        """
        return self == PositionSide.LONG

    @property
    def is_short(self):
        """
        是否是空头
        :return: bool
        """
        return self == PositionSide.SHORT

    @property
    def is_flat(self):
        """
        是否是平仓
        :return: bool
        """
        return self == PositionSide.FLAT


@dataclass
class OrderExecutionState:
    """订单执行状态"""
    order_id: str
    is_open: bool = True
    settle_amount: Decimal = Decimal('0')
    fee: Decimal = Decimal('0')
    friction: Decimal = Decimal('0')
    pnl: Decimal = Decimal('0')
    sz: Decimal = Decimal('0')

    def __post_init__(self):
        if not self.order_id or not isinstance(self.order_id, str):
            raise ValueError("order_id is required and must be a string")
        if not isinstance(self.is_open, bool):
            self.is_open = bool(self.is_open)
        
        if self.settle_amount and not isinstance(self.settle_amount, Decimal):
            self.settle_amount = Decimal(self.settle_amount)
        
        if self.fee and not isinstance(self.fee, Decimal):
            self.fee = Decimal(self.fee)
        
        if self.friction and not isinstance(self.friction, Decimal):
            self.friction = Decimal(self.friction)
        
        if self.pnl and not isinstance(self.pnl, Decimal):
            self.pnl = Decimal(self.pnl)
        
        if self.sz and not isinstance(self.sz, Decimal):
            self.sz = Decimal(self.sz)


@dataclass
class Position:
    """
    表示一个策略在一个交易器下、一个单独标的的仓位。

    属性:
        position_id:      仓位ID
        strategy_id:      策略ID
        signal_id:        信号ID
        executor_id:      执行器ID
        is_fake:          是否是假仓位
        symbol:           交易对符号
        quote_currency:   计价币种（如"USDT"）
        ins_type:         交易品种类型（如现货、永续合约等）
        side:             仓位方向（多/空/平）
        cost_price:       开仓成本价
        amount:           仓位数量
        ratio:            占资金比例
        fee:              手续费
        friction:         摩擦成本
        leverage:         杠杆倍数
        open_time:        开仓时间
    """
    position_id: str  # 仓位ID

    strategy_id: str  # 策略ID
    strategy_instance_id: Union[str, Tuple]       # 策略实例ID

    symbol: str  # 交易标的
    quote_currency: str  # 计价货币
    ins_type: Optional[TradeInsType]  # 合约/现货类型
    asset_type: Optional[AssetType]  # 资产类型（如股票、期货、加密货币等）

    side: Optional[PositionSide]         # 仓位方向（多/空）
    cost_price: Decimal        # 开仓成本价
    amount: Decimal            # 当前价值
    ratio: Decimal             # 占资金比例

    close_price: Optional[Decimal] = None        # 平仓成本价
    current_price: Optional[Decimal] = None      # 当前价格
    total_amount: Decimal = Decimal("0")  # 累计价值
    total_back_amount: Decimal = Decimal("0")  # 累计回款金额
    total_sz: Decimal = Decimal("0")  # 累计仓位数量

    executor_id: Optional[str] = None # 执行器ID
    is_fake: bool = False # 是否是假仓位

    pnl: Decimal = Decimal("0")  # 盈亏
    fee: Decimal = Decimal("0")  # 手续费
    friction: Decimal = Decimal("0")  # 摩擦成本 特指合约资金费之类的磨损， 冲击成本不算
    leverage: Decimal = Decimal("1")  # 杠杆倍数，默认1倍
    open_time: datetime = datetime.now()  # 开仓时间

    # 订单执行状态跟踪
    executor_sz: Dict[str, Decimal] = field(default_factory=dict)  # 执行器的仓位大小
    order_states: Dict[str, OrderExecutionState] = field(default_factory=dict)  # 订单ID -> 执行状态

    @property
    def value(self):
        if self.current_price is None:
            return self.amount
        
        profit = (self.current_price - self.cost_price) * self.sz
        if self.side.is_short:
            profit = -profit
        return self.amount + profit + self.fee + self.friction

    @property
    def sz(self):
        return sum(self.executor_sz.values()) if self.executor_sz else Decimal('0')

    def __post_init__(self):
        # Required string fields validation
        if not self.position_id or not isinstance(self.position_id, str):
            raise ValueError("position_id is required and must be a string")
        if not self.strategy_id or not isinstance(self.strategy_id, str):
            raise ValueError("strategy_id is required and must be a string")
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol is required and must be a string")
        if not self.quote_currency or not isinstance(self.quote_currency, str):
            raise ValueError("quote_currency is required and must be a string")

        self.strategy_instance_id = str(self.strategy_instance_id)
        # Optional string field
        self.executor_id = str(self.executor_id) if self.executor_id is not None else None

        # Boolean field
        self.is_fake = bool(self.is_fake) if self.is_fake is not None else False

        # Convert enum types with null check
        self.ins_type = TradeInsType(self.ins_type) if self.ins_type is not None else None
        self.asset_type = AssetType(self.asset_type) if self.asset_type is not None else None
        self.side = PositionSide(self.side) if self.side is not None else None

        # Convert required decimal fields with null check
        self.ratio = Decimal(str(self.ratio)) if self.ratio is not None else Decimal('0')
        self.cost_price = Decimal(str(self.cost_price)) if self.cost_price is not None else Decimal('0')
        self.amount = Decimal(str(self.amount)) if self.amount is not None else Decimal('0')
        self.leverage = Decimal(str(self.leverage)) if self.leverage is not None else Decimal('1')
        
        # Convert optional decimal fields
        if self.close_price is not None:
            self.close_price = Decimal(str(self.close_price))
        if self.current_price is not None:
            self.current_price = Decimal(str(self.current_price))
        self.total_amount = Decimal(str(self.total_amount)) if self.total_amount is not None else Decimal('0')
        self.total_back_amount = Decimal(str(self.total_back_amount)) if self.total_back_amount is not None else Decimal('0')
        self.total_sz = Decimal(str(self.total_sz)) if self.total_sz is not None else Decimal('0')
        self.pnl = Decimal(str(self.pnl)) if self.pnl is not None else Decimal('0')
        self.fee = Decimal(str(self.fee)) if self.fee is not None else Decimal('0')
        self.friction = Decimal(str(self.friction)) if self.friction is not None else Decimal('0')

        # Ensure datetime type
        self.open_time = self.open_time if isinstance(self.open_time, datetime) else datetime.now()

        # Initialize default collections if None
        self.executor_sz = self.executor_sz if isinstance(self.executor_sz, dict) else {}
        self.order_states = self.order_states if isinstance(self.order_states, dict) else {}

        # Ensure all values in executor_sz are Decimal
        self.executor_sz = {k: Decimal(str(v)) if v is not None else Decimal('0') 
                           for k, v in self.executor_sz.items()}

        # Ensure all values in order_states are OrderExecutionState
        self.order_states = {k: v if isinstance(v, OrderExecutionState) else OrderExecutionState(k)
                            for k, v in self.order_states.items()}


@dataclass
class PositionInfo:
    active_amount: Decimal = Decimal("0")   # 活跃仓位
    active_ratio: Decimal = Decimal("0")    # 活跃比例
    positions: List[Position] = field(default_factory=list) # 仓位列表

