#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List

from .signal import Signal
from .constants import TradeInsType, AssetType
from .order import Order


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
class Position:
    """
    表示一个策略在一个交易器下、一个单独标的的仓位。

    属性:
        position_id:      仓位ID
        strategy_id:      策略ID
        signal_id:        信号ID
        executor_id:      执行器ID
        order_id:         主订单ID
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
        orders:           相关订单列表
        signals:          相关信号列表
    """
    position_id: str  # 仓位ID

    strategy_id: str  # 策略ID
    symbol: str  # 交易标的
    quote_currency: str  # 计价货币
    ins_type: TradeInsType  # 合约/现货类型
    asset_type: AssetType  # 资产类型（如股票、期货、加密货币等）

    side: PositionSide         # 仓位方向（多/空/平）
    cost_price: Decimal        # 开仓成本价
    amount: Decimal            # 仓位数量
    ratio: Decimal             # 占资金比例
    executing_amount: Decimal = None  # 仓位数量
    executing_ratio: Decimal = None   # 占资金比例

    executor_id: str = None # 执行器ID
    order_id: str = None  # 主订单ID
    is_fake: bool = False # 是否是假仓位

    pnl: Decimal = Decimal("0")  # 盈亏
    fee: Decimal = Decimal("0")  # 手续费
    friction: Decimal = Decimal("0")  # 摩擦成本 特指合约资金费之类的磨损， 冲击成本不算
    leverage: Decimal = Decimal("1")  # 杠杆倍数，默认1倍
    open_time: datetime = datetime.now()  # 开仓时间
    orders: Optional[List[Order]] = field(default=list)  # 相关订单列表
    signals: Optional[List[Signal]] = field(default=list)  # 相关信号列表


@dataclass
class PositionContext:
    active_amount: Decimal = Decimal("0")   # 活跃仓位
    active_ratio: Decimal = Decimal("0")    # 活跃比例
    positions: List[Position] = field(default_factory=list) # 仓位列表

