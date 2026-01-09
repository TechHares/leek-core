#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
订单模型定义
"""
import copy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, List, Any


from .constants import PositionSide, OrderType, TradeMode, TradeInsType, OrderStatus, AssetType
from .transaction import Transaction, TransactionType


@dataclass
class Order:
    """
    订单数据结构。
    
    用于描述一笔交易订单的完整信息，覆盖下单、成交、结算等全生命周期。
    
    字段说明：
        order_id:         订单唯一ID（字符串）
        position_id:      仓位唯一ID（字符串）
        exec_order_id:    执行订单ID（字符串）
        signal_id:        信号ID（字符串）
        order_status:     订单状态，OrderStatus 枚举，如 CREATED、FILLED、CANCELED 等
        signal_time:      信号发出时间（datetime）
        order_time:       订单创建时间（datetime）
        symbol:           交易标的，如 "BTC"、"AAPL" 等
        quote_currency:   计价货币，如 "USDT"、"USD"、"CNY" 等
        ins_type:         交易品种类型，TradeInsType 枚举（现货/合约等）
        asset_type:       资产类型，AssetType 枚举（股票/期货/加密货币等）
        side:             交易方向，PositionSide 枚举（LONG/SHORT/NETURAL等）
        is_open:          是否为开仓订单，True=开仓，False=平仓
        order_amount:     下单数量/金额（Decimal）
        order_price:      下单价格（Decimal），市价单可为0
        order_type:       订单类型，OrderType 枚举（市价、限价等）
        ratio:            仓位比例
        settle_amount:    实际成交数量/金额（Decimal）
        execution_price:  实际成交均价（Decimal）
        fee:              手续费（Decimal）
        pnl:              已实现盈亏（Decimal）
        unrealized_pnl:   未实现盈亏（Decimal）
        finish_time:      订单完成时间（datetime，可为None）
        friction:         摩擦成本（Decimal）
        leverage:         杠杆倍数（Decimal，默认1倍）
        executor_id:      执行ID
        trade_mode:       交易模式
        sub_order_id:     子订单ID
        extra:            附加信息（扩展字段）
        market_order_id:  市场订单ID
    """
    order_id: str  # 订单ID
    position_id: str  # 仓位ID
    strategy_id: str  # 策略ID
    strategy_instance_id: str  # 策略实例ID
    signal_id: str  # 信号ID
    exec_order_id: str  # 执行订单ID
    order_status: OrderStatus  # 订单状态
    signal_time: datetime  # 信号时间
    order_time: datetime  # 订单时间

    symbol: str  # 交易标的
    quote_currency: str  # 计价货币
    ins_type: TradeInsType  # 合约/现货类型
    asset_type: AssetType  # 资产类型（如股票、期货、加密货币等）
    side: PositionSide  # 交易方向

    is_open: bool  # 是否开仓
    is_fake: bool  # 是否是虚拟仓位
    order_amount: Decimal  # 订单金额
    order_price: Decimal  # 订单价格
    ratio: Decimal  # 比例
    order_type: OrderType = None  # 订单类型

    settle_amount: Decimal = None  # 实际成交金额
    execution_price: Decimal = None  # 实际成交价格
    sz: Decimal = None  # 订单数量
    sz_value: Decimal = 1  # 订单数量乘数，用于计算和实际下单的之间的差异换算，例如 1股和1手的换算, 这里=0.01
    fee: Decimal = None  # 手续费
    pnl: Decimal = None  # 已实现盈亏
    unrealized_pnl: Decimal = None  # 未实现盈亏
    finish_time: datetime = None  # 完成时间
    friction: Decimal = Decimal(0)  # 摩擦成本 特指合约资金费之类的磨损， 冲击成本不算
    leverage: Decimal = Decimal(1)  # 杠杆倍数，默认1倍

    executor_id: str = None         # 执行ID
    trade_mode: TradeMode = None     # 交易模式
    extra: dict[str, str] = None     # 附加信息
    market_order_id: str = None      # 市场订单ID

    @property
    def asset_key(self) -> str:
        return f"{self.symbol}_{self.quote_currency}_{self.ins_type.value}_{self.asset_type.value}_{self.side.value}"

    def settle(self, activate_amount: Decimal, factor: Decimal=Decimal(1)) -> Transaction:
        if (self.settle_amount is None or self.settle_amount <= 0 or not self.order_status.is_finished):
            return
        return Transaction(
            strategy_id=self.strategy_id,
            strategy_instance_id=self.strategy_instance_id,
            position_id=self.position_id,
            exec_order_id=self.exec_order_id,
            order_id=self.order_id,
            signal_id=self.signal_id,
            asset_key=self.asset_key,
            type=TransactionType.TRADE,
            amount=self.settle_amount * factor,
            balance_before=activate_amount,
            balance_after=activate_amount + self.settle_amount * factor,
            desc=f"{'扣款' if self.is_open else '回款'}[{self.order_id}]: {self.settle_amount}",
        )
    def settle_fee(self, activate_amount: Decimal, factor: Decimal=Decimal(1)) -> Transaction:
        if (self.fee is None or self.fee == 0 or not self.order_status.is_finished):
            return
        return Transaction(
            strategy_id=self.strategy_id,
            strategy_instance_id=self.strategy_instance_id,
            position_id=self.position_id,
            exec_order_id=self.exec_order_id,
            order_id=self.order_id,
            signal_id=self.signal_id,
            asset_key=self.asset_key,
            type=TransactionType.FEE,
            amount=self.fee * factor,
            balance_before=activate_amount,
            balance_after=activate_amount + self.fee * factor,
            desc=f"手续费[{self.order_id}]: {self.fee}",
        )
        
    def __deepcopy__(self, memo):
        return Order(
            order_id=self.order_id,
            position_id=self.position_id,
            strategy_id=self.strategy_id,
            strategy_instance_id=self.strategy_instance_id,
            signal_id=self.signal_id,
            exec_order_id=self.exec_order_id,
            order_status=self.order_status,
            signal_time=self.signal_time,
            order_time=self.order_time,
            symbol=self.symbol,
            quote_currency=self.quote_currency,
            ins_type=self.ins_type,
            asset_type=self.asset_type,
            side=self.side,
            is_open=self.is_open,
            is_fake=self.is_fake,
            order_amount=self.order_amount,
            order_price=self.order_price,
            ratio=self.ratio,
            order_type=self.order_type,
            settle_amount=self.settle_amount,
            execution_price=self.execution_price,
            sz=self.sz,
            sz_value=self.sz_value,
            fee=self.fee,
            pnl=self.pnl,
            unrealized_pnl=self.unrealized_pnl,
            finish_time=self.finish_time,
            friction=self.friction,
            leverage=self.leverage,
            executor_id=self.executor_id,
            trade_mode=self.trade_mode,
            extra=copy.deepcopy(self.extra, memo) if self.extra is not None else None,
            market_order_id=self.market_order_id,
        )
    
@dataclass
class OrderUpdateMessage:
    order_id: str
    order_status: OrderStatus = None  # 订单状态
    settle_amount: Decimal = None  # 实际成交金额
    execution_price: Decimal = None  # 实际成交价格
    sz: Decimal = None  # 订单数量
    fee: Decimal = None  # 手续费
    pnl: Decimal = None  # 已实现盈亏
    unrealized_pnl: Decimal = None  # 未实现盈亏
    finish_time: datetime = None  # 完成时间
    friction: Decimal = Decimal(0)  # 摩擦成本 特指合约资金费之类的磨损， 冲击成本不算
    sz_value: Decimal = 1

    extra: dict[str, str] = None     # 附加信息
    market_order_id: str = None      # 市场订单ID

@dataclass
class ExecutionAsset:
    """
    资产约束模型，用于描述交易信号中的具体资产及其配置信息。
    
    该模型在风控、仓位管理和交易执行等环节中用于资产筛选与约束。它包含了资产类型、
    交易品种、交易标的、交易方向、建议价格、持仓比例等关键信息。

    属性:
        asset_type (AssetType): 资产类型，如股票、期货、加密货币等
        ins_type (TradeInsType): 交易品种类型，如现货、永续合约等
        symbol (str): 交易对或合约标识，例如 "BTC"、"AAPL"
        side (PositionSide): 多空方向，如 LONG/SHORT/NEUTRAL
        price (Decimal): 交易价格
        ratio (Decimal): 仓位比例
        amount (Decimal, optional): 交易金额
        quote_currency (str, optional): 计价币种，如 USDT、USD、CNY
        extra (Any, optional): 其他扩展信息，如信号置信度、触发条件、备注等
    """
    asset_type: AssetType      # 资产类型（如股票、期货、加密货币等），参见 AssetType 枚举
    ins_type: TradeInsType     # 交易品种类型（如现货、永续合约等），参见 TradeInsType 枚举
    symbol: str               # 交易对或合约标识，例如 "BTC"、"AAPL"

    side: PositionSide        # 多空方向，PositionSide 枚举（LONG/SHORT/NEUTRAL等）
    price: Decimal            # 交易价格
    is_open: bool             # 是否开仓

    ratio: Decimal            # 此次出手仓位比例
    sz: Decimal = None        # 订单数量
    amount: Decimal = None    # 交易金额
    virtual_sz: Decimal = None # 虚拟仓位大小

    quote_currency: str = None# 计价币种，如 USDT、USD、CNY
    extra: Any = None         # 其他扩展信息（如信号置信度、触发条件、备注等，可选）
    position_id: str = None   # 仓位ID
    actual_pnl: Decimal = None # 实际盈亏
    virtual_pnl: Decimal = None # 虚拟仓位盈亏
    executor_sz: Dict[str, Decimal] = None  # 执行器的仓位大小

    @property
    def asset_key(self) -> str:
        return f"{self.symbol}_{self.quote_currency}_{self.ins_type.value}_{self.asset_type.value}_{self.side.value}"

@dataclass
class ExecutionContext:
    """
    执行上下文信息，用于在订单执行过程中传递上下文信息。
    
    该类包含了订单执行所需的关键信息，包括策略标识、目标执行器以及待执行的资产列表。

    属性:
        strategy_id (str): 策略的唯一标识符
        strategy_instance_id (str): 策略实例的唯一标识符
        target_executor_id (str): 目标执行器的ID
        execution_assets (List[ExecutionAsset]): 待执行的资产列表
    """
    context_id: str
    signal_id: str
    strategy_id: str
    strategy_instance_id: str
    strategy_cls: str
    target_executor_id: str

    leverage: int
    order_type: OrderType
    trade_type: TradeInsType
    trade_mode: TradeMode

    execution_assets: List[ExecutionAsset]
    created_time: datetime

    actual_ratio: Decimal = None
    actual_amount: Decimal = None
    actual_pnl: Decimal = None
    is_finish: bool = False
    extra: dict[str, str] = None


    @property
    def open_amount(self) -> Decimal:
        """
        计算所有执行资产的总金额。

        Returns:
            Decimal: 所有资产金额的总和。如果某个资产没有设置金额，则不计入总和。
        """
        total = Decimal('0')
        for asset in self.execution_assets:
            if asset.is_open:
                total += asset.amount
        return total

    @property
    def open_ratio(self) -> Decimal:
        """
        计算所有执行资产的总比例。

        Returns:
            Decimal: 所有资产比例的总和。
        """
        total = Decimal('0')
        for asset in self.execution_assets:
            if asset.is_open:
                total += asset.ratio
        return total

    @property
    def close_amount(self) -> Decimal:
        """
        计算所有执行资产的总金额。
        Returns:
            Decimal: 所有资产金额的总和。如果某个资产没有设置金额，则不计入总和。
        """
        total = Decimal('0')
        for asset in self.execution_assets:
            if not asset.is_open:
                total += asset.amount
        return total

    @property
    def close_ratio(self) -> Decimal:
        """
        计算所有执行资产的总比例。
        Returns:
            Decimal: 所有资产比例的总和。
        """
        total = Decimal('0')
        for asset in self.execution_assets:
            if not asset.is_open:
                total += asset.ratio
        return total
