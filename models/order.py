#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
订单模型定义
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List

from .constants import PositionSide, OrderType, TradeMode, TradeInsType, OrderStatus, AssetType


@dataclass
class Order:
    """
    订单数据结构。
    
    用于描述一笔交易订单的完整信息，覆盖下单、成交、结算等全生命周期。
    
    字段说明：
        order_id:         订单唯一ID（字符串）
        position_id:      仓位唯一ID（字符串）
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
        sub_orders:       子订单列表（如分批成交明细，Optional[List[SubOrder]]）
        settle_amount:    实际成交数量/金额（Decimal）
        execution_price:  实际成交均价（Decimal）
        fee:              手续费（Decimal）
        pnl:              已实现盈亏（Decimal）
        unrealized_pnl:   未实现盈亏（Decimal）
        finish_time:      订单完成时间（datetime，可为None）
        friction:         摩擦成本（Decimal）
        leverage:         杠杆倍数（Decimal，默认1倍）
    """
    order_id: str  # 订单ID
    position_id: str  # 仓位ID
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
    order_type: OrderType = None  # 订单类型
    ratio: Decimal = Decimal(0)  # 仓位比例

    sub_orders: Optional[List['SubOrder']] = None  # 子订单列表
    settle_amount: Decimal = None  # 实际成交金额
    execution_price: Decimal = None  # 实际成交价格
    fee: Decimal = None  # 手续费
    pnl: Decimal = None  # 已实现盈亏
    unrealized_pnl: Decimal = None  # 未实现盈亏
    finish_time: datetime = None  # 完成时间
    friction: Decimal = Decimal(0)  # 摩擦成本 特指合约资金费之类的磨损， 冲击成本不算
    leverage: Decimal = Decimal(1)  # 杠杆倍数，默认1倍


@dataclass
class SubOrder(Order):
    """
    子订单数据结构

    属性：
        execution_id:    执行ID
        trade_mode:      交易模式
        sub_order_id:    子订单ID
        extra:           附加信息（扩展字段）
        market_order_id: 市场订单ID
    """
    execution_id: str = None         # 执行ID
    trade_mode: TradeMode = None     # 交易模式
    sub_order_id: str = None         # 子订单ID
    extra: dict[str, str] = None     # 附加信息
    market_order_id: str = None      # 市场订单ID
