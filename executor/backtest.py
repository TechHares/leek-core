#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测交易执行模块
"""

import random
from decimal import Decimal

from models import Data, Field, OrderType, SubOrder, FieldType, ChoiceType
from utils import get_logger, decimal_quantize
from .base import Executor

logger = get_logger(__name__)


class BacktestExecutor(Executor):
    """
    回测交易执行器
    """
    display_name = "回测交易"
    just_backtest = True
    init_params = [
        Field(name="slippage", label="滑点幅度(0.0 - 1.0)",
              description="成交价会在该幅度内随机产生 [(1-slippage)*报价, (1+slippage)*报价] 仅针对市价单有效",
              type=FieldType.FLOAT, default=0.0, min=0, max=1, required=True),
        Field(name="fee_type", label="费用收取方式", description="", type=FieldType.RADIO, default=0, min=0, max=3,
              required=True, choices=[(0, "无费用"), (1, "固定费用"), (2, "成交额固定比例"), (3, "单位成交固定费用")],
              choice_type=ChoiceType.INT),
        Field(name="fee", label="费率",
              description="用收取方式类型无费用时无效， 固定费用时表示固定费用， 成交额固定比例时表示固定比例",
              type=FieldType.FLOAT, default=0.0, min=0, max=1, required=True),
        Field(name="limit_order_execution_rate", label="限价单成交率(1-100)",
              description="仅针对限价单有效, 成交额=报单额*random(成交率% ~ 1)",
              type=FieldType.FLOAT, default=100, min=1, max=100, required=True)
    ]
    init_params += Executor.init_params

    def __init__(self, instance_id: str=None, name: str=None, slippage: Decimal = 0.0, fee_type: int = 0, fee: Decimal = 0,
                 limit_order_execution_rate: int = 100):
        """
        初始化回测交易器
        
        参数:
            bus: 事件总线
            slippage: 滑点幅度，0.0 - 1.0 成交价会在该幅度内随机产生 [(1-slippage)*报价, (1+slippage)*报价] 仅针对市价单有效
            fee_type: ，0 无费用，1 固定费用，2 成交额固定比例，3 单位成交固定费用
            fee: 费率， 费用收取方式类型 0 时无效， 1 时表示固定费用， 2 时表示固定比例
            limit_order_execution_rate: 限价单成交率， 1 - 100, 仅针对限价单有效, 成交额=报单额*random(limit_order_execution_rate% ~ 1)
        """
        super().__init__(instance_id, name)
        self.slippage = Decimal(slippage)
        if self.slippage > 1:
            self.slippage = Decimal(1)
        if self.slippage < 0:
            self.slippage = Decimal(0)

        self.fee_type = int(fee_type)
        if self.fee_type not in [0, 1, 2, 3]:
            self.fee_type = 0
        self.fee = Decimal(fee)

        self.limit_order_execution_rate = Decimal(limit_order_execution_rate)
        if self.limit_order_execution_rate < 1:
            self.limit_order_execution_rate = 1
        if self.limit_order_execution_rate > 100:
            self.limit_order_execution_rate = 100


    def send_order(self, order: SubOrder) -> SubOrder:
        """
        处理订单（参数补全、风控、成交模拟、推送）
        """
        # 参数补全与风控（风格与实盘一致）
        if getattr(order, 'lever', None) is None:
            order.lever = 1
        if getattr(order, 'type', None) is None:
            order.type = OrderType.LimitOrder
        if getattr(order, 'trade_ins_type', None) is None:
            order.trade_ins_type = 3
        if getattr(order, 'trade_mode', None) is None:
            order.trade_mode = 'isolated'

        logger.info(f"回测交易处理订单: {order.symbol} {order.side} {order.price} {order.amount}")

        pos_trade = Data()
        pos_trade.order_id = getattr(order, 'order_id', None)
        pos_trade.strategy_id = getattr(order, 'strategy_id', None)
        pos_trade.symbol = order.symbol
        pos_trade.lever = order.lever
        pos_trade.side = order.side
        pos_trade.ct_val = 1
        pos_trade.cancel_source = ""
        pos_trade.state = "filled"  # 默认全部成交

        # 1. 计算成交价
        pos_trade.transaction_price = order.price
        if order.type == OrderType.MarketOrder and self.slippage > 0:
            slippage = Decimal(random.random()) * (2 * self.slippage) + (1 - self.slippage)
            pos_trade.transaction_price = decimal_quantize(order.price * slippage, 8)

        # 2. 计算成交量
        if getattr(order, 'sz', None) is None:
            if pos_trade.transaction_price == 0:
                pos_trade.sz = 0
            else:
                pos_trade.sz = decimal_quantize(order.amount / pos_trade.transaction_price, 6)
        else:
            pos_trade.sz = Decimal(order.sz)
        pos_trade.transaction_volume = pos_trade.sz
        if order.type == OrderType.LimitOrder:
            random_num = random.randint(int(self.limit_order_execution_rate), 100)
            pos_trade.transaction_volume = decimal_quantize(pos_trade.sz * random_num / 100, 6)

        # 3. 计算成交额
        pos_trade.transaction_amount = decimal_quantize(pos_trade.transaction_volume * pos_trade.transaction_price, 2,
                                                        1)

        # 4. 计算手续费
        fee = Decimal(0)
        if self.fee_type == 0:
            fee = Decimal(0)
        elif self.fee_type == 1:
            fee = self.fee
        elif self.fee_type == 2:
            fee = pos_trade.transaction_amount * self.fee
        elif self.fee_type == 3:
            fee = pos_trade.transaction_volume * self.fee
        pos_trade.fee = abs(decimal_quantize(fee, 10, 1))
        pos_trade.pnl = 0

        logger.info(f"回测交易结果: {pos_trade.__dict__}")
        self._trade_callback(pos_trade)
        return order

    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        回测撤单接口，模拟撤单推送
        """
        logger.info(f"回测撤单: order_id={order_id}, symbol={symbol}")
        pos_trade = Data()
        pos_trade.order_id = order_id
        pos_trade.symbol = symbol
        pos_trade.state = "canceled"
        pos_trade.cancel_source = "backtest"
        self._trade_callback(pos_trade)
        return pos_trade
