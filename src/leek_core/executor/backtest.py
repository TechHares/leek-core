#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测交易执行模块
"""

import random
from decimal import Decimal
from typing import List

from leek_core.models import Data, Field, OrderType, Order, FieldType, ChoiceType, OrderUpdateMessage,OrderStatus
from leek_core.utils import get_logger, decimal_quantize, generate_str

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

    def __init__(self, slippage: Decimal = 0.0, fee_type: int = 0, fee: Decimal = 0,
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

        self._holder_size = {}
        self._holder_price = {}


    def send_order(self, orders: List[Order]):
        """
        处理订单（参数补全、风控、成交模拟、推送）
        """
        for order in orders:
            logger.info(f"回测交易处理订单: {order.symbol} {order.side} {order.order_price} {order.order_amount}")
            key = (order.symbol, order.quote_currency, order.asset_type, order.ins_type)
            # 1. 计算成交价
            transaction_price = order.order_price
            if order.order_type == OrderType.MarketOrder and self.slippage > 0:
                slippage = Decimal(random.random()) * (2 * self.slippage) + (1 - self.slippage)
                transaction_price = decimal_quantize(transaction_price * slippage, 18)

            # 2. 计算成交量
            if order.is_open:
                transaction_volume = decimal_quantize(order.order_amount / transaction_price, 6)
            else:
                transaction_volume = Decimal(order.sz)
            if order.order_type == OrderType.LimitOrder:
                random_num = random.randint(int(self.limit_order_execution_rate), 100)
                transaction_volume = decimal_quantize(transaction_volume * random_num / 100, 6)

            pnl = 0
            if order.is_open:
                hold_size = self._holder_size.get(key, Decimal(0))
                hold_price = self._holder_price.get(key, Decimal(0))

                self._holder_price[key] =decimal_quantize((transaction_price * transaction_volume + hold_size * hold_price) / (hold_size + transaction_volume), 18)
                self._holder_size[key] = hold_size + transaction_volume
            else:
                self._holder_size[key] -= transaction_volume
                assert self._holder_size[key] >= 0, "交易数量不能大于持仓数量"
                pnl = (transaction_price - self._holder_price[key]) * transaction_volume * (1 if order.side.is_short else -1)
            
            # 3. 计算成交额
            transaction_amount = decimal_quantize(transaction_volume * transaction_price, 2, 1) if order.is_open else order.order_amount + pnl

            # 4. 计算手续费
            fee = Decimal(0)
            if self.fee_type == 0:
                fee = Decimal(0)
            elif self.fee_type == 1:
                fee = self.fee
            elif self.fee_type == 2:
                fee = transaction_amount * self.fee
            elif self.fee_type == 3:
                fee = transaction_volume * self.fee

            assert transaction_amount > 0, "交易金额不能为0"
            assert transaction_volume > 0, "交易数量不能为0"
            assert transaction_price > 0, "交易价格不能为0"

            
            msg = OrderUpdateMessage(
                order_id=order.order_id,
                order_status=OrderStatus.FILLED,
                market_order_id="F" + generate_str(),
                settle_amount=transaction_amount,
                execution_price=transaction_price,
                sz=transaction_volume,
                fee=-abs(decimal_quantize(fee, 10, 1)),
                pnl=pnl,
                unrealized_pnl=Decimal(0),
                friction=Decimal(0),
                finish_time=order.order_time,
                sz_value=Decimal("1"),
            )
            logger.info(f"回测交易结果: {msg.__dict__}")
            self._trade_callback(msg)

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
