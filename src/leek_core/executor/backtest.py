#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测交易执行模块。

新方案在原瞬时市价撮合基础上,新增限价单挂单 / 按 bar OHLC 撮合 / 标准撤单回调。
状态机与实盘对齐: CREATED → SUBMITTED → FILLED / CANCELED。
"""

import random
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from leek_core.models import (
    ChoiceType,
    Data,
    Field,
    FieldType,
    Order,
    OrderStatus,
    OrderType,
    OrderUpdateMessage,
    PositionSide,
)
from leek_core.utils import decimal_quantize, generate_str, get_logger

from .base import Executor

logger = get_logger(__name__)


class BacktestExecutor(Executor):
    """
    回测交易执行器。

    撮合规则:
        市价单: 立即按 order_price ± 滑点成交
        限价单: 进入 _pending_orders, 由 on_bar(data) 在新 bar 到达时按 OHLC 判断
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
              type=FieldType.FLOAT, default=100, min=1, max=100, required=True),
    ]
    init_params += Executor.init_params

    def __init__(self, slippage: Decimal = 0.0, fee_type: int = 0, fee: Decimal = 0,
                 limit_order_execution_rate: int = 100, check_hold_size: bool = True):
        self.slippage = Decimal(slippage)
        self.check_hold_size = bool(check_hold_size)
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

        # 持仓追踪(无需 wrapper, 用于回测内部 hold 校验)
        self._holder_size = {}
        self._holder_price = {}

        # 新方案: 未成交限价单池, key=order_id
        self._pending_orders: Dict[str, Order] = {}

    # ============================================================
    # 公开接口
    # ============================================================

    def send_order(self, orders: List[Order]):
        """按 order_type 分支处理"""
        for order in orders:
            logger.info(f"回测下单: {order}")
            if order.order_type == OrderType.MarketOrder:
                self._fill_market(order)
            elif order.order_type == OrderType.LimitOrder:
                self._enqueue_limit(order)
            else:
                # 未识别类型按市价处理(向后兼容)
                logger.warning(f"未知 order_type {order.order_type}, 按市价处理")
                self._fill_market(order)

    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        撤单: 从 _pending_orders 移除并回调标准 OrderUpdateMessage(CANCELED)。

        kwargs 可能包含 leek_order_id: 来自 ExecutorManager 的内部 ID 兜底。
        """
        leek_id = kwargs.get("leek_order_id") or order_id
        order = self._pending_orders.pop(leek_id, None)
        if order is None:
            # 也按 market_order_id 找一次
            for k, o in list(self._pending_orders.items()):
                if o.market_order_id == order_id:
                    order = self._pending_orders.pop(k)
                    break
        if order is None:
            logger.warning(f"回测撤单失败,挂单不存在: order_id={order_id}, leek_id={leek_id}")
            return
        msg = OrderUpdateMessage(
            order_id=order.order_id,
            order_status=OrderStatus.CANCELED,
            settle_amount=Decimal("0"),
            execution_price=order.order_price,
            sz=Decimal("0"),
            fee=Decimal("0"),
            pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            finish_time=datetime.now(),
            friction=Decimal("0"),
        )
        logger.info(f"回测撤单成功: {order.order_id}")
        self._trade_callback(msg)

    def on_bar(self, data: Data):
        """
        新 bar 到达时由引擎调用, 尝试撮合 _pending_orders 中可成交的限价单。

        撮合条件 (strict 模式):
            LONG  + is_open=True (buy):  bar.low  <= order_price  → fill_price = min(price, bar.open)
            LONG  + is_open=False (sell): bar.high >= order_price → fill_price = max(price, bar.open)
            SHORT + is_open=True (sell):  bar.high >= order_price → fill_price = max(price, bar.open)
            SHORT + is_open=False (buy):  bar.low  <= order_price → fill_price = min(price, bar.open)
        """
        if not self._pending_orders:
            return
        symbol = getattr(data, 'symbol', None)
        quote = getattr(data, 'quote_currency', None)
        if symbol is None:
            return
        bar_low = self._extract(data, 'low')
        bar_high = self._extract(data, 'high')
        bar_open = self._extract(data, 'open')
        if bar_low is None or bar_high is None or bar_open is None:
            return

        for order_id, order in list(self._pending_orders.items()):
            if order.symbol != symbol or (quote and order.quote_currency != quote):
                continue
            fill = self._try_match_limit(order, bar_low, bar_high, bar_open)
            if fill is None:
                continue
            fill_price, _ = fill
            self._pending_orders.pop(order_id, None)
            self._fill_limit_at(order, fill_price)

    # ============================================================
    # 内部撮合
    # ============================================================

    def _enqueue_limit(self, order: Order):
        """限价单进入挂单池, 等待 on_bar 撮合"""
        if not order.order_price or order.order_price <= 0:
            logger.warning(f"限价单缺少 order_price, 改为市价处理: {order.order_id}")
            self._fill_market(order)
            return
        self._pending_orders[order.order_id] = order
        # 不发 OrderUpdateMessage: ExecutorContext.send_order 会自动把 CREATED → SUBMITTED 并发 ORDER_UPDATED

    def _try_match_limit(self, order: Order, bar_low: Decimal, bar_high: Decimal,
                         bar_open: Decimal) -> Optional[Tuple[Decimal, Decimal]]:
        """判断挂单是否在本 bar 内成交, 返回 (fill_price, qty) 或 None"""
        price = order.order_price
        if order.side == PositionSide.LONG and order.is_open:
            # buy limit
            if bar_low <= price:
                return (min(price, bar_open), order.order_amount)
        elif order.side == PositionSide.LONG and not order.is_open:
            # sell long: 平多
            if bar_high >= price:
                return (max(price, bar_open), order.order_amount)
        elif order.side == PositionSide.SHORT and order.is_open:
            # sell short: 开空
            if bar_high >= price:
                return (max(price, bar_open), order.order_amount)
        elif order.side == PositionSide.SHORT and not order.is_open:
            # buy to cover: 平空
            if bar_low <= price:
                return (min(price, bar_open), order.order_amount)
        return None

    def _fill_limit_at(self, order: Order, fill_price: Decimal):
        """限价单成交: 按 fill_price 计算成交量、手续费、回调"""
        key = (order.symbol, order.quote_currency, order.asset_type, order.ins_type)

        # 限价单不加滑点
        transaction_price = fill_price

        # 成交量
        if order.is_open:
            transaction_volume = decimal_quantize(order.order_amount * order.leverage / transaction_price, 6)
        else:
            transaction_volume = Decimal(order.sz)

        # limit_order_execution_rate 部分成交
        rate = random.randint(int(self.limit_order_execution_rate), 100)
        transaction_volume = decimal_quantize(transaction_volume * rate / 100, 6)

        pnl = self._update_hold_and_pnl(order, key, transaction_price, transaction_volume)

        transaction_amount = (
            decimal_quantize(transaction_volume * transaction_price / order.leverage, 2, 1)
            if order.is_open else order.order_amount + pnl
        )

        fee = self._calc_fee(transaction_amount, transaction_volume)

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
            finish_time=datetime.now(),
            sz_value=Decimal("1"),
        )
        logger.info(f"回测限价成交: {order.order_id} @ {transaction_price} sz={transaction_volume}")
        self._trade_callback(msg)

    def _fill_market(self, order: Order):
        """市价单立即成交(沿用原逻辑)"""
        key = (order.symbol, order.quote_currency, order.asset_type, order.ins_type)
        transaction_price = order.order_price
        if order.order_type == OrderType.MarketOrder and self.slippage > 0:
            slippage = Decimal(random.random()) * (2 * self.slippage) + (1 - self.slippage)
            transaction_price = decimal_quantize(transaction_price * slippage, 18)

        if order.is_open:
            transaction_volume = decimal_quantize(order.order_amount * order.leverage / transaction_price, 6)
        else:
            transaction_volume = Decimal(order.sz)

        pnl = self._update_hold_and_pnl(order, key, transaction_price, transaction_volume)

        transaction_amount = (
            decimal_quantize(transaction_volume * transaction_price / order.leverage, 2, 1)
            if order.is_open else order.order_amount + pnl
        )

        fee = self._calc_fee(transaction_amount, transaction_volume)

        assert transaction_price > 0, f"{self.instance_id}交易价格不能为0({transaction_price}, {transaction_volume}, {transaction_amount})"
        assert transaction_volume > 0, f"{self.instance_id}交易数量不能为0({transaction_price}, {transaction_volume}, {transaction_amount})"
        assert order.is_fake or transaction_amount > 0, f"{self.instance_id}交易金额不能为0({transaction_price}, {transaction_volume}, {transaction_amount})"

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
        logger.info(f"回测市价成交: {msg.__dict__}")
        self._trade_callback(msg)

    def _update_hold_and_pnl(self, order: Order, key, transaction_price: Decimal,
                             transaction_volume: Decimal) -> Decimal:
        """更新内部持仓追踪 + 计算 pnl(沿用原逻辑)"""
        pnl = Decimal(0)
        if self.check_hold_size and order.is_fake is False:
            if order.is_open:
                hold_size = self._holder_size.get(key, Decimal(0))
                hold_price = self._holder_price.get(key, Decimal(0))
                self._holder_price[key] = decimal_quantize(
                    (transaction_price * transaction_volume + hold_size * hold_price) /
                    (hold_size + transaction_volume), 18)
                self._holder_size[key] = hold_size + transaction_volume
            elif key in self._holder_size:
                hold_size = self._holder_size[key]
                self._holder_size[key] -= transaction_volume
                assert hold_size >= transaction_volume, (
                    f"{self.instance_id}交易数量不能大于持仓数量: {hold_size} - {transaction_volume}"
                )
                pnl = ((transaction_price - self._holder_price[key]) * transaction_volume
                       * (1 if order.side.is_short else -1))
        return pnl

    def _calc_fee(self, transaction_amount: Decimal, transaction_volume: Decimal) -> Decimal:
        if self.fee_type == 0:
            return Decimal(0)
        if self.fee_type == 1:
            return self.fee
        if self.fee_type == 2:
            return transaction_amount * self.fee
        if self.fee_type == 3:
            return transaction_volume * self.fee
        return Decimal(0)

    @staticmethod
    def _extract(data: Data, field_name: str) -> Optional[Decimal]:
        """从 Data 安全取价格字段"""
        val = getattr(data, field_name, None)
        if val is None:
            return None
        return Decimal(val) if not isinstance(val, Decimal) else val
