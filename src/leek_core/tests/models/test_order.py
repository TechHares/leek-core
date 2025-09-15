#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import copy
from datetime import datetime
from decimal import Decimal

from leek_core.models import (
    Order,
    OrderStatus,
    PositionSide,
    OrderType,
    TradeMode,
    TradeInsType,
    AssetType,
)


class TestOrderDeepcopy(unittest.TestCase):
    def test_order_deepcopy_extra_and_nested(self):
        order = Order(
            order_id="o1",
            position_id="p1",
            strategy_id="s1",
            strategy_instance_id="inst1",
            signal_id="sig1",
            exec_order_id="exec1",
            order_status=OrderStatus.SUBMITTED,
            signal_time=datetime.now(),
            order_time=datetime.now(),
            symbol="BTC",
            quote_currency="USDT",
            ins_type=TradeInsType.SWAP,
            asset_type=AssetType.CRYPTO,
            side=PositionSide.LONG,
            is_open=True,
            is_fake=False,
            order_amount=Decimal("10"),
            order_price=Decimal("50000"),
            ratio=Decimal("0.1"),
            order_type=OrderType.MarketOrder,
            settle_amount=Decimal("5"),
            execution_price=Decimal("49900"),
            sz=Decimal("1"),
            sz_value=Decimal("1"),
            fee=Decimal("-0.1"),
            pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            finish_time=None,
            friction=Decimal("0"),
            leverage=Decimal("2"),
            executor_id="ex1",
            trade_mode=TradeMode.ISOLATED,
            extra={
                "a": {"x": 1},
                "b": [1, 2, 3],
                "enum": PositionSide.LONG,
                "decimal": Decimal("1.23"),
            },
            market_order_id="m1",
        )
        copied = copy.deepcopy(order)
        order.extra["a"]["x"] = 999
        self.assertNotEqual(order.extra["a"]["x"], copied.extra["a"]["x"])
        print(order.extra["a"]["x"], copied.extra["a"]["x"])
        # 对象本身不相同
        self.assertIsNot(copied, order)
        # extra 被深拷贝
        self.assertIsNot(copied.extra, order.extra)
        # 嵌套可变对象也应为不同引用
        self.assertIsNot(copied.extra["a"], order.extra["a"])  # dict
        self.assertIsNot(copied.extra["b"], order.extra["b"])  # list
        # 枚举/Decimal 等不可变语义对象保持等值
        self.assertIs(copied.extra["enum"], PositionSide.LONG)
        self.assertEqual(copied.extra["decimal"], Decimal("1.23"))

        # 修改原对象嵌套应不影响副本
        order.extra["a"]["x"] = 999
        order.extra["b"].append(4)
        self.assertEqual(copied.extra["a"]["x"], 1)
        self.assertEqual(copied.extra["b"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()


