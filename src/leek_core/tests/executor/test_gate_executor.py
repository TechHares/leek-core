#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX WebSocket执行器连接测试
"""

import asyncio
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime

from leek_core.adapts import GateAdapter
from leek_core.executor.okx import OkxWebSocketExecutor, OkxRestExecutor
from leek_core.executor.base import WSStatus, WebSocketExecutor
from leek_core.models import Order, OrderStatus, PositionSide, OrderType, TradeMode, TradeInsType, AssetType
from leek_core.utils import get_logger

logger = get_logger(__name__)

class TestGateExecutorConnectionReal(unittest.TestCase):
    """OKX执行器真实连接测试
    
    运行前需要设置环境变量:
        export GATE_API_KEY=your_api_key
        export GATE_SECRET_KEY=your_secret_key
    """
    
    def setUp(self):
        self.adapter = GateAdapter(api_key="a46b69df9fee81168b3856e168ca99f3", secret_key="839f5bdcc72d1d50a898a3431fa3b4273e9b1ea9c4446a36b4a931235c70d2fe")
        
    def test_real_connection(self):
        result = self.adapter.get_futures_position_close("usdt", "CRV_USDT")
        print(f"result: {result}")
        if result and result.get("code") == "0":
            position_closes = result.get("data", [])
            # 通过text字段匹配我们的订单ID
            for close_record in position_closes:
                # if close_record.get("text") == "t-272617036989861888":
                # 从平仓记录获取准确的pnl和手续费
                pnl = close_record.get("pnl", "0")
                pnl_pnl = close_record.get("pnl_pnl", "0")
                pnl_fee = close_record.get("pnl_fee", "0")
                print(f"text: {close_record.get('text')}, pnl: {pnl}, pnl_pnl: {pnl_pnl}, pnl_fee: {pnl_fee}")