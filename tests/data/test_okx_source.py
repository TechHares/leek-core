#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX WebSocket 数据源测试用例
使用 unittest.mock 模拟 WebSocket 连接和交互
"""

import asyncio
import json
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import time

from data.okx_source import OkxDataSource
from models import TimeFrame
# Assuming utils.get_logger exists and works without full setup for tests
from utils import get_logger

logger = get_logger(__name__)

class OkxSourceTest(unittest.TestCase):

    def setUp(self):
        # Create a new event loop for each test to ensure isolation
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Mock callback function
        self.mock_callback = AsyncMock() # Use AsyncMock for async callbacks

    def tearDown(self):
        # Clean up the event loop
        self.loop.close()
        asyncio.set_event_loop(None) # Reset the global event loop policy if needed

    def test_connect_disconnect_ping(self):
        """测试连接、断开连接和 ping 任务管理"""
        source = OkxDataSource(work_flag="2")

        async def run_test():
            # Test connection
            connected = source.connect()
            self.assertTrue(connected)
            self.assertTrue(source.is_connected)
            self.assertIsNotNone(source._ping_task)
            self.assertFalse(source._ping_task.done())

            # Give ping loop a chance to run and potentially send ping
            await asyncio.sleep(0.1)
            print("="*10+ "断开连接" +"="*10)
            # Test disconnection
            disconnected = source.disconnect()
            self.assertTrue(disconnected)
            self.assertFalse(source.is_connected)
            self.assertEqual(source._ping_task, None)
            # Resetting source._ping_task to None might happen slightly after await returns
            # So check cancellation/done status mainly

        self.loop.run_until_complete(run_test())

    def test_subscribe_unsubscribe(self):
        """测试订阅和取消订阅"""
        source = OkxDataSource(work_flag="2")
        def callback(t, d):
            print(t, d)
        source.set_callback(callback)

        print(source.get_supported_parameters())
        async def run_test():
            # Connect
            connected = source.connect()
            self.assertTrue(connected)
            await asyncio.sleep(5)

            source.subscribe(symbol="BTC", timeframe=TimeFrame.H4, )
            await asyncio.sleep(15)
            source.unsubscribe(symbol="BTC", timeframe=TimeFrame.H4, )
            await asyncio.sleep(5)
            source.disconnect()
            await asyncio.sleep(5)

            # Test subscribe
        self.loop.run_until_complete(run_test())

if __name__ == "__main__":
    unittest.main()
