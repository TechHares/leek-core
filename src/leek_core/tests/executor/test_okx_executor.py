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

from leek_core.executor.okx import OkxWebSocketExecutor, OkxRestExecutor
from leek_core.executor.base import WSStatus, WebSocketExecutor
from leek_core.models import Order, OrderStatus, PositionSide, OrderType, TradeMode, TradeInsType, AssetType
from leek_core.utils import get_logger

logger = get_logger(__name__)

class TestOkxExecutorConnectionReal(unittest.TestCase):
    """OKX执行器真实连接测试
    
    运行前需要设置环境变量:
        export OKX_API_KEY=your_api_key
        export OKX_SECRET_KEY=your_secret_key
        export OKX_PASSPHRASE=your_passphrase
    """
    
    def setUp(self):
        self.executor = OkxWebSocketExecutor(
            api_key=os.getenv("OKX_API_KEY"),
            secret_key=os.getenv("OKX_SECRET_KEY"),
            passphrase=os.getenv("OKX_PASSPHRASE")
        )
        
    def tearDown(self):
        """清理：停止连接"""
        try:
            if self.executor._status == WSStatus.CONNECTED:
                self.executor.on_stop()
        except Exception as e:
            logger.warning(f"tearDown异常: {e}")
            
    def test_real_connection(self):
        """测试真实WebSocket连接"""
        logger.info("开始测试真实WebSocket连接...")
        
        try:
            # 启动连接
            self.executor.on_start()
            time.sleep(5)
            # 再次验证状态
            self.assertEqual(self.executor.status, WSStatus.CONNECTED)
            
        except TimeoutError as e:
            logger.error(f"连接超时: {e}")
            self.fail(f"WebSocket连接超时: {e}")
        except Exception as e:
            logger.error(f"连接异常: {e}")
            self.fail(f"WebSocket连接异常: {e}")


class TestOkxRestExecutor(unittest.TestCase):
    """OKX REST执行器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.api_key = "test_api_key"
        self.secret_key = "test_secret_key"
        self.passphrase = "test_passphrase"
        self.executor = OkxRestExecutor(
            api_key=self.api_key,
            secret_key=self.secret_key,
            passphrase=self.passphrase,
            slippage_level=4,
            ccy="USDT"
        )
        
    def tearDown(self):
        """测试后清理"""
        try:
            if self.executor._polling_thread and self.executor._polling_thread.is_alive():
                self.executor.on_stop()
                time.sleep(0.5)  # 等待线程停止
        except Exception as e:
            logger.warning(f"tearDown异常: {e}")
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor.api_key, self.api_key)
        self.assertEqual(self.executor.secret_key, self.secret_key)
        self.assertEqual(self.executor.passphrase, self.passphrase)
        self.assertEqual(self.executor.slippage_level, 4)
        self.assertEqual(self.executor.ccy, "USDT")
        self.assertIsNotNone(self.executor.adapter)
        self.assertEqual(len(self.executor._pending_orders), 0)
        self.assertIsNone(self.executor._polling_thread)
    
    def test_on_start_stop(self):
        """测试启动和停止"""
        # 启动执行器
        self.executor.on_start()
        self.assertIsNotNone(self.executor._polling_thread)
        self.assertTrue(self.executor._polling_thread.is_alive())
        
        # 等待一小段时间确保线程启动
        time.sleep(0.5)
        
        # 停止执行器
        self.executor.on_stop()
        time.sleep(0.5)
        
        # 验证线程已停止
        self.assertFalse(self.executor._polling_thread.is_alive())
        self.assertEqual(len(self.executor._pending_orders), 0)
    
    def _create_test_order(self, order_id="test_order_001", symbol="BTC", 
                          order_price=Decimal("50000"), order_amount=Decimal("1000"),
                          ins_type=TradeInsType.SWAP):
        """创建测试订单"""
        return Order(
            order_id=order_id,
            position_id="pos_001",
            strategy_id="strategy_001",
            strategy_instance_id="strategy_instance_001",
            signal_id="signal_001",
            exec_order_id="exec_order_001",
            order_status=OrderStatus.CREATED,
            signal_time=datetime.now(),
            order_time=datetime.now(),
            symbol=symbol,
            quote_currency="USDT",
            ins_type=ins_type,
            asset_type=AssetType.CRYPTO,
            side=PositionSide.LONG,
            is_open=True,
            is_fake=False,
            order_amount=order_amount,
            order_price=order_price,
            ratio=Decimal("0.1"),
            order_type=OrderType.LimitOrder,
            leverage=Decimal("3"),
            trade_mode=TradeMode.ISOLATED
        )
    
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.place_order')
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.get_instruments')
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.get_orderbook')
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.get_account_config')
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.set_leverage')
    def test_send_order_success(self, mock_set_leverage, mock_get_account_config,
                                 mock_get_orderbook, mock_get_instruments, mock_place_order):
        """测试下单成功"""
        # Mock 账户配置
        mock_get_account_config.return_value = {
            "code": "0",
            "data": [{"posMode": "net_mode"}]
        }
        
        # Mock 交易对信息
        mock_get_instruments.return_value = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "lotSz": "1",
                "ctVal": "0.01",
                "minSz": "1",
                "maxMktSz": "1000",
                "maxLmtSz": "1000"
            }]
        }
        
        # Mock 订单簿
        mock_get_orderbook.return_value = {
            "code": "0",
            "data": [{
                "asks": [["50001", "1", "0", "1"]],
                "bids": [["49999", "1", "0", "1"]]
            }]
        }
        
        # Mock 下单响应
        mock_place_order.return_value = {
            "code": "0",
            "msg": "",
            "data": [{
                "ordId": "exchange_order_123",
                "clOrdId": "test_order_001"
            }]
        }
        
        # Mock 设置杠杆
        mock_set_leverage.return_value = {"code": "0"}
        
        # 创建测试订单
        order = self._create_test_order()
        
        # 启动执行器
        self.executor.on_start()
        
        try:
            # 下单
            self.executor.send_order(order)
            
            # 验证下单被调用
            mock_place_order.assert_called_once()
            
            # 验证订单被添加到待查询列表
            with self.executor._orders_lock:
                self.assertIn(order.order_id, self.executor._pending_orders)
                self.assertEqual(self.executor._pending_orders[order.order_id]["ordId"], "exchange_order_123")
        
        finally:
            self.executor.on_stop()
    
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.cancel_order')
    def test_cancel_order_success(self, mock_cancel_order):
        """测试撤单成功"""
        # Mock 撤单响应
        mock_cancel_order.return_value = {
            "code": "0",
            "msg": "",
            "data": [{"clOrdId": "test_order_001"}]
        }
        
        # 启动执行器
        self.executor.on_start()
        
        try:
            # 添加一个待查询订单
            order = self._create_test_order()
            with self.executor._orders_lock:
                self.executor._pending_orders[order.order_id] = {
                    "instId": "BTC-USDT-SWAP",
                    "ordId": "exchange_order_123",
                    "order": order,
                    "instType": "SWAP"
                }
            
            # 撤单
            self.executor.cancel_order(order.order_id, "BTC-USDT-SWAP")
            
            # 验证撤单被调用
            mock_cancel_order.assert_called_once()
            call_args = mock_cancel_order.call_args
            self.assertEqual(call_args.kwargs["inst_id"], "BTC-USDT-SWAP")
            self.assertEqual(call_args.kwargs["cl_ord_id"], order.order_id)
        
        finally:
            self.executor.on_stop()
    
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.get_orders')
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.get_instruments')
    def test_polling_mechanism(self, mock_get_instruments, mock_get_orders):
        """测试轮询机制"""
        # Mock 交易对信息
        mock_get_instruments.return_value = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "lotSz": "1",
                "ctVal": "0.01",
                "minSz": "1",
                "maxMktSz": "1000",
                "maxLmtSz": "1000"
            }]
        }
        
        # Mock 订单查询响应 - 第一次返回 live 状态，第二次返回 filled 状态
        call_count = [0]
        def mock_get_orders_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一次查询，订单还在 live 状态
                return {
                    "code": "0",
                    "data": [{
                        "instId": "BTC-USDT-SWAP",
                        "instType": "SWAP",
                        "clOrdId": "test_order_001",
                        "ordId": "exchange_order_123",
                        "state": "live",
                        "side": "buy",
                        "posSide": "long"
                    }]
                }
            else:
                # 第二次查询，订单已完成
                return {
                    "code": "0",
                    "data": [{
                        "instId": "BTC-USDT-SWAP",
                        "instType": "SWAP",
                        "clOrdId": "test_order_001",
                        "ordId": "exchange_order_123",
                        "state": "filled",
                        "side": "buy",
                        "posSide": "long",
                        "avgPx": "50000",
                        "accFillSz": "1",
                        "fee": "0.1",
                        "pnl": "10",
                        "lever": "3",
                        "fillTime": "1640995200000",
                        "uTime": "1640995200000"
                    }]
                }
        
        mock_get_orders.side_effect = mock_get_orders_side_effect
        
        # 创建测试订单
        order = self._create_test_order()
        
        # 添加回调函数来捕获订单更新
        callback_calls = []
        def test_callback(order_update):
            callback_calls.append(order_update)
        
        self.executor.callback = test_callback
        
        # 启动执行器
        self.executor.on_start()
        
        try:
            # 添加订单到待查询列表
            with self.executor._orders_lock:
                self.executor._pending_orders[order.order_id] = {
                    "instId": "BTC-USDT-SWAP",
                    "ordId": "exchange_order_123",
                    "order": order,
                    "instType": "SWAP"
                }
            
            # 等待轮询处理（最多等待3秒）
            max_wait = 3
            waited = 0
            while len(callback_calls) == 0 and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            
            # 验证订单更新被触发
            self.assertGreater(len(callback_calls), 0, "订单更新回调应该被触发")
            order_update = callback_calls[0]
            self.assertEqual(order_update.order_id, order.order_id)
            self.assertEqual(order_update.order_status, OrderStatus.FILLED)
            
            # 验证订单已从待查询列表中移除
            with self.executor._orders_lock:
                self.assertNotIn(order.order_id, self.executor._pending_orders)
        
        finally:
            self.executor.on_stop()
    
    @patch('leek_core.adapts.okx_adapter.OkxAdapter.place_order')
    def test_send_order_failure(self, mock_place_order):
        """测试下单失败"""
        # Mock 下单失败响应
        mock_place_order.return_value = {
            "code": "51000",
            "msg": "下单失败：余额不足"
        }
        
        # 创建测试订单
        order = self._create_test_order()
        
        # 启动执行器
        self.executor.on_start()
        
        try:
            # 下单应该抛出异常
            with self.assertRaises(RuntimeError) as context:
                # 需要 mock 其他依赖方法以避免实际调用
                with patch.object(self.executor, '_check_sz', return_value=Decimal("1")):
                    with patch.object(self.executor, '_get_book', return_value={"asks": [["50001", "1"]], "bids": [["49999", "1"]]}):
                        with patch.object(self.executor, '_get_instrument', return_value={"lotSz": "1", "ctVal": "0.01", "minSz": "1", "maxMktSz": "1000", "maxLmtSz": "1000"}):
                            with patch.object(self.executor, 'init_account_mode'):
                                with patch.object(self.executor, 'set_leverage'):
                                    self.executor.send_order(order)
            
            self.assertIn("下单失败", str(context.exception))
        
        finally:
            self.executor.on_stop()
            
    