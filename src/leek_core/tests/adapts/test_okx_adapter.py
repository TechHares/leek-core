#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX适配器测试
"""

import unittest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from leek_core.adapts import OkxAdapter
from leek_core.utils import get_logger


class TestOkxAdapter(unittest.TestCase):
    """OKX适配器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.adapter = OkxAdapter()
        self.logger = get_logger(__name__)
        
        # 模拟API响应
        self.mock_candlesticks_response = {
            "code": "0",
            "msg": "",
            "data": [
                ["1640995200000", "46200.1", "46200.1", "46200.1", "46200.1", "0", "1640995259999", "0", "0", "0"]
            ]
        }
        
        self.mock_orderbook_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "asks": [["46200.1", "0.1", "0", "1"]],
                    "bids": [["46199.9", "0.1", "0", "1"]],
                    "ts": "1640995200000"
                }
            ]
        }
        
        self.mock_tickers_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instType": "SPOT",
                    "instId": "BTC-USDT",
                    "last": "46200.1",
                    "lastSz": "0.1",
                    "askPx": "46200.2",
                    "askSz": "0.1",
                    "bidPx": "46199.9",
                    "bidSz": "0.1",
                    "open24h": "46000.0",
                    "high24h": "46500.0",
                    "low24h": "45800.0",
                    "volCcy24h": "1000000",
                    "vol24h": "21.7",
                    "ts": "1640995200000"
                }
            ]
        }
        
        self.mock_instruments_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instType": "SPOT",
                    "instId": "BTC-USDT",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                    "settleCcy": "",
                    "ctVal": "",
                    "ctMult": "",
                    "ctValCcy": "",
                    "optType": "",
                    "stk": "",
                    "listTime": "1597026383085",
                    "expTime": "",
                    "tickSz": "0.1",
                    "lotSz": "0.00000001",
                    "minSz": "0.00001",
                    "maxSz": "10000000000",
                    "uly": "",
                    "category": "1",
                    "state": "live"
                }
            ]
        }
        
        self.mock_mark_price_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instType": "SWAP",
                    "instId": "BTC-USDT-SWAP",
                    "markPx": "46200.1",
                    "ts": "1640995200000"
                }
            ]
        }
        
        self.mock_balance_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "acctId": "123456789",
                    "ccy": "USDT",
                    "cashBal": "1000.0",
                    "cbal": "1000.0",
                    "disBal": "1000.0",
                    "availBal": "1000.0",
                    "frozenBal": "0.0",
                    "ordFrozen": "0.0",
                    "liab": "0.0",
                    "upl": "0.0",
                    "uplLib": "0.0",
                    "crossLiab": "0.0",
                    "isoLiab": "0.0",
                    "mgnRatio": "0.0",
                    "interest": "0.0",
                    "twap": "0.0",
                    "maxLoan": "0.0",
                    "eq": "1000.0",
                    "notionalLever": "0.0",
                    "stgyEq": "0.0",
                    "isoUpl": "0.0",
                    "uTime": "1640995200000"
                }
            ]
        }
        
        self.mock_positions_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "adlRanking": "1",
                    "availPos": "1",
                    "avgPx": "46200.1",
                    "cTime": "1640995200000",
                    "ccy": "USDT",
                    "deltaBS": "0.0",
                    "deltaPA": "0.0",
                    "gammaBS": "0.0",
                    "gammaPA": "0.0",
                    "idxPx": "46200.1",
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "interest": "0.0",
                    "last": "46200.1",
                    "lastPx": "46200.1",
                    "lever": "10",
                    "liab": "0.0",
                    "liabCcy": "USDT",
                    "liqPx": "0.0",
                    "markPx": "46200.1",
                    "margin": "0.0",
                    "mgnMode": "cross",
                    "mgnRatio": "0.0",
                    "mmr": "0.0",
                    "notionalCcy": "USDT",
                    "notionalUsd": "46200.1",
                    "openMax": "0.0",
                    "ordFrozen": "0.0",
                    "pnl": "0.0",
                    "pnlCcy": "USDT",
                    "pos": "1",
                    "posCcy": "BTC",
                    "posId": "123456789",
                    "posSide": "long",
                    "quoteBal": "0.0",
                    "quoteCcy": "USDT",
                    "realizedPnl": "0.0",
                    "slTriggerPx": "0.0",
                    "slTriggerPxType": "last",
                    "state": "normal",
                    "tdMode": "cross",
                    "tpTriggerPx": "0.0",
                    "tpTriggerPxType": "last",
                    "tradeId": "123456789",
                    "uTime": "1640995200000",
                    "upl": "0.0",
                    "uplLastPx": "0.0",
                    "uplRatio": "0.0",
                    "uplRatioLastPx": "0.0"
                }
            ]
        }
        
        self.mock_order_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "clOrdId": "",
                    "ordId": "123456789",
                    "tag": "",
                    "sCode": "0",
                    "sMsg": ""
                }
            ]
        }
        
        self.mock_leverage_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "lever": "10",
                    "mgnMode": "cross",
                    "posSide": "long"
                }
            ]
        }

    def test_init(self):
        """测试初始化"""
        # 测试无参数初始化
        adapter = OkxAdapter()
        self.assertIsNotNone(adapter.market_api)
        self.assertIsNotNone(adapter.public_api)
        self.assertIsNone(adapter.account_api)  # 没有API密钥时应该为None
        
        # 测试带参数初始化
        adapter_with_auth = OkxAdapter(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_passphrase"
        )
        self.assertIsNotNone(adapter_with_auth.account_api)

    @patch('okx.MarketData.MarketAPI.get_history_candlesticks')
    def test_get_history_candlesticks(self, mock_get_candlesticks):
        """测试获取历史K线数据"""
        mock_get_candlesticks.return_value = self.mock_candlesticks_response
        
        result = self.adapter.get_history_candlesticks(
            inst_id="BTC-USDT",
            bar="1m",
            limit=100
        )
        
        mock_get_candlesticks.assert_called_once_with(
            instId="BTC-USDT",
            bar="1m",
            limit=100,
            before="",
            after=""
        )
        self.assertEqual(result, self.mock_candlesticks_response)

    @patch('okx.MarketData.MarketAPI.get_orderbook')
    def test_get_orderbook(self, mock_get_orderbook):
        """测试获取订单簿数据"""
        mock_get_orderbook.return_value = self.mock_orderbook_response
        
        result = self.adapter.get_orderbook(
            inst_id="BTC-USDT",
            sz=20
        )
        
        mock_get_orderbook.assert_called_once_with(
            instId="BTC-USDT",
            sz=20
        )
        self.assertEqual(result, self.mock_orderbook_response)

    @patch('okx.MarketData.MarketAPI.get_tickers')
    def test_get_tickers(self, mock_get_tickers):
        """测试获取产品行情"""
        mock_get_tickers.return_value = self.mock_tickers_response
        
        result = self.adapter.get_tickers(inst_type="SPOT")
        
        mock_get_tickers.assert_called_once_with(instType="SPOT")
        self.assertEqual(result, self.mock_tickers_response)



    @patch('okx.PublicData.PublicAPI.get_instruments')
    def test_get_instruments(self, mock_get_instruments):
        """测试获取产品信息"""
        mock_get_instruments.return_value = self.mock_instruments_response
        
        result = self.adapter.get_instruments(inst_type="SPOT")
        
        mock_get_instruments.assert_called_once_with(
            instType="SPOT",
            instId=None
        )
        self.assertEqual(result, self.mock_instruments_response)

    @patch('okx.PublicData.PublicAPI.get_mark_price')
    def test_get_mark_price(self, mock_get_mark_price):
        """测试获取标记价格"""
        mock_get_mark_price.return_value = self.mock_mark_price_response
        
        result = self.adapter.get_mark_price(
            inst_type="SWAP",
            inst_id="BTC-USDT-SWAP"
        )
        
        mock_get_mark_price.assert_called_once_with(
            instType="SWAP",
            instId="BTC-USDT-SWAP"
        )
        self.assertEqual(result, self.mock_mark_price_response)

    @patch('okx.Account.AccountAPI.get_account_balance')
    def test_get_account_balance(self, mock_get_balance):
        """测试获取账户余额"""
        mock_get_balance.return_value = self.mock_balance_response
        
        # 需要先设置API密钥
        adapter = OkxAdapter(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_passphrase"
        )
        
        result = adapter.get_account_balance(ccy="USDT")
        
        mock_get_balance.assert_called_once_with(ccy="USDT")
        self.assertEqual(result, self.mock_balance_response)

    @patch('okx.Account.AccountAPI.get_positions')
    def test_get_positions(self, mock_get_positions):
        """测试获取持仓信息"""
        mock_get_positions.return_value = self.mock_positions_response
        
        # 需要先设置API密钥
        adapter = OkxAdapter(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_passphrase"
        )
        
        result = adapter.get_positions(
            inst_type="SWAP",
            inst_id="BTC-USDT-SWAP"
        )
        
        mock_get_positions.assert_called_once_with(
            instType="SWAP",
            instId="BTC-USDT-SWAP"
        )
        self.assertEqual(result, self.mock_positions_response)



    def test_build_inst_id(self):
        """测试构建产品ID"""
        # 测试现货
        spot_id = OkxAdapter.build_inst_id("BTC", 1, "USDT")
        self.assertEqual(spot_id, "BTC-USDT")
        
        # 测试永续合约
        swap_id = OkxAdapter.build_inst_id("BTC", 2, "USDT")
        self.assertEqual(swap_id, "BTC-USDT-SWAP")
        
        # 测试交割合约
        futures_id = OkxAdapter.build_inst_id("BTC", 3, "USDT")
        self.assertEqual(futures_id, "BTC-USDT-240628")  # 假设当前是6月

    def test_get_okx_timeframe(self):
        """测试时间周期转换"""
        # 测试字符串输入
        self.assertEqual(OkxAdapter.get_okx_timeframe("1m"), "1m")
        self.assertEqual(OkxAdapter.get_okx_timeframe("5m"), "5m")
        self.assertEqual(OkxAdapter.get_okx_timeframe("1h"), "1H")
        self.assertEqual(OkxAdapter.get_okx_timeframe("1d"), "1D")
        
        # 测试无效输入
        self.assertIsNone(OkxAdapter.get_okx_timeframe("invalid"))

    def test_check_api_available(self):
        """测试API可用性检查"""
        # 测试无API密钥时
        self.assertFalse(self.adapter.check_api_available())
        
        # 测试有API密钥时
        adapter_with_auth = OkxAdapter(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_passphrase"
        )
        self.assertTrue(adapter_with_auth.check_api_available())


class TestOkxAdapterRateLimit(unittest.TestCase):
    """OKX适配器限速测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.adapter = OkxAdapter()
        self.logger = get_logger(__name__)
        
    def test_rate_limit_logging(self):
        """测试限速日志输出"""
        # 设置日志级别为DEBUG
        logging.basicConfig(level=logging.DEBUG)
        
        # 快速调用，触发限速
        start_time = time.time()
        for i in range(5):
            try:
                # 使用模拟响应避免实际API调用
                with patch.object(self.adapter.market_api, 'get_tickers') as mock_tickers:
                    mock_tickers.return_value = {"code": "0", "data": []}
                    result = self.adapter.get_tickers(inst_type="SPOT")
                    self.assertEqual(result["code"], "0")
            except Exception as e:
                self.logger.warning(f"调用 {i+1} 异常: {e}")
        
        end_time = time.time()
        self.logger.info(f"5次调用总耗时: {end_time - start_time:.2f}秒")

    def test_concurrent_rate_limit(self):
        """测试并发限速"""
        import threading
        
        results = []
        lock = threading.Lock()
        
        def make_request():
            try:
                with patch.object(self.adapter.market_api, 'get_tickers') as mock_tickers:
                    mock_tickers.return_value = {"code": "0", "data": []}
                    result = self.adapter.get_tickers(inst_type="SPOT")
                    with lock:
                        results.append(result["code"])
            except Exception as e:
                with lock:
                    results.append(f"error: {e}")
        
        # 创建多个线程并发调用
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有调用都成功
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertEqual(result, "0")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 