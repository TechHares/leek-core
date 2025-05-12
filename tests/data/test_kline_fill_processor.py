#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试K线数据填充处理器的功能。
"""

import time
import unittest

from info_fabricator.kline_fill import KLineFillProcessor
from models.constants import TimeFrame, TradeInsType
from models.data import KLine


class TestKLineFillProcessor(unittest.TestCase):
    """测试K线数据填充处理器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.processor = KLineFillProcessor()
        
        # 当前时间（毫秒）
        self.current_time_ms = int(time.time() * 1000)
        
        # 基础K线数据，用于创建测试K线
        self.base_kline_data = {
            "symbol": "BTC",
            "market": "test",
            "quote_currency": "USDT",
            "ins_type": TradeInsType.SPOT,
            "data_source_id": "test_source",
            "open": "100.0",
            "close": "101.0",
            "high": "102.0",
            "low": "99.0",
            "volume": "10.0",
            "amount": "1000.0",
            "timeframe": TimeFrame.M1,
            "is_finished": True
        }
    
    def create_test_kline(self, **kwargs):
        """创建测试K线对象"""
        # 合并基础数据和自定义参数
        data = self.base_kline_data.copy()
        data.update(kwargs)
        
        # 如果没有指定时间，设置默认时间
        if 'start_time' not in data:
            data['start_time'] = self.current_time_ms - 60000  # 1分钟前
        if 'end_time' not in data:
            data['end_time'] = self.current_time_ms
        if 'current_time' not in data:
            data['current_time'] = self.current_time_ms
            
        return KLine(**data)
    
    def test_normal_sequence(self):
        """测试正常顺序的K线（无需填充）"""
        # 创建一个连续的K线序列
        kline1 = self.create_test_kline(
            start_time=self.current_time_ms - 120000,  # 2分钟前
            end_time=self.current_time_ms - 60000,     # 1分钟前
            current_time=self.current_time_ms - 60000
        )
        
        kline2 = self.create_test_kline(
            start_time=self.current_time_ms - 60000,   # 1分钟前
            end_time=self.current_time_ms,             # 当前时间
            current_time=self.current_time_ms
        )
        
        # 处理第一个K线
        result1 = self.processor.process(kline1)
        self.assertEqual(len(result1), 1)
        self.assertEqual(result1[0], kline1)
        
        # 处理第二个K线
        result2 = self.processor.process(kline2)
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0], kline2)
    
    def test_missing_one_kline(self):
        """测试缺失一个K线的情况"""
        # 创建两个不连续的K线
        kline1 = self.create_test_kline(
            start_time=self.current_time_ms - 180000,  # 3分钟前
            end_time=self.current_time_ms - 120000,    # 2分钟前
            current_time=self.current_time_ms - 120000
        )
        
        kline2 = self.create_test_kline(
            start_time=self.current_time_ms - 60000,   # 1分钟前
            end_time=self.current_time_ms,             # 当前时间
            current_time=self.current_time_ms
        )
        
        # 处理第一个K线
        result1 = self.processor.process(kline1)
        self.assertEqual(len(result1), 1)
        
        # 处理第二个K线，应该填充中间缺失的K线
        result2 = self.processor.process(kline2)
        self.assertEqual(len(result2), 2)  # 应该有填充K线和原始K线
        print(result1[0].start_time, result1[0].end_time)
        print(result2[0].start_time, result2[0].end_time)
        print(result2[1].start_time, result2[1].end_time)
        self.assertTrue(result2[0].end_time + result1[0].start_time == result1[0].end_time + result2[0].start_time)
        # 验证填充K线的属性
        filled_kline = result2[0]
        self.assertTrue('is_filled' in filled_kline.metadata)
        self.assertTrue(filled_kline.metadata['is_filled'])
        self.assertEqual(filled_kline.start_time, kline1.end_time)
        self.assertEqual(filled_kline.end_time, filled_kline.start_time + 60000)  # 1分钟
        self.assertEqual(filled_kline.open, kline1.close)  # 填充K线的开盘价应该是前一个K线的收盘价
        self.assertTrue(filled_kline.is_finished)  # 填充K线应该标记为已完成
    
    def test_missing_multiple_klines(self):
        """测试缺失多个K线的情况"""
        # 创建相距较远的两个K线（中间缺少2个K线）
        kline1 = self.create_test_kline(
            start_time=self.current_time_ms - 240000,  # 4分钟前
            end_time=self.current_time_ms - 180000,    # 3分钟前
            current_time=self.current_time_ms - 180000
        )
        
        kline2 = self.create_test_kline(
            start_time=self.current_time_ms - 60000,   # 1分钟前
            end_time=self.current_time_ms,             # 当前时间
            current_time=self.current_time_ms
        )
        
        # 处理第一个K线
        result1 = self.processor.process(kline1)
        self.assertEqual(len(result1), 1)
        
        # 处理第二个K线，应该填充中间缺失的K线（应该有2个填充的K线）
        result2 = self.processor.process(kline2)
        self.assertEqual(len(result2), 3)  # 2个填充K线 + 1个原始K线
        
        # 验证第一个填充K线
        filled_kline1 = result2[0]
        self.assertTrue(filled_kline1.metadata['is_filled'])
        self.assertEqual(filled_kline1.start_time, kline1.end_time)
        self.assertEqual(filled_kline1.end_time, filled_kline1.start_time + 60000)
        
        # 验证第二个填充K线
        filled_kline2 = result2[1]
        self.assertTrue(filled_kline2.metadata['is_filled'])
        self.assertEqual(filled_kline2.start_time, filled_kline1.end_time)
        self.assertEqual(filled_kline2.end_time, filled_kline2.start_time + 60000)
    
    def test_unfinished_kline(self):
        """测试未完成K线的处理"""
        # 创建一个已完成的K线和一个未完成的K线
        kline1 = self.create_test_kline(
            start_time=self.current_time_ms - 120000,  # 2分钟前
            end_time=self.current_time_ms - 60000,     # 1分钟前
            current_time=self.current_time_ms - 60000
        )
        
        # 未完成的K线
        kline2 = self.create_test_kline(
            start_time=self.current_time_ms - 60000,   # 1分钟前
            end_time=self.current_time_ms,             # 当前时间
            current_time=self.current_time_ms - 30000,  # 仅过了一半的时间
            is_finished=False  # 标记为未完成
        )
        
        # 新的K线，与未完成K线同一时间段
        kline3 = self.create_test_kline(
            start_time=self.current_time_ms - 60000,   # 1分钟前
            end_time=self.current_time_ms,             # 当前时间
            current_time=self.current_time_ms
        )
        
        # 处理第一个K线
        self.processor.process(kline1)
        
        # 处理未完成的K线
        result2 = self.processor.process(kline2)
        self.assertEqual(len(result2), 1)
        
        # 处理与未完成K线相同时间段的新K线
        result3 = self.processor.process(kline3)
        self.assertEqual(len(result3), 1)
        self.assertEqual(result3[0], kline3)
    
    def test_different_timeframes(self):
        """测试不同时间周期的K线"""
        # 创建两个不同时间周期的K线（应该分别缓存）
        kline1_m1 = self.create_test_kline(
            timeframe=TimeFrame.M1,
            start_time=self.current_time_ms - 60000,
            end_time=self.current_time_ms
        )
        
        kline1_m5 = self.create_test_kline(
            timeframe=TimeFrame.M15,
            start_time=self.current_time_ms - 300000,  # 5分钟前
            end_time=self.current_time_ms
        )
        
        # 处理不同时间周期的K线
        result1 = self.processor.process(kline1_m1)
        result2 = self.processor.process(kline1_m5)
        
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        
        # 创建不连续的K线
        kline2_m1 = self.create_test_kline(
            timeframe=TimeFrame.M1,
            start_time=self.current_time_ms + 60000,    # 未来1分钟
            end_time=self.current_time_ms + 120000      # 未来2分钟
        )
        
        kline2_m5 = self.create_test_kline(
            timeframe=TimeFrame.M15,
            start_time=self.current_time_ms + 300000,   # 未来5分钟
            end_time=self.current_time_ms + 600000      # 未来10分钟
        )
        
        # 处理不连续的K线，分别填充
        result3 = self.processor.process(kline2_m1)
        result4 = self.processor.process(kline2_m5)
        
        # M1应该填充一个K线
        self.assertEqual(len(result3), 2)  # 1个填充 + 1个原始
        
        # M5不应该填充（因为正好是一个周期的间隔）
        self.assertEqual(len(result4), 1)  # 只有原始K线
    
    def test_different_symbols(self):
        """测试不同交易对的K线"""
        # 创建两个不同交易对的K线
        kline1_btc = self.create_test_kline(
            symbol="BTC",
            start_time=self.current_time_ms - 60000,
            end_time=self.current_time_ms
        )
        
        kline1_eth = self.create_test_kline(
            symbol="ETH",
            start_time=self.current_time_ms - 60000,
            end_time=self.current_time_ms
        )
        
        # 处理不同交易对的K线
        self.processor.process(kline1_btc)
        self.processor.process(kline1_eth)
        
        # 创建不连续的K线
        kline2_btc = self.create_test_kline(
            symbol="BTC",
            start_time=self.current_time_ms + 120000,   # 未来2分钟
            end_time=self.current_time_ms + 180000      # 未来3分钟
        )
        
        kline2_eth = self.create_test_kline(
            symbol="ETH",
            start_time=self.current_time_ms + 120000,   # 未来2分钟
            end_time=self.current_time_ms + 180000      # 未来3分钟
        )
        
        # 处理不连续的K线，应该分别为不同交易对填充
        result_btc = self.processor.process(kline2_btc)
        result_eth = self.processor.process(kline2_eth)
        
        # 两个交易对都应该填充一个K线
        self.assertEqual(len(result_btc), 2)  # 1个填充 + 1个原始
        self.assertEqual(len(result_eth), 2)  # 1个填充 + 1个原始
        
        # 验证填充的K线是针对正确的交易对
        self.assertEqual(result_btc[0].symbol, "BTC")
        self.assertEqual(result_eth[0].symbol, "ETH")


if __name__ == "__main__":
    unittest.main()
