#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance WebSocket 数据源测试用例
测试订阅K线数据和获取历史K线数据
"""

import time
import unittest

from leek_core.data import BinanceDataSource
from leek_core.models import TimeFrame, TradeInsType, KLine


class TestBinanceDataSource(unittest.TestCase):
    """Binance 数据源测试"""

    def test_subscribe_btc_kline_30s(self):
        """
        用例1: 订阅BTC现货K线数据30秒
        """
        source = BinanceDataSource()
        received_data = []

        def callback(data):
            """回调函数，收集接收到的K线数据"""
            if isinstance(data, KLine):
                status = "✓完成" if data.is_finished else "◎进行中"
                print(f"[{status}] {data.symbol}/{data.quote_currency} | {data.timeframe.value} | "
                      f"开:{data.open} 收:{data.close} 高:{data.high} 低:{data.low} | "
                      f"时间:{data.start_datetime}")
                received_data.append(data)
            elif data == "reconnect":
                print("WebSocket 连接断开，需要重连")

        # 设置回调
        source.callback = callback

        try:
            # 启动数据源
            print("正在连接 Binance WebSocket...")
            source.on_start()
            print("连接成功!")

            # 构建订阅的 row_key
            row_key = KLine.pack_row_key(
                symbol="BTC",
                quote_currency="USDT",
                ins_type=TradeInsType.SWAP,
                timeframe=TimeFrame.M1  # 1分钟K线
            )

            # 订阅 BTC/USDT 现货 1分钟K线
            print(f"订阅 BTC/USDT 1分钟K线...")
            result = source.subscribe(row_key)
            print(f"订阅结果: {result}")

            # 等待30秒接收数据
            print("等待30秒接收K线数据...")
            time.sleep(30)

            # 取消订阅
            print("取消订阅...")
            source.unsubscribe(row_key)

            print(f"\n共收到 {len(received_data)} 条K线数据")

        finally:
            # 断开连接
            print("断开连接...")
            source.on_stop()
            print("测试完成!")

    def test_subscribe_multiple_symbols(self):
        """
        用例2: 同时订阅多个交易对K线数据
        """
        source = BinanceDataSource()
        received_data = {}

        def callback(data):
            """回调函数，收集接收到的K线数据"""
            if isinstance(data, KLine):
                key = f"{data.symbol}/{data.quote_currency}"
                if key not in received_data:
                    received_data[key] = []
                received_data[key].append(data)
                
                status = "✓完成" if data.is_finished else "◎进行中"
                print(f"[{status}] {key} | {data.timeframe.value} | "
                      f"收:{data.close} | 时间:{data.start_datetime}")

        # 设置回调
        source.callback = callback

        try:
            # 启动数据源
            print("正在连接 Binance WebSocket...")
            source.on_start()
            print("连接成功!")

            symbols = ["BTC", "ETH", "SOL"]
            row_keys = []

            for symbol in symbols:
                row_key = KLine.pack_row_key(
                    symbol=symbol,
                    quote_currency="USDT",
                    ins_type=TradeInsType.SPOT,
                    timeframe=TimeFrame.M1
                )
                row_keys.append(row_key)
                print(f"订阅 {symbol}/USDT 1分钟K线...")
                result = source.subscribe(row_key)
                print(f"订阅结果: {result}")

            # 等待30秒接收数据
            print("等待30秒接收K线数据...")
            time.sleep(30)

            # 取消所有订阅
            print("取消所有订阅...")
            for row_key in row_keys:
                source.unsubscribe(row_key)

            print(f"\n数据统计:")
            for key, data_list in received_data.items():
                print(f"  {key}: 收到 {len(data_list)} 条K线")

        finally:
            # 断开连接
            print("断开连接...")
            source.on_stop()
            print("测试完成!")

    def test_get_btc_history_klines(self):
        """
        用例3: 获取BTC现货最近100根K线，验证返回顺序
        """
        source = BinanceDataSource()

        # 构建查询的 row_key
        row_key = KLine.pack_row_key(
            symbol="BTC",
            quote_currency="USDT",
            ins_type=TradeInsType.SPOT,
            timeframe=TimeFrame.M1  # 1分钟K线
        )

        print(f"\n当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"获取 BTC/USDT 现货最近100根1分钟K线...")

        # 获取历史数据
        klines = list(source.get_history_data(
            row_key=row_key,
            limit=100
        ))

        print(f"共获取到 {len(klines)} 根K线")
        
        # 打印第一条和最后一条，验证顺序
        if klines:
            print(f"\n第1条 (klines[0]): {klines[0].start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"最后1条 (klines[-1]): {klines[-1].start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 判断顺序
            if klines[0].start_time < klines[-1].start_time:
                print("\n✅ 返回顺序: 升序（从旧到新）")
            else:
                print("\n⚠️ 返回顺序: 降序（从新到旧）")
        
        print("\n最近10根K线 (klines[-10:]):")
        print("-" * 100)
        print(f"{'时间':<20} {'开盘':<12} {'最高':<12} {'最低':<12} {'收盘':<12} {'成交量':<15}")
        print("-" * 100)

        for kline in klines[-10:]:
            print(f"{kline.start_datetime.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                  f"{str(kline.open):<12} {str(kline.high):<12} "
                  f"{str(kline.low):<12} {str(kline.close):<12} "
                  f"{str(kline.volume):<15}")

        # 断言至少获取到数据
        self.assertGreater(len(klines), 0, "应该获取到K线数据")
        self.assertLessEqual(len(klines), 100, "获取的K线数量不应超过100")

        # 验证K线数据结构
        for kline in klines:
            self.assertEqual(kline.symbol, "BTC")
            self.assertEqual(kline.quote_currency, "USDT")
            self.assertEqual(kline.market, "binance")
            self.assertIsNotNone(kline.open)
            self.assertIsNotNone(kline.close)
            self.assertIsNotNone(kline.high)
            self.assertIsNotNone(kline.low)

        print("\n测试完成!")

    def test_get_eth_history_klines_5m(self):
        """
        用例4: 获取ETH现货5分钟K线
        """
        source = BinanceDataSource()

        # 构建查询的 row_key
        row_key = KLine.pack_row_key(
            symbol="ETH",
            quote_currency="USDT",
            ins_type=TradeInsType.SPOT,
            timeframe=TimeFrame.M5  # 5分钟K线
        )

        print(f"获取 ETH/USDT 现货最近50根5分钟K线...")

        # 获取历史数据
        klines = list(source.get_history_data(
            row_key=row_key,
            limit=50
        ))

        print(f"共获取到 {len(klines)} 根K线")
        print("\n最近10根K线:")
        print("-" * 100)
        print(f"{'时间':<20} {'开盘':<12} {'最高':<12} {'最低':<12} {'收盘':<12} {'成交量':<15}")
        print("-" * 100)

        for kline in klines[-10:]:
            print(f"{kline.start_datetime.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                  f"{str(kline.open):<12} {str(kline.high):<12} "
                  f"{str(kline.low):<12} {str(kline.close):<12} "
                  f"{str(kline.volume):<15}")

        # 断言
        self.assertGreater(len(klines), 0, "应该获取到K线数据")
        self.assertEqual(klines[0].timeframe, TimeFrame.M5)

        print("\n测试完成!")


if __name__ == "__main__":
    # 可以单独运行某个测试
    import sys

    if len(sys.argv) > 1:
        # 运行指定的测试
        if sys.argv[1] == "subscribe":
            test = TestBinanceDataSource()
            test.test_subscribe_btc_kline_30s()
        elif sys.argv[1] == "multi":
            test = TestBinanceDataSource()
            test.test_subscribe_multiple_symbols()
        elif sys.argv[1] == "history":
            test = TestBinanceDataSource()
            test.test_get_btc_history_klines()
        elif sys.argv[1] == "history5m":
            test = TestBinanceDataSource()
            test.test_get_eth_history_klines_5m()
    else:
        # 运行所有测试
        unittest.main()
