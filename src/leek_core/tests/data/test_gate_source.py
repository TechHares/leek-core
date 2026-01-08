#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gate.io WebSocket 数据源测试用例
测试订阅K线数据和获取历史K线数据
"""

import time
import unittest

from leek_core.data import GateDataSource
from leek_core.models import TimeFrame, TradeInsType, KLine


class TestGateDataSource(unittest.TestCase):
    """Gate.io 数据源测试"""

    def test_subscribe_eth_kline_30s(self):
        """
        用例1: 订阅ETH合约K线数据30秒
        """
        source = GateDataSource(settle="usdt")
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
            print("正在连接 Gate.io WebSocket...")
            source.on_start()
            print("连接成功!")

            # 构建订阅的 row_key
            row_key = KLine.pack_row_key(
                symbol="CRV",
                quote_currency="USDT",
                ins_type=TradeInsType.SWAP,
                timeframe=TimeFrame.M1  # 1分钟K线
            )

            # 订阅 ETH/USDT 永续合约 1分钟K线
            print(f"订阅 ETH/USDT 1分钟K线...")
            result = source.subscribe(row_key)
            print(f"订阅结果: {result}")

            # 等待30秒接收数据
            print("等待30秒接收K线数据...")
            time.sleep(120)

            # 取消订阅
            print("取消订阅...")
            source.unsubscribe(row_key)

            print(f"\n共收到 {len(received_data)} 条K线数据")

        finally:
            # 断开连接
            print("断开连接...")
            source.on_stop()
            print("测试完成!")

    def test_get_eth_history_klines(self):
        """
        用例2: 获取ETH合约最近100根K线
        """
        source = GateDataSource(settle="usdt")

        # 构建查询的 row_key
        row_key = KLine.pack_row_key(
            symbol="ETH",
            quote_currency="USDT",
            ins_type=TradeInsType.SWAP,
            timeframe=TimeFrame.M1  # 1分钟K线
        )

        print(f"获取 ETH/USDT 永续合约最近100根1分钟K线...")

        # 获取历史数据
        klines = list(source.get_history_data(
            row_key=row_key,
            limit=100
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

        # 断言至少获取到数据
        self.assertGreater(len(klines), 0, "应该获取到K线数据")
        self.assertLessEqual(len(klines), 100, "获取的K线数量不应超过100")

        # 验证K线数据结构
        for kline in klines:
            self.assertEqual(kline.symbol, "ETH")
            self.assertEqual(kline.quote_currency, "USDT")
            self.assertEqual(kline.market, "gate")
            self.assertIsNotNone(kline.open)
            self.assertIsNotNone(kline.close)
            self.assertIsNotNone(kline.high)
            self.assertIsNotNone(kline.low)

        print("\n测试完成!")


if __name__ == "__main__":
    # 可以单独运行某个测试
    import sys

    if len(sys.argv) > 1:
        # 运行指定的测试
        if sys.argv[1] == "subscribe":
            test = TestGateDataSource()
            test.test_subscribe_eth_kline_30s()
        elif sys.argv[1] == "history":
            test = TestGateDataSource()
            test.test_get_eth_history_klines()
    else:
        # 运行所有测试
        unittest.main()
