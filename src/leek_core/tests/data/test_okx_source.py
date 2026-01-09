#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX WebSocket 数据源测试用例
测试获取历史K线数据的顺序
"""

import time
import unittest

from leek_core.data import OkxDataSource
from leek_core.models import TimeFrame, TradeInsType, KLine


class TestOkxDataSource(unittest.TestCase):
    """OKX 数据源测试"""

    def test_get_btc_history_klines(self):
        """
        用例: 获取BTC永续合约最近100根1分钟K线，验证返回顺序
        """
        source = OkxDataSource()

        # 构建查询的 row_key
        row_key = KLine.pack_row_key(
            symbol="BTC",
            quote_currency="USDT",
            ins_type=TradeInsType.SWAP,
            timeframe=TimeFrame.M1  # 1分钟K线
        )

        print(f"\n当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"获取 BTC/USDT 永续合约最近100根1分钟K线...")

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

        # 断言
        self.assertGreater(len(klines), 0, "应该获取到K线数据")

        print("\n测试完成!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        test = TestOkxDataSource()
        test.test_get_btc_history_klines()
    else:
        unittest.main()
