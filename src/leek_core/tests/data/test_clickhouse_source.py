#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ClickHouse数据源测试用例
使用unittest.mock模拟ClickHouse客户端
"""
import os

os.environ['LEEK_LOG_LEVEL']='DEBUG'
import unittest
from datetime import datetime

from leek_core.data import ClickHouseKlineDataSource
from leek_core.models import TimeFrame, TradeInsType
from leek_core.utils import get_logger

logger = get_logger(__name__)


class ClickHouseKlineSourceTest(unittest.TestCase):
    """ClickHouse数据源测试用例"""
    
    def setUp(self):
        """每个测试前准备工作"""
        # 测试数据源配置
        self.host = "localhost"
        self.port = 9000
        self.user = "default"
        self.password = ""
        self.database = "default"
        self.table = "klines"
        self.data_source = ClickHouseKlineDataSource(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            name="clickhouse",
            instance_id="test_instance"
        )
    
    def test_clickhouse_source(self):
        """
        测试ClickHouse数据源的同步方法
        """
        # 设置mock的Client实例

        # 连接数据源
        connected = self.data_source.connect()
        self.assertTrue(connected, "数据源连接失败")
        logger.info(f"数据源连接状态: {'成功' if connected else '失败'}")
        
        # 获取可用的交易对
        symbols = self.data_source.get_supported_parameters()
        self.assertIsNotNone(symbols, "获取交易对失败")
        logger.info(f"可用的交易对: {symbols}")
        

        # 获取K线数据
        # start_time = int((datetime.now() - pd.Timedelta(days=1)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        klines = self.data_source.get_history_data(
            symbol="BTC",
            timeframe=TimeFrame.M1,
            market="okx",
            quote_currency="USDT",
            ins_type=TradeInsType.SWAP,
            start_time=None,
            end_time=end_time,
            limit=10
        )
        
        # 将迭代器转换为列表以进行测试
        kline_list = list(klines)
        for k in kline_list:
            logger.info(f"获取到的K线数据: {k}")
        self.assertIsNotNone(kline_list, "获取K线数据失败")
        self.assertTrue(len(kline_list) > 0, "K线数据为空")
        self.assertLessEqual(len(kline_list), 10, "K线数据条数超出预期")
        
        # 检查第一个K线对象是否有正确的属性
        if kline_list:
            first_kline = kline_list[0]
            logger.info(f"获取到的第一个K线数据: {first_kline}")
            self.assertEqual(first_kline.symbol, "BTC", "交易对不正确")
            self.assertEqual(first_kline.market, "okx", "市场不正确")
            self.assertEqual(first_kline.quote_currency, "USDT", "计价币种不正确")
        
        # 移除了市场状态测试，因为对应方法已删除
        
        # 测试不支持的方法
        try:
            self.data_source.subscribe(
                symbol="BTC",
                timeframe=TimeFrame.M1,
                callback=lambda x: None
            )
            self.fail("ClickHouse应该抛出异常而不支持订阅功能")
        except NotImplementedError:
            pass  # 预期的异常
        
        try:
            self.data_source.unsubscribe(
                symbol="BTC",
                timeframe=TimeFrame.M1
            )
            self.fail("ClickHouse应该抛出异常而不支持取消订阅功能")
        except NotImplementedError:
            pass  # 预期的异常
        
        # 断开连接
        disconnected = self.data_source.disconnect()
        self.assertTrue(disconnected, "数据源断开连接失败")
        logger.info(f"数据源断开状态: {'成功' if disconnected else '失败'}")


if __name__ == "__main__":
    unittest.main()
