#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource

class TestIndicatorEngine(unittest.TestCase):
    def test_bias(self):
        view = IndicatorView([MA(14), BiasRatio(14)], symbol="CRV")
        view.start({
            "name": "ma",
            "row": 1
        },{
            "name": "bias",
            "row": 2
        })

    def test_bollinger(self):
        view = IndicatorView([BollBand(20)], [3], symbol="CRV")
        view.start({
            "name": "low",
        }, {
            "name": "mid",
        }, {
            "name": "up",
        })

    def test_cci(self):
        view = IndicatorView([CCI(14)], [1], symbol="CRV")
        view.start({
            "name": "cci",
            "row": 2
        })

    def test_cci2(self):
        view = IndicatorView([CCIV2(14)], [1], symbol="CRV")
        view.start({
            "name": "cci",
            "row": 2
        })

    def test_demark(self):
        view = IndicatorView([DeMarker(14)], [1], symbol="CRV")
        view.start({
            "name": "demark",
            "row": 2
        })

    def test_tdsq(self):
        view = IndicatorView([TDSequence(perfect_countdown=True)], [1], symbol="CRV")
        view.start({
            "name": "demark",
            "row": 2
        })

    def test_tdl(self):
        view = IndicatorView([TDTrendLine()], [2], symbol="CRV")
        view.start({
            "name": "up",
            "row": 1
        },{
            "name": "down",
            "row": 1
        })

    def test_dk(self):
        view = IndicatorView([DK()], [1], symbol="CRV")
        view.start({
            "name": "dk",
            "type": "mark",
            "row": 2
        })

    def test_dm(self):
        view = IndicatorView([DMI()], [1], symbol="CRV")
        view.start({
            "name": "dmi",
            "row": 2
        })

    def test_reflex(self):
        view = IndicatorView([Reflex()], [1], symbol="CRV")
        view.start({
            "name": "Reflex",
            "row": 2
        })
    def test_trendflex(self):
        view = IndicatorView([TrendFlex()], [1], symbol="CRV")
        view.start({
            "name": "TrendFlex",
            "row": 2
        })

    def test_volume_profile(self):
        """
        测试成交量轮廓指标并绘制图表
        """
        view = IndicatorView([VolumeProfile(lookback=90, bins=20)], [3], symbol="BTC",
        data_source=ClickHouseKlineDataSource(password="default"))
        view.start({
            "name": "poc_distance_pct",
            "row": 3
        }, {
            "name": "is_in_hvn",
            "row": 4,
            "type": "bar"
        }, {
            "name": "is_in_lvn",
            "row": 4,
            "type": "bar"
        }, show_volume=True, height=900)

    def test_spike_detector(self):
        """
        测试价格尖峰检测指标并绘制图表
        
        基于SNN论文的尖峰检测方法：
        - spike_strength: 尖峰强度（窗口内平均绝对收益率）
        - spike_threshold: 尖峰阈值（滚动中位数）
        - is_spike: 是否为尖峰事件
        """
        view = IndicatorView(
            [SpikeDetector(threshold_window=60, strength_window=3)], 
            [3], 
            symbol="BTC",
            data_source=ClickHouseKlineDataSource(password="default")
        )
        view.start({
            "name": "spike_strength",
            "row": 3
        }, {
            "name": "spike_threshold",
            "row": 3
        }, {
            "name": "is_spike",
            "row": 4,
            "type": "bar"
        }, show_volume=True, height=900)


if __name__ == '__main__':
    unittest.main()