#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from datetime import datetime, timedelta

from leek_core.indicators import GannHiLo, MA, EMA, SuperTrend
from leek_core.models import TimeFrame
from leek_core.engine import IndicatorView
from leek_core.data import ClickHouseKlineDataSource


class TestGannHiLo(unittest.TestCase):
    def test_gann_hilo_plot(self):
        """
        Gann Hi-Lo 指标可视化测试
        """
        view = IndicatorView(
            [GannHiLo(1)],
            symbol="BTC",
            timeframe=TimeFrame.H1,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            data_source=ClickHouseKlineDataSource(password="default")
        )
        view.start(
            {"name": "GannHiLo(1)", "color": "#e377c2", "width": 2},
        )

    def test_gann_hilo_multi_period(self):
        """
        不同周期 Gann Hi-Lo 对比可视化
        """
        view = IndicatorView(
            [GannHiLo(10), GannHiLo(13), GannHiLo(20)],
            symbol="ETH",
            timeframe=TimeFrame.H4,
            start_time=datetime.now() - timedelta(days=60),
            end_time=datetime.now(),
            data_source=ClickHouseKlineDataSource(password="default")
        )
        view.start(
            {"name": "GannHiLo(10)", "color": "#1f77b4", "width": 2},
            {"name": "GannHiLo(13)", "color": "#ff7f0e", "width": 2},
            {"name": "GannHiLo(20)", "color": "#2ca02c", "width": 2},
        )

    def test_gann_hilo_vs_supertrend(self):
        """
        Gann Hi-Lo 与 SuperTrend 对比
        """
        view = IndicatorView(
            [GannHiLo(13), SuperTrend(14, 3.0)],
            symbol="BTC",
            timeframe=TimeFrame.H1,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            data_source=ClickHouseKlineDataSource(password="default")
        )
        view.start(
            {"name": "GannHiLo(13)", "color": "#e377c2", "width": 2},
            {"name": "SuperTrend(14,3)", "color": "#17becf", "width": 2},
        )

    def test_gann_hilo_vs_ma(self):
        """
        Gann Hi-Lo 与 MA 对比
        """
        view = IndicatorView(
            [GannHiLo(13), MA(13), EMA(13)],
            symbol="BTC",
            timeframe=TimeFrame.M15,
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            data_source=ClickHouseKlineDataSource(password="default")
        )
        view.start(
            {"name": "GannHiLo(13)", "color": "#e377c2", "width": 2},
            {"name": "MA(13)", "color": "#ff7f0e", "width": 1},
            {"name": "EMA(13)", "color": "#2ca02c", "width": 1},
        )


if __name__ == '__main__':
    unittest.main()
