#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource

class TestIndicator2(unittest.TestCase):
    def test_ic(self):
        view = IndicatorView([IchimokuCloud()], symbol="CRV")
        view.start({
            "name": "tenkan-sen",
            "row": 1
        },{
            "name": "kijun-sen",
            "row": 1
        },{
            "name": "span A",
            "row": 1
        },{
            "name": "span B",
            "row": 1
        })

    def test_kdj(self):
        view = IndicatorView([KDJ()], symbol="CRV")
        view.start({
            "name": "kdj",
            "row": 2
        })

    def test_macd(self):
        view = IndicatorView([MACD()], symbol="CRV")
        view.start({
            "name": "dif",
            "row": 2
        },{
            "name": "dea",
            "row": 2
        })

    def test_rsi(self):
        view = IndicatorView([RSI()], symbol="CRV")
        view.start({
            "name": "rsi",
            "row": 2
        })

    def test_rsi2(self):
        view = IndicatorView([StochRSI()], symbol="CRV")
        view.start({
            "name": "rsi",
            "row": 2
        },{
            "name": "dea",
            "row": 2
        })

    def test_rsrs(self):
        view = IndicatorView([RSRS()], symbol="CRV")
        view.start({
            "name": "rsrs",
            "row": 2
        })


    def test_sar(self):
        view = IndicatorView([SAR()], symbol="CRV")
        view.start({
            "name": "sar",
            "type": "mark",
            "row": 1
        },{
            "name": "islong",
            "type": "mark",
            "row": 2
        }
        )

    def test_wr(self):
        view = IndicatorView([WR()], symbol="CRV")
        view.start({
            "name": "wr",
            "row": 2
        }
        )



if __name__ == '__main__':
    unittest.main()