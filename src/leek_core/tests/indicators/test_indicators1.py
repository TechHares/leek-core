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



if __name__ == '__main__':
    unittest.main()