#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource

class TestIndicatorEngine(unittest.TestCase):
    def test_ma(self):
        view = IndicatorView([MA(5), MA(20)], symbol="CRV")
        view.start({
            "name": "ma5"
        },{
            "name": "ma20"
        })

    def test_ema(self):
        view = IndicatorView([EMA(5), EMA(20)], symbol="CRV")
        view.start({
            "name": "ma5"
        },{
            "name": "ma20"
        })

    def test_wma(self):
        view = IndicatorView([WMA(5), WMA(20)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "ma5"
        },{
            "name": "ma20"
        })

    def test_hma(self):
        view = IndicatorView([HMA(5), HMA(20)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "ma5"
        },{
            "name": "ma20"
        })

    def test_mul_ma(self):
        view = IndicatorView([MA(5), EMA(20), WMA(60), HMA(120)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "ma5"
        },{
            "name": "ema20"
        },{
            "name": "wma60"
        },{
            "name": "hma120"
        })

    def test_mul_ma1(self):
        view = IndicatorView([KAMA(5), FRAMA(20), LLT(60), SuperSmoother(120), UltimateOscillator(250)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "kama5"
        },{
            "name": "frama20"
        },{
            "name": "llt60"
        },{
            "name": "ss120"
        },{
            "name": "uo250"
        })

    def test_mul_ma2(self):
        view = IndicatorView([UltimateOscillator(120), UltimateOscillator(250)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "uo120"
        },{
            "name": "uo250"
        })

    def test_mul_ma3(self):
        view = IndicatorView([SuperSmoother(60), SuperSmoother(120), SuperSmoother(250)], symbol="CRV", timeframe=TimeFrame.M15)
        view.start({
            "name": "ss60"
        },{
            "name": "ss120"
        },{
            "name": "ss250"
        })


if __name__ == '__main__':
    unittest.main()