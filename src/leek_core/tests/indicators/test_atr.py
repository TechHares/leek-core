#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource

class TestIndicatorEngine(unittest.TestCase):
    def test_TR(self):
        view = IndicatorView([TR(), ATR(14)], symbol="CRV")
        view.start({
            "name": "tr",
            "row": 2
        },{
            "name": "atr",
            "row": 2
        })



if __name__ == '__main__':
    unittest.main()