#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource

class TestIndicatorEngine(unittest.TestCase):
    def test_engine(self):
        view = IndicatorView([MA(5), MA(20)], symbol="CRV")
        view.start({
            "name": "ma5"
        },{
            "name": "ma20"
        })


if __name__ == '__main__':
    unittest.main()