#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import StrategyDebugView
from leek_core.data import DataSource, ClickHouseKlineDataSource
from leek_core.strategy.strategy_debug import DebugStrategy

class TestIndicatorEngine(unittest.TestCase):
    def test_engine(self):
        view = StrategyDebugView(strategy=DebugStrategy(), symbol="CRV")
        view.start()


if __name__ == '__main__':
    unittest.main()