#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from decimal import Decimal
from os import times
from typing import Dict, Any

from leek_core.indicators import DMI
from leek_core.strategy import CTAStrategy, StrategyCommand
from leek_core.utils import get_logger
from leek_core.models import Field, FieldType, KLine, PositionSide
logger = get_logger(__name__)


class DebugStrategy(CTAStrategy):
    
    """
    Debug 用
    """
    display_name: str = "debug专用"
    init_params = []

    def __init__(self):
        self.open = False
        self.sell = False

    def on_kline(self, kline: KLine):
        print("on_kline", kline.start_time, kline.is_finished)

    def should_open(self) -> PositionSide | StrategyCommand:
        print("should_open", self.open)
        if self.open:
            self.open = False
            return PositionSide.LONG
        return None

    def should_close(self, position_side: PositionSide) -> bool | Decimal:
        # time.sleep(10000)
        print("should_close", self.sell)
        return self.sell

    def get_state(self) -> Dict[str, Any]:
        return {
            "open": self.open,
            "sell": self.sell,
            "row": "rx = sin(x) * 45"
        }

    def load_state(self, state: Dict[str, Any]):
        print("数据加载", state)
        self.open = state.get("open", False)
        self.sell = state.get("sell", False)
