#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from decimal import Decimal
from os import times

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
        self.p = True

    def on_kline(self, kline: KLine):
        ...

    def should_open(self) -> PositionSide | StrategyCommand:
        if self.p:
            self.p = False
            return PositionSide.LONG
        return None

    def should_close(self, position_side: PositionSide) -> bool | Decimal:
        # time.sleep(10000)
        return True

