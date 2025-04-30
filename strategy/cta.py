#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC
from decimal import Decimal

from models import Data, DataType, KLine, PositionSide
from strategy import Strategy


class STAStrategy(Strategy, ABC):
    """
    择时策略基类，定义策略的基本接口
    """
    def __init__(self):
        super().__init__()
