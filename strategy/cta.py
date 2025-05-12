#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC
from decimal import Decimal

from models import Data, DataType, KLine, PositionSide
from .base import Strategy, StrategyCommand


class CTAStrategy(Strategy, ABC):
    """
    择时策略基类，定义策略的基本接口
    """
    def __init__(self):
        super().__init__()

    def on_data(self, data: Data = None):
        """
        处理数据，子类可以选择性重写此方法

        参数:
            data: 接收到的数据，可以是任何类型
            data_type: 数据类型，如果为None，则由策略自行判断
        """
        # 如果是K线数据，调用on_kline方法
        if data.data_type == DataType.KLINE and isinstance(data, KLine):
            self.on_kline(data)

    def on_kline(self, kline: KLine):
        """
        处理K线数据，为了向后兼容而保留

        参数:
            kline: K线数据
        """
        ...

    def should_open(self) -> PositionSide | StrategyCommand:
        """
        判断是否应该开仓

        返回:
            是否应该开仓 PositionSide 表示全仓开， StrategyCommand 可以自定义比例
        """
        ...


    def should_close(self, position_side: PositionSide) -> bool | Decimal:

        """
        判断是否应该平仓
        参数:
            position_side: 当前仓位方向
        返回:
            是否应该平仓 True 表示平仓，False 表示不平仓，Decimal 表示平仓比例
        """
        return False
