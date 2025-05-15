#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal
from typing import List, Set

from leek_core.models import PositionSide, Position, Data, DataType, Field, FieldType
from .position import PositionPolicy


class PositionTakeProfit(PositionPolicy):
    """
    仓位止盈风控策略。

    用于判断当前持仓是否达到预设的止盈条件，若达到则触发平仓信号。
    
    主要逻辑：
    1. 多头仓位：当最新收盘价高于开仓均价的一定比例（profit_ratio）时，触发止盈。
    2. 空头仓位：当最新收盘价低于开仓均价的一定比例（profit_ratio）时，触发止盈。
    
    参数说明：
    - profit_ratio: 止盈比例，默认为5%。
    
    使用说明：
    - 仅对K线数据（DataType.KLINE）生效。
    - 需配合仓位管理模块使用。
    """
    display_name: str = "止盈"
    init_params: List[Field] = [
        Field(
            name="profit_ratio",
            label="止盈比例",
            type=FieldType.FLOAT,
            default=0.05,
            description="止盈比例，默认为5%"
        )
    ]
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    def __init__(self, profit_ratio: Decimal = Decimal('0.05')):
        """
        初始化仓位止盈策略。
        :param profit_ratio: 止盈比例，默认为5%
        """
        self.profit_ratio = profit_ratio

    def evaluate(self, data: Data, position: Position) -> bool:
        """
        检查仓位是否满足止盈条件。
        :param data: 行情数据
        :param position: 仓位信息
        :return: 是否满足止盈条件（True=未达到止盈，False=应平仓）
        """
        assert data.data_type == DataType.KLINE, "数据类型错误"
        if position.side == PositionSide.LONG:
            return data.close < position.cost_price * (1 + self.profit_ratio)
        elif position.side == PositionSide.SHORT:
            return data.close > position.cost_price * (1 - self.profit_ratio)