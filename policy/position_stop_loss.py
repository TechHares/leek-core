#!/usr/bin/env python
# -*- coding: utf-8 -*-

from decimal import Decimal
from typing import List, Set

from models import PositionSide, Position, Data, DataType, Field, FieldType
from .position import PositionPolicy


class PositionStopLoss(PositionPolicy):
    """
    仓位止损风控策略。

    用于判断当前持仓是否达到预设的止损条件，若达到则触发平仓信号。
    
    主要逻辑：
    1. 多头仓位：当最新收盘价低于开仓均价的一定比例（stop_loss_ratio）时，触发止损。
    2. 空头仓位：当最新收盘价高于开仓均价的一定比例（stop_loss_ratio）时，触发止损。
    
    参数说明：
    - stop_loss_ratio: 止损比例，默认为5%。
    
    使用说明：
    - 仅对K线数据（DataType.KLINE）生效。
    - 需配合仓位管理模块使用。
    """
    display_name: str = "止损"
    init_params: List[Field] = [
        Field(
            name="stop_loss_ratio",
            label="止损比例",
            type=FieldType.FLOAT,
            default=0.05,
            description="止损比例，默认为5%"
        )
    ]
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    def __init__(self, stop_loss_ratio: Decimal = Decimal('0.05')):
        """
        初始化仓位止损策略。
        :param stop_loss_ratio: 止损比例，默认为5%
        """
        self.stop_loss_ratio = stop_loss_ratio

    def evaluate(self, data: Data, position: Position) -> bool:
        """
        检查仓位是否满足止损条件。
        :param data: 行情数据
        :param position: 仓位信息
        :return: 是否满足止损条件（True=未达到止损，False=应平仓）
        """
        assert data.data_type == DataType.KLINE, "数据类型错误"
        if position.side == PositionSide.LONG:
            return data.close > position.cost_price * (1 - self.stop_loss_ratio)
        elif position.side == PositionSide.SHORT:
            return data.close < position.cost_price * (1 + self.stop_loss_ratio)