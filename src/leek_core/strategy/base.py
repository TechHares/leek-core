#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略基础模块，提供策略的抽象基类和通用功能。
"""

from abc import ABC
from dataclasses import dataclass
from decimal import Decimal
from logging import Logger
from typing import Any, Set, List

from leek_core.base import LeekComponent
from leek_core.models import PositionSide, Field, Data, DataType
from leek_core.utils import get_logger
from .strategy_mode import StrategyMode, KlineSimple

logger = get_logger(__name__)

@dataclass
class StrategyCommand:
    """
    策略指令。

    属性:
        side: 仓位方向
        ratio: 仓位比例
    """
    side: PositionSide
    ratio: Decimal

class Strategy(LeekComponent, ABC):
    """
    择时策略抽象基类，定义策略的基本接口。
    
    类属性:
        display_name: 策略展示名称
        open_just_no_pos: 是否只在没有仓位时开仓，默认为True
        accepted_data_types: 策略接受的数据类型列表，默认接受K线数据
        strategy_mode: 策略运行模式，默认为单标的单时间周期模式
    """
    
    # 策略展示名称
    display_name: str = "未命名策略"
    
    # 是否只在没有仓位时开仓
    open_just_no_pos: bool = True
    
    # 策略接受的数据类型
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    
    # 策略运行模式
    strategy_mode: StrategyMode = KlineSimple()
    # 参数
    init_params: List[Field] = []

    def __init__(self):
        """
        初始化策略
        """
        # 初始化日志器
        self.logger: Logger = None
    
    def on_data(self, data: Data = None):
        """
        处理数据，子类可以选择性重写此方法
        
        参数:
            data: 接收到的数据，可以是任何类型
            data_type: 数据类型，如果为None，则由策略自行判断
        """
        ...
    
    def should_open(self) -> Any:
        """
        判断是否应该开仓
        
        返回:
            是否应该开仓 PositionSide 表示全仓开， StrategyCommand 可以自定义比例
        """
        ...
    
    def should_close(self, position_side:PositionSide) -> Any:
        """
        判断是否应该平仓
        参数:
            position_side: 当前仓位方向
        返回:
            是否应该平仓 True 表示平仓，False 表示不平仓，Decimal 表示平仓比例
        """
        return False
