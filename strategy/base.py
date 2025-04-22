#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略基础模块，提供策略的抽象基类和通用功能。
"""

from abc import ABC
from decimal import Decimal
from typing import Dict, Any, ClassVar, Set, List

from models import PositionSide, Field, Component
from models.constants import DataType
from models.data import KLine, Data
from utils import get_logger
from .strategy_mode import StrategyMode, KlineSimple

logger = get_logger(__name__)

class Strategy(Component, ABC):
    """
    择时策略抽象基类，定义策略的基本接口。
    
    类属性:
        display_name: 策略展示名称
        open_just_no_pos: 是否只在没有仓位时开仓，默认为True
        accepted_data_types: 策略接受的数据类型列表，默认接受K线数据
        strategy_mode: 策略运行模式，默认为单标的单时间周期模式
        
    实例属性:
        instance_id: 策略实例ID，用于跟踪具体实例产生的数据和策略表现
        event_bus: 事件总线，用于发布事件
    """
    
    # 策略展示名称
    display_name: ClassVar[str] = "未命名策略"
    
    # 是否只在没有仓位时开仓
    open_just_no_pos: ClassVar[bool] = True
    
    # 策略接受的数据类型
    accepted_data_types: ClassVar[Set[DataType]] = {DataType.KLINE}
    
    # 策略运行模式
    strategy_mode: ClassVar[StrategyMode] = KlineSimple()
    # 参数
    params: ClassVar[List[Field]] = []

    def __init__(self, instance_id: str=None, name: str=None):
        """
        初始化策略
        """
        # 初始化日志器
        super().__init__(instance_id, name)
        self.logger = get_logger(f"strategy.{self.__class__.display_name}.{name}.{instance_id}")
    
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
    
    def should_open(self) -> PositionSide:
        """
        判断是否应该开仓
        
        返回:
            是否应该开仓
        """
        ...
    
    def should_close(self, position_side:PositionSide) -> bool:
        """
        判断是否应该平仓
        参数:
            position_side: 当前仓位方向
        返回:
            是否应该平仓
        """
        return False

    def get_position_change_rate(self) -> Decimal:
        """
        获取当前仓位变化的比例
        在返回开仓或者平仓之后会被调用， 如策略不自定义仓位管理， 无需覆写

        返回:
            仓位比例，0-1之间的小数， 如0.5，标识此次会开/平仓50%(相较于全部投入)的仓位
        """
        return Decimal('1')
