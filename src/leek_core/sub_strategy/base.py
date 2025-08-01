#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
附属策略基类模块，定义了进出场策略的基本接口。
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import List, Set

from leek_core.base import LeekComponent
from leek_core.models import Field, DataType, PositionSide, Data


class SubStrategy(LeekComponent, ABC):
    # 声明可接受的数据类型 只有数据类型和主策略有重合， 主策略才可以绑定该附属策略
    accepted_data_types: Set[DataType] = set(DataType)
    # 声明附属策略初始化参数
    init_params: List[Field] = []
    """
    附属策略抽象基类，定义了进出场策略的基本接口。
    
    附属策略负责处理特定的策略逻辑，如入场、出场等，
    可以被主策略组合使用，实现策略的模块化和复用。
    """
    
    def __init__(self):
        """
        初始化附属策略
        """
        super().__init__()
        self.progress = Decimal('0')  # 进度，0-1之间的小数
    
    def on_data(self, data: Data):
        ...

    @abstractmethod
    def ratio(self, position_side: PositionSide):
        """
        此次建议的仓位比例
        
        返回:
            仓位比例，0-1之间的小数，0表示空仓，1表示满仓
        """
        pass
    
    def reset(self):
        """
        重置策略状态
        """
        self.progress = Decimal('0')

    @property
    def is_finished(self) -> bool:
        """
        判断策略是否完成
        """
        return self.progress >= Decimal('1')


class EnterStrategy(SubStrategy):
    """
    入场策略抽象基类，用于定义入场策略的基本接口。
    
    入场策略负责决定何时入场以及入场的仓位比例， 默认直接进场。
    """
    display_name = "直接入场"

    def __init__(self):
        """
        初始化入场策略
        
        参数:
            name: 策略名称，如果为None则使用类名
        """
        super().__init__()

    def ratio(self, position_side: PositionSide):
        self.progress = Decimal('1')
        return self.progress


class ExitStrategy(SubStrategy):
    """
    出场策略抽象基类，用于定义出场策略的基本接口。
    
    出场策略负责决定何时出场以及出场的仓位比例， 默认直接出场。
    """
    display_name = "直接出场"
    
    def __init__(self):
        """
        初始化出场策略
        """
        super().__init__()

    def ratio(self, position_side: PositionSide):
        self.progress = Decimal('1')
        return self.progress