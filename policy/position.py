#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控基础模块，提供风控的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from typing import Set

from base import LeekComponent
from models import Data, Position, DataType
from utils import get_logger

logger = get_logger(__name__)


class PositionPolicy(LeekComponent, ABC):
    # 策略接受的数据类型
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    """
    仓位风控策略基类。

    本抽象基类定义了所有仓位风控策略的统一接口和通用行为，便于扩展和集成多种风控规则。
    
    主要职责：
    1. 统一风控规则的接口，子类需实现 evaluate 方法，对仓位进行风控检查。
    2. 支持灵活扩展多种风控逻辑，如最大持仓、止损、止盈、持仓周期等。
    3. 可与仓位管理、策略等模块解耦集成，实现灵活的风控组合。
    
    使用说明：
    - 所有自定义仓位风控策略需继承本类并实现 evaluate 方法。
    - evaluate 方法根据传入的行情数据和仓位需不需，判断是否满足风控要求。
    """
    @abstractmethod
    def evaluate(self, data: Data, position: Position) -> bool:
        """
        检查信号是否符合风控规则

        参数：
            signal: 信号数据

        返回：
            是否通过检查
        """
        ...
