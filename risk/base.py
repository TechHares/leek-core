#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控基础模块，提供风控的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Set

from models import Component, Data, Position, DataType
from utils import get_logger

logger = get_logger(__name__)


class RiskPlugin(Component, ABC):
    accepted_data_types: ClassVar[Set[DataType]] = {DataType.KLINE}
    """
    风控策略抽象基类，定义风控策略的基本接口。
    
    风控策略用于检查订单是否符合风控规则，如果不符合则拒绝交易。
    """

    def __init__(self, instance_id: str=None, name: str=None):
        """
        初始化风控策略

        参数:
            instance_id: 策略ID，如果不提供则自动生成
            name: 风控组件名称
        """
        super().__init__(instance_id=instance_id, name=name)

    @abstractmethod
    def trigger(self, position: Position, data: Data) -> bool:
        """
        检查订单是否符合风控规则

        参数:
            data: 数据

        返回:
            是否通过检查
        """
        pass
