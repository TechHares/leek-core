#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控基础模块，提供风控的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from typing import List

from leek_core.models import Position, PositionInfo
from leek_core.utils import get_logger
from leek_core.base import LeekComponent

logger = get_logger(__name__)


class RiskPlugin(LeekComponent, ABC):
    """
    风控插件基类。

    本抽象基类定义了所有风控插件的统一接口和通用行为，便于扩展和集成多种风控规则。
    
    主要职责：
    1. 统一风控插件的接口，子类需实现 trigger 方法，对仓位集合进行风控检查。
    2. 支持灵活扩展多种风控逻辑，如批量止损、止盈、强平、风控组合等。
    3. 可与仓位管理、策略等模块解耦集成，实现灵活的风控组合与批量处理。
    
    使用说明：
    - 所有自定义风控插件需继承本类并实现 trigger 方法。
    - trigger 方法根据传入的仓位信息，返回需要被平掉的仓位列表。
    """

    @abstractmethod
    def trigger(self, info: PositionInfo) -> List[Position]:
        """
        检查仓位是否符合风控规则

        参数:
            info: 仓位信息

        返回:
            需要平掉的仓位
        """
        pass
