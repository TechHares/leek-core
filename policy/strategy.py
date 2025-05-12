#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from base import LeekComponent
from models import Signal, PositionInfo
from utils import get_logger

logger = get_logger(__name__)


class StrategyPolicy(LeekComponent, ABC):
    """
    策略风控基类。

    本模块定义了策略级风控的抽象基类和通用接口，用于对策略信号进行统一的风控检查和管理。
    
    主要职责：
    1. 统一策略风控规则接口，便于扩展多种风控逻辑（如信号过滤、风控组合等）。
    2. 支持与策略模块解耦，提升系统灵活性和可维护性。
    3. 可扩展实现如信号有效性校验、极端行情过滤、风控日志记录等功能。
    
    使用说明：
    - 所有自定义策略风控需继承本类并实现相关方法。
    - 通过实现 evaluate 等方法，对策略信号进行风控校验。
    """

    @abstractmethod
    def evaluate(self, signal: Signal, context: PositionInfo) -> bool:
        """
        检查信号是否符合风控规则

        参数:
            signal: 信号数据

        返回:
            是否通过检查
        """
        ...
