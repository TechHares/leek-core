#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from leek_core.base import LeekComponent
from leek_core.models import ExecutionContext, PositionInfo
from leek_core.utils import get_logger

if TYPE_CHECKING:
    from leek_core.event import EventBus

logger = get_logger(__name__)


class StrategyPolicy(LeekComponent, ABC):
    """
    信号级风控基类（pre-trade）。

    主要职责：
    1. 统一信号风控规则接口，便于扩展多种风控逻辑（如信号过滤、风控组合等）。
    2. 与策略模块解耦，对策略产出的 ExecutionContext 进行准入检查。
    3. 通过 evaluate 决定是否放行开仓信号。

    使用说明：
    - 所有自定义信号风控需继承本类并实现 evaluate 方法。
    """

    # 事件总线引用，由 Context 注入
    event_bus: "EventBus" = None
    # 策略实例ID，由 Context 注入
    policy_instance_id: str = None

    @abstractmethod
    def evaluate(self, signal: ExecutionContext, context: PositionInfo) -> bool:
        """
        检查信号是否符合风控规则

        参数:
            signal: 信号数据

        返回:
            是否通过检查
        """
        ...
