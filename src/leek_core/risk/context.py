#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any

from leek_core.base import LeekContext
from leek_core.event import EventBus
from leek_core.models import LeekComponentConfig, PositionInfo, Position
from .base import RiskPlugin


class RiskContextContext(LeekContext):
    """
    风控上下文容器。

    用于管理风控插件的生命周期与统一调用入口，封装风控插件的实例化和触发逻辑。
    
    主要职责：
    1. 负责风控插件的创建、依赖注入和生命周期管理。
    2. 对外提供统一的 trigger 方法，批量检测仓位是否触发风控。
    3. 便于在引擎或上层模块中灵活组合和复用多种风控插件。
    
    使用说明：
    - 构造时传入 event_bus 和风控插件配置。
    - 通过 trigger 方法传入 PositionInfo，返回需要平掉的仓位集合。
    """
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[RiskPlugin, Dict[str, Any]]):
        super().__init__(event_bus, config)

        self._risk_plugin = self.create_component()

    def trigger(self, info: PositionInfo) -> set[Position]:
        ps = self._risk_plugin.trigger(info)
        return set(ps)
