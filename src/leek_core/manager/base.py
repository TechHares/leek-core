#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import TypeVar, Generic, List, Dict

from leek_core.event import EventBus
from leek_core.base import LeekComponent, LeekContext
from leek_core.models import LeekComponentConfig
from leek_core.utils import run_func_timeout, get_logger
logger = get_logger(__name__)

CTX = TypeVar('CTX', bound=LeekContext)
T = TypeVar('T', bound=LeekComponent)
CFG = TypeVar('CFG')


class ComponentManager(LeekContext, Generic[CTX, T, CFG]):
    """
    通用简化组件管理器（ComponentManager）。

    该类用于统一管理和调度多种数据源或组件，适用于插件式、批量化场景。
    支持数据源的批量添加、移除、启动和销毁。

    参数泛型：
        CTX: 继承自 LeekContext 的上下文类型
        T:   继承自 LeekComponent 的组件类型
        CFG: 组件配置类型

    属性：
        components: Dict[str, LeekContext]
            管理所有已注册的实例，key 为实例 ID。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[CTX, None]):
        """
        初始化 SimpleEngine。

        参数：
            event_bus: 事件总线实例
            config:   顶层配置，包含多个数据源的配置信息
        """
        super().__init__(event_bus, config)
        self.components: Dict[str, LeekContext] = {}

    def add(self, config: LeekComponentConfig[T, CFG]):
        """
        添加实例并启动。

        参数：
            data_source_config: 需要添加的配置
        """
        if config.instance_id in self.components:
            return
        self.components[config.instance_id] = self.config.cls(self.event_bus, config)
        is_finish = run_func_timeout(self.components[config.instance_id].on_start, [], {}, 20)
        if not is_finish:
            self.components.pop(config.instance_id)
            logger.error(f"组件{config.name}启动超时")
            return
        logger.info(f"组件{config.name}启动完成")

    def get(self, instance_id: str) -> CTX:
        """
        获取指定实例。
        参数：
            instance_id: 要获取实例ID
        """
        return self.components.get(instance_id)
    
    def update(self, config: LeekComponentConfig[T, CFG]):
        """
        更新指定实例。
        """
        if config.instance_id not in self.components:
            self.add(config)
            return
        self.components[config.instance_id].update(config)

    def remove(self, instance_id: str):
        """
        移除并停止指定实例。

        参数：
            instance_id: 要移除实例ID
        """
        if instance_id not in self.components:
            return
        source = self.components[instance_id]
        run_func_timeout(source.on_stop, [], {})
        self.components.pop(instance_id)

    def on_stop(self):
        """
        停止所有实例并清空管理器。
        """
        for source in self.components.values():
            run_func_timeout(source.on_stop, [], {})
        self.components.clear()

    def __len__(self):
        return len(self.components)
