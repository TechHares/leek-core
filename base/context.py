#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Dict, Any

from .component import LeekComponent
from .util import create_component
from .plugin import Plugin

from event import EventBus
from models import Field, FieldType, LeekComponentConfig


class LeekContext(LeekComponent):
    """
    上下文组件基类，定义组件的基本属性和接口
    """
    init_params: List[Field] = None # context不需要初始化参数描述

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig):
        """
        初始化上下文
        Args:
            event_bus: 事件总线
            instance_id: 组件ID
            name: 组件名称
        """
        self.instance_id = config.instance_id if config.instance_id else "%s" % id(self)
        self.name = config.name if config.name else self.__class__.__name__

        self.event_bus = event_bus
        self.config = config


    def create_component(self):
        """
        新建一个组件实例
        Returns:
            组件实例
        """
        return create_component(self.config.cls, **self.config.config)

    def on_start(self):
        """
        启动组件
        """
        ...

    def on_stop(self):
        """
        停止组件
        """
        ...


class PluginContext(LeekComponent):
    """
    上下文组件基类，定义组件的基本属性和接口
    """
    init_params: List[Field] = None # context不需要初始化参数描述， 由固定参数构成

    def __init__(self, event_bus: EventBus, instance_id: str, name: str, plugins: Dict[type[Plugin], Dict[str, Any]]):
        """
        初始化上下文
        Args:
            event_bus: 事件总线
            instance_id: 组件ID
            name: 组件名称
        """
        self.instance_id = instance_id if instance_id else "%s" % id(self)
        self.name = name if name else self.__class__.__name__

        self.plugins = plugins
        self.event_bus = event_bus

        self._plugin_instances: List[Plugin] = []
        for plugin_cls, plugin_params in self.plugins.items():
            plugin = create_component(plugin_cls, **plugin_params)
            self._plugin_instances.append(plugin)

        self._plugin_instances.sort(key=lambda p: p.priority)

    def on_start(self):
        """
        启动组件
        """
        for plugin in self._plugin_instances:
            plugin.on_start()

    def on_stop(self):
        """
        停止组件
        """
        for plugin in self._plugin_instances:
            plugin.on_stop()

