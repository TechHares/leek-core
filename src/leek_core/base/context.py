#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List

from leek_core.event import EventBus
from .component import LeekComponent
from .util import create_component


class LeekContext(LeekComponent):
    """
    上下文组件基类，定义组件的基本属性和接口
    """
    init_params: List["Field"] = None # context不需要初始化参数描述

    def __init__(self, event_bus: EventBus, config: "LeekComponentConfig"):
        """
        初始化上下文
        Args:
            event_bus: 事件总线
            config: 组件配置
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
    
    def check_component(self):
        """
        检查组件状态
        """
        ...

