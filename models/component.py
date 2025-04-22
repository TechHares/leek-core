#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any


class Component:
    """
    组件基类，定义组件的基本接口
    """

    def __init__(self, instance_id: str=None, name: str=None):
        """
        初始化组件
        Args:
            instance_id: 组件名称
            name: 组件名称
        """
        self.instance_id = instance_id if instance_id else "%s" % id(self)
        self.name = name if name else self.__class__.__name__

    def start(self):
        """
        启动组件
        """
        ...

    def stop(self):
        """
        停止组件
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        序列化组件状态
        """
        return {}

    def load_state(self, state: Dict[str, Any]):
        """
        加载组件状态
        """
        ...