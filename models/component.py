#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, ClassVar

from .parameter import Field, FieldType
from utils import EventSource


class Component:
    display_name: ClassVar[str] = None
    init_params: ClassVar[List[Field]] = [
        Field(
            name="instance_id",
            label="组件名称",
            type=FieldType.STR,
            default="",
            description="组件实例ID",
        ),
        Field(
            name="name",
            label="组件名称",
            type=FieldType.STR,
            default="",
            description="组件名称",
        )
    ]
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

    def _event_source(self) -> EventSource:
        """
        获取事件源
        """
        return EventSource(
            instance_id=self.instance_id,
            name=self.name,
            cls=self.__class__.__name__,
        )



def create_instance(cls: type[Component], **kwargs):
    """
    创建实例
    参数:
        cls: 类
        kwargs: 关键字参数
    返回:
        实例
    """
    define_params: List[Field] = cls.init_params
    params = {}
    for field in define_params:
        if field.name in kwargs:
            params[field.name] = field.covert(kwargs[field.name])
        elif field.default is not None:
            params[field.name] = field.covert(field.default)

    instance = cls(**params)
    return instance