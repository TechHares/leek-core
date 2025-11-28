#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
import importlib

from leek_core.base import LeekComponent


def create_component(cls: type[LeekComponent], **kwargs):
    """
    创建实例
    参数:
        cls: 类
        kwargs: 关键字参数
    返回:
        实例
    """
    from leek_core.models import Field
    define_params: List[Field] = cls.init_params
    params = {}
    for field in define_params:
        if field.name in kwargs:
            params[field.name] = field.covert(kwargs[field.name])
        elif field.default is not None:
            params[field.name] = field.covert(field.default)

    instance = cls(**params)
    return instance


def load_class_from_str(class_path: str):
    """
    通过 'module_path|ClassName' 字符串动态加载类对象
    """
    if "|" not in class_path:
        raise ValueError("class_path must be in 'module|ClassName' format")
    module_path, class_name = class_path.split("|", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls