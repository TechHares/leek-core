#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List

from base import LeekComponent
from models import Field


def create_component(cls: type[LeekComponent], **kwargs):
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