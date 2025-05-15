#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC

from leek_core.base import LeekComponent


class Plugin(LeekComponent, ABC):
    """
    插件基类，定义插件的基本接口
    """
    priority = 10  # 优先级，数字越小优先级越高
    display_name = "插件"  # 插件的显示名称
