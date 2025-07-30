#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
适配器模块
提供各种外部服务的桥接类
"""

from .okx_adapter import OkxAdapter

__all__ = [
    'OkxAdapter',
] 