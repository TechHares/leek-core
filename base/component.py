#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any, List

from models.parameter import Field


class LeekComponent:
    display_name: str = None
    init_params: List[Field] = []

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
