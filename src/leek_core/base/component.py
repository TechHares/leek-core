#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Dict, Any, List, Set


if TYPE_CHECKING:
    from leek_core.event import EventType, Event
    from leek_core.models import Field


class LeekComponent:
    display_name: str = None
    init_params: List["Field"] = []

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

    def on_event(self, event: "Event"):
        """
        处理事件，子类可重写。默认不做任何处理。

        参数:
            event: 事件对象
        """
        ...
