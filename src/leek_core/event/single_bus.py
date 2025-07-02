#!/usr/bin/env python
# -*- coding: utf-8 -*-
from queue import Queue
from typing import Dict, Set, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import threading
import time

from leek_core.utils import get_logger
from .types import EventType, Event
from .bus import EventBus

logger = get_logger(__name__)
"""
事件定义和事件总线扩展
"""


class SerializableEventBus(EventBus):
    """串行化任务事件总线，负责事件分发和处理"""

    def publish_event(self, event: Event):
        """
        发布事件，支持分发给所有订阅者（包括订阅所有事件的回调）
        串行化
        """
        callbacks = []
        
        # 收集具体类型订阅者
        if event.event_type in self._subscribers:
            callbacks.extend(list(self._subscribers[event.event_type]))
        
        # 收集全局订阅者
        callbacks.extend(list(self._all_event_subscribers))
        
        if not callbacks:
            return

        # 在线程池中执行所有回调
        for cb in callbacks:
            cb(event)
