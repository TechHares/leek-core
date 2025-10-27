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
            for subscriber in self._subscribers[event.event_type]:
                subscriber(event)
        
        # 全局订阅者
        for subscriber in self._all_event_subscribers:
            subscriber(event)
