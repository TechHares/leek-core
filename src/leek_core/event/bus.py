#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Set, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

from leek_core.utils import get_logger
from .types import EventType, Event

logger = get_logger(__name__)
"""
事件定义和事件总线扩展
"""


class EventBus:
    """简化版事件总线，负责事件分发和处理"""

    def __init__(self):
        """初始化事件总线"""
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._all_event_subscribers: Set[Callable] = set()
        self._executor = ThreadPoolExecutor(max_workers=10)

    def subscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        订阅事件。如果 event_type 为空，则订阅所有事件。

        参数:
            event_type: 事件类型（可为 None 或空字符串，表示订阅全部事件）
            callback: 回调函数

        返回:
            是否成功订阅
        """
        if not event_type:
            self._all_event_subscribers.add(callback)
            return True
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        return True

    def unsubscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        取消订阅事件

        参数:
            event_type: 事件类型
            callback: 回调函数

        返回:
            是否成功取消订阅
        """
        if event_type is None:
            self._all_event_subscribers.discard(callback)

        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            return True
        return False

    def _run_callback(self, cb, event):
        """在线程池中执行回调函数"""
        try:
            return cb(event)
        except Exception as e:
            logger.error(f"事件处理异常: {e}", exc_info=True)

    def publish_event(self, event: Event):
        """
        发布事件，支持分发给所有订阅者（包括订阅所有事件的回调）
        使用线程池非阻塞执行所有回调
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
            self._executor.submit(self._run_callback, cb, event)
