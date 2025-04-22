#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
from typing import Dict, Set, Callable, List

from models import EventType, Event
from .logging import get_logger

logger = get_logger(__name__)
class EventBus:
    """简化版事件总线，负责事件分发和处理"""

    def __init__(self):
        """初始化事件总线"""
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._interceptors: Dict[EventType, List[Callable]] = {}

    def subscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        订阅事件

        参数:
            event_type: 事件类型
            callback: 回调函数

        返回:
            是否成功订阅
        """
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
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            return True
        return False

    def add_interceptor(self, event_type: EventType, interceptor: Callable) -> bool:
        """
        添加事件拦截器

        参数:
            event_type: 事件类型
            interceptor: 拦截器函数

        返回:
            是否成功添加拦截器
        """
        if event_type not in self._interceptors:
            self._interceptors[event_type] = []
        self._interceptors[event_type].append(interceptor)
        return True

    def remove_interceptor(self, event_type: EventType, interceptor: Callable) -> bool:
        """
        移除事件拦截器

        参数:
            event_type: 事件类型
            interceptor: 拦截器函数

        返回:
            是否成功移除拦截器
        """
        if event_type in self._interceptors and interceptor in self._interceptors[event_type]:
            self._interceptors[event_type].remove(interceptor)
            return True
        return False

    async def publish_event(self, event: Event) -> bool:
        """
        发布事件

        参数:
            event: 事件对象

        返回:
            是否成功发布
        """
        # 应用拦截器
        should_continue = await self._apply_interceptors(event)
        if not should_continue:
            logger.debug(f"事件被拦截: {event.event_type}, 来源: {event.source}")
            return False

        # 分发事件给订阅者
        if event.event_type in self._subscribers:
            tasks = []
            for subscriber in self._subscribers[event.event_type]:
                if asyncio.iscoroutinefunction(subscriber):
                    tasks.append(asyncio.create_task(subscriber(event)))
                else:
                    try:
                        subscriber(event)
                    except Exception as e:
                        logger.error(f"事件处理错误: {e}", exc_info=True)

            # 等待所有异步任务完成
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return True

    async def _apply_interceptors(self, event: Event) -> bool:
        """
        应用事件拦截器

        参数:
            event: 事件对象

        返回:
            是否继续处理事件
        """
        if event.event_type not in self._interceptors:
            return True

        for interceptor in self._interceptors[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(event)
                else:
                    result = interceptor(event)

                if result is False:  # 明确返回False才阻止事件传播
                    return False
            except Exception as e:
                logger.error(f"拦截器错误: {e}", exc_info=True)

        return True
