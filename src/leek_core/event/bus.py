#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Set, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import threading
from queue import Queue, Empty
import time

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
        # 为每个event_type和callback组合维护单独的队列
        self._callback_queues: Dict[str, Queue] = {}
        self._queue_lock = threading.Lock()
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

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
            # 检查是否已经订阅过
            if callback in self._all_event_subscribers:
                logger.debug(f"回调函数已经订阅了所有事件: {id(callback)}")
                return True
                
            self._all_event_subscribers.add(callback)
            # 为全局订阅者创建队列
            self._create_callback_queue("__all__", callback)
            return True
            
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        
        # 检查是否已经订阅过
        if callback in self._subscribers[event_type]:
            logger.debug(f"回调函数已经订阅了事件类型 {event_type}: {id(callback)}")
            return True
            
        self._subscribers[event_type].add(callback)
        
        # 为每个event_type和callback组合创建队列
        self._create_callback_queue(event_type, callback)
        return True

    def _create_callback_queue(self, event_type: EventType, callback: Callable):
        """为指定的event_type和callback创建队列和工作者线程"""
        queue_key = f"{event_type}_{id(callback)}"
        
        with self._queue_lock:
            if queue_key not in self._callback_queues:
                # 创建队列
                queue = Queue()
                self._callback_queues[queue_key] = queue
                
                # 创建工作者线程
                worker_thread = threading.Thread(
                    target=self._queue_worker,
                    args=(queue_key, queue, callback),
                    daemon=True
                )
                self._worker_threads[queue_key] = worker_thread
                worker_thread.start()
                
                logger.debug(f"为 {event_type} 创建队列和工作者线程: {queue_key}")
            else:
                logger.debug(f"队列已存在，跳过创建: {queue_key}")

    def _queue_worker(self, queue_key: str, queue: Queue, callback: Callable):
        """队列工作者线程，按顺序处理事件"""
        logger.debug(f"队列工作者线程启动: {queue_key}")
        
        while not self._stop_event.is_set():
            try:
                # 从队列中获取事件，超时1秒
                event = queue.get(timeout=1)
                if event is None:  # 停止信号
                    break
                    
                # 执行回调
                self._run_callback(callback, event)
                queue.task_done()
                
            except Empty:
                # 队列超时是正常情况，继续循环
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"队列工作者线程异常: {queue_key}, {e}", exc_info=True)
                break
                
        logger.debug(f"队列工作者线程退出: {queue_key}")

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
            # 取消全局订阅
            if callback in self._all_event_subscribers:
                self._all_event_subscribers.discard(callback)
                queue_key = f"__all___{id(callback)}"
                
                with self._queue_lock:
                    # 停止工作者线程
                    if queue_key in self._worker_threads:
                        queue = self._callback_queues.get(queue_key)
                        if queue:
                            queue.put(None)  # 发送停止信号
                        self._worker_threads[queue_key].join(timeout=5)
                        del self._worker_threads[queue_key]
                        del self._callback_queues[queue_key]
                        logger.debug(f"已取消全局订阅并清理队列: {queue_key}")
                return True
            return False

        # 取消特定事件类型订阅
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            queue_key = f"{event_type}_{id(callback)}"
            
            with self._queue_lock:
                # 停止工作者线程
                if queue_key in self._worker_threads:
                    queue = self._callback_queues.get(queue_key)
                    if queue:
                        queue.put(None)  # 发送停止信号
                    self._worker_threads[queue_key].join(timeout=5)
                    del self._worker_threads[queue_key]
                    del self._callback_queues[queue_key]
                    logger.debug(f"已取消订阅并清理队列: {queue_key}")
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
        使用队列保证同一event_type的同一callback按顺序执行
        """
        callbacks = []
        
        # 收集具体类型订阅者
        if event.event_type in self._subscribers:
            callbacks.extend(list(self._subscribers[event.event_type]))
        
        # 收集全局订阅者
        callbacks.extend(list(self._all_event_subscribers))
        
        if not callbacks:
            return

        # 将事件放入每个回调的队列中，保证顺序执行
        with self._queue_lock:
            for cb in callbacks:
                # 确定队列键
                if cb in self._all_event_subscribers:
                    queue_key = f"__all___{id(cb)}"
                else:
                    queue_key = f"{event.event_type}_{id(cb)}"
                
                # 将事件放入队列
                if queue_key in self._callback_queues:
                    self._callback_queues[queue_key].put(event)
                else:
                    logger.warning(f"队列不存在: {queue_key}，直接执行回调")
                    self._executor.submit(self._run_callback, cb, event)

    def shutdown(self):
        """关闭事件总线，清理所有资源"""
        logger.info("正在关闭事件总线...")
        self._stop_event.set()
        
        with self._queue_lock:
            # 停止所有工作者线程
            for queue_key, queue in self._callback_queues.items():
                queue.put(None)  # 发送停止信号
            
            # 等待所有线程结束
            for queue_key, thread in self._worker_threads.items():
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"工作者线程未能在5秒内结束: {queue_key}")
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        logger.info("事件总线已关闭")

    def get_subscription_info(self) -> dict:
        """获取当前订阅信息，用于调试"""
        with self._queue_lock:
            return {
                "subscribers": {str(k): len(v) for k, v in self._subscribers.items()},
                "all_event_subscribers_count": len(self._all_event_subscribers),
                "callback_queues_count": len(self._callback_queues),
                "worker_threads_count": len(self._worker_threads),
                "callback_queues": list(self._callback_queues.keys()),
                "worker_threads": list(self._worker_threads.keys())
            }
