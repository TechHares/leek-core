#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set, Callable, Tuple
import threading
from queue import Queue, Empty

from leek_core.utils import get_logger
from .types import EventType, Event

logger = get_logger(__name__)
"""
事件定义和事件总线扩展
"""


class EventBus:
    """优化版事件总线，使用主队列分发机制，保证顺序性的同时控制线程数量"""

    def __init__(self, max_workers: int = 10):
        """
        初始化事件总线
        
        参数:
            max_workers: 线程池最大工作线程数
        """
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="EventBus")
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._all_event_subscribers: Set[Callable] = set()

        # 主事件队列，所有事件都先放入这里
        self._main_event_queue: Queue = Queue()
        
        # 每个订阅者的事件队列，保证顺序处理
        self._subscriber_queues: Dict[Tuple[Callable, EventType], Queue] = {}
        
        # 记录哪些订阅者当前空闲，可以处理事件
        self._ready_subscribers: Set[Tuple[Callable, EventType]] = set()
        
        # 线程同步锁
        self._lock = threading.Lock()
        
        # 事件分发线程
        self._dispatcher_thread = None
        
        # 停止标志
        self._stop_event = threading.Event()
        
        # 是否已启动
        self._started = False

    def _ensure_started(self):
        """确保事件总线已启动"""
        if not self._started:
            with self._lock:
                if not self._started:  # 双重检查锁定
                    self._start_dispatcher()
                    self._started = True

    def _start_dispatcher(self):
        """启动事件分发线程"""
        self._dispatcher_thread = threading.Thread(
            target=self._event_dispatcher, 
            daemon=True, 
            name="EventBus-Dispatcher"
        )
        self._dispatcher_thread.start()
        logger.debug("事件分发线程已启动")

    def _get_subscriber_key(self, callback: Callable, event_type: EventType) -> Tuple[Callable, EventType]:
        """生成订阅者的唯一键"""
        return (callback, event_type)

    def subscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        订阅事件。如果 event_type 为空，则订阅所有事件。

        参数:
            event_type: 事件类型（可为 None 或空字符串，表示订阅全部事件）
            callback: 回调函数

        返回:
            是否成功订阅
        """
        with self._lock:
            if not event_type:
                # 全局事件订阅
                if callback in self._all_event_subscribers:
                    logger.debug(f"回调函数已经订阅了所有事件: {id(callback)}")
                    return True
                
                self._all_event_subscribers.add(callback)
                subscriber_key = self._get_subscriber_key(callback, None)
                
            else:
                # 特定事件类型订阅
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = set()
                
                if callback in self._subscribers[event_type]:
                    logger.debug(f"回调函数已经订阅了事件类型 {event_type}: {id(callback)}")
                    return True
                
                self._subscribers[event_type].add(callback)
                subscriber_key = self._get_subscriber_key(callback, event_type)
            
            # 为订阅者创建队列
            if subscriber_key not in self._subscriber_queues:
                self._subscriber_queues[subscriber_key] = Queue()
                self._ready_subscribers.add(subscriber_key)
                logger.debug(f"为订阅者创建队列: {subscriber_key}")
            
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
        with self._lock:
            if event_type is None:
                # 取消全局订阅
                if callback in self._all_event_subscribers:
                    self._all_event_subscribers.discard(callback)
                    subscriber_key = self._get_subscriber_key(callback, None)
                    self._cleanup_subscriber(subscriber_key)
                    logger.debug(f"已取消全局订阅: {id(callback)}")
                    return True
                return False

            # 取消特定事件类型订阅
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].discard(callback)
                subscriber_key = self._get_subscriber_key(callback, event_type)
                self._cleanup_subscriber(subscriber_key)
                logger.debug(f"已取消订阅: {event_type} - {id(callback)}")
                return True
            return False

    def _cleanup_subscriber(self, subscriber_key: Tuple[Callable, EventType]):
        """清理订阅者的资源"""
        if subscriber_key in self._subscriber_queues:
            # 向队列发送停止信号
            self._subscriber_queues[subscriber_key].put(None)
            del self._subscriber_queues[subscriber_key]
        
        self._ready_subscribers.discard(subscriber_key)

    def _event_dispatcher(self):
        """事件分发线程主循环"""
        logger.debug("事件分发线程开始运行")
        
        while not self._stop_event.is_set():
            try:
                # 从主队列获取事件
                event = self._main_event_queue.get(timeout=1)
                if event is None:  # 停止信号
                    break
                
                # 处理调度信号
                if event == "__schedule__":
                    self._schedule_subscribers()
                else:
                    # 分发事件到对应的订阅者队列
                    self._dispatch_to_subscribers(event)
                    
                    # 调度空闲的订阅者处理事件
                    self._schedule_subscribers()
                
            except Empty:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"事件分发异常: {e}", exc_info=True)
        
        logger.debug("事件分发线程退出")

    def _dispatch_to_subscribers(self, event: Event):
        """将事件分发到对应的订阅者队列"""
        with self._lock:
            # 分发给全局订阅者
            for callback in self._all_event_subscribers:
                subscriber_key = self._get_subscriber_key(callback, None)
                if subscriber_key in self._subscriber_queues:
                    self._subscriber_queues[subscriber_key].put(event)
            
            # 分发给特定事件类型订阅者
            if event.event_type in self._subscribers:
                for callback in self._subscribers[event.event_type]:
                    subscriber_key = self._get_subscriber_key(callback, event.event_type)
                    if subscriber_key in self._subscriber_queues:
                        self._subscriber_queues[subscriber_key].put(event)

    def _schedule_subscribers(self):
        """调度空闲的订阅者处理事件"""
        to_schedule = []
        
        with self._lock:
            ready_subscribers_snapshot = list(self._ready_subscribers)
        
        for subscriber_key in ready_subscribers_snapshot:
            callback, event_type = subscriber_key
            with self._lock:
                if subscriber_key not in self._ready_subscribers or subscriber_key not in self._subscriber_queues:
                    continue
                
                try:
                    # 非阻塞地获取事件
                    event = self._subscriber_queues[subscriber_key].get_nowait()
                    if event is None:
                        continue
                    
                    # 立即标记订阅者为忙碌状态
                    self._ready_subscribers.discard(subscriber_key)
                    
                    # 收集调度信息
                    to_schedule.append((subscriber_key, callback, event))
                except Empty:
                    continue
        
        for subscriber_key, callback, event in to_schedule:
            self._thread_pool.submit(self._process_event, subscriber_key, callback, event)
        
        if len(to_schedule) > 0:
            logger.debug(f"调度了 {len(to_schedule)} 个订阅者处理事件")

    def _process_event(self, subscriber_key: Tuple[Callable, EventType], callback: Callable, event: Event):
        """在线程池中处理事件"""
        try:
            callback(event)
        except Exception as e:
            logger.error(f"事件处理异常: {e}", exc_info=True)
        finally:
            with self._lock:
                self._ready_subscribers.add(subscriber_key)
            
            self._main_event_queue.put("__schedule__")

    def publish_event(self, event: Event):
        """
        发布事件，支持分发给所有订阅者（包括订阅所有事件的回调）
        使用主队列+分发机制保证顺序性
        """
        # 确保事件总线已启动
        self._ensure_started()
        
        # 将事件放入主队列
        self._main_event_queue.put(event)

    def shutdown(self):
        """关闭事件总线，清理所有资源"""
        logger.info("正在关闭事件总线...")
        self._stop_event.set()
        
        # 向主队列发送停止信号
        self._main_event_queue.put(None)
        
        # 等待分发线程结束
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=5)
            if self._dispatcher_thread.is_alive():
                logger.warning("事件分发线程未能在5秒内结束")
        
        # 关闭线程池
        self._thread_pool.shutdown(wait=True)
        
        logger.info("事件总线已关闭")

    def get_subscription_info(self) -> dict:
        """获取当前订阅信息，用于调试"""
        with self._lock:
            return {
                "subscribers": {str(k): len(v) for k, v in self._subscribers.items()},
                "all_event_subscribers_count": len(self._all_event_subscribers),
                "subscriber_queues_count": len(self._subscriber_queues),
                "ready_subscribers_count": len(self._ready_subscribers),
                "thread_pool_max_workers": self._thread_pool._max_workers,
                "dispatcher_thread_alive": self._dispatcher_thread.is_alive() if self._dispatcher_thread else False,
                "started": self._started
            }