#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import threading
import time
from unittest.mock import Mock, patch
from queue import Queue

from leek_core.event.bus import EventBus
from leek_core.event.types import EventType, Event


class TestEventBus(unittest.TestCase):
    """事件总线测试用例"""

    def setUp(self):
        """测试前准备"""
        self.event_bus = EventBus()
        self.received_events = []
        self.event_order = []

    def tearDown(self):
        """测试后清理"""
        self.event_bus.shutdown()

    def test_basic_subscription_and_publish(self):
        """测试基本的订阅和发布功能"""
        # 创建回调函数
        def callback(event):
            self.received_events.append(event.data)

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)

        # 发布事件
        test_data = {"test": "data"}
        event = Event(event_type=EventType.POSITION_UPDATE, data=test_data)
        self.event_bus.publish_event(event)

        # 等待事件处理完成
        time.sleep(0.1)

        # 验证事件被接收
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0], test_data)

    def test_event_order_guarantee(self):
        """测试事件顺序保证"""
        # 创建回调函数，记录事件顺序
        def callback(event):
            self.event_order.append(event.data)
            time.sleep(0.01)  # 模拟处理时间

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)

        # 发布多个事件
        for i in range(5):
            event = Event(event_type=EventType.POSITION_UPDATE, data=f"event_{i}")
            self.event_bus.publish_event(event)

        # 等待所有事件处理完成
        time.sleep(0.5)

        # 验证事件按顺序处理
        self.assertEqual(len(self.event_order), 5)
        for i in range(5):
            self.assertEqual(self.event_order[i], f"event_{i}")

    def test_multiple_event_types_concurrent(self):
        """测试多个事件类型并发处理"""
        position_events = []
        order_events = []

        def position_callback(event):
            position_events.append(event.data)
            time.sleep(0.01)

        def order_callback(event):
            order_events.append(event.data)
            time.sleep(0.01)

        # 订阅不同事件类型
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, position_callback)
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, order_callback)

        # 并发发布不同事件类型
        for i in range(3):
            self.event_bus.publish_event(Event(EventType.POSITION_UPDATE, f"pos_{i}"))
            self.event_bus.publish_event(Event(EventType.ORDER_UPDATED, f"order_{i}"))

        # 等待处理完成
        time.sleep(0.5)

        # 验证每种事件类型内部有序，但不同类型可以并发
        self.assertEqual(len(position_events), 3)
        self.assertEqual(len(order_events), 3)
        
        for i in range(3):
            self.assertEqual(position_events[i], f"pos_{i}")
            self.assertEqual(order_events[i], f"order_{i}")

    def test_duplicate_subscription(self):
        """测试重复订阅处理"""
        callback = Mock()

        # 第一次订阅
        result1 = self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)
        self.assertTrue(result1)

        # 重复订阅
        result2 = self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)
        self.assertTrue(result2)  # 应该返回True，但不会创建新的队列

        # 发布事件
        event = Event(EventType.POSITION_UPDATE, "test_data")
        self.event_bus.publish_event(event)

        # 等待处理完成
        time.sleep(0.1)

        # 验证回调只被调用一次（因为只有一个队列）
        self.assertEqual(callback.call_count, 1)

    def test_unsubscribe(self):
        """测试取消订阅"""
        callback = Mock()

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)

        # 发布事件
        event = Event(EventType.POSITION_UPDATE, "test_data")
        self.event_bus.publish_event(event)

        # 等待处理完成
        time.sleep(0.1)
        self.assertEqual(callback.call_count, 1)

        # 取消订阅
        result = self.event_bus.unsubscribe_event(EventType.POSITION_UPDATE, callback)
        self.assertTrue(result)

        # 再次发布事件
        event2 = Event(EventType.POSITION_UPDATE, "test_data_2")
        self.event_bus.publish_event(event2)

        # 等待处理完成
        time.sleep(0.1)

        # 验证回调没有被再次调用
        self.assertEqual(callback.call_count, 1)

    def test_global_subscription(self):
        """测试全局订阅（订阅所有事件）"""
        global_events = []

        def global_callback(event):
            global_events.append((event.event_type, event.data))

        # 订阅所有事件
        self.event_bus.subscribe_event(None, global_callback)

        # 发布不同类型的事件
        events = [
            Event(EventType.POSITION_UPDATE, "pos_data"),
            Event(EventType.ORDER_UPDATED, "order_data"),
            Event(EventType.STRATEGY_SIGNAL, "signal_data")
        ]

        for event in events:
            self.event_bus.publish_event(event)

        # 等待处理完成
        time.sleep(0.2)

        # 验证全局回调接收了所有事件
        self.assertEqual(len(global_events), 3)
        self.assertEqual(global_events[0], (EventType.POSITION_UPDATE, "pos_data"))
        self.assertEqual(global_events[1], (EventType.ORDER_UPDATED, "order_data"))
        self.assertEqual(global_events[2], (EventType.STRATEGY_SIGNAL, "signal_data"))

    def test_multiple_callbacks_same_event(self):
        """测试同一事件类型的多个回调"""
        callback1_events = []
        callback2_events = []

        def callback1(event):
            callback1_events.append(event.data)
            time.sleep(0.01)

        def callback2(event):
            callback2_events.append(event.data)
            time.sleep(0.01)

        # 订阅同一事件类型的两个回调
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback1)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback2)

        # 发布事件
        for i in range(3):
            self.event_bus.publish_event(Event(EventType.POSITION_UPDATE, f"event_{i}"))

        # 等待处理完成
        time.sleep(0.3)

        # 验证两个回调都按顺序处理了事件
        self.assertEqual(len(callback1_events), 3)
        self.assertEqual(len(callback2_events), 3)
        
        for i in range(3):
            self.assertEqual(callback1_events[i], f"event_{i}")
            self.assertEqual(callback2_events[i], f"event_{i}")

    def test_callback_exception_handling(self):
        """测试回调异常处理"""
        def failing_callback(event):
            raise Exception("Test exception")

        def working_callback(event):
            self.received_events.append(event.data)

        # 订阅一个会失败的回调和一个正常回调
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, failing_callback)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, working_callback)

        # 发布事件
        event = Event(EventType.POSITION_UPDATE, "test_data")
        self.event_bus.publish_event(event)

        # 等待处理完成
        time.sleep(0.2)

        # 验证正常回调仍然工作
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0], "test_data")

    def test_subscription_info(self):
        """测试获取订阅信息"""
        callback1 = Mock()
        callback2 = Mock()

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback1)
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, callback2)
        self.event_bus.subscribe_event(None, callback1)  # 全局订阅

        # 获取订阅信息
        info = self.event_bus.get_subscription_info()

        # 验证信息正确
        self.assertEqual(info["subscribers"]["EventType.POSITION_UPDATE"], 1)
        self.assertEqual(info["subscribers"]["EventType.ORDER_UPDATED"], 1)
        self.assertEqual(info["all_event_subscribers_count"], 1)
        self.assertEqual(info["callback_queues_count"], 3)  # 2个特定类型 + 1个全局
        self.assertEqual(info["worker_threads_count"], 3)

    def test_shutdown(self):
        """测试关闭事件总线"""
        callback = Mock()

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)

        # 发布事件
        event = Event(EventType.POSITION_UPDATE, "test_data")
        self.event_bus.publish_event(event)

        # 等待事件开始处理
        time.sleep(0.05)

        # 关闭事件总线
        self.event_bus.shutdown()

        # 验证回调被调用（事件在关闭前被处理）
        self.assertEqual(callback.call_count, 1)

    def test_high_concurrency(self):
        """测试高并发场景"""
        event_count = 100
        callback_events = []

        def callback(event):
            callback_events.append(event.data)

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, callback)

        # 快速发布大量事件
        threads = []
        for i in range(10):  # 10个线程并发发布
            thread = threading.Thread(
                target=lambda start: [
                    self.event_bus.publish_event(Event(EventType.POSITION_UPDATE, f"event_{start + j}"))
                    for j in range(10)
                ],
                args=(i * 10,)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 等待事件处理完成
        time.sleep(1.0)

        # 验证所有事件都被处理
        self.assertEqual(len(callback_events), event_count)
        
        # 验证事件按顺序处理（虽然发布是并发的，但处理应该有序）
        # 注意：由于并发发布，顺序可能不完全按event_0, event_1...，但每个线程内部应该有序
        event_numbers = [int(event.split('_')[1]) for event in callback_events]
        # 检查是否有重复或丢失
        self.assertEqual(len(set(event_numbers)), event_count)
        self.assertEqual(min(event_numbers), 0)
        self.assertEqual(max(event_numbers), event_count - 1)


if __name__ == '__main__':
    unittest.main()
