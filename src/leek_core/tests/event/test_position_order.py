#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import threading
import time
from decimal import Decimal

from leek_core.event.bus import EventBus
from leek_core.event.types import EventType, Event


class TestPositionOrder(unittest.TestCase):
    """测试仓位更新顺序保证"""

    def setUp(self):
        """测试前准备"""
        self.event_bus = EventBus()
        self.position_updates = []
        self.lock = threading.Lock()

    def tearDown(self):
        """测试后清理"""
        self.event_bus.shutdown()

    def position_update_callback(self, event):
        """仓位更新回调函数"""
        with self.lock:
            self.position_updates.append({
                'position_id': event.data.get('position_id'),
                'amount': event.data.get('amount'),
                'executor_sz': event.data.get('executor_sz'),
                'update_time': event.data.get('update_time'),
                'timestamp': time.time()
            })

    def test_position_update_sequence(self):
        """测试仓位更新序列，模拟你遇到的实际问题"""
        # 订阅仓位更新事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.position_update_callback)

        # 模拟你的两个仓位更新事件
        position_data_1 = {
            'position_id': '220019776674598912',
            'strategy_id': '2',
            'strategy_instance_id': 'SHIB_USDT_3_5m',
            'symbol': 'SHIB',
            'quote_currency': 'USDT',
            'ins_type': 3,
            'asset_type': 'crypto',
            'side': 2,
            'cost_price': '0.000012309',
            'amount': '1.8463500000',  # 还有仓位
            'ratio': '0.01083092',
            'close_price': '0.000012023',
            'current_price': '0.000012166',
            'total_amount': '2.4618000000',
            'total_back_amount': '0.6297500000',
            'total_sz': '400000.0',
            'executor_id': '1',
            'pnl': '0.0112299000',
            'fee': '-0.0030701',
            'friction': '0',
            'leverage': '2',
            'open_time': 1756541196828,
            'update_time': 1756683782687,  # 第一个时间戳
            'executor_sz': {'1': '300000.0'},  # 还有执行器仓位
            'order_states': {
                '220019768386654208': {
                    'order_id': '220019768386654208',
                    'is_open': True,
                    'settle_amount': '0',
                    'fee': '0',
                    'friction': '0',
                    'pnl': '0',
                    'sz': '0'
                },
                '220689934334300160': {
                    'order_id': '220689934334300160',
                    'is_open': False,
                    'settle_amount': '0.6297500000',
                    'fee': '-0.0006083',
                    'friction': '0',
                    'pnl': '0.0143',
                    'sz': '100000.0'
                }
            },
            'virtual_positions': []
        }

        position_data_2 = {
            'position_id': '220019776674598912',
            'strategy_id': '2',
            'strategy_instance_id': 'SHIB_USDT_3_5m',
            'symbol': 'SHIB',
            'quote_currency': 'USDT',
            'ins_type': 3,
            'asset_type': 'crypto',
            'side': 2,
            'cost_price': '0.000012309',
            'amount': '0E-10',  # 仓位清零
            'ratio': '0E-8',
            'close_price': '0.000012023',
            'current_price': '0.000012166',
            'total_amount': '2.4618000000',
            'total_back_amount': '2.5190000000',
            'total_sz': '400000.0',
            'executor_id': '1',
            'pnl': '0.0523050000',
            'fee': '-0.0048950',
            'friction': '0',
            'leverage': '2',
            'open_time': 1756541196828,
            'update_time': 1756683782692,  # 第二个时间戳，比第一个大5毫秒
            'executor_sz': {'1': '0.0'},  # 执行器仓位清零
            'order_states': {
                '220019768386654208': {
                    'order_id': '220019768386654208',
                    'is_open': True,
                    'settle_amount': '0',
                    'fee': '0',
                    'friction': '0',
                    'pnl': '0',
                    'sz': '0'
                },
                '220689934334300160': {
                    'order_id': '220689934334300160',
                    'is_open': False,
                    'settle_amount': '2.5190000000',
                    'fee': '-0.0024332',
                    'friction': '0',
                    'pnl': '0.0572',
                    'sz': '400000.0'
                }
            },
            'virtual_positions': []
        }

        # 发布第一个事件
        event1 = Event(EventType.POSITION_UPDATE, position_data_1)
        self.event_bus.publish_event(event1)

        # 等待一小段时间，模拟网络延迟
        time.sleep(0.01)

        # 发布第二个事件
        event2 = Event(EventType.POSITION_UPDATE, position_data_2)
        self.event_bus.publish_event(event2)

        # 等待事件处理完成
        time.sleep(0.5)

        # 验证事件按顺序处理
        self.assertEqual(len(self.position_updates), 2)

        # 验证第一个事件
        first_update = self.position_updates[0]
        self.assertEqual(first_update['position_id'], '220019776674598912')
        self.assertEqual(first_update['amount'], '1.8463500000')
        self.assertEqual(first_update['executor_sz'], {'1': '300000.0'})
        self.assertEqual(first_update['update_time'], 1756683782687)

        # 验证第二个事件（最终状态）
        second_update = self.position_updates[1]
        self.assertEqual(second_update['position_id'], '220019776674598912')
        self.assertEqual(second_update['amount'], '0E-10')  # 仓位清零
        self.assertEqual(second_update['executor_sz'], {'1': '0.0'})  # 执行器仓位清零
        self.assertEqual(second_update['update_time'], 1756683782692)

        # 验证时间戳顺序
        self.assertLess(first_update['update_time'], second_update['update_time'])
        self.assertLess(first_update['timestamp'], second_update['timestamp'])

    def test_reverse_order_publish(self):
        """测试逆序发布事件（第二个事件先到）"""
        # 订阅仓位更新事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.position_update_callback)

        # 创建两个事件，第二个事件时间戳更大
        position_data_1 = {
            'position_id': 'test_position_123',
            'amount': '100.0',
            'executor_sz': {'1': '100.0'},
            'update_time': 1000
        }

        position_data_2 = {
            'position_id': 'test_position_123',
            'amount': '0.0',  # 清零
            'executor_sz': {'1': '0.0'},  # 清零
            'update_time': 2000  # 更大的时间戳
        }

        # 先发布第二个事件（时间戳更大）
        event2 = Event(EventType.POSITION_UPDATE, position_data_2)
        self.event_bus.publish_event(event2)

        # 等待一小段时间
        time.sleep(0.01)

        # 再发布第一个事件（时间戳更小）
        event1 = Event(EventType.POSITION_UPDATE, position_data_1)
        self.event_bus.publish_event(event1)

        # 等待事件处理完成
        time.sleep(0.5)

        # 验证事件按发布顺序处理（事件总线的正确行为）
        self.assertEqual(len(self.position_updates), 2)

        # 第一个处理的事件应该是先发布的（时间戳大的）
        first_update = self.position_updates[0]
        self.assertEqual(first_update['update_time'], 2000)
        self.assertEqual(first_update['amount'], '0.0')

        # 第二个处理的事件应该是后发布的（时间戳小的）
        second_update = self.position_updates[1]
        self.assertEqual(second_update['update_time'], 1000)
        self.assertEqual(second_update['amount'], '100.0')

        # 验证最终状态是第二个事件（后发布的覆盖了先发布的）
        # 这模拟了你的实际问题：后到的事件覆盖了先到的事件

    def test_concurrent_position_updates(self):
        """测试并发仓位更新"""
        # 订阅仓位更新事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.position_update_callback)

        # 创建多个仓位更新事件
        events = []
        for i in range(10):
            position_data = {
                'position_id': f'test_position_{i}',
                'amount': str(100 - i * 10),
                'executor_sz': {'1': str(100 - i * 10)},
                'update_time': 1000 + i * 100  # 递增的时间戳
            }
            events.append(Event(EventType.POSITION_UPDATE, position_data))

        # 并发发布事件
        threads = []
        for event in events:
            thread = threading.Thread(target=self.event_bus.publish_event, args=(event,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 等待事件处理完成
        time.sleep(1.0)

        # 验证所有事件都被处理
        self.assertEqual(len(self.position_updates), 10)

        # 验证事件按时间戳顺序处理
        update_times = [update['update_time'] for update in self.position_updates]
        self.assertEqual(update_times, sorted(update_times))

        # 验证每个仓位ID只出现一次（最终状态）
        position_ids = [update['position_id'] for update in self.position_updates]
        unique_position_ids = list(set(position_ids))
        self.assertEqual(len(unique_position_ids), len(position_ids))

    def test_multiple_positions_concurrent(self):
        """测试多个不同仓位的并发更新"""
        # 订阅仓位更新事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.position_update_callback)

        # 创建多个不同仓位的更新事件
        events = []
        for i in range(5):  # 5个不同的仓位
            for j in range(3):  # 每个仓位3个更新
                position_data = {
                    'position_id': f'position_{i}',
                    'amount': str(100 - j * 30),
                    'executor_sz': {'1': str(100 - j * 30)},
                    'update_time': 1000 + i * 1000 + j * 100  # 确保每个仓位内部有序
                }
                events.append(Event(EventType.POSITION_UPDATE, position_data))

        # 并发发布事件
        threads = []
        for event in events:
            thread = threading.Thread(target=self.event_bus.publish_event, args=(event,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 等待事件处理完成
        time.sleep(1.5)

        # 验证所有事件都被处理
        self.assertEqual(len(self.position_updates), 15)

        # 验证每个仓位的事件按时间戳顺序处理
        position_groups = {}
        for update in self.position_updates:
            pos_id = update['position_id']
            if pos_id not in position_groups:
                position_groups[pos_id] = []
            position_groups[pos_id].append(update)

        for pos_id, updates in position_groups.items():
            update_times = [update['update_time'] for update in updates]
            self.assertEqual(update_times, sorted(update_times), 
                           f"仓位 {pos_id} 的事件顺序不正确")


if __name__ == '__main__':
    unittest.main()
