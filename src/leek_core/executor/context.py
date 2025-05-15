#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any

from leek_core.base import LeekContext
from leek_core.event import EventBus, Event, EventType
from leek_core.models import LeekComponentConfig, SubOrder, Order
from .base import Executor


class ExecutorContext(LeekContext):
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        super().__init__(event_bus, config)

        self.orders: dict[str, SubOrder] = {}
        self.executor: Executor = self.create_component()
        self.executor.callback = self._trade_callback

    def check_order(self, order: Order) -> bool:
        """
        检查订单是否可执行

        参数:
            order: 订单信息
        返回:
            bool: True 表示可执行，False 表示不可执行
        """
        return self.executor.check_order(order)

    def send_order(self, order: SubOrder):
        """
        下单

        参数:
            order: 订单信息
        """
        self.executor.send_order(order)


    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        撤单接口，子类需实现。
        :param order_id: 订单ID
        :param symbol: 交易对
        """
        self.executor.cancel_order(order_id, symbol, **kwargs)

    def get_order(self, order_id: str) -> 'SubOrder|None':
        """
        根据订单ID获取订单对象
        """
        return self.orders.get(order_id)

    def update_order(self, order: 'SubOrder'):
        """
        新增或更新订单对象
        """
        self.orders[order.order_id] = order

    def remove_order(self, order_id: str):
        """
        删除已完成（或撤销）订单
        """
        if order_id in self.orders:
            del self.orders[order_id]


    def _trade_callback(self, order):
        """
        交易回调，反馈成交详细等信息。
        若订单已完成或撤销，则自动删除。
        """
        # 用户自定义回调
        self.event_bus.publish_event(Event(
            event_type=EventType.EXECUTOR_UPDATE,
            data=order,
        ))
        # 自动清理已完成订单
        if hasattr(order, 'order_id') and hasattr(order, 'state'):
            if order.state in ("filled", "canceled"):  # 成交或撤单即删除
                self.remove_order(order.order_id)

    def on_start(self):
        self.executor.on_start()

    def on_stop(self):
        self.executor.on_stop()