#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any
import copy

from leek_core.base import LeekContext
from leek_core.event import EventBus, Event, EventType
from leek_core.models import LeekComponentConfig, Order, OrderUpdateMessage, OrderStatus
from .base import Executor
from leek_core.utils import run_func_timeout
from leek_core.utils import get_logger
import time

logger = get_logger(__name__)


class ExecutorContext(LeekContext):
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        super().__init__(event_bus, config)

        self.orders: dict[str, Order] = {}
        self.executor: Executor = self.create_component()
        self.executor.instance_id = self.instance_id
        self.executor.callback = self._trade_callback

        self.max_retry_count = 3
        self.retry_interval = 1

    def check_order(self, order: Order) -> bool:
        """
        检查订单是否可执行

        参数:
            order: 订单信息
        返回:
            bool: True 表示可执行，False 表示不可执行
        """
        return self.executor.check_order(order)
    
    def update(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        self.config = config
        run_func_timeout(self.executor.on_stop, [], {}, 5)
        self.executor = self.create_component()
        self.executor.callback = self._trade_callback
        is_finish = run_func_timeout(self.executor.on_start, [], {}, 20)
        if not is_finish:
            self.is_connected = False
            logger.error(f"执行器{self.name}更新超时")
            return

    def send_order(self, orders: Order):
        """
        下单

        参数:
            order: 订单信息
        """
        for order in orders:
            order.executor_id = self.instance_id
            self.orders[order.order_id] = order
        self.event_bus.publish_event(Event(EventType.ORDER_CREATED, orders))
        retry_count = max(self.max_retry_count, 1)
        while retry_count > 0:
            retry_count -= 1
            try:
                self.executor.send_order(orders)
                for order in orders:
                    if order.order_status == OrderStatus.CREATED:
                        order.order_status = OrderStatus.SUBMITTED
                        # 发布事件时深拷贝，避免后续修改影响已派发事件
                        self.event_bus.publish_event(Event(EventType.ORDER_UPDATED, copy.deepcopy(order)))
                return
            except Exception as e:
                if retry_count <= 1:
                    logger.error(f"{self.instance_id}/{self.name}执行订单失败: {e}", exc_info=True)
                    for order in orders:
                        order.order_status = OrderStatus.ERROR
                        # 发布事件时深拷贝，避免后续修改影响已派发事件
                        self.event_bus.publish_event(Event(EventType.ORDER_UPDATED, copy.deepcopy(order)))
                    return
                else:
                    logger.warning(f"{self.instance_id}/{self.name}执行订单失败, 重试次数: {retry_count}: {e}, ")
                if self.retry_interval > 0:
                    time.sleep(self.retry_interval)



    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        撤单接口，子类需实现。
        :param order_id: 订单ID
        :param symbol: 交易对
        """
        self.executor.cancel_order(order_id, symbol, **kwargs)


    def _trade_callback(self, msg: OrderUpdateMessage):
        """
        交易回调，反馈成交详细等信息。
        若订单已完成或撤销，则自动删除。
        """
        if msg.order_id not in self.orders:
            return

        order = self.orders[msg.order_id]
        order.order_status = msg.order_status
        order.settle_amount = msg.settle_amount
        order.execution_price = msg.execution_price
        order.sz = msg.sz
        order.fee = msg.fee
        order.pnl = msg.pnl
        order.unrealized_pnl = msg.unrealized_pnl
        order.finish_time = msg.finish_time
        order.friction = msg.friction
        if msg.extra:
            if not order.extra:
                order.extra = {}
            order.extra.update(msg.extra)
        order.market_order_id = msg.market_order_id
        try:
            # 用户自定义回调
            # 发布事件时深拷贝，避免后续修改影响已派发事件
            self.event_bus.publish_event(Event(
                event_type=EventType.ORDER_UPDATED,
                data=copy.deepcopy(order),
            ))
        finally:
            if order.order_status.is_finished:
                del self.orders[order.order_id]

    def on_start(self):
        self.executor.on_start()

    def on_stop(self):
        self.executor.on_stop()