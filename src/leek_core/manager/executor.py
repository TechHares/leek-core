#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
执行器管理器模块，提供执行器的管理和调度功能。
"""
import logging
from typing import Dict, List, Optional, Any

from leek_core.event import EventBus, EventType
from leek_core.executor import Executor, ExecutorContext
from leek_core.models import Order, SubOrder, LeekComponentConfig
from .base import ComponentManager
from leek_core.utils import get_logger, log_method

logger = get_logger(__name__)


class ExecutorManager(ComponentManager[ExecutorContext, Executor, Dict[str, Any]]):
    """
    执行器管理器，负责订单的接收、路由和调度到具体执行器。
    支持多执行器注册、动态路由、状态查询等功能。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[ExecutorContext, None]):
        super().__init__(event_bus, config)

    @log_method(level=logging.DEBUG, log_execution_time=True)
    def handle_order(self, order: Order):
        """
        处理订单，根据订单属性决定路由到哪个执行器。
        :param order: 订单对象，需包含executor_id/route等字段
        :return: 执行结果
        """
        executor_id = order.target_executor_id or self.route_order(order)
        executor = self.get(executor_id)
        if executor is None:
            raise ValueError(f"未找到对应执行器: {executor_id}")
        return executor.send_order(order)

    def route_order(self, order) -> Optional[str]:
        """
        路由策略：根据订单属性决定分发到哪个执行器。
        可自定义扩展（如按类型、标的、策略等）。
        :param order: 订单对象
        :return: 执行器ID
        """
        # 示例：默认取订单的executor_id字段
        return getattr(order, 'executor_id', None)

    def list_executors(self) -> List[str]:
        """
        返回所有已注册执行器ID。
        """
        return list(self.components.keys())

    def get_executor_status(self, executor_id: str):
        """
        查询指定执行器的状态。
        :param executor_id: 执行器ID
        :return: 状态信息（由具体执行器实现）
        """
        executor = self.get(executor_id)
        if executor and hasattr(executor, 'get_status'):
            return executor.get_status()
        return None

    def executor_callback(self, order: SubOrder):
        ...

    def on_start(self):
        self.event_bus.subscribe_event(EventType.ORDER_CREATED, lambda e: self.handle_order(e.data))
        logger.info(f"事件订阅: 执行器管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.ORDER_CREATED]]}")
