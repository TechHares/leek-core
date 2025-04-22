#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
执行器管理器模块，提供执行器的管理和调度功能。
"""
from typing import Dict, List, Optional

from executor import Executor
from models import Order


class ExecutorManager:
    """
    执行器管理器，负责订单的接收、路由和调度到具体执行器。
    支持多执行器注册、动态路由、状态查询等功能。
    """

    def __init__(self, event_bus, instance_id: str=None, name: str=None, executors: List[Executor]=None):
        self.event_bus = event_bus
        self.instance_id = instance_id
        self.name = name
        self.executors: Dict[str, Executor] = {}  # 执行器ID -> 执行器实例
        if executors:
            for executor in executors:
                self.add_executor(executor)

    def add_executor(self, executor: Executor):
        """
        注册一个新的执行器。
        :param executor: 执行器实例
        """
        self.executors[executor.instance_id] = executor

    def remove_executor(self, executor_id: str):
        """
        移除指定ID的执行器。
        :param executor_id: 执行器ID
        """
        self.executors.pop(executor_id, None)

    def get_executor(self, executor_id: str):
        """
        获取指定ID的执行器实例。
        :param executor_id: 执行器ID
        :return: 执行器实例或None
        """
        return self.executors.get(executor_id)

    def handle_order(self, order: Order):
        """
        处理订单，根据订单属性决定路由到哪个执行器。
        :param order: 订单对象，需包含executor_id/route等字段
        :return: 执行结果
        """
        executor_id = getattr(order, 'executor_id', None) or self.route_order(order)
        executor = self.get_executor(executor_id)
        if executor is None:
            raise ValueError(f"未找到对应执行器: {executor_id}")
        return executor.execute(order)

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
        return list(self.executors.keys())

    def get_executor_status(self, executor_id: str):
        """
        查询指定执行器的状态。
        :param executor_id: 执行器ID
        :return: 状态信息（由具体执行器实现）
        """
        executor = self.get_executor(executor_id)
        if executor and hasattr(executor, 'get_status'):
            return executor.get_status()
        return None
