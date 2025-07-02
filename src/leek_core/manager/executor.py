#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
执行器管理器模块，提供执行器的管理和调度功能。
"""
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from leek_core.event import EventBus, EventType, Event
from leek_core.executor import Executor, ExecutorContext
from leek_core.models import Order, LeekComponentConfig, ExecutionContext, OrderStatus, OrderUpdateMessage
from leek_core.utils.decorator import thread_lock
from .base import ComponentManager
from leek_core.utils import get_logger, log_method, generate_str

logger = get_logger(__name__)


class ExecutorManager(ComponentManager[ExecutorContext, Executor, Dict[str, Any]]):
    """
    执行器管理器，负责订单的接收、路由和调度到具体执行器。
    支持多执行器注册、动态路由、状态查询等功能。
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[ExecutorContext, None]):
        super().__init__(event_bus, config)

        self.execution_order_map = {}
        self.order_map = {}

    @log_method(level=logging.DEBUG, log_execution_time=True)
    def handle_order(self, execution_order: ExecutionContext):
        """
        处理订单，根据订单属性决定路由到哪个执行器。
        :param order: 订单对象，需包含executor_id/route等字段
        :return: 执行结果
        """
        if len(self.components) == 0:
            logger.warning(f"执行器管理器-{self.name}@{self.instance_id} 没有执行器，跳过订单处理")
            execution_order.actual_ratio = Decimal(0)
            execution_order.actual_amount = Decimal(0)
            execution_order.actual_pnl = Decimal(0)
            execution_order.is_finish = True
            self.event_bus.publish_event(Event(EventType.EXEC_ORDER_UPDATED, execution_order))
            return
        
        self.execution_order_map[execution_order.context_id] = execution_order
        exec_map = self.route_order(execution_order)
        self.order_map[execution_order.context_id]=set()
        for exec, orders in exec_map.items():
            for order in orders:
                self.order_map[execution_order.context_id].add(order.order_id)
            exec.send_order(orders)

    @thread_lock()
    def order_update(self, order: Order):
        try:
            if not order.order_status.is_finished or order.exec_order_id not in self.order_map:
                return

            execution_order = self.execution_order_map[order.exec_order_id]
            if execution_order.actual_amount is None:
                execution_order.actual_amount = Decimal(0)
            execution_order.actual_amount += order.settle_amount or 0
            if order.is_open:
                execution_order.actual_ratio = (execution_order.actual_ratio or 0) + order.ratio
            
            for asset in execution_order.execution_assets:
                if asset.symbol == order.symbol and asset.quote_currency == order.quote_currency and asset.ins_type == order.ins_type and asset.asset_type == order.asset_type:
                    asset.sz = order.sz
                    if asset.amount is None:
                        asset.amount = order.settle_amount
                    break
            if order.pnl:
                execution_order.actual_pnl = (execution_order.actual_pnl or 0) + order.pnl
            

            self.order_map[order.exec_order_id].remove(order.order_id)
            if len(self.order_map[order.exec_order_id]) == 0:
                self.order_map.pop(order.exec_order_id)
                self.execution_order_map.pop(order.exec_order_id)
                execution_order.is_finish = True
            self.event_bus.publish_event(Event(EventType.EXEC_ORDER_UPDATED, execution_order))
        except BaseException as e:
            logger.error(f"执行订单更新异常: {e}", exc_info=True)

    def route_order(self, execution_order: ExecutionContext) -> Dict[ExecutorContext, List[Order]]:
        """
        路由策略：根据订单属性决定分发到哪个执行器。
        可自定义扩展（如按类型、标的、策略等）。
        :param order: 订单对象
        :return: 执行器ID
        """
        # 暂时只支持单个执行器，由指定取对应执行器， 没有取第一个支持的执行器
        exec_map = {}
        exec = self.get(execution_order.target_executor_id if execution_order.target_executor_id else list(self.components.keys())[0])
        orders = self.to_asset_order(execution_order, exec)
        exec_map[exec] = orders
        return exec_map
    
    def to_asset_order(self, execution_order: ExecutionContext, exec: Executor) -> List[Order]:
        """
        将执行上下文转换为订单列表。
        
        Args:
            execution_order: 执行上下文，包含策略信息和待执行的资产列表
            
        Returns:
            List[Order]: 转换后的订单列表
        """
        orders = []
        for asset in execution_order.execution_assets:
            order = Order(
                order_id=generate_str(),
                position_id=asset.position_id,
                signal_id=execution_order.signal_id,
                strategy_id=execution_order.strategy_id,
                strategy_instant_id=execution_order.strategy_instant_id,
                exec_order_id=execution_order.context_id,
                order_status=OrderStatus.CREATED,
                signal_time=execution_order.created_time,
                order_time=datetime.now(),
                ratio=asset.ratio,
                sz=min(asset.executor_sz[exec.instance_id], asset.sz) if not asset.is_open else None,
                
                symbol=asset.symbol,
                quote_currency=asset.quote_currency,
                ins_type=asset.ins_type,
                asset_type=asset.asset_type,
                side=asset.side,
                
                is_open=asset.is_open,
                is_fake=asset.is_fake,
                order_amount=asset.amount,
                order_price=asset.price,
                order_type=execution_order.order_type,
                
                leverage=Decimal(execution_order.leverage),
                trade_mode=execution_order.trade_mode,
                extra=asset.extra
            )
            orders.append(order)
        return orders

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

    def on_start(self):
        self.event_bus.subscribe_event(EventType.EXEC_ORDER_CREATED, lambda e: self.handle_order(e.data))
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, lambda e: self.order_update(e.data))
        logger.info(f"事件订阅: 执行器管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.EXEC_ORDER_CREATED]]}")
