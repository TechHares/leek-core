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
from leek_core.executor import BacktestExecutor, Executor, ExecutorContext
from leek_core.models import ExecutionAsset, Order, LeekComponentConfig, ExecutionContext, OrderStatus, OrderUpdateMessage
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
            if order.is_open:
                execution_order.actual_amount = (execution_order.actual_amount or 0) + (0 if order.order_status.is_failed else order.settle_amount)
                execution_order.actual_ratio = (execution_order.actual_ratio or 0) + (0 if order.order_status.is_failed else order.ratio)
            
            for asset in execution_order.execution_assets:
                if asset.symbol == order.symbol and asset.quote_currency == order.quote_currency and asset.ins_type == order.ins_type and asset.asset_type == order.asset_type:
                    asset.sz = order.sz
                    asset.amount = (asset.amount or 0) + (0 if order.order_status.is_failed else order.settle_amount)
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
        exec_map = {}
        
        # 获取所有执行器，虚拟执行器ID为-1
        virtual_executor = self.get("-1")
        real_executors = [self.get(exec_id) for exec_id in self.components.keys() if exec_id != "-1"]
        
        if not real_executors:
            logger.error("没有找到真实执行器")
            return exec_map
            
        policy_id = execution_order.extra.get("policy_id") if execution_order.extra else None
        # 按资产分组处理
        for asset in execution_order.execution_assets:
            if asset.is_open:
                # 开仓逻辑
                if policy_id:
                    # 虚拟资产路由到虚拟执行器
                    orders = self.to_asset_order(execution_order, virtual_executor, [asset])
                    self._set_orders_fake(orders, policy_id)
                    exec_map.setdefault(virtual_executor, []).extend(orders)
                else:
                    # 真实资产路由到第一个真实执行器
                    orders = self.to_asset_order(execution_order, real_executors[0], [asset])
                    exec_map.setdefault(real_executors[0], []).extend(orders)
            else:
                # 平仓逻辑
                orders = self._route_close_orders(execution_order, asset, virtual_executor, real_executors)
                for executor, order_list in orders.items():
                    exec_map.setdefault(executor, []).extend(order_list)
        
        return exec_map
    
    def _set_orders_fake(self, orders:List[Order], policy_id: str = None):
        """
        设置订单为虚拟订单
        """
        for order in orders:
            order.is_fake = True
            if policy_id:
                if not order.extra:
                    order.extra = {}
                order.extra["policy_id"] = policy_id
    
    def _route_close_orders(self, execution_order: ExecutionContext, asset: ExecutionAsset, 
                           virtual_executor: ExecutorContext, real_executors: List[ExecutorContext]) -> Dict[ExecutorContext, List[Order]]:
        """
        路由平仓订单，优先平虚拟仓位
        """
        orders = {}
        # 计算需要平仓的数量
        virtual_close_sz = asset.virtual_sz or 0
        real_close_sz = asset.sz
        # 平虚拟仓位
        if virtual_close_sz > 0:
            virtual_asset = self.copy_asset_with_newsz(asset, virtual_close_sz)
            virtual_orders = self.to_asset_order(execution_order, virtual_executor, [virtual_asset])
            self._set_orders_fake(virtual_orders)
            orders[virtual_executor] = virtual_orders
        
        # 平真实仓位
        if real_close_sz > 0:
            # 根据每个执行器的持仓量分配平仓数量
            for executor in real_executors:
                executor_id = executor.instance_id
                if executor_id in asset.executor_sz and asset.executor_sz[executor_id] > 0:
                    # 该执行器有持仓，计算平仓数量
                    executor_sz = asset.executor_sz[executor_id]
                    close_sz = min(executor_sz, real_close_sz)
                    
                    if close_sz > 0:
                        real_asset = self.copy_asset_with_newsz(asset, close_sz)
                        real_orders = self.to_asset_order(execution_order, executor, [real_asset])
                        orders[executor] = real_orders
                        real_close_sz -= close_sz
                        
                        if real_close_sz <= 0:
                            break
        
        return orders
    
    def copy_asset_with_newsz(self, original_asset: ExecutionAsset, close_sz: Decimal) -> ExecutionAsset:
        """
        创建虚拟平仓资产
        """
        return ExecutionAsset(
            asset_type=original_asset.asset_type,
            ins_type=original_asset.ins_type,
            symbol=original_asset.symbol,
            side=original_asset.side,
            price=original_asset.price,
            is_open=False,
            ratio=original_asset.ratio * (close_sz / original_asset.sz) if original_asset.sz else original_asset.ratio,
            sz=close_sz,
            amount=original_asset.amount * (close_sz / original_asset.sz) if original_asset.sz else original_asset.amount,
            quote_currency=original_asset.quote_currency,
            extra=original_asset.extra,
            position_id=original_asset.position_id,
            executor_sz=original_asset.executor_sz
        )
    
    def to_asset_order(self, execution_order: ExecutionContext, exec: Executor, assets: List[ExecutionAsset] = None) -> List[Order]:
        """
        将执行上下文转换为订单列表。
        
        Args:
            execution_order: 执行上下文，包含策略信息和待执行的资产列表
            exec: 执行器
            assets: 指定的资产列表，如果为None则使用execution_order中的所有资产
            
        Returns:
            List[Order]: 转换后的订单列表
        """
        orders = []
        target_assets = assets if assets is not None else execution_order.execution_assets
        
        for asset in target_assets:
            order = Order(
                order_id=generate_str(),
                position_id=asset.position_id,
                signal_id=execution_order.signal_id,
                strategy_id=execution_order.strategy_id,
                strategy_instance_id=execution_order.strategy_instance_id,
                exec_order_id=execution_order.context_id,
                order_status=OrderStatus.CREATED,
                signal_time=execution_order.created_time,
                order_time=datetime.now(),
                ratio=asset.ratio,
                sz=asset.sz if not asset.is_open else None,
                
                symbol=asset.symbol,
                quote_currency=asset.quote_currency,
                ins_type=asset.ins_type,
                asset_type=asset.asset_type,
                side=asset.side,
                
                is_open=asset.is_open,
                is_fake=False,
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
        self.add(
            LeekComponentConfig(
                instance_id="-1",
                name="虚拟执行器",
                cls=BacktestExecutor,
                config={
                    "slippage": Decimal(0),
                    "fee_type": 0,
                    "fee": Decimal(0),
                    "limit_order_execution_rate": 100,
                    "check_hold_size": False,
                },
                data=None
            )
        )
        logger.info(f"事件订阅: 执行器管理-{self.name}@{self.instance_id} 订阅 {[e.value for e in [EventType.EXEC_ORDER_CREATED]]}")
