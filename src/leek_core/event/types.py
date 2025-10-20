#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件定义和事件总线扩展
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class EventType(Enum):
    """
    事件类型定义
    """
    # 数据事件
    DATA_SOURCE_SUBSCRIBE = "data_source_subscribe"  # 数据源订阅
    DATA_SOURCE_UNSUBSCRIBE = "data_source_unsubscribe"  # 数据源取消订阅
    DATA_RECEIVED = "data_received"  # 接收到数据
    DATA_REQUEST = "data_request"  # 数据请求

    # 策略事件
    STRATEGY_SIGNAL_MANUAL = "strategy_signal_manual"  # 手动信号(前置信号)

    STRATEGY_SIGNAL = "strategy_signal"  # 策略产生信号
    STRATEGY_SIGNAL_FINISH = "strategy_signal_finish"  # 策略信号完成

    # 风控事件
    RISK_TRIGGERED = "risk_triggered"  # 风控触发事件

    # 仓位管理事件
    POSITION_POLICY_ADD = "position_policy_add"  # 仓位风控添加
    POSITION_POLICY_DEL = "position_policy_del"  # 仓位风控删除
    POSITION_INIT = "position_init"  # 仓位管理初始化
    POSITION_UPDATE = "position_update"  # 仓位更新

    # 订单执行事件
    EXEC_ORDER_CREATED = "exec_order_created"  # 订单创建
    EXEC_ORDER_UPDATED = "exec_order_updated"  # 订单更新
    ORDER_CREATED = "order_created"  # 订单路由
    ORDER_UPDATED = "order_updated"  # 订单更新

    # 资金管理事件
    TRANSACTION = "transaction"  # 资金流水


@dataclass
class EventSource:
    """
    事件源
    """
    instance_id: str
    name: str
    cls: str = None
    extra: dict = field(default_factory=dict)


class Event:
    """
    事件对象
    """

    def __init__(self, event_type: EventType, data: Any = None, source: EventSource = None):
        """
        初始化事件

        参数:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
        """
        self.event_type = event_type
        self.data = data
        self.source = source

    def __str__(self):
        return f"Event(type={self.event_type.value}, source={self.source}, data={self.data})"
