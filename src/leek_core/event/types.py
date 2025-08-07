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
    # 引擎生命周期事件
    ENGINE_INIT = "engine_init"  # 引擎初始化 todo
    ENGINE_START = "engine_start"  # 引擎启动 todo
    ENGINE_STOP = "engine_stop"  # 引擎停止 todo

    # 数据源事件
    DATA_SOURCE_INIT = "data_source_init"  # 数据源初始化 todo
    DATA_SOURCE_CONNECT = "data_source_connect"  # 数据源连接 todo
    DATA_SOURCE_DISCONNECT = "data_source_disconnect"  # 数据源断开 todo
    DATA_SOURCE_RECONNECT = "data_source_reconnect"  # 数据源重连 todo
    DATA_SOURCE_ERROR = "data_source_error"  # 数据源错误 todo
    DATA_SOURCE_STATUS_CHANGE = "data_source_status_change"  # 数据源状态改变 todo

    # 数据事件
    DATA_SOURCE_SUBSCRIBE = "data_source_subscribe"  # 数据源订阅
    DATA_SOURCE_UNSUBSCRIBE = "data_source_unsubscribe"  # 数据源取消订阅
    DATA_RECEIVED = "data_received"  # 接收到数据
    DATA_REQUEST = "data_request"  # 数据请求
    DATA_RESPONSE = "data_response"  # 数据响应

    # 策略事件
    STRATEGY_INIT = "strategy_init"  # 策略初始化 todo
    STRATEGY_START = "strategy_start"  # 策略启动 todo
    STRATEGY_STOP = "strategy_stop"  # 策略停止 todo
    STRATEGY_SIGNAL = "strategy_signal"  # 策略产生信号
    STRATEGY_SIGNAL_FINISH = "strategy_signal_finish"  # 策略信号完成

    # 风控插件事件
    RISK_PLUGIN_INIT = "risk_plugin_init"  # 插件初始化 todo
    RISK_PLUGIN_START = "risk_plugin_start"  # 插件绑定仓位启动 todo
    RISK_PLUGIN_STOP = "risk_plugin_stop"  # 插件绑定停止 todo

    # 风控事件
    RISK_MANAGER_INIT = "risk_manager_init"  # 风控检查开始 todo
    RISK_MANAGER_START = "risk_manager_start"  # 风控检查开始 todo
    RISK_MANAGER_STOP = "risk_manager_stop"  # 风控检查开始 todo
    RISK_MANAGER_UPDATE = "risk_manager_update"  # 风控检查开始 todo

    RISK_CHECK_START = "risk_check_start"  # 风控检查开始 todo
    RISK_CHECK_PASS = "risk_check_pass"  # 风控检查通过 todo
    RISK_CHECK_REJECT = "risk_check_reject"  # 风控检查拒绝 todo

    # 仓位管理事件
    POSITION_POLICY_ADD = "position_policy_add"  # 仓位风控添加 todo
    POSITION_POLICY_DEL = "position_policy_del"  # 仓位风控删除 todo
    POSITION_INIT = "position_init"  # 仓位管理初始化
    POSITION_OPEN = "position_open"  # 开仓 todo
    POSITION_CLOSE = "position_close"  # 平仓 todo
    POSITION_UPDATE = "position_update"  # 仓位更新

    # 执行器事件
    EXECUTOR_INIT = "executor_load_data"  # 执行器初始化 todo
    EXECUTOR_START = "executor_start"  # 执行器启动 todo
    EXECUTOR_STOP = "executor_stop"  # 执行器停止 todo
    EXECUTOR_UPDATE = "executor_update"  # 执行器更新 todo

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
