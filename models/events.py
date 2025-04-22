#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事件定义和事件总线扩展
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .component import Component


class EventType(Enum):
    """
    事件类型定义
    """
    # 引擎生命周期事件
    ENGINE_INIT = "engine_init"    # 引擎初始化 todo
    ENGINE_START = "engine_start"  # 引擎启动 todo
    ENGINE_STOP = "engine_stop"    # 引擎停止 todo

    # 数据源事件
    DATA_SOURCE_INIT = "data_source_init"                    # 数据源初始化 todo
    DATA_SOURCE_CONNECT = "data_source_connect"              # 数据源连接 todo
    DATA_SOURCE_DISCONNECT = "data_source_disconnect"        # 数据源断开 todo
    DATA_SOURCE_SUBSCRIBE = "data_source_subscribe"          # 数据源订阅 todo
    DATA_SOURCE_UNSUBSCRIBE = "data_source_unsubscribe"      # 数据源取消订阅 todo
    DATA_SOURCE_RECONNECT = "data_source_reconnect"          # 数据源重连 todo
    DATA_SOURCE_ERROR = "data_source_error"                  # 数据源错误 todo
    DATA_SOURCE_STATUS_CHANGE = "data_source_status_change"  # 数据源状态改变 todo
    DATA_SOURCE_DATA = "data_source_data"                    # 数据源数据 todo

    # 数据事件
    DATA_RECEIVED = "data_received"      # 接收到数据
    DATA_PROCESSING = "data_processing"  # 数据处理
    DATA_PROCESSED = "data_processed"    # 数据处理完成

    # 策略事件
    STRATEGY_INIT = "strategy_init"      # 策略初始化 todo
    STRATEGY_START = "strategy_start"    # 策略启动 todo
    STRATEGY_STOP = "strategy_stop"      # 策略停止 todo
    STRATEGY_SIGNAL = "strategy_signal"  # 策略产生信号

    # 风控事件
    RISK_CHECK_START = "risk_check_start"    # 风控检查开始 todo
    RISK_CHECK_PASS = "risk_check_pass"      # 风控检查通过 todo
    RISK_CHECK_REJECT = "risk_check_reject"  # 风控检查拒绝 todo

    # 仓位管理事件
    POSITION_INIT = "position_init"      # 仓位管理初始化 todo
    POSITION_OPEN = "position_open"      # 开仓 todo
    POSITION_CLOSE = "position_close"    # 平仓 todo
    POSITION_UPDATE = "position_update"  # 仓位更新 todo

    # 交易执行事件
    ORDER_CREATED = "order_created"      # 订单创建 todo
    ORDER_SENT = "order_sent"            # 订单发送 todo
    ORDER_FILLED = "order_filled"        # 订单成交 todo
    ORDER_CANCELED = "order_canceled"    # 订单取消 todo
    ORDER_REJECTED = "order_rejected"    # 订单拒绝 todo

    # 资金管理事件
    FUND_ALLOCATED = "fund_allocated"  # 资金分配 todo
    FUND_RECLAIMED = "fund_reclaimed"  # 资金回收 todo

@dataclass
class EventSource:
    """
    事件源
    """
    instance_id: str
    name: str
    cls: str
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
        assert source is not None, "Event source is None"
        self.source = source

    def __str__(self):
        return f"Event(type={self.event_type.value}, source={self.source}, data={self.data})"
