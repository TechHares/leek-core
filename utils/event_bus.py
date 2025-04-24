#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
from typing import Dict, Set, Callable, List

from .logging import get_logger

logger = get_logger(__name__)
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
    ENGINE_INIT = "engine_init"  # 引擎初始化
    ENGINE_START = "engine_start"  # 引擎启动
    ENGINE_STOP = "engine_stop"  # 引擎停止

    # 数据源事件
    DATA_SOURCE_INIT = "data_source_init"  # 数据源初始化 todo
    DATA_SOURCE_CONNECT = "data_source_connect"  # 数据源连接 todo
    DATA_SOURCE_DISCONNECT = "data_source_disconnect"  # 数据源断开 todo
    DATA_SOURCE_SUBSCRIBE = "data_source_subscribe"  # 数据源订阅 todo
    DATA_SOURCE_UNSUBSCRIBE = "data_source_unsubscribe"  # 数据源取消订阅 todo
    DATA_SOURCE_RECONNECT = "data_source_reconnect"  # 数据源重连 todo
    DATA_SOURCE_ERROR = "data_source_error"  # 数据源错误 todo
    DATA_SOURCE_STATUS_CHANGE = "data_source_status_change"  # 数据源状态改变 todo
    DATA_SOURCE_DATA = "data_source_data"  # 数据源数据 todo

    # 数据事件
    DATA_RECEIVED = "data_received"  # 接收到数据
    DATA_PROCESSING = "data_processing"  # 数据处理
    DATA_PROCESSED = "data_processed"  # 数据处理完成

    # 策略事件
    STRATEGY_INIT = "strategy_init"  # 策略初始化 todo
    STRATEGY_START = "strategy_start"  # 策略启动 todo
    STRATEGY_STOP = "strategy_stop"  # 策略停止 todo
    STRATEGY_SIGNAL = "strategy_signal"  # 策略产生信号

    # 风控插件事件
    RISK_PLUGIN_INIT = "risk_plugin_init"  # 插件初始化
    RISK_PLUGIN_START = "risk_plugin_start"  # 插件绑定仓位启动
    RISK_PLUGIN_STOP = "risk_plugin_stop"  # 插件绑定停止

    # 风控事件
    RISK_MANAGER_INIT = "risk_manager_init"  # 风控检查开始 todo
    RISK_MANAGER_START = "risk_manager_start"  # 风控检查开始 todo
    RISK_MANAGER_STOP = "risk_manager_stop"  # 风控检查开始 todo
    RISK_MANAGER_UPDATE = "risk_manager_update"  # 风控检查开始 todo

    RISK_CHECK_START = "risk_check_start"  # 风控检查开始 todo
    RISK_CHECK_PASS = "risk_check_pass"  # 风控检查通过 todo
    RISK_CHECK_REJECT = "risk_check_reject"  # 风控检查拒绝 todo

    # 仓位管理事件
    POSITION_INIT = "position_init"  # 仓位管理初始化 todo
    POSITION_OPEN = "position_open"  # 开仓 todo
    POSITION_CLOSE = "position_close"  # 平仓 todo
    POSITION_UPDATE = "position_update"  # 仓位更新 todo

    # 交易执行事件
    ORDER_CREATED = "order_created"  # 订单创建 todo
    ORDER_SENT = "order_sent"  # 订单发送 todo
    ORDER_FILLED = "order_filled"  # 订单成交 todo
    ORDER_CANCELED = "order_canceled"  # 订单取消 todo
    ORDER_REJECTED = "order_rejected"  # 订单拒绝 todo

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


class EventBus:
    """简化版事件总线，负责事件分发和处理"""

    def __init__(self):
        """初始化事件总线"""
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._all_event_subscribers: Set[Callable] = set()

    def subscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        订阅事件。如果 event_type 为空，则订阅所有事件。

        参数:
            event_type: 事件类型（可为 None 或空字符串，表示订阅全部事件）
            callback: 回调函数

        返回:
            是否成功订阅
        """
        if not event_type:
            self._all_event_subscribers.add(callback)
            return True
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        return True

    def unsubscribe_event(self, event_type: EventType, callback: Callable) -> bool:
        """
        取消订阅事件

        参数:
            event_type: 事件类型
            callback: 回调函数

        返回:
            是否成功取消订阅
        """
        if event_type is None:
            self._all_event_subscribers.discard(callback)

        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            return True
        return False

    def publish_event(self, event: 'Event') -> bool:
        """
        发布事件，支持分发给所有订阅者（包括订阅所有事件的回调）
        """
        dispatched = False
        # 先分发给具体类型订阅者
        if event.event_type in self._subscribers:
            for cb in list(self._subscribers[event.event_type]):
                try:
                    cb(event)
                    dispatched = True
                except Exception as e:
                    logger.error(f"事件处理异常: {e}")
        # 再分发给全局订阅者
        for cb in self._all_event_subscribers:
            try:
                cb(event)
                dispatched = True
            except Exception as e:
                logger.error(f"全局事件处理异常: {e}")
        return dispatched
