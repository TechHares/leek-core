#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Callable, Dict
from models import Component, Position, Order, InstanceInitConfig, Data
from risk.base import RiskPlugin
from utils import EventBus


class RiskManager(Component):
    def __init__(self, event_bus: EventBus, instance_id: str=None, name: str=None, risk_plugins: List[InstanceInitConfig]=None):
        super().__init__(instance_id=instance_id, name=name)
        self.event_bus: EventBus = event_bus
        self.risk_plugins: List[InstanceInitConfig] = risk_plugins if risk_plugins else []  # 风控插件列表
        self.positions: Dict[str, Position] = {}  # 仓位字典，键为仓位ID
        self.position_plugins: Dict[Position, List[RiskPlugin]] = {}       # 键为仓位ID

    def update_plugin(self, risk_plugins: List[InstanceInitConfig]=None):
        """
        添加风控插件。
        :param risk_plugins: RiskPlugin 实例
        """
        self.risk_plugins = risk_plugins if risk_plugins else []

    def risk_process(self, data) -> list[Position]:
        """
        主动风控检查：轮询所有插件，若有插件触发风控则执行平仓。
        :param data: 市场数据、行情等
        :return: 是否触发风控（True=有插件触发）
        """
        closed_positions = []
        for position, plugins in self.position_plugins.items():
            # todo 判断仓位需不需要调用插件
            if any(plugin.trigger(position, data) for plugin in plugins):
                closed_positions.append(position)
        return closed_positions
