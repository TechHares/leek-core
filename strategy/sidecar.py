#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略的伴生组件，管理策略实例的生命周期。
"""

from abc import ABC
from decimal import Decimal
from typing import Dict, Any

from models import *
from .cta import STAStrategy
from sub_strategy import EnterStrategy, ExitStrategy
from utils import get_logger, EventBus

logger = get_logger(__name__)

class CTAStrategySidecar:
    """
    管理择时策略实例的生命周期和状态。
    
    职责:
    1. 管理策略的状态
    2. 管理进出场策略
    """
    
    def __init__(self, event_bus: EventBus, strategy: STAStrategy, enter_strategy: EnterStrategy, exit_strategy: ExitStrategy):
        """
        初始化策略上下文
        
        参数:
            strategy: 策略实例
            enter_strategy: 进场策略
            exit_strategy: 出场策略
        """
        super().__init__()
        self.event_bus = event_bus
        self.strategy = strategy

        # 进出场策略
        self.enter_strategy = enter_strategy
        self.exit_strategy = exit_strategy
        
        # 策略状态
        self.state = StrategyInstanceState.CREATED
        self.position_rate = Decimal("0")  # 仓位比例，0-1之间的小数
        self.current_position_rate = Decimal("0") # 进/出场状态中需要变化的比例
        self.current_position_side = None

    def on_data(self, data: Data = None) -> (PositionSide, Decimal):
        """
        处理数据，更新进出场策略状态
        
        参数:
            data: 接收到的数据
            data_type: 数据类型
        """
        if self.state in [StrategyInstanceState.STOPPED, StrategyInstanceState.CREATED]:
            return
        # 更新计算信息
        self.strategy.on_data(data)
        self.enter_strategy.on_data(data)
        self.exit_strategy.on_data(data)

        if self.state == StrategyInstanceState.READY: # 无仓位
            return self.no_position_handler()
        elif self.state == StrategyInstanceState.ENTERING:
            return self.entering_handler()
        elif self.state == StrategyInstanceState.HOLDING:
            return self.holding_handler()
        elif self.state == StrategyInstanceState.EXITING:
            return self.exiting_handler()
        elif self.state == StrategyInstanceState.STOPPING:
            return self.stopping_handler()

    def no_position_handler(self):
        position_side = self.strategy.should_open()
        if position_side:
            self.state = StrategyInstanceState.ENTERING
            self.current_position_side = position_side
            self.current_position_rate = self.strategy.get_position_change_rate()
            return self.entering_handler()

    def entering_handler(self):
        rt = self.enter_strategy.position_rate(self.current_position_side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate += position_change
            if self.enter_strategy.is_finished():
                self.state = StrategyInstanceState.HOLDING
                self.enter_strategy.reset()
            return self.current_position_side, position_change

    def holding_handler(self):
        if self.strategy.should_close(self.current_position_side):
            self.state = StrategyInstanceState.EXITING
            self.current_position_rate = self.strategy.get_position_change_rate()
            return self.exiting_handler()

        # 策略要求重复开仓
        if self.strategy.open_just_no_pos is False:
            return self.no_position_handler()

    def exiting_handler(self):
        rt = self.exit_strategy.position_rate(self.current_position_side)
        if rt > 0:
            position_change = self.current_position_rate * rt
            self.position_rate += position_change
            if self.enter_strategy.is_finished():
                self.state = StrategyInstanceState.HOLDING if self.position_rate > 0 else StrategyInstanceState.READY
                self.enter_strategy.reset()
            return self.current_position_side.switch(), position_change

    def stopping_handler(self):
        if self.state == StrategyInstanceState.READY or self.position_rate == 0:
            self.state = StrategyInstanceState.STOPPED
            return

        res = self.exiting_handler()
        if res is not None and self.state == StrategyInstanceState.READY:
            self.state = StrategyInstanceState.STOPPED
        return res

    def get_position_rate(self) -> Decimal:
        """
        获取当前仓位比例
        
        返回:
            当前仓位比例，0-1之间的小数
        """
        return self.current_position_rate
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取策略上下文状态
        
        返回:
            状态字典
        """
        return {
            "strategy": self.strategy.get_state(),
            "state": self.state,
            "position_rate": str(self.position_rate),
            "current_position_rate": str(self.current_position_rate),
            "current_position_side": self.current_position_side,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载策略上下文状态
        
        参数:
            state: 状态字典
        """
        self.strategy.load_state(state.get("strategy", {}))
        self.state = state.get("state", self.state)
        self.position_rate = Decimal(state.get("position_rate", str(self.position_rate)))
        self.current_position_rate = Decimal(state.get("current_position_rate", str(self.current_position_rate)))
        self.current_position_side = state.get("current_position_side", self.current_position_side)

    def start(self):
        """
        启动组件
        """
        self.strategy.initialize()

    def stop(self):
        """
        停止组件
        """
        self.strategy.stop()
