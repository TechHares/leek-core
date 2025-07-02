#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import time
from typing import List

from leek_core.data import DataSourceContext
from leek_core.engine import Engine
from leek_core.event import EventBus
from leek_core.executor import ExecutorContext
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.models import LeekComponentConfig, BacktestEngineConfig, Signal, Order
from leek_core.strategy import StrategyContext
from leek_core.utils import get_logger, run_func_timeout

logger = get_logger(__name__)


@dataclass
class DailyData:
    date: int
    position_value: Decimal
    signals: List[Signal]
    orders: List[Order]


class BacktestEngine(Engine):
    """
    回测指标:
    策略收益
    策略年化收益
    超额收益
    基准收益
    阿尔法
    贝塔
    夏普比率
    胜率
    盈亏比
    最大回撤
    索提诺比率
    日均超额收益
    超额收益最大回撤
    超额收益夏普比率
    日胜率
    盈利次数
    亏损次数
    信息比率
    策略波动率
    基准波动率
    最大回撤区间
    """
    def __init__(self, config: LeekComponentConfig[None, BacktestEngineConfig] = None):
        super().__init__()
        self.event_bus = EventBus()
        self.config = config
        self.strategy_manager: StrategyManager = StrategyManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-策略管理",
            cls=StrategyContext,
            config=None
        ))

        self.position_manager: PositionManager = PositionManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-仓位管理",
            cls=None,
            config=config.config.position_config
        ))
        self.executor_manager: ExecutorManager = ExecutorManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-执行器管理",
            cls=ExecutorContext,
            config=None
        ))

        self.daily_data: List[DailyData] = []


    def on_start(self):
        ...




