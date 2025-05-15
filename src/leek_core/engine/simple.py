#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from leek_core.data import DataSourceContext
from leek_core.engine import Engine
from leek_core.event import EventBus
from leek_core.executor import ExecutorContext
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.models import LeekComponentConfig, SimpleEngineConfig
from leek_core.strategy import StrategyContext
from leek_core.utils import get_logger, run_func_timeout

logger = get_logger(__name__)


class SimpleEngine(Engine):
    def __init__(self, event_bus: EventBus=EventBus(), config: LeekComponentConfig[None, SimpleEngineConfig]=None):
        super().__init__(event_bus, config)

        self.data_source_manager: DataManager = DataManager(
            event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-数据源管理",
            cls=DataSourceContext,
            config=None
        ))
        self.strategy_manager: StrategyManager = StrategyManager(
            event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-策略管理",
            cls=StrategyContext,
            config=None
        ))

        self.position_manager: PositionManager = PositionManager(
            event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-仓位管理",
            cls=None,
            config=config.config.position_config
        ))
        self.executor_manager: ExecutorManager = ExecutorManager(
            event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-执行器管理",
            cls=ExecutorContext,
            config=None
        ))

        self.running_manager=[]
        self.timeout = self.config.config.timeout if self.config.config.timeout else 10

    def on_start(self):
        self.running = True
        logger.info("引擎开始启动")
        for manager in [self.executor_manager, self.position_manager, self.data_source_manager, self.strategy_manager]:
            if self._run_func_timeout(manager.on_start, [], {}):
                self.running_manager.append(manager)
                logger.info(f"{manager.name}模块启动完成")
            else:
                logger.error(f"{manager.name}模块启动超时")
                self.on_stop()
                return
        self.running_manager.reverse()
        if self.config.config.executor_configs:
            for executor in self.config.config.executor_configs:
                logger.info(f"添加执行器: {executor.name}@{executor.instance_id}")
                self.executor_manager.add(executor)
        if self.config.config.data_sources:
            for data_source in self.config.config.data_sources.config:
                logger.info(f"添加数据源: {data_source.name}@{data_source.instance_id}")
                self.data_source_manager.add(data_source)
        if self.config.config.strategy_configs:
            for strategy in self.config.config.strategy_configs:
                logger.info(f"添加策略: {strategy.name}@{strategy.instance_id}")
                self.strategy_manager.add(strategy)
        logger.info("引擎启动完成")
        while self.running:
            time.sleep(0.1)
        logger.warning("引擎停止")

    def on_stop(self):
        logger.info("开始停止引擎")
        for manager in self.running_manager:
            if self._run_func_timeout(manager.on_stop, [], {}):
                logger.info(f"{manager.name}模块停止完成")
            else:
                logger.error(f"{manager.name}模块停止超时")
        self.running = False

    def _run_func_timeout(self, func, args, kwargs):
        """
        执行函数并设置超时
        """
        return run_func_timeout(func, args, kwargs, self.timeout)



