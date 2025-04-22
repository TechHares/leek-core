#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易引擎核心实现
"""

import asyncio
import threading
from typing import Dict, Any, Callable

from .risk_manager import RiskManager
from .position_manager import PositionManager
from .data_manager import DataManager
from .executor_manager import ExecutorManager
from .strategy_manager import StrategyManager

from utils import EventBus, EventType, Event, EventSource, get_logger
from strategy import StrategyContext, Strategy
from models import Component, DataType
from data import DataSource


logger = get_logger(__name__)


class Engine(Component):
    """
    异步交易引擎
    """

    def __init__(self, instance_id, name: str = "LeekEngine", event_bus: EventBus = EventBus()):
        """
        初始化交易引擎
        
        参数:
            name: 引擎名称
        """
        super().__init__(instance_id, name)
        self.instance_id = instance_id
        self.name = name

        self.event_bus = event_bus
        self.data_manager = DataManager(event_bus)
        self.strategy_manager = StrategyManager(event_bus)
        self.position_manager = PositionManager(event_bus, None, None, None)
        self.risk_manager = RiskManager(event_bus)
        self.executor_manager = ExecutorManager(event_bus)

        # 创建事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # 注册默认事件监听器
        self._register_default_listeners()

    def _register_default_listeners(self):
        """注册默认事件监听器"""
        # 引擎生命周期事件
        # self.event_bus.subscribe_event(EventType.ENGINE_INIT, self._on_engine_init)
        # self.event_bus.subscribe_event(EventType.ENGINE_START, self._on_engine_start)
        # self.event_bus.subscribe_event(EventType.ENGINE_STOP, self._on_engine_stop)
        #
        # 数据事件
        self.event_bus.subscribe_event(EventType.DATA_PROCESSED, self._on_data_received)
        #
        # 策略事件
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self._on_strategy_signal)
        # 订单创建
        self.event_bus.subscribe_event(EventType.ORDER_CREATED, self._on_position_order)

    def _on_data_received(self, event: Event):
        """
        接收处理完成数据

        参数:
            event: 数据
        """
        self.risk_manager.risk_process(event.data)
        self.strategy_manager.on_data_event(event)

    def _on_strategy_signal(self, event: Event):
        """策略信号事件处理"""
        # 将信号发送给风控
        signal = event.data
        order = self.position_manager.process_signal(signal)
        self.event_bus.publish_event(Event(
            event_type=EventType.ORDER_CREATED,
            data=order,
            source=EventSource(
                instance_id=self.position_manager.instance_id,
                name=self.position_manager.name,
                cls=self.position_manager.__class__.__name__
            )
        ))

    def _on_position_order(self, event: Event):
        """仓位订单信号事件处理"""
        # 将信号发送给执行模块处理
        order = event.data
        self.executor_manager.handle_order(order)

    def _on_order_update(self, event: Event):
        """订单更新事件处理"""
        order = event.data
        self.position_manager.update_order(order)

    def add_data_source(self, data_source: DataSource):
        """
        添加数据源
        
        参数:
            data_source: 数据源实例
        """
        # 设置数据回调
        data_source.set_callback(self._on_data_callback)

        # 添加到数据管理器
        self.data_manager.add_data_source(data_source)
        logger.info(f"已添加数据源: {data_source.name}")

    def add_strategy(self, strategy: Strategy, context: StrategyContext = None):
        """
        添加策略
        
        参数:
            strategy: 策略实例
            context: 策略上下文
        """
        # 确保策略有事件总线
        if not hasattr(strategy, 'event_bus'):
            strategy.event_bus = self.event_bus

        # 确保策略有名称
        name = getattr(strategy, 'name', strategy.__class__.__name__)
        strategy.name = name

        # 设置上下文
        if context:
            strategy.context = context
            context.event_bus = self.event_bus

        self.strategies[name] = strategy
        logger.info(f"已添加策略: {name}")

    def add_risk_controller(self, controller: Any, name: str = None):
        """
        添加风控
        
        参数:
            controller: 风控实例
            name: 风控名称
        """
        if name is None:
            name = controller.__class__.__name__

        # 确保风控有事件总线
        if not hasattr(controller, 'event_bus'):
            controller.event_bus = self.event_bus

        # 确保风控有名称
        controller.name = name

        self.risk_controllers[name] = controller
        logger.info(f"已添加风控: {name}")

    def add_position_manager(self, manager: Any, name: str = None):
        """
        添加仓位管理器
        
        参数:
            manager: 仓位管理器实例
            name: 仓位管理器名称
        """
        if name is None:
            name = manager.__class__.__name__

        # 确保仓位管理器有事件总线
        if not hasattr(manager, 'event_bus'):
            manager.event_bus = self.event_bus

        # 确保仓位管理器有名称
        manager.name = name

        self.position_managers[name] = manager
        logger.info(f"已添加仓位管理器: {name}")

    def add_trader(self, trader: Any, name: str = None):
        """
        添加交易执行器
        
        参数:
            trader: 交易执行器实例
            name: 交易执行器名称
        """
        if name is None:
            name = trader.__class__.__name__

        # 确保交易执行器有事件总线
        if not hasattr(trader, 'event_bus'):
            trader.event_bus = self.event_bus

        # 确保交易执行器有名称
        trader.name = name

        self.traders[name] = trader
        logger.info(f"已添加交易执行器: {name}")

    def subscribe_event(self, event_type: EventType, callback: Callable):
        """
        订阅事件
        
        参数:
            event_type: 事件类型
            callback: 回调函数
        """
        self.event_bus.subscribe_event(event_type, callback)

    def add_interceptor(self, event_type: EventType, interceptor: Callable):
        """
        添加事件拦截器
        
        参数:
            event_type: 事件类型
            interceptor: 拦截器函数
        """
        self.event_bus.add_interceptor(event_type, interceptor)

    async def initialize(self):
        """初始化引擎"""
        logger.info(f"初始化引擎: {self.name}")

        # 初始化数据管理器
        self.data_manager.in_service = True

        # 初始化所有策略
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'initialize') and asyncio.iscoroutinefunction(strategy.initialize):
                await strategy.initialize()
            elif hasattr(strategy, 'initialize'):
                strategy.initialize()

        # 初始化所有风控
        for name, controller in self.risk_controllers.items():
            if hasattr(controller, 'initialize') and asyncio.iscoroutinefunction(controller.initialize):
                await controller.initialize()
            elif hasattr(controller, 'initialize'):
                controller.initialize()

        # 初始化所有仓位管理器
        for name, manager in self.position_managers.items():
            if hasattr(manager, 'initialize') and asyncio.iscoroutinefunction(manager.initialize):
                await manager.initialize()
            elif hasattr(manager, 'initialize'):
                manager.initialize()

        # 初始化所有交易执行器
        for name, trader in self.traders.items():
            if hasattr(trader, 'initialize') and asyncio.iscoroutinefunction(trader.initialize):
                await trader.initialize()
            elif hasattr(trader, 'initialize'):
                trader.initialize()

        # 发布引擎初始化事件
        await self.publish_event(EventType.ENGINE_INIT)

    async def start(self):
        """启动引擎"""
        if self.running:
            logger.warning("引擎已经在运行中")
            return

        logger.info(f"启动引擎: {self.name}")
        self.running = True

        # 连接所有数据源
        self.data_manager._connect_all()

        # 启动所有策略
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'start') and asyncio.iscoroutinefunction(strategy.start):
                await strategy.start()
            elif hasattr(strategy, 'start'):
                strategy.start()

        # 启动所有风控
        for name, controller in self.risk_controllers.items():
            if hasattr(controller, 'start') and asyncio.iscoroutinefunction(controller.start):
                await controller.start()
            elif hasattr(controller, 'start'):
                controller.start()

        # 启动所有仓位管理器
        for name, manager in self.position_managers.items():
            if hasattr(manager, 'start') and asyncio.iscoroutinefunction(manager.start):
                await manager.start()
            elif hasattr(manager, 'start'):
                manager.start()

        # 启动所有交易执行器
        for name, trader in self.traders.items():
            if hasattr(trader, 'start') and asyncio.iscoroutinefunction(trader.start):
                await trader.start()
            elif hasattr(trader, 'start'):
                trader.start()

        # 发布引擎启动事件
        await self.publish_event(EventType.ENGINE_START)

        # 启动主循环
        self._main_task = asyncio.create_task(self._main_loop())

    async def _main_loop(self):
        """引擎主循环"""
        logger.info("引擎主循环开始")

        try:
            while self.running:
                # 主循环逻辑
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("引擎主循环被取消")
        except Exception as e:
            logger.error(f"引擎主循环异常: {e}", exc_info=True)
            await self.stop()

        logger.info("引擎主循环结束")

    async def stop(self):
        """停止引擎"""
        if not self.running:
            logger.warning("引擎已经停止")
            return

        logger.info(f"停止引擎: {self.name}")
        self.running = False

        # 取消主循环任务
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        # 发布引擎停止事件
        await self.publish_event(EventType.ENGINE_STOP)

        # 停止所有交易执行器
        for name, trader in self.traders.items():
            if hasattr(trader, 'stop') and asyncio.iscoroutinefunction(trader.stop):
                await trader.stop()
            elif hasattr(trader, 'stop'):
                trader.stop()

        # 停止所有仓位管理器
        for name, manager in self.position_managers.items():
            if hasattr(manager, 'stop') and asyncio.iscoroutinefunction(manager.stop):
                await manager.stop()
            elif hasattr(manager, 'stop'):
                manager.stop()

        # 停止所有风控
        for name, controller in self.risk_controllers.items():
            if hasattr(controller, 'stop') and asyncio.iscoroutinefunction(controller.stop):
                await controller.stop()
            elif hasattr(controller, 'stop'):
                controller.stop()

        # 停止所有策略
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'stop') and asyncio.iscoroutinefunction(strategy.stop):
                await strategy.stop()
            elif hasattr(strategy, 'stop'):
                strategy.stop()

        # 断开所有数据源
        self.data_manager.shutdown()

    def run(self):
        """
        在单独的线程中运行引擎
        
        该方法会启动一个新的线程来运行事件循环，适用于在非异步环境中使用引擎
        """

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run())

        thread = threading.Thread(target=run_loop)
        thread.daemon = True
        thread.start()

        return thread

    async def _run(self):
        """实际的运行方法"""
        try:
            await self.initialize()
            await self.start()

            # 保持运行直到被停止
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("接收到键盘中断，停止引擎")
        except Exception as e:
            logger.error(f"引擎运行异常: {e}", exc_info=True)
        finally:
            await self.stop()

    def stop_sync(self):
        """同步停止引擎"""
        asyncio.run_coroutine_threadsafe(self.stop(), self._loop).result()
