#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易引擎核心实现
"""

from leek_core.base import LeekContext
from leek_core.event import EventBus
from leek_core.models import LeekComponentConfig

from leek_core.utils import get_logger

logger = get_logger(__name__)


class Engine(LeekContext):
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig):
        super().__init__(event_bus, config)
        self.running = False
    # def add_strategy(self):
    #     ...
    #
    # def add_risk_plugin(self):
    #    ...
    #
    # def add_position_policy(self):
    #   ...
    #
    # def add_strategy_policy(self):
    #     ...
    #
    # def add_data_source(self, data_source_config: LeekComponentConfig[DataSource, Dict[str, Any]]):
    #    ...
    #
    # def add_executor(self):
    #    ...
    #
    # def remove_strategy(self):
    #   ...
    #
    # def remove_risk_plugin(self):
    #   ...
    #
    # def remove_position_policy(self):
    #     ...
    #
    # def remove_strategy_policy(self):
    #    ...
    #
    # def remove_data_source(self, instance_id: str):
    #    ...
    #
    # def remove_executor(self):
    #    ...


#
#
#
#
# class Engine(Component):
#     """
#     交易引擎
#     """
#
#     def __init__(self, instance_id: str, name: str, event_bus: EventBus,
#                  data_manager: DataManager, strategy_manager: StrategyManager,
#                  position_manager: PositionManager, risk_manager: RiskManager,
#                  executor_manager: ExecutorManager):
#         """
#         初始化交易引擎
#
#         参数:
#             instance_id: 引擎实例唯一标识
#             name: 引擎名称
#             event_bus: 事件总线（用于组件间事件通信）
#             data_manager: 数据管理器（负责行情、数据源管理）
#             strategy_manager: 策略管理器（负责策略加载、调度与管理）
#             position_manager: 仓位管理器（负责资金与仓位管理）
#             risk_manager: 风控管理器（负责风险控制与风控规则）
#             executor_manager: 执行器管理器（负责订单执行与撮合）
#
#         说明：
#             所有核心组件均通过依赖注入方式传入，便于扩展和测试。
#             初始化时会创建独立事件循环，并注册引擎默认事件监听器。
#         """
#         super().__init__(instance_id, name)
#         self.event_bus = event_bus
#         self.data_manager = data_manager
#         self.strategy_manager = strategy_manager
#         self.position_manager = position_manager
#         self.risk_manager = risk_manager
#         self.executor_manager = executor_manager
#
#         # 创建事件循环
#         self._loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self._loop)
#         # 注册默认事件监听器
#         self.running = False
#         self._register_default_listeners()
#
#     def _register_default_listeners(self):
#         """注册默认事件监听器"""
#         # 引擎生命周期事件
#         # self.event_bus.subscribe_event(EventType.ENGINE_INIT, self._on_engine_init)
#         # self.event_bus.subscribe_event(EventType.ENGINE_START, self._on_engine_start)
#         # self.event_bus.subscribe_event(EventType.ENGINE_STOP, self._on_engine_stop)
#         #
#         # 数据事件
#         self.event_bus.subscribe_event(EventType.DATA_PROCESSED, self._on_data_received)
#         #
#         # 策略事件
#         self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self._on_strategy_signal)
#         # 订单创建
#         self.event_bus.subscribe_event(EventType.ORDER_CREATED, self._on_position_order)
#
#     def _on_data_received(self, event: Event):
#         """
#         接收处理完成数据
#
#         参数:
#             event: 数据
#         """
#         self.risk_manager.risk_process(event.data)
#         self.strategy_manager.on_data_event(event)
#
#     def _on_strategy_signal(self, event: Event):
#         """策略信号事件处理"""
#         # 将信号发送给风控
#         signal = event.data
#         order = self.position_manager.process_signal(signal)
#         self.event_bus.publish_event(Event(
#             event_type=EventType.ORDER_CREATED,
#             data=order,
#             source=EventSource(
#                 instance_id=self.position_manager.instance_id,
#                 name=self.position_manager.name,
#                 cls=self.position_manager.__class__.__name__
#             )
#         ))
#
#     def _on_position_order(self, event: Event):
#         """仓位订单信号事件处理"""
#         # 将信号发送给执行模块处理
#         order = event.data
#         self.executor_manager.handle_order(order)
#
#     def _on_order_update(self, event: Event):
#         """订单更新事件处理"""
#         order = event.data
#         self.position_manager.update_order(order)
#
#     def add_risk_plugin(self, plugin: InstanceInitConfig):
#         """
#         添加风控插件
#
#         参数:
#             plugin: 插件初始化配置
#         """
#         self.risk_manager.add_plugin(plugin)
#
#     def add_data_source(self, data_source: DataSource):
#         """
#         添加数据源
#
#         参数:
#             data_source: 数据源实例
#         """
#         # 设置数据回调
#         data_source.set_callback(self._on_data_callback)
#
#         # 添加到数据管理器
#         self.data_manager.add_data_source(data_source)
#         logger.info(f"已添加数据源: {data_source.name}")
#
#     def add_strategy(self, strategy: Strategy, context: StrategyContext = None):
#         """
#         添加策略
#
#         参数:
#             strategy: 策略实例
#             context: 策略上下文
#         """
#         # 确保策略有事件总线
#         if not hasattr(strategy, 'event_bus'):
#             strategy.event_bus = self.event_bus
#
#         # 确保策略有名称
#         name = getattr(strategy, 'name', strategy.__class__.__name__)
#         strategy.name = name
#
#         # 设置上下文
#         if context:
#             strategy.context = context
#             context.event_bus = self.event_bus
#
#         self.strategies[name] = strategy
#         logger.info(f"已添加策略: {name}")
#
#     def add_risk_policy(self, policy: Policy):
#         """
#         添加风控
#
#         参数:
#             controller: 风控实例
#             name: 风控名称
#         """
#         self.position_manager.add_policy(policy)
#         logger.info(f"已添加风控: {policy.name} @ {policy.instance_id}")
#
#     def add_position_manager(self, manager: Any, name: str = None):
#         """
#         添加仓位管理器
#
#         参数:
#             manager: 仓位管理器实例
#             name: 仓位管理器名称
#         """
#         if name is None:
#             name = manager.__class__.__name__
#
#         # 确保仓位管理器有事件总线
#         if not hasattr(manager, 'event_bus'):
#             manager.event_bus = self.event_bus
#
#         # 确保仓位管理器有名称
#         manager.name = name
#
#         self.position_managers[name] = manager
#         logger.info(f"已添加仓位管理器: {name}")
#
#     def add_executor(self, executor: Executor, run_data: Dict[str, Any]=None):
#         """
#         添加交易执行器
#
#         参数:
#             trader: 交易执行器实例
#             name: 交易执行器名称
#         """
#         self.executor_manager.add_executor(executor, run_data)
#         logger.info(f"已添加交易执行器: {executor.name} {executor.instance_id}")
#
#     def subscribe_event(self, event_type: EventType, callback: Callable):
#         """
#         订阅事件
#
#         参数:
#             event_type: 事件类型
#             callback: 回调函数
#         """
#         self.event_bus.subscribe_event(event_type, callback)
#
#     def load_state(self, state: Dict[str, Any]=None):
#         """
#         加载组件状态
#         """
#         self.initialize()
#         self.event_bus.publish_event(Event(
#             event_type=EventType.ENGINE_INIT,
#             data=state,
#             source=EventSource(
#                 instance_id=self.instance_id,
#                 name=self.name,
#                 cls=self.__class__.__name__
#             )
#         ))
#
#     def initialize(self):
#         """初始化引擎"""
#         ...
#
#     def on_start(self):
#         """启动引擎"""
#         if self.running:
#             logger.warning("引擎已经启动")
#             return
#         logger.info(f"启动引擎: {self.name}")
#         self.running = True
#         self.risk_manager.on_start()
#         self.position_manager.on_start()
#         self.executor_manager.on_start()
#         self.strategy_manager.on_start()
#         self.data_manager.on_start()
#         self.event_bus.publish_event(Event(
#             event_type=EventType.ENGINE_START,
#             data={},
#             source=EventSource(
#                 instance_id=self.instance_id,
#                 name=self.name,
#                 cls=self.__class__.__name__
#             )
#         ))
#         # 启动事件循环
#         self._loop.run_until_complete(self._main_loop())
#
#     async def _main_loop(self):
#         """引擎主循环"""
#         logger.info("引擎主循环开始")
#
#         try:
#             while self.running:
#                 # 主循环逻辑
#                 await asyncio.sleep(0.1)
#         except asyncio.CancelledError:
#             logger.info("引擎主循环被取消")
#         except Exception as e:
#             logger.error(f"引擎主循环异常: {e}", exc_info=True)
#             await self.on_stop()
#
#         logger.info("引擎主循环结束")
#
#     def on_stop(self):
#         """停止引擎"""
#         if not self.running:
#             logger.warning("引擎已经停止")
#             return
#
#         logger.info(f"停止引擎: {self.name}")
#         self.running = False
#
#         self.data_manager.on_stop()
#         self.strategy_manager.on_stop()
#         self.executor_manager.on_stop()
#         self.position_manager.on_stop()
#         self.risk_manager.on_stop()
#         # 发布引擎停止事件
#         self.event_bus.publish_event(Event(
#             event_type=EventType.ENGINE_STOP,
#             data={},
#             source=EventSource(
#                 instance_id=self.instance_id,
#                 name=self.name,
#                 cls=self.__class__.__name__
#             )
#         ))
