# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import copy
# from typing import List, Callable, Dict
# from models import Component, Position, Order, InstanceInitConfig, Data, create_instance
# from risk.base import RiskPlugin
# from utils import EventBus, Event, EventType, EventSource, get_logger
# logger = get_logger(__name__)
#
#
# class RiskManager(Component):
#     def __init__(self, instance_id: str, name: str, event_bus: EventBus):
#         super().__init__(instance_id=instance_id, name=name)
#         self.event_bus: EventBus = event_bus
#
#         self.risk_plugins: Dict[type[RiskPlugin], InstanceInitConfig] = {}  # 风控插件列表
#         self.positions: Dict[str, Position] = {}  # 仓位字典，键为仓位ID
#         self.position_plugins: Dict[Position, List[RiskPlugin]] = {}       # 键为仓位ID
#
#     def add_plugin(self, risk_plugin: InstanceInitConfig):
#         """
#         添加风控插件。
#         :param risk_plugin: RiskPlugin 实例
#         """
#         assert issubclass(risk_plugin.cls, RiskPlugin)
#         self.risk_plugins[risk_plugin.cls] = risk_plugin
#         # 对所有已有仓位，若未包含该插件，则初始化并on_start
#         for position, plugin_list in self.position_plugins.items():
#             # 检查是否已存在该类型插件
#             if not any(isinstance(pl, risk_plugin.cls) for pl in plugin_list):
#                 instance = self._init_plugin(risk_plugin, position)
#                 self._start_plugin(instance, position)
#                 plugin_list.append(instance)
#
#     def _init_plugin(self, plugin_config: InstanceInitConfig, position: Position):
#         params = copy.deepcopy(plugin_config.config)
#         params["name"] = plugin_config.cls.display_name if plugin_config.cls.display_name else plugin_config.cls.__name__
#         params["instance_id"] = "%s@%s" % (params["name"], position.position_id)
#         instance = create_instance(
#             plugin_config.cls,
#             **plugin_config.config,
#         )
#         self.event_bus.publish_event(Event(
#             event_type=EventType.RISK_PLUGIN_INIT,
#             source=self._event_source(),
#             data={
#                 "instance_id": instance.instance_id,
#                 "name": instance.name,
#                 "position_id": position.position_id
#             }
#         ))
#         return instance
#
#     def _start_plugin(self, plugin: RiskPlugin, position: Position):
#         try:
#             plugin.on_start()
#         except Exception as e:
#             logger.error("Failed to start plugin %s: %s", plugin.name, e)
#             return
#         self.event_bus.publish_event(Event(
#             event_type=EventType.RISK_PLUGIN_START,
#             source=self._event_source(),
#             data={
#                 "instance_id": plugin.instance_id,
#                 "name": plugin.name,
#                 "position_id": position.position_id
#             }
#         ))
#
#     def _stop_plugin(self, plugin: RiskPlugin, position: Position):
#         try:
#             plugin.on_stop()
#         except Exception:
#             ...
#         self.event_bus.publish_event(Event(
#             event_type=EventType.RISK_PLUGIN_STOP,
#             source=self._event_source(),
#             data={
#                 "instance_id": plugin.instance_id,
#                 "name": plugin.name,
#                 "position_id": position.position_id
#             }
#         ))
#
#     def remove_plugin(self, cls: type[RiskPlugin]):
#         """
#         移除风控插件。
#         :param cls: cls 类型
#         """
#         self.risk_plugins.pop(cls, None)
#         # 对所有仓位，移除该类型插件并on_stop
#         for position, plugin_list in self.position_plugins.items():
#             to_remove = [pl for pl in plugin_list if isinstance(pl, cls)]
#             for pl in to_remove:
#                 self._stop_plugin(pl, position)
#                 plugin_list.remove(pl)
#
#     def on_position(self, position: Position):
#         """
#         当有新仓位创建时，初始化风控插件。
#         :param position: 新仓位
#         """
#         self.positions[position.position_id] = position
#         pls = []
#         for plugin_cls, plugin_config in self.risk_plugins.items():
#             instance = self._init_plugin(plugin_config, position)
#             self._start_plugin(instance, position)
#             pls.append(instance)
#         self.position_plugins[position] = pls
#
#     def on_position_close(self, position: Position):
#         """
#         当有仓位关闭时，移除风控插件。
#         :param position: 新仓位
#         """
#         self.positions.pop(position.position_id, None)
#         plugin_list = self.position_plugins.pop(position, None)
#         for pl in plugin_list:
#             self._stop_plugin(pl, position)
#
#     def risk_process(self, data) -> list[Position]:
#         """
#         主动风控检查：轮询所有插件，若有插件触发风控则执行平仓。
#         :param data: 市场数据、行情等
#         :return: 是否触发风控（True=有插件触发）
#         """
#         closed_positions = []
#         for position, plugins in self.position_plugins.items():
#             # todo 判断仓位需不需要调用插件
#             if any(plugin.trigger(position, data) for plugin in plugins):
#                 closed_positions.append(position)
#         return closed_positions
