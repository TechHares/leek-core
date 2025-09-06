#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易引擎核心实现
"""
from abc import ABC, abstractmethod
from decimal import Decimal
import os
import time
from typing import Dict, Any

import psutil

from leek_core.base import LeekComponent, load_class_from_str, create_component
from leek_core.data import DataSource
from leek_core.event import EventBus, EventType, Event
from leek_core.executor import Executor, ExecutorContext
from leek_core.models import LeekComponentConfig, StrategyConfig, PositionConfig, StrategyPositionConfig
from leek_core.strategy import Strategy, StrategyContext
from leek_core.manager import PositionManager, ExecutorManager, StrategyManager, DataManager
from leek_core.data import DataSourceContext
from leek_core.utils import get_logger

logger = get_logger(__name__)


class Engine(LeekComponent, ABC):
    def __init__(self, instance_id: str, name: str, position_config: PositionConfig = None):
        super().__init__()
        self.running = False
        self.instance_id = instance_id
        self.name = name
        self.position_config = position_config
        self._start_time = time.time()
        
        # 引擎组件
        self.event_bus = EventBus()
        self.data_source_manager: DataManager = DataManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-数据源管理",
                cls=DataSourceContext,
                config=None
            ))
        self.strategy_manager: StrategyManager = StrategyManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-策略管理",
                cls=StrategyContext,
                config=None
            ))
        self.position_manager: PositionManager = PositionManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-仓位管理",
                cls=None,
                config=position_config,
                data=position_config.data if position_config else None
            ))
        self.executor_manager: ExecutorManager = ExecutorManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-执行器管理",
                cls=ExecutorContext,
                config=None
            ))
    
    def check_component(self):
        """
        检查组件状态
        """
        return {
            "data_source_manager": self.data_source_manager.check_component(),
            "strategy_manager": self.strategy_manager.check_component(),
            "position_manager": self.position_manager.check_component(),
            "executor_manager": self.executor_manager.check_component(),
        }
    
    def handle_event(self, event: Event):
        if not event.event_type.value.startswith("data_"):
            self._handle_event(event)

    def get_position_state(self):
        """
        获取仓位状态
        """
        return self.position_manager.get_state()

    def get_strategy_state(self):
        """
        获取策略状态
        """
        return {instance_id: strategy.get_state() for instance_id, strategy in self.strategy_manager.components.items()}
    
    def update_strategy_state(self, instance_id: str, state: Dict):
        """
        更新策略
        """
        logger.info(f"修改策略状态: {instance_id} {state}")
        self.strategy_manager.update_state(instance_id, state)

    def clear_strategy_state(self, strategy_id: str, instance_id: str):
        """
        清除策略状态
        """
        self.strategy_manager.clear_state(strategy_id, instance_id)
        return self.get_strategy_state()

    def handle_message(self, msg: Dict):
        """
        处理主进程发来的消息，自动分发到对应方法
        """
        if not isinstance(msg, dict) or "action" not in msg:
            logger.error(f"未知消息: {msg}")
            return
            
        action = msg["action"]
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})
        msg_id = msg.get("msg_id")
        
        # 反序列化参数
        args = [arg for arg in args]
        kwargs = {k: v for k, v in kwargs.items()}
        
        if hasattr(self, action):
            method = getattr(self, action)
            try:
                result = method(*args, **kwargs)
                # 如果是invoke请求（有msg_id），则发送响应
                if msg_id:
                    self.send_msg("response", msg_id=msg_id, result=result)
            except Exception as e:
                logger.error(f"处理命令 {action} 时出错，参数{args} {kwargs}，错误信息: {e}", exc_info=True)
                if msg_id:
                    self.send_msg("response", msg_id=msg_id, error=str(e))
        else:
            logger.error(f"未知action: {action}")
            if msg_id:
                self.send_msg("response", msg_id=msg_id, error=f"未知action: {action}")

    def engine_state(self):
        """
        返回引擎当前状态，包括数据源、策略、仓位、执行器等
        """
        # 获取系统信息
        process = psutil.Process(os.getpid())
        process_id = process.pid

        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0)

        # 获取内存使用情况
        memory = psutil.virtual_memory()
        mem_used = round(memory.used / (1024 ** 3), 1)  # GB
        mem_total = round(memory.total / (1024 ** 3), 1)  # GB
        mem_percent = round((memory.used / memory.total) * 100, 1)

        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_used = round(disk.used / (1024 ** 3), 1)  # GB
        disk_total = round(disk.total / (1024 ** 3), 1)  # GB
        disk_percent = disk.percent
        return {
            "state": {
                "process_id": process_id,
                "data_source_count": len(self.data_source_manager),
                "strategy_count": len(self.strategy_manager),
                "executor_count": len(self.executor_manager),
            },
            "resources": {
                "cpu": {
                    "percent": cpu_percent,
                    "value": f"{cpu_percent}%",
                    "status": "success" if cpu_percent < 60 else "warning" if cpu_percent < 85 else "error"
                },
                "memory": {
                    "percent": mem_percent,
                    "value": f"{mem_used}G/{mem_total}G",
                    "status": "success" if mem_percent < 60 else "warning" if mem_percent < 85 else "error"
                },
                "disk": {
                    "percent": disk_percent,
                    "value": f"{disk_used}G/{disk_total}G",
                    "status": "success" if disk_percent < 60 else "warning" if disk_percent < 85 else "error"
                }
            }
        }
    def format_strategy_config(self, config: Dict):
        return LeekComponentConfig(
            instance_id=str(config.get("id")),
            name=config.get("name", ""),
            cls=load_class_from_str(config.get("class_name")),
            data=config.get("data"),
            config=StrategyConfig(
                data_source_configs=[LeekComponentConfig(
                    instance_id=str(cfg.get("id")),
                    name=cfg.get("name", ""),
                    cls=load_class_from_str(cfg.get("class_name")),
                    config=cfg.get("config", {})
                ) for cfg in config.get("data_source_config", [])],
                info_fabricator_configs=[LeekComponentConfig(
                    instance_id=str(config.get("id")),
                    name=config.get("name", ""),
                    cls=load_class_from_str(cfg.get("class_name")),
                    config=cfg.get("config", {})
                ) for cfg in config.get("info_fabricator_configs", [])],
                strategy_config=config.get("params", {}),
                strategy_position_config=StrategyPositionConfig(**config.get("position_config")) if config.get("position_config") else None,
                enter_strategy_cls=load_class_from_str(config.get("enter_strategy_class_name")),
                enter_strategy_config=config.get("enter_strategy_config", {}),
                exit_strategy_cls=load_class_from_str(config.get("exit_strategy_class_name")),
                exit_strategy_config=config.get("exit_strategy_config", {}),
                risk_policies=[LeekComponentConfig(
                    instance_id=str(config.get("id")),
                    name=config.get("name", ""),
                    cls=load_class_from_str(cfg.get("class_name")),
                    config=cfg.get("config", {})
                ) for cfg in config.get("risk_policies", [])],
            )
        )

    # 下面实现所有Engine抽象方法（这里只做简单打印/存储，实际可接入管理器）
    def add_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        logger.info(f"添加策略: {config}")
        if isinstance(config, dict):
            config = self.format_strategy_config(config)
        self.strategy_manager.add(config)

    def update_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        logger.info(f"更新策略: {config}")
        if isinstance(config, dict):
            config = self.format_strategy_config(config)
        self.strategy_manager.update(config)

    def remove_strategy(self, instance_id):
        logger.info(f"移除策略: {instance_id}")
        self.strategy_manager.remove(instance_id)

    def format_component_config(self, config: Dict):
        return LeekComponentConfig(
            instance_id=str(config.get("id")),
            name=config.get("name", ""),
            cls=load_class_from_str(config.get("class_name")),
            config=config.get("params", {}),
            data=config.get("data")
        )

    def add_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        logger.info(f"添加执行器: {config}")
        if isinstance(config, dict):
            config = self.format_component_config(config)
        self.executor_manager.add(config)

    def update_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        logger.info(f"更新执行器: {config}")
        if isinstance(config, dict):
            config = self.format_component_config(config)
        self.executor_manager.update(config)

    def remove_executor(self, instance_id):
        logger.info(f"移除执行器: {instance_id}")
        self.executor_manager.remove(instance_id)

    def add_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        logger.info(f"添加数据源: {config}")
        if isinstance(config, dict):
            config = self.format_component_config(config)
        self.data_source_manager.add(config)

    def update_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        logger.info(f"更新数据源: {config}")
        if isinstance(config, dict):
            config = self.format_component_config(config)
        self.data_source_manager.update(config)

    def remove_data_source(self, instance_id):
        logger.info(f"移除数据源: {instance_id}")
        self.data_source_manager.remove(instance_id)

    def update_position_config(self, position_config: PositionConfig, data: Dict=None) -> None:
        logger.info(f"更新仓位配置: {position_config}")
        if isinstance(position_config, dict):
            # 为 PositionConfig 提供默认值
            position_config.setdefault('init_amount', Decimal('100000'))
            position_config.setdefault('max_strategy_amount', Decimal('50000'))
            position_config.setdefault('max_strategy_ratio', Decimal('0.5'))
            position_config.setdefault('max_symbol_amount', Decimal('25000'))
            position_config.setdefault('max_symbol_ratio', Decimal('0.25'))
            position_config.setdefault('max_amount', Decimal('10000'))
            position_config.setdefault('max_ratio', Decimal('0.1'))
            
            position_config = PositionConfig(**position_config)
            position_config.data = data


        self.position_config = position_config
        self.position_manager.update(position_config)

    def add_position_policy(self, config: LeekComponentConfig[LeekComponent, Dict[str, Any]]):
        """添加全局仓位风控策略"""
        logger.info(f"添加仓位风控策略: {config}")
        if isinstance(config, dict):
            data = {
                "scope": config.get("scope"),
                "strategy_template_ids": config.get("strategy_template_ids"),
                "strategy_instance_ids": config.get("strategy_instance_ids"),
            }
            config = self.format_component_config(config)
            config.extra = data
        self.position_manager.add_policy(config)

    def update_position_policy(self, config: LeekComponentConfig[LeekComponent, Dict[str, Any]]):
        """更新全局仓位风控策略：按类名先移除后添加"""
        logger.info(f"更新仓位风控策略: {config}")
        if isinstance(config, dict):
            instance_id = str(config.get("id"))
        else:
            instance_id = config.instance_id
        self.remove_position_policy(instance_id)
        self.add_position_policy(config)

    def remove_position_policy(self, instance_id: str):
        """移除全局仓位风控策略，优先按实例ID移除，其次按类名移除"""
        self.position_manager.remove_policy(instance_id)

    def ping(self, instance_id: str):
        """响应主进程的 ping 消息，回复 pong"""
        if instance_id == self.instance_id:
            return True
        return False
    
    def close_position(self, position_id: str):
        """
        关闭仓位
        """
        position = self.position_manager.get_position(str(position_id))
        logger.info(f"关闭仓位: {position_id}, {'已找到仓位' if position else '未找到仓位'}")
        if position:
            self.strategy_manager.close_position(position)
            return True
        return False

    def reset_position_state(self):
        """
        重置仓位状态
        """
        self.position_manager.reset_position_state()
        return self.position_manager.get_state()

    def on_start(self):
        """启动引擎组件"""
        self.data_source_manager.on_start()
        self.strategy_manager.on_start()
        self.position_manager.on_start()
        self.executor_manager.on_start()
        self.event_bus.subscribe_event(None, self.handle_event)
        self.running = True

    def on_stop(self):
        """引擎停止时的回调"""
        self.running = False
        # 先取消全局事件订阅，避免停止期间产生无关处理
        try:
            self.event_bus.unsubscribe_event(None, self.handle_event)
        except Exception:
            pass

        # 停止引擎组件（顺序很重要）
        # 1) 策略先停止（会发布数据源取消订阅事件）
        self.strategy_manager.on_stop()
        # 2) 执行器停止
        self.executor_manager.on_stop()
        # 3) 仓位管理停止
        self.position_manager.on_stop()
        # 4) 数据源最后停止（确保能处理取消订阅）
        self.data_source_manager.on_stop()

        # 最后关闭事件总线
        try:
            self.event_bus.shutdown()
        except Exception:
            pass
        logger.info(f"引擎已停止: {self.instance_id} {self.name}")
    
    def start(self) -> None:
        ...

    def _handle_event(self, event: Event):
        ...

    def shutdown(self):
        ...
