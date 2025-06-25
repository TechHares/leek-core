#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易引擎核心实现
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from leek_core.base import LeekComponent
from leek_core.data import DataSource
from leek_core.event import EventBus
from leek_core.executor import Executor
from leek_core.models import LeekComponentConfig, StrategyConfig, PositionConfig
from leek_core.strategy import Strategy

from leek_core.utils import get_logger

logger = get_logger(__name__)


class Engine(LeekComponent, ABC):
    def __init__(self):
        self.running = False

    def add_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]) -> None:
        raise NotImplementedError

    def update_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]) -> None:
        raise NotImplementedError

    def remove_strategy(self, instance_id: str) -> None:
        raise NotImplementedError

    def add_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]) -> None:
        raise NotImplementedError

    def update_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]) -> None:
        raise NotImplementedError

    def remove_executor(self, instance_id: str) -> None:
        raise NotImplementedError

    def add_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]) -> None:
        raise NotImplementedError

    def update_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]) -> None:
        raise NotImplementedError

    def remove_data_source(self, instance_id: str) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def update_position_config(self, position_config: PositionConfig) -> None:
        raise NotImplementedError
    

