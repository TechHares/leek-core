#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar, Dict, Any, List, Type, Tuple, TypeVar, Generic

from base import LeekComponent
from .constants import OrderType

@dataclass
class EngineRuntimeData:
    """
    引擎运行时数据
    """
    data: Dict[str, Any] = field(default_factory=dict)
    data_manager_data: Dict[str, Any] = field(default_factory=dict)
    strategy_manager_data: Dict[str, Any] = field(default_factory=dict)
    position_manager_data: Dict[str, Any] = field(default_factory=dict)
    risk_manager_data: Dict[str, Any] = field(default_factory=dict)
    executor_manager_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionConfig:
    """
    仓位配置
    """
    init_amount: Decimal          # 初始余额
    max_strategy_amount: Decimal  # 单个策略最大仓位
    max_strategy_ratio: Decimal    # 单个策略最大仓位比例

    max_symbol_amount: Decimal    # 单个标的最大仓位
    max_symbol_ratio: Decimal      # 单个标的最大仓位比例

    max_amount: Decimal    # 单次开仓最大仓位
    max_ratio: Decimal      # 单单次开仓最大仓位比例


T = TypeVar('T', bound=LeekComponent)
CFG = TypeVar('CFG')
@dataclass
class LeekComponentConfig(Generic[T, CFG]):
    """
    各种组件的类型和初始化配置
    """
    instance_id: str
    name: str
    cls: type[T]
    config: CFG



@dataclass
class StrategyPositionConfig:
    """
    策略仓位配置
    """
    principal: Decimal = None # 策略本金
    leverage: Decimal = None # 策略杠杆倍数
    order_type: OrderType = None  # 订单类型
    executor_id: str = None  # 指定执行器ID


@dataclass
class StrategyConfig:
    """
    策略配置信息
    """
    data_source_configs: List[LeekComponentConfig["DataSource", Dict[str, Any]]]
    strategy_config: Dict[str, Any] = None
    strategy_position_config: StrategyPositionConfig = None

    enter_strategy_cls: Type["EnterStrategy"] = None
    enter_strategy_config: Dict[str, Any] = None
    exit_strategy_cls: Type['ExitStrategy'] = None
    exit_strategy_config: Dict[str, Any] = None

    risk_policies: List[LeekComponentConfig["PositionPolicy", Dict[str, Any]]] = field(default_factory=list)