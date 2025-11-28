#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar

from leek_core.base import LeekComponent

from .constants import OrderType, TradeInsType, TradeMode

if TYPE_CHECKING:
    from leek_core.data import DataSource, DataSourceContext
    from leek_core.executor import Executor
    from leek_core.info_fabricator import Fabricator
    from leek_core.strategy import Strategy
    from leek_core.sub_strategy import SubStrategy


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


T = TypeVar('T', bound=LeekComponent)
CFG = TypeVar('CFG')
@dataclass
class LeekComponentConfig(Generic[T, CFG]):
    """
    各种组件的类型和初始化配置
    """
    instance_id: str = None
    name: str = None
    cls: Optional[type[T]] = None
    config: CFG = None
    data: Optional[Dict[str, Any]] = None  # 存储组件的运行数据
    extra: Optional[Dict[str, Any]] = None  # 存储组件的扩展数据

@dataclass
class StrategyPositionConfig:
    """
    策略仓位配置
    """
    principal: Decimal = None # 策略本金
    leverage: Decimal = None # 策略杠杆倍数
    order_type: OrderType = None  # 订单类型
    executor_id: str = None  # 指定执行器ID

    def __post_init__(self):
        self.order_type = OrderType(self.order_type) if self.order_type else None
        self.leverage = Decimal(self.leverage) if self.leverage else None
        self.principal = Decimal(self.principal) if self.principal else None
        self.executor_id = str(self.executor_id) if self.executor_id else None


@dataclass
class StrategyConfig:
    """
    策略配置信息
    """
    data_source_configs: List[LeekComponentConfig["DataSource", Dict[str, Any]]]
    info_fabricator_configs: List[LeekComponentConfig["Fabricator", Dict[str, Any]]] = field(default_factory=list)

    strategy_config: Dict[str, Any] = None
    strategy_position_config: StrategyPositionConfig = None

    risk_policies: List[LeekComponentConfig["SubStrategy", Dict[str, Any]]] = field(default_factory=list)
    runtime_data: Dict[str, Dict[str, Any]] = None


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

    default_leverage: int = 1 # 默认杠杆倍数
    order_type: OrderType = OrderType.MarketOrder # 订单类型
    trade_type: TradeInsType = TradeInsType.SPOT # 交易类型
    trade_mode: TradeMode = TradeMode.CROSS # 交易模式
    data: Dict[str, Any] = None

    def __post_init__(self):
        # 转换 Decimal 字段
        for field_name in ['init_amount', 'max_strategy_amount', 'max_strategy_ratio', 'max_symbol_amount', 'max_symbol_ratio', 'max_amount', 'max_ratio']:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Decimal(value))

        # 转换枚举字段
        if isinstance(self.order_type, int):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.trade_type, int):
            self.trade_type = TradeInsType(self.trade_type)
        if isinstance(self.trade_mode, str):
            self.trade_mode = TradeMode(self.trade_mode)


@dataclass
class BacktestEngineConfig:
    """
    引擎配置 包含引擎启动所有组件的配置
    """
    data_sources: LeekComponentConfig["DataSourceContext", List[LeekComponentConfig["DataSource", Dict[str, Any]]]]
    strategy_configs: List[LeekComponentConfig["Strategy", StrategyConfig]] = field(default_factory=list)
    position_config: PositionConfig = None
    executor_configs: List[LeekComponentConfig["Executor", Dict[str, Any]]] = field(default_factory=list)
    timeout: int = 10 # 超时时间，单位秒
