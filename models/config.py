#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar, Dict, Any, List

from models import OrderType


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


@dataclass
class InstanceInitConfig:
    """
    用户各种灵活启用的配置
    """
    cls: ClassVar
    config: Dict[str, Any]

    def get_instance(self):
        sig = inspect.signature(self.cls)
        params = sig.parameters
        parameter_map = {k: v for k, v in self.config.items() if k in list(params.keys())}
        return self.cls(**parameter_map)



@dataclass
class StrategyPositionConfig:
    """
    策略仓位配置
    """
    principal: Decimal = None # 策略本金
    leverage: Decimal = None # 策略杠杆倍数
    order_type: OrderType = None  # 订单类型
    executor_id: str = None  # 指定执行器ID

    risk_policy: List[InstanceInitConfig] = field(default_factory=list)