#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控事件模型定义
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum

from .constants import PositionSide, AssetType, TradeInsType


class RiskEventType(Enum):
    """风控事件类型"""
    EMBEDDED = "embedded"  # 策略内嵌风控
    SIGNAL = "signal"      # 信号风控
    ACTIVE = "active"      # 主动风控


@dataclass
class RiskEvent:
    """
    风控触发事件模型
    
    用于在风控系统中传递风控触发信息，包含触发的风控类型、
    相关的策略信息、仓位信息、触发原因等。
    """
    # 基本信息
    risk_type: RiskEventType
    
    # 策略相关
    strategy_id: Optional[int] = None
    strategy_instance_id: Optional[str] = None
    strategy_class_name: Optional[str] = None
    
    # 风控策略信息
    risk_policy_id: Optional[int] = None
    risk_policy_class_name: str = ""
    
    # 触发信息
    trigger_time: datetime = None
    trigger_reason: Optional[str] = None
    
    # 信号相关（仅 signal 类型）
    signal_id: Optional[int] = None
    execution_order_id: Optional[int] = None
    
    # 仓位相关（embedded 和 active 类型）
    position_id: Optional[int] = None
    
    # 风控结果
    original_amount: Optional[Decimal] = None
    pnl: Optional[Decimal] = None
    
    # 扩展信息
    extra_info: Optional[Dict[str, Any]] = None
    tags: Optional[list] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.trigger_time is None:
            self.trigger_time = datetime.now()
        
        # 确保 risk_type 是枚举类型
        if isinstance(self.risk_type, str):
            self.risk_type = RiskEventType(self.risk_type)
        
        # 转换 Decimal 字段
        if self.original_amount is not None and not isinstance(self.original_amount, Decimal):
            self.original_amount = Decimal(str(self.original_amount))
        if self.pnl is not None and not isinstance(self.pnl, Decimal):
            self.pnl = Decimal(str(self.pnl))
