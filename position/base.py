#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控基础模块，提供风控的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
import uuid

from models import Component, Signal, PositionContext
from models.constants import PositionSide
from utils import get_logger

logger = get_logger(__name__)


class Policy(Component, ABC):
    """
    风控策略抽象基类，定义风控策略的基本接口。
    
    风控策略用于检查交易信号是否符合风控规则，如果不符合则拒绝交易或使用虚拟仓位。
    """
    
    def __init__(self, instance_id: str, name: str, **kwargs):
        """
        初始化风控策略
        
        参数:
            policy_id: 策略ID，如果不提供则自动生成
        """
        super().__init__(instance_id, name)
        self.args = kwargs

        self.reject_reason = ""

    @abstractmethod
    def _check(self, signal: Signal, context: PositionContext) -> str:
        """
        检查信号是否符合风控规则
        
        参数:
            signal: 信号数据
            
        返回:
            拒绝原因
        """
        pass
    
    def check(self, signal: Signal, context: PositionContext) -> bool:
        """
        检查信号是否符合风控规则

        参数:
            signal: 信号数据

        返回:
            是否通过检查
        """
        self.reject_reason = self._check(signal, context)
        return self.reject_reason == ""
