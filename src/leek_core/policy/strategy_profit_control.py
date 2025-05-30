#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
from decimal import Decimal
from typing import Optional, Dict, Any

from leek_core.policy.strategy import StrategyPolicy
from leek_core.models import Signal, PositionInfo, Field, FieldType
from leek_core.utils import get_logger

logger = get_logger(__name__)


class StrategyProfitControl(StrategyPolicy):
    """
    基于策略最近盈利情况的风控策略。

    该策略通过跟踪策略最近的盈利情况来决定是否放行新的交易信号。
    当策略最近表现不佳时，会限制新的交易信号，以防止进一步亏损。

    主要功能：
    1. 跟踪最近N笔交易的盈亏情况
    2. 根据盈亏比例决定是否放行新的交易信号
    3. 可配置的盈亏阈值和观察窗口大小
    4. 支持连续亏损次数限制
    5. 支持动态调整风控阈值
    """
    display_name = "策略盈利风控"
    init_params = [
        Field(
            name='window_size',
            label='观察窗口大小',
            type=FieldType.INT,
            default=10,
            required=True,
            description='观察窗口大小，即跟踪最近多少笔交易'
        ),
        Field(
            name='min_profit_ratio',
            label='最小盈利比例',
            type=FieldType.FLOAT,
            default=Decimal(-0.05),
            required=True,
            description='最小盈利比例阈值，低于此值将限制新信号'
        ),
        Field(
            name='max_loss_ratio',
            label='最大亏损比例',
            type=FieldType.FLOAT,
            default=Decimal(-0.1),
            required=True,
            description='最大亏损比例阈值，低于此值将完全停止新信号'
        ),
        Field(
            name='max_consecutive_losses',
            label='最大连续亏损次数',
            type=FieldType.INT,
            default=3,
            required=True,
            description='最大连续亏损次数，超过此值将暂停交易'
        ),
        Field(
            name='recovery_threshold',
            label='恢复阈值',
            type=FieldType.FLOAT,
            default=0.02,
            required=True,
            description='恢复阈值，当盈利超过此值时重置风控状态'
        )
    ]

    def __init__(
        self,
        window_size: int = 10,
        min_profit_ratio: Decimal = Decimal(-0.05),
        max_loss_ratio: Decimal = Decimal(-0.1),
        max_consecutive_losses: int = 3,
        recovery_threshold: Decimal = Decimal(0.02),
    ):
        """
        初始化策略盈利风控

        参数:
            window_size: 观察窗口大小，即跟踪最近多少笔交易
            min_profit_ratio: 最小盈利比例阈值，低于此值将限制新信号
            max_loss_ratio: 最大亏损比例阈值，低于此值将完全停止新信号
            max_consecutive_losses: 最大连续亏损次数
            recovery_threshold: 恢复阈值
        """
        super().__init__()
        self.window_size = window_size
        self.min_profit_ratio = min_profit_ratio
        self.max_loss_ratio = max_loss_ratio
        self.max_consecutive_losses = max_consecutive_losses
        self.recovery_threshold = recovery_threshold
        
        self.profit_history = deque(maxlen=window_size)
        self.consecutive_losses = 0
        self.is_trading_paused = False

    def update_profit(self, profit_ratio: float):
        """
        更新策略盈利历史

        参数:
            profit_ratio: 单笔交易的盈亏比例
        """
        self.profit_history.append(profit_ratio)
        
        # 更新连续亏损计数
        if profit_ratio < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # 检查是否需要暂停交易
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_trading_paused = True
            logger.warning(f"Trading paused due to {self.consecutive_losses} consecutive losses")
            
        # 检查是否可以恢复交易
        if self.is_trading_paused and profit_ratio >= self.recovery_threshold:
            self.is_trading_paused = False
            logger.info(f"Trading resumed after reaching recovery threshold {self.recovery_threshold:.2%}")
            
        logger.info(f"Updated profit history: {list(self.profit_history)}")

    def get_recent_profit_ratio(self) -> float:
        """
        获取最近交易的盈亏比例

        返回:
            最近交易的盈亏比例
        """
        if not self.profit_history:
            return 0.0
        return sum(self.profit_history) / len(self.profit_history)

    def evaluate(self, signal: Signal, context: PositionInfo) -> bool:
        """
        评估信号是否应该放行

        参数:
            signal: 交易信号
            context: 持仓信息

        返回:
            是否放行信号
        """
        # 如果交易被暂停，直接拒绝信号
        if self.is_trading_paused:
            logger.warning("Signal rejected: Trading is currently paused")
            return False
            
        recent_profit = self.get_recent_profit_ratio()
        
        # 如果亏损超过最大阈值，完全停止新信号
        if recent_profit <= self.max_loss_ratio:
            logger.warning(
                f"Signal rejected: Recent profit ratio {recent_profit:.2%} below max loss threshold {self.max_loss_ratio:.2%}"
            )
            return False
        
        # 如果亏损超过最小阈值，限制新信号
        if recent_profit <= self.min_profit_ratio:
            logger.warning(
                f"Signal rejected: Recent profit ratio {recent_profit:.2%} below min profit threshold {self.min_profit_ratio:.2%}"
            )
            return False
        
        logger.info(f"Signal accepted: Recent profit ratio {recent_profit:.2%} within acceptable range")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        获取风控策略当前状态

        返回:
            包含风控策略状态信息的字典
        """
        return {
            "recent_profit_ratio": self.get_recent_profit_ratio(),
            "consecutive_losses": self.consecutive_losses,
            "is_trading_paused": self.is_trading_paused,
            "profit_history": list(self.profit_history),
            "window_size": self.window_size,
            "min_profit_ratio": self.min_profit_ratio,
            "max_loss_ratio": self.max_loss_ratio,
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_threshold": self.recovery_threshold
        } 