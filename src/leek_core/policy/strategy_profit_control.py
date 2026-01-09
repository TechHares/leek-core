#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略盈利风控模块

基于策略最近盈利情况的风控策略，通过跟踪策略的盈亏表现来决定是否放行新的交易信号。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Set, List, Optional

from leek_core.policy.strategy import StrategyPolicy
from leek_core.models import ExecutionContext, PositionInfo, Field, FieldType
from leek_core.event import Event, EventType
from leek_core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime          # 交易时间
    profit_pct: float            # 盈利百分比（正盈负亏）
    strategy_key: str            # 策略唯一标识 (strategy_id:strategy_instance_id)


class StrategyProfitControl(StrategyPolicy):
    """
    基于策略最近盈利情况的风控策略。

    该策略通过跟踪策略最近的盈利情况来决定是否放行新的交易信号。
    当策略最近表现不佳时，会限制新的交易信号转为虚拟单，以防止进一步亏损。
    当虚拟单表现恢复后，会重新放行真实交易。

    风控触发条件（满足其一即触发）：
    1. 连续亏损累计和超过阈值
    2. 连续亏损笔数超过阈值

    恢复条件（满足其一即恢复）：
    1. 从风控触发点累计盈利超过阈值
    2. 虚拟单连续盈利笔数超过阈值
    3. 风控封禁时间超过最大封禁时长
    """
    display_name = "策略盈利风控"
    init_params = [
        Field(
            name='observation_hours',
            label='观察时间窗口',
            type=FieldType.INT,
            default=24,
            required=True,
            description='观察时间窗口（小时），只统计该时间范围内的交易记录'
        ),
        Field(
            name='observation_count',
            label='观察交易笔数',
            type=FieldType.INT,
            default=10,
            required=True,
            description='观察交易笔数上限，最多保留最近N笔交易记录'
        ),
        Field(
            name='max_cumulative_loss_pct',
            label='连续亏损累计阈值',
            type=FieldType.FLOAT,
            default=10.0,
            required=True,
            description='连续亏损累计百分比阈值（正数），连续亏损累计和超过此值将触发风控'
        ),
        Field(
            name='max_consecutive_losses',
            label='最大连续亏损笔数',
            type=FieldType.INT,
            default=3,
            required=True,
            description='最大连续亏损笔数，连续亏损达到此值将触发风控'
        ),
        Field(
            name='recovery_win_count',
            label='恢复连续盈利笔数',
            type=FieldType.INT,
            default=2,
            required=True,
            description='恢复所需虚拟单连续盈利笔数，达到此值可恢复真实交易'
        ),
        Field(
            name='recovery_cumulative_profit_pct',
            label='恢复累计盈利阈值',
            type=FieldType.FLOAT,
            default=2.0,
            required=True,
            description='恢复累计盈利百分比阈值（正数），从风控触发点累计盈利超过此值可恢复真实交易'
        ),
        Field(
            name='max_pause_hours',
            label='最大封禁时长',
            type=FieldType.INT,
            default=0,
            required=True,
            description='最大封禁时长（小时），超过此时长自动恢复交易；0或负数表示无限封禁直到满足恢复条件'
        ),
    ]

    def __init__(
        self,
        observation_hours: int = 24,
        observation_count: int = 10,
        max_cumulative_loss_pct: float = 10.0,
        max_consecutive_losses: int = 3,
        recovery_win_count: int = 2,
        recovery_cumulative_profit_pct: float = 2.0,
        max_pause_hours: int = 0,
    ):
        """
        初始化策略盈利风控

        参数:
            observation_hours: 观察时间窗口（小时）
            observation_count: 观察交易笔数上限
            max_cumulative_loss_pct: 连续亏损累计百分比阈值
            max_consecutive_losses: 最大连续亏损笔数
            recovery_win_count: 恢复所需连续盈利笔数
            recovery_cumulative_profit_pct: 恢复累计盈利百分比阈值
            max_pause_hours: 最大封禁时长（小时），0或负数表示无限封禁
        """
        super().__init__()
        self.observation_hours = observation_hours
        self.observation_count = observation_count
        self.max_cumulative_loss_pct = max_cumulative_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.recovery_win_count = recovery_win_count
        self.recovery_cumulative_profit_pct = recovery_cumulative_profit_pct
        self.max_pause_hours = max_pause_hours
        
        # 交易记录列表
        self._trade_records: List[TradeRecord] = []
        
        # 跟踪的策略唯一标识集合 (strategy_id:strategy_instance_id)
        self._tracked_strategies: Set[str] = set()
        
        # === 风控触发相关 ===
        # 连续亏损累计和（百分比，正数表示亏损）
        self._cumulative_loss: float = 0.0
        # 连续亏损笔数
        self._consecutive_losses: int = 0
        # 风控触发时间
        self._pause_start_time: Optional[datetime] = None
        
        # === 恢复相关 ===
        # 从风控触发点开始的累计盈利（百分比）
        self._recovery_cumulative_profit: float = 0.0
        # 连续盈利笔数（用于恢复判断）
        self._consecutive_wins: int = 0
        
        # 是否暂停交易
        self._is_trading_paused: bool = False

    def _cleanup_expired_records(self):
        """
        清理过期的交易记录
        按时间和笔数两个维度清理
        """
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.observation_hours)
        
        # 按时间过滤
        self._trade_records = [ r for r in self._trade_records if r.timestamp >= cutoff_time]
        
        # 按笔数过滤，保留最近的 observation_count 笔
        if len(self._trade_records) > self.observation_count:
            self._trade_records = self._trade_records[-self.observation_count:]

    def _record_trade(self, profit_ratio: float, strategy_key: str):
        """
        记录交易并更新状态
        
        参数:
            profit_ratio: 盈利比例（小数形式，如0.05表示5%）
            strategy_key: 策略唯一标识 (strategy_id:strategy_instance_id)
        """
        profit_pct = profit_ratio * 100  # 转百分比
        
        # 记录交易
        record = TradeRecord(timestamp=datetime.now(), profit_pct=profit_pct, strategy_key=strategy_key)
        self._trade_records.append(record)
        self._cleanup_expired_records()
        
        if not self._is_trading_paused:
            # 未暂停状态：检查是否触发风控
            if profit_ratio < 0:
                self._cumulative_loss += abs(profit_pct)
                self._consecutive_losses += 1
                
                logger.info(
                    f"[{self.policy_instance_id}] 记录亏损: {profit_pct:.2f}%, "
                    f"累计亏损: {self._cumulative_loss:.2f}%, 连续亏损: {self._consecutive_losses}笔"
                )
                
                # 检查是否触发风控
                if (self._cumulative_loss > self.max_cumulative_loss_pct or 
                    self._consecutive_losses >= self.max_consecutive_losses):
                    self._is_trading_paused = True
                    self._pause_start_time = datetime.now()
                    # 重置恢复相关计数
                    self._recovery_cumulative_profit = 0.0
                    self._consecutive_wins = 0
                    logger.error(
                        f"[{self.policy_instance_id}] 风控触发! "
                        f"累计亏损: {self._cumulative_loss:.2f}% (阈值: {self.max_cumulative_loss_pct}%), "
                        f"连续亏损: {self._consecutive_losses}笔 (阈值: {self.max_consecutive_losses}笔)"
                    )
            else:
                # 盈利：重置连续亏损统计
                logger.info(
                    f"[{self.policy_instance_id}] 记录盈利: {profit_pct:.2f}%, 重置连续亏损统计"
                )
                self._cumulative_loss = 0.0
                self._consecutive_losses = 0
        else:
            # 已暂停状态：检查恢复条件（不区分虚拟/真实）
            # 累加恢复累计盈利（盈亏都算）
            self._recovery_cumulative_profit += profit_pct
            
            if profit_ratio > 0:
                self._consecutive_wins += 1
            else:
                self._consecutive_wins = 0
            
            logger.info(
                f"[{self.policy_instance_id}] 风控期间交易: {profit_pct:.2f}%, "
                f"恢复累计盈利: {self._recovery_cumulative_profit:.2f}%, "
                f"连续盈利: {self._consecutive_wins}笔"
            )
            
            # 检查恢复条件
            if (self._recovery_cumulative_profit >= self.recovery_cumulative_profit_pct or
                self._consecutive_wins >= self.recovery_win_count):
                self._resume_trading(
                    f"累计盈利: {self._recovery_cumulative_profit:.2f}% (阈值: {self.recovery_cumulative_profit_pct}%), "
                    f"连续盈利: {self._consecutive_wins}笔 (阈值: {self.recovery_win_count}笔)"
                )

    def _resume_trading(self, reason: str):
        """
        恢复交易，重置所有状态
        
        参数:
            reason: 恢复原因描述
        """
        self._is_trading_paused = False
        self._pause_start_time = None
        # 重置所有计数
        self._cumulative_loss = 0.0
        self._consecutive_losses = 0
        self._recovery_cumulative_profit = 0.0
        self._consecutive_wins = 0
        logger.info(f"[{self.policy_instance_id}] 交易恢复! {reason}")

    def _check_pause_timeout(self) -> bool:
        """
        检查是否超过最大封禁时长
        
        返回:
            是否超时（超时返回True）
        """
        if self.max_pause_hours <= 0:
            # 0或负数表示无限封禁
            return False
        
        if not self._pause_start_time:
            return False
        
        elapsed = datetime.now() - self._pause_start_time
        return elapsed >= timedelta(hours=self.max_pause_hours)

    @staticmethod
    def _make_strategy_key(strategy_id: str, strategy_instance_id: str) -> str:
        """
        生成策略唯一标识
        
        参数:
            strategy_id: 策略ID
            strategy_instance_id: 策略实例ID
            
        返回:
            策略唯一标识 (strategy_id:strategy_instance_id)
        """
        return f"{strategy_id}:{strategy_instance_id}"

    def evaluate(self, signal: ExecutionContext, context: PositionInfo) -> bool:
        """
        评估信号是否应该放行

        参数:
            signal: 交易信号
            context: 持仓信息

        返回:
            是否放行信号（True=放行真实交易，False=转为虚拟单）
        """
        # 记录策略唯一标识到跟踪集合
        strategy_key = self._make_strategy_key(signal.strategy_id, signal.strategy_instance_id)
        self._tracked_strategies.add(strategy_key)
        
        # 如果交易被暂停，检查是否超时恢复
        if self._is_trading_paused:
            if self._check_pause_timeout():
                elapsed_hours = (datetime.now() - self._pause_start_time).total_seconds() / 3600
                self._resume_trading(
                    f"封禁超时自动恢复 (已封禁 {elapsed_hours:.1f} 小时, 阈值: {self.max_pause_hours} 小时)"
                )
            else:
                logger.warning(
                    f"[{self.policy_instance_id}] 信号被风控拒绝: strategy={strategy_key}, "
                    f"累计亏损={self._cumulative_loss:.2f}%, 连续亏损={self._consecutive_losses}笔"
                )
                return False
        
        logger.debug(
            f"[{self.policy_instance_id}] 信号放行: strategy={strategy_key}"
        )
        return True

    def on_start(self):
        """
        启动风控策略，订阅执行订单更新事件
        """
        if self.event_bus:
            self.event_bus.subscribe_event(EventType.EXEC_ORDER_UPDATED, self._on_exec_order_updated)
            logger.info(f"[{self.policy_instance_id}] 已订阅 EXEC_ORDER_UPDATED 事件")
    
    def on_stop(self):
        """
        停止风控策略，取消订阅事件并清理状态
        """
        if self.event_bus:
            self.event_bus.unsubscribe_event(EventType.EXEC_ORDER_UPDATED, self._on_exec_order_updated)
            logger.info(f"[{self.policy_instance_id}] 已取消订阅 EXEC_ORDER_UPDATED 事件")
        
        # 清理状态
        self._tracked_strategies.clear()
        self._trade_records.clear()
        self._cumulative_loss = 0.0
        self._consecutive_losses = 0
        self._pause_start_time = None
        self._recovery_cumulative_profit = 0.0
        self._consecutive_wins = 0
        self._is_trading_paused = False
    
    def _on_exec_order_updated(self, event: Event):
        """
        处理执行订单更新事件，统计平仓盈亏
        
        参数:
            event: 执行订单更新事件，data 为 ExecutionContext 对象
        """
        try:
            exec_ctx: ExecutionContext = event.data
            if not exec_ctx:
                return
            
            # 检查是否完成
            if not exec_ctx.is_finish:
                return
            
            # 生成策略唯一标识
            strategy_key = self._make_strategy_key(exec_ctx.strategy_id, exec_ctx.strategy_instance_id)
            
            # 检查策略是否在跟踪集合中
            if strategy_key not in self._tracked_strategies:
                return
            
            # 检查是否有平仓资产
            if all(asset.is_open for asset in exec_ctx.execution_assets):
                return
            
            # 从 execution_assets 中汇总盈亏：actual_pnl + virtual_pnl
            total_actual_pnl = sum((asset.actual_pnl or 0) for asset in exec_ctx.execution_assets if asset.is_open is False)
            total_virtual_pnl = sum((asset.virtual_pnl or 0) for asset in exec_ctx.execution_assets if asset.is_open is False)
            total_pnl = total_actual_pnl + total_virtual_pnl
            
            # 计算盈利比例
            if exec_ctx.close_amount and exec_ctx.close_amount > 0:
                profit_ratio = float(total_pnl / exec_ctx.close_amount)
                
                logger.info(
                    f"[{self.policy_instance_id}] 收到订单完成: "
                    f"strategy={strategy_key}, "
                    f"actual_pnl={total_actual_pnl}, virtual_pnl={total_virtual_pnl}, "
                    f"total_pnl={total_pnl}, close_amount={exec_ctx.close_amount}, "
                    f"profit_ratio={profit_ratio:.4f}"
                )
                
                self._record_trade(profit_ratio, strategy_key)
                
        except Exception as e:
            logger.error(f"[{self.policy_instance_id}] 处理订单更新事件异常: {e}", exc_info=True)
