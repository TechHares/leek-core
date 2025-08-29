#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

from leek_core.policy.strategy import StrategyPolicy
from leek_core.models import ChoiceType, ExecutionContext, PositionInfo, Field, FieldType, PositionSide
from leek_core.utils import get_logger

logger = get_logger(__name__)


class StrategyTimeWindow(StrategyPolicy):
    """
    按时间点风控策略

    在指定的时间戳点前后给定秒数范围内，限制指定方向/类型的开仓信号。

    参数:
    - action_scope: 风控的操作范围（all_open: 所有开仓；open_long: 开多；open_short: 开空）
    - pre_seconds: 时间点之前的窗口秒数
    - post_seconds: 时间点之后的窗口秒数
    - timestamps: 多个时间点（unix时间戳，秒或毫秒均可）
    """

    display_name = "按时间点风控"
    init_params = [
        Field(
            name='action_scope',
            label='风控操作范围',
            type=FieldType.RADIO,
            default=0,
            choices=[
                (0, '所有开仓'),
                (1, '仅多头开仓'),
                (2, '仅空头开仓'),
            ],
            choice_type=ChoiceType.INT,
            required=True,
            description='风控操作范围'
        ),
        Field(
            name='pre_seconds',
            label='时间点之前(秒)',
            type=FieldType.INT,
            default=1800,
            required=True,
            description='在时间点之前多长时间内生效，单位秒'
        ),
        Field(
            name='post_seconds',
            label='时间点之后(秒)',
            type=FieldType.INT,
            default=1800,
            required=True,
            description='在时间点之后多长时间内生效，单位秒'
        ),
        Field(
            name='timestamps',
            label='时间点(Unix时间戳)',
            type=FieldType.ARRAY,
            default=[],
            required=True,
            description='多个时间点，支持秒或毫秒级时间戳'
        ),
    ]

    def __init__(self,
                 action_scope: int = 0,
                 pre_seconds: int = 1800,
                 post_seconds: int = 1800,
                 timestamps: List[int] = None):
        super().__init__()
        
        # 调试信息：打印原始参数
        self.action_scope = action_scope or 0
        self.pre_seconds = int(pre_seconds or 0)
        self.post_seconds = int(post_seconds or 0)
        
        # 处理 timestamps 参数
        if timestamps is None:
            self.timestamps = []
        elif isinstance(timestamps, str):
            # 如果是字符串，尝试解析为数组
            try:
                import json
                self.timestamps = json.loads(timestamps)
            except:
                # 如果不是JSON格式，尝试按逗号分割
                self.timestamps = [int(x.strip()) for x in timestamps.split(',') if x.strip().isdigit()]
        elif isinstance(timestamps, (list, tuple)):
            self.timestamps = [int(x) for x in timestamps if x is not None]
        else:
            self.timestamps = []

    def _ts_sec(self, ts: int | float) -> int:
        """将传入的时间戳转换为秒级（支持毫秒传入）。"""
        try:
            ts = int(ts)
        except Exception:
            return 0
        # 粗略判断毫秒
        return ts // 1000 if ts > 10**12 else ts

    def _in_window(self, now_sec: int) -> bool:
        for raw in self.timestamps:
            sec = self._ts_sec(raw)
            if sec <= 0:
                logger.debug(f"StrategyTimeWindow: 跳过无效时间戳 {raw}")
                continue
            window_start = sec - self.pre_seconds
            window_end = sec + self.post_seconds
            logger.debug(f"StrategyTimeWindow: 检查时间点 {sec}, 窗口 [{window_start}, {window_end}], 当前时间 {now_sec}")
            if window_start <= now_sec <= window_end:
                logger.debug(f"StrategyTimeWindow: 命中时间窗口 {sec}")
                return True
        logger.debug(f"StrategyTimeWindow: 未命中任何时间窗口")
        return False

    def _match_action(self, ctx: ExecutionContext) -> bool:
        if not ctx.execution_assets:
            return False
        # 仅判断开仓方向
        for asset in ctx.execution_assets:
            if not asset.is_open:
                continue
            # 调试信息
            logger.debug(f"StrategyTimeWindow: 检查资产 {asset.side}, action_scope={self.action_scope} (type: {type(self.action_scope)})")
            
            # 处理 action_scope 可能是字符串或整数的情况
            action_scope = int(self.action_scope) if isinstance(self.action_scope, str) and self.action_scope.isdigit() else self.action_scope
            
            if action_scope == 0:
                logger.debug(f"StrategyTimeWindow: 匹配所有开仓")
                return True
            if action_scope == 1 and asset.side == PositionSide.LONG:
                logger.debug(f"StrategyTimeWindow: 匹配多头开仓")
                return True
            if action_scope == 2 and asset.side == PositionSide.SHORT:
                logger.debug(f"StrategyTimeWindow: 匹配空头开仓")
                return True
        logger.debug(f"StrategyTimeWindow: 未匹配任何操作范围")
        return False

    def evaluate(self, signal: ExecutionContext, context: PositionInfo) -> bool:
        # 非目标操作，直接放行
        if not self._match_action(signal):
            logger.debug(f"StrategyTimeWindow: 非目标操作，放行信号 {signal.signal_id}")
            return True
        # 取执行上下文创建时间
        now_sec = int(signal.created_time.timestamp()) if signal.created_time else 0
        if now_sec <= 0:
            logger.debug(f"StrategyTimeWindow: 无效时间戳，放行信号 {signal.signal_id}")
            return True
        
        # 调试信息
        logger.debug(f"StrategyTimeWindow: 检查信号 {signal.signal_id}")
        logger.debug(f"StrategyTimeWindow: 信号时间 {now_sec} ({signal.created_time})")
        logger.debug(f"StrategyTimeWindow: 配置时间点 {self.timestamps}")
        logger.debug(f"StrategyTimeWindow: 窗口范围 {self.pre_seconds}秒前 - {self.post_seconds}秒后")
        
        # 如果在窗口内，则拒绝
        in_win = self._in_window(now_sec)
        if in_win:
            logger.info(f"StrategyTimeWindow: 命中时间窗口，拒绝信号 {signal.signal_id} @ {now_sec}")
            return False
        
        logger.debug(f"StrategyTimeWindow: 未命中时间窗口，放行信号 {signal.signal_id} @ {now_sec}")
        return True


