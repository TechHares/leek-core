#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, Set

from leek_core.base import LeekContext
from leek_core.event import EventBus, Event, EventType, EventSource
from leek_core.models import LeekComponentConfig, ExecutionContext, PositionInfo
from leek_core.models import RiskEventType, RiskEvent
from leek_core.utils import get_logger
from .strategy import StrategyPolicy

logger = get_logger(__name__)


class StrategyPolicyContext(LeekContext):
    """
    策略风控上下文。

    - 统一保存策略风控的实例、配置与实例ID
    - 提供 evaluate 入口封装
    - 统一生命周期管理 on_start/on_stop
    """

    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[StrategyPolicy, Dict[str, Any]]):
        super().__init__(event_bus, config)
        self.policy: StrategyPolicy = self.create_component()
        # 过滤配置
        data = config.extra or {}
        self.scope: str = data.get('scope', 'all')
        # 前端“实例”选择的是策略ID
        self.allowed_strategy_ids: Set[str] = set(str(x) for x in (data.get('strategy_instance_ids') or []))
        # 前端“模板”选择的是 模块|类 名称
        self.allowed_strategy_templates: Set[str] = set(data.get('strategy_template_ids') or [])

    def evaluate(self, signal: ExecutionContext, context: PositionInfo) -> bool:
        result = self.policy.evaluate(signal, context)
        
        # 如果风控策略评估失败，发布风控事件
        if not result:
            self._publish_policy_risk_event(signal, context)
        
        return result

    def is_applicable(self, signal: ExecutionContext) -> bool:
        # 全部适用
        if self.scope == 'all' or not self.scope:
            return True
        # 实例/策略ID 过滤
        if self.scope in ('strategy_instances', 'mixed'):
            if self.allowed_strategy_ids and str(signal.strategy_id) in self.allowed_strategy_ids:
                return True
            # 如果指定了实例过滤而未命中，则视为不适用
            if self.scope == 'strategy_instances':
                return False
        # 模板过滤（需要策略类名，可选）
        if self.scope in ('strategy_templates', 'mixed') and signal.strategy_cls:
            if self.allowed_strategy_templates and signal.strategy_cls in self.allowed_strategy_templates:
                return True
            if self.scope == 'strategy_templates':
                return False
        # mixed 情况下，如果上述都未命中，则不适用
        return self.scope == 'mixed' and False

    def on_start(self):
        self.policy.on_start()

    def on_stop(self):
        self.policy.on_stop()

    def _publish_policy_risk_event(self, signal: ExecutionContext, context: PositionInfo):
        """
        发布策略风控事件
        
        Args:
            signal: 执行上下文
            context: 仓位信息
        """
        try:
            # 创建风控事件数据
            data = RiskEvent(
                risk_type=RiskEventType.SIGNAL,
                strategy_id=signal.strategy_id,
                strategy_instance_id=signal.strategy_instance_id,
                strategy_class_name=signal.strategy_cls,
                risk_policy_id=self.instance_id,
                risk_policy_class_name=f"{self.policy.__class__.__module__}|{self.policy.__class__.__name__}",
                trigger_time=datetime.now(),
                trigger_reason=f"风控 {self.name} 拒绝信号",
                signal_id=signal.signal_id,
                execution_order_id=signal.context_id,
                position_id=None,
                original_amount=Decimal(signal.open_amount),
                pnl=None,
                extra_info={
                },
            )
            # 发布风控触发事件
            event = Event(
                event_type=EventType.RISK_TRIGGERED,
                data=data,
                source=EventSource(
                    instance_id=self.instance_id,
                    name=self.name,
                    cls=self.policy.__class__.__name__
                )
            )
            self.event_bus.publish_event(event)
        except Exception as e:
            logger.error(f"发布策略风控事件失败: {e}", exc_info=True)


