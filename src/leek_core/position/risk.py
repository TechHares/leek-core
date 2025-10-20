#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Dict, Any

from leek_core.base import LeekComponent
from leek_core.event import EventBus, Event, EventType
from leek_core.models import ExecutionContext, PositionInfo, LeekComponentConfig
from leek_core.policy import StrategyPolicy
from leek_core.policy.context import StrategyPolicyContext
from leek_core.utils import get_logger

logger = get_logger(__name__)


class RiskManager(LeekComponent):
    """
    风控管理器组件 - 纯风控策略管理
    
    职责：
    - 风控策略管理
    - 风控检查执行
    - 风控事件发布
    - 风控策略生命周期管理
    """
    
    def __init__(self, event_bus: EventBus):
        """
        初始化风控管理器
        
        参数:
            event_bus: 事件总线
        """
        super().__init__()
        self.event_bus = event_bus
        self.policies: List[StrategyPolicyContext] = []
        
        logger.info("RiskManager 初始化完成")
    
    def evaluate_risk(self, execution_context: ExecutionContext, position_info: PositionInfo) -> bool:
        """
        执行风控策略检查
        
        参数:
            execution_context: 执行上下文
            position_info: 仓位信息
            
        返回:
            bool: True表示风控拦截，False表示风控通过
        """
        # 如果是纯减仓操作，跳过风控检查
        if all(asset.is_open is False for asset in execution_context.execution_assets):
            logger.debug(f"纯减仓操作，跳过风控检查: {execution_context.signal_id}")
            return False
        
        for policy in self.policies:
            if policy.is_applicable(execution_context) and not policy.evaluate(execution_context, position_info):
                logger.warning(
                    f"Risk policy {policy.instance_id}/{policy.name} rejected signal {execution_context}")
                execution_context.extra = (execution_context.extra or {}) | {
                    "policy_id": policy.instance_id,
                }
                return True
        return False
    
    def add_policy(self, policy_config: LeekComponentConfig[StrategyPolicy, Dict[str, Any]]):
        """
        添加风控策略
        
        参数:
            policy_config: 策略配置
        """
        try:
            ctx = StrategyPolicyContext(self.event_bus, policy_config)
            ctx.on_start()
            self.policies.append(ctx)
            
            # 发布策略添加事件
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_POLICY_ADD,
                data={
                    "policy_id": ctx.instance_id,
                    "policy_name": ctx.name,
                    "policy_class": policy_config.cls.__name__ if policy_config.cls else "Unknown"
                }
            ))
            
            logger.info(f"风控策略添加成功: {ctx.instance_id}/{ctx.name}")
            
        except Exception as e:
            logger.error(f"添加风控策略失败: {e}", exc_info=True)
            raise
    
    def remove_policy(self, instance_id: str):
        """
        移除风控策略
        
        参数:
            instance_id: 策略实例ID
        """
        del_policies = [p for p in self.policies if p.instance_id == instance_id]
        self.policies = [p for p in self.policies if p.instance_id != instance_id]
        for policy in del_policies:
            try:
                policy.on_stop()
            finally:
                self.event_bus.publish_event(Event(
                    event_type=EventType.POSITION_POLICY_DEL,
                    data={
                        "instance_id": policy.instance_id,
                        "name": policy.name,
                    }
                ))
    
    def on_stop(self):
        """清空所有风控策略"""
        for policy in self.policies:
            try:
                policy.on_stop()
            except Exception as e:
                logger.error(f"停止风控策略失败: {e}", exc_info=True)
        
        self.policies.clear()
        logger.info("所有风控策略已清空")
    
    def _publish_risk_event(self, execution_context: ExecutionContext, 
                           policy: StrategyPolicyContext, action: str):
        """
        发布风控事件
        
        参数:
            execution_context: 执行上下文
            policy: 风控策略
            action: 动作（REJECTED, APPROVED等）
        """
        self.event_bus.publish_event(Event(
            event_type=EventType.RISK_TRIGGERED,
            data={
                "signal_id": execution_context.signal_id,
                "strategy_id": execution_context.strategy_id,
                "policy_id": policy.instance_id,
                "policy_name": policy.name,
                "action": action,
                "execution_context": execution_context
            }
        ))
    
    def reset(self):
        ...