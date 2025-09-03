#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal
from threading import RLock
from typing import Dict, List, Any

from leek_core.base import LeekContext
from leek_core.event import EventBus, Event, EventType
from leek_core.models import (LeekComponentConfig, PositionConfig, Signal, ExecutionContext, ExecutionAsset, PositionInfo, Order, OrderStatus)
from leek_core.utils import get_logger, generate_str, decimal_quantize, thread_lock
from .capital_account import CapitalAccount
from .position_tracker import PositionTracker
from .risk import RiskManager

logger = get_logger(__name__)
_portfolio_lock = RLock()


class Portfolio(LeekContext):
    """
    投资组合管理器 - 组合使用 CapitalAccount 和 PositionTracker
    
    职责：
    - 信号处理和订单生成
    - 资金和仓位的协调使用
    - 风险控制和策略执行
    - 对外提供统一的业务接口
    """
    
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig[None, PositionConfig], 
                capital_account: CapitalAccount = None, 
                position_tracker: PositionTracker = None, 
                risk_manager: RiskManager = None):
        """
        初始化投资组合管理器
        
        参数:
            event_bus: 事件总线
            config: 配置信息
            capital_account: 资金账户
            position_tracker: 仓位跟踪器
            risk_manager: 风控管理器
        """
        super().__init__(event_bus, config)
        self.position_config = config.config
        # 组合使用轻量级组件
        self.capital_account = capital_account or CapitalAccount(event_bus, self.position_config.init_amount)
        self.position_tracker = position_tracker or PositionTracker(event_bus)
        self.risk_manager = risk_manager or RiskManager(event_bus)
        
        # 重要的统计数据（从原 PositionContext 迁移）
        self.pnl = Decimal(0)  # 总盈亏
        self.friction = Decimal(0)  # 摩擦成本
        self.fee = Decimal(0)  # 手续费
        self.virtual_pnl = Decimal(0)  # 虚拟盈亏
        
        # 信号和订单管理
        self.signals: Dict[str, Signal] = {}
        logger.info(f"Portfolio 初始化完成，初始资金: {self.position_config.init_amount}")
        self.load_state(config.data)
    
    @property
    def total_amount(self) -> Decimal:
        """总资产（初始资金 + 盈亏 + 费用）"""
        return self.position_config.init_amount + self.pnl + self.friction + self.fee
    
    @property
    def available_amount(self) -> Decimal:
        """可用资金"""
        return self.capital_account.available_balance
    
    @property
    def position_value(self) -> Decimal:
        """仓位价值"""
        return self.position_tracker.get_total_position_value()
    
    @property
    def total_value(self) -> Decimal:
        """总价值（可用资金 + 仓位价值）"""
        if len(self.position_tracker.positions) == 0:
            return self.available_amount
        return self.available_amount + self.position_value
    
    @property
    def profit(self) -> Decimal:
        """总收益"""
        return self.pnl + self.friction + self.fee
    
    @thread_lock(_portfolio_lock)
    def process_signal(self, signal: Signal, cls: str = None):
        """
        处理策略信号 - Portfolio 的核心业务逻辑
        
        参数:
            signal: 策略信号
            cls: 策略类名
            
        返回:
            ExecutionContext: 执行上下文
        """
        logger.debug(f"Portfolio 处理信号: {signal.signal_id}")
        # 保存信号
        self.signals[signal.signal_id] = signal
        # 评估信号可执行性
        execution_assets = self._evaluate_signal(signal)
        logger.info(f"处理策略信号: {signal} -> {execution_assets}")
        if not execution_assets:
            logger.warning(f"信号无法执行: {signal.signal_id}")
            self.exec_update(signal.signal_id)
            return
        
        # 创建执行上下文
        execution_context = ExecutionContext(
            context_id=generate_str(),
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            strategy_instant_id=signal.strategy_instance_id,
            target_executor_id=signal.config.executor_id if signal.config else None,
            execution_assets=execution_assets,
            created_time=signal.signal_time,
            leverage=signal.config.leverage if signal.config and signal.config.leverage else self.position_config.default_leverage,
            order_type=signal.config.order_type if signal.config and signal.config.order_type else self.position_config.order_type,
            trade_type=self.position_config.trade_type,
            trade_mode=self.position_config.trade_mode,
            strategy_cls=cls,
        )
        
        # 执行风控检查
        if not self._do_risk_policy(execution_context):
            # 风控通过，冻结资金
            if not self.capital_account.freeze_amount(execution_context):
                logger.warning(f"Portfolio 冻结资金失败: {execution_context.signal_id}")
                execution_context.is_finish = True
                self.exec_update(execution_context.signal_id)
                return
        
        # 发布执行事件
        logger.info(f"Portfolio 信号处理完成: {signal.signal_id}, {execution_context}")
        self.event_bus.publish_event(Event(
            event_type=EventType.EXEC_ORDER_CREATED,
            data=execution_context
        ))

    def exec_update(self, execution_context: ExecutionContext | str):
        if isinstance(execution_context, str):
            signal = self.signals.pop(execution_context, None)
            logger.error(f"信号分配资金失败: {signal}")
            if not signal:
                return
            for signal_asset in signal.assets:
                signal_asset.actual_ratio = 0
        else:
            logger.info(f"执行订单更新: {execution_context.is_finish}, {execution_context}")
            if not execution_context.is_finish:
                return
            signal = self.signals.pop(execution_context.signal_id, None)
            if not signal:
                return
            for signal_asset in signal.assets:
                signal_asset.actual_ratio = signal_asset.actual_ratio or 0
                for asset in execution_context.execution_assets:
                    if asset.asset_key == signal_asset.asset_key and asset.is_open == signal_asset.is_open:
                        signal_asset.actual_ratio += asset.ratio
        self.event_bus.publish_event(Event(
            event_type=EventType.STRATEGY_SIGNAL_FINISH,
            data=signal
        ))
    
    def _evaluate_signal(self, signal: Signal) -> List[ExecutionAsset]:
        """
        评估信号可执行性
        
        参数:
            signal: 策略信号
            
        返回:
            List[ExecutionAsset]: 执行资产列表
        """
        if not signal.assets:
            return []
        
        available_balance = self.capital_account.available_balance
        if available_balance <= 0:
            logger.warning("可用资金不足")
            return []
        
        # 获取策略当前使用的资金和比例
        strategy_used_principal, strategy_used_ratio = self.position_tracker.get_strategy_used(signal.strategy_id)
        
        # 计算策略可用比例
        strategy_ratio = min(self.position_config.max_strategy_ratio - strategy_used_ratio, self.position_config.max_ratio)
        
        # 计算本次可投入本金
        principal = decimal_quantize(self.total_amount * strategy_ratio, 8)
        principal = min(
            self.position_config.max_amount, 
            principal, 
            self.position_config.max_strategy_amount - strategy_used_principal, 
            available_balance
        )
        
        # 如果信号配置了本金，则使用信号配置的本金取小
        if signal.config and signal.config.principal:
            principal = min(signal.config.principal, principal)
        
        ratios = []
        max_amounts = []
        for asset in signal.assets:
            if not asset.is_open:
                continue
            ratios.append(asset.ratio)
            symbol_used_principal, symbol_used_ratio = self.position_tracker.get_symbol_used(asset.symbol, asset.quote_currency)
            symbol_amount = self.position_config.max_symbol_amount - symbol_used_principal

            symbol_amount = min(symbol_amount, decimal_quantize(self.total_amount * (self.position_config.max_symbol_ratio - symbol_used_ratio), 8))
            if symbol_amount <= 0:
                symbol_amount = 0
            elif principal * asset.ratio > symbol_amount:
                principal = symbol_amount / asset.ratio
            max_amounts.append(symbol_amount)
        if sum(ratios) > 1:
            logger.error(f"信号 {signal.signal_id} 的资产比例之和大于1，无法计算单位金额")
            return None
        just_close = any(amount == 0 for amount in max_amounts)
        if just_close and len(signal.assets) > len(max_amounts):
            logger.warning(f"信号 {signal.signal_id} 的资产资金限制，仅减仓")
        # 处理每个资产
        execution_assets = []
        for asset in signal.assets:
            if just_close and not asset.is_open:
                continue
            current_position = self.position_tracker.find_position(
                strategy_id=signal.strategy_id,
                strategy_instance_id=signal.strategy_instance_id,
                symbol=asset.symbol,
                quote_currency=asset.quote_currency,
                ins_type=asset.ins_type,
                asset_type=asset.asset_type
            )
            if current_position:
                current_position = current_position[0]
            elif asset.is_open is False:
                logger.error(f"信号 {signal.signal_id} 的平仓资产 {asset.asset_key} 不存在")
                return None

            execution_asset = ExecutionAsset(
                asset_type=asset.asset_type,
                ins_type=asset.ins_type,
                symbol=asset.symbol,
                side=asset.side,
                price=asset.price,
                ratio=asset.ratio,
                amount=0,
                is_open=asset.is_open,
                quote_currency=asset.quote_currency,
                extra=asset.extra,
            )
            
            execution_assets.append(execution_asset)
            if current_position:
                execution_asset.executor_sz = current_position.executor_sz
                execution_asset.position_id = current_position.position_id
                if not asset.is_open:
                    # 计算总仓位（真实仓位 + 虚拟仓位）
                    total_sz = current_position.sz + sum(vp.sz for vp in current_position.virtual_positions)
                    total_ratio = current_position.ratio + sum(vp.ratio for vp in current_position.virtual_positions)
                    
                    close_ratio = min(1, asset.ratio / total_ratio)
                    close_sz = total_sz * close_ratio
                    virtual_sz = sum(vp.sz for vp in current_position.virtual_positions)
                    execution_asset.virtual_sz = min(virtual_sz, close_sz)
                    execution_asset.sz = max(0, close_sz - virtual_sz)
                    execution_asset.ratio = total_ratio * close_ratio if execution_asset.sz < total_sz else total_ratio
                    execution_asset.amount = decimal_quantize(current_position.amount * close_ratio, 8)
                    continue
            
            execution_asset.is_open = True
            execution_asset.amount = decimal_quantize(principal * asset.ratio, 8)
            execution_asset.ratio = decimal_quantize(execution_asset.amount / self.total_amount, 8)
        return execution_assets
    
    def _do_risk_policy(self, execution_context: ExecutionContext) -> bool:
        """
        执行风控策略检查
        
        参数:
            execution_context: 执行上下文
            
        返回:
            bool: True表示风控拦截，False表示风控通过
        """
        # 构建仓位信息上下文
        position_info = PositionInfo(
            positions=list(self.position_tracker.positions.values())
        )
        
        # 委托给风控管理器执行检查
        return self.risk_manager.evaluate_risk(execution_context, position_info)
    
    def update_config(self, config: PositionConfig):
        """
        更新投资组合配置
        
        参数:
            config: 新的配置
        """
        # 计算资金变化
        delta_amount = config.init_amount - self.position_config.init_amount
        if delta_amount != 0:
            self.capital_account.change_amount(delta_amount, f"修改初始化资金: {self.position_config.init_amount} -> {config.init_amount}")
        
        # 更新配置
        self.config.config = config
        self.position_config = config
        
        logger.info(f"Portfolio 配置更新完成: 初始资金 {self.position_config.init_amount}")
    
    @thread_lock(_portfolio_lock)
    def order_update(self, order: Order):
        """
        处理订单更新
        
        参数:
            order: 订单对象
        """
        logger.info(f"仓位处理收到订单更新: {order}， finish={order.order_status.is_finished}")
        if order.order_status == OrderStatus.SUBMITTED or order.order_status == OrderStatus.CREATED:
            return
        if order.order_status.is_finished:
            logger.info(f"仓位处理收到订单更新: {order.order_id}, 解冻资金 {order.is_fake} {order.is_open}")
            self.capital_account.unfreeze_amount(order)
        
        if order.order_status.is_failed:
            if not order.position_id:
                return
        delta_amount, delta_fee, delta_friction, delta_pnl, delta_sz = self.position_tracker.get_order_change(order)
        if delta_amount is None: # 先发出的信息后到 不处理
            return
        
        pnl = self.position_tracker.order_update(order, delta_amount, delta_fee, delta_friction, delta_sz)
        if order.is_fake:
            self.virtual_pnl += pnl
        else:
            self.pnl += delta_pnl
            self.friction += delta_friction
            self.fee += delta_fee
    
    def reset_stats(self):
        """重置统计数据"""
        # 重置资金账户
        self.capital_account.reset()
        # 重置仓位跟踪器
        self.position_tracker.reset()
        # 重置信号
        self.signals = {}
        self.pnl = Decimal(0)
        self.fee = Decimal(0)
        self.friction = Decimal(0)
        self.virtual_pnl = Decimal(0)
        logger.info("Portfolio 重置状态完成")
    
    def load_state(self, state: dict):
        """
        加载状态 - 参考原始 PositionContext.load_state 实现
        
        参数:
            state: 状态数据
        """
        if not state:
            return

        # 处理重置标志
        if state.get("reset_position_state", False):
            self.reset_stats()
            return

        # 加载统计数据
        self.pnl = Decimal(state.get('pnl', self.pnl))
        self.friction = Decimal(state.get('friction', self.friction))
        self.fee = Decimal(state.get('fee', self.fee))
        self.virtual_pnl = Decimal(state.get('virtual_pnl', self.virtual_pnl))
        
        # 加载资金状态
        self.capital_account.load_state(state.get('capital', None))
        # 加载仓位状态
        self.position_tracker.load_state(state.get('position', None))
        # 加载风控状态
        self.risk_manager.load_state(state.get('risk', None))

        # 加载信号
        signals_data = state.get('signals', [])
        for signal in signals_data:
            if isinstance(signal, dict):
                signal = Signal(**signal)
            self.signals[signal.signal_id] = signal
        logger.info("Portfolio 状态加载完成")
    
    def get_state(self) -> dict:
        """
        获取当前状态 - 包含所有重要的统计数据
        
        返回:
            dict: 状态数据
        """
        return {
            # 资金相关
            'activate_amount': str(self.available_amount),  # 可用资金（对应原 activate_amount）
            'total_amount': str(self.total_amount),  # 总资产
            'total_value': str(self.total_value),  # 总价值
            'available_amount': str(self.available_amount),  # 可用资金
            'position_value': str(self.position_value),  # 仓位价值
            
            # 盈亏统计
            'pnl': str(self.pnl),  # 总盈亏
            'friction': str(self.friction),  # 摩擦成本
            'fee': str(self.fee),  # 手续费
            'virtual_pnl': str(self.virtual_pnl),  # 虚拟盈亏
            'profit': str(self.profit),  # 总收益
            
            # 仓位统计
            'asset_count': len(set((pos.symbol, pos.quote_currency) for pos in self.position_tracker.positions.values())),  # 资产数量
            'position_count': len(self.position_tracker.positions),  # 仓位数量
            'positions': list(self.position_tracker.positions.values()),  # 仓位列表
            'signals': self.signals,  # 信号

            # 详细信息
            'capital': self.capital_account.get_state(),  # 资金详情
            'position': self.position_tracker.get_state(),  # 仓位详情
            'risk': self.risk_manager.get_state(),  # 风控详情
        }
