#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal
from threading import RLock
from typing import Dict, List
from datetime import datetime

from leek_core.base import LeekComponent
from leek_core.event import EventBus, Event, EventType, EventSource
from leek_core.models import Position, Data, DataType, KLine, TradeInsType, AssetType, Order, OrderStatus, VirtualPosition, OrderExecutionState
from leek_core.utils import get_logger, thread_lock
from leek_core.utils import generate_str

logger = get_logger(__name__)
_position_lock = RLock()


class PositionTracker(LeekComponent):
    """
    仓位跟踪器组件 - 纯仓位跟踪
    
    职责：
    - 仓位状态跟踪
    - 仓位数据查询
    - 仓位更新通知
    - 仓位统计计算
    """
    
    def __init__(self, event_bus: EventBus):
        """
        初始化仓位跟踪器
        
        参数:
            event_bus: 事件总线
        """
        super().__init__()
        self.event_bus = event_bus
        
        # 仓位存储
        self.positions: Dict[str, Position] = {}  # 仓位字典，键为仓位ID
    
    def find_position(self, position_id: str=None, strategy_id: str=None, strategy_instance_id: str=None, 
                            symbol: str=None, quote_currency: str=None, ins_type: TradeInsType=None, asset_type: AssetType=None) -> List[Position]:
        """
        获取仓位
        
        参数:
            position_id: 仓位ID
            strategy_id: 策略ID
            strategy_instance_id: 策略实例ID
            symbol: 交易对
            quote_currency: 计价货币
            ins_type: 交易类型
            asset_type: 资产类型

        返回:
            List[Position]: 仓位对象列表
        """
        res = []
        for position in list(self.positions.values()):
            if position_id and position.position_id != position_id:
                continue
            if strategy_id and position.strategy_id != strategy_id:
                continue
            if strategy_instance_id and position.strategy_instance_id != strategy_instance_id:
                continue
            if symbol and position.symbol != symbol:
                continue
            if quote_currency and position.quote_currency != quote_currency:
                continue
            if ins_type and position.ins_type != ins_type:
                continue
            if asset_type and position.asset_type != asset_type:
                continue
            if position.is_closed:
                self.positions.pop(position.position_id, None)
                continue
            res.append(position)
        return res
    
    def get_strategy_used(self, strategy_id: str) -> Decimal:
        """
        获取策略已使用本金, 比例
        
        参数:
            strategy_id: 策略ID
        """
        postions = self.find_position(strategy_id=strategy_id)
        if not postions:
            return Decimal(0), Decimal(0)
        total_amount = sum(pos.amount + pos.pnl for pos in postions)
        total_ratio = sum(pos.ratio for pos in postions)
        return total_amount, total_ratio
    
    def get_symbol_used(self, symbol: str, quote_currency: str) -> Decimal:
        """
        获取标的已使用本金, 比例
        
        参数:
            symbol: 交易对
            quote_currency: 计价货币
        """
        postions = self.find_position(symbol=symbol, quote_currency=quote_currency)
        if not postions:
            return Decimal(0), Decimal(0)
        total_amount = sum(pos.amount + pos.pnl for pos in postions)
        total_ratio = sum(pos.ratio for pos in postions)
        return total_amount, total_ratio
    
    def get_total_position_value(self) -> Decimal:
        """
        获取总仓位价值
        
        返回:
            Decimal: 总价值
        """
        if not self.positions:
            return Decimal('0')
        return sum(pos.value for pos in list(self.positions.values()))

    
    def on_data(self, data: Data):
        """
        处理数据更新
        
        参数:
            data: 数据对象
        """
        if data.data_type == DataType.KLINE and isinstance(data, KLine):
            positions = self.find_position(symbol=data.symbol, quote_currency=data.quote_currency)
            for position in positions:
                position.current_price = data.close
    
    @thread_lock(_position_lock)
    def order_update(self, order: Order, delta_amount: Decimal, delta_fee: Decimal, delta_friction: Decimal, delta_sz: Decimal):
        """
        处理订单更新
        
        参数:
            order: 订单对象
        """
        position = self._get_position_by_order(order)
        logger.info(f"仓位处理订单更新原仓位: {position}, 订单: {order}")
        try:
            pnl = Decimal(0)
            if order.is_fake:
                pnl = self._update_virtual_position(position, order)
            else:
                self._update_real_position(position, order, delta_amount, delta_fee, delta_friction, delta_sz)
            logger.info(f"仓位更新完成, 总价值: {self.get_total_position_value()}, 总仓位: {len(self.positions)}, 虚拟仓位: {order.is_fake}")
            return pnl
        except Exception as e:
            logger.error(f"更新仓位信息失败: {e}", exc_info=True)
        finally:
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_UPDATE,
                data=position
            ))
            if position.is_closed:
                self.positions.pop(position.position_id, None)
    
    def get_order_change(self, order: Order) -> (Decimal, Decimal, Decimal, Decimal, Decimal):
        """
        获取订单执行状态
        
        参数:
            order: 订单对象
        """
        position = self._get_position_by_order(order)
        order_state = position.order_states.get(order.order_id)
        if not order_state:
            order_state = OrderExecutionState(order_id=order.order_id, is_open=order.is_open)
            position.order_states[order.order_id] = order_state
        # 计算本次更新的变化量
        delta_amount = (order.settle_amount or Decimal('0')) - order_state.settle_amount
        delta_fee = (order.fee or Decimal('0')) - order_state.fee
        delta_friction = (order.friction or Decimal('0')) - order_state.friction
        delta_pnl = (order.pnl or Decimal('0')) - order_state.pnl
        delta_sz = (order.sz or Decimal('0')) - order_state.sz
        if delta_sz < 0: # 先发出的信息后到 不处理
            return None, None, None, None, None

        # 更新订单执行状态
        order_state.settle_amount = order.settle_amount or Decimal('0')
        order_state.fee = order.fee or Decimal('0')
        order_state.friction = order.friction or Decimal('0')
        order_state.pnl = order.pnl or Decimal('0')
        order_state.sz = order.sz or Decimal('0')
        return delta_amount, delta_fee, delta_friction, delta_pnl, delta_sz
    
    def _update_virtual_position(self, position: Position, order: Order):
        """
        更新虚拟仓位信息
        
        参数:
            position: 仓位对象
            order: 订单对象
        """
        if not order.order_status.is_finished:
            return 0
        if order.is_open: # 开虚拟仓位
            virtual_position = VirtualPosition(
                policy_id=order.extra.get("policy_id") if order.extra else None,
                signal_id=order.signal_id,
                amount=order.settle_amount,
                ratio=order.ratio,
                cost_price=order.execution_price,
                sz=order.sz,
                timestamp=int(order.order_time.timestamp() * 1000)
            )
            position.virtual_positions.append(virtual_position)
            return 0
        # 平虚拟仓位，根据订单的virtual_sz来平仓
        pnl = Decimal(0)
        if position.virtual_positions:
            # 按时间戳排序，优先平最近的
            position.virtual_positions.sort(key=lambda x: x.timestamp, reverse=True)
            remaining_close_sz = abs(order.sz)
            for virtual_pos in position.virtual_positions:
                if remaining_close_sz <= 0:
                    break
                
                if virtual_pos.sz <= 0:
                    continue
                
                close_sz = min(remaining_close_sz, virtual_pos.sz)
                virtual_pos.closed_sz += close_sz
                virtual_pos.sz -= close_sz
                
                # 计算加权平均平仓价格
                if virtual_pos.closed_sz > 0:
                    if virtual_pos.closed_sz == close_sz:
                        # 第一次平仓，直接使用执行价格
                        virtual_pos.close_price = order.execution_price
                    else:
                        # 多次平仓，计算加权平均价格
                        virtual_pos.close_price = (virtual_pos.close_price * (virtual_pos.closed_sz - close_sz) + order.execution_price * close_sz) / virtual_pos.closed_sz
                
                # 计算虚拟仓位的盈亏
                if position.side.is_long:
                    virtual_pos.pnl = (virtual_pos.close_price - virtual_pos.cost_price) * close_sz
                else:
                    virtual_pos.pnl = (virtual_pos.cost_price - virtual_pos.close_price) * close_sz
                
                remaining_close_sz -= close_sz
                pnl += virtual_pos.pnl
        # 返回虚拟盈亏
        return pnl
    
    def _update_real_position(self, position: Position, order: Order, delta_amount: Decimal,
                                 delta_fee: Decimal, delta_friction: Decimal, delta_sz: Decimal):
        """
        更新真实仓位信息
        
        参数:
            position: 仓位对象
            order: 订单对象
            delta_amount: 变化量
            delta_fee: 手续费
            delta_friction: 滑点
            delta_sz: 数量
        """
        position.current_price = order.execution_price
        position.friction += delta_friction
        position.fee += delta_fee
        position.pnl += delta_friction + delta_fee
        
        if order.is_open: # 如果是开仓订单
            position.executor_sz[order.executor_id] = position.executor_sz.get(order.executor_id, Decimal('0')) + delta_sz
            position.total_amount += delta_amount
            position.total_sz += delta_sz
            if position.total_sz > 0:
                position.cost_price = order.leverage * position.total_amount / position.total_sz
            if order.order_status == OrderStatus.FILLED:
                position.ratio += order.ratio
        else:
            position.total_back_amount += delta_amount
            position.executor_sz[order.executor_id] = position.executor_sz.get(order.executor_id, Decimal('0')) - delta_sz
            position.pnl += (1 if position.side.is_long else -1) * (order.execution_price - position.cost_price) * delta_sz
            closed_sz = (position.total_sz - position.sz)
            if closed_sz > 0:
                position.close_price = position.total_back_amount * position.leverage / closed_sz
                if position.side.is_short:
                    position.close_price = 2 * position.cost_price - position.close_price
            if order.order_status == OrderStatus.FILLED:
                position.ratio -= order.ratio
        
        position.amount = position.sz * position.cost_price / position.leverage
        return 0

    def _get_position_by_order(self, order: Order) -> Position:
        """
        处理订单更新
        
        参数:
            order: 订单对象
        """
        if order.is_open and not order.position_id: # 兼容下前面消息堆积的情况
            positions = self.find_position(strategy_id=order.strategy_id, strategy_instance_id=order.strategy_instance_id)
            for position in positions:
                if position.order_states and order.order_id in position.order_states:
                    order.position_id = position.position_id
                    break
        # 如果是开仓订单且没有position_id，创建新仓位
        if order.is_open and not order.position_id:
            position = Position(
                position_id=generate_str(),
                strategy_id=order.strategy_id,
                strategy_instance_id=order.strategy_instance_id,
                symbol=order.symbol,
                quote_currency=order.quote_currency,
                ins_type=order.ins_type,
                asset_type=order.asset_type,
                side=order.side,
                cost_price=0,
                amount=0,
                ratio=0,
                current_price=order.execution_price,
                executor_id=order.executor_id,
                fee=0,
                friction=0,
                leverage=order.leverage,
                open_time=order.order_time
            )
            order.position_id = position.position_id
            self.positions[position.position_id] = position
            self.event_bus.publish_event(Event(
                event_type=EventType.POSITION_INIT,
                data=order
            ))
        return self.positions.get(order.position_id)
    
    def load_state(self, state: dict):
        """
        加载仓位状态
        
        参数:
            state: 状态数据
        """
        if not state:
            return
        
        # 清空现有数据
        self.positions.clear()
        # 加载仓位状态
        for p in state.get('positions', []):
            p = Position(**p) if isinstance(p, dict) else p
            self.positions[p.position_id] = p
        logger.info(f"PositionTracker 状态加载完成，加载仓位数量: {len(self.positions)}")

    def get_state(self) -> dict:
        """
        获取当前状态
        
        返回:
            dict: 状态数据
        """ 
        return {
            'positions': list(self.positions.values()),
            'position_count': len(self.positions),
            "asset_count": len(set((pos.symbol, pos.quote_currency) for pos in self.positions.values())),
            'total_value': str(self.get_total_position_value())
        }
    
    def reset(self):
        """重置仓位跟踪器"""
        self.positions.clear()
        logger.info("PositionTracker 重置完成")