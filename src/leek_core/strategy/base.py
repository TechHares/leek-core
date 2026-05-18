#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略基础模块，提供策略的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union

from leek_core.base import LeekComponent
from leek_core.event import Event, EventType
from leek_core.models import Data, DataType, Field, Order, OrderType, Position, PositionSide, Signal
from leek_core.utils import StrategyStateSerializer, get_logger
from .strategy_mode import KlineSimple, StrategyMode

logger = get_logger(__name__)


@dataclass
class StrategyCommand:
    """
    策略意图: 表达"我想要持有的仓位 / 想挂的单"。

    新方案中扩展了挂单字段, 全部带默认值, 向后兼容旧的 StrategyCommand(side, ratio) 调用。

    字段:
        side:        仓位方向 (LONG / SHORT)
        ratio:       仓位比例 (0~1), 相对策略实例配置的最大资金
        order_type:  订单类型 (None 时走 Portfolio 默认配置)
        price:       限价单挂单价 (LimitOrder 必填; MarketOrder 忽略)
        expire_bars: 限价单存活 bar 数 (None=不过期, 仅对 LimitOrder 生效)
        extra:       策略元信息, 用于 diff 引擎匹配 (如 {"tag": "stop_loss", "grid_level": 3})
    """
    side: PositionSide
    ratio: Decimal
    order_type: Optional[OrderType] = None
    price: Optional[Decimal] = None
    expire_bars: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CancelCommand:
    """
    撤单意图: 仅 Tier 2 / Tier 3 callback (on_order_update / on_position_update / on_event)
    返回值可包含此命令。should_open / close 不接受 CancelCommand
    (它们通过返回值差异由 diff 引擎自动撤单)。

    三选一字段 (优先级 order_id > cancel_all > tag):
        order_id:   撤指定订单
        cancel_all: 撤当前策略实例的所有未成交挂单
        tag:        按 order.extra 匹配撤单
                    - tag="entry"          匹配 extra.get("tag") == "entry"
                    - tag="grid_level=3"   匹配 str(extra.get("grid_level")) == "3"
                    - tag="grid_level"     只要 extra 中含此键即匹配
    """
    order_id: Optional[str] = None
    cancel_all: bool = False
    tag: Optional[str] = None


# 类型别名: 所有 callback 的返回值意图均为此联合
StrategyAction = Union[StrategyCommand, CancelCommand]

class Strategy(LeekComponent, ABC):
    """
    择时策略抽象基类，定义策略的基本接口。
    
    类属性:
        display_name: 策略展示名称
        open_just_no_pos: 是否只在没有仓位时开仓，默认为True
        accepted_data_types: 策略接受的数据类型列表，默认接受K线数据
        strategy_mode: 策略运行模式，默认为单标的单时间周期模式
    """
    
    # 策略展示名称
    display_name: str = "未命名策略"
    
    # 是否只在没有仓位时开仓
    open_just_no_pos: bool = True
    
    # 策略接受的数据类型
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    
    # 策略运行模式
    strategy_mode: StrategyMode = KlineSimple()
    # 参数
    init_params: List[Field] = []

    def __init__(self):
        """
        初始化策略
        """
        # 初始化日志器
        # self.position_status: PositionStatus = None
        self.position: Dict[str, Position] = {}
    
    def on_data(self, data: Data = None):
        """
        处理数据，子类可以选择性重写此方法
        
        参数:
            data: 接收到的数据，可以是任何类型
            data_type: 数据类型，如果为None，则由策略自行判断
        """
        ...
    
    def should_open(self) -> Any:
        """
        判断是否应该开仓
        
        返回:
            是否应该开仓 PositionSide 表示全仓开， StrategyCommand 可以自定义比例
        """
        ...

    def close(self, position: Position) -> Any:
        ...

    def after_risk_control(self):
        """
        策略风控策略执行
            触发风控时调用，一般无需特别处理， 但是策略自己管理仓位的话需要重写此方法清空自己仓位管理相关的信息
        """
        ...

    def on_order_update(self, order: Order) -> Optional[List[StrategyAction]]:
        """
        订单进入终态时触发 (FILLED / CANCELED / REJECTED / EXPIRED / ERROR)。
        中间态 (SUBMITTED / PARTIALLY_FILLED) 不触发, 需要时覆写 on_event 兜底。

        典型用途:
            - 入场单成交 → 挂止损单
            - 网格某档成交 → 挂下一档
            - 限价单被拒/过期 → 触发降级逻辑

        返回:
            None                              → 什么也不做 (常见)
            List[StrategyCommand]             → 新挂单 (增量, 不做 diff)
            List[CancelCommand]               → 撤其他挂单
            List[StrategyAction]              → 混合 (先撤后挂)
        """
        return None

    def on_position_update(self, position: Position) -> Optional[List[StrategyAction]]:
        """
        持仓变化时触发。绝大多数策略不需要重写。
        返回意图列表语义同 on_order_update。
        """
        return None

    def on_event(self, event: Event) -> Optional[List[StrategyAction]]:
        """
        订阅原始事件 (包含中间态订单更新等高级场景)。

        默认无操作。仅在需要部分成交细节、风控事件等高级场景时覆写。
        返回意图列表语义同 on_order_update。
        """
        return None



    def load_state(self, state: Dict[str, Any]):
        """
        加载策略状态
        
        参数:
            state: 状态字典，包含field_extra字段记录字段类型信息
        """
        StrategyStateSerializer.deserialize(self, state)    
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取策略状态
        
        返回:
            包含策略状态和字段类型信息的字典
        """
        return StrategyStateSerializer.serialize(self, {field.name for field in self.init_params})
