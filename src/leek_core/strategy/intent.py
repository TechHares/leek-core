#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略意图 diff 引擎与挂单索引工具。

本模块为 StrategyWrapper 内部使用,不对策略开发者暴露。

主要职责:
1. PendingOrderIndex: 维护当前策略实例的未成交挂单, 支持按 order_id / tag / extra 业务键查询
2. intent_key:        把 StrategyCommand.extra 序列化为业务键, 用于 diff 引擎匹配
3. diff_intents:      比对"新意图列表"与"现有挂单", 产出 IntentPatch (要撤的单 + 要新挂的单)
4. match_cancel:      解析 CancelCommand 找出实际要撤的挂单列表
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from leek_core.models import Order, OrderType, PositionSide


# 与 StrategyCommand / CancelCommand 解耦的轻量类型(避免循环引用)
# 使用时由 StrategyWrapper 把 StrategyCommand 转 dict 传入


def intent_key(extra: Optional[Dict[str, Any]]) -> str:
    """
    把 extra 序列化为业务键, 用于 diff 引擎匹配现有挂单与新意图。

    优先级:
        1. extra 为空 → "__default__"     (只允许 1 笔默认意图)
        2. "tag" in extra → "tag=<value>"
        3. 否则 → 全部键值排序拼接 "k1=v1;k2=v2;..."

    保留的内部键(_expire_bars / _submit_bar / _expired / policy_id)
    不参与业务键计算。
    """
    if not extra:
        return "__default__"
    # 过滤内部保留键
    business = {k: v for k, v in extra.items() if not k.startswith("_") and k != "policy_id"}
    if not business:
        return "__default__"
    if "tag" in business:
        return f"tag={business['tag']}"
    return ";".join(f"{k}={v}" for k, v in sorted(business.items()))


def intents_match(order: Order, cmd_side: PositionSide, cmd_ratio: Decimal,
                  cmd_order_type: Optional[OrderType], cmd_price: Optional[Decimal]) -> bool:
    """
    判断现有挂单是否与新意图字段完全一致(无需改单)。
    """
    if order.side != cmd_side:
        return False
    target_type = cmd_order_type or OrderType.MarketOrder
    if order.order_type != target_type:
        return False
    if target_type == OrderType.LimitOrder:
        order_price = order.order_price or Decimal(0)
        cmd_price_v = cmd_price or Decimal(0)
        if order_price != cmd_price_v:
            return False
    # ratio 比较: 允许小数误差
    order_ratio = order.ratio or Decimal(0)
    if abs(order_ratio - cmd_ratio) > Decimal("0.0001"):
        return False
    return True


@dataclass
class IntentPatch:
    """diff 引擎产出的差异补丁"""
    to_cancel: List[Order] = field(default_factory=list)         # 要撤的现有挂单
    to_submit: List[Any] = field(default_factory=list)            # 要新挂的 StrategyCommand


class PendingOrderIndex:
    """
    策略实例本地挂单索引。

    特性:
    - 按 order_id 索引: O(1) 查/增/删
    - 按 intent_key (extra 业务键) 索引: 支持 diff 引擎快速匹配
    - 按 tag 索引: 支持 CancelCommand(tag=...) 快速命中

    注意: 这是 wrapper 内嵌的轻量级容器, 不做并发同步(wrapper.lock 保证单线程访问)。
    """

    def __init__(self):
        self._by_id: Dict[str, Order] = {}

    # ---- 基本读写 ----
    def add(self, order: Order) -> None:
        self._by_id[order.order_id] = order

    def remove(self, order_id: str) -> Optional[Order]:
        return self._by_id.pop(order_id, None)

    def get(self, order_id: str) -> Optional[Order]:
        return self._by_id.get(order_id)

    def all(self) -> List[Order]:
        return list(self._by_id.values())

    def __len__(self):
        return len(self._by_id)

    def __contains__(self, order_id: str):
        return order_id in self._by_id

    # ---- 业务查询 ----
    def open_orders(self) -> List[Order]:
        """所有未成交的开仓挂单"""
        return [o for o in self._by_id.values() if o.is_open]

    def close_orders(self) -> List[Order]:
        """所有未成交的平仓挂单"""
        return [o for o in self._by_id.values() if not o.is_open]

    def limit_open_orders(self) -> List[Order]:
        """未成交的限价开仓单(diff 引擎只对限价单生效)"""
        return [o for o in self._by_id.values()
                if o.is_open and o.order_type == OrderType.LimitOrder]

    def limit_close_orders(self) -> List[Order]:
        """未成交的限价平仓单"""
        return [o for o in self._by_id.values()
                if not o.is_open and o.order_type == OrderType.LimitOrder]

    def match_tag(self, tag: str) -> List[Order]:
        """
        按 CancelCommand.tag 规则匹配挂单:
            "entry"          → extra.get("tag") == "entry"
            "grid_level=3"   → str(extra.get("grid_level")) == "3"
            "grid_level"     → "grid_level" in extra
        """
        if "=" in tag:
            key, value = tag.split("=", 1)
            return [
                o for o in self._by_id.values()
                if o.extra and str(o.extra.get(key)) == value
            ]
        # 单独键: 先看是否有 tag 字段直接匹配, 否则按"键存在"匹配
        return [
            o for o in self._by_id.values()
            if o.extra and (o.extra.get("tag") == tag or tag in o.extra)
        ]


def diff_intents(
    existing: List[Order],
    desired: List[Tuple[str, Any]],  # (intent_key, StrategyCommand)
) -> IntentPatch:
    """
    意图 diff 算法。

    参数:
        existing: 现有同方向同类型挂单列表 (通常是 limit_open_orders / limit_close_orders)
        desired:  新意图列表, 每项为 (业务键, StrategyCommand 对象)

    返回:
        IntentPatch(to_cancel, to_submit)
    """
    key_existing: Dict[str, Order] = {}
    for o in existing:
        k = intent_key(o.extra)
        # 同键多个挂单的情况(策略写出 bug): 取最后一个,前面的视为冗余撤掉
        if k in key_existing:
            # 重复: 之前那个移到 to_cancel
            pass
        key_existing[k] = o

    key_desired: Dict[str, Any] = {}
    for k, cmd in desired:
        key_desired[k] = cmd  # 同样取最后(策略 bug)

    patch = IntentPatch()
    # 现有但 desired 里没有 → 撤
    for k, order in key_existing.items():
        if k not in key_desired:
            patch.to_cancel.append(order)
            continue
        cmd = key_desired[k]
        if not intents_match(order, cmd.side, cmd.ratio, cmd.order_type, cmd.price):
            patch.to_cancel.append(order)
            patch.to_submit.append(cmd)
    # desired 里有但现有没有 → 新挂
    for k, cmd in key_desired.items():
        if k not in key_existing:
            patch.to_submit.append(cmd)
    return patch
