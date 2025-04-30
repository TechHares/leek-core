#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import unittest
from datetime import datetime
from decimal import Decimal
from threading import Thread
from typing import ClassVar, List

from engine import *
from executor import BacktestExecutor
from models import Data, Position, Field, FieldType, KLine, create_instance, InstanceInitConfig, Signal, \
    PositionContext, PositionSide, ChoiceType
from position import Policy
from risk import RiskPlugin
from utils import EventBus, get_logger, Event, EventType

logger = get_logger(__name__)

class FixedRateRiskPlugin(RiskPlugin):
    # 参数
    init_params: ClassVar[List[Field]] = [
        Field(
            name="rate",
            label="亏损率",
            type=FieldType.FLOAT,
            default=0.05,
            description="风险率，0.05=5%",
            required=True,
            min=0.0,
            max=1.0,
        )
    ]
    init_params += RiskPlugin.init_params
    """测试风控插件"""
    def __init__(self, instance_id: str, name: str, rate: Decimal=Decimal("0.05")):
        super().__init__(instance_id, name)
        self.risk_rate = -abs(rate)  # 风险率，5%

    def trigger(self, position: Position, data: KLine) -> bool:
        """
        触发风控
        :param position: 仓位
        :param data: 市场数据
        :return: 是否触发风控
        """
        pnl_ratio = data.close_price / position.cost_price - 1
        if position.side.is_short:
            pnl_ratio *= -1
        return  pnl_ratio <= self.risk_rate

class SidePolicy(Policy):
    # 参数
    init_params = Policy.init_params + [
        Field(
            name="side",
            label="方向",
            type=FieldType.RADIO,
            description="限制仓位方向",
            required=True,
            min=1,
            max=2,
            choices=[(1, "多"), (2, "空")],
            choice_type=ChoiceType.INT,
        )
    ]
    """测试风控插件"""
    def __init__(self, instance_id: str, name: str, side: PositionSide, start_time: datetime = None, end_time: datetime = None):
        super().__init__(instance_id, name, start_time, end_time)
        self.side = PositionSide(side)

    def _check(self, signal: Signal, context: PositionContext) -> str:
        """
        检查信号是否符合风控规则
        :param signal: 信号
        :param context: 上下文
        :return: 拒绝原因
        """
        return "" if signal.side == self.side else f"方向不匹配，当前方向：{signal.side.name}"

class TestEngine(unittest.TestCase):
    """测试K线策略上下文"""

    def setUp(self):
        cfg = {
            "instance_id": "test_engine",
            "name": "测试引擎",
        }
        instance_id = cfg.get("instance_id")
        name = cfg.get("name")

        event_bus = EventBus()
        def on_event(event: Event):
            logger.info(f"收到事件: {event.event_type} -> {event}")
        event_bus.subscribe_event(None, on_event)

        risk_manager = RiskManager(instance_id, name, event_bus)
        executor_manager = ExecutorManager(instance_id, name, event_bus)
        position_manager = PositionManager(instance_id, name, event_bus)
        strategy_manager = StrategyManager(instance_id, name, event_bus)
        data_manager = DataManager(instance_id, name, event_bus)


        self.engine = Engine(instance_id, name, event_bus, data_manager, strategy_manager, position_manager, risk_manager, executor_manager)
        self.engine.load_state({})
        # 1. 添加一个风控插件
        self.engine.add_risk_plugin(InstanceInitConfig(
            cls=FixedRateRiskPlugin,
            config={
                "rate": Decimal("0.05"),
            })
        )
        # 2. 添加一个执行器
        self.engine.add_executor(BacktestExecutor(
            instance_id="backtest_executor",
            name="回测执行器",
        ), None)

        Thread(target=self.engine.on_start).start()
        # self.engine.start()

    def tearDown(self):
        self.engine.on_stop()

    def test_engine(self):
        time.sleep(20)
        self.engine.on_stop()
        time.sleep(5)
        self.assertFalse(self.engine.running)


if __name__ == '__main__':
    unittest.main()