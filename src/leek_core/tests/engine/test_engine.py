#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque

from leek_core.executor import OkxWebSocketExecutor
from leek_core.info_fabricator import KLineFillFabricator, DataThrottleFabricator, KlineInitFabricator
from leek_core.models import StrategyPositionConfig, OrderType, PositionConfig
from leek_core.policy import PositionStopLoss
from leek_core.strategy import CTAStrategy
from leek_core.strategy import StrategyCommand
from leek_core.sub_strategy import EnterStrategy, ExitStrategy
from leek_core.utils import get_logger, setup_logging

setup_logging(use_colors=True, level="INFO")
import time
import unittest
from decimal import Decimal
from threading import Thread
from typing import List

from leek_core.data import DataSourceContext, OkxDataSource
from leek_core.engine import *
from leek_core.event import EventBus, Event
from leek_core.models import LeekComponentConfig, BacktestEngineConfig, StrategyConfig, KLine, PositionSide, Field, FieldType
import os
logger = get_logger(__name__)


class CTAStrategyTest(CTAStrategy):
    # 策略展示名称
    display_name: str = "未命名策略"

    # 参数
    init_params: List[Field] = [
        Field(
            name="period",
            label="移动平均线周期",
            type=FieldType.INT,
            default=10,
            required=True,
            description="移动平均线的周期"
        )
    ]
    """
    简单的CTA策略
    """

    def __init__(self, period: int = 10):
        super().__init__()
        self.period = period
        self.kline_buffer = deque(maxlen=period)
        self.ma = None

    def on_kline(self, kline: KLine):
        logger.info(f"K线数据: {kline}")
        self.ma = None
        if len(self.kline_buffer) < self.period:
            self.kline_buffer.append(kline)
            return
        ma = sum(k.close for k in self.kline_buffer)
        self.ma = kline.close > ma

    def should_open(self) -> PositionSide | StrategyCommand:
        if self.ma:
            return PositionSide.LONG

    def should_close(self, position_side: PositionSide) -> bool | Decimal:
        if self.ma is False:
            return True


class TestEngine(unittest.TestCase):
    """测试K线策略上下文"""

    def setUp(self):
        engine_cfg = LeekComponentConfig(instance_id="test_engine", name="测试引擎", cls=None, config=None)

        event_bus = EventBus()

        def on_event(event: Event):
            logger.debug(f"收到事件: {event.event_type} -> {event}")

        event_bus.subscribe_event(None, on_event)

        cfg = BacktestEngineConfig(
            data_sources=LeekComponentConfig(
                instance_id="data_manager",
                name="数据管理",
                cls=DataSourceContext,
                config=[
                    LeekComponentConfig(
                        instance_id="1",
                        name="okx测试",
                        cls=OkxDataSource,
                        config={
                            "symbols": ["BTC", "ETH"],
                        }
                    )
                ]
            ),
            strategy_configs=[LeekComponentConfig(
                instance_id="1",
                name="测试策略",
                cls=CTAStrategyTest,
                config=StrategyConfig(
                    data_source_configs=[
                        LeekComponentConfig(
                            instance_id="1",
                            cls=OkxDataSource,
                            config={
                                "symbols": ["BTC"],
                                "timeframes": ["1m"],
                                "ins_types": [3],
                                "quote_currencies": ["USDT"],
                            }
                        )
                    ],
                    strategy_config={
                        "period": 20
                    },
                    strategy_position_config=StrategyPositionConfig(
                        principal=Decimal("20"),
                        leverage=Decimal("1"),
                        order_type=OrderType.MarketOrder,
                        executor_id="1"
                    ),
                    enter_strategy_cls=EnterStrategy,
                    enter_strategy_config={},
                    exit_strategy_cls=ExitStrategy,
                    exit_strategy_config={},

                    risk_policies=[
                        LeekComponentConfig(
                            instance_id="1",
                            name="测试风险策略",
                            cls=PositionStopLoss,
                            config={
                                "stop_loss_ratio": "0.05",
                            }
                        )
                    ],
                    info_fabricator_configs=[
                        LeekComponentConfig(
                            cls=KLineFillFabricator,
                            config={}
                        ),
                        LeekComponentConfig(
                            cls=KlineInitFabricator,
                            config={
                                "num": 20,
                            }
                        ),
                        LeekComponentConfig(
                            cls=DataThrottleFabricator,
                            config={
                                "price_change_ratio": "0.01",
                                "time_interval": 12,
                            }
                        )
                    ]
                ))
            ],
            position_config=PositionConfig(
                init_amount=Decimal("10000"),
                max_strategy_amount=Decimal("10000"),
                max_strategy_ratio=Decimal("0.1"),
                max_symbol_amount=Decimal("10000"),
                max_symbol_ratio=Decimal("0.1"),
                max_amount=Decimal("10000"),
                max_ratio=Decimal("0.1"),
                risk_policies=[]
            ),
            executor_configs=[
                LeekComponentConfig(
                    instance_id="1",
                    name="测试执行器",
                    cls=OkxWebSocketExecutor,
                    config={
                        "slippage_level": "10",
                        "td_mode": "isolated",
                        "ccy": "USDT",
                        "lever": "3",
                        "heartbeat_interval": "25",
                        "order_type": "market",
                        "trade_ins_type": "3",
                    }
                )
            ]
        )
        engine_cfg.config = cfg
        self.engine = SimpleEngine(event_bus, engine_cfg)

        Thread(target=self.engine.on_start).start()
        # self.engine.start()

    def test_engine(self):
        time.sleep(130)
        self.engine.on_stop()
        self.assertFalse(self.engine.running)


if __name__ == '__main__':
    unittest.main()
