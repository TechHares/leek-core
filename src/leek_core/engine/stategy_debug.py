#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from leek_core.strategy import StrategyWrapper, Strategy
from leek_core.sub_strategy import EnterStrategy, ExitStrategy
from leek_core.policy import PositionPolicy
from .base import Engine

from leek_core.event import SerializableEventBus, EventType, Event, EventSource
from leek_core.indicators import T, MERGE
from leek_core.data import DataSource, ClickHouseKlineDataSource
from leek_core.models import TimeFrame, TradeInsType, LeekComponentConfig, PositionConfig, Signal, Order, Position
from leek_core.executor import ExecutorContext
from leek_core.manager import PositionManager, ExecutorManager
from leek_core.position import PositionContext
from leek_core.executor import BacktestExecutor
from leek_core.utils import get_logger, DateTimeUtils, decimal_quantize, generate_str

logger = get_logger(__name__)
colors = [
    '#7f7f7f',  # 灰色系 - 中灰
    '#2ca02c',  # 绿色系 - 绿
    '#bcbd22',  # 黄绿色系 - 橄榄绿
    '#17becf'   # 青色系 - 宝石蓝
]

class StrategyDebugView(Engine):
    def __init__(self, strategy: Strategy, enter_strategy: EnterStrategy = EnterStrategy(), exit_strategy: ExitStrategy=ExitStrategy(),
                 policies: List[PositionPolicy]=[],
                 symbol: str = "ETH", start_time: datetime|str = datetime.now() - timedelta(days=10),
                 end_time: datetime|str = datetime.now(),
                 timeframe: TimeFrame = TimeFrame.M5, market: str = "okx", quote_currency: str = "USDT",
                 executor: type[ExecutorContext] = BacktestExecutor,
                 executor_cfg: Dict[str, Any] = {},
                 ins_type: TradeInsType = TradeInsType.SWAP, data_source: DataSource = ClickHouseKlineDataSource()):
        super().__init__()

        self.strategy = StrategyWrapper(strategy, enter_strategy, exit_strategy, policies)
        self.strategy.on_start()
        self.data_source = data_source
        self.symbol = symbol
        self.start_time = start_time
        if isinstance(start_time, str):
            self.start_time = DateTimeUtils.to_datetime(DateTimeUtils.to_timestamp(start_time))
        self.end_time = end_time
        if isinstance(end_time, str):
            self.end_time = DateTimeUtils.to_datetime(DateTimeUtils.to_timestamp(end_time))
        self.timeframe = timeframe
        self.market = market
        self.quote_currency = quote_currency
        self.ins_type = ins_type

        self.color_index = 0
        self.initial_balance = 10000
        self.bechmark = None

        self.event_bus = SerializableEventBus()
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, self.process_order_update)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.on_position_update_event)

        limit_amt = self.initial_balance * 1000
        self.position_context: PositionContext = PositionContext(
            self.event_bus, LeekComponentConfig(
                instance_id="p0",
                name="仓位",
                cls=None,
                config=PositionConfig(init_amount=self.initial_balance, max_amount=limit_amt,
                                      max_strategy_amount=limit_amt, max_strategy_ratio=Decimal("0.5"),
                                      max_symbol_amount=limit_amt, max_symbol_ratio=Decimal("1"), max_ratio=Decimal("1")),
            ))
        self.position_context.on_start()
        self.executor_manager: ExecutorManager = ExecutorManager(
            self.event_bus, LeekComponentConfig(
                instance_id="p1",
                name="执行器管理",
                cls=ExecutorContext,
                config=None
            ))
        self.executor_manager.on_start()
        self.executor_manager.add(LeekComponentConfig(
            instance_id="p3",
            name="执行器",
            cls=executor,
            config=executor_cfg
        ))

    def on_position_update_event(self, event: Event):
        assert isinstance(event.data, Position)
        self.strategy.on_position_update(event.data)

    def process_order_update(self, event: Event):
        assert isinstance(event.data, Order)
        self.position_context.order_update(event.data)


    def get_color(self, color=None):
        if color is not None:
            return color
        color = colors[self.color_index % len(colors)]
        self.color_index += 1
        return color

    def start(self, row=None, custom_draw=None, **kwargs):
        assert self.data_source.connect(), "数据源连接失败"
        count = 0
        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "bechmark": [],
            "sell": [],
            "buy": [],
            "profit": [],
            "time": []
        }

        position_volume = 0
        position_ratio = 0
        custom_key = []
        for kline in self.data_source.get_history_data(start_time=self.start_time, end_time=self.end_time,
                                                       symbol=self.symbol, timeframe=self.timeframe,
                                                       market=self.market, quote_currency=self.quote_currency,
                                                       ins_type=self.ins_type):
            count += 1
            assets = self.strategy.on_data(kline)
            self.position_context.on_data(kline)
            if self.bechmark is None:
                self.bechmark = kline.close
                for k in kline.dynamic_attrs.keys():
                    custom_key.append(k)
                    if k not in data:
                        data[k] = []
                print("========================")
                print(f"自定义KEY: {custom_key}")
                print("========================")
            data["open"].append(kline.open)
            data["high"].append(kline.high)
            data["low"].append(kline.low)
            data["close"].append(kline.close)
            data["bechmark"].append((kline.close - self.bechmark) / self.bechmark * 100)
            data["volume"].append(kline.volume)
            data["time"].append(DateTimeUtils.to_datetime(kline.start_time))
            data["buy"].append(None)
            data["sell"].append(None)
            for k in custom_key:
                data[k].append(kline.dynamic_attrs.get(k, None))
            if assets is not None and len(assets) > 0:
                signal = Signal(
                    signal_id=generate_str(),
                    data_source_instance_id=kline.data_source_id,
                    strategy_id="p1",
                    strategy_instance_id=("debug", ),
                    config=None,
                    signal_time=datetime.now(),
                    assets=assets
                )
                if assets[0].side.is_long:
                    data["buy"][-1] = kline.low * Decimal("0.98")
                if assets[0].side.is_short:
                    data["sell"][-1] = kline.high * Decimal("1.02")
                self.position_context.process_signal(signal)
            data["profit"].append((self.position_context.value - self.initial_balance) / self.initial_balance * 100)
        self.data_source.disconnect()
        logger.info(f"数据执行完成，共{count}条")
        self.draw(data, row, custom_draw, **kwargs)

    def draw(self, data, row=None, custom_draw=None, **kwargs) -> None:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = pd.DataFrame(data)
        fig = make_subplots(rows=row or 3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name=self.symbol),
                      row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['buy'],
            mode='markers+text',
            text="多",
            marker=dict(color='green', size=4)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['sell'],
            mode='markers+text',
            text="空",
            marker=dict(color='red', size=4)
        ), row=1, col=1)

        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume'),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=df['time'], y=df["bechmark"], mode='lines', name="bechmark",
                                 line=dict(color=self.get_color(), width=1)),
                      row=3, col=1)

        fig.add_trace(go.Scatter(x=df['time'], y=df["profit"], mode='lines', name="return",
                                 line=dict(color=self.get_color(), width=2)),
                      row=3, col=1)
        # 设置 x 轴标签格式为百分比
        import numpy as np
        fig.update_xaxes(
            tickvals=np.linspace(-100, 100, 5),  # 设置刻度值
            ticktext=[f"{x}%" for x in np.linspace(-100, 100, 5)],  # 格式化为百分比
            row=3, col=1
        )

        if custom_draw is not None:
            custom_draw(fig, df)
        fig.update_layout(height=kwargs.get("height", 600))
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.show()

