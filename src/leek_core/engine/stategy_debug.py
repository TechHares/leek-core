#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from leek_core.models.data import KLine
from leek_core.strategy import StrategyWrapper, Strategy
 
from leek_core.sub_strategy import SubStrategy
from .base import LeekComponent

from leek_core.event import SerializableEventBus, EventType, Event, EventSource
from leek_core.indicators import T, MERGE
from leek_core.data import DataSource, ClickHouseKlineDataSource
from leek_core.models import TimeFrame, TradeInsType, LeekComponentConfig, PositionConfig, Signal, Order, Position
from leek_core.executor import ExecutorContext
from leek_core.manager import PositionManager, ExecutorManager
from leek_core.position import Portfolio
from leek_core.executor import BacktestExecutor
from leek_core.utils import get_logger, DateTimeUtils, decimal_quantize, generate_str

logger = get_logger(__name__)
colors = [
    '#7f7f7f',  # 灰色系 - 中灰
    '#2ca02c',  # 绿色系 - 绿
    '#bcbd22',  # 黄绿色系 - 橄榄绿
    '#17becf'   # 青色系 - 宝石蓝
]

class StrategyDebugView(LeekComponent):
    def __init__(self, strategy: Strategy,
                 policies: List[SubStrategy]=[],
                 symbol: str = "ETH", start_time: datetime|str = datetime.now() - timedelta(days=10),
                 end_time: datetime|str = datetime.now(),
                 timeframe: TimeFrame = TimeFrame.M5, market: str = "okx", quote_currency: str = "USDT",
                 executor: type[ExecutorContext] = BacktestExecutor,
                 executor_cfg: Dict[str, Any] = {},
                 ins_type: TradeInsType = TradeInsType.SWAP, data_source: DataSource = ClickHouseKlineDataSource()):
        super().__init__()

        self.event_bus = SerializableEventBus()
        self.strategy = StrategyWrapper(self.event_bus, strategy, policies)
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

        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, self.process_order_update)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self.on_position_update_event)

        limit_amt = self.initial_balance * 1000
        self.position_context: Portfolio = Portfolio(
            self.event_bus, LeekComponentConfig(
                instance_id="p0",
                name="仓位",
                cls=None,
                config=PositionConfig(init_amount=self.initial_balance, max_amount=limit_amt,
                                      max_strategy_amount=limit_amt, max_strategy_ratio=Decimal("0.5"),
                                      max_symbol_amount=limit_amt, max_symbol_ratio=Decimal("1"), max_ratio=Decimal("1")),
            ))
        self.position_context.on_start()
        self.strategy.positon_getter = lambda: self.position_context.position_tracker.positions.values()
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
            "open_long": [],
            "close_long": [],
            "open_short": [],
            "close_short": [],
            "profit": [],
            "position": [],
            "time": []
        }

        custom_key = []
        for kline in self.data_source.get_history_data(start_time=self.start_time, end_time=self.end_time,
                                                       row_key=KLine.pack_row_key(self.symbol, self.quote_currency, self.ins_type, self.timeframe),
                                                       market=self.market):
            count += 1
            assets = self.strategy.on_data(kline)
            data["position"].append(self.strategy.position_rate * 100)
            self.position_context.position_tracker.on_data(kline)
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
            data["open_long"].append(None)
            data["close_long"].append(None)
            data["open_short"].append(None)
            data["close_short"].append(None)
            for k in custom_key:
                data[k].append(kline.dynamic_attrs.get(k, None))
            if assets is not None and len(assets) > 0:
                signal = Signal(
                    signal_id=generate_str(),
                    data_source_instance_id=kline.data_source_id,
                    strategy_id="p1",
                    strategy_instance_id="debug",
                    config=None,
                    signal_time=datetime.now(),
                    assets=assets
                )
                if assets[0].side.is_long:
                    if assets[0].is_open:
                        data["open_long"][-1] = kline.low * Decimal("0.98")
                    else:
                        data["close_short"][-1] = kline.high * Decimal("0.98")
                if assets[0].side.is_short:
                    if assets[0].is_open:
                        data["open_short"][-1] = kline.high * Decimal("1.02")
                    else:
                        data["close_long"][-1] = kline.low * Decimal("1.02")
                self.position_context.process_signal(signal)
            data["profit"].append((self.position_context.total_value - self.initial_balance) / self.initial_balance * 100)
        self.data_source.disconnect()
        logger.info(f"数据执行完成，共{count}条")
        self.draw(data, row, custom_draw, **kwargs)

    def draw(self, data, row=None, custom_draw=None, **kwargs) -> None:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = pd.DataFrame(data)
        rows_count = max(row or 2, 2)
        # 动态生成specs，只有第二行支持secondary_y
        specs = []
        for i in range(rows_count):
            if i == 1:  # 第二行（索引为1）
                specs.append([{"secondary_y": True}])
            else:
                specs.append([{"secondary_y": False}])
        fig = make_subplots(rows=rows_count, cols=1, shared_xaxes=True, specs=specs)
        fig.add_trace(go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name=self.symbol), row=1, col=1)
        


        fig.add_trace(go.Scatter(x=df['time'],y=df['open_long'],mode='markers+text',text="多", textposition='bottom center', textfont=dict(family='Courier New', color='green', size=14), marker=dict(color='#bcbd22', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'],y=df['close_long'],mode='markers+text',text="多",textposition='top center', textfont=dict(family='Courier New', color='red', size=14), marker=dict(color='#17becf', size=4)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['time'],y=df['open_short'],mode='markers+text',text="空",textposition='top center', textfont=dict(family='Courier New', color='red', size=14), marker=dict(color='#bcbd22', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'],y=df['close_short'],mode='markers+text',text="空",textposition='bottom center', textfont=dict(family='Courier New', color='green', size=14), marker=dict(color='#17becf', size=4)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['time'], y=df["bechmark"], mode='lines', name="bechmark",
                                 line=dict(color=self.get_color(), width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df["profit"], mode='lines', name="return",
                                 line=dict(color=self.get_color(), width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df["position"], mode='lines', name="position",
                                 line=dict(color=self.get_color(), width=1)), row=2, col=1, secondary_y=True)
        # 设置 x 轴标签格式为百分比
        import numpy as np
        fig.update_xaxes(
            tickvals=np.linspace(-100, 100, 5),  # 设置刻度值
            ticktext=[f"{x}%" for x in np.linspace(-100, 100, 5)],  # 格式化为百分比
            row=2, col=1
        )

        if custom_draw is not None:
            custom_draw(fig, df)
        fig.update_layout(height=kwargs.get("height", 600 + rows_count * 100))
        # 设置右边y轴的标题
        fig.update_yaxes(title_text="Position Rate", secondary_y=True, row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.show()

