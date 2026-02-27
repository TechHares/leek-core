#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from leek_core.models.data import KLine
from leek_core.strategy import StrategyWrapper, Strategy, StrategyContext

from leek_core.sub_strategy import SubStrategy
from leek_core.base import LeekComponent

from leek_core.event import SerializableEventBus, EventType, Event, EventSource
from leek_core.indicators import T, MERGE
from leek_core.data import DataSource, ClickHouseKlineDataSource
from leek_core.models import TimeFrame, TradeInsType, LeekComponentConfig, PositionConfig, Signal, Order, Position, \
    StrategyConfig
from leek_core.executor import ExecutorContext
from leek_core.manager import ExecutorManager
from leek_core.position import Portfolio
from leek_core.executor import BacktestExecutor
from leek_core.utils import get_logger, DateTimeUtils, decimal_quantize, generate_str
from .engine import SimpleEngine

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
                 ins_type: TradeInsType = TradeInsType.SWAP, data_source: DataSource = ClickHouseKlineDataSource(),
                 simulate_kline: bool = False, base_timeframe: TimeFrame = TimeFrame.M1):
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
        self.simulate_kline = simulate_kline
        self.base_timeframe = base_timeframe
        
        # 模拟K线参数校验
        self.merge_window = None
        if self.simulate_kline:
            if self.timeframe.milliseconds is None or self.base_timeframe.milliseconds is None:
                raise ValueError("模拟K线模式不支持 TICK 周期")
            if self.timeframe.milliseconds % self.base_timeframe.milliseconds != 0:
                raise ValueError(f"timeframe({self.timeframe.value})必须能被base_timeframe({self.base_timeframe.value})整除")
            self.merge_window = self.timeframe.milliseconds // self.base_timeframe.milliseconds
            if self.merge_window <= 1:
                raise ValueError(f"timeframe必须大于base_timeframe，当前窗口为{self.merge_window}")
            logger.info(f"模拟K线模式已启用: base={self.base_timeframe.value}, target={self.timeframe.value}, window={self.merge_window}")
        
        self.executor = ExecutorContext(self.event_bus, LeekComponentConfig(
            instance_id="1",
            name="debug",
            cls=executor,
            config=executor_cfg
        ))

        self.color_index = 0
        self.initial_balance = 10000
        self.bechmark = None
        
        # 交易数据收集
        self.trades_data = []
        # 仓位数据收集
        self.positions_data = {}

        limit_amt = self.initial_balance * 1000
        cfg = PositionConfig(init_amount=self.initial_balance, max_amount=limit_amt,
                       max_strategy_amount=limit_amt, max_strategy_ratio=Decimal("0.5"),
                       max_symbol_amount=limit_amt, max_symbol_ratio=Decimal("1"), max_ratio=Decimal("1"))
        self.engine = SimpleEngine("p1", "debug", cfg, 0, self.event_bus)
        self.strategy.positon_getter = self.engine.position_tracker.find_position

    def get_color(self, color=None):
        if color is not None:
            return color
        color = colors[self.color_index % len(colors)]
        self.color_index += 1
        return color
    
    def _on_order_updated(self, event: Event):
        """处理订单更新事件，收集交易数据"""
        if not isinstance(event.data, Order):
            return
        order = event.data
        if order.is_open or not order.order_status.is_finished:
            return
        # 订单pnl需要包含手续费和摩擦成本，与仓位统计保持一致
        pnl_val = float((order.pnl or 0) + (order.fee or 0) + (order.friction or 0))
        
        trade_data = {
            "timestamp": int(order.order_time.timestamp() * 1000),
            "symbol": order.symbol,
            "side": str(order.side.switch()),
            "size": float(order.sz),
            "entry_amount": float(order.order_amount),
            "pnl": pnl_val,
        }
        self.trades_data.append(trade_data)
    
    def _on_position_update(self, event: Event):
        """处理仓位更新事件，收集平仓完成的仓位数据（sz归0）"""
        if not isinstance(event.data, Position):
            return
        pos = event.data
        # 只收集已平仓的仓位数据
        if not pos.is_closed:
            return
        position_data = {
                "timestamp": DateTimeUtils.now_timestamp(),
                "symbol": pos.symbol,
                "side": str(pos.side),
                "pnl": float(pos.pnl or 0),
            }
        self.positions_data[str(pos.position_id)]=position_data
    
    def _print_backtest_metrics(self, equity_values: List[float]):
        """计算并输出回测指标：胜率、盈亏比、区间收益"""
        import numpy as np
        
        # 计算区间收益
        if equity_values and len(equity_values) > 0 and equity_values[0] != 0:
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
        else:
            total_return = 0.0
        
        # 计算最大回撤
        max_drawdown = 0.0
        if equity_values and len(equity_values) > 1:
            peak = equity_values[0]
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100 if peak > 0 else 0.0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # 计算胜率和盈亏比
        total_trades = len(self.trades_data)
        if total_trades == 0:
            win_rate = 0.0
            win_loss_ratio = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        else:
            # 计算胜率
            win_trades = [t for t in self.trades_data if t.get("pnl", 0) > 0]
            loss_trades = [t for t in self.trades_data if t.get("pnl", 0) < 0]
            win_rate = len(win_trades) / total_trades * 100
            
            # 计算盈亏比
            profits = [t["pnl"] for t in win_trades]
            losses = [abs(t["pnl"]) for t in loss_trades]
            
            avg_win = float(np.mean(profits)) if profits else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0.0
        
        # 计算仓位胜率和盈亏比
        total_positions = len(self.positions_data)
        if total_positions == 0:
            position_win_rate = 0.0
            position_win_loss_ratio = 0.0
            position_avg_win = 0.0
            position_avg_loss = 0.0
        else:
            # 计算仓位胜率
            win_positions = [p for p in self.positions_data.values() if p.get("pnl", 0) > 0]
            loss_positions = [p for p in self.positions_data.values() if p.get("pnl", 0) < 0]
            position_win_rate = len(win_positions) / total_positions * 100
            
            # 计算仓位盈亏比
            position_profits = [p["pnl"] for p in win_positions]
            position_losses = [abs(p["pnl"]) for p in loss_positions]
            
            position_avg_win = float(np.mean(position_profits)) if position_profits else 0.0
            position_avg_loss = float(np.mean(position_losses)) if position_losses else 0.0
            position_win_loss_ratio = position_avg_win / position_avg_loss if position_avg_loss > 0 else float('inf') if position_avg_win > 0 else 0.0
        
        # 输出结果
        print("\n" + "=" * 50)
        print("回测统计结果")
        print("=" * 50)
        print("【订单统计】")
        print(f"总交易次数: {total_trades}")
        print(f"胜率: {win_rate:.2f}%")
        if win_loss_ratio == float('inf'):
            print(f"盈亏比: ∞ (无亏损交易)")
        else:
            print(f"盈亏比: {win_loss_ratio:.4f}")
        print(f"平均盈利: {avg_win:.2f}")
        print(f"平均亏损: {avg_loss:.2f}")
        print("\n【仓位统计】")
        print(f"总仓位次数: {total_positions}")
        print(f"仓位胜率: {position_win_rate:.2f}%")
        if position_win_loss_ratio == float('inf'):
            print(f"仓位盈亏比: ∞ (无亏损仓位)")
        else:
            print(f"仓位盈亏比: {position_win_loss_ratio:.4f}")
        print(f"平均盈利仓位: {position_avg_win:.2f}")
        print(f"平均亏损仓位: {position_avg_loss:.2f}")
        print("\n【收益统计】")
        print(f"区间收益: {total_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print("=" * 50 + "\n")

    def start(self, row=None, custom_draw=None, **kwargs):
        self.engine.on_start()
        self.executor.on_start()
        self.data_source.on_start()
        self.engine.executor_manager.components["1"]=self.executor
        
        # 订阅订单更新事件
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, self._on_order_updated)
        # 订阅仓位更新事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self._on_position_update)
        
        # 重置交易数据
        self.trades_data = []
        self.positions_data = {}

        ctx = StrategyContext(self.event_bus, config=LeekComponentConfig(
            instance_id="p1",
            name="debug",
            cls=Strategy,
            config=StrategyConfig(data_source_configs=[])
        ))
        ctx.strategies["debug"] = self.strategy
        self.engine.strategy_manager.components["p1"] = ctx
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
        
        # 收集净值数据用于计算区间收益
        equity_values = []

        # 模拟K线模式：使用MERGE合并K线
        merge = MERGE(window=self.merge_window) if self.simulate_kline else None
        query_timeframe = self.base_timeframe if self.simulate_kline else self.timeframe

        custom_key = []
        for raw_kline in self.data_source.get_history_data(start_time=self.start_time, end_time=self.end_time,
                                                       row_key=KLine.pack_row_key(self.symbol, self.quote_currency, self.ins_type, query_timeframe),
                                                       market=self.market):
            # 模拟K线模式：合并K线
            if self.simulate_kline:
                kline = merge.update(raw_kline)
                if kline is None:
                    continue  # 等待合并窗口对齐
            else:
                kline = raw_kline
            count += 1
            assets = self.strategy.on_data(kline)
            self.engine.position_tracker.on_data(kline)
            
            # 模拟模式下只记录完成的K线数据，非模拟模式记录所有数据
            should_record = not self.simulate_kline or kline.is_finished
            
            if should_record:
                data["position"].append(self.strategy.position_rate * 100)
                if self.bechmark is None:
                    self.bechmark = kline.close
                    for k in kline.dynamic_attrs.keys():
                        custom_key.append(k)
                        if k not in data:
                            data[k] = []
                    # print("========================")
                    # print(f"自定义KEY: {custom_key}")
                    # print("========================")
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
                    strategy_cls=f"{self.strategy.strategy.__class__.__module__}|{self.strategy.strategy.__class__.__name__}",
                    config=None,
                    signal_time=datetime.now(),
                    assets=assets
                )
                ctx.signals[signal.signal_id] = signal
                if should_record:
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
                self.engine._on_signal(signal)
            
            if should_record:
                total_value = self.engine.portfolio.total_value
                equity_values.append(float(total_value))
                data["profit"].append((total_value - self.initial_balance) / self.initial_balance * 100)
        
        logger.info(f"数据执行完成，共{count}条")
        if kwargs.get("print_metrics", True):
            # 计算并输出回测指标
            self._print_backtest_metrics(equity_values)
        
        self.data_source.on_stop()
        self.engine.on_stop()
        if kwargs.get("draw", True):
            self.draw(data, row, custom_draw, **kwargs)

    def draw(self, data, row=None, custom_draw=None, **kwargs) -> None:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = pd.DataFrame(data)

        if kwargs.get("sample_draw", False):
            return self._sample_draw(df, custom_draw, **kwargs)

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

        if kwargs.get("draw_2", True):
            if kwargs.get("draw_bechmark", True):
                fig.add_trace(go.Scatter(x=df['time'], y=df["bechmark"], mode='lines', name="bechmark",
                                    line=dict(color=self.get_color(), width=1)), row=2, col=1)
            if kwargs.get("draw_profit", True):
                fig.add_trace(go.Scatter(x=df['time'], y=df["profit"], mode='lines', name="return",
                                    line=dict(color=self.get_color(), width=2)), row=2, col=1)
            if kwargs.get("draw_position", True):
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

    def _sample_draw(self, df, custom_draw=None, **kwargs) -> None:
        """简单模式画图：close和return归一化为百分比变化，画在同一张图上"""
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        # close 归一化为百分比变化（相对于第一个值）
        first_close = df['close'].iloc[0]
        close_pct = (df['close'] - first_close) / first_close * 100

        # 计算自定义图的行数
        custom_rows = 0
        if custom_draw is not None:
            # 先用临时fig探测custom_draw会添加多少行
            custom_rows = kwargs.get("sample_custom_rows", 0)

        rows_count = max(1 + custom_rows, 1)
        specs = [[{"secondary_y": False}] for _ in range(rows_count)]
        fig = make_subplots(rows=rows_count, cols=1, shared_xaxes=True, specs=specs)

        # close 百分比变化曲线
        fig.add_trace(go.Scatter(
            x=df['time'], y=close_pct, mode='lines', name='close %',
            line=dict(color='#1f77b4', width=1.5)
        ), row=1, col=1)

        # return 曲线（已经是百分比）
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['profit'], mode='lines', name='return %',
            line=dict(color='#ff7f0e', width=2)
        ), row=1, col=1)

        fig.update_yaxes(title_text="%", ticksuffix="%", row=1, col=1)

        if custom_draw is not None:
            custom_draw(fig, df)

        fig.update_layout(height=kwargs.get("height", 500 + custom_rows * 150))
        fig.show()

