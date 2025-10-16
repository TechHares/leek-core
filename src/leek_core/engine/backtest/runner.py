#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import asdict
from decimal import Decimal
from re import L
from typing import Optional, Callable, Dict, Any, Type, List
import sys

import numpy as np

from leek_core.utils import get_logger, DateTimeUtils, setup_logging
from leek_core.base import load_class_from_str, create_component
from leek_core.data import DataSource
from leek_core.models import PositionConfig, LeekComponentConfig, Position, Signal, StrategyConfig, KLine
from leek_core.manager import PositionManager, ExecutorManager, StrategyManager
from leek_core.event import SerializableEventBus, Event, EventType
from .performance import vectorized_operations
from leek_core.executor import ExecutorContext
from leek_core.strategy import StrategyContext
from leek_core.sub_strategy import SubStrategy

from .types import RunConfig, PerformanceMetrics, BacktestResult
from .data_cache import DataCache
from concurrent_log_handler import ConcurrentRotatingFileHandler

logger = get_logger(__name__)
def run_backtest(config: Dict[str, Any]):
    """运行回测"""
    runner = BacktestRunner(config)
    try:
        handler = ConcurrentRotatingFileHandler(filename=f'{config.get("id", 0)}.log', mode='a', maxBytes=0, backupCount=0, encoding='utf-8')
        setup_logging(console=False, external_handlers=[handler])
        global logger
        logger = get_logger(__name__)
        return runner.run()
    finally:
        try:
            runner.on_stop()
        except Exception as e:
            print(f"停止回测失败: {e}")

class BacktestRunner:
    def __init__(self, config: Dict[str, Any]):
        self.original_config = config
        self.config = RunConfig(**config)

        self.data_source = None
        self.event_bus = None

        # 统计变量
        self.trades_data = []
        self.positions_data = []
        self.signals_data = []
        self.equity_values = []
        self.equity_times = []
        self.benchmark_prices = []  # 收集基准价格数据

        # 交易统计
        self.trades_counter = {"win": 0, "loss": 0, "total": 0}
        self.trade_profit_sum = 0.0
        self.trade_loss_sum = 0.0

    def _load_mount_dirs(self):
        # 加载模块
        for path in self.config.mount_dirs:
            if path and path not in sys.path:
                sys.path.insert(0, path)

    def _init_data_source(self):
        datasource: DataSource = create_component(load_class_from_str(self.config.datasource_class), **(self.config.datasource_config | {}))
        if self.config.use_cache:
            self.data_source = DataCache(datasource)
        else:
            self.data_source = datasource
        self.data_source.on_start()

    def _init_position_manager(self):
        self.position_manager = PositionManager(
            self.event_bus,
            LeekComponentConfig(
                instance_id="pos_mgr",
                name="仓位管理",
                cls=None,
                config=PositionConfig(
                    init_amount=self.config.initial_balance,
                    max_amount=self.config.initial_balance * 1000,
                    max_strategy_amount=self.config.initial_balance * 1000,
                    max_strategy_ratio=Decimal("0.5"),
                    max_symbol_amount=self.config.initial_balance * 1000,
                    max_symbol_ratio=Decimal("1"),
                    max_ratio=Decimal("1"),
                ),
            ),
        )
        self.position_manager.on_start()

    def _init_executor(self):
        self.executor_manager = ExecutorManager(
            self.event_bus,
            LeekComponentConfig(
                instance_id="exec_mgr",
                name="执行器管理",
                cls=ExecutorContext,
                config=None,
            ),
        )
        self.executor_manager.on_start()
        executor_cls = load_class_from_str(self.config.executor_class)
        self.executor_manager.add(
            LeekComponentConfig(
                instance_id="executor",
                name="执行器",
                cls=executor_cls,
                config=self.config.executor_config or {},
            )
        )

    def _init_strategy(self):
        # 创建策略管理器
        self.strategy_manager = StrategyManager(
            self.event_bus,
            LeekComponentConfig(
                instance_id="strat_mgr",
                name="策略管理",
                cls=StrategyContext,
                config=None
            ),
        )
        self.strategy_manager.on_start()
        # 准备风控策略
        risk_policies_cfg = []
        for risk_cfg in self.config.risk_policies:
            cls_path: Type[SubStrategy] = load_class_from_str(risk_cfg.get("class_name") or risk_cfg.get("cls"))
            risk_policies_cfg.append(LeekComponentConfig(
                instance_id=f"T1",
                name=cls_path.display_name,
                cls=cls_path,
                config=risk_cfg.get("config", {})
            ))

        # 添加策略
        strategy_cfg = StrategyConfig(
            data_source_configs=[],
            info_fabricator_configs=[],
            strategy_config=self.config.strategy_params,
            strategy_position_config=None,
            risk_policies=risk_policies_cfg,
        )

        strategy_cls = load_class_from_str(self.config.strategy_class)
        strategy_component = LeekComponentConfig(
            instance_id="strategy",
            name="策略",
            cls=strategy_cls,
            config=strategy_cfg,
        )
        self.strategy_manager.add(strategy_component)

    def run(self):
        self.event_bus = SerializableEventBus()
        self._load_mount_dirs()
        self._init_data_source()
        self._init_position_manager()
        self._init_executor()
        self._init_strategy()

        # 订阅事件
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self._on_signal_generated)

        # 执行回测
        last_position_rate = Decimal("0")
        turnover_acc = Decimal("0")
        row_key = KLine.pack_row_key(self.config.symbol, self.config.quote_currency, self.config.ins_type, self.config.timeframe)
        logger.info(f"Generated row_key: {row_key}")
        strategy_ctx = self.strategy_manager.get("strategy")
        wrapper = None
        for kline in self.data_source.get_history_data(
                start_time=min(self.config.start_time, self.config.pre_start) if  self.config.pre_start else self.config.start_time,
                end_time=max(self.config.end_time, self.config.pre_end) if self.config.pre_end else self.config.end_time,
               row_key=row_key, market=self.config.market):
            if kline.start_time < self.config.start_time:
                continue
            if kline.end_time > self.config.end_time:
                break
            # 更新仓位管理
            self.position_manager.process_data_update(Event(EventType.DATA_RECEIVED, kline))
            # 处理策略数据
            strategy_ctx.on_data(kline)

            if wrapper is None:
                wrapper = list(strategy_ctx.strategies.values())[0]

            # 获取当前仓位率
            current_rate = wrapper.position_rate if wrapper else last_position_rate
            turnover_acc += abs(current_rate - last_position_rate)
            last_position_rate = current_rate

            # 记录净值
            total_value = self.position_manager.total_value
            self.equity_values.append(float(total_value))
            self.equity_times.append(int(kline.end_time))

            # 记录基准价格（与净值一一对应）
            self.benchmark_prices.append(float(kline.close))

        # 计算性能指标（使用优化的并行计算）
        metrics = self._calculate_performance_metrics_optimized(turnover=float(turnover_acc))

        # 计算回撤曲线
        drawdown_curve = self._calculate_drawdown_curve()

        # 计算基准曲线（如果有基准价格）
        benchmark_curve = self._calculate_benchmark_curve_from_prices()

        return BacktestResult(
            config=self.original_config,
            metrics=metrics,
            equity_curve=self.equity_values,
            equity_times=self.equity_times,
            trades=self.trades_data,
            positions=self.positions_data,
            signals=self.signals_data,
            drawdown_curve=drawdown_curve,
            benchmark_curve=benchmark_curve,
            metadata={
                "total_bars": len(self.equity_values),
                "turnover": float(turnover_acc)
            }
        )

    def _calculate_benchmark_curve_from_prices(self):
        if not self.benchmark_prices:
            return []

        prices = np.array(self.benchmark_prices, dtype=np.float64)
        first_price = self.benchmark_prices[0]
        init_balance = float(self.config.initial_balance)
        # 向量化计算
        benchmark_curve = (prices - first_price) / first_price * init_balance + init_balance
        return benchmark_curve.tolist()

    def _on_position_update(self, event: Event):
        """处理仓位更新事件"""
        if isinstance(event.data, Position):
            pos = event.data
            cur_sz = float(pos.sz or 0)
            total_sz = float(pos.total_sz or 0)

            self.positions_data.append({
                "timestamp": DateTimeUtils.now_timestamp(),
                "symbol": pos.symbol,
                "side": str(pos.side),
                "size": cur_sz,
                "avg_price": float(pos.cost_price or 0),
                "pnl": float(pos.pnl or 0),
                "unrealized_pnl": float(getattr(pos, 'unrealized_pnl', 0) or 0)
            })

            # 统计交易（平仓时记录）
            if total_sz > 0 and cur_sz == 0:  # 平仓
                pnl_val = float(pos.pnl or 0)
                self.trades_counter["total"] += 1

                trade_data = {
                    "timestamp": DateTimeUtils.now_timestamp(),
                    "symbol": pos.symbol,
                    "side": str(pos.side),
                    "size": total_sz,
                    "entry_price": float(pos.cost_price or 0),
                    "exit_price": float(pos.current_price or pos.cost_price or 0),
                    "pnl": pnl_val,
                    "duration": 0
                }
                self.trades_data.append(trade_data)

                if pnl_val > 0:
                    self.trades_counter["win"] += 1
                    self.trade_profit_sum += pnl_val
                elif pnl_val < 0:
                    self.trades_counter["loss"] += 1
                    self.trade_loss_sum += abs(pnl_val)

    def _on_signal_generated(self, event: Event):
        """处理信号生成事件"""
        if isinstance(event.data, Signal):
            signal_ev = event.data
            self.signals_data.append({
                "timestamp": int(signal_ev.signal_time.timestamp() * 1000),
                "strategy_id": signal_ev.strategy_id,
                "assets": [asdict(asset) for asset in signal_ev.assets]
            })

    def _calculate_drawdown_curve(self) -> List[float]:
        """计算回撤曲线"""
        if not self.equity_values:
            return []

        equity_array = np.array(self.equity_values)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        return drawdown.tolist()

    def _calculate_performance_metrics_optimized(self, turnover: float = 0.0) -> PerformanceMetrics:
        """优化的性能指标计算"""
        if not self.equity_values:
            return PerformanceMetrics()

        # 使用向量化操作
        equity_array = np.array(self.equity_values)
        periods_per_year = (24 * 3600 * 1000 / self.config.timeframe.milliseconds) * 365
        # 并行计算基础指标
        vectorized_results = vectorized_operations(equity_array, ['returns', 'moving_average', 'volatility'], periods_per_year=periods_per_year)

        returns = vectorized_results.get('returns', np.array([]))

        # 基础指标
        total_return = (self.equity_values[-1] - self.equity_values[0]) / self.equity_values[0] if self.equity_values[0] != 0 else 0.0

        # 计算年化收益率
        if len(self.equity_times) > 1:
            days = (self.equity_times[-1] - self.equity_times[0]) / (24 * 3600 * 1000)
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        else:
            annual_return = total_return

        # 波动率
        volatility = vectorized_results.get('full_volatility', 0)
        # 夏普比率
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0

        # 回撤相关指标
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)

        # 回撤持续期
        in_drawdown = drawdown < 0
        if np.any(in_drawdown):
            drawdown_periods = np.sum(np.diff(np.concatenate(([False], in_drawdown, [False])).astype(int)) == 1)

            # 计算最大回撤持续期
            max_dd_duration = 0
            current_duration = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_duration += 1
                    max_dd_duration = max(max_dd_duration, current_duration)
                else:
                    current_duration = 0
        else:
            drawdown_periods = 0
            max_dd_duration = 0

        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Sortino比率
        negative_returns = returns[returns < 0] if len(returns) > 0 else np.array([])
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 1 else 0.0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0.0

        # 交易统计
        total_trades = len(self.trades_data)
        win_trades = len([t for t in self.trades_data if t.get("pnl", 0) > 0])
        loss_trades = len([t for t in self.trades_data if t.get("pnl", 0) < 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0

        # 做多和做空交易分类
        # 支持多种side值格式：字符串'LONG'/'SHORT'，数字1/2，或枚举值
        long_trades = [t for t in self.trades_data if str(t.get("side", "")) in ["LONG", "1", "PositionSide.LONG"]]
        short_trades = [t for t in self.trades_data if str(t.get("side", "")) in ["SHORT", "2", "PositionSide.SHORT"]]

        # 做多胜率
        long_win_trades = len([t for t in long_trades if t.get("pnl", 0) > 0])
        long_win_rate = long_win_trades / len(long_trades) if long_trades else 0.0

        # 做空胜率
        short_win_trades = len([t for t in short_trades if t.get("pnl", 0) > 0])
        short_win_rate = short_win_trades / len(short_trades) if short_trades else 0.0

        # 盈亏比
        profits = [t["pnl"] for t in self.trades_data if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in self.trades_data if t.get("pnl", 0) < 0]

        avg_win = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = sum(profits) / sum(losses) if losses else float('inf') if profits else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0.0

        # VaR和CVaR
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0.0

            # 偏度和峰度
            if len(returns) > 1 and np.std(returns) > 0:
                standardized = (returns - np.mean(returns)) / np.std(returns)
                skewness = float(np.mean(standardized ** 3))
                kurtosis = float(np.mean(standardized ** 4)) - 3
            else:
                skewness = 0.0
                kurtosis = 0.0
        else:
            var_95 = 0.0
            cvar_95 = 0.0
            skewness = 0.0
            kurtosis = 0.0

        # Omega比率
        threshold = 0.0
        if len(returns) > 0:
            gains = returns[returns > threshold]
            losses_omega = returns[returns <= threshold]
            omega_ratio = np.sum(gains - threshold) / abs(np.sum(losses_omega - threshold)) if len(
                losses_omega) > 0 and np.sum(losses_omega - threshold) != 0 else float('inf') if len(gains) > 0 else 0.0
        else:
            omega_ratio = 0.0

        return PerformanceMetrics(
            total_return=float(total_return),
            annual_return=float(annual_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            calmar_ratio=float(calmar_ratio),
            sortino_ratio=float(sortino_ratio),
            omega_ratio=float(omega_ratio),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_dd_duration),
            drawdown_periods=int(drawdown_periods),
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=float(win_rate),
            long_win_rate=float(long_win_rate),
            short_win_rate=float(short_win_rate),
            profit_factor=float(profit_factor),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            win_loss_ratio=float(win_loss_ratio),
            largest_win=float(max(profits)) if profits else 0.0,
            largest_loss=float(min([-abs(l) for l in losses])) if losses else 0.0,
            turnover=float(turnover),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
        )

    def on_stop(self):
        self.strategy_manager.on_stop()
        self.executor_manager.on_stop()
        self.position_manager.on_stop()
        self.data_source.on_stop()
