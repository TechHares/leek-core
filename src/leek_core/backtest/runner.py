#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import asdict
from decimal import Decimal
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Type
import logging
import os
import sys

import numpy as np

from leek_core.base import create_component, load_class_from_str
from leek_core.data import DataSource
from leek_core.engine import SimpleEngine
from leek_core.event import Event, EventType, SerializableEventBus
from leek_core.executor import ExecutorContext
from leek_core.manager import ExecutorManager, StrategyManager
from leek_core.models import (
    KLine,
    LeekComponentConfig,
    Order,
    Position,
    PositionConfig,
    Signal,
    StrategyConfig,
    TimeFrame,
)
from leek_core.indicators import MERGE
from leek_core.strategy import StrategyContext
from leek_core.sub_strategy import SubStrategy
from leek_core.utils import DateTimeUtils, get_logger, set_worker_id, setup_logging

from .data_cache import DataCache
from .performance import vectorized_operations
from .statistical_tests import calculate_statistical_tests
from .types import BacktestResult, PerformanceMetrics, RunConfig

logger = get_logger(__name__)
def run_backtest(config: Dict[str, Any]):
    """运行回测"""
    runner = BacktestRunner(config)
    try:
        set_worker_id(os.getpid() % 1024)
        if config.get("log_file", False):
            from concurrent_log_handler import ConcurrentRotatingFileHandler
            handler = ConcurrentRotatingFileHandler(filename=f'{config.get("id", 0)}.log', mode='a', maxBytes=0, backupCount=0, encoding='utf-8')
            setup_logging(console=False, external_handlers=[handler])
        else:
            setup_logging(console=False, external_handlers=[logging.NullHandler()], log_level="CRITICAL")
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

        # K线模拟相关
        self.merge = None
        self.merge_window = None
        if self.config.simulate_kline:
            # 固定使用1分钟K线作为基础周期
            base_timeframe = TimeFrame.M1
            if self.config.timeframe.milliseconds is None or base_timeframe.milliseconds is None:
                raise ValueError("K线模拟模式不支持 TICK 周期")
            if self.config.timeframe.milliseconds % base_timeframe.milliseconds != 0:
                raise ValueError(f"timeframe({self.config.timeframe.value})必须能被1分钟整除")
            self.merge_window = self.config.timeframe.milliseconds // base_timeframe.milliseconds
            if self.merge_window <= 1:
                raise ValueError(f"timeframe必须大于1分钟，当前窗口为{self.merge_window}")
            self.merge = MERGE(window=self.merge_window)
            logger.info(f"K线模拟模式已启用: base=1m, target={self.config.timeframe.value}, window={self.merge_window}")

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

    def _init_executor(self):
        executor_cls = load_class_from_str(self.config.executor_class)
        self.engine.add_executor(
            LeekComponentConfig(
                instance_id="executor",
                name="执行器",
                cls=executor_cls,
                config=self.config.executor_config or {},
            )
        )

    def _init_strategy(self):
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
            instance_id="0",
            name="策略",
            cls=strategy_cls,
            config=strategy_cfg,
        )
        self.engine.add_strategy(strategy_component)

    def _init(self):
        self.event_bus = SerializableEventBus()
        self._load_mount_dirs()
        self.engine = SimpleEngine("p2", "backtest", PositionConfig(
            init_amount=self.config.initial_balance,
            max_amount=self.config.initial_balance * 1000,
            max_strategy_amount=self.config.initial_balance * 1000,
            max_strategy_ratio=Decimal("0.5"),
            max_symbol_amount=self.config.initial_balance * 1000,
            max_symbol_ratio=Decimal("1"),
            max_ratio=Decimal("1"),
        ), 0, self.event_bus)
        self.engine.on_start()
        self._init_data_source()
        self._init_executor()
        self._init_strategy()

        # 订阅事件
        self.event_bus.subscribe_event(EventType.ORDER_UPDATED, self._on_order_updated)
        self.event_bus.subscribe_event(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, self._on_signal_generated)

    def run(self):
        time_start = DateTimeUtils.now_timestamp()
        self._init()

        # 执行回测
        last_position_rate = Decimal("0")
        turnover_acc = Decimal("0")
        
        # K线模拟模式：使用1分钟K线查询，否则使用目标周期
        query_timeframe = TimeFrame.M1 if self.config.simulate_kline else self.config.timeframe
        row_key = KLine.pack_row_key(self.config.symbol, self.config.quote_currency, self.config.ins_type, query_timeframe)
        logger.info(f"Generated row_key: {row_key}, simulate_kline={self.config.simulate_kline}")
        strategy_ctx = self.engine.strategy_manager.get("0")
        wrapper = None
        time_init = DateTimeUtils.now_timestamp()
        time_data = None
        time_run = None
        for raw_kline in self.data_source.get_history_data(start_time=self.config.start_time, end_time=self.config.end_time,
                            pre_load_start_time=self.config.pre_start, pre_load_end_time=self.config.pre_end,
                            row_key=row_key, market=self.config.market):
            if time_data is None:
                time_data = DateTimeUtils.now_timestamp()
                time_run = time_data
            
            # K线模拟模式：合并K线
            if self.config.simulate_kline:
                kline = self.merge.update(raw_kline)
                if kline is None:
                    continue  # 等待合并窗口对齐
            else:
                kline = raw_kline
            
            # 更新仓位管理
            kline.target_instance_id = set(["0"])
            signal = strategy_ctx._process_data(kline)
            if signal:
                self.engine._on_signal(signal)
            # 处理策略数据
            if wrapper is None:
                wrapper = list(strategy_ctx.strategies.values())[0]

            # 获取当前仓位率
            current_rate = wrapper.position_rate if wrapper else last_position_rate
            turnover_acc += abs(current_rate - last_position_rate)
            last_position_rate = current_rate

            # 记录净值（K线模拟模式下只记录完成的K线）
            should_record = not self.config.simulate_kline or kline.is_finished
            if should_record:
                total_value = self.engine.portfolio.total_value
                self.equity_values.append(float(total_value))
                self.equity_times.append(int(kline.end_time))

                # 记录基准价格（与净值一一对应）
                self.benchmark_prices.append(float(kline.close))
        time_end = DateTimeUtils.now_timestamp()
        # 计算性能指标（使用优化的并行计算）
        metrics = self._calculate_performance_metrics_optimized(turnover=float(turnover_acc))

        # 计算回撤曲线
        drawdown_curve = self._calculate_drawdown_curve()

        # 计算基准曲线（如果有基准价格）
        benchmark_curve = self._calculate_benchmark_curve_from_prices()
        time_metrics = DateTimeUtils.now_timestamp()
        return BacktestResult(
            times=[time_start, time_init, time_data or time_init, time_run or time_init, time_end, time_metrics],
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

    def _on_order_updated(self, event: Event):
        if not isinstance(event.data, Order):
            return
        order = event.data
        if order.is_open or not order.order_status.is_finished:
            return
        pnl_val = float(order.pnl or 0)
        self.trades_counter["total"] += 1

        trade_data = {
            "timestamp": int(order.order_time.timestamp() * 1000),
            "symbol": order.symbol,
            "side": str(order.side.switch()),
            "size": float(order.sz),
            "entry_amount": float(order.order_amount),
            "pnl": pnl_val,
        }
        self.trades_data.append(trade_data)

        if pnl_val > 0:
            self.trades_counter["win"] += 1
            self.trade_profit_sum += pnl_val
        elif pnl_val < 0:
            self.trades_counter["loss"] += 1
            self.trade_loss_sum += abs(pnl_val)

    def _on_position_update(self, event: Event):
        """处理仓位更新事件"""
        if isinstance(event.data, Position):
            pos = event.data
            cur_sz = float(pos.sz or 0)
            self.positions_data.append({
                "timestamp": DateTimeUtils.now_timestamp(),
                "symbol": pos.symbol,
                "side": str(pos.side),
                "size": cur_sz,
                "avg_price": float(pos.cost_price or 0),
                "pnl": float(pos.pnl or 0),
                "unrealized_pnl": float(getattr(pos, 'unrealized_pnl', 0) or 0)
            })


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
        periods_per_year = int((24 * 3600 * 1000 / self.config.timeframe.milliseconds) * 365)
        # 并行计算基础指标
        vectorized_results = vectorized_operations(equity_array, ['returns', 'moving_average', 'volatility'], periods_per_year=periods_per_year)

        returns = vectorized_results.get('returns', np.array([]))

        # 基础指标
        total_return = (self.equity_values[-1] - self.equity_values[0]) / self.equity_values[0] if self.equity_values[0] != 0 else 0.0

        # 计算年化收益率
        n_periods = len(self.equity_values) - 1  # 实际交易期数
        if n_periods > 0 and total_return > -1:
            # 年化收益率 = (1 + 总收益率)^(年期数/实际期数) - 1
            annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        else:
            annual_return = 0.0

        # 波动率（已年化）
        volatility = vectorized_results.get('full_volatility', 0)
        # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
        # 无风险利率暂设为0
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

        # Calmar比率 = 年化收益率 / |最大回撤|
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        # Sortino比率 = 年化收益率 / 年化下行标准差
        negative_returns = returns[returns < 0] if len(returns) > 0 else np.array([])
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year) if len(negative_returns) > 1 else 0.0
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

        # 做多/做空的平均利润和平均亏损
        long_profits = [t["pnl"] for t in long_trades if t.get("pnl", 0) > 0]
        long_losses = [abs(t["pnl"]) for t in long_trades if t.get("pnl", 0) < 0]
        short_profits = [t["pnl"] for t in short_trades if t.get("pnl", 0) > 0]
        short_losses = [abs(t["pnl"]) for t in short_trades if t.get("pnl", 0) < 0]

        avg_win_long = float(np.mean(long_profits)) if long_profits else 0.0
        avg_loss_long = float(np.mean(long_losses)) if long_losses else 0.0
        avg_win_short = float(np.mean(short_profits)) if short_profits else 0.0
        avg_loss_short = float(np.mean(short_losses)) if short_losses else 0.0

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

        # 单笔平均收益（绝对值）
        avg_pnl = float(np.mean([t.get("pnl", 0.0) for t in self.trades_data])) if total_trades > 0 else 0.0
        avg_pnl_long = float(np.mean([t.get("pnl", 0.0) for t in long_trades])) if len(long_trades) > 0 else 0.0
        avg_pnl_short = float(np.mean([t.get("pnl", 0.0) for t in short_trades])) if len(short_trades) > 0 else 0.0

        # 单笔平均收益率（基于名义本金：入场价 * 数量）
        def trade_return_rate(trade: Dict[str, Any]) -> float:
            denom = trade["entry_amount"]
            pnl_val = float(trade["pnl"])
            return pnl_val / denom if denom > 0 else 0.0

        returns_all = [trade_return_rate(t) for t in self.trades_data]
        returns_long = [trade_return_rate(t) for t in long_trades]
        returns_short = [trade_return_rate(t) for t in short_trades]

        avg_return_per_trade = float(np.mean(returns_all)) if total_trades > 0 else 0.0
        avg_return_long = float(np.mean(returns_long)) if len(long_trades) > 0 else 0.0
        avg_return_short = float(np.mean(returns_short)) if len(short_trades) > 0 else 0.0

        # 单笔最大亏损比例（小数），供评估与 -0.1 比较、.1% 展示
        loss_return_rates = [
            trade_return_rate(t) for t in self.trades_data
            if t.get("pnl", 0) < 0 and t.get("entry_amount", 0) > 0
        ]
        largest_loss_ratio = float(min(loss_return_rates)) if loss_return_rates else 0.0

        # 统计检验（训练阶段可跳过以提升性能）
        statistical_results = {}
        if self.config.skip_statistical_tests is False:
            statistical_results = calculate_statistical_tests(
                equity_curve=self.equity_values,
                benchmark_curve=self.benchmark_prices if self.benchmark_prices else None,
                trades=self.trades_data,
                n_bootstrap=1000
            )

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
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_win_trades=long_win_trades,
            short_win_trades=short_win_trades,
            long_win_rate=float(long_win_rate),
            short_win_rate=float(short_win_rate),
            profit_factor=float(profit_factor),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            avg_win_long=float(avg_win_long),
            avg_loss_long=float(avg_loss_long),
            avg_win_short=float(avg_win_short),
            avg_loss_short=float(avg_loss_short),
            win_loss_ratio=float(win_loss_ratio),
            largest_win=float(max(profits)) if profits else 0.0,
            largest_loss=largest_loss_ratio,
            turnover=float(turnover),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            avg_pnl=float(avg_pnl),
            avg_pnl_long=float(avg_pnl_long),
            avg_pnl_short=float(avg_pnl_short),
            avg_return_per_trade=float(avg_return_per_trade),
            avg_return_long=float(avg_return_long),
            avg_return_short=float(avg_return_short),
            # 统计检验结果
            t_statistic=statistical_results.get('t_statistic', 0.0),
            t_pvalue=statistical_results.get('t_pvalue', 1.0),
            paired_t_statistic=statistical_results.get('paired_t_statistic', 0.0),
            paired_t_pvalue=statistical_results.get('paired_t_pvalue', 1.0),
            bootstrap_sharpe_ci_lower=statistical_results.get('bootstrap_sharpe_ci_lower', 0.0),
            bootstrap_sharpe_ci_upper=statistical_results.get('bootstrap_sharpe_ci_upper', 0.0),
            bootstrap_annual_return_ci_lower=statistical_results.get('bootstrap_annual_return_ci_lower', 0.0),
            bootstrap_annual_return_ci_upper=statistical_results.get('bootstrap_annual_return_ci_upper', 0.0),
            win_rate_pvalue=statistical_results.get('win_rate_pvalue', 1.0),
            alpha_pvalue=statistical_results.get('alpha_pvalue', 1.0),
        )

    def on_stop(self):
        self.engine.on_stop()
        self.data_source.on_stop()
