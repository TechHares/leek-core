#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from leek_core.data import DataSource
from leek_core.models import TimeFrame, TradeInsType
from leek_core.utils import DateTimeUtils


class BacktestMode(Enum):
    """回测模式"""
    SINGLE = "single"  # 单次回测，固定参数
    NORMAL = "normal"  # 普通回测
    PARAM_SEARCH = "param_search"  # 参数搜索
    WALK_FORWARD = "walk_forward"  # 走向前验证
    MONTE_CARLO = "monte_carlo"  # 蒙特卡洛模拟


class OptimizationObjective(Enum):
    """优化目标"""
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    CUSTOM = "custom"


@dataclass
class RunConfig:
    """运行配置"""
    id: int = 0
    # 策略配置
    strategy_class: str = None
    strategy_params: Dict[str, Any] = None
    risk_policies: List[Dict[str, Any]] = None

    # 数据源配置
    datasource_class: str = None
    datasource_config: Dict[str, Any] = None

    # 执行器配置
    executor_class: str = None
    executor_config: Dict[str, Any] = None

    # 数据配置
    pre_start: int = None
    pre_end: int = None
    start_time: str|datetime|int = None
    end_time: str|datetime|int = None
    market: str = "okx"
    quote_currency: str = "USDT"
    ins_type: TradeInsType = TradeInsType.SWAP
    symbol: str = None
    timeframe: TimeFrame = None

    # 资金配置
    initial_balance: Decimal = Decimal("10000")

    # 其它配置
    mount_dirs: List[str] = field(default_factory=list)

    # 性能优化选项
    use_cache: bool = False
    skip_statistical_tests: bool = False  # 是否跳过统计检验（用于训练阶段加速）
    # 日志选项
    log_file: bool = False

    def __post_init__(self):
        if isinstance(self.timeframe, str):
            self.timeframe = TimeFrame(self.timeframe)
        if isinstance(self.ins_type, int):
            self.ins_type = TradeInsType(self.ins_type)
        if isinstance(self.ins_type, str):
            self.ins_type = TradeInsType(TradeInsType[self.ins_type])
        if isinstance(self.start_time, str):
            self.start_time = DateTimeUtils.to_timestamp(self.start_time)
        if isinstance(self.start_time, datetime):
            self.start_time = DateTimeUtils.datetime_to_timestamp(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = DateTimeUtils.to_timestamp(self.end_time)
        if isinstance(self.end_time, datetime):
            self.end_time = DateTimeUtils.datetime_to_timestamp(self.end_time)
        assert self.strategy_class is not None
        assert self.datasource_class is not None
        assert self.executor_class is not None
        assert self.start_time is not None
        assert self.end_time is not None
        assert self.timeframe is not None
        assert self.symbol is not None
        assert self.ins_type is not None
        assert self.initial_balance is not None

@dataclass
class BacktestConfig:
    """回测配置"""
    # 基础配置
    id: int
    name: str
    mode: BacktestMode

    # 策略配置
    strategy_class: str
    strategy_params: Dict[str, Any] = None

    # 数据配置
    symbols: List[str] = None
    timeframes: List[TimeFrame] = None
    start_time: Union[str, datetime] = None
    end_time: Union[str, datetime] = None
    market: str = "okx"
    quote_currency: str = "USDT"
    ins_type: TradeInsType = TradeInsType.SWAP

    # 执行配置
    initial_balance: Decimal = Decimal("10000")
    executor_class: str = "leek_core.executor.BacktestExecutor"
    executor_config: Dict[str, Any] = None

    # 参数搜索配置
    param_space: Dict[str, List[Any]] = None
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    custom_objective: Callable[[Dict[str, Any]], float] = None

    # 走向前配置
    train_days: int = None
    test_days: int = None
    embargo_days: int = 0
    cv_splits: int = 0
    # 窗口模式：rolling | expanding
    wf_window_mode: str = "rolling"

    # 并行配置
    max_workers: int = 1
    min_window_size: int = 1  # 最小执行粒度窗口

    # 风险管理
    risk_policies: List[Dict[str, Any]] = None

    # 数据源配置
    data_source: DataSource = None
    data_source_config: Dict[str, Any] = None

    # 策略路径（用于子进程/线程导入）
    mount_dirs: List[str] = None

    # 性能优化配置
    use_cache: bool = True  # 是否使用缓存
    # 日志选项
    log_file: bool = False

    # Optuna 配置（仅 WFA 使用，可选）
    optuna_enabled: bool = False
    optuna_n_trials: int = 80

@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基础指标
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0

    # 风险调整收益指标
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    omega_ratio: float = 0.0
    sterling_ratio: float = 0.0
    information_ratio: float = 0.0

    # 回撤指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    drawdown_periods: int = 0

    # 交易指标
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    # 多空拆分笔数与胜率
    long_trades: int = 0
    short_trades: int = 0
    long_win_trades: int = 0
    short_win_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_long: float = 0.0  # 做多平均利润（只算赚钱的单子）
    avg_loss_long: float = 0.0  # 做多平均亏损（只算亏钱的单子，取绝对值）
    avg_win_short: float = 0.0  # 做空平均利润（只算赚钱的单子）
    avg_loss_short: float = 0.0  # 做空平均亏损（只算亏钱的单子，取绝对值）
    win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # 单笔收益（绝对值与收益率）
    avg_pnl: float = 0.0
    avg_pnl_long: float = 0.0
    avg_pnl_short: float = 0.0
    avg_return_per_trade: float = 0.0  # 单笔平均收益率（小数）
    avg_return_long: float = 0.0
    avg_return_short: float = 0.0

    # 其他指标
    turnover: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR
    beta: float = 0.0
    alpha: float = 0.0
    r_squared: float = 0.0

    # 时间相关
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0

    # 统计检验结果
    t_statistic: float = 0.0  # t统计量
    t_pvalue: float = 1.0  # t检验p值
    paired_t_statistic: float = 0.0  # 配对t检验统计量
    paired_t_pvalue: float = 1.0  # 配对t检验p值
    bootstrap_sharpe_ci_lower: float = 0.0  # Bootstrap Sharpe 95%置信区间下界
    bootstrap_sharpe_ci_upper: float = 0.0  # Bootstrap Sharpe 95%置信区间上界
    bootstrap_annual_return_ci_lower: float = 0.0  # Bootstrap年化收益率95%置信区间下界
    bootstrap_annual_return_ci_upper: float = 0.0  # Bootstrap年化收益率95%置信区间上界
    win_rate_pvalue: float = 1.0  # 胜率二项检验p值
    alpha_pvalue: float = 1.0  # alpha显著性检验p值

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """回测结果"""
    times: List[int]
    config: Dict[str, Any]
    metrics: PerformanceMetrics
    equity_curve: List[float]
    equity_times: List[int]
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]
    drawdown_curve: List[float]
    benchmark_curve: List[float] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": self.times,
            "config": self.config,
            "metrics": self.metrics.to_dict(),
            "equity_curve": self.equity_curve,
            "equity_times": self.equity_times,
            "trades": self.trades,
            "positions": self.positions,
            "signals": self.signals,
            "drawdown_curve": self.drawdown_curve,
            "benchmark_curve": self.benchmark_curve,
            "execution_time": self.execution_time,
            "metadata": self.metadata or {}
        }

@dataclass
class WindowResult:
    """Walk-Forward窗口结果"""
    window_idx: int
    symbol: str
    timeframe: str
    train_period: Tuple[int, int]
    test_period: Tuple[int, int]
    train_result: Optional[BacktestResult]
    test_result: BacktestResult
    best_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "train_period": (str(self.train_period[0]), str(self.train_period[1])),
            "test_period": (str(self.test_period[0]), str(self.test_period[1])),
            "train_result": self.train_result.to_dict() if self.train_result else None,
            "test_result": self.test_result.to_dict(),
            "best_params": self.best_params
        }


@dataclass
class WalkForwardResult:
    """Walk-Forward总体结果"""
    config: BacktestConfig
    window_results: List[WindowResult]
    aggregated_metrics: PerformanceMetrics
    equity_curve: List[float]
    equity_times: List[int]
    drawdown_curve: List[float]
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "window_results": [wr.to_dict() for wr in self.window_results],
            "aggregated_metrics": self.aggregated_metrics.to_dict(),
            "equity_curve": self.equity_curve,
            "equity_times": self.equity_times,
            "drawdown_curve": self.drawdown_curve,
            "execution_time": self.execution_time
        }


@dataclass
class NormalBacktestResult:
    """普通回测（多标的 × 多周期）总体结果"""
    results: List[BacktestResult]
    aggregated_metrics: PerformanceMetrics
    combined_equity_times: List[int]
    combined_equity_values: List[float]
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregated_metrics": self.aggregated_metrics.to_dict(),
            "combined_equity_times": self.combined_equity_times,
            "combined_equity_values": self.combined_equity_values,
            "execution_time": self.execution_time,
        }

    @staticmethod
    def from_backtest_results(results: List[BacktestResult], execution_time) -> "NormalBacktestResult":
        # 聚合指标（简单平均）
        agg = PerformanceMetrics()
        if results:
            count = float(len(results))
            for f in fields(agg):
                value = sum(getattr(r.metrics, f.name, 0.0) or 0.0 for r in results) / count
                setattr(agg, f.name, value)

        # 组合净值（等权）：对齐时间，平均简单收益并复合
        union_times = sorted({t for r in results for t in (r.equity_times or [])})
        combined_values = []
        if union_times:
            initial_values = [float(r.equity_curve[0]) for r in results if r.equity_curve]
            combined_current = (sum(initial_values) / float(len(initial_values))) if initial_values else 1.0
            last_equities = []
            idx_by_result = []
            times_by_result = []
            for r in results:
                times_by_result.append(r.equity_times or [])
                idx_by_result.append(0)
                last_equities.append(float(r.equity_curve[0]) if r.equity_curve else 0.0)

            combined_values.append(combined_current)
            for t in union_times[1:]:
                step_returns = []
                for i, r in enumerate(results):
                    times = times_by_result[i]
                    idx = idx_by_result[i]
                    while idx + 1 < len(times) and times[idx + 1] <= t:
                        idx += 1
                        idx_by_result[i] = idx
                        last_equities[i] = float(r.equity_curve[idx])
                    if idx >= 1:
                        prev_eq = float(r.equity_curve[idx - 1])
                        curr_eq = float(r.equity_curve[idx])
                        ret = (curr_eq / prev_eq - 1.0) if prev_eq != 0 else 0.0
                    else:
                        ret = 0.0
                    step_returns.append(ret)
                avg_ret = sum(step_returns) / float(len(step_returns)) if step_returns else 0.0
                combined_current = combined_current * (1.0 + avg_ret)
                combined_values.append(combined_current)

        return NormalBacktestResult(
            results=results,
            aggregated_metrics=agg,
            combined_equity_times=union_times if results else [],
            combined_equity_values=combined_values if results else [],
            execution_time=execution_time,
        )

@dataclass
class FactorEvaluationConfig:
    """回测配置"""
    # 基础配置
    id: int
    name: str

    # 数据配置
    symbols: List[str] = None
    timeframes: List[TimeFrame] = None
    start_time: Union[int] = None
    end_time: Union[int] = None
    market: str = "okx"
    quote_currency: str = "USDT"
    ins_type: TradeInsType = TradeInsType.SWAP

    # 因子配置
    factor_classes: Dict[int, str] = None  # {factor_id: factor_class_name}
    factor_params: Dict[int, Dict[str, Any]] = None  # {factor_id: factor_params}
    data_source_class: str = None
    data_source_config: Dict[str, Any] = None
    future_periods: int = 1
    quantile_count: int = 5
    ic_window: Optional[int] = None  # IC计算窗口大小（None表示累计，int表示固定窗口大小）

    # 并行配置
    max_workers: int = 1

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = DateTimeUtils.to_timestamp(self.start_time)
        if isinstance(self.start_time, datetime):
            self.start_time = DateTimeUtils.datetime_to_timestamp(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = DateTimeUtils.to_timestamp(self.end_time)
        if isinstance(self.end_time, datetime):
            self.end_time = DateTimeUtils.datetime_to_timestamp(self.end_time)