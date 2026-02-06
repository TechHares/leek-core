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

    # K线模拟选项
    simulate_kline: bool = False  # 是否启用K线模拟（固定使用1分钟K线）

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

    # K线模拟选项
    simulate_kline: bool = False  # 是否启用K线模拟（固定使用1分钟K线）

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

    def _calculate_grade(self) -> str:
        """基于多维度指标计算策略等级 (A+/A/B+/B/C/D/F)"""
        score = 0
        
        # 收益维度 (0-25分)
        if self.annual_return > 0.5:
            score += 25
        elif self.annual_return > 0.3:
            score += 20
        elif self.annual_return > 0.15:
            score += 15
        elif self.annual_return > 0:
            score += 10
        elif self.annual_return > -0.1:
            score += 5
        
        # 风险调整收益维度 (0-25分)
        if self.sharpe_ratio > 2.0:
            score += 25
        elif self.sharpe_ratio > 1.5:
            score += 20
        elif self.sharpe_ratio > 1.0:
            score += 15
        elif self.sharpe_ratio > 0.5:
            score += 10
        elif self.sharpe_ratio > 0:
            score += 5
        
        # 回撤控制维度 (0-25分)
        if self.max_drawdown > -0.1:
            score += 25
        elif self.max_drawdown > -0.2:
            score += 20
        elif self.max_drawdown > -0.3:
            score += 15
        elif self.max_drawdown > -0.5:
            score += 10
        elif self.max_drawdown > -0.7:
            score += 5
        
        # 统计显著性维度 (0-25分)
        if self.t_pvalue < 0.01 and self.total_trades >= 50:
            score += 25
        elif self.t_pvalue < 0.05 and self.total_trades >= 30:
            score += 20
        elif self.t_pvalue < 0.1 and self.total_trades >= 20:
            score += 15
        elif self.total_trades >= 30:
            score += 10
        elif self.total_trades >= 10:
            score += 5
        
        # 转换为等级
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def _identify_concerns(self) -> List[str]:
        """识别策略潜在风险和问题"""
        concerns = []
        
        # 交易次数不足
        if self.total_trades < 30:
            concerns.append(f"交易次数不足({self.total_trades}笔)，统计结果可能不可靠")
        
        # 统计显著性问题
        if self.t_pvalue >= 0.05:
            concerns.append(f"收益统计不显著(p={self.t_pvalue:.3f})，可能是随机结果")
        
        # 回撤过大
        if self.max_drawdown < -0.3:
            concerns.append(f"最大回撤过大({self.max_drawdown:.1%})，风险较高")
        
        # 夏普比率过低
        if self.sharpe_ratio < 0.5 and self.total_return > 0:
            concerns.append(f"夏普比率偏低({self.sharpe_ratio:.2f})，风险调整后收益不佳")
        
        # 胜率与盈亏比不匹配
        if self.win_rate > 0 and self.win_loss_ratio > 0:
            expectancy = self.win_rate * self.win_loss_ratio - (1 - self.win_rate)
            if expectancy < 0.1:
                concerns.append("期望收益较低，胜率与盈亏比组合不佳")
        
        # 多空不平衡
        if self.long_trades > 0 and self.short_trades > 0:
            if self.long_win_rate > 0 and self.short_win_rate > 0:
                if abs(self.long_win_rate - self.short_win_rate) > 0.2:
                    concerns.append("多空胜率差异较大，策略可能存在方向偏好")
        
        # 单边交易
        if self.long_trades == 0 or self.short_trades == 0:
            direction = "只做多" if self.long_trades > 0 else "只做空"
            concerns.append(f"策略{direction}，未充分利用双向交易机会")
        
        # 最大单笔亏损过大
        if self.largest_loss < -0.1:
            concerns.append(f"最大单笔亏损过大({self.largest_loss:.1%})，建议加强止损")
        
        # 收益分布偏态
        if self.skewness < -1:
            concerns.append("收益分布负偏，存在较多极端亏损")
        
        # VaR 风险
        if self.var_95 < -0.03:
            concerns.append(f"95% VaR较高({self.var_95:.1%})，日波动风险较大")
        
        return concerns

    def _generate_text_summary(self) -> str:
        """生成一句话策略评估摘要"""
        parts = []
        
        # 收益评估
        if self.annual_return > 0.5:
            parts.append(f"年化收益优秀({self.annual_return:.0%})")
        elif self.annual_return > 0.2:
            parts.append(f"年化收益良好({self.annual_return:.0%})")
        elif self.annual_return > 0:
            parts.append(f"年化收益一般({self.annual_return:.0%})")
        else:
            parts.append(f"策略亏损({self.annual_return:.0%})")
        
        # 风险评估
        if self.max_drawdown > -0.1:
            parts.append("回撤控制优秀")
        elif self.max_drawdown > -0.2:
            parts.append("回撤可接受")
        elif self.max_drawdown > -0.3:
            parts.append("回撤偏大")
        else:
            parts.append(f"回撤过大({self.max_drawdown:.0%})")
        
        # 风险调整收益
        if self.sharpe_ratio > 1.5:
            parts.append(f"夏普比率优秀({self.sharpe_ratio:.2f})")
        elif self.sharpe_ratio > 1.0:
            parts.append(f"夏普比率良好({self.sharpe_ratio:.2f})")
        elif self.sharpe_ratio > 0.5:
            parts.append(f"夏普比率一般({self.sharpe_ratio:.2f})")
        else:
            parts.append(f"夏普比率偏低({self.sharpe_ratio:.2f})")
        
        # 统计显著性
        if self.t_pvalue < 0.01:
            parts.append("结果高度显著")
        elif self.t_pvalue < 0.05:
            parts.append("结果统计显著")
        elif self.t_pvalue < 0.1:
            parts.append("结果边缘显著")
        else:
            parts.append(f"结果可能是随机的(p={self.t_pvalue:.2f})")
        
        return "，".join(parts)

    def to_agent_summary(self) -> Dict[str, Any]:
        """
        生成 AI Agent 友好的策略评估摘要
        
        返回精简的、结构化的评估数据，适合 AI 分析和决策。
        包含约15个核心指标 + 自动评估结论，而不是全部50+指标。
        """
        return {
            # ===== 核心收益指标 (5个) =====
            "total_return": round(self.total_return, 4),
            "annual_return": round(self.annual_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            
            # ===== 交易质量指标 (5个) =====
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "avg_return_per_trade": round(self.avg_return_per_trade, 4),
            "win_loss_ratio": round(self.win_loss_ratio, 3),
            
            # ===== 统计显著性 (4个) =====
            "t_pvalue": round(self.t_pvalue, 4),
            "bootstrap_sharpe_ci": [
                round(self.bootstrap_sharpe_ci_lower, 3),
                round(self.bootstrap_sharpe_ci_upper, 3)
            ],
            "bootstrap_annual_return_ci": [
                round(self.bootstrap_annual_return_ci_lower, 4),
                round(self.bootstrap_annual_return_ci_upper, 4)
            ],
            "alpha_pvalue": round(self.alpha_pvalue, 4),
            
            # ===== 自动评估结论 =====
            "evaluation": {
                "overall_grade": self._calculate_grade(),
                "is_profitable": self.total_return > 0,
                "is_statistically_significant": self.t_pvalue < 0.05,
                "risk_adjusted_good": self.sharpe_ratio > 1.0,
                "drawdown_acceptable": self.max_drawdown > -0.3,
                "has_enough_trades": self.total_trades >= 30,
                "concerns": self._identify_concerns(),
                "summary": self._generate_text_summary()
            }
        }


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

    def to_agent_summary(self) -> Dict[str, Any]:
        """
        生成 AI Agent 友好的回测结果摘要
        
        返回精简的回测信息，去除大量时间序列数据，保留核心评估信息。
        适合 AI 分析策略是否可用、是否需要优化。
        """
        # 配置摘要
        config_summary = {}
        if self.config:
            config_summary = {
                "symbol": self.config.get("symbol"),
                "timeframe": str(self.config.get("timeframe")),
                "initial_balance": float(self.config.get("initial_balance", 0)),
                "strategy_class": self.config.get("strategy_class"),
                "strategy_params": self.config.get("strategy_params"),
            }
        
        # 交易摘要统计
        trade_summary = {}
        if self.trades:
            pnls = [t.get("pnl", 0) for t in self.trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            trade_summary = {
                "total_count": len(self.trades),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
                "best_trade": round(max(pnls), 2) if pnls else 0,
                "worst_trade": round(min(pnls), 2) if pnls else 0,
                "avg_winning": round(sum(winning_trades) / len(winning_trades), 2) if winning_trades else 0,
                "avg_losing": round(sum(losing_trades) / len(losing_trades), 2) if losing_trades else 0,
            }
        
        # 净值曲线摘要（只保留关键点）
        equity_summary = {}
        if self.equity_curve and len(self.equity_curve) > 0:
            equity_summary = {
                "initial": round(self.equity_curve[0], 2),
                "final": round(self.equity_curve[-1], 2),
                "peak": round(max(self.equity_curve), 2),
                "trough": round(min(self.equity_curve), 2),
                "data_points": len(self.equity_curve),
            }
        
        # 元数据
        meta_summary = {}
        if self.metadata:
            meta_summary = {
                "total_bars": self.metadata.get("total_bars", 0),
                "turnover": round(self.metadata.get("turnover", 0), 4),
            }
        
        return {
            "config": config_summary,
            "metrics": self.metrics.to_agent_summary(),
            "trade_summary": trade_summary,
            "equity_summary": equity_summary,
            "metadata": meta_summary,
            "execution_time": round(self.execution_time, 2),
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

    def to_agent_summary(self) -> Dict[str, Any]:
        """
        生成 AI Agent 友好的 Walk-Forward 验证结果摘要
        
        重点展示样本外测试表现、参数稳定性、过拟合风险评估。
        """
        # 配置摘要
        config_summary = {
            "strategy_class": self.config.strategy_class,
            "symbols": self.config.symbols,
            "timeframes": [str(tf) for tf in (self.config.timeframes or [])],
            "train_days": self.config.train_days,
            "test_days": self.config.test_days,
            "window_mode": self.config.wf_window_mode,
        }
        
        # 窗口结果摘要
        window_summaries = []
        train_sharpes = []
        test_sharpes = []
        params_history = []
        
        for wr in self.window_results:
            train_sharpe = wr.train_result.metrics.sharpe_ratio if wr.train_result else None
            test_sharpe = wr.test_result.metrics.sharpe_ratio
            
            if train_sharpe is not None:
                train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)
            params_history.append(wr.best_params)
            
            window_summaries.append({
                "window_idx": wr.window_idx,
                "train_sharpe": round(train_sharpe, 3) if train_sharpe else None,
                "test_sharpe": round(test_sharpe, 3),
                "train_return": round(wr.train_result.metrics.total_return, 4) if wr.train_result else None,
                "test_return": round(wr.test_result.metrics.total_return, 4),
                "best_params": wr.best_params,
            })
        
        # 过拟合分析
        overfit_analysis = {}
        if train_sharpes and test_sharpes and len(train_sharpes) == len(test_sharpes):
            avg_train = sum(train_sharpes) / len(train_sharpes)
            avg_test = sum(test_sharpes) / len(test_sharpes)
            decay_ratio = (avg_train - avg_test) / avg_train if avg_train > 0 else 0
            
            overfit_analysis = {
                "avg_train_sharpe": round(avg_train, 3),
                "avg_test_sharpe": round(avg_test, 3),
                "performance_decay": round(decay_ratio, 3),  # 越大越可能过拟合
                "is_likely_overfit": decay_ratio > 0.5,  # 衰减超过50%认为可能过拟合
                "consistent_windows": sum(1 for ts in test_sharpes if ts > 0),  # 正收益窗口数
                "total_windows": len(test_sharpes),
            }
        
        # 参数稳定性分析
        param_stability = {}
        if params_history and len(params_history) > 1:
            # 检查参数是否稳定
            all_params = set()
            for p in params_history:
                all_params.update(p.keys())
            
            param_changes = {}
            for param in all_params:
                values = [p.get(param) for p in params_history if param in p]
                if len(set(values)) == 1:
                    param_changes[param] = "stable"
                elif len(set(values)) <= len(values) / 2:
                    param_changes[param] = "moderate"
                else:
                    param_changes[param] = "unstable"
            
            param_stability = {
                "parameter_changes": param_changes,
                "is_params_stable": all(v == "stable" for v in param_changes.values()),
            }
        
        # 净值曲线摘要
        equity_summary = {}
        if self.equity_curve and len(self.equity_curve) > 0:
            equity_summary = {
                "initial": round(self.equity_curve[0], 2),
                "final": round(self.equity_curve[-1], 2),
                "peak": round(max(self.equity_curve), 2),
                "trough": round(min(self.equity_curve), 2),
            }
        
        return {
            "config": config_summary,
            "metrics": self.aggregated_metrics.to_agent_summary(),
            "window_count": len(self.window_results),
            "windows": window_summaries,
            "overfit_analysis": overfit_analysis,
            "param_stability": param_stability,
            "equity_summary": equity_summary,
            "execution_time": round(self.execution_time, 2),
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

    def to_agent_summary(self) -> Dict[str, Any]:
        """
        生成 AI Agent 友好的多标的回测结果摘要
        
        展示各标的表现对比、策略普适性评估。
        """
        # 各标的结果摘要
        result_summaries = []
        symbols_performance = {}
        timeframes_performance = {}
        
        for r in self.results:
            symbol = r.config.get("symbol", "unknown") if r.config else "unknown"
            timeframe = str(r.config.get("timeframe", "unknown")) if r.config else "unknown"
            
            summary = {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_return": round(r.metrics.total_return, 4),
                "sharpe_ratio": round(r.metrics.sharpe_ratio, 3),
                "max_drawdown": round(r.metrics.max_drawdown, 4),
                "win_rate": round(r.metrics.win_rate, 3),
                "total_trades": r.metrics.total_trades,
            }
            result_summaries.append(summary)
            
            # 按标的统计
            if symbol not in symbols_performance:
                symbols_performance[symbol] = []
            symbols_performance[symbol].append(r.metrics.sharpe_ratio)
            
            # 按周期统计
            if timeframe not in timeframes_performance:
                timeframes_performance[timeframe] = []
            timeframes_performance[timeframe].append(r.metrics.sharpe_ratio)
        
        # 标的普适性分析
        symbol_analysis = {}
        if symbols_performance:
            avg_by_symbol = {
                s: round(sum(v) / len(v), 3) 
                for s, v in symbols_performance.items()
            }
            positive_symbols = sum(1 for v in avg_by_symbol.values() if v > 0)
            symbol_analysis = {
                "avg_sharpe_by_symbol": avg_by_symbol,
                "positive_symbols": positive_symbols,
                "total_symbols": len(symbols_performance),
                "universality_ratio": round(positive_symbols / len(symbols_performance), 2) if symbols_performance else 0,
            }
        
        # 周期普适性分析
        timeframe_analysis = {}
        if timeframes_performance:
            avg_by_tf = {
                tf: round(sum(v) / len(v), 3) 
                for tf, v in timeframes_performance.items()
            }
            positive_tfs = sum(1 for v in avg_by_tf.values() if v > 0)
            timeframe_analysis = {
                "avg_sharpe_by_timeframe": avg_by_tf,
                "positive_timeframes": positive_tfs,
                "total_timeframes": len(timeframes_performance),
            }
        
        # 组合净值摘要
        equity_summary = {}
        if self.combined_equity_values and len(self.combined_equity_values) > 0:
            equity_summary = {
                "initial": round(self.combined_equity_values[0], 2),
                "final": round(self.combined_equity_values[-1], 2),
                "peak": round(max(self.combined_equity_values), 2),
                "trough": round(min(self.combined_equity_values), 2),
            }
        
        # 整体评估
        overall_evaluation = {
            "is_universal": symbol_analysis.get("universality_ratio", 0) >= 0.7,  # 70%标的正收益
            "best_symbol": max(result_summaries, key=lambda x: x["sharpe_ratio"])["symbol"] if result_summaries else None,
            "worst_symbol": min(result_summaries, key=lambda x: x["sharpe_ratio"])["symbol"] if result_summaries else None,
            "strategy_consistency": "consistent" if symbol_analysis.get("universality_ratio", 0) >= 0.8 else (
                "moderate" if symbol_analysis.get("universality_ratio", 0) >= 0.5 else "inconsistent"
            ),
        }
        
        return {
            "metrics": self.aggregated_metrics.to_agent_summary(),
            "result_count": len(self.results),
            "results": result_summaries,
            "symbol_analysis": symbol_analysis,
            "timeframe_analysis": timeframe_analysis,
            "overall_evaluation": overall_evaluation,
            "equity_summary": equity_summary,
            "execution_time": round(self.execution_time, 2),
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
    
    # AlphaEval 配置
    enable_robustness: bool = False  # 是否启用鲁棒性评估
    robustness_noise_level: float = 0.05  # 鲁棒性测试噪声水平
    robustness_trials: int = 5  # 鲁棒性测试次数
    scoring_weights: Optional[Dict[str, float]] = None  # 综合评分权重配置

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