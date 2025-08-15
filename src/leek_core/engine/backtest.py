#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

from leek_core.data import DataSource, DataSourceContext, ClickHouseKlineDataSource
from leek_core.engine import Engine
from leek_core.event import EventBus, Event, EventType, SerializableEventBus
from leek_core.executor import ExecutorContext, BacktestExecutor
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.models import (
    LeekComponentConfig,
    BacktestEngineConfig,
    Signal,
    Order,
    Position,
    PositionConfig,
    StrategyConfig,
    TimeFrame,
    TradeInsType,
)
from leek_core.models.data import KLine
from leek_core.strategy import StrategyContext, Strategy, StrategyWrapper
from leek_core.manager import StrategyManager
from leek_core.sub_strategy import EnterStrategy, ExitStrategy
from leek_core.position import PositionContext
from leek_core.analysis.performance import calculate_performance_from_values
from leek_core.utils import get_logger, run_func_timeout, DateTimeUtils, generate_str

logger = get_logger(__name__)


@dataclass
class DailyData:
    date: int
    position_value: Decimal
    signals: List[Signal]
    orders: List[Order]


class BacktestEngine(Engine):
    """
    回测指标:
    策略收益
    策略年化收益
    超额收益
    基准收益
    阿尔法
    贝塔
    夏普比率
    胜率
    盈亏比
    最大回撤
    索提诺比率
    日均超额收益
    超额收益最大回撤
    超额收益夏普比率
    日胜率
    盈利次数
    亏损次数
    信息比率
    策略波动率
    基准波动率
    最大回撤区间
    """
    def __init__(self, config: LeekComponentConfig[None, BacktestEngineConfig] = None):
        super().__init__()
        self.event_bus = EventBus()
        self.config = config
        self.strategy_manager: StrategyManager = StrategyManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-策略管理",
            cls=StrategyContext,
            config=None
        ))

        self.position_manager: PositionManager = PositionManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-仓位管理",
            cls=None,
            config=config.config.position_config
        ))
        self.executor_manager: ExecutorManager = ExecutorManager(
            self.event_bus, LeekComponentConfig(
            instance_id=config.instance_id,
            name=config.name + "-执行器管理",
            cls=ExecutorContext,
            config=None
        ))

        self.daily_data: List[DailyData] = []


    def on_start(self):
        ...

class WalkForwardOptimizer:
    def __init__(self, objective: Callable[[Dict[str, Any]], float] = None):
        if objective is None:
            objective = default_objective
        self.objective = objective

    def walk_forward(
        self,
        strategy: 'StrategySearchConfig',
        evaluation: 'EvaluationConfig',
        executor_cfg: 'ExecutorConfig',
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward optimization.

        on_progress: optional callback. Called as on_progress(done, total)
        when each window finishes evaluation.
        """
        results: List[Dict[str, Any]] = []
        base_bt_cfg = (executor_cfg.executor_cfg or {}).copy()
        if executor_cfg.fee_rate is not None:
            base_bt_cfg["fee_rate"] = float(executor_cfg.fee_rate)
        if executor_cfg.slippage_bps is not None:
            base_bt_cfg["slippage_bps"] = float(executor_cfg.slippage_bps)

        # Build all evaluation jobs at granularity: (symbol, timeframe, window)
        pairs: List[Tuple[str, TimeFrame]] = [
            (symbol, timeframe)
            for symbol in evaluation.symbols
            for timeframe in evaluation.timeframes
        ]
        windows: List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]] = list(
            generate_windows(
                evaluation.start,
                evaluation.end,
                evaluation.train_days,
                evaluation.test_days,
                evaluation.embargo_days,
                evaluation.mode,
            )
        )
        jobs: List[Tuple[str, TimeFrame, Tuple[Any, Any], Tuple[Any, Any]]] = [
            (symbol, timeframe, (tr_s, tr_e), (te_s, te_e))
            for symbol, timeframe in pairs
            for (tr_s, tr_e), (te_s, te_e) in windows
        ]

        total = max(1, len(jobs))
        from threading import Lock
        progress_lock = Lock()
        done = 0

        def eval_job(symbol: str, timeframe: TimeFrame, tr: Tuple[Any, Any], te: Tuple[Any, Any]) -> Dict[str, Any]:
            nonlocal done
            (tr_s, tr_e) = tr
            (te_s, te_e) = te
            bt = StrategyBacktester(
                symbol=symbol,
                timeframe=timeframe,
                market=executor_cfg.market,
                quote_currency=executor_cfg.quote_currency,
                ins_type=executor_cfg.ins_type,
                data_source=executor_cfg.data_source or ClickHouseKlineDataSource(),
                initial_balance=executor_cfg.initial_balance,
                executor=executor_cfg.executor,
                executor_cfg=base_bt_cfg,
            )
            best_params: Optional[Dict[str, Any]] = None
            best_score: float = float("-inf")
            candidates = list(param_grid(strategy.param_space))
            train_length_ok = True
            try:
                ts = _to_datetime(tr_s)
                te_dt = _to_datetime(tr_e)
                train_length_ok = te_dt > ts and int(evaluation.train_days or 0) > 0
            except Exception:
                train_length_ok = False
            if not train_length_ok:
                best_params = candidates[0] if candidates else {}
            else:
                folds = _split_cv_windows(
                    _to_datetime(tr_s),
                    _to_datetime(tr_e),
                    int(evaluation.cv_splits or 0),
                    int(evaluation.embargo_days or 0),
                )
                for p in candidates:
                    scores: List[float] = []
                    for fs, fe in folds:
                        train_res = bt.run(strategy.strategy_cls, p, fs, fe)
                        scores.append(self.objective(train_res.metrics))
                    score = _median(scores)
                    if score > best_score:
                        best_params, best_score = p, score
            assert best_params is not None, "参数搜索未产生结果"
            test_res = bt.run(strategy.strategy_cls, best_params, te_s, te_e)
            res = {
                "symbol": symbol,
                "timeframe": timeframe.value if hasattr(timeframe, "value") else str(timeframe),
                "train": (tr_s, tr_e),
                "test": (te_s, te_e),
                "params": best_params,
                "train_score": float(best_score),
                "test_sharpe": float(test_res.metrics.get("sharpe", 0.0)),
                "test_mdd": float(test_res.metrics.get("max_drawdown", 0.0)),
                "test_trades": int(test_res.metrics.get("trades", 0)),
                "win_trades": int(test_res.metrics.get("win_trades", 0)),
                "test_turnover": float(test_res.metrics.get("turnover", 0.0)),
                "test_pnl": float(test_res.metrics.get("pnl_net", 0.0)),
                # 盈亏相关指标（窗口级）
                "profit_factor": (lambda v: float(v) if v is not None else None)(test_res.metrics.get("profit_factor")),
                "avg_win": (lambda v: float(v) if v is not None else None)(test_res.metrics.get("avg_win")),
                "avg_loss": (lambda v: float(v) if v is not None else None)(test_res.metrics.get("avg_loss")),
                "win_loss_ratio": (lambda v: float(v) if v is not None else None)(test_res.metrics.get("win_loss_ratio")),
                "equity_values": [float(v) for v in (getattr(test_res, "equity_values", []) or [])],
                "equity_times": list(getattr(test_res, "equity_times", []) or []),
            }
            if on_progress:
                with progress_lock:
                    done += 1
                    on_progress(done, total)
            return res

        # Execute jobs with optional concurrency
        max_workers = max(1, int(evaluation.max_workers or 1))
        if max_workers == 1:
            for symbol, timeframe, tr, te in jobs:
                results.append(eval_job(symbol, timeframe, tr, te))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = [executor.submit(eval_job, symbol, timeframe, tr, te) for (symbol, timeframe, tr, te) in jobs]
                for fut in as_completed(futs):
                    results.append(fut.result())
        summary = self.summarize_results(
            results,
            sharpe_median_min=evaluation.sharpe_median_min,
            sharpe_p25_min=evaluation.sharpe_p25_min,
            mdd_median_max=evaluation.mdd_median_max,
            min_trades_per_window=evaluation.min_trades_per_window,
        )
        return {"windows": results, "summary": summary}

    def summarize_results(
        self,
        results: List[Dict[str, Any]],
        sharpe_median_min: Optional[float] = None,
        sharpe_p25_min: Optional[float] = None,
        mdd_median_max: Optional[float] = None,
        min_trades_per_window: int = 0,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in results:
            key = (str(r["symbol"]), str(r["timeframe"]))
            groups.setdefault(key, []).append(r)

        summary: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, rows in groups.items():
            sharpe_list = [float(x.get("test_sharpe", 0.0)) for x in rows]
            mdd_list = [float(x.get("test_mdd", 0.0)) for x in rows]
            trades_list = [int(x.get("test_trades", 0)) for x in rows]
            turnover_list = [float(x.get("test_turnover", 0.0)) for x in rows]
            pnl_list = [float(x.get("test_pnl", 0.0)) for x in rows]
            # 窗口胜率：窗口级别盈利占比
            wins = sum(1 for x in rows if float(x.get("test_pnl", 0.0)) > 0.0)
            win_rate = float(wins) / float(len(rows)) if rows else 0.0
            # 交易胜率：统计各窗口的胜率（若窗口提供 win_trades/total_trades），再加权平均
            trade_win = 0
            trade_total = 0
            for x in rows:
                wt = int(x.get("win_trades", 0))
                tt = int(x.get("test_trades", 0))
                trade_win += wt
                trade_total += tt
            trade_win_rate = float(trade_win) / float(trade_total) if trade_total > 0 else None

            agg = {
                "windows": len(rows),
                "sharpe_median": _median(sharpe_list),
                "sharpe_p25": _percentile(sharpe_list, 25.0),
                "mdd_median": _median(mdd_list),
                "trades_median": _median(trades_list),
                "turnover_median": _median(turnover_list),
                "pnl_median": _median(pnl_list),
                "win_rate": win_rate,  # 窗口胜率
                "trade_win_rate": trade_win_rate,
            }

            pass_flag = True
            if sharpe_median_min is not None and agg["sharpe_median"] < sharpe_median_min:
                pass_flag = False
            if sharpe_p25_min is not None and agg["sharpe_p25"] < sharpe_p25_min:
                pass_flag = False
            if mdd_median_max is not None and abs(agg["mdd_median"]) > abs(mdd_median_max):
                pass_flag = False
            if min_trades_per_window > 0 and any(t < min_trades_per_window for t in trades_list):
                pass_flag = False
            agg["pass_thresholds"] = bool(pass_flag)

            summary[key] = agg

        return summary
############################
# Lightweight backtester API
############################

def _to_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return DateTimeUtils.to_datetime(DateTimeUtils.to_timestamp(value))


def param_grid(space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    from itertools import product
    keys = list(space.keys())
    values = [list(v) for v in space.values()]
    for vals in product(*values):
        yield dict(zip(keys, vals))


def generate_windows(
    start: datetime | str,
    end: datetime | str,
    train_days: int,
    test_days: int,
    embargo_days: int = 0,
    mode: str = "rolling",
) -> Iterable[Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]]:
    """Generate train/test windows.

    Rules:
    - Step by test_days (non-overlapping test windows)
    - Expanding: train_start = start, train_end = current_train_end
    - Rolling:   train length = train_days, train window precedes each test
    - If final leftover test window length < 0.8 * test_days, merge it into the previous window
    - If test_days<=0 or total_days < test_days: single test window = entire period
    """
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end)
    total_days = (end_dt - start_dt).days
    # Test-only
    if test_days <= 0 or total_days < test_days:
        train_end = start_dt if train_days <= 0 else start_dt + timedelta(days=train_days)
        yield (start_dt, train_end), (start_dt, end_dt)
        return

    windows: List[Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]] = []
    cur_train_anchor = start_dt
    while True:
        if mode == "expanding":
            train_start = start_dt
            train_end = cur_train_anchor + timedelta(days=max(0, train_days))
        else:
            train_start = cur_train_anchor
            train_end = train_start + timedelta(days=max(0, train_days))
        test_start = train_end + timedelta(days=max(0, embargo_days))
        test_end = test_start + timedelta(days=test_days)
        if test_start >= end_dt:
            break
        if test_end > end_dt:
            # leftover segment
            leftover_days = (end_dt - test_start).days
            if windows and leftover_days < int(0.8 * test_days):
                # merge leftover into previous test window
                prev_train, (prev_ts, prev_te) = windows[-1]
                windows[-1] = (prev_train, (prev_ts, end_dt))
            else:
                windows.append(((train_start, train_end), (test_start, end_dt)))
            break
        windows.append(((train_start, train_end), (test_start, test_end)))
        # step by test period
        cur_train_anchor = test_end

    for w in windows:
        yield w


def default_objective(metrics: Dict[str, Any]) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    mdd = abs(float(metrics.get("max_drawdown", 0.0)))
    turnover = float(metrics.get("turnover", 0.0))
    return sharpe - 0.5 * mdd - 0.1 * turnover


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[int(k)])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def _split_cv_windows(
    start: datetime, end: datetime, cv_splits: int, embargo_days: int = 0
) -> List[Tuple[datetime, datetime]]:
    if cv_splits <= 1:
        return [(start, end)]
    total_days = (end - start).days
    fold_days = max(1, total_days // cv_splits)
    folds: List[Tuple[datetime, datetime]] = []
    cur = start
    for i in range(cv_splits):
        fs = cur
        fe = fs + timedelta(days=fold_days)
        if i == cv_splits - 1:
            fe = end
        folds.append((fs, fe))
        cur = fe + timedelta(days=embargo_days)
        if cur >= end:
            break
    return folds


@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    equity_values: List[Decimal]
    equity_times: List[int]
    trades: int


@dataclass
class ExecutorConfig:
    market: str = "okx"
    quote_currency: str = "USDT"
    ins_type: TradeInsType = TradeInsType.SWAP
    data_source: DataSource | None = None
    executor: type[ExecutorContext] = BacktestExecutor
    executor_cfg: Optional[Dict[str, Any]] = None
    initial_balance: Decimal | float = Decimal("10000")
    fee_rate: Optional[float] = None
    slippage_bps: Optional[float] = None


@dataclass
class StrategySearchConfig:
    strategy_cls: type[Strategy]
    param_space: Dict[str, Iterable[Any]]


@dataclass
class EvaluationConfig:
    symbols: List[str]
    timeframes: List[TimeFrame]
    start: datetime | str
    end: datetime | str
    train_days: int
    test_days: int
    embargo_days: int = 0
    mode: str = "rolling"
    cv_splits: int = 0
    max_workers: int = 1
    sharpe_median_min: Optional[float] = None
    sharpe_p25_min: Optional[float] = None
    mdd_median_max: Optional[float] = None
    min_trades_per_window: int = 0
    export_csv_path: Optional[str] = None
    summary_export_csv_path: Optional[str] = None


class StrategyBacktester:
    def __init__(
        self,
        symbol: str = "ETH",
        timeframe: TimeFrame = TimeFrame.M5,
        market: str = "okx",
        quote_currency: str = "USDT",
        ins_type: TradeInsType = TradeInsType.SWAP,
        data_source: DataSource | None = None,
        initial_balance: Decimal | float = Decimal("10000"),
        executor: type[ExecutorContext] = BacktestExecutor,
        executor_cfg: Dict[str, Any] | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.market = market
        self.quote_currency = quote_currency
        self.ins_type = ins_type
        # 数据源：优先使用传入实例；否则按 executor_cfg 中提供的类与参数动态创建
        if data_source is not None:
            self.data_source = data_source
        else:
            ds = None
            try:
                ds_cls_path = (executor_cfg or {}).get("data_source_cls")
                ds_params = (executor_cfg or {}).get("data_source_params") or {}
                # ensure mount dirs are added for dynamic imports in worker threads
                try:
                    mount_dirs = (executor_cfg or {}).get("mount_dirs") or []
                    import sys, os
                    for d in mount_dirs:
                        if d == "default":
                            continue
                        if d and os.path.isdir(d) and d not in sys.path:
                            sys.path.append(d)
                except Exception:
                    ...
                if ds_cls_path:
                    from leek_core.base import load_class_from_str, create_component
                    ds_cls = load_class_from_str(ds_cls_path)
                    ds = create_component(cls=ds_cls, **ds_params)
            except Exception:
                ds = None
            self.data_source = ds or ClickHouseKlineDataSource()
        self.initial_balance = Decimal(str(initial_balance))
        self.executor_cls = executor
        self.executor_cfg = executor_cfg or {}

    def run(
        self,
        strategy_cls: type[Strategy],
        params: Dict[str, Any],
        start: datetime | str,
        end: datetime | str,
        enter_strategy: EnterStrategy | None = None,
        exit_strategy: ExitStrategy | None = None,
    ) -> BacktestResult:
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end)

        event_bus = SerializableEventBus()
        # 统一使用一个计数器来避免重复计数  
        trades_win_counter = {"win": 0, "loss": 0, "total": 0}
        # 不做前后态对比，按事件当前态判定即可（该事件不会重复推送）
        last_pos_sz: dict[str, float] = {}
        # 统计盈亏金额：用于盈亏比（Profit Factor）和平均盈亏比
        trade_profit_sum = 0.0  # 累计盈利（>0）
        trade_loss_sum = 0.0    # 累计亏损的绝对值（>0）

        def _on_position_update(event: Event):
            """基于仓位平仓事件统计交易次数和盈亏"""
            try:
                if isinstance(event.data, Position):
                    pos: Position = event.data
                    cur_sz = float(pos.sz or 0)
                    total_sz = float(pos.total_sz or 0)
                    # 当仓位从持有状态变为平仓状态，计为一次交易
                    if total_sz > 0 and cur_sz == 0:
                        try:
                            pnl_val = float(pos.pnl or 0)
                            # 统计所有平仓交易，包括无盈无亏的
                            trades_win_counter["total"] += 1
                            if pnl_val > 0:
                                trades_win_counter["win"] += 1
                                trade_profit_sum += pnl_val
                            elif pnl_val < 0:
                                trades_win_counter["loss"] += 1
                                trade_loss_sum += abs(pnl_val)
                            # pnl_val == 0 的情况也计入总交易数，但不计入盈亏
                        except Exception:
                            ...
            except Exception:
                ...

        event_bus.subscribe_event(EventType.POSITION_UPDATE, _on_position_update)

        # 用 Manager 包装 PositionContext，统一事件订阅
        from leek_core.manager import PositionManager
        position_manager: PositionManager = PositionManager(
            event_bus,
            LeekComponentConfig(
                instance_id="p0",
                name="仓位管理",
                cls=None,
                config=PositionConfig(
                    init_amount=self.initial_balance,
                    max_amount=self.initial_balance * 1000,
                    max_strategy_amount=self.initial_balance * 1000,
                    max_strategy_ratio=Decimal("0.5"),
                    max_symbol_amount=self.initial_balance * 1000,
                    max_symbol_ratio=Decimal("1"),
                    max_ratio=Decimal("1"),
                ),
            ),
        )
        position_manager.on_start()

        executor_manager: ExecutorManager = ExecutorManager(
            event_bus,
            LeekComponentConfig(
                instance_id="p1",
                name="执行器管理",
                cls=ExecutorContext,
                config=None,
            ),
        )
        executor_manager.on_start()
        executor_manager.add(
            LeekComponentConfig(
                instance_id="p3",
                name="执行器",
                cls=self.executor_cls,
                config=self.executor_cfg,
            )
        )

        enter = enter_strategy or EnterStrategy()
        exit_ = exit_strategy or ExitStrategy()
        # 使用 StrategyManager + StrategyContext 管理策略
        strat_manager: StrategyManager = StrategyManager(
            event_bus,
            LeekComponentConfig(instance_id="s0", name="策略管理", cls=StrategyContext, config=None),
        )
        strat_manager.on_start()
        strat_cfg = StrategyConfig(
            data_source_configs=[],
            info_fabricator_configs=[],
            strategy_config=params or {},
            strategy_position_config=None,
            enter_strategy_cls=EnterStrategy,
            enter_strategy_config={},
            exit_strategy_cls=ExitStrategy,
            exit_strategy_config={},
            risk_policies=[],
        )
        strat_component = LeekComponentConfig(
            instance_id="s1",
            name="策略",
            cls=strategy_cls,
            config=strat_cfg,
        )
        strat_manager.add(strat_component)
        strat_ctx = strat_manager.components.get("s1")

        assert self.data_source.connect(), "数据源连接失败"
        equity_values: List[Decimal] = []
        equity_times: List[int] = []
        last_position_rate: Decimal = Decimal("0")
        turnover_acc: Decimal = Decimal("0")

        for kline in self.data_source.get_history_data(
            start_time=start_dt,
            end_time=end_dt,
            row_key=KLine.pack_row_key(
                self.symbol, self.quote_currency, self.ins_type, self.timeframe
            ),
            market=self.market,
        ):
            # 直接驱动策略上下文处理数据（简化数据通路）
            strat_ctx.process_data(kline)
            # 读取当前实例key对应的策略实例仓位
            key = strat_ctx.strategy_mode.build_instance_key(kline)
            wrapper = strat_ctx.strategies.get(key)
            current_rate = wrapper.position_rate if wrapper else last_position_rate
            turnover_acc += abs(current_rate - last_position_rate)
            last_position_rate = current_rate
            # 推给仓位管理处理
            position_manager.process_data_update(Event(EventType.DATA_RECEIVED, kline))
            equity_values.append(position_manager.position_context.value)
            try:
                equity_times.append(int(getattr(kline, "current_time", None) or getattr(kline, "end_time", None) or 0))
            except Exception:
                equity_times.append(0)

        self.data_source.disconnect()

        perf = calculate_performance_from_values(equity_values)
        sharpe = float(perf.get("sharpe_ratio", 0.0))
        mdd = float(perf.get("max_drawdown", {}).get("max_drawdown", 0.0))

        # 计算盈亏比/平均盈亏
        wins = int(trades_win_counter["win"])
        losses = int(trades_win_counter["loss"])
        total_trades = int(trades_win_counter["total"])
        profit_factor = (float(trade_profit_sum) / float(trade_loss_sum)) if trade_loss_sum > 0 else None
        avg_win = (float(trade_profit_sum) / float(wins)) if wins > 0 else None
        avg_loss = (float(trade_loss_sum) / float(losses)) if losses > 0 else None
        win_loss_ratio = (avg_win / avg_loss) if (avg_win is not None and avg_loss and avg_loss > 0) else None

        result = {
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "turnover": float(turnover_acc),
            "trades": int(trades_win_counter["total"]),
            "win_trades": wins,
            "profit_sum": float(trade_profit_sum),
            "loss_sum": float(trade_loss_sum),
            "profit_count": int(wins),
            "loss_count": int(losses),
            "pnl_net": float(equity_values[-1] - equity_values[0]) if equity_values else 0.0,
            "profit_factor": float(profit_factor) if profit_factor is not None else None,
            "avg_win": float(avg_win) if avg_win is not None else None,
            "avg_loss": float(avg_loss) if avg_loss is not None else None,
            "win_loss_ratio": float(win_loss_ratio) if win_loss_ratio is not None else None,
        }
        return BacktestResult(metrics=result, equity_values=equity_values, equity_times=equity_times, trades=result["trades"])




