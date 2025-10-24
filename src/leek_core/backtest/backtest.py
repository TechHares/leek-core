#!/usr/bin/env python
# -*- coding: utf-8 -*-
import concurrent
import time
from dataclasses import fields
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable, Union, Dict, List, Tuple, Any

from leek_core.utils import get_logger, DateTimeUtils

from .types import BacktestConfig, BacktestResult, WalkForwardResult, BacktestMode, NormalBacktestResult, PerformanceMetrics, WindowResult, OptimizationObjective
from .runner import run_backtest

logger = get_logger(__name__)
class EnhancedBacktester:
    def __init__(self, config: BacktestConfig, _progress_callback):
        self.config = config
        self._progress_callback: Optional[Callable[[int, int], None]] = _progress_callback
        self.executor = ProcessPoolExecutor(max_workers=config.max_workers)

        self.data_source = None
        self._lock = None
        self._active_executor = None
        self.strategy = None
        self.strategy_class = None
        self.strategy_kwargs = None
        self.strategy_instance = None

    def run(self) -> Union[BacktestResult, NormalBacktestResult, WalkForwardResult]:
        """运行回测"""
        try:
            logger.info(f"Running backtest for config: {self.config}")
            if self.config.mode in (BacktestMode.SINGLE, BacktestMode.NORMAL):
                return self._run_normal_backtest()
            elif self.config.mode == BacktestMode.WALK_FORWARD:
                return self._run_walk_forward()
            else:
                raise ValueError(f"Unsupported backtest mode: {self.config.mode}")

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def _run_normal_backtest(self) -> NormalBacktestResult|BacktestResult:
        start_time = time.time()
        completed = 0
        params_list = []
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                cfg = self._build_run_config(symbol, timeframe, self.config.strategy_params, self.config.start_time, self.config.end_time)
                params_list.append(cfg)
                if self.config.mode == BacktestMode.SINGLE:
                    break

        futures = {}
        for params in params_list:
            future = self.executor.submit(run_backtest, params)
            futures[future] = params

        results = []
        for future in concurrent.futures.as_completed(futures):
            params = futures[future]
            result = future.result()
            if result is not None:
                result.config = params
                results.append(result)
            completed += 1
            self._progress_callback(completed, len(params_list), result.times)

        execution_time = time.time() - start_time
        if self.config.mode == BacktestMode.SINGLE:
            results[0].execution_time = execution_time
            return results[0]
        return NormalBacktestResult.from_backtest_results(results, execution_time)

    # ===================== Walk-Forward with per-window optimization =====================
    def _run_walk_forward(self) -> WalkForwardResult:
        """Run walk-forward optimization: per-window parameter search on train, evaluate best on test."""
        start_time = time.time()
        windows = self._generate_windows()

        symbols = self.config.symbols or []
        timeframes = self.config.timeframes or []

        # Build param combinations
        param_space: Dict[str, List[Any]] = self.config.param_space or {}
        param_combos: List[Dict[str, Any]] = self._expand_param_space(param_space) if param_space else [self.config.strategy_params or {}]

        # Progress estimation (include CV folds)
        cv_factor = max(1, int(getattr(self.config, 'cv_splits', 0) or 0))
        total_train_jobs = len(windows) * cv_factor * max(1, len(param_combos)) * max(1, len(symbols) * len(timeframes))
        total_test_jobs = len(windows) * max(1, len(symbols) * len(timeframes))
        total_jobs = max(1, total_train_jobs + total_test_jobs)
        done_jobs = 0
        objective = self.config.optimization_objective

        window_results: List[WindowResult] = []
        # For aggregated metrics across windows (test results)
        agg_metrics = PerformanceMetrics()
        agg_count = 0.0

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Train: evaluate ALL param sets across all symbol×timeframe and CV folds in parallel, aggregate objective
            best_params: Dict[str, Any] = None
            best_tuple = (float('-inf'), float('inf'), float('-inf'))  # (avg_score, avg_abs_mdd, avg_trades_negated)

            # Submit all train jobs for this window
            param_acc: Dict[int, Dict[str, float]] = {}
            # build cv folds
            cv_folds = self._generate_cv_folds(train_start, train_end, self.config.cv_splits)
            all_cfgs = []
            for p_idx, params in enumerate(param_combos):
                param_acc[p_idx] = {"score_sum": 0.0, "mdd_sum": 0.0, "trades_sum": 0.0, "count": 0.0}
                for symbol in symbols:
                    for timeframe in timeframes:
                        for (cv_start, cv_end) in cv_folds:
                            cfg = self._build_run_config(symbol, timeframe, params, cv_start, cv_end)
                            cfg["pre_start"] = DateTimeUtils.to_timestamp(self.config.start_time)
                            cfg["pre_end"] = DateTimeUtils.to_timestamp(self.config.end_time)
                            all_cfgs.append((cfg, p_idx))
            all_cfgs.sort(key=lambda x: (x[0]["symbol"], x[0]["timeframe"]))
            futures = {}
            for cfg, p_idx in all_cfgs:
                futures[self.executor.submit(run_backtest, cfg)] = p_idx
            # collect
            for future in concurrent.futures.as_completed(futures):
                p_idx = futures[future]
                br: BacktestResult = future.result()
                if br is not None:
                    score = self._score_objective(objective, br.metrics)
                    acc = param_acc[p_idx]
                    acc["score_sum"] += float(score)
                    acc["mdd_sum"] += abs(float(getattr(br.metrics, 'max_drawdown', 0.0) or 0.0))
                    acc["trades_sum"] += float(getattr(br.metrics, 'total_trades', 0) or 0)
                    acc["count"] += 1.0
                done_jobs += 1
                self._progress_callback(done_jobs, total_jobs, br.times)

            # choose best params using tuple ordering with tie-breakers
            for p_idx, params in enumerate(param_combos):
                acc = param_acc.get(p_idx) or {"count": 0.0}
                c = acc.get("count", 0.0) or 0.0
                if c <= 0:
                    continue
                avg_score = acc["score_sum"] / c
                avg_abs_mdd = acc["mdd_sum"] / c
                avg_trades = acc["trades_sum"] / c
                # Higher objective better, lower max drawdown better, higher trades better
                candidate = (avg_score, -avg_abs_mdd, avg_trades)
                if candidate > best_tuple:
                    best_tuple = candidate
                    best_params = params

            if best_params is None:
                best_params = self.config.strategy_params or {}

            # Test: evaluate best params across all s×tf and record WindowResult
            futures = {}
            for symbol in symbols:
                for timeframe in timeframes:
                    cfg = self._build_run_config(symbol, timeframe, best_params, test_start, test_end)
                    cfg["pre_start"] = DateTimeUtils.to_timestamp(self.config.start_time)
                    cfg["pre_end"] = DateTimeUtils.to_timestamp(self.config.end_time)
                    futures[self.executor.submit(run_backtest, cfg)] = (symbol, timeframe)

            for future in concurrent.futures.as_completed(futures):
                symbol, timeframe = futures[future]
                test_br: BacktestResult = future.result()
                if test_br is not None:
                    # accumulate aggregated metrics
                    agg_count += 1.0
                    for f in fields(PerformanceMetrics):
                        val = getattr(test_br.metrics, f.name, 0.0) or 0.0
                        cur = getattr(agg_metrics, f.name, 0.0) or 0.0
                        setattr(agg_metrics, f.name, cur + float(val))

                    window_results.append(WindowResult(
                        window_idx=w_idx + 1,
                        symbol=symbol,
                        timeframe=timeframe.value,
                        train_period=(windows[w_idx][0], windows[w_idx][1]),
                        test_period=(windows[w_idx][2], windows[w_idx][3]),
                        train_result=None,
                        test_result=test_br,
                        best_params=best_params,
                    ))
                done_jobs += 1
                self._progress_callback(done_jobs, total_jobs, test_br.times)

        # finalize aggregated metrics (mean)
        if agg_count > 0:
            count_fields = {
                "total_trades", "win_trades", "loss_trades",
                "positive_months", "negative_months",
            }
            for f in fields(PerformanceMetrics):
                cur = getattr(agg_metrics, f.name, 0.0) or 0.0
                if f.name in count_fields:
                    # keep as integer sum
                    try:
                        setattr(agg_metrics, f.name, int(round(cur)))
                    except Exception:
                        setattr(agg_metrics, f.name, int(cur))
                else:
                    setattr(agg_metrics, f.name, float(cur) / agg_count)

        return WalkForwardResult(
            config=self.config,
            window_results=window_results,
            aggregated_metrics=agg_metrics,
            equity_curve=[],
            equity_times=[],
            drawdown_curve=[],
            execution_time=time.time() - start_time,
        )

    def _generate_windows(self) -> List[Tuple[int, int, int, int]]:
        """Generate (train_start, train_end, test_start, test_end) windows using rolling/expanding mode."""
        start_dt = DateTimeUtils.to_timestamp(self.config.start_time)
        end_dt = DateTimeUtils.to_timestamp(self.config.end_time)
        train_days = int(self.config.train_days or 0)
        test_days = int(self.config.test_days or 0)
        embargo_days = int(self.config.embargo_days or 0)
        mode = getattr(self.config, 'wf_window_mode', 'rolling') or 'rolling'

        windows: List[Tuple[int, int, int, int]] = []
        test_delta = int(timedelta(days=test_days).total_seconds() * 1000)
        train_delta = int(timedelta(days=train_days).total_seconds() * 1000)
        embargo_delta = int(timedelta(days=embargo_days).total_seconds() * 1000)

        if test_days <= 0:
            return [(start_dt, start_dt, start_dt, end_dt)]
        # If no train_days, run single test window across entire period
        if train_days <= 0:
            # slide tests along
            cur = start_dt
            while cur + test_delta <= end_dt:
                test_start = cur
                test_end = test_start + test_delta
                windows.append((test_start, test_start, test_start, test_end))
                cur = test_start + test_delta
            return windows

        if mode == 'expanding':
            # expanding mode: train from start_dt to current window end, test advances by test_days
            cur_train_end = start_dt + train_delta
            while True:
                train_start = start_dt
                train_end = cur_train_end
                test_start = train_end + embargo_delta
                test_end = test_start + test_delta
                if test_end > end_dt:
                    break
                windows.append((train_start, train_end, test_start, test_end))
                # expand training end by test_days
                cur_train_end = cur_train_end + test_delta
        else:
            # rolling mode
            cur = start_dt
            while True:
                train_start = cur
                train_end = train_start + train_delta
                test_start = train_end + embargo_delta
                test_end = test_start + test_delta
                if test_end > end_dt:
                    break
                windows.append((train_start, train_end, test_start, test_end))
                # advance by test_days (rolling)
                cur = test_start
        return windows

    def _generate_cv_folds(self, start_dt: int, end_dt: int, cv_splits: int) -> List[Tuple[int, int]]:
        """Split [start_dt, end_dt] into cv_splits contiguous folds; if cv_splits<=1, return the full span."""
        if cv_splits is None or cv_splits <= 1:
            return [(start_dt, end_dt)]
        fold_period = (end_dt - start_dt) // cv_splits
        folds = []
        cur_start = start_dt
        for i in range(cv_splits - 1):
            cur_end = cur_start + fold_period
            folds.append((cur_start, cur_end))
            cur_start = cur_end
        # 最后一个 fold 吃掉剩余部分
        folds.append((cur_start, end_dt))
        return folds

    def _build_run_config(self, symbol: str, timeframe, params: Dict[str, Any], start_time, end_time) -> Dict[str, Any]:
        """Build RunConfig dict for runner with overridden params and time bounds."""
        return {
            "id": self.config.id,
            "strategy_class": self.config.strategy_class,
            "strategy_params": params,
            "risk_policies": self.config.risk_policies,
            "datasource_class": self.config.data_source,
            "datasource_config": self.config.data_source_config,
            "executor_class": self.config.executor_class,
            "executor_config": self.config.executor_config,
            "start_time": start_time,
            "end_time": end_time,
            "market": self.config.market,
            "quote_currency": self.config.quote_currency,
            "ins_type": self.config.ins_type.value,
            "symbol": symbol,
            "timeframe": timeframe.value,
            "initial_balance": self.config.initial_balance,
            "mount_dirs": self.config.mount_dirs,
            "use_cache": self.config.use_cache,
            "log_file": self.config.log_file,
        }

    def _expand_param_space(self, space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Cartesian product expansion of param_space into list of param dicts."""
        if not space:
            return [{}]
        keys = list(space.keys())
        arrays = [list(space[k]) for k in keys]
        combos: List[Dict[str, Any]] = []

        def backtrack(i: int, cur: Dict[str, Any]):
            if i == len(keys):
                combos.append(cur.copy())
                return
            k = keys[i]
            for v in arrays[i]:
                cur[k] = v
                backtrack(i + 1, cur)
                cur.pop(k, None)

        backtrack(0, {})
        return combos

    def _score_objective(self, obj: OptimizationObjective, m: PerformanceMetrics) -> float:
        """Map objective to score from metrics; higher is better."""
        try:
            if obj == OptimizationObjective.SHARPE_RATIO:
                return float(m.sharpe_ratio or 0.0)
            if obj == OptimizationObjective.CALMAR_RATIO:
                return float(m.calmar_ratio or 0.0)
            if obj == OptimizationObjective.SORTINO_RATIO:
                return float(m.sortino_ratio or 0.0)
            if obj == OptimizationObjective.PROFIT_FACTOR:
                return float(m.profit_factor or 0.0)
            if obj == OptimizationObjective.WIN_RATE:
                return float(m.win_rate or 0.0)
            # CUSTOM: default to Sharpe until custom function wiring exists
            return float(m.sharpe_ratio or 0.0)
        except Exception:
            return 0.0
