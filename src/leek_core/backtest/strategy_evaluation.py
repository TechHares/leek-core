#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略评估模块：在同一份数据上跑 N 个策略（N>=1），输出对比矩阵 + 按市场状态切片的指标。

设计原则：
  - 单策略 / 多策略统一接口（strategies 列表长度决定模式）。
  - 不预设 verdict / 推荐组合等主观结论，输出纯事实指标。
  - 复用 BacktestRunner，不修改其代码。
"""
from concurrent.futures import as_completed
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
from joblib.externals.loky import ProcessPoolExecutor

from leek_core.base import create_component, load_class_from_str
from leek_core.data import DataSource
from leek_core.indicators import ADX
from leek_core.models import KLine, TimeFrame, TradeInsType
from leek_core.utils import get_logger

from .data_cache import DataCache
from .runner import run_backtest
from .types import (
    BacktestResult,
    StrategyEvaluationConfig,
    StrategyEvaluationResult,
)

logger = get_logger(__name__)


REGIME_TRENDING = "trending"
REGIME_RANGING = "ranging"
REGIME_WARMUP = "warmup"   # ADX 未稳定输出阶段，不参与切片统计


class StrategyEvaluator:
    """策略评估器：单策略详细诊断 + 多策略横向对比。

    典型用法：

        # 单策略
        result = StrategyEvaluator.evaluate(
            strategy=(MyStrategy, {"period": 20}),
            symbol="BTC_USDT", timeframe="3m",
            start_time="2024-01-01", end_time="2025-01-01",
            datasource_class="...", datasource_config={...},
        )

        # 多策略（同函数，strategies 传 list）
        result = StrategyEvaluator.evaluate(
            strategies=[(MAStrategy, {}), (RSIStrategy, {})],
            symbol="BTC_USDT", timeframe="3m",
            start_time="2024-01-01", end_time="2025-01-01",
            datasource_class="...", datasource_config={...},
            max_workers=4,
        )

        summary = result.to_agent_summary()   # 给 Agent
        full = result.to_dict()               # 完整结构化数据
    """

    def __init__(self, config: StrategyEvaluationConfig):
        self.config = config

    # ---------------------- 便捷入口 ----------------------

    @classmethod
    def evaluate(
        cls,
        *,
        strategy: Optional[Tuple[Union[str, type], Optional[Dict[str, Any]]]] = None,
        strategies: Optional[List[Tuple[Union[str, type], Optional[Dict[str, Any]]]]] = None,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_time: Union[str, datetime, int],
        end_time: Union[str, datetime, int],
        datasource_class: str,
        datasource_config: Dict[str, Any],
        executor_class: str = "leek_core.executor.BacktestExecutor",
        executor_config: Optional[Dict[str, Any]] = None,
        market: str = "okx",
        quote_currency: str = "USDT",
        ins_type: Union[TradeInsType, int, str] = TradeInsType.SWAP,
        initial_balance: Union[Decimal, float, int] = Decimal("10000"),
        risk_policies: Optional[List[Dict[str, Any]]] = None,
        mount_dirs: Optional[List[str]] = None,
        regime_method: str = "adx",
        adx_threshold: float = 25.0,
        adx_smoothing: int = 6,
        adx_di_length: int = 14,
        max_workers: int = 1,
        use_cache: bool = True,
        simulate_kline: bool = False,
    ) -> StrategyEvaluationResult:
        """便捷入口。`strategy` 与 `strategies` 二选一。

        策略入参支持 `(类对象, params)` 或 `("module.ClassName", params)`。
        """
        if strategy is None and strategies is None:
            raise ValueError("strategy 或 strategies 必须传一个")
        if strategy is not None and strategies is not None:
            raise ValueError("strategy 和 strategies 不能同时传")
        if strategy is not None:
            strategies = [strategy]

        normalized: List[Tuple[str, Dict[str, Any]]] = []
        for entry in strategies:
            s_cls, params = entry
            if isinstance(s_cls, type):
                cls_str = f"{s_cls.__module__}.{s_cls.__name__}"
            else:
                cls_str = s_cls
            normalized.append((cls_str, dict(params or {})))

        cfg = StrategyEvaluationConfig(
            strategies=normalized,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            market=market,
            quote_currency=quote_currency,
            ins_type=ins_type,
            datasource_class=datasource_class,
            datasource_config=datasource_config or {},
            executor_class=executor_class,
            executor_config=executor_config or {},
            initial_balance=Decimal(str(initial_balance)),
            risk_policies=risk_policies or [],
            mount_dirs=mount_dirs or [],
            regime_method=regime_method,
            adx_threshold=adx_threshold,
            adx_smoothing=adx_smoothing,
            adx_di_length=adx_di_length,
            max_workers=max_workers,
            use_cache=use_cache,
            simulate_kline=simulate_kline,
        )
        return cls(cfg).run()

    # ---------------------- 主流程 ----------------------

    def run(self) -> StrategyEvaluationResult:
        t0 = time.time()

        bt_results = self._run_all_backtests()

        regime_at_time: Dict[int, str] = {}
        if self.config.regime_method == "adx" and bt_results:
            regime_at_time = self._compute_regime_labels()

        regime_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sid, br in bt_results.items():
            regime_metrics[sid] = self._slice_metrics_by_regime(br, regime_at_time)

        corr_matrix: Optional[List[List[float]]] = None
        strategy_ids = list(bt_results.keys())
        if len(bt_results) > 1:
            corr_matrix = self._compute_correlation_matrix(bt_results, strategy_ids)

        return StrategyEvaluationResult(
            config=self._config_summary(),
            per_strategy=bt_results,
            regime_metrics=regime_metrics,
            correlation_matrix=corr_matrix,
            strategy_ids=strategy_ids,
            execution_time=time.time() - t0,
        )

    # ---------------------- 实现细节 ----------------------

    def _config_summary(self) -> Dict[str, Any]:
        tf = self.config.timeframe
        return {
            "symbol": self.config.symbol,
            "timeframe": tf.value if hasattr(tf, "value") else str(tf),
            "start_time": self.config.start_time,
            "end_time": self.config.end_time,
            "strategy_count": len(self.config.strategies),
            "regime_method": self.config.regime_method,
            "adx_threshold": self.config.adx_threshold,
        }

    def _build_run_config(self, strategy_class: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ins_type = self.config.ins_type
        return {
            "id": self.config.id,
            "strategy_class": strategy_class,
            "strategy_params": params,
            "risk_policies": self.config.risk_policies or [],
            "datasource_class": self.config.datasource_class,
            "datasource_config": self.config.datasource_config or {},
            "executor_class": self.config.executor_class,
            "executor_config": self.config.executor_config or {},
            "start_time": self.config.start_time,
            "end_time": self.config.end_time,
            "market": self.config.market,
            "quote_currency": self.config.quote_currency,
            "ins_type": ins_type.value if hasattr(ins_type, "value") else ins_type,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe.value,
            "initial_balance": self.config.initial_balance,
            "mount_dirs": self.config.mount_dirs or [],
            "use_cache": self.config.use_cache,
            "simulate_kline": self.config.simulate_kline,
        }

    @staticmethod
    def _strategy_id(idx: int, strategy_class: str) -> str:
        short = strategy_class.rsplit(".", 1)[-1]
        return f"{idx:02d}_{short}"

    def _run_all_backtests(self) -> Dict[str, BacktestResult]:
        """并行跑各策略；为避免 run_backtest 内 setup_logging 污染主进程日志，始终走子进程。"""
        results: Dict[str, BacktestResult] = {}
        run_configs: List[Tuple[str, Dict[str, Any]]] = [
            (self._strategy_id(i, s_cls), self._build_run_config(s_cls, params))
            for i, (s_cls, params) in enumerate(self.config.strategies)
        ]
        cfg_map: Dict[str, Dict[str, Any]] = dict(run_configs)

        workers = max(1, min(self.config.max_workers, len(run_configs)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_backtest, cfg): sid for sid, cfg in run_configs}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    br: Optional[BacktestResult] = future.result()
                    if br is not None:
                        br.config = cfg_map[sid]
                        results[sid] = br
                except Exception as e:
                    logger.error(f"策略 {sid} 回测失败: {e}", exc_info=True)
        return results

    def _compute_regime_labels(self) -> Dict[int, str]:
        """从 datasource 读 K 线，逐根算 ADX，返回 {kline.end_time_ms: regime}。"""
        try:
            datasource: DataSource = create_component(
                load_class_from_str(self.config.datasource_class),
                **(self.config.datasource_config or {}),
            )
        except Exception as e:
            logger.warning(f"创建 datasource 失败，跳过 regime 切片: {e}")
            return {}

        if self.config.use_cache:
            datasource = DataCache(datasource)

        try:
            datasource.on_start()
        except Exception as e:
            logger.warning(f"datasource.on_start 失败，跳过 regime 切片: {e}")
            return {}

        adx = ADX(
            adx_smoothing=self.config.adx_smoothing,
            di_length=self.config.adx_di_length,
        )
        threshold = self.config.adx_threshold
        regime_at_time: Dict[int, str] = {}

        try:
            row_key = KLine.pack_row_key(
                self.config.symbol,
                self.config.quote_currency,
                self.config.ins_type,
                self.config.timeframe,
            )
            for kline in datasource.get_history_data(
                start_time=self.config.start_time,
                end_time=self.config.end_time,
                pre_load_start_time=self.config.start_time,
                pre_load_end_time=self.config.end_time,
                row_key=row_key,
                market=self.config.market,
            ):
                adx_val = adx.update(kline)
                if adx_val is None or not kline.is_finished:
                    continue
                regime_at_time[int(kline.end_time)] = (
                    REGIME_TRENDING if adx_val > threshold else REGIME_RANGING
                )
        except Exception as e:
            logger.warning(f"ADX 标注异常: {e}")
        finally:
            try:
                datasource.on_stop()
            except Exception:
                pass

        return regime_at_time

    def _slice_metrics_by_regime(
        self, result: BacktestResult, regime_at_time: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """按 regime 标签把 equity 步进收益切片重算指标。"""
        if not result.equity_curve or len(result.equity_curve) < 2 or not regime_at_time:
            return {}

        eq = np.asarray(result.equity_curve, dtype=np.float64)
        times = result.equity_times
        # step return r[i] 对应 equity_curve[i+1] 这根 K 线，关联其 end_time 的 regime
        prev = eq[:-1]
        returns = np.where(np.abs(prev) > 1e-12, np.diff(eq) / prev, 0.0)
        labels = np.asarray(
            [regime_at_time.get(int(times[i + 1]), REGIME_WARMUP) for i in range(len(returns))]
        )

        periods_per_year = int(
            (24 * 3600 * 1000 / self.config.timeframe.milliseconds) * 365
        )
        breakdown: Dict[str, Dict[str, float]] = {}
        total = len(returns)
        for regime in (REGIME_TRENDING, REGIME_RANGING):
            mask = labels == regime
            n = int(mask.sum())
            if n < 2:
                continue
            r = returns[mask]
            mean_r = float(np.mean(r))
            std_r = float(np.std(r, ddof=1))
            sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else 0.0
            virtual_eq = np.cumprod(1.0 + r)
            peak = np.maximum.accumulate(virtual_eq)
            dd = (virtual_eq - peak) / np.maximum(peak, 1e-12)
            breakdown[regime] = {
                "sharpe": round(sharpe, 3),
                "max_drawdown": round(float(np.min(dd)), 4),
                "total_return": round(float(virtual_eq[-1] - 1.0), 4),
                "mean_return_per_period": round(mean_r, 6),
                "n_periods": n,
                "time_share": round(n / total, 3) if total > 0 else 0.0,
            }
        return breakdown

    def _compute_correlation_matrix(
        self, results: Dict[str, BacktestResult], strategy_ids: List[str]
    ) -> Optional[List[List[float]]]:
        """各策略 step return 序列的相关矩阵；按时间戳取交集对齐。"""
        series_by_id: Dict[str, Dict[int, float]] = {}
        for sid in strategy_ids:
            br = results[sid]
            d: Dict[int, float] = {}
            if br.equity_curve and len(br.equity_curve) >= 2:
                eq = br.equity_curve
                ts = br.equity_times
                for i in range(1, len(eq)):
                    prev = eq[i - 1]
                    if prev == 0:
                        continue
                    d[int(ts[i])] = eq[i] / prev - 1.0
            series_by_id[sid] = d

        common: Optional[set] = None
        for sid in strategy_ids:
            keys = set(series_by_id[sid].keys())
            common = keys if common is None else common & keys
        if not common or len(common) < 3:
            return None

        sorted_times = sorted(common)
        mat = np.array(
            [[series_by_id[sid][t] for t in sorted_times] for sid in strategy_ids],
            dtype=np.float64,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(mat)
        if corr.ndim == 0:
            return [[1.0]]
        corr = np.nan_to_num(corr, nan=0.0)
        return [[round(float(c), 3) for c in row] for row in corr]
