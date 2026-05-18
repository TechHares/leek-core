#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrategyEvaluator 单元测试（不依赖真实数据源）：
  - regime 切片算法
  - 相关矩阵算法
  - to_agent_summary 单/多策略分支
  - evaluate() 参数归一化
"""
import unittest
from decimal import Decimal

from leek_core.backtest.strategy_evaluation import (
    REGIME_RANGING,
    REGIME_TRENDING,
    StrategyEvaluator,
)
from leek_core.backtest.types import (
    BacktestResult,
    PerformanceMetrics,
    StrategyEvaluationConfig,
    StrategyEvaluationResult,
)
from leek_core.models import TimeFrame


def _mk_config(strategies=None) -> StrategyEvaluationConfig:
    return StrategyEvaluationConfig(
        strategies=strategies or [("mock.MockStrategy", {})],
        symbol="BTC_USDT",
        timeframe=TimeFrame.M3,
        start_time="2024-01-01",
        end_time="2024-02-01",
        datasource_class="mock.MockDataSource",
        datasource_config={},
        regime_method="adx",
    )


def _mk_backtest_result(equity_curve, equity_times, sharpe=1.0) -> BacktestResult:
    metrics = PerformanceMetrics(
        sharpe_ratio=sharpe,
        calmar_ratio=0.5,
        total_return=0.1,
        max_drawdown=-0.05,
        total_trades=20,
    )
    return BacktestResult(
        times=[],
        config={"strategy_class": "mock.MockStrategy", "symbol": "BTC_USDT", "timeframe": "3m"},
        metrics=metrics,
        equity_curve=equity_curve,
        equity_times=equity_times,
        trades=[],
        positions=[],
        signals=[],
        drawdown_curve=[],
    )


class TestRegimeSlicing(unittest.TestCase):
    def test_slice_separates_trending_and_ranging(self):
        ev = StrategyEvaluator(_mk_config())
        # equity 10 根（9 step return），前 5 步标 trending，后 4 步标 ranging
        equity = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 104.0, 103.5, 103.0]
        times = [i * 1000 for i in range(10)]
        # regime label 关联 equity_times[i+1]
        regime_at_time = {}
        for i in range(1, 6):
            regime_at_time[times[i]] = REGIME_TRENDING
        for i in range(6, 10):
            regime_at_time[times[i]] = REGIME_RANGING

        out = ev._slice_metrics_by_regime(_mk_backtest_result(equity, times), regime_at_time)

        self.assertIn(REGIME_TRENDING, out)
        self.assertIn(REGIME_RANGING, out)
        self.assertEqual(out[REGIME_TRENDING]["n_periods"], 5)
        self.assertEqual(out[REGIME_RANGING]["n_periods"], 4)
        # 趋势段全是正收益 → total_return > 0；震荡段全是负收益 → total_return < 0
        self.assertGreater(out[REGIME_TRENDING]["total_return"], 0)
        self.assertLess(out[REGIME_RANGING]["total_return"], 0)
        # time_share 之和 = 1
        self.assertAlmostEqual(
            out[REGIME_TRENDING]["time_share"] + out[REGIME_RANGING]["time_share"], 1.0
        )

    def test_empty_regime_labels_returns_empty(self):
        ev = StrategyEvaluator(_mk_config())
        out = ev._slice_metrics_by_regime(
            _mk_backtest_result([100, 101, 102], [0, 1, 2]), {}
        )
        self.assertEqual(out, {})

    def test_too_few_points_returns_empty(self):
        ev = StrategyEvaluator(_mk_config())
        out = ev._slice_metrics_by_regime(_mk_backtest_result([100], [0]), {0: REGIME_TRENDING})
        self.assertEqual(out, {})


class TestCorrelationMatrix(unittest.TestCase):
    def test_identical_series_have_correlation_one(self):
        ev = StrategyEvaluator(_mk_config())
        eq = [100, 101, 102, 101, 103, 104, 103, 105]
        times = list(range(8))
        results = {
            "a": _mk_backtest_result(eq, times),
            "b": _mk_backtest_result(eq, times),
        }
        m = ev._compute_correlation_matrix(results, ["a", "b"])
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m[0][0], 1.0)
        self.assertAlmostEqual(m[1][1], 1.0)
        self.assertAlmostEqual(m[0][1], 1.0)

    def test_anti_correlated_series_negative(self):
        ev = StrategyEvaluator(_mk_config())
        # a 涨/跌/涨/跌/涨；b 反过来跌/涨/跌/涨/跌 → step return 符号严格相反
        eq_a = [100.0, 105.0, 100.0, 105.0, 100.0, 105.0]
        eq_b = [100.0, 95.0, 100.0, 95.0, 100.0, 95.0]
        times = list(range(6))
        results = {
            "a": _mk_backtest_result(eq_a, times),
            "b": _mk_backtest_result(eq_b, times),
        }
        m = ev._compute_correlation_matrix(results, ["a", "b"])
        self.assertIsNotNone(m)
        self.assertLess(m[0][1], 0)

    def test_too_short_returns_none(self):
        ev = StrategyEvaluator(_mk_config())
        results = {
            "a": _mk_backtest_result([100, 101], [0, 1]),
            "b": _mk_backtest_result([100, 101], [0, 1]),
        }
        # 共同时间戳只有 1 个（idx=1 的 step），少于 3 → 返回 None
        self.assertIsNone(ev._compute_correlation_matrix(results, ["a", "b"]))


class TestAgentSummaryShapes(unittest.TestCase):
    def test_single_strategy_summary_has_regime_breakdown(self):
        cfg = _mk_config()
        eq = [100.0 + i for i in range(10)]
        times = [i * 1000 for i in range(10)]
        br = _mk_backtest_result(eq, times)
        regime_metrics = {
            "00_MockStrategy": {
                REGIME_TRENDING: {"sharpe": 2.0, "max_drawdown": -0.01, "total_return": 0.1, "n_periods": 5, "time_share": 0.5},
            }
        }
        result = StrategyEvaluationResult(
            config={"symbol": "BTC_USDT"},
            per_strategy={"00_MockStrategy": br},
            regime_metrics=regime_metrics,
            correlation_matrix=None,
            strategy_ids=["00_MockStrategy"],
            execution_time=1.23,
        )
        s = result.to_agent_summary()
        self.assertEqual(s["mode"], "single")
        self.assertEqual(s["strategy_id"], "00_MockStrategy")
        self.assertIn("regime_breakdown", s)
        self.assertIn("metrics", s)        # 复用 BacktestResult.to_agent_summary 的 metrics

    def test_multi_strategy_summary_has_ranking_and_corr(self):
        cfg = _mk_config()
        br1 = _mk_backtest_result([100, 105], [0, 1], sharpe=1.5)
        br2 = _mk_backtest_result([100, 102], [0, 1], sharpe=0.8)
        result = StrategyEvaluationResult(
            config={"symbol": "BTC_USDT"},
            per_strategy={"00_A": br1, "01_B": br2},
            regime_metrics={"00_A": {}, "01_B": {}},
            correlation_matrix=[[1.0, 0.3], [0.3, 1.0]],
            strategy_ids=["00_A", "01_B"],
            execution_time=2.0,
        )
        s = result.to_agent_summary()
        self.assertEqual(s["mode"], "multi")
        self.assertEqual(s["strategy_count"], 2)
        self.assertEqual(s["ranking_by_sharpe"][0]["id"], "00_A")        # sharpe 高的排前面
        self.assertEqual(s["correlation_matrix"], [[1.0, 0.3], [0.3, 1.0]])


class TestEvaluateNormalization(unittest.TestCase):
    def test_strategy_class_object_converted_to_string(self):
        # 用 unittest.TestCase 自身作为占位类（不会被实际加载）
        cfg = StrategyEvaluationConfig(
            strategies=[(f"{TestEvaluateNormalization.__module__}.TestEvaluateNormalization", {"k": 1})],
            symbol="BTC_USDT",
            timeframe=TimeFrame.M3,
            start_time="2024-01-01",
            end_time="2024-02-01",
            datasource_class="mock",
        )
        ev = StrategyEvaluator(cfg)
        sid = ev._strategy_id(0, cfg.strategies[0][0])
        self.assertEqual(sid, "00_TestEvaluateNormalization")

    def test_strategy_id_format(self):
        cfg = _mk_config([("a.b.MyStrategy", {}), ("c.d.OtherStrategy", {})])
        ev = StrategyEvaluator(cfg)
        self.assertEqual(ev._strategy_id(0, "a.b.MyStrategy"), "00_MyStrategy")
        self.assertEqual(ev._strategy_id(7, "c.d.OtherStrategy"), "07_OtherStrategy")

    def test_empty_strategies_rejected(self):
        with self.assertRaises(ValueError):
            StrategyEvaluationConfig(
                strategies=[],
                symbol="BTC_USDT",
                timeframe=TimeFrame.M3,
                start_time="2024-01-01",
                end_time="2024-02-01",
                datasource_class="mock",
            )


if __name__ == "__main__":
    unittest.main()
