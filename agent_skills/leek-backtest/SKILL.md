---
name: leek-backtest
description: 使用 leek 回测系统。配置 BacktestConfig/RunConfig 运行策略回测，支持单次回测、参数搜索、Walk-Forward验证等模式。当用户要进行策略回测、参数优化、历史数据测试时使用。Use when backtesting strategies, optimizing parameters, or testing with historical data.
---

# Leek 回测系统

## 快速开始

```python
from leek_core.backtest import BacktestRunner, BacktestConfig, BacktestMode
from leek_core.models import TimeFrame, TradeInsType
from decimal import Decimal

# 配置
config = {
    "strategy_class": "my_module.MyStrategy",
    "strategy_params": {"period": 14},
    "datasource_class": "leek_core.data.ClickHouseKlineDataSource",
    "datasource_config": {"connection_string": "..."},
    "executor_class": "leek_core.executor.BacktestExecutor",
    "executor_config": {"slippage": 0.001},
    "symbol": "BTCUSDT",
    "timeframe": TimeFrame.H1,
    "start_time": "2024-01-01",
    "end_time": "2024-06-01",
    "initial_balance": Decimal("10000"),
    "market": "okx",
    "quote_currency": "USDT",
    "ins_type": TradeInsType.SWAP,
}

# 运行
runner = BacktestRunner(config)
result = runner.run()

# 查看结果
print(f"总收益: {result.metrics.total_return:.2%}")
print(f"夏普比率: {result.metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {result.metrics.max_drawdown:.2%}")
```

## 回测模式

| 模式 | 说明 | 用途 |
|-----|------|------|
| `SINGLE` | 单次回测 | 固定参数快速测试 |
| `NORMAL` | 普通回测 | 多标的×多周期 |
| `PARAM_SEARCH` | 参数搜索 | 网格搜索最优参数 |
| `WALK_FORWARD` | 走向前验证 | 样本外验证防过拟合 |
| `MONTE_CARLO` | 蒙特卡洛模拟 | 风险评估 |

## 配置详解

### RunConfig（单次回测）

```python
from leek_core.backtest import RunConfig
from leek_core.models import TimeFrame, TradeInsType

config = RunConfig(
    # 策略配置
    strategy_class="my_module.MyStrategy",
    strategy_params={"period": 14, "threshold": 0.5},
    risk_policies=[
        {"class": "leek_core.sub_strategy.StopLoss", "params": {"rate": 0.05}}
    ],
    
    # 数据源配置
    datasource_class="leek_core.data.ClickHouseKlineDataSource",
    datasource_config={"connection_string": "clickhouse://..."},
    
    # 执行器配置
    executor_class="leek_core.executor.BacktestExecutor",
    executor_config={
        "slippage": 0.001,    # 滑点
        "commission": 0.001,  # 手续费率
    },
    
    # 数据配置
    symbol="BTCUSDT",
    timeframe=TimeFrame.H1,
    start_time="2024-01-01",  # 支持字符串/datetime/时间戳
    end_time="2024-06-01",
    market="okx",
    quote_currency="USDT",
    ins_type=TradeInsType.SWAP,
    
    # 资金配置
    initial_balance=Decimal("10000"),
    
    # 性能选项
    use_cache=True,
    skip_statistical_tests=False,
    simulate_kline=False,
)
```

### BacktestConfig（高级回测）

```python
from leek_core.backtest import BacktestConfig, BacktestMode, OptimizationObjective

config = BacktestConfig(
    id=1,
    name="我的回测",
    mode=BacktestMode.WALK_FORWARD,
    
    # 策略配置
    strategy_class="my_module.MyStrategy",
    strategy_params={"period": 14},
    
    # 多标的多周期
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=[TimeFrame.H1, TimeFrame.H4],
    
    # 时间范围
    start_time="2023-01-01",
    end_time="2024-01-01",
    
    # 参数搜索空间
    param_space={
        "period": [7, 14, 21, 28],
        "threshold": [0.3, 0.5, 0.7],
    },
    
    # 优化目标
    optimization_objective=OptimizationObjective.SHARPE_RATIO,
    
    # Walk-Forward 配置
    train_days=60,
    test_days=30,
    embargo_days=1,
    wf_window_mode="rolling",  # rolling | expanding
    
    # 并行配置
    max_workers=4,
    
    # 风控策略
    risk_policies=[
        {"class": "leek_core.sub_strategy.StopLoss", "params": {"rate": 0.05}}
    ],
)
```

## 优化目标

```python
from leek_core.backtest import OptimizationObjective

# 内置目标
OptimizationObjective.SHARPE_RATIO    # 夏普比率
OptimizationObjective.CALMAR_RATIO    # 卡玛比率
OptimizationObjective.SORTINO_RATIO   # 索提诺比率
OptimizationObjective.PROFIT_FACTOR   # 盈亏比
OptimizationObjective.WIN_RATE        # 胜率
OptimizationObjective.CUSTOM          # 自定义

# 自定义目标函数
def my_objective(metrics: dict) -> float:
    return metrics["sharpe_ratio"] * metrics["win_rate"]

config = BacktestConfig(
    optimization_objective=OptimizationObjective.CUSTOM,
    custom_objective=my_objective,
    ...
)
```

## 性能指标 (PerformanceMetrics)

### 收益指标

| 指标 | 说明 |
|-----|------|
| `total_return` | 总收益率 |
| `annual_return` | 年化收益率 |
| `volatility` | 波动率 |

### 风险调整收益

| 指标 | 说明 |
|-----|------|
| `sharpe_ratio` | 夏普比率 |
| `calmar_ratio` | 卡玛比率 |
| `sortino_ratio` | 索提诺比率 |
| `omega_ratio` | 欧米伽比率 |
| `information_ratio` | 信息比率 |

### 回撤指标

| 指标 | 说明 |
|-----|------|
| `max_drawdown` | 最大回撤 |
| `max_drawdown_duration` | 最大回撤持续时间 |
| `avg_drawdown` | 平均回撤 |

### 交易指标

| 指标 | 说明 |
|-----|------|
| `total_trades` | 总交易次数 |
| `win_rate` | 胜率 |
| `profit_factor` | 盈亏因子 |
| `avg_win` | 平均盈利 |
| `avg_loss` | 平均亏损 |
| `win_loss_ratio` | 盈亏比 |
| `long_win_rate` | 做多胜率 |
| `short_win_rate` | 做空胜率 |

### 统计检验

| 指标 | 说明 |
|-----|------|
| `t_statistic` | t统计量 |
| `t_pvalue` | t检验p值 |
| `bootstrap_sharpe_ci_lower/upper` | 夏普比率95%置信区间 |
| `win_rate_pvalue` | 胜率二项检验p值 |

## 回测结果

### BacktestResult

```python
result = runner.run()

# 指标
result.metrics.total_return
result.metrics.sharpe_ratio
result.metrics.max_drawdown

# 权益曲线
result.equity_curve      # [10000, 10100, 10050, ...]
result.equity_times      # [时间戳列表]
result.drawdown_curve    # 回撤曲线

# 交易记录
result.trades            # [{"entry_time": ..., "exit_time": ..., "pnl": ...}, ...]

# 配置和元数据
result.config
result.execution_time
```

### WalkForwardResult

```python
result = backtester.run()

# 窗口结果
for window in result.window_results:
    print(f"窗口 {window.window_idx}")
    print(f"  训练期: {window.train_period}")
    print(f"  测试期: {window.test_period}")
    print(f"  最优参数: {window.best_params}")
    print(f"  测试夏普: {window.test_result.metrics.sharpe_ratio}")

# 聚合指标
result.aggregated_metrics
result.equity_curve
```

## 使用 EnhancedBacktester

```python
from leek_core.backtest import EnhancedBacktester, BacktestConfig, BacktestMode

def progress_callback(completed, total, times, window_data):
    print(f"进度: {completed}/{total}")
    if window_data:
        print(f"  窗口 {window_data['window_idx']}: 夏普={window_data['metrics']['sharpe_ratio']:.2f}")

config = BacktestConfig(
    id=1,
    name="参数优化回测",
    mode=BacktestMode.WALK_FORWARD,
    strategy_class="my_module.MyStrategy",
    symbols=["BTCUSDT"],
    timeframes=[TimeFrame.H1],
    start_time="2023-01-01",
    end_time="2024-01-01",
    param_space={
        "fast_period": [5, 10, 20],
        "slow_period": [20, 50, 100],
    },
    train_days=90,
    test_days=30,
    max_workers=4,
)

backtester = EnhancedBacktester(config, progress_callback)
result = backtester.run()
```

## 数据源配置

### ClickHouse

```python
datasource_config = {
    "connection_string": "clickhouse://user:pass@host:9000/database",
    "table": "klines",
}
```

### Redis + ClickHouse

```python
datasource_config = {
    "redis_url": "redis://localhost:6379",
    "clickhouse_connection": "clickhouse://...",
}
```

## 最佳实践

1. **避免过拟合**：使用 Walk-Forward 验证样本外表现
2. **参数范围**：参数搜索时设置合理的范围，避免过多组合
3. **统计检验**：关注 p值，夏普比率置信区间
4. **多标的测试**：在多个标的上验证策略普适性
5. **滑点手续费**：设置真实的滑点和手续费率
6. **数据质量**：确保回测数据完整无缺失

## 时间格式

支持多种时间格式：

```python
# 字符串
start_time = "2024-01-01"
start_time = "2024-01-01 00:00:00"

# datetime
from datetime import datetime
start_time = datetime(2024, 1, 1)

# 毫秒时间戳
start_time = 1704067200000
```
