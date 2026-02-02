# 30 回测系统

## 概述

回测系统是 Leek Core 的核心功能之一，用于在历史数据上验证交易策略的有效性。系统支持多种回测模式，包括单次回测、参数搜索、走向前验证（Walk-Forward）和蒙特卡洛模拟，提供丰富的绩效指标和统计检验。

## 核心组件

### 组件结构

```text
BacktestRunner
├── SimpleEngine (回测引擎)
│   ├── DataManager
│   ├── StrategyManager
│   ├── ExecutorManager (BacktestExecutor)
│   └── Portfolio
├── DataSource / DataCache (数据源)
└── SerializableEventBus (同步事件总线)
```

### `BacktestRunner` - 回测运行器

```python
class BacktestRunner:
    """回测运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = RunConfig(**config)
        
        # 统计变量
        self.trades_data = []       # 交易记录
        self.positions_data = []    # 仓位记录
        self.signals_data = []      # 信号记录
        self.equity_values = []     # 净值曲线
        self.equity_times = []      # 时间序列
        self.benchmark_prices = []  # 基准价格
    
    def run(self) -> BacktestResult: ...
    def on_stop(self): ...
```

### `RunConfig` - 运行配置

```python
@dataclass
class RunConfig:
    """运行配置"""
    id: int = 0
    
    # 策略配置
    strategy_class: str = None          # 策略类路径
    strategy_params: Dict[str, Any] = None  # 策略参数
    risk_policies: List[Dict] = None    # 风控策略
    
    # 数据源配置
    datasource_class: str = None        # 数据源类路径
    datasource_config: Dict = None      # 数据源参数
    
    # 执行器配置
    executor_class: str = None          # 执行器类路径
    executor_config: Dict = None        # 执行器参数
    
    # 数据配置
    symbol: str = None                  # 交易对
    timeframe: TimeFrame = None         # 时间周期
    start_time: int = None              # 开始时间（毫秒）
    end_time: int = None                # 结束时间（毫秒）
    pre_start: int = None               # 预热开始时间
    pre_end: int = None                 # 预热结束时间
    market: str = "okx"                 # 市场标识
    quote_currency: str = "USDT"        # 计价币种
    ins_type: TradeInsType = TradeInsType.SWAP  # 品种类型
    
    # 资金配置
    initial_balance: Decimal = Decimal("10000")
    
    # 性能选项
    use_cache: bool = False             # 是否使用数据缓存
    skip_statistical_tests: bool = False # 跳过统计检验
    simulate_kline: bool = False        # K线模拟模式
    log_file: bool = False              # 是否输出日志文件
    
    # 模块挂载
    mount_dirs: List[str] = field(default_factory=list)
```

## 回测模式

### 1. 单次回测（SINGLE）

固定参数的单次回测，最常用的模式：

```python
from leek_core.backtest import run_backtest

config = {
    "strategy_class": "my_module.MyStrategy",
    "strategy_params": {"fast_period": 5, "slow_period": 20},
    "datasource_class": "leek_core.data.ClickHouseSource",
    "datasource_config": {"host": "localhost", "database": "market"},
    "executor_class": "leek_core.executor.BacktestExecutor",
    "symbol": "BTC",
    "quote_currency": "USDT",
    "ins_type": 2,  # SWAP
    "timeframe": "1h",
    "start_time": 1704067200000,
    "end_time": 1706745600000,
    "initial_balance": 10000,
}

result = run_backtest(config)
```

### 2. 参数搜索（PARAM_SEARCH）

网格搜索或优化器搜索最优参数：

```python
from leek_core.backtest import BacktestConfig, BacktestMode, OptimizationObjective

config = BacktestConfig(
    id=1,
    name="参数优化",
    mode=BacktestMode.PARAM_SEARCH,
    strategy_class="my_module.MyStrategy",
    param_space={
        "fast_period": [3, 5, 7, 10],
        "slow_period": [15, 20, 25, 30],
    },
    optimization_objective=OptimizationObjective.SHARPE_RATIO,
    symbols=["BTC"],
    timeframes=["1h"],
    start_time="2024-01-01",
    end_time="2024-02-01",
    max_workers=4,  # 并行搜索
)
```

### 3. 走向前验证（WALK_FORWARD）

滚动窗口验证，避免过拟合：

```python
config = BacktestConfig(
    id=2,
    name="走向前验证",
    mode=BacktestMode.WALK_FORWARD,
    strategy_class="my_module.MyStrategy",
    param_space={...},
    train_days=30,       # 训练窗口30天
    test_days=7,         # 测试窗口7天
    embargo_days=1,      # 间隔1天
    wf_window_mode="rolling",  # rolling | expanding
    optuna_enabled=True,       # 使用Optuna优化
    optuna_n_trials=80,
)
```

### 4. 蒙特卡洛模拟（MONTE_CARLO）

随机模拟评估策略稳健性。

## 绩效指标

### `PerformanceMetrics` - 性能指标

```python
@dataclass
class PerformanceMetrics:
    # 基础指标
    total_return: float = 0.0       # 总收益率
    annual_return: float = 0.0      # 年化收益率
    volatility: float = 0.0         # 年化波动率
    
    # 风险调整收益
    sharpe_ratio: float = 0.0       # 夏普比率
    calmar_ratio: float = 0.0       # 卡尔玛比率
    sortino_ratio: float = 0.0      # 索提诺比率
    omega_ratio: float = 0.0        # Omega比率
    
    # 回撤指标
    max_drawdown: float = 0.0       # 最大回撤
    max_drawdown_duration: int = 0  # 最大回撤持续期
    drawdown_periods: int = 0       # 回撤次数
    
    # 交易指标
    total_trades: int = 0           # 总交易次数
    win_trades: int = 0             # 盈利次数
    loss_trades: int = 0            # 亏损次数
    win_rate: float = 0.0           # 胜率
    long_trades: int = 0            # 做多次数
    short_trades: int = 0           # 做空次数
    long_win_rate: float = 0.0      # 做多胜率
    short_win_rate: float = 0.0     # 做空胜率
    profit_factor: float = 0.0      # 盈亏比
    avg_win: float = 0.0            # 平均盈利
    avg_loss: float = 0.0           # 平均亏损
    win_loss_ratio: float = 0.0     # 盈亏比率
    largest_win: float = 0.0        # 最大单笔盈利
    largest_loss: float = 0.0       # 最大单笔亏损
    
    # 单笔收益
    avg_pnl: float = 0.0            # 平均单笔收益
    avg_return_per_trade: float = 0.0  # 平均单笔收益率
    
    # 风险指标
    var_95: float = 0.0             # 95% VaR
    cvar_95: float = 0.0            # 95% CVaR
    skewness: float = 0.0           # 偏度
    kurtosis: float = 0.0           # 峰度
    turnover: float = 0.0           # 换手率
    
    # 统计检验
    t_statistic: float = 0.0        # t统计量
    t_pvalue: float = 1.0           # t检验p值
    bootstrap_sharpe_ci_lower: float = 0.0   # Bootstrap夏普置信区间下界
    bootstrap_sharpe_ci_upper: float = 0.0   # Bootstrap夏普置信区间上界
    win_rate_pvalue: float = 1.0    # 胜率二项检验p值
```

### 指标说明

| 指标 | 公式 | 说明 |
|------|------|------|
| 年化收益率 | $(1 + R)^{N/n} - 1$ | R为总收益率，N为年期数，n为实际期数 |
| 夏普比率 | $(R_a - R_f) / \sigma$ | 年化收益减无风险利率除以年化波动率 |
| 卡尔玛比率 | $R_a / |MDD|$ | 年化收益除以最大回撤绝对值 |
| 索提诺比率 | $R_a / \sigma_d$ | 年化收益除以下行标准差 |
| Omega比率 | $\sum(gains) / |sum(losses)|$ | 超过阈值的收益和除以低于阈值的亏损和 |
| VaR 95% | 5%分位数 | 95%置信度下的最大损失 |
| CVaR 95% | VaR以下的均值 | 尾部风险的期望损失 |

## 回测结果

### `BacktestResult` - 回测结果

```python
@dataclass
class BacktestResult:
    times: List[int]                 # 时间戳 [start, init, data, run, end, metrics]
    config: Dict[str, Any]           # 运行配置
    metrics: PerformanceMetrics      # 绩效指标
    equity_curve: List[float]        # 净值曲线
    equity_times: List[int]          # 净值时间戳
    trades: List[Dict]               # 交易记录
    positions: List[Dict]            # 仓位记录
    signals: List[Dict]              # 信号记录
    drawdown_curve: List[float]      # 回撤曲线
    benchmark_curve: List[float]     # 基准曲线
    metadata: Dict[str, Any]         # 元数据
```

## 使用示例

### 基础回测

```python
from leek_core.backtest import run_backtest

config = {
    # 策略
    "strategy_class": "my_module.DualMAStrategy",
    "strategy_params": {
        "fast_period": 5,
        "slow_period": 20,
    },
    "risk_policies": [
        {
            "class_name": "leek_core.sub_strategy.PositionStopLoss",
            "config": {"stop_loss_pct": 0.05}
        }
    ],
    
    # 数据源
    "datasource_class": "leek_core.data.ClickHouseSource",
    "datasource_config": {
        "host": "localhost",
        "port": 9000,
        "database": "market_data",
    },
    
    # 执行器
    "executor_class": "leek_core.executor.BacktestExecutor",
    
    # 数据
    "symbol": "BTC",
    "quote_currency": "USDT",
    "ins_type": 2,  # SWAP
    "timeframe": "1h",
    "market": "okx",
    "start_time": "2024-01-01",
    "end_time": "2024-03-01",
    
    # 资金
    "initial_balance": 10000,
    
    # 性能优化
    "use_cache": True,
    "skip_statistical_tests": False,
}

result = run_backtest(config)

# 输出结果
print(f"总收益率: {result.metrics.total_return:.2%}")
print(f"年化收益: {result.metrics.annual_return:.2%}")
print(f"夏普比率: {result.metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {result.metrics.max_drawdown:.2%}")
print(f"胜率: {result.metrics.win_rate:.2%}")
print(f"盈亏比: {result.metrics.profit_factor:.2f}")
```

### 带预热的回测

```python
config = {
    ...
    # 预热期：用于初始化指标，不参与交易
    "pre_start": 1701388800000,  # 2023-12-01
    "pre_end": 1704067200000,    # 2024-01-01
    
    # 实际回测期
    "start_time": 1704067200000,  # 2024-01-01
    "end_time": 1706745600000,    # 2024-02-01
}
```

### K线模拟模式

使用1分钟K线模拟更大周期，提高回测精度：

```python
config = {
    ...
    "timeframe": "1h",      # 目标周期
    "simulate_kline": True,  # 启用K线模拟
    # 系统会自动查询1分钟K线，然后合并成1小时K线
}
```

### 结果可视化

```python
import matplotlib.pyplot as plt

result = run_backtest(config)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 净值曲线
ax1 = axes[0]
ax1.plot(result.equity_times, result.equity_curve, label='Strategy')
if result.benchmark_curve:
    ax1.plot(result.equity_times, result.benchmark_curve, label='Benchmark', alpha=0.7)
ax1.set_title('Equity Curve')
ax1.legend()

# 回撤曲线
ax2 = axes[1]
ax2.fill_between(result.equity_times, result.drawdown_curve, 0, alpha=0.3, color='red')
ax2.set_title('Drawdown')

plt.tight_layout()
plt.show()
```

### 多进程并行回测

```python
from multiprocessing import Pool
from leek_core.backtest import run_backtest

# 生成多组配置
configs = []
for symbol in ["BTC", "ETH", "SOL"]:
    for timeframe in ["1h", "4h"]:
        config = {
            "id": len(configs),
            "symbol": symbol,
            "timeframe": timeframe,
            ...
        }
        configs.append(config)

# 并行执行
with Pool(processes=4) as pool:
    results = pool.map(run_backtest, configs)

# 汇总结果
for config, result in zip(configs, results):
    print(f"{config['symbol']}/{config['timeframe']}: "
          f"Return={result.metrics.total_return:.2%}, "
          f"Sharpe={result.metrics.sharpe_ratio:.2f}")
```

## 统计检验

### 可用的统计检验

| 检验 | 说明 | 用途 |
|------|------|------|
| t检验 | 收益率是否显著异于0 | 验证策略是否有alpha |
| 配对t检验 | 策略与基准是否显著不同 | 验证策略是否优于基准 |
| Bootstrap | 夏普/年化收益置信区间 | 评估指标稳定性 |
| 二项检验 | 胜率是否显著高于50% | 验证胜率是否有效 |

### 解读统计结果

```python
result = run_backtest(config)
m = result.metrics

# 检查策略是否有统计显著性
if m.t_pvalue < 0.05:
    print("策略收益统计显著 (p < 0.05)")
    
if m.bootstrap_sharpe_ci_lower > 0:
    print(f"夏普比率95%置信区间: [{m.bootstrap_sharpe_ci_lower:.2f}, {m.bootstrap_sharpe_ci_upper:.2f}]")
    
if m.win_rate_pvalue < 0.05:
    print(f"胜率显著高于50% (p={m.win_rate_pvalue:.4f})")
```

## 数据缓存

### `DataCache` - 数据缓存

启用缓存可以显著提升重复回测的速度：

```python
config = {
    ...
    "use_cache": True,  # 启用数据缓存
}
```

缓存机制：
- 首次查询从数据源加载并缓存
- 后续查询直接从缓存读取
- 缓存按 `row_key` 组织

## 性能优化

### 1. 跳过统计检验

在参数搜索阶段，跳过统计检验以提升速度：

```python
config = {
    ...
    "skip_statistical_tests": True,
}
```

### 2. 使用数据缓存

```python
config = {
    ...
    "use_cache": True,
}
```

### 3. 调整日志级别

```python
config = {
    ...
    "log_file": False,  # 禁用日志文件输出
}
```

### 4. 并行回测

对于参数搜索，使用多进程并行：

```python
from multiprocessing import Pool

with Pool(processes=8) as pool:
    results = pool.map(run_backtest, configs)
```

## 最佳实践

### 1. 避免前视偏差

确保策略只使用当前及之前的数据：

```python
def should_open(self):
    # 正确：使用当前K线数据
    if self.current_price > self.ma20:
        return PositionSide.LONG
    
    # 错误：使用未来数据（前视偏差）
    # if self.next_price > self.current_price:  # 不要这样做
```

### 2. 合理设置预热期

指标需要足够的历史数据才能稳定：

```python
config = {
    # 如果使用20日均线，预热期至少20根K线
    "pre_start": start_time - 20 * timeframe_ms,
    "pre_end": start_time,
    ...
}
```

### 3. 考虑交易成本

回测执行器默认模拟滑点和手续费，确保配置合理：

```python
config = {
    "executor_config": {
        "slippage": 0.001,  # 0.1% 滑点
        "commission": 0.0005,  # 0.05% 手续费
    }
}
```

### 4. 使用走向前验证

避免过拟合，使用滚动窗口验证：

```python
config = BacktestConfig(
    mode=BacktestMode.WALK_FORWARD,
    train_days=60,
    test_days=14,
    embargo_days=1,
)
```

## 相关模块

- [策略模块](01-strategy.md) - 策略开发指南
- [数据源](11-data-sources.md) - 数据源配置
- [因子模块](06-factor.md) - 因子回测
