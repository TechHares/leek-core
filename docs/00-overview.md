# 00 Leek Core 概述

## 简介

Leek Core 是一个专业的量化交易框架核心库，提供完整的策略开发、回测验证和实盘交易能力。框架采用事件驱动架构，支持多数据源、多策略并行运行，内置丰富的技术指标和风控模块。

## 核心特性

- **事件驱动架构**：基于 EventBus 的异步事件处理，组件间松耦合
- **多交易所支持**：内置 Binance、Gate、OKX 等主流交易所适配器
- **完整的策略体系**：支持 CTA 策略、机器学习策略等多种策略类型
- **专业的回测引擎**：高性能回测，支持 K 线模拟、绩效分析、统计检验
- **灵活的风控系统**：多层级风控（策略级、仓位级、全局级）
- **丰富的技术指标**：MACD、RSI、缠论等 20+ 种技术指标

## 架构概览

```text
┌─────────────────────────────────────────────────────────────────┐
│                         SimpleEngine                             │
│                         (执行引擎)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ DataManager  │    │StrategyMgr  │    │ExecutorMgr   │       │
│  │  (数据管理)   │    │ (策略管理)   │    │ (执行器管理)  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  DataSource  │    │   Strategy   │    │   Executor   │       │
│  │   (数据源)    │    │    (策略)    │    │   (执行器)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                         EventBus (事件总线)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Portfolio   │    │PositionTracker│   │CapitalAccount│       │
│  │ (投资组合)    │    │  (仓位跟踪)   │    │  (资金账户)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 数据流

```text
DataSource ──► Engine.on_data() ──► StrategyManager ──► Signal
                                                           │
                                                           ▼
Executor ◄── ExecutorManager ◄── Portfolio.process_signal()
    │
    ▼
Order ──► Engine.on_order_update() ──► Portfolio.order_update()
                                              │
                                              ▼
                                       PositionTracker
```

## 模块说明

| 模块 | 说明 | 文档链接 |
|------|------|----------|
| `strategy/` | 策略抽象与实现 | [01-strategy.md](01-strategy.md) |
| `sub_strategy/` | 子策略（止盈止损） | [02-sub-strategy.md](02-sub-strategy.md) |
| `position/` | 仓位与资金管理 | [03-position.md](03-position.md) |
| `risk/` | 风险控制 | [04-risk.md](04-risk.md) |
| `indicators/` | 技术指标 | [12-indicators.md](12-indicators.md) |
| `data/` | 数据源 | [11-data-sources.md](11-data-sources.md) |
| `engine/` | 引擎核心 | [20-engine.md](20-engine.md) |
| `event/` | 事件总线 | [22-event-bus.md](22-event-bus.md) |
| `executor/` | 订单执行器 | [23-executor.md](23-executor.md) |
| `backtest/` | 回测系统 | [30-backtest.md](30-backtest.md) |
| `models/` | 数据模型 | [10-models.md](10-models.md) |
| `utils/` | 工具类 | [21-logging.md](21-logging.md) |

## 快速开始

### 安装

```bash
pip install leek-core
```

### 编写第一个策略

```python
from leek_core.strategy import CTAStrategy
from leek_core.models import PositionSide, KLine

class MyFirstStrategy(CTAStrategy):
    """简单的双均线策略"""
    display_name = "双均线策略"
    
    def __init__(self):
        super().__init__()
        self.fast_ma = []
        self.slow_ma = []
    
    def on_kline(self, kline: KLine):
        # 计算均线
        self.fast_ma.append(float(kline.close))
        self.slow_ma.append(float(kline.close))
        
        if len(self.fast_ma) > 5:
            self.fast_ma = self.fast_ma[-5:]
        if len(self.slow_ma) > 20:
            self.slow_ma = self.slow_ma[-20:]
    
    def should_open(self) -> PositionSide | None:
        if len(self.slow_ma) < 20:
            return None
        
        fast = sum(self.fast_ma) / len(self.fast_ma)
        slow = sum(self.slow_ma) / len(self.slow_ma)
        
        if fast > slow:
            return PositionSide.LONG
        elif fast < slow:
            return PositionSide.SHORT
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        if len(self.slow_ma) < 20:
            return False
        
        fast = sum(self.fast_ma) / len(self.fast_ma)
        slow = sum(self.slow_ma) / len(self.slow_ma)
        
        if position_side == PositionSide.LONG:
            return fast < slow
        else:
            return fast > slow
```

### 运行回测

```python
from leek_core.backtest import run_backtest

config = {
    "strategy_class": "my_module.MyFirstStrategy",
    "strategy_params": {},
    "datasource_class": "leek_core.data.ClickHouseSource",
    "datasource_config": {
        "host": "localhost",
        "database": "market_data"
    },
    "symbol": "BTC",
    "quote_currency": "USDT",
    "timeframe": "1h",
    "start_time": 1704067200000,  # 2024-01-01
    "end_time": 1706745600000,    # 2024-02-01
    "initial_balance": 10000,
}

result = run_backtest(config)
print(f"总收益率: {result.metrics.total_return:.2%}")
print(f"夏普比率: {result.metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {result.metrics.max_drawdown:.2%}")
```

### 实盘运行

```python
from leek_core.engine import SimpleEngine
from leek_core.models import PositionConfig, LeekComponentConfig
from decimal import Decimal

# 初始化引擎
engine = SimpleEngine(
    instance_id="prod-1",
    name="生产引擎",
    position_config=PositionConfig(
        init_amount=Decimal("10000"),
        max_amount=Decimal("1000"),
        max_ratio=Decimal("0.1"),
    )
)

# 添加数据源
engine.add_data_source(LeekComponentConfig(
    instance_id="gate-ws",
    name="Gate WebSocket",
    cls="leek_core.data.GateSource",
    config={"api_key": "xxx", "api_secret": "xxx"}
))

# 添加策略
engine.add_strategy(LeekComponentConfig(
    instance_id="strategy-1",
    name="双均线策略",
    cls="my_module.MyFirstStrategy",
    config={...}
))

# 添加执行器
engine.add_executor(LeekComponentConfig(
    instance_id="gate-executor",
    name="Gate执行器",
    cls="leek_core.executor.GateExecutor",
    config={"api_key": "xxx", "api_secret": "xxx"}
))

# 启动引擎
engine.on_start()
```

## 核心概念

### 1. 组件（LeekComponent）

所有模块都继承自 `LeekComponent` 基类，提供统一的生命周期管理：

- `on_start()`: 组件启动
- `on_stop()`: 组件停止
- `get_state()`: 获取状态
- `load_state()`: 加载状态

### 2. 事件（Event）

系统通过事件进行组件间通信，主要事件类型：

| 事件类型 | 说明 |
|----------|------|
| `DATA_RECEIVED` | 接收到市场数据 |
| `STRATEGY_SIGNAL` | 策略产生交易信号 |
| `ORDER_CREATED` | 订单创建 |
| `ORDER_UPDATED` | 订单状态更新 |
| `POSITION_UPDATE` | 仓位更新 |

### 3. 信号（Signal）

策略产生的交易意图，包含：

- 交易资产列表
- 开仓/平仓方向
- 目标比例
- 策略配置

### 4. 执行上下文（ExecutionContext）

从信号转换而来的执行指令，经过风控评估后发送给执行器。

## 配置说明

### 仓位配置（PositionConfig）

```python
PositionConfig(
    init_amount=Decimal("100000"),      # 初始资金
    max_amount=Decimal("10000"),        # 单次最大金额
    max_ratio=Decimal("0.1"),           # 单次最大比例
    max_strategy_amount=Decimal("50000"), # 单策略最大金额
    max_strategy_ratio=Decimal("0.5"),   # 单策略最大比例
    max_symbol_amount=Decimal("25000"),  # 单品种最大金额
    max_symbol_ratio=Decimal("0.25"),    # 单品种最大比例
    default_leverage=1,                  # 默认杠杆
)
```

## 目录结构

```text
leek-core/
├── src/leek_core/
│   ├── adapts/          # 交易所适配器
│   ├── alarm/           # 告警模块
│   ├── backtest/        # 回测系统
│   ├── base/            # 基础组件
│   ├── data/            # 数据源
│   ├── engine/          # 引擎核心
│   ├── event/           # 事件总线
│   ├── executor/        # 订单执行器
│   ├── indicators/      # 技术指标
│   ├── manager/         # 管理器
│   ├── models/          # 数据模型
│   ├── policy/          # 策略控制
│   ├── position/        # 仓位管理
│   ├── risk/            # 风险控制
│   ├── strategy/        # 策略模块
│   ├── sub_strategy/    # 子策略
│   └── utils/           # 工具类
└── docs/                # 文档
```

## 相关文档

- [策略开发指南](01-strategy.md)
- [因子模块](06-factor.md)
- [数据源配置](11-data-sources.md)
- [引擎架构](20-engine.md)
- [日志系统](21-logging.md)
