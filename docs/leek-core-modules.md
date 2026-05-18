# leek-core 模块职责划分

## 概述

leek-core 是 Leek 量化交易系统的核心引擎，采用事件驱动架构，通过 EventBus 实现组件间解耦通信。系统支持实时交易和回测两种运行模式。

## 核心模块

### 1. engine - 执行引擎

**路径**: `src/leek_core/engine/`

**核心文件**:
- `engine.py`: SimpleEngine 主引擎类
- `grpc_engine.py`: gRPC 分布式引擎
- `strategy_debug.py`: 策略调试工具
- `indicator_view.py`: 指标视图

**职责**:
- 协调各组件（数据源、策略、执行器、风控）的生命周期
- 处理数据事件流：接收数据 → 分发给策略 → 收集信号 → 执行订单
- 管理事件总线 EventBus
- 提供策略管理器 (StrategyManager)、数据管理器 (DataManager)、执行器管理器 (ExecutorManager)

---

### 2. executor - 交易执行器

**路径**: `src/leek_core/executor/`

**核心文件**:
- `base.py`: Executor 抽象基类
- `binance.py`: Binance 执行器实现
- `okx.py`: OKX 执行器实现
- `gate.py`: Gate.io 执行器实现
- `backtest.py`: 回测专用执行器
- `context.py`: 执行上下文

**职责**:
- 接收订单 (Order) 并发送到交易所
- 实现下单、撤单、查询订单状态
- 处理订单成交回调
- 支持模拟交易和实盘交易模式

---

### 3. data - 数据源层

**路径**: `src/leek_core/data/`

**核心文件**:
- `base.py`: DataSource 抽象基类
- `binance_source.py`: Binance 数据源
- `okx_source.py`: OKX 数据源
- `gate_source.py`: Gate.io 数据源
- `clickhouse_source.py`: ClickHouse 历史数据源
- `redis_clickhouse_source.py`: Redis+ClickHouse 混合源
- `websocket.py`: WebSocket 实时数据
- `context.py`: 数据上下文

**职责**:
- 获取历史 K 线数据
- 订阅实时行情数据
- 提供统一的数据接口 Data
- 支持多种数据源类型（REST API、WebSocket、数据库）

---

### 4. strategy - 策略层

**路径**: `src/leek_core/strategy/`

**核心文件**:
- `base.py`: Strategy 抽象基类
- `cta.py`: CTA 策略框架
- `xgboost_strategy.py`: XGBoost 机器学习策略
- `gru_strategy.py`: GRU 深度学习策略
- `ml.py`: 机器学习策略框架
- `context.py`: 策略上下文
- `strategy_dmi.py`: DMI 指标策略示例
- `strategy_mode.py`: 策略运行模式定义

**职责**:
- 实现择时交易逻辑
- 生成交易信号 (Signal)
- 管理策略状态序列化/反序列化
- 支持单标的和多标的策略模式

---

### 5. risk - 风控层

**路径**: `src/leek_core/risk/`

**核心文件**:
- `base.py`: RiskPlugin 抽象基类
- `context.py`: 风控上下文
- `plugins/`: 风控插件目录

**职责**:
- 定义风控规则接口
- 检查仓位是否符合风控条件
- 触发止损、止盈、强平等风控操作
- 支持灵活扩展多种风控逻辑

---

### 6. backtest - 回测引擎

**路径**: `src/leek_core/backtest/`

**核心文件**:
- `backtest.py`: EnhancedBacktester 主回测类
- `runner.py`: 回测运行器
- `types.py`: 回测类型定义
- `factor_evaluation.py`: 因子评估工具
- `performance.py`: 性能指标计算
- `data_cache.py`: 回测数据缓存
- `statistical_tests.py`: 统计检验

**职责**:
- 执行策略回测
- 支持单次回测和 Walk-Forward 分析
- 参数优化 (Optuna)
- 计算性能指标 (收益率、夏普比率、最大回撤等)
- 因子评估与统计检验

---

### 7. position - 仓位与资金管理

**路径**: `src/leek_core/position/`

**核心文件**:
- `portfolio.py`: Portfolio 投资组合类
- `position_tracker.py`: PositionTracker 仓位跟踪器
- `capital_account.py`: CapitalAccount 资金账户
- `risk.py`: 风控相关

**职责**:
- 跟踪和管理所有仓位状态
- 计算持仓盈亏
- 管理资金账户和可用余额
- 仓位数据查询和统计

---

## 适配器与工具层

### 8. adapts - 交易所适配器

**路径**: `src/leek_core/adapts/`

**核心文件**:
- `okx_adapter.py`: OKX 交易所适配器
- `binance_adapter.py`: Binance 交易所适配器
- `gate_adapter.py`: Gate.io 交易所适配器

**职责**:
- 封装各交易所 API 差异
- 提供统一的消息格式
- 处理交易所特有逻辑（签名、重试、心跳等）

---

### 9. indicators - 技术指标

**路径**: `src/leek_core/indicators/`

**核心模块**:
- `ma.py`: 均线系列 (MA, EMA, LLT, KAMA, FRAMA, WMA, HMA)
- `macd.py`: MACD 及其变体
- `kdj.py`: KDJ 指标
- `boll.py`: 布林带
- `atr.py`: ATR、TR
- `cci.py`: CCI 顺势指标
- `rsi.py`: RSI 及其变体
- `chan/`: 缠论指标 (笔、线段、走势中枢)
- `czsc/`: CZSC 禅师理论指标
- 其他: DK, WR, SAR, BBI, Bias, DMI, ADX, Ichimoku, DeMarker 等

**职责**:
- 提供 40+ 常用技术指标
- 支持自定义指标组合
- 为策略提供技术分析工具

---

### 10. event - 事件总线

**路径**: `src/leek_core/event/`

**核心文件**:
- `bus.py`: EventBus 事件总线实现
- `single_bus.py`: 单线程事件总线
- `types.py`: 事件类型定义

**职责**:
- 实现组件间解耦通信
- 管理事件订阅和分发
- 支持异步事件处理
- 定义系统事件类型 (ORDER_UPDATED, STRATEGY_SIGNAL 等)

---

### 11. models - 数据模型

**路径**: `src/leek_core/models/`

**核心文件**:
- `order.py`: Order 订单模型
- `position.py`: Position 仓位模型
- `data.py`: Data 数据模型
- `signal.py`: Signal 信号模型
- `config.py`: 配置模型
- `constants.py`: 常量定义
- `transaction.py`: 交易流水
- `risk_event.py`: 风控事件

**职责**:
- 定义系统核心数据结构
- 提供数据序列化/反序列化
- 统一数据交换格式

---

### 12. base - 基础组件

**路径**: `src/leek_core/base/`

**职责**:
- 定义 LeekComponent 组件基类
- 提供组件创建和加载机制
- 实现生命周期管理

---

### 13. manager - 组件管理器

**路径**: `src/leek_core/manager/`

**核心管理器**:
- `StrategyManager`: 策略管理器
- `DataManager`: 数据管理器
- `ExecutorManager`: 执行器管理器
- `ComponentManager`: 通用组件管理器

**职责**:
- 统一管理各类组件的注册、启动、停止
- 维护组件依赖关系
- 提供组件查询和状态监控

---

### 14. 其他辅助模块

| 模块 | 路径 | 职责 |
|------|------|------|
| alarm | `src/leek_core/alarm/` | 告警通知 |
| policy | `src/leek_core/policy/` | 策略决策 |
| ml | `src/leek_core/ml/` | 机器学习工具 |
| info_fabricator | `src/leek_core/info_fabricator/` | 数据生成 |
| sub_strategy | `src/leek_core/sub_strategy/` | 子策略框架 |
| utils | `src/leek_core/utils/` | 工具函数 |

---

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                         Engine                              │
│  (协调者: 接收数据 → 分发策略 → 收集信号 → 执行订单)           │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    DataSource      Strategy       Risk         Executor
         │              │              │              │
         ▼              ▼              ▼              ▼
    [binance]       [cta/ML]     [RiskPlugin]   [binance/okx/gate]
    [okx]          [策略信号]     [风控检查]     [交易所API]
    [gate]                                                   
                                                             
    Position ←──────────────────────────┘
    (Portfolio/PositionTracker/CapitalAccount)

    EventBus ←──────── 所有组件通过事件总线通信
```

## 数据流

```
交易所行情 → DataSource → Engine.on_data() → StrategyManager 
                                            ↓
                                      Signal 生成
                                            ↓
                                      Portfolio
                                            ↓
                                      ExecutorManager
                                            ↓
                                   交易所订单执行
```

---

*文档版本: 1.0*
*最后更新: 2026-02-02*
