# 99 版本更新日志

本文档记录 Leek Core 的版本更新历史。

## 版本号规范

项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范：

- **主版本号 (MAJOR)**：不兼容的 API 变更
- **次版本号 (MINOR)**：向后兼容的功能新增
- **修订号 (PATCH)**：向后兼容的问题修复

## 更新记录

---

### [未发布]

#### 新增
- 完善项目文档体系
  - `00-overview.md` - 项目总览
  - `02-sub-strategy.md` - 子策略模块
  - `03-position.md` - 仓位管理
  - `04-risk.md` - 风险控制
  - `10-models.md` - 数据模型
  - `12-indicators.md` - 技术指标
  - `20-engine.md` - 引擎架构
  - `22-event-bus.md` - 事件总线
  - `23-executor.md` - 订单执行器
  - `30-backtest.md` - 回测系统
  - `40-alarm.md` - 告警系统
  - `41-ml.md` - 机器学习集成
  - `42-grpc.md` - gRPC服务
  - `90-architecture.md` - 整体架构
  - `91-component-guide.md` - 组件开发指南
  - `92-testing.md` - 测试指南

---

### [1.0.0] - 待定

#### 核心功能
- **引擎模块**
  - `SimpleEngine` 执行引擎
  - 事件驱动架构
  - 组件动态管理
  - gRPC 远程调用支持

- **策略模块**
  - `Strategy` 策略基类
  - `CTAStrategy` CTA策略基类
  - `StrategyMode` 策略运行模式
  - `StrategyWrapper` 策略生命周期管理

- **子策略模块**
  - `PositionStopLoss` 固定止损
  - `PositionTakeProfit` 固定止盈
  - `PositionTargetTrailingExit` 目标追踪离场

- **仓位管理**
  - `Portfolio` 投资组合管理
  - `CapitalAccount` 资金账户
  - `PositionTracker` 仓位跟踪
  - 多层级资金控制

- **风险控制**
  - `RiskManager` 风控管理器
  - `StrategyPolicy` 策略风控策略
  - 信号频率限制
  - 交易时间窗口
  - 盈利控制

- **数据源**
  - `DataSource` 数据源基类
  - `BinanceSource` Binance数据源
  - `GateSource` Gate.io数据源
  - `OKXSource` OKX数据源
  - `ClickHouseSource` ClickHouse数据源
  - WebSocket 实时数据支持

- **执行器**
  - `Executor` 执行器基类
  - `WebSocketExecutor` WebSocket执行器
  - `GateRestExecutor` Gate REST执行器
  - `BinanceExecutor` Binance执行器
  - `OKXExecutor` OKX执行器
  - `BacktestExecutor` 回测执行器

- **技术指标**
  - 趋势指标：MA, EMA, MACD, BOLL, SuperTrend
  - 震荡指标：RSI, KDJ, CCI, WR
  - 波动率指标：ATR, Keltner
  - 特色指标：缠论(Chan), CZSC, Hurst, RSRS

- **回测系统**
  - `BacktestRunner` 回测运行器
  - 多种回测模式（单次/参数搜索/走向前）
  - 完整的绩效指标
  - 统计检验支持

- **机器学习**
  - `FeatureEngine` 特征工程引擎
  - Alpha101/Alpha158/Alpha191 因子库
  - XGBoost/GRU 训练器
  - 模型评估器

- **事件系统**
  - `EventBus` 事件总线
  - 异步事件处理
  - 订阅者队列机制

- **告警系统**
  - `DingDingAlarmSender` 钉钉告警
  - `FeishuAlarmSender` 飞书告警

- **工具模块**
  - 日志系统
  - 时间工具
  - 序列化工具
  - 装饰器工具

---

## 变更类型说明

- **新增 (Added)**: 新功能
- **变更 (Changed)**: 对现有功能的变更
- **弃用 (Deprecated)**: 即将被移除的功能
- **移除 (Removed)**: 已移除的功能
- **修复 (Fixed)**: Bug修复
- **安全 (Security)**: 安全相关修复

---

## 升级指南

### 从 0.x 升级到 1.0

1. **API 变更**
   - `PositionContext` 重构为 `Portfolio`
   - 组件配置使用 `LeekComponentConfig`

2. **配置变更**
   - 仓位配置使用 `PositionConfig` 数据类
   - 策略配置使用 `StrategyConfig` 数据类

3. **事件类型**
   - 事件类型统一使用 `EventType` 枚举

---

## 贡献指南

提交更新日志时请遵循以下格式：

```markdown
### [版本号] - 日期

#### 新增
- 功能描述 (#Issue号)

#### 变更
- 变更描述

#### 修复
- Bug描述 (#Issue号)
```

---

## 相关链接

- [GitHub Releases](https://github.com/xxx/leek-core/releases)
- [项目总览](00-overview.md)
- [整体架构](90-architecture.md)
