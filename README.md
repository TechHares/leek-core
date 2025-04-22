# Leek Core

核心量化交易引擎，整合数据处理、策略开发、风控和交易执行功能。

## 项目亮点
- 事件驱动、组件化架构，类似 Spring Boot，支持灵活扩展与插件
- 数据源、策略、风控、仓位、执行五大核心子系统，解耦协作
- 支持实盘（OKX等）与回测统一接口，易于策略无缝切换
- 完善的订单生命周期与风控体系，支持多账户/多策略
- 丰富的日志、埋点、性能监控与告警扩展能力

## 项目结构

```
leek-core/
├── data/               # 数据模块，处理多源数据接入、对齐、存储和清洗
├── models/             # 数据模型模块，包含各模块间交互的DTO和通用数据结构
├── strategy/           # 策略开发模块，包含策略模板和机器学习集成
├── risk/               # 风控模块，提供全面的风险管理功能
├── executor/           # 执行模块，抽象交易接口与实盘/回测实现
├── backtest/           # 回测引擎，支持向量化和分布式并行回测
├── analysis/           # 绩效分析工具，包括收益归因和策略评估
├── engine/             # 交易引擎，事件驱动、组件管理、依赖注入
├── position/           # 仓位管理，支持多账户、多标的
├── utils/              # 通用工具和辅助函数
└── config/             # 配置文件管理
```

## 快速开始

### 安装
```bash
pip install leek-core
```
或源码安装：
```bash
git clone https://github.com/shenglin-li/leek-core.git
cd leek-core
pip install .
```

### 典型用法
```python
from engine import Engine
from executor import OkxWebSocketExecutor, BacktestExecutor
from strategy import MyStrategy

engine = Engine(instance_id="engine1")
engine.add_trader(OkxWebSocketExecutor(...))
engine.add_trader(BacktestExecutor(...))
engine.add_strategy(MyStrategy(...))
engine.run()
```

### 日志系统示例
```python
from utils import get_logger
logger = get_logger("app.module")
logger.info("系统初始化完成")
logger.error("操作失败", extra={"error_code": 500, "reason": "连接超时"})
```

## 开发指南
- 详见 [docs/](docs/) 目录，包含组件开发、插件扩展、策略模板等说明
- 支持自定义数据源、风控插件、仓位管理、策略等

## 贡献与支持
欢迎提交 issue/pr 参与共建！