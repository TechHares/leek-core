# 20 引擎模块

## 概述

引擎模块（Engine）是 Leek Core 的核心调度中心，负责协调数据源、策略、执行器和仓位管理等各个组件的协同工作。引擎采用事件驱动架构，通过 EventBus 实现组件间的松耦合通信。

## 核心组件

### 组件层次结构

```text
SimpleEngine
├── EventBus                 # 事件总线
├── DataManager              # 数据源管理器
│   └── DataSourceContext    # 数据源上下文
│       └── DataSource[]     # 数据源实例
├── StrategyManager          # 策略管理器
│   └── StrategyContext      # 策略上下文
│       └── StrategyWrapper  # 策略包装器
│           ├── Strategy     # 策略实例
│           └── SubStrategy[]# 子策略
├── ExecutorManager          # 执行器管理器
│   └── ExecutorContext      # 执行器上下文
│       └── Executor[]       # 执行器实例
├── Portfolio                # 投资组合管理
│   ├── CapitalAccount       # 资金账户
│   ├── PositionTracker      # 仓位跟踪器
│   └── RiskManager          # 风险管理器
└── ComponentManager[]       # 通用组件管理器
```

### `SimpleEngine` - 执行引擎

执行引擎是系统的主入口，负责：

- 初始化和管理所有子组件
- 处理数据流转和信号处理
- 订单执行和仓位更新的协调

```python
class SimpleEngine(LeekComponent):
    """执行引擎"""
    
    def __init__(
        self, 
        instance_id: str,           # 引擎实例ID
        name: str,                  # 引擎名称
        position_config: PositionConfig = None,  # 仓位配置
        max_workers: int = 0,       # 策略并行工作线程数
        event_bus: EventBus = None  # 自定义事件总线
    ):
        ...
    
    # 数据处理
    def on_data(self, data: Data) -> None: ...
    
    # 订单更新处理
    def on_order_update(self, event: Event) -> None: ...
    
    # 生命周期
    def on_start(self) -> None: ...
    def on_stop(self) -> None: ...
    
    # 组件管理
    def add_strategy(self, config: LeekComponentConfig) -> None: ...
    def remove_strategy(self, instance_id: str) -> None: ...
    def add_executor(self, config: LeekComponentConfig) -> None: ...
    def remove_executor(self, instance_id: str) -> None: ...
    def add_data_source(self, config: LeekComponentConfig) -> None: ...
    def remove_data_source(self, instance_id: str) -> None: ...
    
    # 状态查询
    def engine_state(self) -> dict: ...
    def get_position_state(self) -> dict: ...
    def get_strategy_state(self) -> dict: ...
```

## 数据流程

### 1. 数据接收流程

```text
DataSource.push_data()
        │
        ▼
DataManager.on_data()
        │
        ▼
Engine.on_data(data)
        │
        ├──► StrategyManager.process_data(data)
        │           │
        │           ▼
        │    Strategy.on_data(data)
        │           │
        │           ▼
        │    Signal (if any)
        │           │
        │           ▼
        │    Engine._on_signal(signal)
        │
        └──► PositionTracker.on_data(data)
                    │
                    ▼
             更新仓位市价
```

### 2. 信号处理流程

```text
Engine._on_signal(signal)
        │
        ├──► EventBus.publish(STRATEGY_SIGNAL)
        │
        ▼
Portfolio.process_signal(signal)
        │
        ▼
ExecutionContext (or None)
        │
        ▼
RiskManager.evaluate_risk()
        │
        ▼
CapitalAccount.freeze_amount()
        │
        ▼
EventBus.publish(EXEC_ORDER_CREATED)
        │
        ▼
ExecutorManager.handle_order()
```

### 3. 订单更新流程

```text
Executor._trade_callback(order)
        │
        ▼
EventBus.publish(ORDER_UPDATED)
        │
        ▼
Engine.on_order_update(event)
        │
        ├──► Portfolio.order_update(order)
        │           │
        │           ├── CapitalAccount.unfreeze_amount()
        │           └── PositionTracker.order_update()
        │
        └──► ExecutorManager.order_update()
                    │
                    ▼
             StrategyManager.exec_update()
```

## 生命周期

### 启动顺序

```python
def on_start(self):
    """启动引擎组件"""
    # 1. 启动数据源管理器
    self.data_source_manager.on_start()
    
    # 2. 启动策略管理器
    self.strategy_manager.on_start()
    
    # 3. 启动执行器管理器
    self.executor_manager.on_start()
    
    self.running = True
```

### 停止顺序

```python
def on_stop(self):
    """引擎停止时的回调"""
    self.running = False
    
    # 1. 策略先停止（会发布数据源取消订阅事件）
    self.strategy_manager.on_stop()
    
    # 2. 执行器停止
    self.executor_manager.on_stop()
    
    # 3. 数据源最后停止（确保能处理取消订阅）
    self.data_source_manager.on_stop()
    
    # 4. 关闭事件总线
    self.event_bus.shutdown()
```

## 配置说明

### 引擎初始化配置

```python
from leek_core.engine import SimpleEngine
from leek_core.models import PositionConfig
from decimal import Decimal

engine = SimpleEngine(
    instance_id="engine-001",
    name="主引擎",
    position_config=PositionConfig(
        init_amount=Decimal("100000"),      # 初始资金
        max_amount=Decimal("10000"),        # 单次最大金额
        max_ratio=Decimal("0.1"),           # 单次最大比例
        max_strategy_amount=Decimal("50000"), # 单策略最大金额
        max_strategy_ratio=Decimal("0.5"),   # 单策略最大比例
        max_symbol_amount=Decimal("25000"),  # 单品种最大金额
        max_symbol_ratio=Decimal("0.25"),    # 单品种最大比例
        default_leverage=1,                  # 默认杠杆
    ),
    max_workers=4,  # 策略处理线程数，0表示同步处理
)
```

### 组件配置格式

```python
from leek_core.models import LeekComponentConfig

# 策略配置
strategy_config = LeekComponentConfig(
    instance_id="strategy-001",
    name="双均线策略",
    cls=MyStrategy,  # 或字符串 "module.MyStrategy"
    config=StrategyConfig(
        data_source_configs=[...],      # 数据源配置
        info_fabricator_configs=[...],  # 信息处理器配置
        strategy_config={"param1": 10}, # 策略参数
        strategy_position_config=None,  # 策略仓位配置
        risk_policies=[...],            # 风控策略配置
    ),
    data=None,  # 初始化数据
)

# 执行器配置
executor_config = LeekComponentConfig(
    instance_id="executor-001",
    name="Gate执行器",
    cls="leek_core.executor.GateExecutor",
    config={
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
    }
)

# 数据源配置
datasource_config = LeekComponentConfig(
    instance_id="datasource-001",
    name="Gate WebSocket",
    cls="leek_core.data.GateSource",
    config={
        "ws_url": "wss://...",
        "symbols": ["BTC_USDT", "ETH_USDT"],
    }
)
```

## 使用示例

### 基础用法

```python
from leek_core.engine import SimpleEngine
from leek_core.models import PositionConfig, LeekComponentConfig
from decimal import Decimal

# 1. 创建引擎
engine = SimpleEngine(
    instance_id="prod-engine",
    name="生产引擎",
    position_config=PositionConfig(
        init_amount=Decimal("100000"),
        max_amount=Decimal("5000"),
        max_ratio=Decimal("0.05"),
    )
)

# 2. 添加组件
engine.add_data_source(LeekComponentConfig(...))
engine.add_strategy(LeekComponentConfig(...))
engine.add_executor(LeekComponentConfig(...))

# 3. 启动引擎
engine.on_start()

# 4. 运行中查询状态
state = engine.engine_state()
print(f"数据源数量: {state['state']['data_source_count']}")
print(f"策略数量: {state['state']['strategy_count']}")
print(f"CPU使用率: {state['resources']['cpu']['value']}")

# 5. 停止引擎
engine.on_stop()
```

### 动态管理组件

```python
# 运行时添加策略
engine.add_strategy(LeekComponentConfig(
    instance_id="strategy-002",
    name="新策略",
    cls="my_module.NewStrategy",
    config=StrategyConfig(...)
))

# 运行时移除策略
engine.remove_strategy("strategy-001")

# 更新策略状态
engine.update_strategy_state("strategy-002", {"param1": 20})

# 更新仓位配置
engine.update_position_config(PositionConfig(
    init_amount=Decimal("200000"),
    ...
))
```

### 手动平仓

```python
# 查找并关闭仓位
position_state = engine.get_position_state()
for pos in position_state.get("position", {}).get("positions", []):
    if pos["symbol"] == "BTC_USDT":
        engine.close_position(pos["position_id"])
```

## 状态查询

### 引擎状态

```python
state = engine.engine_state()
# 返回结构:
{
    "state": {
        "process_id": 12345,
        "data_source_count": 2,
        "strategy_count": 3,
        "executor_count": 1,
    },
    "resources": {
        "cpu": {"percent": 25.5, "value": "25.5%", "status": "success"},
        "memory": {"percent": 45.2, "value": "8.1G/16G", "status": "success"},
        "disk": {"percent": 60.0, "value": "120G/200G", "status": "warning"},
    }
}
```

### 仓位状态

```python
position_state = engine.get_position_state()
# 返回结构:
{
    "total_value": "105000.00",
    "position_value": "25000.00",
    "pnl": "5000.00",
    "friction": "-50.00",
    "fee": "-100.00",
    "profit": "4850.00",
    "capital": {...},
    "position": {...},
}
```

### 策略状态

```python
strategy_state = engine.get_strategy_state()
# 返回结构:
{
    "strategy-001": {
        "instance_id": "strategy-001",
        "state": {...},
        "positions": [...],
    },
    ...
}
```

## 最佳实践

### 1. 资源管理

```python
# 使用 try-finally 确保引擎正确关闭
try:
    engine.on_start()
    # 主循环或等待
    while engine.running:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    engine.on_stop()
```

### 2. 错误处理

引擎内部已经对关键流程进行了异常捕获，但建议在外层也添加保护：

```python
try:
    engine.add_strategy(config)
except Exception as e:
    logger.error(f"添加策略失败: {e}")
```

### 3. 状态持久化

```python
import json

# 保存状态
state = {
    "position": engine.get_position_state(),
    "strategy": engine.get_strategy_state(),
}
with open("engine_state.json", "w") as f:
    json.dump(state, f)

# 恢复状态（需要在组件配置中传入）
with open("engine_state.json", "r") as f:
    saved_state = json.load(f)
# 通过 position_config.data 传入仓位状态
```

## 相关模块

- [事件总线](22-event-bus.md) - EventBus 详细说明
- [策略模块](01-strategy.md) - 策略开发指南
- [执行器](23-executor.md) - 订单执行器说明
- [仓位管理](03-position.md) - Portfolio 详细说明
