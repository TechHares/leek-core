# 22 事件总线

## 概述

事件总线（EventBus）是 Leek Core 的核心通信组件，采用发布-订阅模式实现组件间的松耦合通信。事件总线使用主队列分发机制，保证事件处理的顺序性，同时通过线程池实现高效的并发处理。

## 核心组件

### `EventBus` - 事件总线

```python
class EventBus:
    """优化版事件总线，使用主队列分发机制，保证顺序性的同时控制线程数量"""
    
    def __init__(self, max_workers: int = 10):
        """
        初始化事件总线
        
        参数:
            max_workers: 线程池最大工作线程数
        """
        ...
    
    # 订阅管理
    def subscribe_event(self, event_type: EventType, callback: Callable) -> bool: ...
    def unsubscribe_event(self, event_type: EventType, callback: Callable) -> bool: ...
    
    # 事件发布
    def publish_event(self, event: Event) -> None: ...
    
    # 生命周期
    def shutdown(self) -> None: ...
    
    # 调试信息
    def get_subscription_info(self) -> dict: ...
```

### `Event` - 事件对象

```python
class Event:
    """事件对象"""
    
    def __init__(
        self, 
        event_type: EventType,      # 事件类型
        data: Any = None,           # 事件数据
        source: EventSource = None  # 事件源
    ):
        self.event_type = event_type
        self.data = data
        self.source = source
```

### `EventType` - 事件类型

```python
class EventType(Enum):
    """事件类型定义"""
    
    # 数据事件
    DATA_SOURCE_SUBSCRIBE = "data_source_subscribe"      # 数据源订阅
    DATA_SOURCE_UNSUBSCRIBE = "data_source_unsubscribe"  # 数据源取消订阅
    DATA_RECEIVED = "data_received"                      # 接收到数据
    DATA_REQUEST = "data_request"                        # 数据请求
    
    # 策略事件
    STRATEGY_SIGNAL_MANUAL = "strategy_signal_manual"    # 手动信号
    STRATEGY_SIGNAL = "strategy_signal"                  # 策略产生信号
    STRATEGY_SIGNAL_FINISH = "strategy_signal_finish"    # 策略信号完成
    
    # 风控事件
    RISK_TRIGGERED = "risk_triggered"                    # 风控触发事件
    
    # 仓位管理事件
    POSITION_POLICY_ADD = "position_policy_add"          # 仓位风控添加
    POSITION_POLICY_DEL = "position_policy_del"          # 仓位风控删除
    POSITION_INIT = "position_init"                      # 仓位管理初始化
    POSITION_UPDATE = "position_update"                  # 仓位更新
    
    # 订单执行事件
    EXEC_ORDER_CREATED = "exec_order_created"            # 执行订单创建
    EXEC_ORDER_UPDATED = "exec_order_updated"            # 执行订单更新
    ORDER_CREATED = "order_created"                      # 订单创建
    ORDER_UPDATED = "order_updated"                      # 订单更新
    
    # 资金管理事件
    TRANSACTION = "transaction"                          # 资金流水
```

### `EventSource` - 事件源

```python
@dataclass
class EventSource:
    """事件源"""
    instance_id: str          # 实例ID
    name: str                 # 名称
    cls: str = None           # 类名
    extra: dict = field(default_factory=dict)  # 额外信息
```

## 架构设计

### 内部结构

```text
EventBus
├── _main_event_queue        # 主事件队列
├── _thread_pool             # 线程池 (ThreadPoolExecutor)
├── _dispatcher_thread       # 分发线程
├── _subscribers             # 特定事件订阅者 Dict[EventType, Set[Callable]]
├── _all_event_subscribers   # 全局事件订阅者 Set[Callable]
├── _subscriber_queues       # 订阅者队列 Dict[Tuple, Queue]
└── _ready_subscribers       # 空闲订阅者集合
```

### 事件分发流程

```text
publish_event(event)
        │
        ▼
    主事件队列
        │
        ▼
  分发线程(_event_dispatcher)
        │
        ├──► 全局订阅者队列
        │
        └──► 特定类型订阅者队列
                    │
                    ▼
            调度空闲订阅者
                    │
                    ▼
              线程池执行
                    │
                    ▼
            callback(event)
                    │
                    ▼
            标记订阅者空闲
```

### 顺序保证机制

每个订阅者维护独立的事件队列，确保同一订阅者处理事件的顺序性：

```text
订阅者A的队列: [Event1] -> [Event2] -> [Event3]
                    ↓
              顺序处理
              
订阅者B的队列: [Event1] -> [Event2]
                    ↓
              顺序处理
              
（不同订阅者之间并行处理）
```

## 使用示例

### 基础用法

```python
from leek_core.event import EventBus, Event, EventType

# 创建事件总线
event_bus = EventBus(max_workers=10)

# 定义回调函数
def on_data_received(event: Event):
    print(f"收到数据: {event.data}")

def on_order_updated(event: Event):
    print(f"订单更新: {event.data}")

# 订阅事件
event_bus.subscribe_event(EventType.DATA_RECEIVED, on_data_received)
event_bus.subscribe_event(EventType.ORDER_UPDATED, on_order_updated)

# 发布事件
event_bus.publish_event(Event(
    event_type=EventType.DATA_RECEIVED,
    data={"symbol": "BTC_USDT", "price": 50000}
))

# 关闭事件总线
event_bus.shutdown()
```

### 订阅所有事件

```python
def on_any_event(event: Event):
    print(f"事件: {event.event_type.value}, 数据: {event.data}")

# 传入 None 或空字符串订阅所有事件
event_bus.subscribe_event(None, on_any_event)
```

### 带事件源的事件

```python
from leek_core.event import EventSource

# 创建带来源的事件
event = Event(
    event_type=EventType.STRATEGY_SIGNAL,
    data=signal,
    source=EventSource(
        instance_id="strategy-001",
        name="双均线策略",
        cls="MyStrategy"
    )
)

event_bus.publish_event(event)
```

### 取消订阅

```python
# 取消特定事件订阅
event_bus.unsubscribe_event(EventType.DATA_RECEIVED, on_data_received)

# 取消全局订阅
event_bus.unsubscribe_event(None, on_any_event)
```

### 查看订阅信息

```python
info = event_bus.get_subscription_info()
print(info)
# 输出:
# {
#     "subscribers": {"DATA_RECEIVED": 2, "ORDER_UPDATED": 1},
#     "all_event_subscribers_count": 1,
#     "subscriber_queues_count": 4,
#     "ready_subscribers_count": 4,
#     "thread_pool_max_workers": 10,
#     "dispatcher_thread_alive": True,
#     "started": True
# }
```

## 常用事件流

### 数据流事件

```text
DataSource
    │
    ▼
DATA_RECEIVED ──► StrategyManager
                        │
                        ▼
                STRATEGY_SIGNAL ──► Portfolio
                                        │
                                        ▼
                                EXEC_ORDER_CREATED ──► ExecutorManager
```

### 订单流事件

```text
Executor
    │
    ▼
ORDER_UPDATED ──► Engine.on_order_update()
                        │
                        ├──► Portfolio
                        │       │
                        │       ▼
                        │   POSITION_UPDATE
                        │
                        └──► EXEC_ORDER_UPDATED ──► StrategyManager
```

### 风控事件

```text
RiskManager
    │
    ▼
RISK_TRIGGERED ──► (可被外部监听)
```

## 配置说明

### 线程池配置

```python
# 默认10个工作线程
event_bus = EventBus(max_workers=10)

# 高负载场景增加线程数
event_bus = EventBus(max_workers=20)

# 低延迟场景减少线程切换
event_bus = EventBus(max_workers=5)
```

### 在引擎中使用

```python
from leek_core.engine import SimpleEngine
from leek_core.event import EventBus

# 使用自定义事件总线
custom_bus = EventBus(max_workers=15)
engine = SimpleEngine(
    instance_id="engine-001",
    name="主引擎",
    event_bus=custom_bus,
    ...
)
```

## 最佳实践

### 1. 回调函数异常处理

事件总线内部会捕获回调异常并记录日志，但建议在回调中也添加保护：

```python
def on_order_updated(event: Event):
    try:
        order = event.data
        # 处理逻辑
    except Exception as e:
        logger.error(f"处理订单更新失败: {e}", exc_info=True)
```

### 2. 避免阻塞操作

回调函数中避免长时间阻塞操作，以免影响其他事件处理：

```python
# 不推荐
def on_event(event: Event):
    time.sleep(10)  # 阻塞会延迟后续事件处理

# 推荐：异步处理耗时操作
def on_event(event: Event):
    threading.Thread(target=long_running_task, args=(event.data,)).start()
```

### 3. 正确关闭

确保在程序退出前关闭事件总线：

```python
try:
    # 主逻辑
    pass
finally:
    event_bus.shutdown()
```

### 4. 事件数据不可变

建议事件数据使用不可变对象或深拷贝，避免多个订阅者修改同一数据：

```python
from copy import deepcopy

def on_event(event: Event):
    data = deepcopy(event.data)  # 使用副本
    data["processed"] = True
```

## SerializableEventBus

回测场景使用的同步事件总线，用于确保事件处理的确定性：

```python
from leek_core.event import SerializableEventBus

# 回测使用同步事件总线
event_bus = SerializableEventBus()
```

特点：
- 同步处理事件，保证顺序确定性
- 无线程池开销
- 适用于回测等需要确定性的场景

## 相关模块

- [引擎架构](20-engine.md) - Engine 中的事件使用
- [策略模块](01-strategy.md) - 策略信号事件
- [执行器](23-executor.md) - 订单相关事件
