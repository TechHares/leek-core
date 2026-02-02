# 23 订单执行器

## 概述

订单执行器（Executor）是 Leek Core 与交易所交互的核心组件，负责将系统产生的交易信号转化为实际的订单操作。执行器模块采用抽象设计，支持多种交易所的 REST API 和 WebSocket 实现。

## 核心组件

### 组件层次结构

```text
ExecutorManager (执行器管理器)
└── ExecutorContext (执行器上下文)
        └── Executor (执行器实例)
                ├── GateRestExecutor    # Gate.io REST执行器
                ├── GateWebSocketExecutor # Gate.io WS执行器
                ├── BinanceExecutor     # Binance执行器
                ├── OKXExecutor         # OKX执行器
                └── BacktestExecutor    # 回测执行器
```

### `Executor` - 执行器抽象基类

```python
class Executor(LeekComponent, ABC):
    """交易执行抽象基类"""
    
    just_backtest = False  # 仅用于回测，不实际执行
    
    def __init__(self):
        self.instance_id = None
        self.callback = None  # 回调函数，用于处理订单状态变化
    
    def check_order(self, order: Order) -> bool:
        """检查订单是否可执行"""
        return True
    
    @abstractmethod
    def send_order(self, order: Order | List[Order]):
        """下单"""
        raise NotImplementedError()
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """撤单"""
        raise NotImplementedError()
    
    def _trade_callback(self, order_update_message: OrderUpdateMessage):
        """交易回调，反馈成交信息"""
        if self.callback:
            self.callback(order_update_message)
```

### `WebSocketExecutor` - WebSocket执行器基类

```python
class WebSocketExecutor(Executor, ABC):
    """基于WebSocket的异步执行器抽象基类"""
    
    init_params = [
        Field(name="ws_url", label="WebSocket URL", type=FieldType.STRING, required=True),
        Field(name="heartbeat_interval", label="心跳间隔", type=FieldType.FLOAT, default=-1),
        Field(name="reconnect_interval", label="重连间隔", type=FieldType.FLOAT, default=5),
        Field(name="max_retries", label="最大重连次数", type=FieldType.INT, default=5),
    ]
    
    def __init__(self, ws_url: str, heartbeat_interval: float = -1, 
                 reconnect_interval: float = 5, max_retries: int = 5, **kwargs):
        ...
    
    @property
    def status(self) -> WSStatus: ...  # 连接状态
    
    def on_start(self): ...    # 启动WebSocket连接
    def on_stop(self): ...     # 停止WebSocket连接
    
    async def send(self, msg): ...           # 异步发送消息
    def async_send(self, message: str) -> bool: ...  # 同步上下文发送
    
    # 子类实现的回调方法
    async def on_message(self, msg): ...     # 收到消息
    async def on_open(self): ...             # 连接建立
    async def on_close(self): ...            # 连接关闭
    async def on_error(self, error): ...     # 错误处理
```

### `ExecutorContext` - 执行器上下文

```python
class ExecutorContext(LeekContext):
    """执行器上下文，管理单个执行器实例"""
    
    def __init__(self, event_bus: EventBus, config: LeekComponentConfig):
        self.orders: dict[str, Order] = {}  # 待处理订单
        self.executor: Executor = ...        # 执行器实例
        self.max_retry_count = 3             # 最大重试次数
        self.retry_interval = 1              # 重试间隔
    
    def check_order(self, order: Order) -> bool: ...
    def send_order(self, orders: List[Order]): ...
    def cancel_order(self, order_id: str, symbol: str, **kwargs): ...
```

## WebSocket 状态机

```python
class WSStatus(Enum):
    INIT = auto()          # 初始化
    CONNECTING = auto()    # 连接中
    CONNECTED = auto()     # 已连接
    DISCONNECTING = auto() # 断开中
    DISCONNECTED = auto()  # 已断开
    RECONNECTING = auto()  # 重连中
```

状态流转：

```text
INIT ──► CONNECTING ──► CONNECTED
              │              │
              │              ▼
              │      (正常运行)
              │              │
              │              ▼
              └──► RECONNECTING ◄── 连接断开
                       │
                       ▼
                 超过重试次数?
                   │      │
                  Yes     No
                   │      │
                   ▼      └──► CONNECTING
             DISCONNECTED
```

## 内置执行器

### Gate.io 执行器

#### GateRestExecutor

```python
class GateRestExecutor(Executor):
    """Gate.io REST API执行器"""
    
    display_name = "Gate.io REST"
    init_params = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, required=True),
        Field(name="secret_key", label="API Secret Key", type=FieldType.PASSWORD, required=True),
        Field(name="testnet", label="测试网", type=FieldType.BOOLEAN, default=False),
        Field(name="slippage_level", label="允许滑档", type=FieldType.INT, default=5),
    ]
```

特点：
- 支持现货和合约交易
- 内置订单状态轮询机制
- 支持限价单和市价单

#### GateWebSocketExecutor

WebSocket 实时订单推送，更低延迟。

### Binance 执行器

```python
class BinanceExecutor(Executor):
    """Binance执行器"""
    
    init_params = [
        Field(name="api_key", ...),
        Field(name="api_secret", ...),
        Field(name="testnet", ...),
    ]
```

### OKX 执行器

```python
class OKXExecutor(Executor):
    """OKX执行器"""
    
    init_params = [
        Field(name="api_key", ...),
        Field(name="api_secret", ...),
        Field(name="passphrase", ...),  # OKX特有
    ]
```

### 回测执行器

```python
class BacktestExecutor(Executor):
    """回测执行器，模拟订单执行"""
    
    just_backtest = True  # 标记为回测专用
```

## 订单数据模型

### Order - 订单

```python
@dataclass
class Order:
    order_id: str              # 订单ID
    position_id: str           # 仓位ID
    strategy_id: str           # 策略ID
    executor_id: str           # 执行器ID
    symbol: str                # 交易对
    quote_currency: str        # 计价币种
    ins_type: TradeInsType     # 交易品种类型
    side: PositionSide         # 方向
    sz: Decimal                # 数量
    order_price: Decimal       # 下单价格
    order_type: OrderType      # 订单类型
    order_status: OrderStatus  # 订单状态
    order_time: datetime       # 下单时间
    finish_time: datetime      # 完成时间
    execution_price: Decimal   # 成交价格
    settle_amount: Decimal     # 结算金额
    fee: Decimal               # 手续费
    pnl: Decimal               # 盈亏
    is_open: bool              # 是否开仓
    is_fake: bool              # 是否虚拟订单
    market_order_id: str       # 交易所订单ID
```

### OrderStatus - 订单状态

```python
class OrderStatus(Enum):
    CREATED = 0      # 已创建
    SUBMITTED = 1    # 已提交
    PARTIAL = 2      # 部分成交
    FILLED = 3       # 完全成交
    CANCELED = 4     # 已取消
    EXPIRED = 5      # 已过期
    ERROR = 6        # 错误
    
    @property
    def is_finished(self) -> bool:
        """订单是否已完结"""
        return self in [FILLED, CANCELED, EXPIRED, ERROR]
    
    @property
    def is_failed(self) -> bool:
        """订单是否失败"""
        return self in [CANCELED, EXPIRED, ERROR]
```

### OrderUpdateMessage - 订单更新消息

```python
@dataclass
class OrderUpdateMessage:
    order_id: str
    order_status: OrderStatus
    execution_price: Decimal
    sz: Decimal
    settle_amount: Decimal
    fee: Decimal
    pnl: Decimal
    finish_time: datetime
    market_order_id: str
    extra: dict
```

## 使用示例

### 添加执行器到引擎

```python
from leek_core.engine import SimpleEngine
from leek_core.models import LeekComponentConfig

engine = SimpleEngine(...)

# 添加Gate执行器
engine.add_executor(LeekComponentConfig(
    instance_id="gate-executor",
    name="Gate.io执行器",
    cls="leek_core.executor.GateRestExecutor",
    config={
        "api_key": "your_api_key",
        "secret_key": "your_secret_key",
        "testnet": False,
        "slippage_level": 5,
    }
))
```

### 自定义执行器

```python
from leek_core.executor import Executor
from leek_core.models import Order, OrderUpdateMessage, OrderStatus, Field, FieldType

class MyCustomExecutor(Executor):
    """自定义执行器"""
    
    display_name = "自定义执行器"
    init_params = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, required=True),
        Field(name="endpoint", label="API地址", type=FieldType.STRING, default="https://api.example.com"),
    ]
    
    def __init__(self, api_key: str, endpoint: str = "https://api.example.com", **kwargs):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = None
    
    def on_start(self):
        """初始化API客户端"""
        self.client = MyAPIClient(self.api_key, self.endpoint)
    
    def on_stop(self):
        """清理资源"""
        if self.client:
            self.client.close()
    
    def send_order(self, orders: Order | list[Order]):
        """发送订单"""
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            try:
                result = self.client.place_order(
                    symbol=order.symbol,
                    side="buy" if order.side == PositionSide.LONG else "sell",
                    quantity=float(order.sz),
                    price=float(order.order_price) if order.order_price else None,
                )
                
                # 更新订单状态
                self._trade_callback(OrderUpdateMessage(
                    order_id=order.order_id,
                    order_status=OrderStatus.SUBMITTED,
                    market_order_id=result["order_id"],
                    ...
                ))
            except Exception as e:
                self._trade_callback(OrderUpdateMessage(
                    order_id=order.order_id,
                    order_status=OrderStatus.ERROR,
                    ...
                ))
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """取消订单"""
        self.client.cancel_order(order_id, symbol)
```

### WebSocket 执行器实现

```python
from leek_core.executor import WebSocketExecutor
import json

class MyWSExecutor(WebSocketExecutor):
    """WebSocket执行器示例"""
    
    display_name = "My WS Executor"
    
    def __init__(self, ws_url: str, api_key: str, **kwargs):
        super().__init__(ws_url, **kwargs)
        self.api_key = api_key
    
    async def on_open(self):
        """连接建立后认证"""
        auth_msg = json.dumps({
            "action": "auth",
            "api_key": self.api_key,
        })
        await self.send(auth_msg)
    
    async def on_message(self, msg):
        """处理服务器消息"""
        data = json.loads(msg)
        
        if data.get("type") == "order_update":
            self._trade_callback(OrderUpdateMessage(
                order_id=data["client_order_id"],
                order_status=self._parse_status(data["status"]),
                execution_price=Decimal(data.get("price", 0)),
                sz=Decimal(data.get("filled_qty", 0)),
                ...
            ))
    
    def send_order(self, orders: Order | list[Order]):
        """通过WebSocket发送订单"""
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            msg = json.dumps({
                "action": "place_order",
                "client_order_id": order.order_id,
                "symbol": order.symbol,
                "side": "buy" if order.side == PositionSide.LONG else "sell",
                "quantity": str(order.sz),
            })
            self.async_send(msg)
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        msg = json.dumps({
            "action": "cancel_order",
            "order_id": order_id,
            "symbol": symbol,
        })
        self.async_send(msg)
```

## 配置说明

### 执行器通用配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `instance_id` | str | 执行器实例ID |
| `name` | str | 执行器名称 |
| `cls` | str/class | 执行器类 |
| `config` | dict | 执行器参数 |

### WebSocket 执行器配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ws_url` | str | - | WebSocket连接URL |
| `heartbeat_interval` | float | -1 | 心跳间隔秒数，-1不发送 |
| `reconnect_interval` | float | 5 | 重连间隔秒数 |
| `max_retries` | int | 5 | 最大重连次数 |

## 事件流程

### 订单创建流程

```text
ExecutorManager.handle_order(context)
        │
        ▼
选择合适的ExecutorContext
        │
        ▼
ExecutorContext.send_order(orders)
        │
        ├──► EventBus.publish(ORDER_CREATED)
        │
        └──► Executor.send_order(orders)
                    │
                    ▼
              交易所API/WebSocket
                    │
                    ▼
              订单提交成功
                    │
                    ▼
              EventBus.publish(ORDER_UPDATED, SUBMITTED)
```

### 订单回调流程

```text
交易所推送/轮询
        │
        ▼
Executor._trade_callback(msg)
        │
        ▼
ExecutorContext._trade_callback(msg)
        │
        ├──► 更新本地Order对象
        │
        └──► EventBus.publish(ORDER_UPDATED)
                    │
                    ▼
              Engine.on_order_update()
```

## 最佳实践

### 1. 错误处理

```python
def send_order(self, orders):
    for order in orders:
        try:
            # 下单逻辑
            result = self.client.place_order(...)
        except NetworkError as e:
            logger.warning(f"网络错误，将重试: {e}")
            raise  # 让上层重试
        except APIError as e:
            logger.error(f"API错误: {e}")
            # 标记订单失败
            self._trade_callback(OrderUpdateMessage(
                order_id=order.order_id,
                order_status=OrderStatus.ERROR,
            ))
```

### 2. 数量精度处理

```python
def _check_sz(self, order: Order) -> Decimal:
    """检查并调整订单数量精度"""
    # 获取交易对精度信息
    info = self.get_symbol_info(order.symbol)
    step_size = Decimal(info["step_size"])
    
    # 调整到有效精度
    quantity = (order.sz // step_size) * step_size
    
    # 检查最小数量
    if quantity < Decimal(info["min_qty"]):
        raise ValueError(f"数量小于最小值: {info['min_qty']}")
    
    return quantity
```

### 3. 连接管理

```python
class MyWSExecutor(WebSocketExecutor):
    async def on_error(self, error):
        """错误处理"""
        logger.error(f"WebSocket错误: {error}")
        # 可以在这里发送告警
        
    async def on_close(self):
        """连接关闭"""
        logger.warning("WebSocket连接关闭")
        # 清理未完成订单的状态
```

## 相关模块

- [引擎架构](20-engine.md) - 执行器在引擎中的使用
- [事件总线](22-event-bus.md) - 订单事件
- [仓位管理](03-position.md) - 订单与仓位关系
