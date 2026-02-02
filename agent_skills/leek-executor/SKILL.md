---
name: leek-executor
description: 开发 leek 交易执行器。继承 Executor/WebSocketExecutor 基类，实现 send_order/cancel_order 方法执行交易。支持 REST API 和 WebSocket 两种模式。当用户要开发交易执行器、对接交易所下单接口时使用。Use when developing trade executors, connecting to exchange APIs, or implementing order execution.
---

# Leek 交易执行器

## 内置执行器

| 执行器 | 类型 | 用途 |
|-------|------|------|
| `BinanceRestExecutor` | REST | Binance REST API 下单 |
| `OkxWebSocketExecutor` | WebSocket | OKX WebSocket 下单 |
| `OkxRestExecutor` | REST | OKX REST API 下单 |
| `GateRestExecutor` | REST | Gate.io REST API 下单 |
| `BacktestExecutor` | 模拟 | 回测模拟成交 |

## 基类

### Executor

所有执行器的抽象基类：

```python
from leek_core.executor import Executor

class Executor(LeekComponent, ABC):
    just_backtest: bool = False  # 是否仅用于回测
    
    def __init__(self):
        self.instance_id = None
        self.callback = None  # 订单状态回调
    
    # 核心方法
    def check_order(self, order: Order) -> bool: ...  # 检查订单可执行性
    def send_order(self, order: Order | List[Order]): ...  # 下单（抽象）
    def cancel_order(self, order_id: str, symbol: str, **kwargs): ...  # 撤单（抽象）
    def _trade_callback(self, order_update_message: OrderUpdateMessage): ...  # 交易回调
```

### WebSocketExecutor

WebSocket 执行器基类，支持自动重连、心跳：

```python
from leek_core.executor import WebSocketExecutor, WSStatus

class WebSocketExecutor(Executor, ABC):
    init_params = [
        Field(name="ws_url", type=FieldType.STRING, required=True),
        Field(name="heartbeat_interval", type=FieldType.FLOAT, default=-1),
        Field(name="reconnect_interval", type=FieldType.FLOAT, default=5),
        Field(name="max_retries", type=FieldType.INT, default=5),
    ]
    
    def __init__(self, ws_url, heartbeat_interval=-1, 
                 reconnect_interval=5, max_retries=5):
        self._status: WSStatus  # 连接状态
    
    # 生命周期
    def on_start(self): ...
    def on_stop(self): ...
    
    # 回调方法（子类实现）
    async def on_message(self, msg): ...  # 收到消息
    async def on_open(self): ...          # 连接建立
    async def on_close(self): ...         # 连接关闭
    async def on_error(self, error): ...  # 错误处理
    
    # 发送消息
    async def send(self, msg): ...
    def async_send(self, message: str) -> bool: ...  # 同步上下文调用
```

### WSStatus 连接状态

```python
class WSStatus(Enum):
    INIT = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    DISCONNECTED = auto()
    RECONNECTING = auto()
```

## 数据模型

### Order 订单

```python
from leek_core.models import Order, OrderType, OrderSide, OrderStatus

order = Order(
    order_id="order_123",
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=Decimal("50000"),
    amount=Decimal("0.1"),
    status=OrderStatus.PENDING,
)
```

### OrderUpdateMessage 订单更新消息

```python
from leek_core.models import OrderUpdateMessage

update = OrderUpdateMessage(
    order_id="order_123",
    symbol="BTCUSDT",
    status=OrderStatus.FILLED,
    filled_amount=Decimal("0.1"),
    filled_price=Decimal("50000"),
    fee=Decimal("0.0001"),
    timestamp=1704067200000,
)
```

## 开发自定义执行器

### REST 执行器

```python
import requests
from typing import List
from leek_core.executor import Executor
from leek_core.models import (
    Order, OrderUpdateMessage, OrderStatus,
    Field, FieldType
)

class MyRestExecutor(Executor):
    display_name = "我的REST执行器"
    
    init_params = [
        Field(name="api_key", label="API Key", 
              type=FieldType.PASSWORD, required=True),
        Field(name="api_secret", label="API Secret", 
              type=FieldType.PASSWORD, required=True),
        Field(name="base_url", label="API地址", 
              type=FieldType.STRING, default="https://api.example.com"),
    ]
    
    def __init__(self, api_key: str, api_secret: str, 
                 base_url: str = "https://api.example.com"):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.pending_orders = {}
    
    def on_start(self):
        """启动时可以初始化连接池等"""
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": self.api_key})
    
    def on_stop(self):
        """停止时清理资源"""
        self.session.close()
    
    def check_order(self, order: Order) -> bool:
        """检查订单有效性"""
        if order.amount <= 0:
            return False
        return True
    
    def send_order(self, order: Order | List[Order]):
        """发送订单"""
        orders = [order] if isinstance(order, Order) else order
        
        for o in orders:
            try:
                resp = self.session.post(
                    f"{self.base_url}/order",
                    json={
                        "symbol": o.symbol,
                        "side": o.side.value,
                        "type": o.order_type.value,
                        "price": str(o.price),
                        "amount": str(o.amount),
                    },
                    headers=self._sign_request(o)
                )
                
                data = resp.json()
                if data.get("success"):
                    o.order_id = data["order_id"]
                    self.pending_orders[o.order_id] = o
                    self._start_polling(o)
                else:
                    self._handle_error(o, data.get("error"))
                    
            except Exception as e:
                self._handle_error(o, str(e))
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """撤销订单"""
        try:
            resp = self.session.delete(
                f"{self.base_url}/order/{order_id}",
                params={"symbol": symbol}
            )
            
            if resp.json().get("success"):
                self._notify_cancelled(order_id)
                
        except Exception as e:
            logger.error(f"撤单失败: {e}")
    
    def _start_polling(self, order: Order):
        """启动订单状态轮询"""
        # 实现轮询逻辑，查询订单状态
        # 成交后调用 _trade_callback
        pass
    
    def _handle_error(self, order: Order, error: str):
        """处理错误"""
        update = OrderUpdateMessage(
            order_id=order.order_id,
            symbol=order.symbol,
            status=OrderStatus.REJECTED,
            error_message=error,
        )
        self._trade_callback(update)
    
    def _notify_cancelled(self, order_id: str):
        """通知订单已撤销"""
        order = self.pending_orders.pop(order_id, None)
        if order:
            update = OrderUpdateMessage(
                order_id=order_id,
                symbol=order.symbol,
                status=OrderStatus.CANCELLED,
            )
            self._trade_callback(update)
    
    def _sign_request(self, order: Order) -> dict:
        """签名请求"""
        # 实现签名逻辑
        return {"X-Signature": "..."}
```

### WebSocket 执行器

```python
import json
from leek_core.executor import WebSocketExecutor, WSStatus
from leek_core.models import (
    Order, OrderUpdateMessage, OrderStatus,
    Field, FieldType
)

class MyWebSocketExecutor(WebSocketExecutor):
    display_name = "我的WebSocket执行器"
    
    init_params = [
        Field(name="ws_url", label="WebSocket地址", 
              type=FieldType.STRING, required=True),
        Field(name="api_key", label="API Key", 
              type=FieldType.PASSWORD, required=True),
        Field(name="api_secret", label="API Secret", 
              type=FieldType.PASSWORD, required=True),
    ]
    
    def __init__(self, ws_url: str, api_key: str, api_secret: str):
        super().__init__(
            ws_url=ws_url,
            heartbeat_interval=30,
            reconnect_interval=5,
            max_retries=10
        )
        self.api_key = api_key
        self.api_secret = api_secret
        self.pending_orders = {}
    
    async def on_open(self):
        """连接建立后进行认证"""
        auth_msg = json.dumps({
            "op": "auth",
            "key": self.api_key,
            "sign": self._generate_signature(),
        })
        await self.send(auth_msg)
        
        # 订阅订单更新
        sub_msg = json.dumps({
            "op": "subscribe",
            "channel": "orders"
        })
        await self.send(sub_msg)
    
    async def on_message(self, msg: str):
        """处理收到的消息"""
        try:
            data = json.loads(msg)
            
            if data.get("channel") == "orders":
                self._handle_order_update(data)
            elif data.get("op") == "auth":
                if data.get("success"):
                    logger.info("认证成功")
                else:
                    logger.error(f"认证失败: {data.get('error')}")
                    
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
    
    def send_order(self, order: Order | List[Order]):
        """发送订单"""
        orders = [order] if isinstance(order, Order) else order
        
        for o in orders:
            msg = json.dumps({
                "op": "order",
                "data": {
                    "client_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "type": o.order_type.value,
                    "price": str(o.price),
                    "amount": str(o.amount),
                }
            })
            
            self.pending_orders[o.order_id] = o
            self.async_send(msg)
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """撤销订单"""
        msg = json.dumps({
            "op": "cancel",
            "data": {
                "order_id": order_id,
                "symbol": symbol,
            }
        })
        self.async_send(msg)
    
    def _handle_order_update(self, data: dict):
        """处理订单更新"""
        order_data = data.get("data", {})
        
        update = OrderUpdateMessage(
            order_id=order_data["order_id"],
            symbol=order_data["symbol"],
            status=OrderStatus(order_data["status"]),
            filled_amount=Decimal(order_data.get("filled_amount", "0")),
            filled_price=Decimal(order_data.get("filled_price", "0")),
            fee=Decimal(order_data.get("fee", "0")),
        )
        
        self._trade_callback(update)
        
        # 清理已完成订单
        if update.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            self.pending_orders.pop(update.order_id, None)
    
    def _generate_signature(self) -> str:
        """生成认证签名"""
        # 实现签名逻辑
        return "..."
    
    async def on_error(self, error):
        """错误处理"""
        logger.error(f"WebSocket错误: {error}")
    
    async def on_close(self):
        """连接关闭"""
        logger.warning("WebSocket连接关闭")
```

## 回测执行器

```python
from leek_core.executor import Executor
from leek_core.models import Order, OrderUpdateMessage, OrderStatus

class BacktestExecutor(Executor):
    """回测模拟执行器"""
    
    just_backtest = True
    display_name = "回测执行器"
    
    init_params = [
        Field(name="slippage", label="滑点", 
              type=FieldType.FLOAT, default=0.001),
        Field(name="commission", label="手续费率", 
              type=FieldType.FLOAT, default=0.001),
    ]
    
    def __init__(self, slippage=0.001, commission=0.001):
        super().__init__()
        self.slippage = Decimal(str(slippage))
        self.commission = Decimal(str(commission))
        self.current_price = {}  # 当前价格缓存
    
    def update_price(self, symbol: str, price: Decimal):
        """更新当前价格（由回测引擎调用）"""
        self.current_price[symbol] = price
    
    def send_order(self, order: Order | List[Order]):
        """模拟下单"""
        orders = [order] if isinstance(order, Order) else order
        
        for o in orders:
            # 计算成交价（加入滑点）
            price = self.current_price.get(o.symbol, o.price)
            if o.side == OrderSide.BUY:
                fill_price = price * (1 + self.slippage)
            else:
                fill_price = price * (1 - self.slippage)
            
            # 计算手续费
            fee = o.amount * fill_price * self.commission
            
            # 立即成交
            update = OrderUpdateMessage(
                order_id=o.order_id,
                symbol=o.symbol,
                status=OrderStatus.FILLED,
                filled_amount=o.amount,
                filled_price=fill_price,
                fee=fee,
            )
            
            self._trade_callback(update)
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """回测中撤单立即成功"""
        update = OrderUpdateMessage(
            order_id=order_id,
            symbol=symbol,
            status=OrderStatus.CANCELLED,
        )
        self._trade_callback(update)
```

## 使用执行器

```python
# 创建执行器
executor = MyRestExecutor(
    api_key="your_key",
    api_secret="your_secret"
)

# 设置回调
def on_order_update(update: OrderUpdateMessage):
    print(f"订单更新: {update.order_id} -> {update.status}")

executor.callback = on_order_update

# 启动
executor.on_start()

# 下单
order = Order(
    order_id="order_001",
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=Decimal("50000"),
    amount=Decimal("0.1"),
)

if executor.check_order(order):
    executor.send_order(order)

# 撤单
executor.cancel_order("order_001", "BTCUSDT")

# 停止
executor.on_stop()
```

## 最佳实践

1. **签名安全**：API 密钥不要硬编码，使用环境变量或配置文件
2. **错误处理**：所有网络请求都要有异常处理
3. **订单跟踪**：维护 pending_orders 字典跟踪未完成订单
4. **重连机制**：WebSocket 使用内置重连，可配置重试次数
5. **回调及时**：订单状态变化及时调用 `_trade_callback`
6. **资源清理**：`on_stop` 中确保关闭所有连接和线程
