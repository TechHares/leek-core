# 91 组件开发指南

## 概述

本指南介绍如何开发 Leek Core 的自定义组件，包括策略、指标、数据源、执行器、风控等。所有组件都继承自 `LeekComponent` 基类，遵循统一的接口规范。

## 组件基类

### `LeekComponent`

```python
class LeekComponent:
    """组件基类"""
    
    # 类属性
    display_name: str = None           # 组件显示名称
    init_params: List[Field] = []      # 初始化参数定义
    
    def on_start(self):
        """
        组件启动回调
        - 初始化资源（如网络连接、文件句柄）
        - 启动后台任务
        """
        ...
    
    def on_stop(self):
        """
        组件停止回调
        - 释放资源
        - 停止后台任务
        - 保存状态
        """
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """
        序列化组件状态
        - 返回可 JSON 序列化的字典
        - 用于状态持久化和恢复
        """
        return {}
    
    def load_state(self, state: Dict[str, Any]):
        """
        加载组件状态
        - 从字典恢复组件状态
        - 用于重启后状态恢复
        """
        ...
    
    def on_event(self, event: Event):
        """
        处理事件
        - 响应事件总线推送的事件
        - 子类可选重写
        """
        ...
```

## 参数定义

### `Field` 类

用于定义组件的初始化参数：

```python
from leek_core.models import Field, FieldType

init_params = [
    Field(
        name="period",           # 参数名（与__init__参数对应）
        label="周期",            # 显示标签
        type=FieldType.INT,      # 参数类型
        default=20,              # 默认值
        required=True,           # 是否必填
        min=1,                   # 最小值
        max=1000,                # 最大值
        description="计算周期"    # 描述
    ),
    Field(
        name="api_key",
        label="API Key",
        type=FieldType.STRING,
        required=True,
    ),
    Field(
        name="secret",
        label="密钥",
        type=FieldType.PASSWORD,  # 密码类型（UI隐藏显示）
        required=True,
    ),
    Field(
        name="mode",
        label="模式",
        type=FieldType.SELECT,    # 下拉选择
        options=[
            {"value": "fast", "label": "快速"},
            {"value": "normal", "label": "普通"},
            {"value": "safe", "label": "安全"},
        ],
        default="normal",
    ),
    Field(
        name="enabled",
        label="启用",
        type=FieldType.BOOLEAN,
        default=True,
    ),
]
```

### `FieldType` 枚举

| 类型 | 说明 | UI控件 |
|------|------|--------|
| `STRING` | 字符串 | 文本框 |
| `INT` | 整数 | 数字输入框 |
| `FLOAT` | 浮点数 | 数字输入框 |
| `BOOLEAN` | 布尔值 | 开关 |
| `PASSWORD` | 密码 | 密码框 |
| `SELECT` | 单选 | 下拉框 |
| `MULTISELECT` | 多选 | 多选下拉框 |

## 策略组件

### 基础策略

```python
from leek_core.strategy import Strategy
from leek_core.models import Data, Position, PositionSide, Field, FieldType
from typing import List, Set
from decimal import Decimal

class MyStrategy(Strategy):
    """自定义策略"""
    
    display_name = "我的策略"
    
    # 参数定义
    init_params: List[Field] = [
        Field(name="threshold", label="阈值", type=FieldType.FLOAT, default=0.02),
    ]
    
    # 策略接受的数据类型
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    
    def __init__(self, threshold: float = 0.02):
        super().__init__()
        self.threshold = Decimal(str(threshold))
        self._last_price = None
    
    def on_data(self, data: Data):
        """
        处理数据
        - 每次收到数据时调用
        - 更新内部状态
        """
        if data.data_type == DataType.KLINE:
            self._last_price = data.close
    
    def should_open(self) -> PositionSide | None:
        """
        判断是否开仓
        返回:
            PositionSide.LONG: 开多
            PositionSide.SHORT: 开空
            None: 不开仓
        """
        if self._last_price is None:
            return None
        
        # 开仓逻辑
        if some_condition:
            return PositionSide.LONG
        return None
    
    def close(self, position: Position) -> bool | Decimal | None:
        """
        判断是否平仓
        参数:
            position: 当前持仓
        返回:
            True: 全部平仓
            Decimal: 部分平仓比例
            None/False: 不平仓
        """
        if some_close_condition:
            return True
        return None
    
    def after_risk_control(self):
        """
        风控后回调
        - 在子策略风控检查后调用
        """
        pass
    
    def get_state(self) -> dict:
        """保存策略状态"""
        return {
            "last_price": str(self._last_price) if self._last_price else None,
        }
    
    def load_state(self, state: dict):
        """恢复策略状态"""
        if state.get("last_price"):
            self._last_price = Decimal(state["last_price"])
```

### CTA 策略

```python
from leek_core.strategy import CTAStrategy
from leek_core.models import KLine, PositionSide

class MyCTAStrategy(CTAStrategy):
    """CTA策略（K线专用）"""
    
    display_name = "我的CTA策略"
    
    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self._prices = []
    
    def on_kline(self, kline: KLine):
        """
        处理K线数据
        - CTAStrategy 自动将 KLINE 数据转发到此方法
        """
        self._prices.append(float(kline.close))
        if len(self._prices) > self.slow:
            self._prices = self._prices[-self.slow:]
    
    def should_open(self) -> PositionSide | None:
        if len(self._prices) < self.slow:
            return None
        
        fast_ma = sum(self._prices[-self.fast:]) / self.fast
        slow_ma = sum(self._prices) / self.slow
        
        if fast_ma > slow_ma:
            return PositionSide.LONG
        elif fast_ma < slow_ma:
            return PositionSide.SHORT
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        """
        CTA策略的简化平仓接口
        参数:
            position_side: 当前持仓方向
        返回:
            True: 平仓
            False: 不平仓
        """
        if len(self._prices) < self.slow:
            return False
        
        fast_ma = sum(self._prices[-self.fast:]) / self.fast
        slow_ma = sum(self._prices) / self.slow
        
        if position_side == PositionSide.LONG:
            return fast_ma < slow_ma
        else:
            return fast_ma > slow_ma
```

## 指标组件

```python
from leek_core.indicators import T
from leek_core.models import KLine
from decimal import Decimal
from typing import Optional

class MyIndicator(T):
    """自定义指标"""
    
    def __init__(self, period: int = 14, max_cache: int = 10):
        super().__init__(max_cache=max_cache)
        self.period = period
        self._buffer = []
    
    def update(self, data: KLine) -> Optional[Decimal]:
        """
        更新指标
        参数:
            data: K线数据
        返回:
            指标值，数据不足时返回 None
        """
        self._buffer.append(data.close)
        
        if len(self._buffer) > self.period:
            self._buffer = self._buffer[-self.period:]
        
        if len(self._buffer) < self.period:
            return None
        
        # 计算指标值
        result = sum(self._buffer) / len(self._buffer)
        
        # 缓存历史值（仅在K线完成时）
        if data.is_finished:
            self.cache.append(result)
            if len(self.cache) > self.max_cache:
                self.cache = self.cache[-self.max_cache:]
        
        return result
```

## 数据源组件

```python
from leek_core.data import DataSource
from leek_core.models import KLine, TimeFrame, Field, FieldType
import pandas as pd
from typing import List, Dict, Any

class MyDataSource(DataSource):
    """自定义数据源"""
    
    display_name = "我的数据源"
    init_params = [
        Field(name="host", label="主机", type=FieldType.STRING, required=True),
        Field(name="port", label="端口", type=FieldType.INT, default=8080),
    ]
    
    def __init__(self, host: str, port: int = 8080):
        super().__init__()
        self.host = host
        self.port = port
        self._client = None
    
    def on_start(self):
        """连接数据源"""
        self._client = MyClient(self.host, self.port)
        self._client.connect()
    
    def on_stop(self):
        """断开连接"""
        if self._client:
            self._client.close()
    
    async def get_klines(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        start_time: int,
        end_time: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取历史K线
        返回包含 open, high, low, close, volume, timestamp 的 DataFrame
        """
        data = await self._client.fetch_klines(
            symbol=symbol,
            interval=timeframe.value,
            start=start_time,
            end=end_time,
        )
        return pd.DataFrame(data)
    
    async def subscribe_klines(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        callback
    ) -> bool:
        """订阅实时K线"""
        await self._client.subscribe(
            f"{symbol}_{timeframe.value}",
            callback
        )
        return True
    
    async def unsubscribe_klines(
        self, 
        symbol: str, 
        timeframe: TimeFrame
    ) -> bool:
        """取消订阅"""
        await self._client.unsubscribe(f"{symbol}_{timeframe.value}")
        return True
```

## 执行器组件

```python
from leek_core.executor import Executor
from leek_core.models import Order, OrderUpdateMessage, OrderStatus, Field, FieldType
from typing import List

class MyExecutor(Executor):
    """自定义执行器"""
    
    display_name = "我的执行器"
    init_params = [
        Field(name="api_key", label="API Key", type=FieldType.STRING, required=True),
        Field(name="api_secret", label="API Secret", type=FieldType.PASSWORD, required=True),
    ]
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
    
    def on_start(self):
        """初始化API客户端"""
        self._client = MyTradingClient(self.api_key, self.api_secret)
    
    def on_stop(self):
        """清理资源"""
        if self._client:
            self._client.close()
    
    def send_order(self, orders: Order | List[Order]):
        """
        发送订单
        - 调用交易所API下单
        - 通过 _trade_callback 回调订单状态
        """
        if isinstance(orders, Order):
            orders = [orders]
        
        for order in orders:
            try:
                result = self._client.place_order(
                    symbol=f"{order.symbol}_{order.quote_currency}",
                    side="buy" if order.side == PositionSide.LONG else "sell",
                    quantity=float(order.sz),
                    price=float(order.order_price) if order.order_price else None,
                )
                
                # 回调订单已提交
                self._trade_callback(OrderUpdateMessage(
                    order_id=order.order_id,
                    order_status=OrderStatus.SUBMITTED,
                    market_order_id=result["order_id"],
                    execution_price=None,
                    sz=None,
                    settle_amount=None,
                    fee=None,
                    pnl=None,
                    finish_time=None,
                ))
                
            except Exception as e:
                # 回调订单失败
                self._trade_callback(OrderUpdateMessage(
                    order_id=order.order_id,
                    order_status=OrderStatus.ERROR,
                ))
    
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """取消订单"""
        self._client.cancel_order(order_id, symbol)
```

## 子策略组件

```python
from leek_core.sub_strategy import SubStrategy
from leek_core.models import Data, Position, DataType, PositionSide, Field, FieldType
from decimal import Decimal
from typing import Set

class MySubStrategy(SubStrategy):
    """自定义子策略（风控）"""
    
    display_name = "我的风控"
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    
    init_params = [
        Field(name="max_loss", label="最大亏损", type=FieldType.FLOAT, default=0.05),
    ]
    
    def __init__(self, max_loss: float = 0.05):
        self.max_loss = Decimal(str(max_loss))
    
    def evaluate(self, data: Data, position: Position) -> bool:
        """
        评估仓位是否应该继续持有
        参数:
            data: 当前市场数据
            position: 当前持仓
        返回:
            True: 继续持有
            False: 应该平仓
        """
        if data.data_type != DataType.KLINE:
            return True
        
        # 计算盈亏比例
        if position.side == PositionSide.LONG:
            pnl_ratio = (data.close - position.cost_price) / position.cost_price
        else:
            pnl_ratio = (position.cost_price - data.close) / position.cost_price
        
        # 亏损超过阈值，平仓
        if pnl_ratio < -self.max_loss:
            return False
        
        return True
```

## 告警组件

```python
from leek_core.alarm import AlarmSender, AlarmLevel
from leek_core.models import Field, FieldType
import requests

class MyAlarmSender(AlarmSender):
    """自定义告警发送器"""
    
    display_name = "我的告警"
    init_params = [
        Field(name="webhook_url", label="Webhook URL", type=FieldType.STRING, required=True),
    ]
    
    def __init__(self, webhook_url: str):
        super().__init__()
        self.webhook_url = webhook_url
    
    def send(self, level: str, message: str, **kwargs):
        """
        发送告警
        参数:
            level: 告警级别 (AlarmLevel.INFO/WARNING/ERROR/CRITICAL)
            message: 告警消息
        """
        if level not in [AlarmLevel.ERROR, AlarmLevel.CRITICAL]:
            return
        
        payload = {
            "level": level,
            "message": message,
            "extra": kwargs,
        }
        
        try:
            requests.post(self.webhook_url, json=payload)
        except Exception as e:
            print(f"告警发送失败: {e}")
```

## 最佳实践

### 1. 参数验证

```python
def __init__(self, period: int = 20):
    if period < 1:
        raise ValueError("period 必须大于 0")
    self.period = period
```

### 2. 资源管理

```python
def on_start(self):
    self._connection = create_connection()

def on_stop(self):
    if self._connection:
        try:
            self._connection.close()
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")
        finally:
            self._connection = None
```

### 3. 状态持久化

```python
def get_state(self) -> dict:
    return {
        "counter": self._counter,
        "last_value": str(self._last_value),
        "history": [str(v) for v in self._history],
    }

def load_state(self, state: dict):
    self._counter = state.get("counter", 0)
    if state.get("last_value"):
        self._last_value = Decimal(state["last_value"])
    self._history = [Decimal(v) for v in state.get("history", [])]
```

### 4. 日志记录

```python
from leek_core.utils import get_logger

logger = get_logger(__name__)

class MyComponent(LeekComponent):
    def on_start(self):
        logger.info(f"组件启动: {self.display_name}")
    
    def on_stop(self):
        logger.info(f"组件停止: {self.display_name}")
```

### 5. 异常处理

```python
def send_order(self, order):
    try:
        result = self._client.place_order(...)
    except NetworkError as e:
        logger.warning(f"网络错误: {e}")
        raise  # 让上层重试
    except APIError as e:
        logger.error(f"API错误: {e}")
        self._trade_callback(OrderUpdateMessage(
            order_id=order.order_id,
            order_status=OrderStatus.ERROR,
        ))
```

## 相关文档

- [整体架构](90-architecture.md) - 架构设计
- [策略模块](01-strategy.md) - 策略开发详情
- [指标模块](12-indicators.md) - 指标开发详情
- [执行器](23-executor.md) - 执行器开发详情
