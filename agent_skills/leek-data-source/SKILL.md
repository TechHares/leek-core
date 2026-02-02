---
name: leek-data-source
description: 配置和开发 leek 数据源。继承 DataSource/WebSocketDataSource 基类，实现数据获取和订阅。支持 Binance/OKX/Gate WebSocket 和 ClickHouse 数据库。当用户要配置数据源、开发自定义数据源、获取K线数据时使用。Use when configuring data sources, developing custom data feeds, or fetching market data.
---

# Leek 数据源

## 内置数据源

| 数据源 | 类型 | 用途 |
|-------|------|------|
| `BinanceDataSource` | WebSocket | Binance 实时K线 |
| `OkxDataSource` | WebSocket | OKX 实时K线 |
| `GateDataSource` | WebSocket | Gate.io 实时K线 |
| `ClickHouseKlineDataSource` | 数据库 | ClickHouse 历史数据（回测） |
| `RedisClickHouseDataSource` | 缓存+数据库 | Redis缓存 + ClickHouse（回测） |

## 基类

### DataSource

所有数据源的抽象基类：

```python
from leek_core.data import DataSource

class DataSource(LeekComponent, ABC):
    # 类属性
    supported_data_type: DataType = DataType.KLINE
    supported_asset_type: AssetType = AssetType.STOCK
    just_backtest: bool = False  # 是否仅用于回测
    display_name: str = "数据源名称"
    
    # 核心方法
    def parse_row_key(self, **kwargs) -> List[tuple]: ...
    def subscribe(self, row_key: str): ...
    def unsubscribe(self, row_key: str): ...
    def get_history_data(self, row_key, start_time, end_time, limit) -> Iterator: ...
    def get_supported_parameters(self) -> List[Field]: ...
    def send_data(self, data: Data): ...  # 发送数据到回调
```

### WebSocketDataSource

WebSocket 数据源基类，管理连接生命周期：

```python
from leek_core.data import WebSocketDataSource

class WebSocketDataSource(DataSource, ABC):
    init_params = [
        Field(name="ws_url", type=FieldType.STRING, required=True),
    ]
    
    def __init__(self, ws_url: str, ping_interval=25, ping_timeout=10): ...
    
    # 生命周期
    def on_start(self): ...      # 启动连接
    def on_stop(self): ...       # 断开连接
    def on_connect(self): ...    # 连接成功回调
    def on_disconnect(self): ... # 断开前回调
    
    # 消息处理
    async def on_message(self, message: str): ...  # 必须实现
    async def on_connection_closed(self, exception): ...
    
    # 发送消息
    async def send(self, message: str) -> bool: ...
    def async_send(self, message: str) -> bool: ...  # 同步上下文调用
```

## 开发自定义数据源

### 数据库数据源

```python
from typing import Iterator, List
from datetime import datetime
from leek_core.data import DataSource
from leek_core.models import Field, FieldType, KLine, AssetType

class MyDatabaseSource(DataSource):
    display_name = "我的数据源"
    just_backtest = True  # 仅用于回测
    supported_asset_type = AssetType.CRYPTO
    
    init_params = [
        Field(name="connection_string", label="连接字符串", 
              type=FieldType.STRING, required=True),
    ]
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.conn = None
    
    def on_start(self):
        """连接数据库"""
        self.conn = connect(self.connection_string)
    
    def on_stop(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
    
    def parse_row_key(self, **kwargs) -> List[tuple]:
        """解析参数为数据键"""
        symbol = kwargs.get("symbol")
        timeframe = kwargs.get("timeframe")
        return [(symbol, timeframe)]
    
    def get_supported_parameters(self) -> List[Field]:
        """返回支持的查询参数"""
        return [
            Field(name="symbol", label="交易对", type=FieldType.STRING),
            Field(name="timeframe", label="周期", type=FieldType.STRING),
        ]
    
    def get_history_data(self, row_key: str, 
                         start_time: datetime = None,
                         end_time: datetime = None,
                         limit: int = None, **kwargs) -> Iterator[KLine]:
        """获取历史数据"""
        symbol, timeframe = row_key.split("_")
        
        query = f"SELECT * FROM klines WHERE symbol='{symbol}'"
        if start_time:
            query += f" AND time >= '{start_time}'"
        if end_time:
            query += f" AND time <= '{end_time}'"
        if limit:
            query += f" LIMIT {limit}"
        
        for row in self.conn.execute(query):
            yield KLine(
                symbol=row['symbol'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                is_finished=True,
            )
```

### WebSocket 数据源

```python
import json
from leek_core.data import WebSocketDataSource
from leek_core.models import Field, FieldType, KLine, TimeFrame

class MyWebSocketSource(WebSocketDataSource):
    display_name = "我的WebSocket数据源"
    
    init_params = [
        Field(name="ws_url", label="WebSocket地址", 
              type=FieldType.STRING, required=True),
        Field(name="api_key", label="API Key", 
              type=FieldType.PASSWORD, required=False),
    ]
    
    def __init__(self, ws_url: str, api_key: str = None):
        super().__init__(ws_url)
        self.api_key = api_key
        self.subscriptions = {}
    
    def on_connect(self):
        """连接成功后的初始化"""
        # 如果需要认证
        if self.api_key:
            auth_msg = json.dumps({"op": "auth", "key": self.api_key})
            self.async_send(auth_msg)
    
    def parse_row_key(self, **kwargs) -> List[tuple]:
        symbol = kwargs.get("symbol")
        timeframe = kwargs.get("timeframe")
        return [(f"{symbol}_{timeframe}",)]
    
    def get_supported_parameters(self) -> List[Field]:
        return [
            Field(name="symbol", label="交易对", type=FieldType.STRING),
            Field(name="timeframe", label="周期", type=FieldType.STRING),
        ]
    
    def subscribe(self, row_key: str):
        """订阅实时数据"""
        symbol, timeframe = row_key.split("_")
        sub_msg = json.dumps({
            "op": "subscribe",
            "channel": "kline",
            "symbol": symbol,
            "interval": timeframe
        })
        self.async_send(sub_msg)
        self.subscriptions[row_key] = True
    
    def unsubscribe(self, row_key: str):
        """取消订阅"""
        symbol, timeframe = row_key.split("_")
        unsub_msg = json.dumps({
            "op": "unsubscribe",
            "channel": "kline",
            "symbol": symbol,
            "interval": timeframe
        })
        self.async_send(unsub_msg)
        self.subscriptions.pop(row_key, None)
    
    async def on_message(self, message: str):
        """处理收到的消息"""
        try:
            data = json.loads(message)
            
            if data.get("channel") == "kline":
                kline = self._parse_kline(data)
                self.send_data(kline)  # 发送到回调
                
        except Exception as e:
            logger.error(f"解析消息失败: {e}")
    
    def _parse_kline(self, data: dict) -> KLine:
        """解析K线数据"""
        return KLine(
            symbol=data["symbol"],
            market="my_exchange",
            timeframe=TimeFrame.from_string(data["interval"]),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
            start_time=data["start_time"],
            end_time=data["end_time"],
            is_finished=data.get("is_finished", False),
        )
    
    def get_history_data(self, row_key, start_time=None, 
                         end_time=None, limit=None, **kwargs):
        """WebSocket数据源通常不支持历史数据"""
        raise NotImplementedError("此数据源不支持历史数据获取")
    
    async def on_connection_closed(self, exception):
        """连接关闭处理"""
        logger.warning(f"连接关闭: {exception}")
        # 可以在这里实现重连逻辑
```

## 数据模型

### KLine

```python
from leek_core.models import KLine, TimeFrame
from decimal import Decimal

kline = KLine(
    symbol="BTCUSDT",
    market="binance",
    timeframe=TimeFrame.M1,
    open=Decimal("50000"),
    high=Decimal("50100"),
    low=Decimal("49900"),
    close=Decimal("50050"),
    volume=Decimal("100"),
    amount=Decimal("5000000"),
    start_time=1704067200000,  # 毫秒时间戳
    end_time=1704067260000,
    is_finished=True,
    quote_currency="USDT",
    ins_type="SPOT",
)
```

### TimeFrame

```python
from leek_core.models import TimeFrame

# 常用周期
TimeFrame.S1   # 1秒
TimeFrame.M1   # 1分钟
TimeFrame.M5   # 5分钟
TimeFrame.M15  # 15分钟
TimeFrame.H1   # 1小时
TimeFrame.H4   # 4小时
TimeFrame.D1   # 日线
TimeFrame.W1   # 周线

# 从字符串创建
tf = TimeFrame.from_string("1m")

# 获取毫秒数
ms = TimeFrame.M1.milliseconds  # 60000
```

## row_key 格式

row_key 是数据的唯一标识，格式通常为：

```
{symbol}_{quote_currency}_{ins_type}_{timeframe}
```

示例：`BTCUSDT_USDT_SPOT_1m`

## 使用数据源

```python
# 创建数据源
source = MyWebSocketSource(
    ws_url="wss://stream.example.com",
    api_key="your_api_key"
)

# 设置回调
def on_data(data):
    print(f"收到数据: {data}")

source.callback = on_data

# 启动
source.on_start()

# 订阅
source.subscribe("BTCUSDT_USDT_SPOT_1m")

# ... 运行中 ...

# 取消订阅
source.unsubscribe("BTCUSDT_USDT_SPOT_1m")

# 停止
source.on_stop()
```

## 最佳实践

1. **错误处理**：`on_message` 中捕获所有异常，避免中断监听
2. **重连机制**：在 `on_connection_closed` 中实现重连逻辑
3. **资源清理**：在 `on_stop` 中确保释放所有资源
4. **心跳保活**：WebSocket 使用内置 ping/pong，可调整 `ping_interval`
5. **线程安全**：使用 `async_send` 从同步上下文发送消息

## 详细参考

完整文档见 [reference/11-data-sources.md](reference/11-data-sources.md)
