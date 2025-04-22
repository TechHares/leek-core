# 11 数据源

## 概述

数据源模块是交易平台的核心组件之一，负责从各种来源获取金融市场数据，包括但不限于K线数据、逐笔成交、订单簿和市场指标。该模块提供了统一的抽象接口，使系统能够以一致的方式处理来自不同来源的数据，无论是实时API、WebSocket流、历史数据文件还是数据库。

## 核心组件

### `DataSource` 抽象基类

`DataSource` 是所有数据源实现的抽象基类，定义了统一的接口方法：

```python
class DataSource(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """初始化数据源"""
        
    async def connect(self) -> bool:
        """连接到数据源"""
        
    async def disconnect(self) -> bool:
        """断开与数据源的连接"""
        
    async def get_klines(self, symbol: str, timeframe: TimeFrame, ...) -> pd.DataFrame:
        """获取历史K线数据"""
        
    async def subscribe_klines(self, symbol: str, timeframe: TimeFrame, callback) -> bool:
        """订阅实时K线更新"""
        
    async def unsubscribe_klines(self, symbol: str, timeframe: TimeFrame) -> bool:
        """取消订阅实时K线更新"""
        
    async def get_symbols(self) -> List[Dict[str, Any]]:
        """获取可用交易对列表"""
        
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """获取支持的时间周期"""
        
    def get_supported_data_types(self) -> List[DataType]:
        """获取支持的数据类型"""
        
    async def get_market_status(self, symbol: Optional[str] = None) -> MarketStatus:
        """获取市场状态"""
```

### `DataManager` 管理类

`DataManager` 管理多个数据源，提供统一的接口来添加、移除和获取数据源：

```python
class DataManager:
    def __init__(self):
        """初始化数据管理器"""
        
    def add_data_source(self, source: DataSource, make_default: bool = False) -> None:
        """添加数据源"""
        
    def get_data_source(self, name: Optional[str] = None) -> DataSource:
        """获取数据源，如果未指定名称则返回默认数据源"""
        
    def remove_data_source(self, name: str) -> None:
        """移除数据源"""
        
    async def connect_all(self) -> Dict[str, bool]:
        """连接所有数据源"""
        
    async def disconnect_all(self) -> Dict[str, bool]:
        """断开所有数据源连接"""
```

## 枚举类型

系统使用以下枚举类型来标准化数据源交互：

### `TimeFrame` - 时间周期

```python
class TimeFrame(Enum):
    """K线时间粒度枚举"""
    TICK = "tick"
    S1 = "1s"     # 1秒
    S5 = "5s"     # 5秒
    # ... 更多时间周期
    M1 = "1m"     # 1分钟
    M5 = "5m"     # 5分钟
    H1 = "1H"     # 1小时
    D1 = "1D"     # 日线
    W1 = "1W"     # 周线
    MON1 = "1M"   # 月线
```

### `DataType` - 数据类型

```python
class DataType(Enum):
    """处理的金融数据类型"""
    KLINE = "kline"               # K线数据
    TICK = "tick"                 # 逐笔成交数据
    ORDER_BOOK = "order_book"     # 订单簿数据
    TRADE = "trade"               # 成交数据
    # ... 更多数据类型
```

### `AssetType` - 资产类型

```python
class AssetType(Enum):
    """金融资产类型"""
    STOCK = "stock"           # 股票
    FUTURES = "futures"       # 期货
    CRYPTO = "crypto"         # 加密货币
    # ... 更多资产类型
```

### `MarketStatus` - 市场状态

```python
class MarketStatus(Enum):
    """市场交易状态"""
    OPEN = "open"                 # 开盘
    CLOSED = "closed"             # 收盘
    PRE_MARKET = "pre_market"     # 盘前
    # ... 更多市场状态
```

## 数据源类型与用法

### 1. REST API 数据源

REST API 数据源通过HTTP请求从交易所或数据提供商的API获取数据。

#### 特点
- 不维持长连接
- 适合获取历史数据
- 请求频率通常受限制

#### 使用示例

```python
# 创建交易所REST API数据源
exchange_source = ExchangeRestApiSource(
    name="binance",
    config={
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "base_url": "https://api.binance.com"
    }
)

# 连接（可能只是初始化客户端）
await exchange_source.connect()

# 获取历史K线数据
df = await exchange_source.get_klines(
    symbol="BTCUSDT",
    timeframe=TimeFrame.H1,
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 1, 31),
    limit=500
)

# 断开连接
await exchange_source.disconnect()
```

### 2. WebSocket 数据源

WebSocket数据源通过WebSocket连接接收实时数据推送。

#### 特点
- 维持长连接
- 适合接收实时数据
- 数据通过回调函数推送
- 连接断开需要重连机制

#### 使用示例

```python
# 创建WebSocket数据源
ws_source = ExchangeWebSocketSource(
    name="binance_ws",
    config={
        "ws_url": "wss://stream.binance.com:9443/ws"
    }
)

# 连接WebSocket
await ws_source.connect()

# 定义回调函数
async def on_kline_update(kline):
    print(f"收到新K线: {kline}")

# 订阅K线更新
await ws_source.subscribe_klines(
    symbol="BTCUSDT",
    timeframe=TimeFrame.M1,
    callback=on_kline_update
)

# 保持应用程序运行以接收数据

# 取消订阅
await ws_source.unsubscribe_klines("BTCUSDT", TimeFrame.M1)

# 断开连接
await ws_source.disconnect()
```

### 3. 文件数据源

文件数据源从本地文件系统读取数据，如CSV、Parquet或自定义格式文件。

#### 特点
- 适合处理历史数据和回测
- 通常不支持实时数据订阅
- 数据加载速度受I/O限制

#### 使用示例

```python
# 创建CSV文件数据源
file_source = CSVDataSource(
    name="historical_data",
    config={
        "base_dir": "/path/to/data",
        "file_pattern": "{symbol}/{timeframe}.csv",
        "datetime_format": "%Y-%m-%d %H:%M:%S"
    }
)

# 连接（验证文件路径）
await file_source.connect()

# 读取历史数据
df = await file_source.get_klines(
    symbol="BTCUSDT",
    timeframe=TimeFrame.D1,
    start_time=datetime(2020, 1, 1),
    end_time=datetime(2022, 12, 31)
)

# 断开连接
await file_source.disconnect()
```

### 4. 数据库数据源

数据库数据源从SQL或NoSQL数据库读取市场数据。

#### 特点
- 适合存储和检索大量历史数据
- 可以通过数据库通知机制支持实时更新
- 支持复杂查询和数据聚合

#### 使用示例

```python
# 创建数据库数据源
db_source = DatabaseDataSource(
    name="market_db",
    config={
        "connection_string": "postgresql://user:pass@localhost/market_data",
        "klines_table": "klines"
    }
)

# 连接到数据库
await db_source.connect()

# 查询K线数据
df = await db_source.get_klines(
    symbol="BTCUSDT",
    timeframe=TimeFrame.H4,
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 3, 31)
)

# 断开数据库连接
await db_source.disconnect()
```

## 混合使用多个数据源

`DataManager` 使您能够管理多个数据源并根据需要使用它们：

```python
# 创建数据管理器
data_manager = DataManager()

# 添加多个数据源
data_manager.add_data_source(realtime_source, make_default=True)
data_manager.add_data_source(historical_db_source)

# 连接所有数据源
await data_manager.connect_all()

# 获取默认数据源
default_source = data_manager.get_data_source()

# 获取指定的数据源
historical_source = data_manager.get_data_source("historical_db_source")

# 从不同数据源获取数据
recent_data = await default_source.get_klines(...)
historical_data = await historical_source.get_klines(...)

# 断开所有连接
await data_manager.disconnect_all()
```

## 实现自定义数据源

要实现自定义数据源，继承 `DataSource` 抽象类并实现所有必需的方法：

```python
class MyCustomDataSource(DataSource):
    """自定义数据源实现"""
    
    async def connect(self) -> bool:
        # 实现连接逻辑
        return True
    
    async def disconnect(self) -> bool:
        # 实现断开连接逻辑
        return True
    
    async def get_klines(self, symbol: str, timeframe: TimeFrame, ...) -> pd.DataFrame:
        # 实现K线数据获取逻辑
        ...
    
    # 实现其他抽象方法...
```

## 最佳实践

1. **连接管理**：始终正确地连接和断开数据源，使用try/finally确保资源释放。

2. **错误处理**：所有数据源方法都应包含适当的错误处理，特别是对于网络操作。

3. **重试机制**：对于易失败的操作（如网络请求），实现重试机制。

4. **资源限制**：注意API速率限制和连接池大小，避免超过限制。

5. **数据一致性**：确保从不同数据源获取的数据格式一致，特别是时间戳格式。

6. **缓存策略**：考虑实现数据缓存以减少重复请求，提高性能。

7. **并发控制**：当多个组件访问同一数据源时，确保并发安全。

## 结论

数据源模块通过提供统一的接口，使交易系统能够无缝地使用各种数据来源。它的抽象设计允许系统轻松切换或添加新的数据源，而不需要修改使用数据的策略或分析组件。

通过正确实现和使用数据源，系统可以可靠地获取市场数据，这是任何高效交易系统的基础。 