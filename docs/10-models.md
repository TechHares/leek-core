# 10 数据模型

## 概述

数据模型模块定义了 Leek Core 中使用的核心数据结构，包括 K 线数据、订单、仓位、信号等。所有数据模型均使用 Python dataclass 实现，支持类型标注和默认值。

## 市场数据模型

### `Data` - 数据基类

```python
@dataclass
class Data(metaclass=ABCMeta):
    """数据源产出的数据结构基类"""
    
    data_source_id: str = None           # 数据源ID
    data_type: DataType = None           # 数据类型
    target_instance_id: Set[str] = field(default_factory=set)  # 目标实例ID
    
    dynamic_attrs: Dict[str, Any] = field(default_factory=dict)  # 动态属性
    metadata: Dict[str, Any] = field(default_factory=dict)       # 元数据
    
    @property
    @abstractmethod
    def row_key(self) -> str:
        """获取数据的唯一标识符"""
        ...
    
    def get(self, name: str, default: Any = None) -> Any: ...
    def set(self, name: str, value: Any) -> None: ...
```

### `KLine` - K线数据

```python
@dataclass
class KLine(Data):
    """K线数据传输对象"""
    
    symbol: str = None           # 交易对符号
    market: str = None           # 市场标识（如"okx"）
    quote_currency: str = "USDT" # 计价币种
    ins_type: TradeInsType = TradeInsType.SPOT  # 交易品种类型
    asset_type: AssetType = AssetType.CRYPTO    # 资产类型
    
    # 价格数据
    open: Decimal = None         # 开盘价
    close: Decimal = None        # 收盘价
    high: Decimal = None         # 最高价
    low: Decimal = None          # 最低价
    
    # 成交数据
    volume: Decimal = None       # 成交量
    amount: Decimal = None       # 成交额
    
    # 时间数据（毫秒时间戳）
    start_time: int = None       # K线开始时间
    end_time: int = None         # K线结束时间
    current_time: int = None     # 当前时间
    timeframe: TimeFrame = None  # K线时间粒度
    is_finished: bool = False    # K线是否已完成
    
    @property
    def duration(self) -> int:
        """K线持续时间(毫秒)"""
        return self.end_time - self.start_time
    
    @property
    def row_key(self) -> str:
        """唯一标识符：{symbol}_{quote_currency}_{ins_type}_{timeframe}"""
        return f"{self.symbol}_{self.quote_currency}_{self.ins_type.value}_{self.timeframe.value}"
    
    @staticmethod
    def pack_row_key(symbol: str, quote_currency: str, ins_type: TradeInsType, timeframe: TimeFrame) -> str:
        """构建行键"""
        ...
    
    @staticmethod
    def parse_row_key(row_key: str) -> tuple:
        """解析行键"""
        ...
```

### `DataType` - 数据类型枚举

```python
class DataType(Enum):
    KLINE = "kline"              # K线数据
    TICK = "tick"                # 逐笔数据
    DEPTH = "depth"              # 深度数据
    TRADE = "trade"              # 成交数据
    INDEX = "index"              # 指数数据
```

### `TimeFrame` - 时间周期枚举

```python
class TimeFrame(Enum):
    TICK = "tick"    # 逐笔
    S1 = "1s"        # 1秒
    M1 = "1m"        # 1分钟
    M3 = "3m"        # 3分钟
    M5 = "5m"        # 5分钟
    M15 = "15m"      # 15分钟
    M30 = "30m"      # 30分钟
    H1 = "1h"        # 1小时
    H2 = "2h"        # 2小时
    H4 = "4h"        # 4小时
    H6 = "6h"        # 6小时
    H12 = "12h"      # 12小时
    D1 = "1d"        # 1天
    W1 = "1w"        # 1周
    MO1 = "1M"       # 1月
    
    @property
    def milliseconds(self) -> int:
        """返回时间周期的毫秒数"""
        ...
```

## 交易数据模型

### `Signal` - 交易信号

```python
@dataclass
class Signal:
    """交易信号模型"""
    
    signal_id: str                # 信号ID
    data_source_instance_id: str  # 数据源实例ID
    strategy_id: str              # 策略ID
    strategy_instance_id: str     # 策略实例ID
    signal_time: datetime         # 信号产生时间
    strategy_cls: str             # 策略类名
    
    assets: list[Asset] = list           # 资产列表
    config: StrategyPositionConfig = None # 仓位配置
    extra: Any = None                     # 扩展信息
```

### `Asset` - 资产约束

```python
@dataclass
class Asset:
    """资产约束模型"""
    
    asset_type: AssetType       # 资产类型
    ins_type: TradeInsType      # 交易品种类型
    symbol: str                 # 交易对标识
    
    side: PositionSide          # 多空方向
    price: Decimal              # 建议价格
    ratio: Decimal              # 仓位比例 (0~1)
    actual_ratio: Decimal = None # 实际仓位比例
    is_open: bool = False       # 是否开仓
    
    quote_currency: str = None  # 计价币种
    extra: Any = None           # 扩展信息
    
    @property
    def asset_key(self) -> str:
        return f"{self.symbol}_{self.quote_currency}_{self.ins_type.value}_{self.asset_type.value}_{self.side.value}"
```

### `Order` - 订单

```python
@dataclass
class Order:
    """订单数据结构"""
    
    order_id: str                # 订单ID
    position_id: str             # 仓位ID
    strategy_id: str             # 策略ID
    strategy_instance_id: str    # 策略实例ID
    signal_id: str               # 信号ID
    exec_order_id: str           # 执行订单ID
    
    order_status: OrderStatus    # 订单状态
    signal_time: datetime        # 信号时间
    order_time: datetime         # 订单时间
    
    symbol: str                  # 交易标的
    quote_currency: str          # 计价货币
    ins_type: TradeInsType       # 合约/现货类型
    asset_type: AssetType        # 资产类型
    side: PositionSide           # 交易方向
    
    is_open: bool                # 是否开仓
    is_fake: bool                # 是否虚拟订单
    order_amount: Decimal        # 订单金额
    order_price: Decimal         # 订单价格
    ratio: Decimal               # 比例
    order_type: OrderType = None # 订单类型
    
    # 成交信息
    settle_amount: Decimal = None    # 实际成交金额
    execution_price: Decimal = None  # 实际成交价格
    sz: Decimal = None               # 订单数量
    fee: Decimal = None              # 手续费
    pnl: Decimal = None              # 已实现盈亏
    unrealized_pnl: Decimal = None   # 未实现盈亏
    finish_time: datetime = None     # 完成时间
    friction: Decimal = Decimal(0)   # 摩擦成本
    leverage: Decimal = Decimal(1)   # 杠杆倍数
    
    executor_id: str = None          # 执行器ID
    trade_mode: TradeMode = None     # 交易模式
    market_order_id: str = None      # 交易所订单ID
    extra: dict = None               # 附加信息
```

### `OrderStatus` - 订单状态

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
        return self in [OrderStatus.FILLED, OrderStatus.CANCELED, 
                       OrderStatus.EXPIRED, OrderStatus.ERROR]
    
    @property
    def is_failed(self) -> bool:
        """订单是否失败"""
        return self in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.ERROR]
```

### `Position` - 仓位

```python
@dataclass
class Position:
    """仓位数据结构"""
    
    position_id: str             # 仓位ID
    strategy_id: str             # 策略ID
    strategy_instance_id: str    # 策略实例ID
    
    symbol: str                  # 交易对
    quote_currency: str          # 计价币种
    ins_type: TradeInsType       # 品种类型
    asset_type: AssetType        # 资产类型
    side: PositionSide           # 方向
    
    # 数量与价格
    sz: Decimal                  # 持仓数量
    cost_price: Decimal          # 成本价
    market_price: Decimal        # 市场价
    executor_sz: Decimal         # 交易所实际数量
    
    # 金额与比例
    amount: Decimal              # 投入本金
    ratio: Decimal               # 占用比例
    leverage: int                # 杠杆倍数
    
    # 盈亏
    pnl: Decimal                 # 已实现盈亏
    unrealized_pnl: Decimal      # 未实现盈亏
    fee: Decimal                 # 手续费
    friction: Decimal            # 摩擦成本
    
    # 时间
    open_time: datetime          # 开仓时间
    update_time: datetime        # 更新时间
    
    # 状态
    is_closed: bool              # 是否已平仓
    
    # 虚拟仓位
    virtual_positions: List[VirtualPosition] = field(default_factory=list)
    
    @property
    def value(self) -> Decimal:
        """仓位当前价值"""
        return self.sz * self.market_price
```

## 常量枚举

### `PositionSide` - 仓位方向

```python
class PositionSide(Enum):
    LONG = 1         # 做多
    SHORT = 2        # 做空
    NEUTRAL = 0      # 中性
    
    def switch(self) -> "PositionSide":
        """切换方向"""
        if self == PositionSide.LONG:
            return PositionSide.SHORT
        elif self == PositionSide.SHORT:
            return PositionSide.LONG
        return self
```

### `OrderType` - 订单类型

```python
class OrderType(Enum):
    MarketOrder = 1   # 市价单
    LimitOrder = 2    # 限价单
```

### `TradeInsType` - 交易品种类型

```python
class TradeInsType(Enum):
    SPOT = 1         # 现货
    SWAP = 2         # 永续合约
    FUTURES = 3      # 交割合约
    MARGIN = 4       # 杠杆现货
```

### `AssetType` - 资产类型

```python
class AssetType(Enum):
    STOCK = 1        # 股票
    FUTURES = 2      # 期货
    CRYPTO = 3       # 加密货币
    FOREX = 4        # 外汇
    BOND = 5         # 债券
```

### `TradeMode` - 交易模式

```python
class TradeMode(Enum):
    ISOLATED = 1     # 逐仓
    CROSS = 2        # 全仓
```

## 配置模型

### `PositionConfig` - 仓位配置

```python
@dataclass
class PositionConfig:
    init_amount: Decimal              # 初始资金
    max_amount: Decimal               # 单次最大金额
    max_ratio: Decimal                # 单次最大比例
    max_strategy_amount: Decimal      # 单策略最大金额
    max_strategy_ratio: Decimal       # 单策略最大比例
    max_symbol_amount: Decimal        # 单品种最大金额
    max_symbol_ratio: Decimal         # 单品种最大比例
    default_leverage: int = 1         # 默认杠杆
    order_type: OrderType = None      # 订单类型
    trade_type: TradeType = None      # 交易类型
    trade_mode: TradeMode = None      # 交易模式
    data: dict = None                 # 初始化数据
```

### `LeekComponentConfig` - 组件配置

```python
@dataclass
class LeekComponentConfig(Generic[T, C]):
    instance_id: str       # 实例ID
    name: str              # 名称
    cls: Type[T]           # 组件类
    config: C = None       # 配置参数
    data: dict = None      # 初始化数据
    extra: dict = None     # 额外信息
```

### `StrategyConfig` - 策略配置

```python
@dataclass
class StrategyConfig:
    data_source_configs: List[LeekComponentConfig]    # 数据源配置
    info_fabricator_configs: List[LeekComponentConfig] # 信息处理器配置
    strategy_config: Dict[str, Any]                   # 策略参数
    strategy_position_config: StrategyPositionConfig  # 策略仓位配置
    risk_policies: List[LeekComponentConfig]          # 风控策略配置
```

## 参数定义

### `Field` - 参数字段

```python
@dataclass
class Field:
    name: str                    # 参数名
    label: str                   # 显示标签
    type: FieldType              # 参数类型
    default: Any = None          # 默认值
    required: bool = False       # 是否必填
    description: str = None      # 描述
    min: float = None            # 最小值
    max: float = None            # 最大值
    options: List[dict] = None   # 选项列表（用于SELECT类型）
```

### `FieldType` - 参数类型

```python
class FieldType(Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PASSWORD = "password"
    SELECT = "select"
    MULTISELECT = "multiselect"
```

## 使用示例

### 创建K线数据

```python
from leek_core.models import KLine, TimeFrame, TradeInsType
from decimal import Decimal

kline = KLine(
    symbol="BTC",
    market="okx",
    quote_currency="USDT",
    ins_type=TradeInsType.SWAP,
    open=Decimal("50000"),
    close=Decimal("50100"),
    high=Decimal("50200"),
    low=Decimal("49900"),
    volume=Decimal("1000"),
    amount=Decimal("50000000"),
    start_time=1704067200000,
    end_time=1704070800000,
    current_time=1704070800000,
    timeframe=TimeFrame.H1,
    is_finished=True,
)

print(kline.row_key)  # BTC_USDT_2_1h
```

### 创建交易信号

```python
from leek_core.models import Signal, Asset, PositionSide, AssetType, TradeInsType
from decimal import Decimal
from datetime import datetime

signal = Signal(
    signal_id="sig-001",
    data_source_instance_id="ds-001",
    strategy_id="strategy-001",
    strategy_instance_id="inst-001",
    signal_time=datetime.now(),
    strategy_cls="MyStrategy",
    assets=[
        Asset(
            asset_type=AssetType.CRYPTO,
            ins_type=TradeInsType.SWAP,
            symbol="BTC",
            side=PositionSide.LONG,
            price=Decimal("50000"),
            ratio=Decimal("0.1"),
            is_open=True,
            quote_currency="USDT",
        )
    ]
)
```

### 动态属性

K线数据支持动态属性，用于存储指标计算结果：

```python
kline = KLine(...)

# 设置动态属性
kline.ma20 = Decimal("50050")
kline.rsi = 65.5

# 获取动态属性
print(kline.ma20)  # Decimal('50050')
print(kline.rsi)   # 65.5

# 使用get方法（带默认值）
print(kline.get("ma50", 0))  # 0
```

## 相关模块

- [数据源](11-data-sources.md) - 数据获取
- [策略模块](01-strategy.md) - 信号生成
- [执行器](23-executor.md) - 订单执行
- [仓位管理](03-position.md) - Position 详情
