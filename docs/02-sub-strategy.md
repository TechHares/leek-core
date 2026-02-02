# 02 子策略模块

## 概述

子策略（SubStrategy）模块提供仓位级别的风控和退出逻辑，用于管理已开仓位的止损、止盈和追踪退出等操作。子策略与主策略分离，实现风控逻辑的复用和灵活组合。

## 核心组件

### 组件层次结构

```text
StrategyWrapper (策略包装器)
├── Strategy (主策略实例)
└── SubStrategy[] (子策略列表)
        ├── PositionStopLoss      # 固定比例止损
        ├── PositionTakeProfit    # 固定比例止盈
        └── PositionTargetTrailingExit  # 目标追踪离场
```

### `SubStrategy` - 子策略抽象基类

```python
class SubStrategy(LeekComponent, ABC):
    """
    仓位风控策略基类
    
    职责：
    1. 统一风控规则的接口，子类需实现 evaluate 方法
    2. 支持灵活扩展多种风控逻辑
    3. 与仓位管理、策略等模块解耦集成
    """
    
    # 策略接受的数据类型
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    
    @abstractmethod
    def evaluate(self, data: Data, position: Position) -> bool:
        """
        检查仓位是否应该继续持有
        
        参数:
            data: 行情数据
            position: 仓位信息
            
        返回:
            True: 继续持有
            False: 应该平仓
        """
        ...
```

## 内置子策略

### 1. `PositionStopLoss` - 固定比例止损

当价格亏损达到设定比例时触发止损平仓。

```python
class PositionStopLoss(SubStrategy):
    """固定比例止损"""
    
    display_name = "止损"
    init_params = [
        Field(
            name="stop_loss_ratio",
            label="止损比例",
            type=FieldType.FLOAT,
            default=0.05,
            description="止损比例，默认为5%"
        )
    ]
```

**逻辑说明：**
- 多头：当收盘价 < 开仓价 × (1 - 止损比例) 时触发
- 空头：当收盘价 > 开仓价 × (1 + 止损比例) 时触发

**使用示例：**

```python
from leek_core.sub_strategy import PositionStopLoss

# 5%止损
stop_loss = PositionStopLoss(stop_loss_ratio=0.05)

# 在策略配置中使用
strategy_config = StrategyConfig(
    risk_policies=[
        LeekComponentConfig(
            instance_id="sl-001",
            name="止损",
            cls=PositionStopLoss,
            config={"stop_loss_ratio": 0.05}
        )
    ]
)
```

### 2. `PositionTakeProfit` - 固定比例止盈

当价格盈利达到设定比例时触发止盈平仓。

```python
class PositionTakeProfit(SubStrategy):
    """固定比例止盈"""
    
    display_name = "止盈"
    init_params = [
        Field(
            name="profit_ratio",
            label="止盈比例",
            type=FieldType.FLOAT,
            default=0.05,
            description="止盈比例，默认为5%"
        )
    ]
```

**逻辑说明：**
- 多头：当收盘价 > 开仓价 × (1 + 止盈比例) 时触发
- 空头：当收盘价 < 开仓价 × (1 - 止盈比例) 时触发

### 3. `PositionTargetTrailingExit` - 目标追踪离场

目标突破后启动追踪止盈，兼具止损保护和利润追踪功能。

```python
class PositionTargetTrailingExit(SubStrategy):
    """目标追踪离场"""
    
    display_name = "目标追踪离场"
    init_params = [
        Field(name="stop_loss_ratio", label="止损比例", default=2, description="相对开仓价的止损百分比"),
        Field(name="target_ratio", label="目标比例", default=5, description="相对开仓价的目标百分比"),
        Field(name="reserve_ratio", label="预留百分比", default=60, description="保留利润的比例(0~100)"),
        Field(name="profit_retention_ratio", label="利润保留比例", default=None, description="大盈利时的保留比例"),
        Field(name="small_profit_threshold", label="小盈利阈值", default=5, description="小盈利阈值百分比"),
        Field(name="large_profit_threshold", label="大盈利阈值", default=15, description="大盈利阈值百分比"),
    ]
```

**工作流程：**

```text
开仓
  │
  ▼
目标未突破阶段
  │
  ├── 触发止损 → 平仓
  │
  └── 价格突破目标 → 进入追踪阶段
                          │
                          ▼
                    追踪阶段
                          │
                          ├── 价格创新高 → 上调追踪止盈价
                          │
                          └── 回落到追踪止盈价 → 平仓
```

**核心逻辑：**

1. **目标未突破阶段**
   - 检查是否触发止损（亏损 > stop_loss_ratio）
   - 检查是否突破目标（盈利 > target_ratio）

2. **追踪阶段（目标突破后）**
   - 记录突破后的极值价格
   - 计算追踪止盈价 = 开仓价 + (极值 - 开仓价) × reserve_ratio
   - 追踪止盈价只单向调整（多头只上调，空头只下调）
   - 价格回落/反弹到追踪止盈价时平仓

3. **动态利润保留**
   - 小盈利（< small_profit_threshold）：使用 reserve_ratio
   - 大盈利（> large_profit_threshold）：使用 profit_retention_ratio
   - 中间盈利：二次函数插值（先慢后快）

**参数示例：**

```python
from leek_core.sub_strategy import PositionTargetTrailingExit

trailing = PositionTargetTrailingExit(
    stop_loss_ratio=2,          # 2%止损
    target_ratio=5,             # 5%目标
    reserve_ratio=60,           # 基础保留60%利润
    profit_retention_ratio=80,  # 大盈利时保留80%
    small_profit_threshold=5,   # 5%以下为小盈利
    large_profit_threshold=15,  # 15%以上为大盈利
)

# 场景示例：
# 开仓价100，目标价105（5%），追踪开始
# 价格涨到110（盈利10%）
# 追踪止盈价 = 100 + (110-100) × 60% = 106
# 价格回落到106时平仓，保留6%利润
```

## 使用示例

### 在策略中配置子策略

```python
from leek_core.models import LeekComponentConfig, StrategyConfig
from leek_core.sub_strategy import PositionStopLoss, PositionTargetTrailingExit

# 策略配置
strategy_config = StrategyConfig(
    data_source_configs=[...],
    strategy_config={"param1": 10},
    risk_policies=[
        # 方式1：使用类
        LeekComponentConfig(
            instance_id="sl-001",
            name="止损",
            cls=PositionStopLoss,
            config={"stop_loss_ratio": 0.05}
        ),
        # 方式2：使用字符串类路径
        LeekComponentConfig(
            instance_id="trailing-001",
            name="追踪止盈",
            cls="leek_core.sub_strategy.PositionTargetTrailingExit",
            config={
                "stop_loss_ratio": 2,
                "target_ratio": 5,
                "reserve_ratio": 60,
            }
        ),
    ]
)
```

### 在回测中使用

```python
config = {
    "strategy_class": "my_module.MyStrategy",
    "strategy_params": {...},
    "risk_policies": [
        {
            "class_name": "leek_core.sub_strategy.PositionStopLoss",
            "config": {"stop_loss_ratio": 0.03}
        },
        {
            "class_name": "leek_core.sub_strategy.PositionTakeProfit",
            "config": {"profit_ratio": 0.10}
        }
    ],
    ...
}
```

### 自定义子策略

```python
from leek_core.sub_strategy import SubStrategy
from leek_core.models import Data, Position, DataType, PositionSide, Field, FieldType
from decimal import Decimal

class PositionTimeLimit(SubStrategy):
    """持仓时间限制"""
    
    display_name = "持仓时间限制"
    init_params = [
        Field(
            name="max_bars",
            label="最大持仓K线数",
            type=FieldType.INT,
            default=24,
            description="最大持仓K线数量"
        )
    ]
    
    def __init__(self, max_bars: int = 24):
        self.max_bars = max_bars
        self._bar_counts = {}  # position_id -> bar_count
    
    def evaluate(self, data: Data, position: Position) -> bool:
        """
        检查是否超过持仓时间限制
        """
        if data.data_type != DataType.KLINE:
            return True
        
        pid = position.position_id
        if pid not in self._bar_counts:
            self._bar_counts[pid] = 0
        
        self._bar_counts[pid] += 1
        
        if self._bar_counts[pid] >= self.max_bars:
            del self._bar_counts[pid]
            return False  # 超时，应平仓
        
        return True  # 继续持有


class PositionDynamicStopLoss(SubStrategy):
    """动态止损：基于ATR"""
    
    display_name = "动态止损"
    init_params = [
        Field(name="atr_period", label="ATR周期", type=FieldType.INT, default=14),
        Field(name="atr_multiplier", label="ATR倍数", type=FieldType.FLOAT, default=2.0),
    ]
    
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        self.atr_period = atr_period
        self.atr_multiplier = Decimal(str(atr_multiplier))
        self._atr_values = {}  # position_id -> List[TR]
    
    def _calculate_atr(self, position_id: str, high: Decimal, low: Decimal, close: Decimal) -> Decimal:
        """计算ATR"""
        if position_id not in self._atr_values:
            self._atr_values[position_id] = []
        
        tr = high - low  # 简化计算
        self._atr_values[position_id].append(tr)
        
        if len(self._atr_values[position_id]) > self.atr_period:
            self._atr_values[position_id] = self._atr_values[position_id][-self.atr_period:]
        
        return sum(self._atr_values[position_id]) / len(self._atr_values[position_id])
    
    def evaluate(self, data: Data, position: Position) -> bool:
        if data.data_type != DataType.KLINE:
            return True
        
        atr = self._calculate_atr(position.position_id, data.high, data.low, data.close)
        stop_distance = atr * self.atr_multiplier
        
        if position.side == PositionSide.LONG:
            stop_price = position.cost_price - stop_distance
            return data.close > stop_price
        else:
            stop_price = position.cost_price + stop_distance
            return data.close < stop_price
```

## 执行流程

### 子策略评估时机

```text
Engine.on_data(data)
        │
        ▼
StrategyManager.process_data(data)
        │
        ▼
StrategyWrapper._process_data(data)
        │
        ├── Strategy.on_data(data)  # 主策略处理
        │
        └── 对每个持仓执行风控检查:
                │
                ▼
            for position in positions:
                for sub_strategy in sub_strategies:
                    if not sub_strategy.evaluate(data, position):
                        # 产生平仓信号
                        close_signal = create_close_signal(position)
```

### 评估结果处理

| 返回值 | 含义 | 处理 |
|--------|------|------|
| `True` | 继续持有 | 无操作 |
| `False` | 应该平仓 | 产生平仓信号 |

## 配置说明

### 通用参数定义

子策略参数通过 `Field` 类定义：

```python
from leek_core.models import Field, FieldType

init_params = [
    Field(
        name="param_name",        # 参数名（与__init__参数对应）
        label="参数标签",         # 显示名称
        type=FieldType.FLOAT,    # 参数类型
        default=0.05,            # 默认值
        min=0,                   # 最小值
        max=1,                   # 最大值
        required=True,           # 是否必填
        description="参数说明"   # 描述
    )
]
```

### 支持的字段类型

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

## 最佳实践

### 1. 合理设置止损

```python
# 建议止损比例 2%-5%
PositionStopLoss(stop_loss_ratio=0.03)  # 3%止损
```

### 2. 止损止盈配合使用

```python
risk_policies = [
    LeekComponentConfig(cls=PositionStopLoss, config={"stop_loss_ratio": 0.03}),
    LeekComponentConfig(cls=PositionTakeProfit, config={"profit_ratio": 0.09}),
]
# 盈亏比 3:1
```

### 3. 追踪止盈适用场景

- 趋势行情：使用较大的 reserve_ratio（如60-70%）
- 震荡行情：使用较小的 reserve_ratio（如40-50%）

### 4. 子策略优先级

多个子策略按添加顺序执行，任一返回 False 即触发平仓。

## 相关模块

- [策略模块](01-strategy.md) - 主策略开发
- [仓位管理](03-position.md) - Position 数据结构
- [风险控制](04-risk.md) - 全局风控
