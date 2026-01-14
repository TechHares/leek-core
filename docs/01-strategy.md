# 01 策略模块

## 概述

策略模块是量化交易系统的核心组件，负责根据市场数据生成交易信号。本模块采用分层设计，将策略的生命周期管理、状态机、信号生成与具体的交易逻辑分离，便于扩展和维护。

## 核心组件

### 组件层次结构

```text
StrategyContext (策略上下文 - 管理策略配置和多实例)
    └── StrategyWrapper (策略包装器 - 管理单个策略实例的生命周期)
            ├── Strategy (策略实例 - 实现具体交易逻辑)
            └── SubStrategy[] (风控策略 - 仓位风险控制)
```

### `Strategy` - 策略抽象基类

所有策略的基类，定义策略的基本接口：

```python
class Strategy(LeekComponent, ABC):
    """
    择时策略抽象基类
    
    类属性:
        display_name: 策略展示名称
        open_just_no_pos: 是否只在没有仓位时开仓，默认True
        accepted_data_types: 策略接受的数据类型列表
        strategy_mode: 策略运行模式
        init_params: 策略参数定义列表
    """
    
    # 类属性定义
    display_name: str = "未命名策略"
    open_just_no_pos: bool = True
    accepted_data_types: Set[DataType] = {DataType.KLINE}
    strategy_mode: StrategyMode = KlineSimple()
    init_params: List[Field] = []
    
    # 核心方法
    def on_data(self, data: Data) -> None: ...
    def should_open(self) -> PositionSide | StrategyCommand | None: ...
    def close(self, position: Position) -> bool | Decimal | None: ...
    def after_risk_control(self) -> None: ...
    def on_event(self, event: Event) -> None: ...
    def get_state(self) -> Dict[str, Any]: ...
    def load_state(self, state: Dict[str, Any]) -> None: ...
```

### `CTAStrategy` - CTA策略基类

CTA（Commodity Trading Advisor）策略基类，提供K线数据处理的便捷封装：

```python
class CTAStrategy(Strategy, ABC):
    """
    CTA择时策略基类，专注于K线数据处理
    """
    
    def on_data(self, data: Data) -> None:
        # 自动将KLINE类型数据转发到on_kline
        if data.data_type == DataType.KLINE:
            self.on_kline(data)
    
    def on_kline(self, kline: KLine) -> None: ...
    def should_open(self) -> PositionSide | StrategyCommand | None: ...
    def should_close(self, position_side: PositionSide) -> bool | Decimal: ...
```

### `StrategyMode` - 策略运行模式

决定如何为不同的数据创建策略实例：

```python
class StrategyMode(ABC):
    """策略运行模式，决定实例划分方式"""
    
    @abstractmethod
    def build_instance_key(self, data: Data) -> str:
        """构建实例Key，相同Key共享同一个策略实例"""
        ...

class Single(StrategyMode):
    """单实例模式：所有数据共享一个策略实例"""
    def build_instance_key(self, data: Data) -> str:
        return "default"

class KlineSimple(StrategyMode):
    """K线模式：按交易对+品种+时间周期创建独立实例"""
    def build_instance_key(self, data: KLine) -> str:
        return data.row_key  # "{symbol}_{quote_currency}_{ins_type}_{timeframe}"
```

### `StrategyCommand` - 策略指令

策略返回的交易指令：

```python
@dataclass
class StrategyCommand:
    """策略指令"""
    side: PositionSide  # 交易方向（LONG/SHORT）
    ratio: Decimal      # 仓位比例（0-1）
```

## 策略实例状态机

策略实例通过 `StrategyInstanceState` 枚举管理生命周期状态：

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                                                         │
                    ▼                                                         │
┌─────────┐   on_start()   ┌─────────┐   should_open()   ┌──────────┐       │
│ CREATED │ ─────────────► │  READY  │ ────────────────► │ ENTERING │       │
└─────────┘                └─────────┘                   └──────────┘       │
                                ▲                              │            │
                                │                              │ signal完成 │
                                │                              ▼            │
                           signal完成                    ┌──────────┐       │
                           (仓位为0)                     │ HOLDING  │       │
                                │                        └──────────┘       │
                                │                              │            │
                                │                              │ close()    │
                                │                              ▼            │
                           ┌────┴────┐                   ┌──────────┐       │
                           │         │ ◄──────────────── │ EXITING  │       │
                           │         │   signal完成      └──────────┘       │
                           │         │   (仓位为0)                          │
                           │         │                                      │
                           └─────────┴──────────────────────────────────────┘
                                                               │
                                          on_stop()            │
                                              │                │
                                              ▼                │
                                        ┌──────────┐           │
                                        │ STOPPING │ ──────────┘
                                        └──────────┘
                                              │
                                              │ 仓位清空
                                              ▼
                                        ┌──────────┐
                                        │ STOPPED  │
                                        └──────────┘
```

### 状态说明

| 状态 | 说明 |
|------|------|
| `CREATED` | 策略实例已创建，尚未启动 |
| `READY` | 空仓状态，等待开仓信号 |
| `ENTERING` | 开仓中，等待订单执行完成 |
| `HOLDING` | 持仓中，等待平仓信号 |
| `EXITING` | 平仓中，等待订单执行完成 |
| `STOPPING` | 停止中，正在清理仓位 |
| `STOPPED` | 已停止，策略实例生命周期结束 |

## 调用流程

### 1. 数据处理流程

```
数据源 ──► StrategyContext.on_data()
                    │
                    ▼
         InfoFabricators处理（计算指标等）
                    │
                    ▼
         StrategyContext._process_data()
                    │
                    ├─── 根据StrategyMode获取instance_key
                    │
                    ├─── 获取或创建StrategyWrapper
                    │
                    ▼
         StrategyWrapper.on_data()
                    │
                    ├─── 执行风控策略检查
                    │    └── SubStrategy.evaluate(data, position)
                    │
                    ├─── 根据状态执行对应处理
                    │    ├── READY    → ready_handler()  → should_open()
                    │    ├── HOLDING  → holding_handler() → close()
                    │    └── STOPPING → stopping_handler()
                    │
                    ▼
         Strategy.on_data() / on_kline()
                    │
                    ▼
         返回Signal或None
```

### 2. 信号生成与执行流程

```
should_open() 返回 PositionSide/StrategyCommand
                    │
                    ▼
         构建Asset对象
                    │
                    ▼
         StrategyContext.build_signal()
                    │
                    ▼
         生成Signal并缓存
                    │
                    ▼
         Signal发送到ExecutionEngine执行
                    │
                    ▼
         执行完成回调 exec_update()
                    │
                    ▼
         StrategyWrapper.on_signal_finish()
                    │
                    ▼
         更新策略状态
```

### 3. 风控检查流程

```
每次on_data()调用时
        │
        ▼
遍历所有风控策略（SubStrategy）
        │
        ├── policy.evaluate(data, position)
        │        │
        │        ├── True  → 继续检查下一个
        │        │
        │        └── False → 触发风控
        │                       │
        │                       ▼
        │               发布风控事件
        │                       │
        │                       ▼
        │               清理仓位
        │                       │
        │                       ▼
        │               调用strategy.after_risk_control()
        │
        ▼
所有检查通过后，继续正常策略逻辑
```

## 实现策略的原则

### 1. 策略类定义规范

```python
from decimal import Decimal
from leek_core.strategy import CTAStrategy, StrategyCommand
from leek_core.models import Field, FieldType, KLine, PositionSide

class MyStrategy(CTAStrategy):
    """
    策略文档字符串：说明策略原理、参数含义、交易逻辑
    """
    
    # 【必需】策略展示名称
    display_name: str = "我的策略"
    
    # 【可选】是否只在空仓时开仓，默认True
    open_just_no_pos: bool = True
    
    # 【必需】参数定义列表
    init_params = [
        Field(
            name="period",           # 参数名，与__init__参数名一致
            label="周期",            # 界面显示名称
            type=FieldType.INT,      # 参数类型
            default=14,              # 默认值
            required=True,           # 是否必填
            description="计算周期"   # 参数说明
        ),
    ]
    
    def __init__(self, period=14):
        """
        初始化策略
        - 调用super().__init__()
        - 初始化指标和状态变量
        """
        super().__init__()
        self.period = period
        # 初始化指标...
```

### 2. 核心方法实现

#### `on_kline(kline: KLine)` - 数据处理

```python
def on_kline(self, kline: KLine):
    """
    处理K线数据，更新指标和状态
    
    - 每根K线都会调用，包括未完成的K线
    - 通过 kline.is_finished 判断K线是否完成
    - 可以在kline上设置动态属性用于调试/绘图
    """
    # 更新指标
    value = self.indicator.update(kline)
    
    # 设置绘图属性（可选）
    kline.my_indicator = value
    
    # 保存前值用于比较
    if kline.is_finished:
        self.prev_value = value
```

#### `should_open()` - 开仓判断

```python
def should_open(self) -> PositionSide | StrategyCommand | None:
    """
    判断是否应该开仓
    
    返回值：
    - None: 不开仓
    - PositionSide.LONG: 全仓做多
    - PositionSide.SHORT: 全仓做空
    - StrategyCommand(side, ratio): 指定比例开仓
    """
    # 检查数据是否充足
    if self.indicator_value is None:
        return None
    
    # 开仓条件判断
    if self.should_go_long():
        return PositionSide.LONG
    elif self.should_go_short():
        return PositionSide.SHORT
    
    return None
```

#### `should_close(position_side)` - 平仓判断

```python
def should_close(self, position_side: PositionSide) -> bool | Decimal:
    """
    判断是否应该平仓
    
    参数：
        position_side: 当前持仓方向
    
    返回值：
    - False: 继续持仓
    - True: 全部平仓
    - Decimal: 平仓比例（0-1）
    """
    # 平仓条件判断
    if position_side.is_long:
        if self.should_close_long():
            return True
    else:
        if self.should_close_short():
            return True
    
    return False
```

#### `after_risk_control()` - 风控后处理

```python
def after_risk_control(self):
    """
    风控触发后的清理工作
    - 当风控策略触发强制平仓时调用
    - 用于清理策略内部的仓位相关状态
    """
    self.internal_position_state = None
```

### 3. 状态持久化

如果策略需要保存/恢复状态，重写以下方法：

```python
def get_state(self) -> Dict[str, Any]:
    """
    获取策略状态，用于持久化
    - 默认实现会序列化init_params中定义的字段
    - 需要额外保存的状态在此添加
    """
    state = super().get_state()
    # 添加额外状态
    state["custom_field"] = self.custom_field
    return state

def load_state(self, state: Dict[str, Any]):
    """
    加载策略状态
    - 默认实现会反序列化init_params中定义的字段
    """
    super().load_state(state)
    # 恢复额外状态
    self.custom_field = state.get("custom_field")
```

### 4. 参数类型定义

`Field` 支持的类型：

| FieldType | Python类型 | 说明 |
|-----------|-----------|------|
| `STRING` | str | 字符串 |
| `INT` | int | 整数 |
| `FLOAT` | float | 浮点数 |
| `BOOL` | bool | 布尔值 |
| `PASSWORD` | str | 密码（界面隐藏） |
| `RADIO` | Any | 单选，需配合choices |
| `SELECT` | Any | 下拉选择，需配合choices |
| `MODEL` | Dict | 模型配置 |

带选项的参数示例：

```python
Field(
    name="mode",
    label="模式",
    type=FieldType.RADIO,
    choices=[("fast", "快速"), ("slow", "慢速")],
    choice_type=FieldType.STRING,
    default="fast",
)
```

## 完整策略示例

### 示例1：简单双均线策略

```python
from decimal import Decimal
from leek_core.strategy import CTAStrategy
from leek_core.models import Field, FieldType, KLine, PositionSide

class DualMAStrategy(CTAStrategy):
    """
    双均线策略
    
    交易逻辑：
    - 快线上穿慢线时做多
    - 快线下穿慢线时做空
    - 反向信号时平仓
    """
    
    display_name = "双均线策略"
    
    init_params = [
        Field(name="fast_period", label="快线周期", type=FieldType.INT, default=5, required=True),
        Field(name="slow_period", label="慢线周期", type=FieldType.INT, default=20, required=True),
    ]
    
    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # 价格缓存
        self.prices = []
        self.fast_ma = None
        self.slow_ma = None
        self.prev_fast_ma = None
        self.prev_slow_ma = None
    
    def on_kline(self, kline: KLine):
        # 只在K线完成时更新
        if not kline.is_finished:
            return
        
        # 更新价格列表
        self.prices.append(float(kline.close))
        if len(self.prices) > self.slow_period:
            self.prices.pop(0)
        
        # 计算均线
        if len(self.prices) >= self.slow_period:
            self.prev_fast_ma = self.fast_ma
            self.prev_slow_ma = self.slow_ma
            self.fast_ma = sum(self.prices[-self.fast_period:]) / self.fast_period
            self.slow_ma = sum(self.prices) / self.slow_period
            
            # 设置绘图属性
            kline.fast_ma = self.fast_ma
            kline.slow_ma = self.slow_ma
    
    def should_open(self) -> PositionSide | None:
        if self.prev_fast_ma is None or self.prev_slow_ma is None:
            return None
        
        # 金叉做多
        if self.prev_fast_ma <= self.prev_slow_ma and self.fast_ma > self.slow_ma:
            return PositionSide.LONG
        
        # 死叉做空
        if self.prev_fast_ma >= self.prev_slow_ma and self.fast_ma < self.slow_ma:
            return PositionSide.SHORT
        
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        if self.prev_fast_ma is None:
            return False
        
        # 持多仓时出现死叉
        if position_side.is_long:
            if self.prev_fast_ma >= self.prev_slow_ma and self.fast_ma < self.slow_ma:
                return True
        # 持空仓时出现金叉
        else:
            if self.prev_fast_ma <= self.prev_slow_ma and self.fast_ma > self.slow_ma:
                return True
        
        return False
```

### 示例2：使用内置指标的策略

```python
from leek_core.strategy import CTAStrategy
from leek_core.indicators import DMI
from leek_core.models import Field, FieldType, KLine, PositionSide

class DMIStrategy(CTAStrategy):
    """
    DMI趋势跟踪策略
    
    使用DMI指标判断趋势方向和强度
    """
    
    display_name = "DMI策略"
    
    init_params = [
        Field(name="adx_smoothing", label="ADX平滑周期", type=FieldType.INT, default=6, required=True),
        Field(name="di_length", label="DI计算周期", type=FieldType.INT, default=14, required=True),
        Field(name="adx_threshold", label="ADX阈值", type=FieldType.INT, default=25, required=True),
    ]
    
    def __init__(self, adx_smoothing=6, di_length=14, adx_threshold=25):
        super().__init__()
        self.adx_threshold = adx_threshold
        self.dmi = DMI(adx_smoothing=adx_smoothing, di_length=di_length)
        
        self.adx = None
        self.up_di = None
        self.down_di = None
        self.prev_adxr = None
        self.adxr = None
    
    def on_kline(self, kline: KLine):
        # 获取前值
        last = self.dmi.last(1)
        if len(last) > 0:
            self.prev_adxr = last[0][3]
        
        # 更新指标
        self.adx, self.up_di, self.down_di, self.adxr = self.dmi.update(kline)
        
        if self.adx is not None:
            kline.adx = self.adx
            kline.up_di = self.up_di
            kline.down_di = self.down_di
    
    def should_open(self) -> PositionSide | None:
        if self.adx is None or self.prev_adxr is None:
            return None
        
        # ADX突破阈值确认趋势
        if self.prev_adxr < self.adx_threshold < self.adxr < self.adx:
            if self.up_di > self.down_di:
                return PositionSide.LONG
            elif self.up_di < self.down_di:
                return PositionSide.SHORT
        
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        if self.adx is None:
            return False
        
        # 趋势反转
        if position_side.is_long and self.up_di < self.down_di:
            return True
        if position_side.is_short and self.up_di > self.down_di:
            return True
        
        return False
```

## 机器学习策略

### `MLStrategy` 基类

用于集成机器学习模型的策略基类：

```python
class MLStrategy(CTAStrategy):
    """
    机器学习策略基类
    
    功能：
    1. 加载训练好的模型
    2. 实时计算特征
    3. 模型预测
    4. 将预测结果转换为交易信号
    """
    
    init_params = [
        Field(name="model_config", label="模型", type=FieldType.MODEL, required=True),
        Field(name="confidence_threshold", label="置信度阈值", type=FieldType.FLOAT, default=0.5),
        Field(name="warmup_periods", label="预热期", type=FieldType.INT, default=0),
    ]
```

### 实现ML策略

继承 `MLStrategy` 并实现 `_predict` 方法：

```python
from leek_core.strategy import MLStrategy
from leek_core.models import PositionSide

class MyMLStrategy(MLStrategy):
    """自定义ML策略"""
    
    display_name = "我的ML策略"
    
    def _predict(self, features):
        """
        模型预测
        
        参数：
            features: 特征向量 numpy array
        
        返回：
            信号字典 {"side": PositionSide, "confidence": float} 或 None
        """
        # 使用模型预测
        proba = self.model.predict_proba(features)
        
        if proba[0][1] > self.confidence_threshold:
            return {
                "side": PositionSide.LONG,
                "confidence": proba[0][1]
            }
        elif proba[0][0] > self.confidence_threshold:
            return {
                "side": PositionSide.SHORT,
                "confidence": proba[0][0]
            }
        
        return None
```

## 风控策略

风控策略通过 `SubStrategy` 基类实现：

```python
from leek_core.sub_strategy import SubStrategy
from leek_core.models import Data, Position

class MyRiskPolicy(SubStrategy):
    """自定义风控策略"""
    
    display_name = "我的风控"
    
    def evaluate(self, data: Data, position: Position) -> bool:
        """
        评估是否通过风控检查
        
        参数：
            data: 当前数据
            position: 当前仓位
        
        返回：
            True: 通过检查，继续持仓
            False: 触发风控，强制平仓
        """
        # 止损检查
        if position.pnl < -position.amount * Decimal("0.05"):
            return False  # 亏损超过5%，触发止损
        
        return True
```

## 最佳实践

1. **数据验证**：在 `should_open()` 和 `should_close()` 开始时检查必要数据是否存在

2. **使用完成的K线**：大多数策略逻辑应在 `kline.is_finished == True` 时执行

3. **状态管理**：合理使用 `get_state()` 和 `load_state()` 确保策略重启后状态一致

4. **日志记录**：在关键决策点添加日志，便于调试和回测分析

5. **参数定义**：所有可配置参数都应通过 `init_params` 定义，便于界面配置

6. **类型注解**：使用正确的返回类型注解，帮助理解代码意图

7. **风控集成**：复杂策略应配合风控策略使用，不要在策略内部实现所有风控逻辑
