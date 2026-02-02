---
name: leek-strategy
description: 开发 leek 量化交易策略。继承 Strategy/CTAStrategy 基类，实现 should_open/close 方法生成交易信号。当用户要开发交易策略、择时策略、CTA策略、ML策略时使用。Use when developing trading strategies, timing strategies, or implementing buy/sell signals.
---

# Leek 策略开发

## 快速开始

继承 `CTAStrategy` 实现 K 线策略：

```python
from leek_core.strategy import CTAStrategy
from leek_core.models import Field, FieldType, KLine, PositionSide

class MyStrategy(CTAStrategy):
    display_name = "我的策略"
    
    init_params = [
        Field(name="period", label="周期", type=FieldType.INT, default=14),
    ]
    
    def __init__(self, period=14):
        super().__init__()
        self.period = period
    
    def on_kline(self, kline: KLine):
        """处理K线数据，更新指标"""
        if kline.is_finished:
            # 更新计算...
            pass
    
    def should_open(self) -> PositionSide | None:
        """开仓判断：返回 LONG/SHORT/None"""
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        """平仓判断：返回 True/False"""
        return False
```

## 核心概念

### 策略基类

| 基类 | 用途 |
|-----|------|
| `Strategy` | 通用策略基类，处理任意数据类型 |
| `CTAStrategy` | K线策略基类，自动转发到 `on_kline()` |
| `MLStrategy` | 机器学习策略，集成模型预测 |

### 关键方法

| 方法 | 调用时机 | 返回值 |
|-----|---------|--------|
| `on_kline(kline)` | 每根K线 | 无 |
| `should_open()` | 空仓时 | `PositionSide.LONG/SHORT` 或 `None` |
| `should_close(side)` | 持仓时 | `True`(平仓) / `False`(持仓) / `Decimal`(比例) |
| `after_risk_control()` | 风控触发后 | 无 |

### 策略指令

```python
from leek_core.strategy import StrategyCommand
from decimal import Decimal

# 全仓开多
return PositionSide.LONG

# 50%仓位开多
return StrategyCommand(side=PositionSide.LONG, ratio=Decimal("0.5"))
```

### 策略模式

```python
from leek_core.strategy import KlineSimple, Single

# 默认：按 symbol+timeframe 创建独立实例
strategy_mode = KlineSimple()

# 单实例模式：所有数据共享一个实例
strategy_mode = Single()
```

## 类属性

```python
class MyStrategy(CTAStrategy):
    # 必需
    display_name: str = "策略名称"
    init_params: List[Field] = [...]
    
    # 可选
    open_just_no_pos: bool = True          # 只在空仓时开仓
    accepted_data_types = {DataType.KLINE} # 接受的数据类型
    strategy_mode = KlineSimple()          # 实例模式
```

## 参数定义

```python
from leek_core.models import Field, FieldType, ChoiceType

init_params = [
    # 基本类型
    Field(name="period", label="周期", type=FieldType.INT, default=14),
    Field(name="threshold", label="阈值", type=FieldType.FLOAT, default=0.5),
    Field(name="enabled", label="启用", type=FieldType.BOOL, default=True),
    
    # 选择类型
    Field(
        name="mode",
        label="模式",
        type=FieldType.RADIO,
        choices=[("fast", "快速"), ("slow", "慢速")],
        choice_type=ChoiceType.STRING,
        default="fast",
    ),
]
```

## 使用指标

```python
from leek_core.indicators import MACD, RSI, BollBand

class MyStrategy(CTAStrategy):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__()
        self.macd = MACD(fast, slow, signal)
        self.rsi = RSI(14)
    
    def on_kline(self, kline: KLine):
        dif, dea = self.macd.update(kline)
        rsi = self.rsi.update(kline)
        
        # 设置绘图属性
        kline.dif = dif
        kline.dea = dea
        kline.rsi = rsi
```

## 状态持久化

```python
def get_state(self) -> Dict[str, Any]:
    """保存额外状态"""
    state = super().get_state()
    state["my_field"] = self.my_field
    return state

def load_state(self, state: Dict[str, Any]):
    """恢复额外状态"""
    super().load_state(state)
    self.my_field = state.get("my_field")
```

## 风控策略

```python
from leek_core.sub_strategy import SubStrategy

class StopLossPolicy(SubStrategy):
    display_name = "止损策略"
    
    init_params = [
        Field(name="stop_loss_pct", label="止损比例", type=FieldType.FLOAT, default=0.05),
    ]
    
    def evaluate(self, data, position) -> bool:
        """返回 False 触发平仓"""
        if position.pnl < -position.amount * self.stop_loss_pct:
            return False
        return True
```

## 完整示例

```python
from leek_core.strategy import CTAStrategy
from leek_core.indicators import MA
from leek_core.models import Field, FieldType, KLine, PositionSide

class DualMAStrategy(CTAStrategy):
    """双均线策略：金叉做多，死叉做空"""
    
    display_name = "双均线策略"
    
    init_params = [
        Field(name="fast_period", label="快线周期", type=FieldType.INT, default=5),
        Field(name="slow_period", label="慢线周期", type=FieldType.INT, default=20),
    ]
    
    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_ma = MA(fast_period)
        self.slow_ma = MA(slow_period)
        self.prev_fast = None
        self.prev_slow = None
        self.cur_fast = None
        self.cur_slow = None
    
    def on_kline(self, kline: KLine):
        fast = self.fast_ma.update(kline)
        slow = self.slow_ma.update(kline)
        
        if kline.is_finished and fast and slow:
            self.prev_fast, self.prev_slow = self.cur_fast, self.cur_slow
            self.cur_fast, self.cur_slow = fast, slow
            kline.fast_ma = fast
            kline.slow_ma = slow
    
    def should_open(self) -> PositionSide | None:
        if not all([self.prev_fast, self.prev_slow, self.cur_fast, self.cur_slow]):
            return None
        
        # 金叉
        if self.prev_fast <= self.prev_slow and self.cur_fast > self.cur_slow:
            return PositionSide.LONG
        # 死叉
        if self.prev_fast >= self.prev_slow and self.cur_fast < self.cur_slow:
            return PositionSide.SHORT
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        if not self.prev_fast:
            return False
        
        if position_side.is_long:
            return self.prev_fast >= self.prev_slow and self.cur_fast < self.cur_slow
        else:
            return self.prev_fast <= self.prev_slow and self.cur_fast > self.cur_slow
```

## 最佳实践

1. **数据验证**：在 `should_open/close` 开始检查数据是否充足
2. **使用完成K线**：主要逻辑在 `kline.is_finished == True` 时执行
3. **参数定义**：所有可配置参数通过 `init_params` 定义
4. **状态管理**：需要持久化的状态通过 `get_state/load_state` 处理
5. **风控分离**：复杂风控逻辑使用 SubStrategy 实现

## 详细参考

完整文档见 [reference/01-strategy.md](reference/01-strategy.md)
