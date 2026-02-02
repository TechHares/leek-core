# 04 风险控制模块

## 概述

风险控制模块提供多层级的风控能力，包括全局风控（RiskManager）和策略级风控（SubStrategy）。全局风控作用于所有策略的交易信号，在订单执行前进行拦截和过滤；策略级风控则针对已开仓位进行止损止盈等操作。

## 核心组件

### 组件层次结构

```text
风控体系
├── RiskManager (全局风控管理器)
│   └── StrategyPolicy[] (风控策略)
│       ├── StrategySignalLimit    # 信号频率限制
│       ├── StrategyTimeWindow     # 交易时间窗口
│       └── StrategyProfitControl  # 盈利控制
├── RiskPlugin (风控插件)
│   └── 用于批量仓位风控检查
└── SubStrategy (策略级风控)
    └── 见 02-sub-strategy.md
```

### `RiskManager` - 风控管理器

```python
class RiskManager(LeekComponent):
    """
    风控管理器组件 - 纯风控策略管理
    
    职责：
    - 风控策略管理
    - 风控检查执行
    - 风控事件发布
    - 风控策略生命周期管理
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.policies: List[StrategyPolicyContext] = []
    
    def evaluate_risk(
        self, 
        execution_context: ExecutionContext, 
        position_info: PositionInfo
    ) -> bool:
        """
        执行风控策略检查
        
        参数:
            execution_context: 执行上下文
            position_info: 仓位信息
            
        返回:
            bool: True表示风控拦截，False表示风控通过
        """
        ...
    
    def add_policy(self, policy_config: LeekComponentConfig): ...
    def remove_policy(self, instance_id: str): ...
```

### `RiskPlugin` - 风控插件基类

```python
class RiskPlugin(LeekComponent, ABC):
    """
    风控插件基类
    
    用于批量仓位风控检查，返回需要平仓的仓位列表。
    """
    
    @abstractmethod
    def trigger(self, info: PositionInfo) -> List[Position]:
        """
        检查仓位是否符合风控规则
        
        参数:
            info: 仓位信息
            
        返回:
            需要平掉的仓位列表
        """
        pass
```

### `StrategyPolicy` - 策略风控策略

```python
class StrategyPolicy(LeekComponent, ABC):
    """
    策略风控策略基类
    
    用于在信号执行前进行风控检查。
    """
    
    @abstractmethod
    def evaluate(
        self, 
        execution_context: ExecutionContext, 
        position_info: PositionInfo
    ) -> bool:
        """
        评估是否允许执行
        
        返回:
            True: 允许执行
            False: 拦截信号
        """
        ...
```

## 内置风控策略

### 1. `StrategySignalLimit` - 信号频率限制

限制策略在指定时间窗口内的信号数量。

```python
class StrategySignalLimit(StrategyPolicy):
    """信号频率限制"""
    
    display_name = "信号频率限制"
    init_params = [
        Field(name="max_signals", label="最大信号数", type=FieldType.INT, default=10),
        Field(name="window_seconds", label="时间窗口(秒)", type=FieldType.INT, default=3600),
    ]
    
    def __init__(self, max_signals: int = 10, window_seconds: int = 3600):
        self.max_signals = max_signals
        self.window_seconds = window_seconds
        self._signal_times = []  # 信号时间记录
```

**使用场景：**
- 防止策略在短时间内频繁交易
- 控制手续费成本
- 避免异常情况下的过度交易

### 2. `StrategyTimeWindow` - 交易时间窗口

限制策略只能在指定时间段内交易。

```python
class StrategyTimeWindow(StrategyPolicy):
    """交易时间窗口"""
    
    display_name = "交易时间窗口"
    init_params = [
        Field(name="start_hour", label="开始小时", type=FieldType.INT, default=9),
        Field(name="start_minute", label="开始分钟", type=FieldType.INT, default=0),
        Field(name="end_hour", label="结束小时", type=FieldType.INT, default=15),
        Field(name="end_minute", label="结束分钟", type=FieldType.INT, default=0),
        Field(name="timezone", label="时区", type=FieldType.STRING, default="Asia/Shanghai"),
    ]
```

**使用场景：**
- 避开开盘收盘的剧烈波动
- 限制夜间交易
- 适配不同市场的交易时段

### 3. `StrategyProfitControl` - 盈利控制

根据策略当日/累计盈亏进行风控。

```python
class StrategyProfitControl(StrategyPolicy):
    """盈利控制"""
    
    display_name = "盈利控制"
    init_params = [
        Field(name="daily_loss_limit", label="日亏损限制", type=FieldType.FLOAT, default=-500),
        Field(name="daily_profit_target", label="日盈利目标", type=FieldType.FLOAT, default=1000),
        Field(name="total_loss_limit", label="总亏损限制", type=FieldType.FLOAT, default=-5000),
    ]
```

**使用场景：**
- 设置每日最大亏损，达到后停止交易
- 达到盈利目标后锁定利润
- 累计亏损限制

## 风控执行流程

### 全局风控流程

```text
Engine._on_signal(signal)
        │
        ▼
Portfolio.process_signal(signal)
        │
        ▼
ExecutionContext 创建
        │
        ▼
RiskManager.evaluate_risk(context, position_info)
        │
        ├── 纯减仓操作? → 跳过风控
        │
        └── 遍历所有 policies
                │
                ├── policy.is_applicable(context)? → 检查是否适用
                │
                └── policy.evaluate(context, position_info)
                        │
                        ├── True → 继续下一个策略
                        │
                        └── False → 拦截，返回 True
```

### 风控拦截处理

```python
# RiskManager.evaluate_risk 返回 True 时
if risk_manager.evaluate_risk(execution_context, position_info):
    # 信号被拦截，不执行
    logger.warning(f"Signal {signal.signal_id} rejected by risk policy")
    return

# 正常执行流程
capital_account.freeze_amount(execution_context)
executor_manager.handle_order(execution_context)
```

## 使用示例

### 添加全局风控策略

```python
from leek_core.models import LeekComponentConfig
from leek_core.policy import StrategySignalLimit, StrategyTimeWindow

# 通过引擎添加
engine.add_position_policy(LeekComponentConfig(
    instance_id="signal-limit-001",
    name="信号频率限制",
    cls=StrategySignalLimit,
    config={
        "max_signals": 10,
        "window_seconds": 3600,
    }
))

engine.add_position_policy(LeekComponentConfig(
    instance_id="time-window-001",
    name="交易时间窗口",
    cls=StrategyTimeWindow,
    config={
        "start_hour": 9,
        "end_hour": 21,
        "timezone": "Asia/Shanghai",
    }
))
```

### 配置带作用域的风控策略

```python
# 仅对特定策略生效
engine.add_position_policy({
    "id": "profit-control-001",
    "name": "盈利控制",
    "class_name": "leek_core.policy.StrategyProfitControl",
    "config": {
        "daily_loss_limit": -500,
        "daily_profit_target": 1000,
    },
    "scope": "strategy",  # 作用范围
    "strategy_template_ids": ["strategy-001"],  # 指定策略模板
    "strategy_instance_ids": None,  # 或指定策略实例
})
```

### 自定义风控策略

```python
from leek_core.policy import StrategyPolicy
from leek_core.models import ExecutionContext, PositionInfo, Field, FieldType
from decimal import Decimal

class MaxPositionCount(StrategyPolicy):
    """最大持仓数量限制"""
    
    display_name = "最大持仓数量"
    init_params = [
        Field(name="max_positions", label="最大持仓数", type=FieldType.INT, default=5),
    ]
    
    def __init__(self, max_positions: int = 5):
        self.max_positions = max_positions
    
    def evaluate(self, execution_context: ExecutionContext, position_info: PositionInfo) -> bool:
        """检查是否超过最大持仓数"""
        # 只检查开仓操作
        if not any(asset.is_open for asset in execution_context.execution_assets):
            return True
        
        current_count = len(position_info.positions)
        
        if current_count >= self.max_positions:
            logger.warning(f"已达最大持仓数 {self.max_positions}，拒绝开仓")
            return False
        
        return True


class DrawdownControl(StrategyPolicy):
    """回撤控制"""
    
    display_name = "回撤控制"
    init_params = [
        Field(name="max_drawdown", label="最大回撤", type=FieldType.FLOAT, default=0.1),
    ]
    
    def __init__(self, max_drawdown: float = 0.1):
        self.max_drawdown = Decimal(str(max_drawdown))
        self._peak_value = None
    
    def evaluate(self, execution_context: ExecutionContext, position_info: PositionInfo) -> bool:
        """检查是否超过最大回撤"""
        # 计算当前总价值
        total_value = sum(pos.value for pos in position_info.positions)
        
        if self._peak_value is None:
            self._peak_value = total_value
        else:
            self._peak_value = max(self._peak_value, total_value)
        
        if self._peak_value <= 0:
            return True
        
        drawdown = (self._peak_value - total_value) / self._peak_value
        
        if drawdown >= self.max_drawdown:
            logger.warning(f"当前回撤 {drawdown:.2%} 超过限制 {self.max_drawdown:.2%}")
            return False
        
        return True
```

### 自定义风控插件

```python
from leek_core.risk import RiskPlugin
from leek_core.models import Position, PositionInfo
from typing import List
from datetime import datetime, timedelta

class MaxHoldingTime(RiskPlugin):
    """最大持仓时间"""
    
    display_name = "最大持仓时间"
    
    def __init__(self, max_hours: int = 24):
        self.max_hours = max_hours
    
    def trigger(self, info: PositionInfo) -> List[Position]:
        """返回超时的仓位"""
        expired_positions = []
        now = datetime.now()
        max_duration = timedelta(hours=self.max_hours)
        
        for position in info.positions:
            if now - position.open_time > max_duration:
                expired_positions.append(position)
        
        return expired_positions
```

## 风控事件

### 事件类型

| 事件类型 | 说明 |
|----------|------|
| `RISK_TRIGGERED` | 风控触发 |
| `POSITION_POLICY_ADD` | 风控策略添加 |
| `POSITION_POLICY_DEL` | 风控策略移除 |

### 监听风控事件

```python
from leek_core.event import EventType

def on_risk_triggered(event):
    data = event.data
    print(f"风控触发: 策略={data['policy_name']}, 信号={data['signal_id']}, 动作={data['action']}")

engine.event_bus.subscribe_event(EventType.RISK_TRIGGERED, on_risk_triggered)
```

## 风控层级对比

| 层级 | 组件 | 作用时机 | 作用对象 |
|------|------|----------|----------|
| 全局风控 | RiskManager | 订单执行前 | 所有策略的信号 |
| 策略级风控 | SubStrategy | K线更新时 | 已开仓位 |
| 仓位批量风控 | RiskPlugin | 定期检查 | 仓位集合 |

## 最佳实践

### 1. 多层防护

```python
# 全局风控：限制交易频率和时间
engine.add_position_policy(LeekComponentConfig(
    cls=StrategySignalLimit,
    config={"max_signals": 20, "window_seconds": 3600}
))

# 策略级风控：止损止盈
strategy_config = StrategyConfig(
    risk_policies=[
        LeekComponentConfig(cls=PositionStopLoss, config={"stop_loss_ratio": 0.05}),
        LeekComponentConfig(cls=PositionTakeProfit, config={"profit_ratio": 0.15}),
    ]
)
```

### 2. 针对性风控

```python
# 针对高频策略
engine.add_position_policy({
    ...,
    "scope": "strategy",
    "strategy_template_ids": ["high-freq-001"],
    "config": {"max_signals": 100, "window_seconds": 60}
})

# 针对趋势策略
engine.add_position_policy({
    ...,
    "scope": "strategy", 
    "strategy_template_ids": ["trend-001"],
    "config": {"max_signals": 5, "window_seconds": 3600}
})
```

### 3. 动态调整

```python
# 根据市场状态调整风控
def adjust_risk_by_volatility(volatility: float):
    if volatility > 0.05:  # 高波动
        engine.update_position_policy({
            "id": "signal-limit-001",
            "config": {"max_signals": 5}  # 减少交易
        })
    else:
        engine.update_position_policy({
            "id": "signal-limit-001",
            "config": {"max_signals": 20}  # 正常交易
        })
```

## 相关模块

- [子策略](02-sub-strategy.md) - 策略级风控
- [仓位管理](03-position.md) - 仓位信息
- [引擎架构](20-engine.md) - 风控在引擎中的位置
