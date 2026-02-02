---
name: leek-risk-position
description: leek 风险管理和仓位控制。使用 Portfolio 管理投资组合，RiskPlugin 实现风控策略，CapitalAccount 管理资金，PositionTracker 跟踪仓位。当用户要实现风控策略、仓位管理、资金管理时使用。Use when implementing risk management, position sizing, capital management, or portfolio management.
---

# Leek 风险与仓位管理

## 架构概览

```
Portfolio (投资组合管理器)
    ├── CapitalAccount (资金账户)
    ├── PositionTracker (仓位跟踪器)
    └── RiskManager (风控管理器)
            └── RiskPlugin[] (风控插件)
```

## 核心组件

### Portfolio 投资组合管理器

统一管理资金和仓位，处理策略信号：

```python
from leek_core.position import Portfolio
from leek_core.models import PositionConfig, Signal

# 配置
config = PositionConfig(
    init_amount=Decimal("10000"),
    max_ratio=Decimal("0.3"),           # 单次最大仓位比例
    max_strategy_ratio=Decimal("0.5"),  # 单策略最大仓位比例
    max_amount=Decimal("5000"),         # 单次最大金额
    max_strategy_amount=Decimal("8000"), # 单策略最大金额
    default_leverage=1,
)

portfolio = Portfolio(event_bus, config)

# 处理信号
execution_context = portfolio.process_signal(signal)

# 查询状态
print(f"总资产: {portfolio.total_amount}")
print(f"可用资金: {portfolio.available_amount}")
print(f"仓位价值: {portfolio.position_value}")
print(f"总盈亏: {portfolio.pnl}")
```

### CapitalAccount 资金账户

管理资金余额、冻结/解冻：

```python
from leek_core.position import CapitalAccount

account = CapitalAccount(event_bus, initial_balance=Decimal("10000"))

# 属性
account.available_balance  # 可用余额
account.frozen_balance     # 冻结金额
account.total_balance      # 总余额

# 冻结资金（由 Portfolio 内部调用）
account.freeze_amount(execution_context)

# 解冻资金
account.unfreeze_amount(order)

# 变更余额
account.change_amount(Decimal("100"), "盈利入账")
```

### PositionTracker 仓位跟踪器

跟踪和查询仓位状态：

```python
from leek_core.position import PositionTracker

tracker = PositionTracker(event_bus)

# 查询仓位
positions = tracker.find_position(
    strategy_id="strategy_001",
    symbol="BTCUSDT",
)

# 获取策略使用情况
used_amount, used_ratio = tracker.get_strategy_used("strategy_001")

# 获取标的使用情况
used_amount, used_ratio = tracker.get_symbol_used("BTCUSDT", "USDT")

# 获取总仓位价值
total_value = tracker.get_total_position_value()
```

### RiskPlugin 风控插件基类

实现自定义风控策略：

```python
from leek_core.risk import RiskPlugin
from leek_core.models import Position, PositionInfo
from typing import List

class RiskPlugin(LeekComponent, ABC):
    @abstractmethod
    def trigger(self, info: PositionInfo) -> List[Position]:
        """
        检查仓位是否触发风控
        
        返回需要平掉的仓位列表
        """
        pass
```

## 开发风控插件

### 基本模式

```python
from leek_core.risk import RiskPlugin
from leek_core.models import Position, PositionInfo, Field, FieldType
from typing import List
from decimal import Decimal

class MaxDrawdownRisk(RiskPlugin):
    """最大回撤风控"""
    
    display_name = "最大回撤风控"
    
    init_params = [
        Field(name="max_drawdown", label="最大回撤", 
              type=FieldType.FLOAT, default=0.1),
    ]
    
    def __init__(self, max_drawdown: float = 0.1):
        self.max_drawdown = Decimal(str(max_drawdown))
    
    def trigger(self, info: PositionInfo) -> List[Position]:
        """检查回撤是否超限"""
        positions_to_close = []
        
        for position in info.positions:
            if position.is_closed:
                continue
            
            # 计算回撤
            if position.max_profit > 0:
                drawdown = (position.max_profit - position.pnl) / position.max_profit
                if drawdown > self.max_drawdown:
                    positions_to_close.append(position)
        
        return positions_to_close
```

### 止损风控

```python
class StopLossRisk(RiskPlugin):
    """固定比例止损"""
    
    display_name = "止损风控"
    
    init_params = [
        Field(name="stop_loss_rate", label="止损比例", 
              type=FieldType.FLOAT, default=0.05),
    ]
    
    def __init__(self, stop_loss_rate: float = 0.05):
        self.stop_loss_rate = Decimal(str(stop_loss_rate))
    
    def trigger(self, info: PositionInfo) -> List[Position]:
        positions_to_close = []
        
        for position in info.positions:
            if position.is_closed:
                continue
            
            # 亏损比例
            loss_rate = -position.pnl / position.amount if position.amount > 0 else 0
            
            if loss_rate > self.stop_loss_rate:
                positions_to_close.append(position)
        
        return positions_to_close
```

### 止盈风控

```python
class TakeProfitRisk(RiskPlugin):
    """固定比例止盈"""
    
    display_name = "止盈风控"
    
    init_params = [
        Field(name="take_profit_rate", label="止盈比例", 
              type=FieldType.FLOAT, default=0.1),
    ]
    
    def __init__(self, take_profit_rate: float = 0.1):
        self.take_profit_rate = Decimal(str(take_profit_rate))
    
    def trigger(self, info: PositionInfo) -> List[Position]:
        positions_to_close = []
        
        for position in info.positions:
            if position.is_closed:
                continue
            
            # 盈利比例
            profit_rate = position.pnl / position.amount if position.amount > 0 else 0
            
            if profit_rate > self.take_profit_rate:
                positions_to_close.append(position)
        
        return positions_to_close
```

### 移动止损风控

```python
class TrailingStopRisk(RiskPlugin):
    """移动止损（追踪止损）"""
    
    display_name = "移动止损"
    
    init_params = [
        Field(name="trail_rate", label="回撤比例", 
              type=FieldType.FLOAT, default=0.05),
        Field(name="activation_rate", label="激活盈利比例", 
              type=FieldType.FLOAT, default=0.03),
    ]
    
    def __init__(self, trail_rate: float = 0.05, activation_rate: float = 0.03):
        self.trail_rate = Decimal(str(trail_rate))
        self.activation_rate = Decimal(str(activation_rate))
    
    def trigger(self, info: PositionInfo) -> List[Position]:
        positions_to_close = []
        
        for position in info.positions:
            if position.is_closed:
                continue
            
            # 检查是否激活移动止损
            profit_rate = position.pnl / position.amount if position.amount > 0 else 0
            if profit_rate < self.activation_rate:
                continue
            
            # 计算从最高点回撤
            if position.max_profit > 0:
                drawdown = (position.max_profit - position.pnl) / position.max_profit
                if drawdown > self.trail_rate:
                    positions_to_close.append(position)
        
        return positions_to_close
```

## 数据模型

### Position 仓位

```python
from leek_core.models import Position, PositionSide

position = Position(
    position_id="pos_001",
    strategy_id="strategy_001",
    symbol="BTCUSDT",
    side=PositionSide.LONG,
    amount=Decimal("1000"),      # 投入金额
    quantity=Decimal("0.02"),    # 持仓数量
    entry_price=Decimal("50000"),# 入场价格
    current_price=Decimal("51000"),
    pnl=Decimal("20"),           # 盈亏
    max_profit=Decimal("50"),    # 历史最高盈利
    ratio=Decimal("0.1"),        # 仓位比例
    leverage=10,
)

# 属性
position.is_long    # 是否多仓
position.is_short   # 是否空仓
position.is_closed  # 是否已平仓
position.pnl_rate   # 盈亏率
```

### PositionInfo 仓位信息

```python
from leek_core.models import PositionInfo

info = PositionInfo(
    positions=positions,           # 仓位列表
    total_amount=Decimal("10000"), # 总资产
    available_amount=Decimal("5000"), # 可用资金
    current_data=kline,            # 当前数据
)
```

### PositionConfig 仓位配置

```python
from leek_core.models import PositionConfig, OrderType, TradeMode, TradeType

config = PositionConfig(
    init_amount=Decimal("10000"),      # 初始资金
    max_ratio=Decimal("0.3"),          # 单次最大仓位比例
    max_strategy_ratio=Decimal("0.5"), # 单策略最大仓位比例
    max_amount=Decimal("5000"),        # 单次最大金额
    max_strategy_amount=Decimal("8000"),# 单策略最大金额
    default_leverage=1,                 # 默认杠杆
    order_type=OrderType.MARKET,        # 订单类型
    trade_mode=TradeMode.CROSS,         # 交易模式（全仓/逐仓）
    trade_type=TradeType.ONEWAY,        # 交易类型（单向/双向）
)
```

## 在策略中使用风控

### 配置风控策略

```python
from leek_core.backtest import RunConfig

config = RunConfig(
    strategy_class="my_module.MyStrategy",
    risk_policies=[
        {
            "class": "my_module.StopLossRisk",
            "params": {"stop_loss_rate": 0.05}
        },
        {
            "class": "my_module.TakeProfitRisk",
            "params": {"take_profit_rate": 0.15}
        },
        {
            "class": "my_module.TrailingStopRisk",
            "params": {"trail_rate": 0.05, "activation_rate": 0.03}
        },
    ],
    ...
)
```

### 策略内部风控回调

```python
from leek_core.strategy import CTAStrategy

class MyStrategy(CTAStrategy):
    def after_risk_control(self):
        """风控触发后的清理工作"""
        # 清理策略内部状态
        self.internal_state = None
        self.signal_pending = False
```

## 内置风控策略

| 策略 | 说明 | 参数 |
|-----|------|------|
| `PositionStopLoss` | 固定止损 | `stop_loss_rate` |
| `PositionTakeProfit` | 固定止盈 | `take_profit_rate` |
| `PositionTargetTrailingExit` | 移动止损 | `trail_rate`, `activation_rate` |

## 最佳实践

1. **多层风控**：组合使用止损、止盈、移动止损
2. **参数合理**：止损不宜过大（建议 3-5%），止盈不宜过小
3. **激活条件**：移动止损设置激活条件，避免过早触发
4. **分离关注**：风控逻辑与策略逻辑分离
5. **状态清理**：策略实现 `after_risk_control` 清理内部状态
6. **资金管理**：设置合理的单次和单策略最大仓位比例
