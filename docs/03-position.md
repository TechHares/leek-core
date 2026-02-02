# 03 仓位管理模块

## 概述

仓位管理模块负责资金账户、仓位跟踪和投资组合的统一管理。模块采用组合模式，将资金管理和仓位跟踪分离，通过 Portfolio 类提供统一的业务接口。

## 核心组件

### 组件层次结构

```text
Portfolio (投资组合管理器)
├── CapitalAccount (资金账户)
│   ├── available_balance    # 可用余额
│   ├── frozen_by_strategy   # 按策略冻结资金
│   └── frozen_by_signal     # 按信号冻结明细
├── PositionTracker (仓位跟踪器)
│   └── positions: Dict[str, Position]
└── RiskManager (风险管理器)
```

### `Portfolio` - 投资组合管理器

```python
class Portfolio:
    """
    投资组合管理器 - 组合使用 CapitalAccount 和 PositionTracker
    
    职责：
    - 信号处理和订单生成
    - 资金和仓位的协调使用
    - 风险控制和策略执行
    - 对外提供统一的业务接口
    """
    
    def __init__(
        self, 
        event_bus: EventBus, 
        config: PositionConfig,
        capital_account: CapitalAccount = None,
        position_tracker: PositionTracker = None,
        risk_manager: RiskManager = None
    ):
        ...
    
    # 核心属性
    @property
    def total_amount(self) -> Decimal: ...      # 总资产
    @property
    def available_amount(self) -> Decimal: ...  # 可用资金
    @property
    def position_value(self) -> Decimal: ...    # 仓位价值
    @property
    def total_value(self) -> Decimal: ...       # 总价值
    @property
    def profit(self) -> Decimal: ...            # 总收益
    
    # 核心方法
    def process_signal(self, signal: Signal) -> ExecutionContext | None: ...
    def order_update(self, order: Order): ...
    def update_config(self, config: PositionConfig): ...
    
    # 状态管理
    def get_state(self) -> dict: ...
    def load_state(self, state: dict): ...
    def reset_stats(self): ...
```

### `CapitalAccount` - 资金账户

```python
class CapitalAccount(LeekComponent):
    """
    资金账户组件 - 纯资金管理
    
    职责：
    - 资金余额管理
    - 资金冻结/解冻
    - 交易流水记录
    - 资金查询统计
    """
    
    def __init__(self, event_bus: EventBus, initial_balance: Decimal = Decimal('0')):
        self.init_balance = initial_balance     # 初始资金
        self.available_balance = initial_balance # 可用余额
        self.frozen_by_strategy: Dict[str, Decimal] = {}  # 按策略冻结
        self.frozen_by_signal: Dict[str, List[Transaction]] = {}  # 按信号冻结
    
    @property
    def total_balance(self) -> Decimal: ...   # 总余额
    @property
    def frozen_balance(self) -> Decimal: ...  # 冻结金额
    
    def freeze_amount(self, execution_context: ExecutionContext) -> bool: ...
    def unfreeze_amount(self, order: Order): ...
    def change_amount(self, delta: Decimal, desc: str): ...
```

### `PositionTracker` - 仓位跟踪器

```python
class PositionTracker(LeekComponent):
    """
    仓位跟踪器组件 - 纯仓位跟踪
    
    职责：
    - 仓位状态跟踪
    - 仓位数据查询
    - 仓位更新通知
    - 仓位统计计算
    """
    
    def __init__(self, event_bus: EventBus):
        self.positions: Dict[str, Position] = {}  # 仓位字典
    
    def find_position(
        self, 
        position_id: str = None,
        strategy_id: str = None,
        strategy_instance_id: str = None,
        symbol: str = None,
        quote_currency: str = None,
        ins_type: TradeInsType = None,
        asset_type: AssetType = None
    ) -> List[Position]: ...
    
    def get_strategy_used(self, strategy_id: str) -> Tuple[Decimal, Decimal]: ...
    def get_symbol_used(self, symbol: str, quote_currency: str) -> Tuple[Decimal, Decimal]: ...
    def get_total_position_value(self) -> Decimal: ...
    def order_update(self, order: Order, ...): ...
    def on_data(self, data: Data): ...  # 更新仓位市价
```

## 数据模型

### `Position` - 仓位

```python
@dataclass
class Position:
    position_id: str           # 仓位ID
    strategy_id: str           # 策略ID
    strategy_instance_id: str  # 策略实例ID
    symbol: str                # 交易对
    quote_currency: str        # 计价币种
    ins_type: TradeInsType     # 品种类型
    asset_type: AssetType      # 资产类型
    side: PositionSide         # 方向 (LONG/SHORT)
    
    # 数量与价格
    sz: Decimal                # 持仓数量
    cost_price: Decimal        # 成本价
    market_price: Decimal      # 市场价
    executor_sz: Decimal       # 交易所实际数量
    
    # 金额与比例
    amount: Decimal            # 投入本金
    ratio: Decimal             # 占用比例
    leverage: int              # 杠杆倍数
    
    # 盈亏
    pnl: Decimal               # 已实现盈亏
    unrealized_pnl: Decimal    # 未实现盈亏
    fee: Decimal               # 手续费
    friction: Decimal          # 摩擦成本
    
    # 时间
    open_time: datetime        # 开仓时间
    update_time: datetime      # 更新时间
    
    # 状态
    is_closed: bool            # 是否已平仓
    
    # 虚拟仓位（用于部分平仓）
    virtual_positions: List[VirtualPosition] = field(default_factory=list)
    
    @property
    def value(self) -> Decimal:
        """仓位当前价值"""
        return self.sz * self.market_price
```

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
    data: dict = None                 # 初始化状态数据
```

### `Transaction` - 资金流水

```python
@dataclass
class Transaction:
    strategy_id: str
    strategy_instance_id: str
    position_id: str
    exec_order_id: str
    order_id: str
    signal_id: str
    asset_key: str
    type: TransactionType      # FROZEN / UNFROZEN / SETTLE
    amount: Decimal
    balance_before: Decimal
    balance_after: Decimal
    desc: str
    timestamp: datetime
```

## 信号处理流程

### 1. 信号评估

```text
Portfolio.process_signal(signal)
        │
        ▼
_evaluate_signal(signal)
        │
        ├── 检查可用资金
        │
        ├── 计算策略已用资金和比例
        │   └── PositionTracker.get_strategy_used()
        │
        ├── 计算本次可投入金额
        │   └── min(max_amount, max_strategy_amount - used, available)
        │
        ├── 检查品种限制
        │   └── PositionTracker.get_symbol_used()
        │
        └── 生成 ExecutionAsset 列表
```

### 2. 资金冻结

```text
CapitalAccount.freeze_amount(execution_context)
        │
        ├── 检查可用余额是否充足
        │
        ├── 扣减可用余额
        │
        ├── 记录冻结流水 (Transaction)
        │
        └── 按策略/信号分组冻结
```

### 3. 订单更新处理

```text
Portfolio.order_update(order)
        │
        ├── 解冻资金
        │   └── CapitalAccount.unfreeze_amount()
        │
        ├── 计算订单变动
        │   └── PositionTracker.get_order_change()
        │
        ├── 更新仓位
        │   └── PositionTracker.order_update()
        │
        └── 更新统计数据
            ├── pnl (已实现盈亏)
            ├── friction (摩擦成本)
            └── fee (手续费)
```

## 使用示例

### 配置仓位参数

```python
from leek_core.models import PositionConfig
from decimal import Decimal

config = PositionConfig(
    init_amount=Decimal("100000"),      # 初始资金10万
    max_amount=Decimal("5000"),         # 单次最大5000
    max_ratio=Decimal("0.05"),          # 单次最大5%
    max_strategy_amount=Decimal("30000"), # 单策略最大3万
    max_strategy_ratio=Decimal("0.3"),   # 单策略最大30%
    max_symbol_amount=Decimal("20000"),  # 单品种最大2万
    max_symbol_ratio=Decimal("0.2"),     # 单品种最大20%
    default_leverage=1,
)
```

### 查询仓位

```python
# 通过引擎获取仓位状态
position_state = engine.get_position_state()

# 查询特定策略的仓位
positions = engine.position_tracker.find_position(
    strategy_id="strategy-001"
)

# 查询特定品种的仓位
positions = engine.position_tracker.find_position(
    symbol="BTC",
    quote_currency="USDT"
)

# 获取策略已使用资金
used_amount, used_ratio = engine.position_tracker.get_strategy_used("strategy-001")
print(f"已用资金: {used_amount}, 已用比例: {used_ratio}")
```

### 状态持久化

```python
# 获取状态
state = engine.portfolio.get_state()
# 输出:
# {
#     'total_value': '105000.00',
#     'position_value': '25000.00',
#     'pnl': '5000.00',
#     'friction': '-50.00',
#     'fee': '-100.00',
#     'profit': '4850.00',
#     'capital': {...},
#     'position': {...},
#     'risk': {...}
# }

# 保存状态
import json
with open("position_state.json", "w") as f:
    json.dump(state, f)

# 恢复状态
with open("position_state.json", "r") as f:
    saved_state = json.load(f)

# 通过配置传入
position_config = PositionConfig(
    init_amount=Decimal("100000"),
    ...,
    data=saved_state,  # 传入保存的状态
)
```

## 资金控制逻辑

### 多层级限制

```text
可投入金额 = min(
    max_amount,                           # 单次最大
    max_strategy_amount - strategy_used,  # 策略剩余
    max_symbol_amount - symbol_used,      # 品种剩余
    available_balance                     # 可用余额
)

可投入比例 = min(
    max_ratio,                           # 单次最大比例
    max_strategy_ratio - strategy_ratio, # 策略剩余比例
    max_symbol_ratio - symbol_ratio      # 品种剩余比例
)
```

### 示例场景

```python
# 配置
config = PositionConfig(
    init_amount=Decimal("100000"),
    max_amount=Decimal("10000"),
    max_ratio=Decimal("0.1"),
    max_strategy_amount=Decimal("30000"),
    max_strategy_ratio=Decimal("0.3"),
    max_symbol_amount=Decimal("20000"),
    max_symbol_ratio=Decimal("0.2"),
)

# 场景1：新策略首次开仓
# 可投入 = min(10000, 30000, 20000, 100000) = 10000

# 场景2：策略已持仓25000
# 可投入 = min(10000, 30000-25000=5000, 20000, 75000) = 5000

# 场景3：品种已持仓18000
# 可投入 = min(10000, 30000, 20000-18000=2000, 82000) = 2000
```

## 虚拟仓位

虚拟仓位用于处理部分平仓场景，避免创建过多实际订单：

```python
@dataclass
class VirtualPosition:
    """虚拟仓位"""
    virtual_id: str
    sz: Decimal
    ratio: Decimal
    open_time: datetime
```

使用场景：
- 策略发出部分平仓信号时，先创建虚拟仓位
- 虚拟仓位汇总后一次性发送实际订单
- 减少交易所API调用次数

## 最佳实践

### 1. 合理设置仓位限制

```python
# 保守配置（适合新手）
config = PositionConfig(
    max_ratio=Decimal("0.02"),          # 单次2%
    max_strategy_ratio=Decimal("0.1"),   # 单策略10%
    max_symbol_ratio=Decimal("0.05"),    # 单品种5%
)

# 激进配置（适合经验丰富）
config = PositionConfig(
    max_ratio=Decimal("0.1"),           # 单次10%
    max_strategy_ratio=Decimal("0.5"),   # 单策略50%
    max_symbol_ratio=Decimal("0.3"),     # 单品种30%
)
```

### 2. 监控资金使用情况

```python
def monitor_capital():
    state = engine.portfolio.get_state()
    capital = state['capital']
    
    usage_rate = 1 - Decimal(capital['available']) / Decimal(capital['total'])
    
    if usage_rate > Decimal("0.8"):
        logger.warning(f"资金使用率较高: {usage_rate:.1%}")
```

### 3. 定期保存状态

```python
import schedule

def save_state():
    state = engine.portfolio.get_state()
    with open(f"state_{datetime.now():%Y%m%d_%H%M}.json", "w") as f:
        json.dump(state, f)

schedule.every(1).hour.do(save_state)
```

## 相关模块

- [引擎架构](20-engine.md) - Portfolio 在引擎中的使用
- [策略模块](01-strategy.md) - 信号生成
- [执行器](23-executor.md) - 订单执行
- [风险控制](04-risk.md) - RiskManager 详情
