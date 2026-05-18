# 05 数据流与生命周期

> 描述 Leek Core 中从行情数据到订单成交的完整事件流转。架构分层见 [`90-architecture.md`](./90-architecture.md);事件总线机制见 [`22-event-bus.md`](./22-event-bus.md);模块职责见 [`leek-core-modules.md`](./leek-core-modules.md)。

---

## 1. 双模式数据流总览

| 模式 | 入口 | 引擎 | 出口 |
|------|------|------|------|
| **实盘** | `DataSource(WebSocket)` | `SimpleEngine / GrpcEngine` | `Executor(OKX / Binance / Gate)` |
| **回测** | `DataSource(ClickHouse)` | `SimpleEngine` | `BacktestExecutor`(本地撮合) |

两种模式**事件流完全一致**:`Engine` / `StrategyManager` / `Portfolio` / `EventBus` 不感知模式差异。

---

## 2. 实盘数据流(K 线进入到下单)

```
交易所 WebSocket
       │
       ▼
┌─────────────┐    KLine     ┌─────────────────────────────────┐
│ DataSource  │ ──────────►  │  Engine.on_data(data)            │
└─────────────┘              │  1. BacktestExecutor.on_bar(回测)│
                             │  2. strategy_manager.process_data│
                             │  3. portfolio.position_tracker   │
                             │     .on_data(data) 更新价格      │
                             └────────────┬─────────────────────┘
                                          │ 策略产 Signal
                                          ▼
                             ┌──────────────────────────────┐
                             │ Engine._on_signal(signal)    │
                             │ 1. publish STRATEGY_SIGNAL   │
                             │ 2. portfolio.process_signal  │
                             │ 3. portfolio.risk_manager    │
                             │    .evaluate_risk            │
                             │ 4. portfolio.capital_account │
                             │    .freeze_amount            │
                             │ 5. executor_manager          │
                             │    .handle_order             │
                             └────────────┬─────────────────┘
                                          │
                                          ▼
                                  ┌──────────────────┐
                                  │ ExecutorContext  │
                                  │ send_order       │ → 交易所
                                  └──────────────────┘
```

**关键点**:
- `Engine` 是 Façade,所有子组件统一通过 `engine.portfolio.{position_tracker, capital_account, risk_manager}` 访问
- 风控(`evaluate_risk`)在 freeze 之前,可对所有 Signal 一视同仁拒绝
- freeze 失败时 wrapper 收到 `exec_update` 信号回收,**不进入 Executor**

---

## 3. 订单回调流转

```
交易所推送 / 回测撮合
        │
        ▼
┌────────────────────┐
│ Executor           │ _trade_callback(OrderUpdateMessage)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐    EventType.ORDER_UPDATED
│ ExecutorContext    │ ────────────────────────────► EventBus
└────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Engine             │ on_order_update(event)
                       └─────────┬──────────┘
                                 │
                  ┌──────────────┼─────────────────────┐
                  ▼              ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
        │ portfolio.       │  │ executor_manager │  │ strategy_manager     │
        │ order_update     │  │ .order_update    │  │ .exec_update(回收 sig)│
        │ (持仓 / 资金)    │  │ (回收 exec_ctx)  │  │ .dispatch_event       │
        └──────────────────┘  └──────────────────┘  │ (转发 Order 给 wrapper)│
                                                    └──────────────────────┘
```

**Engine.on_order_update 关键改造**:
- **中间态(SUBMITTED / PARTIALLY_FILLED)**: 仅调 `dispatch_event` 转给 wrapper(维护 pending_index)
- **终态(FILLED / CANCELED / REJECTED / EXPIRED / ERROR)**:
  - `portfolio.order_update`: unfreeze 资金 + 更新持仓
  - `strategy_manager.exec_update`: 回收 Signal 容器,wrapper 状态机归一
  - `strategy_manager.dispatch_event`: 调 `Strategy.on_order_update` 触发用户 callback

---

## 4. 策略意图 → 信号生命周期

```
on_data(KLine)
     │
     ▼
┌──────────────────────┐
│ Strategy 内部计算    │   on_data 更新指标
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ StrategyWrapper 状态机                │
│   READY ──should_open()──► ENTERING  │
│   HOLDING ──close()──► EXITING       │
│   STOPPING ──stopping_handler        │
└────────┬─────────────────────────────┘
         │
         │ List[StrategyCommand] / List[StrategyAction]
         ▼
┌──────────────────────────────────────┐
│ Wrapper.materialize                  │
│ ┌────────────────┐ ┌──────────────┐ │
│ │ diff 引擎      │ │ pending_index │ │
│ │ (限价单改/撤)  │ │  本地挂单跟踪 │ │
│ └────────────────┘ └──────────────┘ │
└────────┬─────────────────────────────┘
         │ List[Asset]
         ▼
   Signal(包含 1 或多笔 Asset 意图)
         │
         ▼
   Engine._on_signal(signal)  ──► 原 freeze + executor 链路
```

**多笔意图(网格 / 分批)**:`should_open` 可返回 `List[StrategyCommand]`,每笔 Asset 独立携带 `order_type / price / expire_bars`,共用一个 Signal 走完整 freeze 链路。

详细策略开发模型见 [`01-strategy.md`](./01-strategy.md)。

---

## 5. 挂单生命周期(新增)

### 5.1 限价单挂出 → 成交

```
策略意图
   │  StrategyCommand(LimitOrder, price, expire_bars, extra={tag})
   ▼
Wrapper.materialize → Asset(order_type=LIMIT, price=99)
   ▼
Portfolio.process_signal → freeze(按 order_price 计算)
   ▼
Executor.send_order
   │
   ├─── 实盘: 交易所 ACK → ORDER_UPDATED(SUBMITTED)
   │                                 │
   │                                 ▼
   │                  Wrapper.pending_index.add(order)
   │                  (中间态走 strategy.on_event 兜底)
   │
   └─── 回测: 进入 BacktestExecutor._pending_orders
                   │
                   ▼
        新 bar 到达 → Engine.on_data → BacktestExecutor.on_bar(data)
                                      │
                                      ▼
                           按 bar OHLC 撮合(strict 模式):
                             BUY  LIMIT: bar.low  <= price → fill@min(price, bar.open)
                             SELL LIMIT: bar.high >= price → fill@max(price, bar.open)

成交回调 ORDER_UPDATED(FILLED) → Engine.on_order_update
                                   │
                                   ▼
                         Wrapper.pending_index.remove(order)
                         Strategy.on_order_update(order)
```

### 5.2 限价单撤单

策略 / Wrapper 通过发布 `ORDER_CANCEL_REQUEST` 事件触发,ExecutorManager 订阅后路由到对应 Executor。

```
撤单来源:
  - 策略 on_order_update / on_position_update 返回 CancelCommand
  - Wrapper diff 引擎检测到挂单需"撤旧挂新"
  - Wrapper._check_expired 检测到 expire_bars 到期
       │
       ▼
EventBus.publish(ORDER_CANCEL_REQUEST, data=Order)
       │
       ▼
ExecutorManager.on_cancel_request
       │ 按 order.executor_id 路由
       ▼
ExecutorContext.cancel_order(target_id, symbol, leek_order_id=...)
       │
       ├─── 实盘: 交易所确认 → ORDER_UPDATED(CANCELED)
       └─── 回测: BacktestExecutor 从 _pending_orders 移除,
                  推 ORDER_UPDATED(CANCELED)
```

### 5.3 意图 diff 引擎(每根 bar 自动撤改)

当策略每根 bar 重复返回意图时,Wrapper 不会盲目重新挂单,而是与 `pending_index` 中现有挂单 diff:

| 现有挂单 vs 新意图(按 `extra` 业务键匹配) | 行为 |
|------------------------------|------|
| 同 key + price/ratio 一致 | 保留,跳过 |
| 同 key + price/ratio 变化 | 撤旧 + 挂新(改价) |
| 现有有,新意图无 | 撤旧 |
| 现有无,新意图有 | 新挂 |

业务键由 `extra["tag"]` 或 `extra` 全部字段排序决定。详细规则见 [`01-strategy.md`](./01-strategy.md)。

### 5.4 expire_bars 自动过期

```
StrategyCommand(LimitOrder, expire_bars=N)
   │
   ▼
Wrapper._make_asset 写入 Asset.extra["_submit_bar"] / ["_expire_bars"]
   ↓
执行链路透传到 Order.extra
   │
   ▼
每根 bar 到达 → Wrapper._check_expired 扫描 pending_index
   if bar_counter - _submit_bar >= _expire_bars:
       order.extra["_expired"] = True
       publish ORDER_CANCEL_REQUEST
   │
   ▼
Executor.cancel_order → ORDER_UPDATED(CANCELED)
   │
   ▼
Strategy.on_order_update 收到 order.extra["_expired"]=True
```

---

## 6. 事件类型总表

| 事件类型 | 来源 | 订阅者 | 说明 |
|---------|------|--------|------|
| `DATA_RECEIVED` | DataSource | Engine | 行情数据 |
| `DATA_SOURCE_SUBSCRIBE` | Strategy | DataSource | 订阅请求 |
| `STRATEGY_SIGNAL` | Engine | (扩展用) | 策略产生信号(审计) |
| `STRATEGY_SIGNAL_MANUAL` | Wrapper / 引擎 | Engine | Tier 2 callback 产生的信号,触发 `_on_signal` |
| `STRATEGY_SIGNAL_FINISH` | StrategyContext | (扩展用) | 信号完成审计 |
| `EXEC_ORDER_CREATED` | Engine | (扩展用) | freeze 后 ExecutionContext 创建 |
| `EXEC_ORDER_UPDATED` | Engine | (扩展用) | ExecutionContext 完成 |
| `ORDER_CREATED` | ExecutorContext | (扩展用) | 订单发送到交易所 |
| `ORDER_UPDATED` | ExecutorContext | Engine | 订单状态变化(含中间态) |
| `ORDER_CANCEL_REQUEST` | Wrapper / 策略 | ExecutorManager | 撤单请求(本次新增) |
| `PENDING_ORDER_RECONCILE` | Wrapper(启动) | ExecutorManager | 重启后挂单对账(预留) |
| `POSITION_UPDATE` | PositionTracker | Wrapper | 持仓变化 |
| `POSITION_POLICY_ADD/DEL` | Engine | RiskManager | 全局风控策略增删 |
| `POSITION_INIT` | Portfolio | (扩展用) | 仓位初始化 |
| `TRANSACTION` | CapitalAccount | (扩展用) | 资金流水审计 |
| `RISK_TRIGGERED` | Wrapper / RiskManager | (扩展用) | 风控触发审计 |

---

## 7. 模块依赖关系

```
                            ┌─────────┐
                            │  utils  │
                            └────┬────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   ┌─────────┐             ┌─────────┐             ┌─────────┐
   │ models  │             │  event  │             │  base   │
   └────┬────┘             └────┬────┘             └────┬────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
   ┌─────────┐            ┌─────────┐            ┌─────────┐
   │  data   │            │ strategy│            │executor │
   └────┬────┘            └────┬────┘            └────┬────┘
        │                      │                      │
        │                ┌─────┴─────┐                │
        │                ▼           ▼                │
        │          ┌─────────┐ ┌─────────┐            │
        │          │indicator│ │sub_strat│            │
        │          └─────────┘ └─────────┘            │
        │                                             │
        └────────────────────┬────────────────────────┘
                             │
                             ▼
                       ┌─────────┐
                       │position │ (含 Portfolio Façade)
                       └────┬────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │  risk   │        │ policy  │        │ manager │
   └─────────┘        └─────────┘        └─────────┘
                            │
                            ▼
                       ┌─────────┐
                       │ engine  │
                       └────┬────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │backtest │        │  alarm  │        │   ml    │
   └─────────┘        └─────────┘        └─────────┘
```

---

## 相关文档

- [`00-overview.md`](./00-overview.md) — Leek Core 项目概览
- [`90-architecture.md`](./90-architecture.md) — 分层架构与设计原则
- [`leek-core-modules.md`](./leek-core-modules.md) — 各模块详细职责
- [`22-event-bus.md`](./22-event-bus.md) — 事件总线实现
- [`20-engine.md`](./20-engine.md) — 引擎实现
- [`01-strategy.md`](./01-strategy.md) — 策略开发模型(含意图 diff / pending_index)
- [`23-executor.md`](./23-executor.md) — 执行器与回测撮合
- [`03-position.md`](./03-position.md) — Portfolio / CapitalAccount / PositionTracker
- [`04-risk.md`](./04-risk.md) — 风控策略
