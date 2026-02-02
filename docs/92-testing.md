# 92 测试指南

## 概述

本指南介绍 Leek Core 的测试规范和最佳实践，包括单元测试、集成测试和策略回测验证。项目使用 `unittest` 作为测试框架，测试代码位于 `src/leek_core/tests/` 目录下。

## 测试目录结构

```text
src/leek_core/tests/
├── __init__.py
├── adapts/                    # 适配器测试
│   ├── test_okx_adapter.py
│   └── test_rate_limit.py
├── analysis/                  # 分析模块测试
│   └── test_performance.py
├── backtest/                  # 回测模块测试
│   └── __init__.py
├── data/                      # 数据源测试
│   ├── test_binance_source.py
│   ├── test_clickhouse_source.py
│   ├── test_gate_source.py
│   └── test_okx_source.py
├── engine/                    # 引擎测试
│   ├── test_engine.py
│   ├── test_event_source_flow.py
│   ├── test_grpc.py
│   └── test_indicator_view.py
├── event/                     # 事件系统测试
│   ├── test_bus.py
│   └── test_position_order.py
├── executor/                  # 执行器测试
│   ├── test_gate_executor.py
│   └── test_okx_executor.py
├── indicators/                # 指标测试
│   ├── test_atr.py
│   ├── test_chan.py
│   ├── test_macd.py
│   └── test_ma.py
├── ml/                        # 机器学习测试
│   ├── test_factor.py
│   └── test_gru_trainer.py
├── models/                    # 数据模型测试
│   ├── test_data.py
│   └── test_order.py
└── utils/                     # 工具类测试
    ├── test_logging.py
    └── test_retry.py
```

## 运行测试

### 运行全部测试

```bash
# 在项目根目录
cd leek-core
python -m pytest src/leek_core/tests/

# 或使用 unittest
python -m unittest discover -s src/leek_core/tests/ -p "test_*.py"
```

### 运行单个测试文件

```bash
python -m pytest src/leek_core/tests/engine/test_engine.py -v
```

### 运行单个测试方法

```bash
python -m pytest src/leek_core/tests/engine/test_engine.py::TestEngine::test_on_data -v
```

### 带覆盖率运行

```bash
python -m pytest src/leek_core/tests/ --cov=leek_core --cov-report=html
```

## 单元测试

### 指标测试示例

```python
import unittest
from decimal import Decimal
from leek_core.indicators import MA, MACD, RSI
from leek_core.models import KLine, TimeFrame

class TestMA(unittest.TestCase):
    """移动平均线测试"""
    
    def setUp(self):
        """每个测试方法前执行"""
        self.ma = MA(period=5)
    
    def tearDown(self):
        """每个测试方法后执行"""
        pass
    
    def _create_kline(self, close: float, is_finished: bool = True) -> KLine:
        """创建测试用K线"""
        return KLine(
            symbol="BTC",
            market="test",
            quote_currency="USDT",
            open=Decimal(str(close)),
            close=Decimal(str(close)),
            high=Decimal(str(close)),
            low=Decimal(str(close)),
            volume=Decimal("100"),
            amount=Decimal("10000"),
            start_time=1704067200000,
            end_time=1704070800000,
            current_time=1704070800000,
            timeframe=TimeFrame.H1,
            is_finished=is_finished,
        )
    
    def test_insufficient_data(self):
        """测试数据不足时返回None"""
        for i in range(4):
            result = self.ma.update(self._create_kline(100 + i))
            self.assertIsNone(result)
    
    def test_calculation(self):
        """测试计算正确性"""
        # 输入5个价格
        prices = [100, 102, 104, 103, 101]
        for price in prices[:-1]:
            self.ma.update(self._create_kline(price))
        
        result = self.ma.update(self._create_kline(prices[-1]))
        
        expected = sum(prices) / len(prices)
        self.assertEqual(float(result), expected)
    
    def test_sliding_window(self):
        """测试滑动窗口"""
        prices = [100, 102, 104, 103, 101, 105]
        for price in prices[:-1]:
            self.ma.update(self._create_kline(price))
        
        result = self.ma.update(self._create_kline(prices[-1]))
        
        # 应该使用最近5个价格
        expected = sum(prices[-5:]) / 5
        self.assertEqual(float(result), expected)


class TestMACD(unittest.TestCase):
    """MACD测试"""
    
    def setUp(self):
        self.macd = MACD(fast_period=3, slow_period=5, moving_period=2)
    
    def test_output_format(self):
        """测试输出格式"""
        kline = self._create_kline(100)
        result = self.macd.update(kline)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (DIF, DEA)
```

### 策略测试示例

```python
import unittest
from decimal import Decimal
from leek_core.strategy import CTAStrategy
from leek_core.models import KLine, PositionSide, TimeFrame

class TestMyStrategy(unittest.TestCase):
    """策略测试"""
    
    def setUp(self):
        self.strategy = MyStrategy(fast=3, slow=5)
    
    def test_should_open_long(self):
        """测试做多信号"""
        # 模拟价格上涨趋势
        prices = [100, 101, 102, 103, 104, 105]
        for price in prices:
            kline = self._create_kline(price)
            self.strategy.on_kline(kline)
        
        result = self.strategy.should_open()
        self.assertEqual(result, PositionSide.LONG)
    
    def test_should_open_short(self):
        """测试做空信号"""
        # 模拟价格下跌趋势
        prices = [105, 104, 103, 102, 101, 100]
        for price in prices:
            kline = self._create_kline(price)
            self.strategy.on_kline(kline)
        
        result = self.strategy.should_open()
        self.assertEqual(result, PositionSide.SHORT)
    
    def test_should_close(self):
        """测试平仓信号"""
        # 先建立做多仓位
        prices = [100, 101, 102, 103, 104, 105]
        for price in prices:
            self.strategy.on_kline(self._create_kline(price))
        
        # 模拟价格反转
        reverse_prices = [104, 103, 102, 101]
        for price in reverse_prices:
            self.strategy.on_kline(self._create_kline(price))
        
        result = self.strategy.should_close(PositionSide.LONG)
        self.assertTrue(result)
    
    def test_state_persistence(self):
        """测试状态持久化"""
        # 更新一些数据
        for price in [100, 101, 102]:
            self.strategy.on_kline(self._create_kline(price))
        
        # 保存状态
        state = self.strategy.get_state()
        
        # 创建新实例并恢复
        new_strategy = MyStrategy(fast=3, slow=5)
        new_strategy.load_state(state)
        
        # 验证状态一致
        self.assertEqual(
            self.strategy._prices, 
            new_strategy._prices
        )
```

### 事件总线测试示例

```python
import unittest
import threading
import time
from leek_core.event import EventBus, Event, EventType

class TestEventBus(unittest.TestCase):
    """事件总线测试"""
    
    def setUp(self):
        self.event_bus = EventBus(max_workers=5)
        self.received_events = []
        self.lock = threading.Lock()
    
    def tearDown(self):
        self.event_bus.shutdown()
    
    def _callback(self, event):
        with self.lock:
            self.received_events.append(event)
    
    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        self.event_bus.subscribe_event(EventType.DATA_RECEIVED, self._callback)
        
        event = Event(EventType.DATA_RECEIVED, data={"test": "data"})
        self.event_bus.publish_event(event)
        
        # 等待异步处理
        time.sleep(0.1)
        
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].data, {"test": "data"})
    
    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        events1 = []
        events2 = []
        
        self.event_bus.subscribe_event(
            EventType.DATA_RECEIVED, 
            lambda e: events1.append(e)
        )
        self.event_bus.subscribe_event(
            EventType.DATA_RECEIVED, 
            lambda e: events2.append(e)
        )
        
        event = Event(EventType.DATA_RECEIVED, data="test")
        self.event_bus.publish_event(event)
        
        time.sleep(0.1)
        
        self.assertEqual(len(events1), 1)
        self.assertEqual(len(events2), 1)
    
    def test_unsubscribe(self):
        """测试取消订阅"""
        self.event_bus.subscribe_event(EventType.DATA_RECEIVED, self._callback)
        self.event_bus.unsubscribe_event(EventType.DATA_RECEIVED, self._callback)
        
        event = Event(EventType.DATA_RECEIVED, data="test")
        self.event_bus.publish_event(event)
        
        time.sleep(0.1)
        
        self.assertEqual(len(self.received_events), 0)
```

## 集成测试

### 引擎集成测试

```python
import unittest
from decimal import Decimal
from leek_core.engine import SimpleEngine
from leek_core.models import (
    PositionConfig, LeekComponentConfig, StrategyConfig, KLine
)
from leek_core.event import EventBus

class TestEngineIntegration(unittest.TestCase):
    """引擎集成测试"""
    
    def setUp(self):
        self.event_bus = EventBus()
        self.engine = SimpleEngine(
            instance_id="test-engine",
            name="测试引擎",
            position_config=PositionConfig(
                init_amount=Decimal("100000"),
                max_amount=Decimal("10000"),
                max_ratio=Decimal("0.1"),
                max_strategy_amount=Decimal("50000"),
                max_strategy_ratio=Decimal("0.5"),
                max_symbol_amount=Decimal("25000"),
                max_symbol_ratio=Decimal("0.25"),
            ),
            event_bus=self.event_bus,
        )
    
    def tearDown(self):
        try:
            self.engine.on_stop()
        except:
            pass
    
    def test_add_remove_strategy(self):
        """测试添加和移除策略"""
        config = LeekComponentConfig(
            instance_id="strategy-001",
            name="测试策略",
            cls="leek_core.tests.engine.test_engine.CTAStrategyTest",
            config=StrategyConfig(
                data_source_configs=[],
                info_fabricator_configs=[],
                strategy_config={"period": 10},
                strategy_position_config=None,
                risk_policies=[],
            ),
        )
        
        self.engine.add_strategy(config)
        
        state = self.engine.get_strategy_state()
        self.assertIn("strategy-001", state)
        
        self.engine.remove_strategy("strategy-001")
        
        state = self.engine.get_strategy_state()
        self.assertNotIn("strategy-001", state)
    
    def test_data_flow(self):
        """测试数据流"""
        signals = []
        
        def on_signal(event):
            signals.append(event.data)
        
        self.event_bus.subscribe_event(EventType.STRATEGY_SIGNAL, on_signal)
        
        # 添加策略
        self.engine.add_strategy(...)
        self.engine.on_start()
        
        # 模拟数据
        for i in range(20):
            kline = create_test_kline(price=100 + i)
            self.engine.on_data(kline)
        
        # 验证信号产生
        self.assertGreater(len(signals), 0)
```

### 回测集成测试

```python
import unittest
from leek_core.backtest import run_backtest

class TestBacktestIntegration(unittest.TestCase):
    """回测集成测试"""
    
    def test_simple_backtest(self):
        """测试简单回测"""
        config = {
            "strategy_class": "leek_core.tests.engine.test_engine.CTAStrategyTest",
            "strategy_params": {"period": 10},
            "datasource_class": "leek_core.data.ClickHouseSource",
            "datasource_config": {
                "host": "localhost",
                "database": "test_data",
            },
            "executor_class": "leek_core.executor.BacktestExecutor",
            "symbol": "BTC",
            "quote_currency": "USDT",
            "ins_type": 2,
            "timeframe": "1h",
            "start_time": 1704067200000,
            "end_time": 1704672000000,
            "initial_balance": 10000,
        }
        
        result = run_backtest(config)
        
        # 验证结果结构
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.equity_curve)
        self.assertGreater(len(result.equity_curve), 0)
    
    def test_backtest_metrics(self):
        """测试回测指标计算"""
        config = {...}
        result = run_backtest(config)
        
        metrics = result.metrics
        
        # 验证指标范围
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
        self.assertLessEqual(metrics.max_drawdown, 0)
```

## Mock 和 Stub

### Mock 数据源

```python
from unittest.mock import Mock, AsyncMock
import pandas as pd

class MockDataSource:
    """模拟数据源"""
    
    def __init__(self, klines: list[dict]):
        self.klines = klines
        self._index = 0
    
    async def get_klines(self, symbol, timeframe, start, end):
        return pd.DataFrame(self.klines)
    
    def get_history_data(self, **kwargs):
        for kline_data in self.klines:
            yield KLine(**kline_data)


# 使用
mock_source = MockDataSource([
    {"open": 100, "close": 101, "high": 102, "low": 99, ...},
    {"open": 101, "close": 103, "high": 104, "low": 100, ...},
])
```

### Mock 执行器

```python
class MockExecutor:
    """模拟执行器"""
    
    def __init__(self):
        self.orders = []
        self.callback = None
    
    def send_order(self, order):
        self.orders.append(order)
        
        # 模拟立即成交
        if self.callback:
            self.callback(OrderUpdateMessage(
                order_id=order.order_id,
                order_status=OrderStatus.FILLED,
                execution_price=order.order_price,
                sz=order.sz,
                settle_amount=order.order_amount,
                fee=Decimal("0"),
                pnl=Decimal("0"),
                finish_time=datetime.now(),
            ))
    
    def cancel_order(self, order_id, symbol, **kwargs):
        pass
```

## 测试夹具

### conftest.py

```python
# tests/conftest.py
import pytest
from decimal import Decimal
from leek_core.models import KLine, TimeFrame

@pytest.fixture
def sample_kline():
    """提供示例K线"""
    return KLine(
        symbol="BTC",
        market="test",
        quote_currency="USDT",
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

@pytest.fixture
def kline_series():
    """提供K线序列"""
    base_time = 1704067200000
    prices = [100, 101, 102, 101, 103, 105, 104, 106, 108, 107]
    
    klines = []
    for i, price in enumerate(prices):
        klines.append(KLine(
            symbol="BTC",
            market="test",
            open=Decimal(str(price)),
            close=Decimal(str(price)),
            high=Decimal(str(price + 1)),
            low=Decimal(str(price - 1)),
            volume=Decimal("100"),
            amount=Decimal("10000"),
            start_time=base_time + i * 3600000,
            end_time=base_time + (i + 1) * 3600000,
            current_time=base_time + (i + 1) * 3600000,
            timeframe=TimeFrame.H1,
            is_finished=True,
        ))
    return klines

@pytest.fixture
def event_bus():
    """提供事件总线"""
    from leek_core.event import EventBus
    bus = EventBus()
    yield bus
    bus.shutdown()
```

## 最佳实践

### 1. 测试命名

```python
# 好的命名
def test_should_open_returns_long_when_fast_ma_crosses_above_slow_ma(self):
    ...

def test_calculate_returns_none_when_insufficient_data(self):
    ...

# 避免
def test_1(self):
    ...
```

### 2. 单一职责

```python
# 好的做法：每个测试只验证一件事
def test_ma_calculation_accuracy(self):
    # 只测试计算准确性
    ...

def test_ma_handles_insufficient_data(self):
    # 只测试数据不足情况
    ...
```

### 3. 使用断言消息

```python
self.assertEqual(
    result, 
    expected, 
    f"MA计算错误：期望 {expected}，实际 {result}"
)
```

### 4. 测试边界条件

```python
def test_period_of_one(self):
    ma = MA(period=1)
    result = ma.update(kline)
    self.assertEqual(result, kline.close)

def test_zero_volume(self):
    kline = create_kline(volume=0)
    # 验证能正常处理
```

## 相关文档

- [组件开发指南](91-component-guide.md) - 组件开发
- [回测系统](30-backtest.md) - 回测验证
