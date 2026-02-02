# 12 技术指标模块

## 概述

技术指标模块提供丰富的量化交易技术指标实现，支持流式计算（实时更新）模式。所有指标继承自基类 `T`，提供统一的 `update()` 接口，便于在策略中集成使用。

## 指标分类

### 趋势指标

| 指标 | 类名 | 说明 |
|------|------|------|
| 移动平均线 | `MA`, `EMA`, `SMA` | 简单/指数移动平均 |
| MACD | `MACD` | 指数平滑异同移动平均 |
| 布林带 | `BOLL` | 布林带指标 |
| 唐奇安通道 | `Donchian` | 唐奇安通道突破 |
| 一目均衡表 | `IchimokuCloud` | 云图指标 |
| SuperTrend | `SuperTrend` | 超级趋势指标 |
| SAR | `SAR` | 抛物线止损转向 |

### 震荡指标

| 指标 | 类名 | 说明 |
|------|------|------|
| RSI | `RSI` | 相对强弱指标 |
| KDJ | `KDJ` | 随机指标 |
| CCI | `CCI` | 顺势指标 |
| WR | `WR` | 威廉指标 |
| BIAS | `BIAS` | 乖离率 |
| DM/ADX | `DM` | 动向指标 |

### 波动率指标

| 指标 | 类名 | 说明 |
|------|------|------|
| ATR | `ATR` | 平均真实波幅 |
| 布林带宽 | `BOLL` | 布林带宽度 |
| Keltner | `Keltner` | 肯特纳通道 |
| Chaikin Volatility | `ChaikinVolatility` | 蔡金波动率 |

### 成交量指标

| 指标 | 类名 | 说明 |
|------|------|------|
| 量价分布 | `VolumeProfile` | 成交量分布 |

### 特色指标

| 指标 | 类名 | 说明 |
|------|------|------|
| 缠论 | `Chan` | 缠中说禅分析体系 |
| CZSC | `CZSC` | 缠中说禅指标 |
| Hurst指数 | `Hurst` | 赫斯特指数 |
| Lyapunov指数 | `Lyapunov` | 李雅普诺夫指数 |
| RSRS | `RSRS` | 阻力支撑相对强度 |
| DeMark | `DeMark` | 德马克指标 |
| Elder Impulse | `ElderImpulse` | 艾尔德动能系统 |
| Gann HiLo | `GannHiLo` | 江恩高低指标 |

## 基础用法

### 指标基类 `T`

```python
class T:
    """技术指标基类"""
    
    def __init__(self, max_cache: int = 10):
        self.cache = []  # 缓存历史值
        self.max_cache = max_cache
    
    def update(self, data: KLine) -> Any:
        """
        更新指标值
        
        参数:
            data: K线数据
            
        返回:
            计算结果（类型由具体指标决定）
        """
        raise NotImplementedError()
```

### 使用示例

```python
from leek_core.indicators import MA, MACD, RSI, BOLL, ATR
from leek_core.models import KLine

# 创建指标实例
ma20 = MA(period=20)
macd = MACD(fast_period=12, slow_period=26, moving_period=9)
rsi = RSI(period=14)
boll = BOLL(period=20, std_dev=2)
atr = ATR(period=14)

# 在策略中使用
class MyStrategy(CTAStrategy):
    def __init__(self):
        super().__init__()
        self.ma20 = MA(20)
        self.ma50 = MA(50)
        self.macd = MACD()
        self.rsi = RSI(14)
    
    def on_kline(self, kline: KLine):
        # 更新指标
        fast_ma = self.ma20.update(kline)
        slow_ma = self.ma50.update(kline)
        dif, dea = self.macd.update(kline)
        rsi_value = self.rsi.update(kline)
        
        # 存储到K线动态属性（可选）
        kline.ma20 = fast_ma
        kline.ma50 = slow_ma
        kline.dif = dif
        kline.dea = dea
        kline.rsi = rsi_value
```

## 常用指标详解

### 移动平均线 (MA)

```python
class MA(T):
    """简单移动平均线"""
    
    def __init__(self, period: int = 20, ma_type: str = "close"):
        """
        参数:
            period: 周期
            ma_type: 计算类型 ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4")
        """
        ...
    
    def update(self, data: KLine) -> Decimal | None:
        """返回MA值，数据不足时返回None"""
        ...


class EMA(T):
    """指数移动平均线"""
    
    def __init__(self, period: int = 20):
        ...


class SMA(T):
    """平滑移动平均线"""
    
    def __init__(self, period: int = 20):
        ...
```

**使用示例：**

```python
from leek_core.indicators import MA, EMA

ma20 = MA(period=20)
ema12 = EMA(period=12)

for kline in klines:
    ma_value = ma20.update(kline)
    ema_value = ema12.update(kline)
    
    if ma_value and ema_value:
        if ema_value > ma_value:
            print("短期均线在长期均线之上")
```

### MACD

```python
class MACD(T):
    """指数平滑异同移动平均"""
    
    def __init__(
        self, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        moving_period: int = 9,
        ma: type = MA,
        max_cache: int = 10
    ):
        """
        参数:
            fast_period: 快线周期
            slow_period: 慢线周期
            moving_period: 信号线周期
            ma: 均线类型
        """
        ...
    
    def update(self, data: KLine) -> tuple[Decimal, Decimal] | tuple[None, None]:
        """
        返回:
            (DIF, DEA) 元组，数据不足时返回 (None, None) 或 (dif, None)
        """
        ...
```

**MACD 背离检测：**

```python
class Divergence:
    """MACD背离检测"""
    
    def __init__(
        self,
        divergence_threshold: int = 1,    # 背离阈值
        allow_cross_zero: bool = True,    # 允许穿过零轴
        close_price_divergence: bool = True,   # 收盘价背离
        peak_price_divergence: bool = True,    # 峰值背离
        m_area_divergence: bool = True,        # 能量柱面积背离
        dea_pull_back: bool = True,            # DEA线拉回
        pull_back_rate: float = 0,             # 拉回率
        divergence_rate: float = 0.9,          # 背离率
    ):
        ...
    
    def is_top_divergence(self, data) -> bool:
        """检测顶背离"""
        ...
    
    def is_bottom_divergence(self, data) -> bool:
        """检测底背离"""
        ...
```

### RSI

```python
class RSI(T):
    """相对强弱指标"""
    
    def __init__(self, period: int = 14):
        """
        参数:
            period: RSI周期
        """
        ...
    
    def update(self, data: KLine) -> float | None:
        """返回RSI值 (0-100)"""
        ...
```

**使用示例：**

```python
from leek_core.indicators import RSI

rsi = RSI(period=14)

for kline in klines:
    rsi_value = rsi.update(kline)
    
    if rsi_value is not None:
        if rsi_value < 30:
            print("超卖区域")
        elif rsi_value > 70:
            print("超买区域")
```

### 布林带 (BOLL)

```python
class BOLL(T):
    """布林带指标"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        参数:
            period: 周期
            std_dev: 标准差倍数
        """
        ...
    
    def update(self, data: KLine) -> tuple[Decimal, Decimal, Decimal] | tuple[None, None, None]:
        """
        返回:
            (上轨, 中轨, 下轨)
        """
        ...
```

### ATR

```python
class ATR(T):
    """平均真实波幅"""
    
    def __init__(self, period: int = 14):
        ...
    
    def update(self, data: KLine) -> Decimal | None:
        """返回ATR值"""
        ...
```

### KDJ

```python
class KDJ(T):
    """随机指标"""
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        """
        参数:
            n: RSV周期
            m1: K值平滑周期
            m2: D值平滑周期
        """
        ...
    
    def update(self, data: KLine) -> tuple[float, float, float] | tuple[None, None, None]:
        """
        返回:
            (K, D, J)
        """
        ...
```

### SuperTrend

```python
class SuperTrend(T):
    """超级趋势指标"""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        参数:
            period: ATR周期
            multiplier: ATR倍数
        """
        ...
    
    def update(self, data: KLine) -> tuple[Decimal, int] | tuple[None, None]:
        """
        返回:
            (SuperTrend值, 趋势方向: 1=上升, -1=下降)
        """
        ...
```

## 缠论指标

### Chan - 缠论分析

缠论是一套完整的技术分析体系，包含分型、笔、线段、中枢等概念。

```python
from leek_core.indicators.chan import Chan

chan = Chan()

for kline in klines:
    result = chan.update(kline)
    
    # 获取分型
    fxs = chan.fxs        # 分型列表
    
    # 获取笔
    bis = chan.bis        # 笔列表
    
    # 获取线段
    segs = chan.segs      # 线段列表
    
    # 获取中枢
    zss = chan.zss        # 中枢列表
    
    # 获取买卖点
    bsps = chan.bsps      # 买卖点列表
```

**缠论组件：**

| 组件 | 类 | 说明 |
|------|-----|------|
| 分型 | `FX` | 顶分型/底分型 |
| 笔 | `BI` | 相邻分型连接 |
| 线段 | `SEG` | 笔的组合 |
| 中枢 | `ZS` | 至少3笔重叠区域 |
| 买卖点 | `BSP` | 背驰点位 |

### CZSC - 缠中说禅指标

简化版缠论实现：

```python
from leek_core.indicators.czsc import CZSC

czsc = CZSC()

for kline in klines:
    czsc.update(kline)
    
    # 获取当前状态
    trend = czsc.trend     # 趋势方向
    bi = czsc.current_bi   # 当前笔
```

## 自定义指标

### 基础模板

```python
from leek_core.indicators import T
from leek_core.models import KLine
from decimal import Decimal
from typing import Optional

class MyIndicator(T):
    """自定义指标"""
    
    def __init__(self, period: int = 14, param2: float = 1.0, max_cache: int = 10):
        super().__init__(max_cache=max_cache)
        self.period = period
        self.param2 = Decimal(str(param2))
        
        # 内部状态
        self._prices = []
    
    def update(self, data: KLine) -> Optional[Decimal]:
        """
        更新指标
        
        参数:
            data: K线数据
            
        返回:
            指标值，数据不足时返回None
        """
        # 收集数据
        self._prices.append(data.close)
        
        # 保持窗口大小
        if len(self._prices) > self.period:
            self._prices = self._prices[-self.period:]
        
        # 数据不足
        if len(self._prices) < self.period:
            return None
        
        # 计算指标
        result = sum(self._prices) / len(self._prices) * self.param2
        
        # 缓存结果
        if data.is_finished:
            self.cache.append(result)
            if len(self.cache) > self.max_cache:
                self.cache = self.cache[-self.max_cache:]
        
        return result
```

### 组合指标

```python
from leek_core.indicators import MA, RSI, ATR

class TrendStrength(T):
    """趋势强度指标：结合MA、RSI、ATR"""
    
    def __init__(self, ma_period: int = 20, rsi_period: int = 14, atr_period: int = 14):
        super().__init__()
        self.ma = MA(ma_period)
        self.rsi = RSI(rsi_period)
        self.atr = ATR(atr_period)
    
    def update(self, data: KLine) -> dict | None:
        ma_value = self.ma.update(data)
        rsi_value = self.rsi.update(data)
        atr_value = self.atr.update(data)
        
        if ma_value is None or rsi_value is None or atr_value is None:
            return None
        
        # 计算趋势方向
        trend = 1 if data.close > ma_value else -1
        
        # 计算趋势强度 (0-100)
        strength = abs(rsi_value - 50) * 2
        
        # 计算波动性
        volatility = float(atr_value / data.close) * 100
        
        return {
            "trend": trend,
            "strength": strength,
            "volatility": volatility,
            "ma": ma_value,
            "rsi": rsi_value,
            "atr": atr_value,
        }
```

## 在策略中使用

```python
from leek_core.strategy import CTAStrategy
from leek_core.indicators import MA, MACD, RSI, BOLL, SuperTrend
from leek_core.models import PositionSide, KLine

class MultiIndicatorStrategy(CTAStrategy):
    """多指标组合策略"""
    
    display_name = "多指标组合策略"
    
    def __init__(self):
        super().__init__()
        # 趋势指标
        self.ma20 = MA(20)
        self.ma50 = MA(50)
        self.supertrend = SuperTrend(10, 3.0)
        
        # 震荡指标
        self.rsi = RSI(14)
        self.macd = MACD()
        
        # 波动率指标
        self.boll = BOLL(20, 2.0)
    
    def on_kline(self, kline: KLine):
        # 更新所有指标
        ma20 = self.ma20.update(kline)
        ma50 = self.ma50.update(kline)
        st_value, st_trend = self.supertrend.update(kline)
        rsi = self.rsi.update(kline)
        dif, dea = self.macd.update(kline)
        upper, middle, lower = self.boll.update(kline)
        
        # 存储到K线
        kline.ma20 = ma20
        kline.ma50 = ma50
        kline.supertrend = st_value
        kline.st_trend = st_trend
        kline.rsi = rsi
        kline.dif = dif
        kline.dea = dea
        kline.boll_upper = upper
        kline.boll_middle = middle
        kline.boll_lower = lower
    
    def should_open(self) -> PositionSide | None:
        kline = self.current_kline
        
        # 检查指标是否就绪
        if kline.ma20 is None or kline.ma50 is None:
            return None
        
        # 多头条件
        long_conditions = [
            kline.ma20 > kline.ma50,           # 均线多头排列
            kline.st_trend == 1,               # SuperTrend上升
            kline.rsi < 70,                    # RSI未超买
            kline.dif > kline.dea,             # MACD金叉
        ]
        
        # 空头条件
        short_conditions = [
            kline.ma20 < kline.ma50,           # 均线空头排列
            kline.st_trend == -1,              # SuperTrend下降
            kline.rsi > 30,                    # RSI未超卖
            kline.dif < kline.dea,             # MACD死叉
        ]
        
        if all(long_conditions):
            return PositionSide.LONG
        elif all(short_conditions):
            return PositionSide.SHORT
        
        return None
    
    def should_close(self, position_side: PositionSide) -> bool:
        kline = self.current_kline
        
        if position_side == PositionSide.LONG:
            # 多头平仓条件
            return kline.st_trend == -1 or kline.rsi > 80
        else:
            # 空头平仓条件
            return kline.st_trend == 1 or kline.rsi < 20
```

## 最佳实践

### 1. 指标预热

确保策略有足够的历史数据初始化指标：

```python
# 回测配置中设置预热期
config = {
    "pre_start": start_time - period * timeframe_ms,
    "pre_end": start_time,
    "start_time": start_time,
    ...
}
```

### 2. 避免重复计算

在单根K线内只计算一次指标：

```python
def on_kline(self, kline: KLine):
    # 只在K线完成时更新
    if kline.is_finished:
        self.ma_value = self.ma.update(kline)
```

### 3. 使用缓存

利用指标的内置缓存获取历史值：

```python
macd = MACD(max_cache=20)

# 获取最近的MACD值
recent_values = macd.cache  # [(dif, dea), ...]
```

## 相关模块

- [因子模块](06-factor.md) - 基于指标的因子计算
- [策略模块](01-strategy.md) - 在策略中使用指标
- [数据模型](10-models.md) - KLine 数据结构
