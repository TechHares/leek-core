---
name: leek-indicators
description: 使用和开发 leek 技术指标。继承基类 T，实现 update() 方法进行流式计算。内置 50+ 指标（MA/MACD/RSI/KDJ/布林带等）。当用户要使用技术指标、开发自定义指标、计算均线/MACD/RSI 等时使用。Use when working with technical indicators, moving averages, oscillators, or developing custom indicators.
---

# Leek 技术指标

## 快速使用

```python
from leek_core.indicators import MACD, RSI, BollBand, MA

# 创建指标
macd = MACD(fast_period=12, slow_period=26, moving_period=9)
rsi = RSI(window=14)
boll = BollBand(window=20, num_std_dev=2)

# 流式更新
for kline in klines:
    dif, dea = macd.update(kline)
    rsi_value = rsi.update(kline)
    lower, middle, upper = boll.update(kline)

# 获取历史值
last_10 = macd.last(10)  # 最近10个计算结果
```

## 内置指标列表

### 移动平均线

| 指标 | 说明 | 参数 |
|-----|------|------|
| `MA` | 简单移动平均 | `window=9` |
| `EMA` | 指数移动平均 | `window=9` |
| `WMA` | 加权移动平均 | `window=9` |
| `HMA` | Hull 移动平均 | `window=9` |
| `KAMA` | 自适应移动平均 | `window=10, fast=2, slow=30` |
| `FRAMA` | 分形自适应移动平均 | `window=10` |
| `LLT` | 低延迟趋势线 | `window=9, alpha=2` |
| `SuperSmoother` | 超级平滑器 | `window=10` |

### 趋势指标

| 指标 | 说明 | 返回值 |
|-----|------|--------|
| `MACD` | 指数平滑异同 | `(dif, dea)` |
| `DMI` | 方向运动指数 | `(adx, up_di, down_di, adxr)` |
| `ADX` | 平均方向指数 | `adx` |
| `SuperTrend` | 超级趋势 | `(trend, direction)` |
| `IchimokuCloud` | 一目均衡表 | `(tenkan, kijun, senkou_a, senkou_b, chikou)` |
| `SAR` | 抛物线止损 | `sar` |

### 振荡指标

| 指标 | 说明 | 返回值 |
|-----|------|--------|
| `RSI` | 相对强弱指数 | `rsi (0-100)` |
| `StochRSI` | 随机RSI | `[k, d]` |
| `KDJ` | 随机指标 | `(k, d, j)` |
| `CCI` | 商品通道指数 | `cci` |
| `WR` | 威廉指标 | `wr` |
| `BiasRatio` | 乖离率 | `bias` |
| `ZScore` | Z分数 | `zscore` |

### 波动率指标

| 指标 | 说明 | 返回值 |
|-----|------|--------|
| `ATR` | 平均真实波幅 | `atr` |
| `TR` | 真实波幅 | `tr` |
| `BollBand` | 布林带 | `(lower, middle, upper)` |
| `KeltnerChannel` | 肯特纳通道 | `(lower, middle, upper)` |
| `DonchianChannel` | 唐奇安通道 | `(lower, middle, upper)` |
| `ChaikinVolatility` | 佳庆波动率 | `cv` |

### 成交量指标

| 指标 | 说明 | 返回值 |
|-----|------|--------|
| `VolumeProfile` | 成交量分布 | profile dict |
| `ElderImpulse` | Elder冲动系统 | `(impulse, ema, macd_hist)` |

### 其他指标

| 指标 | 说明 |
|-----|------|
| `RSRS` | 阻力支撑相对强度 |
| `DeMarker` | DeMark 指标 |
| `TDSequence` | TD序列 |
| `GannHiLo` | 江恩高低 |
| `Extreme` | 极值检测 |
| `SpikeDetector` | 尖峰检测 |
| `Divergence` | MACD背离检测 |

### 混沌/复杂系统

| 指标 | 说明 |
|-----|------|
| `HurstExponent` | 赫斯特指数（趋势持续性） |
| `LyapunovExponent` | 李雅普诺夫指数（混沌程度） |
| `CorrelationDimension` | 关联维数 |

### 缠论指标

| 指标 | 说明 |
|-----|------|
| `Chan` | 缠论完整分析 |
| `CZSC` | 缠中说禅 |
| `ChanK` | 缠论K线处理 |
| `ChanBI` | 缠论笔 |
| `ChanSegment` | 缠论线段 |
| `ChanZS` | 缠论中枢 |

### 工具类

| 指标 | 说明 |
|-----|------|
| `MERGE` | K线周期合并（如1分钟合并为5分钟） |

## 基类 T

所有指标继承自基类 `T`：

```python
class T:
    def __init__(self, max_cache=100):
        self.cache = deque(maxlen=max_cache)  # 历史结果缓存
    
    def update(self, data: KLine):
        """流式更新，返回计算结果"""
        pass
    
    def last(self, n=100) -> list:
        """获取最近 n 个历史结果"""
        return list(self.cache)[-n:]
```

## 开发自定义指标

### 基本模式

```python
from collections import deque
from leek_core.indicators import T
from leek_core.models import KLine

class MyIndicator(T):
    def __init__(self, window=14, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window)  # 数据缓冲区
    
    def update(self, data: KLine):
        result = None
        try:
            # 1. 数据不足时返回 None
            if len(self.q) < self.window - 1:
                return result
            
            # 2. 计算指标值
            values = list(self.q) + [data.close]
            result = sum(values) / len(values)
            
            return result
        finally:
            # 3. 仅在 K 线完成时更新状态
            if data.is_finished:
                self.q.append(data.close)
                if result is not None:
                    self.cache.append(result)
```

### 关键设计模式

1. **Try-Finally 模式**：确保状态在 K 线完成时更新
2. **延迟缓存**：仅 `is_finished=True` 时写入缓存
3. **多返回值**：返回元组如 `(dif, dea)`
4. **组合使用**：指标内部可使用其他指标

### 组合指标示例

```python
from leek_core.indicators import T, MA, ATR

class MyTrendIndicator(T):
    def __init__(self, ma_period=20, atr_period=14):
        T.__init__(self)
        self.ma = MA(ma_period)
        self.atr = ATR(atr_period)
    
    def update(self, data):
        ma_value = self.ma.update(data)
        atr_value = self.atr.update(data)
        
        if ma_value is None or atr_value is None:
            return None, None
        
        upper = ma_value + 2 * atr_value
        lower = ma_value - 2 * atr_value
        
        if data.is_finished:
            self.cache.append((upper, lower))
        
        return upper, lower
```

## 在策略中使用

```python
from leek_core.strategy import CTAStrategy
from leek_core.indicators import MACD, RSI
from leek_core.models import KLine, PositionSide

class MACDStrategy(CTAStrategy):
    display_name = "MACD策略"
    
    def __init__(self):
        super().__init__()
        self.macd = MACD(12, 26, 9)
        self.prev_m = None
        self.cur_m = None
    
    def on_kline(self, kline: KLine):
        dif, dea = self.macd.update(kline)
        
        if dif is not None and dea is not None:
            m = dif - dea  # MACD柱
            kline.dif = dif
            kline.dea = dea
            kline.macd = m
            
            if kline.is_finished:
                self.prev_m = self.cur_m
                self.cur_m = m
    
    def should_open(self) -> PositionSide | None:
        if self.prev_m is None or self.cur_m is None:
            return None
        
        # MACD柱由负转正
        if self.prev_m < 0 < self.cur_m:
            return PositionSide.LONG
        # MACD柱由正转负
        if self.prev_m > 0 > self.cur_m:
            return PositionSide.SHORT
        
        return None
```

## K线周期合并

```python
from leek_core.indicators import MERGE

# 将1分钟K线合并为5分钟
merge_5m = MERGE(window=5)

for kline_1m in klines_1m:
    kline_5m = merge_5m.update(kline_1m)
    if kline_5m and kline_5m.is_finished:
        # 处理完成的5分钟K线
        process(kline_5m)
```

## 背离检测

```python
from leek_core.indicators import MACD, Divergence

macd = MACD(12, 26, 9)
divergence = Divergence(
    divergence_threshold=1,      # 背离阈值
    close_price_divergence=True, # 收盘价背离
    peak_price_divergence=True,  # 极值价背离
    m_area_divergence=True,      # 能量柱面积背离
)

# 收集MACD数据
macd_data = []
for kline in klines:
    dif, dea = macd.update(kline)
    if dif and dea and kline.is_finished:
        kline.dif = dif
        kline.dea = dea
        kline.m = dif - dea
        macd_data.append(kline)

# 检测背离
if divergence.is_top_divergence(macd_data):
    print("顶背离")
if divergence.is_bottom_divergence(macd_data):
    print("底背离")
```

## 最佳实践

1. **状态管理**：使用 `try-finally` 确保 `is_finished` 时更新状态
2. **数据验证**：返回前检查数据是否充足
3. **缓存大小**：根据需要设置 `max_cache`
4. **组合复用**：复杂指标可组合简单指标
5. **绘图支持**：在策略中设置 `kline.xxx = value` 用于可视化
