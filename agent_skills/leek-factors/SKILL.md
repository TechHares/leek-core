---
name: leek-factors
description: 开发 leek 机器学习因子。继承 DualModeFactor 基类，实现 compute()/update() 方法进行批量和流式计算。内置 Alpha101/Alpha158/Alpha360 等因子集。当用户要开发量化因子、特征工程、ML因子时使用。Use when developing quantitative factors, feature engineering, or machine learning features.
---

# Leek 因子开发

## 概述

因子模块采用**双模式设计**：

1. **流式计算（update）**：实盘交易和逐行回测
2. **向量化计算（compute）**：离线批量计算

## 快速开始

```python
import pandas as pd
from leek_core.ml.factors import DualModeFactor, MAFactor

# 使用内置因子
ma_factor = MAFactor(window=20)

# 批量计算
df = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 
    'close': [...], 'volume': [...]
})
result_df = ma_factor.compute(df)

# 流式计算
for kline in klines:
    value = ma_factor.update(kline)
```

## 基类 DualModeFactor

```python
from leek_core.ml.factors import DualModeFactor

class DualModeFactor(LeekComponent, ABC):
    # 类属性
    display_name: str = None           # 展示名称
    _name: str = None                  # 内部名称（列名前缀）
    _required_buffer_size: int = 360   # 流式计算缓冲区大小
    init_params: List[Field] = []      # 参数定义
    
    # 核心方法
    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...  # 批量计算（必须实现）
    def update(self, kline: KLine) -> float | List[float] | None: ...  # 流式计算
    def get_output_names(self) -> List[str]: ...  # 输出列名
```

## 开发自定义因子

### 单输出因子

```python
import numpy as np
import pandas as pd
from leek_core.ml.factors import DualModeFactor
from leek_core.indicators import MA
from leek_core.models import Field, FieldType

class MAFactor(DualModeFactor):
    """移动平均因子"""
    
    display_name = "MA"
    _name = "MA"
    
    init_params = [
        Field(name="window", label="窗口", type=FieldType.INT, default=20),
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 20))
        self._required_buffer_size = self.window + 10
        self._indicator = MA(self.window)  # 使用指标加速流式计算
        self._factor_name = f"MA_{self.window}"
        super().__init__()
    
    def update(self, kline) -> float:
        """流式计算"""
        val = self._indicator.update(kline)
        return float(val) if val is not None else np.nan
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量计算"""
        result = df['close'].rolling(window=self.window).mean()
        return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    def get_output_names(self) -> list:
        return [self._factor_name]
```

### 多输出因子

```python
class BollFactor(DualModeFactor):
    """布林带因子（3个输出）"""
    
    display_name = "Boll"
    _name = "Boll"
    
    init_params = [
        Field(name="window", type=FieldType.INT, default=20),
        Field(name="num_std", type=FieldType.FLOAT, default=2.0),
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 20))
        self.num_std = float(kwargs.get("num_std", 2.0))
        super().__init__()
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        
        return pd.DataFrame({
            f"Boll_{self.window}_lower": ma - self.num_std * std,
            f"Boll_{self.window}_middle": ma,
            f"Boll_{self.window}_upper": ma + self.num_std * std,
        }, index=df.index)
    
    def get_output_names(self) -> list:
        return [
            f"Boll_{self.window}_lower",
            f"Boll_{self.window}_middle", 
            f"Boll_{self.window}_upper"
        ]
```

### 带选项的因子

```python
from leek_core.models import ChoiceType

class VolumeRatioFactor(DualModeFactor):
    """成交量比率因子"""
    
    display_name = "成交量比率"
    
    init_params = [
        Field(name="window", type=FieldType.INT, default=20),
        Field(
            name="side",
            label="方向",
            type=FieldType.RADIO,
            choices=[("all", "全部"), ("long", "上涨"), ("short", "下跌")],
            choice_type=ChoiceType.STRING,
            default="all",
        ),
        Field(
            name="agg_type",
            label="聚合方式",
            type=FieldType.RADIO,
            choices=[("mean", "平均"), ("sum", "求和"), ("max", "最大")],
            default="mean",
        ),
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 20))
        self.side = kwargs.get("side", "all")
        self.agg_type = kwargs.get("agg_type", "mean")
        super().__init__()
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df['volume'].copy()
        
        # 根据方向过滤
        if self.side == "long":
            mask = df['close'] > df['open']
            volume = volume.where(mask, 0)
        elif self.side == "short":
            mask = df['close'] < df['open']
            volume = volume.where(mask, 0)
        
        # 聚合计算
        if self.agg_type == "mean":
            result = volume.rolling(self.window).mean()
        elif self.agg_type == "sum":
            result = volume.rolling(self.window).sum()
        else:
            result = volume.rolling(self.window).max()
        
        factor_name = f"VolumeRatio_{self.window}_{self.side}_{self.agg_type}"
        return pd.DataFrame({factor_name: result / df['volume']}, index=df.index)
    
    def get_output_names(self) -> list:
        return [f"VolumeRatio_{self.window}_{self.side}_{self.agg_type}"]
```

## 内置因子库

| 因子类 | 说明 | 因子数量 |
|-------|------|----------|
| `MAFactor` | 移动平均 | 1 |
| `RSIFactor` | 相对强弱 | 1 |
| `ATRFactor` | 真实波幅 | 1 |
| `TimeFactor` | 时间特征 | 可配置 |
| `DirectionFactor` | 涨跌比率 | 1 |
| `VolumeAverageFactor` | 成交量平均 | 1 |
| `Alpha101Factor` | WorldQuant 101 | 最多101 |
| `Alpha158Factor` | Qlib Alpha158 | 最多158 |
| `Alpha191Factor` | 国泰君安191 | 最多191 |
| `Alpha360Factor` | Qlib Alpha360 | 最多360 |

### 使用 Alpha158

```python
from leek_core.ml.factors import Alpha158Factor

alpha158 = Alpha158Factor(
    include_kbar=True,     # K线特征
    include_price=True,    # 价格特征
    include_rolling=True,  # 滚动窗口特征
    windows="5,10,20,30,60",
)

# 批量计算
features_df = alpha158.compute(df)
print(f"生成 {len(alpha158.get_output_names())} 个因子")
```

## 因子评价

```python
from leek_core.backtest import FactorEvaluator

# 创建评价器
evaluator = FactorEvaluator(future_periods=1, quantile_count=5)

# 计算IC值
ic = evaluator.evaluate(df, factor_name='MA_20')
```

| 指标 | 说明 | 良好范围 |
|-----|------|----------|
| IC | 信息系数 | \|IC\| > 0.02 |
| IR | 信息比率 | \|IR\| > 0.5 |
| IC_decay | IC衰减 | 缓慢衰减 |

## 输入数据格式

```python
df = pd.DataFrame({
    'start_time': [...],  # 毫秒时间戳
    'symbol': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'amount': [...],      # 可选
})
```

## 最佳实践

### 1. 命名规范

```python
# 因子类名：PascalCase + Factor
class RSIFactor(DualModeFactor):
    display_name = "RSI"
    _name = "RSI"
    # 输出列名：{_name}_{参数}
    # 如：RSI_14
```

### 2. 数值稳定性

```python
# 防止除零
_delta = 1e-20
result = numerator / (denominator + _delta)
```

### 3. 向量化优先

```python
# 推荐：向量化
df['factor'] = df['close'].rolling(window).mean()

# 避免：循环
for i in range(len(df)):
    df.loc[i, 'factor'] = ...
```

### 4. 缓冲区设置

```python
def __init__(self, window=60, **kwargs):
    self.window = window
    self._required_buffer_size = self.window + 10  # 添加余量
    super().__init__()
```

### 5. NaN 处理

```python
# 数据不足时返回 NaN，不要用 0 替代
def update(self, kline) -> float:
    if len(self._buffer) < self.window:
        return np.nan
    return float(self._calculate())
```

### 6. 返回格式

```python
def compute(self, df: pd.DataFrame) -> pd.DataFrame:
    # 正确：返回只包含因子列的新 DataFrame
    return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    # 避免：直接修改原 df
```

## 在 ML 策略中使用

```python
from leek_core.strategy import MLStrategy
from leek_core.ml.factors import Alpha158Factor

class MyMLStrategy(MLStrategy):
    display_name = "我的ML策略"
    
    # 配置使用的因子
    factor_classes = [Alpha158Factor]
    factor_params = [{"include_rolling": True}]
```

## 详细参考

完整文档见 [reference/06-factor.md](reference/06-factor.md)
