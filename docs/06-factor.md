# 06 因子模块

## 概述

因子模块是量化交易系统的特征工程核心组件，用于从市场数据中提取有预测价值的特征。本模块采用**双模式设计**（Dual Mode），同时支持：

1. **流式计算（Update）**：用于实盘交易和逐行回测，实时更新因子值
2. **向量化计算（Vectorized）**：用于离线批量计算，高效处理大规模历史数据

## 核心组件

### `DualModeFactor` - 双模因子基类

所有因子的抽象基类，定义因子的基本接口：

```python
from abc import ABC, abstractmethod
from typing import List, Union
import pandas as pd
from leek_core.base import LeekComponent

class DualModeFactor(LeekComponent, ABC):
    """
    双模因子基类
    
    类属性：
        display_name: 因子展示名称
        _name: 因子内部名称（用于列名前缀）
        _required_buffer_size: 流式计算所需的缓冲区大小，默认360
        init_params: 因子参数定义列表
    """
    
    # 类属性定义
    display_name: str = None
    _name: str = None
    _required_buffer_size: int = 360
    init_params: List[Field] = []
    
    # 核心方法
    def update(self, kline: KLine) -> Union[float, List[float], None]: ...
    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def get_output_names(self) -> List[str]: ...
```

### 输入数据格式

因子接收的 DataFrame 包含以下标准列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `start_time` | int | K线开始时间戳（毫秒） |
| `symbol` | str | 交易对符号 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `volume` | float | 成交量 |
| `amount` | float | 成交额（可选） |

## 编写因子的原则

### 1. 因子类定义规范

```python
from typing import List
import numpy as np
import pandas as pd
from leek_core.models import Field, FieldType
from leek_core.ml.factors import DualModeFactor

class MyFactor(DualModeFactor):
    """
    因子文档字符串：说明因子原理、计算公式、参数含义
    
    计算公式：
    - 描述因子的数学公式
    - 说明各参数的作用
    
    示例：
    - 给出典型的参数配置示例
    """
    
    # 【必需】因子展示名称（用于界面显示）
    display_name: str = "我的因子"
    
    # 【可选】因子内部名称（用于列名前缀），默认使用类名
    _name: str = "MyFactor"
    
    # 【可选】缓冲区大小，流式计算时保留的历史数据条数
    # 应设置为计算所需的最大窗口 + 一些余量
    _required_buffer_size: int = 100
    
    # 【必需】参数定义列表
    init_params = [
        Field(
            name="window",           # 参数名，与__init__参数名一致
            label="窗口大小",        # 界面显示名称
            type=FieldType.INT,      # 参数类型
            default=20,              # 默认值
            required=True,           # 是否必填
            description="计算周期"   # 参数说明
        ),
    ]
    
    def __init__(self, **kwargs):
        """
        初始化因子
        - 从kwargs获取参数（支持关键字参数传入）
        - 设置_required_buffer_size（如需动态计算）
        - 调用super().__init__()
        - 初始化内部状态
        """
        self.window = int(kwargs.get("window", 20))
        self._required_buffer_size = self.window + 10  # 动态设置缓冲区
        super().__init__()
        # 动态生成因子名称
        self._factor_name = f"MyFactor_{self.window}"
```

### 2. 核心方法实现

#### `compute(df)` - 批量计算（必须实现）

这是因子的核心计算方法，在 DataFrame 上计算因子值：

```python
def compute(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    批量计算接口
    
    参数：
        df: 包含 OHLCV 数据的 DataFrame
    
    返回：
        pd.DataFrame: 只包含因子列的新 DataFrame
        
    注意：
        - 返回的 DataFrame 的 index 应与输入 df 的 index 一致
        - 列名应与 get_output_names() 返回的列表一致
        - 前几行可能因数据不足而为 NaN，这是正常的
    """
    # 获取所需数据
    close = df['close']
    
    # 计算因子值
    result = close.rolling(window=self.window).mean()
    
    # 返回包含因子列的 DataFrame
    return pd.DataFrame({self._factor_name: result}, index=df.index)
```

#### `get_output_names()` - 获取输出列名

```python
def get_output_names(self) -> List[str]:
    """
    获取因子输出的列名列表
    
    返回：
        List[str]: 因子列名列表
        
    注意：
        - 单因子返回长度为1的列表
        - 多因子返回包含所有因子名的列表
        - 列名应具有唯一性，通常包含参数信息
    """
    return [self._factor_name]
```

#### `update(kline)` - 流式计算（可选重写）

基类提供了默认实现，会自动缓存数据并调用 `compute()`。如果有性能需求，可以重写：

```python
def update(self, kline: KLine) -> Union[float, List[float], None]:
    """
    流式更新接口
    
    参数：
        kline: 当前K线数据
    
    返回：
        float: 单因子返回单个值
        List[float]: 多因子返回值列表
        None: 数据不足时返回 None
        
    注意：
        - 默认实现会维护一个 deque 缓冲区
        - 只有 kline.is_finished 为 True 时才会添加到缓冲区
        - 可以使用指标类（Indicator）实现更高效的流式计算
    """
    # 使用指标实现高效流式计算
    value = self._indicator.update(kline)
    return float(value) if value is not None else np.nan
```

### 3. 参数类型定义

`Field` 支持的类型：

| FieldType | Python类型 | 说明 |
|-----------|-----------|------|
| `STRING` | str | 字符串 |
| `INT` | int | 整数 |
| `FLOAT` | float | 浮点数 |
| `BOOLEAN` | bool | 布尔值 |
| `RADIO` | Any | 单选，需配合 choices |
| `SELECT` | Any | 下拉选择，需配合 choices |

带选项的参数示例：

```python
Field(
    name="compute_type",
    label="计算方式",
    type=FieldType.RADIO,
    default="ratio",
    choices=[("ratio", "比率"), ("count", "数量"), ("average", "平均")],
    choice_type=ChoiceType.STRING,  # 可选，指定选项值类型
    required=True,
    description="选择计算方式"
)
```

## 完整因子示例

### 示例1：简单单因子 - 移动平均线

```python
import numpy as np
import pandas as pd
from leek_core.indicators import MA
from leek_core.models import Field, FieldType
from leek_core.ml.factors import DualModeFactor

class MAFactor(DualModeFactor):
    """
    移动平均线因子
    
    计算公式：
    MA = SUM(CLOSE, window) / window
    
    用途：
    - 衡量价格趋势
    - 作为基准价格参考
    """
    display_name = "MA"
    _name = "MA"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=20,
            description="移动平均线的窗口大小"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 20))
        self._indicator = MA(self.window)  # 使用指标类实现流式计算
        self._factor_name = f"MA_{self.window}"
        super().__init__()

    def update(self, kline) -> float:
        """使用指标类实现高效的流式计算"""
        val = self._indicator.update(kline)
        return float(val) if val is not None else np.nan

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化批量计算"""
        df[self._factor_name] = df['close'].rolling(window=self.window).mean()
        return pd.DataFrame({self._factor_name: df[self._factor_name]}, index=df.index)
    
    def get_output_names(self) -> list:
        return [self._factor_name]
```

### 示例2：带多参数的因子 - 成交量平均值比率

```python
import numpy as np
import pandas as pd
from leek_core.models import ChoiceType, Field, FieldType
from leek_core.ml.factors import DualModeFactor

class VolumeAverageFactor(DualModeFactor):
    """
    成交量平均值比率因子
    
    计算公式：
    1. 在近 window 个周期内，根据 side 参数过滤K线：
       - FLAT: 统计所有K线
       - LONG: 只统计上涨K线（close > open）
       - SHORT: 只统计下跌K线（close < open）
    
    2. 在过滤后的K线中，根据 type 参数选择成交量：
       - min_volume: 取成交量最小的 top 个周期
       - max_volume: 取成交量最大的 top 个周期
    
    3. 计算这 top 个周期的成交量平均值
    
    4. 最终因子值 = 平均值 / 当前成交量
    """
    display_name = "成交量平均值"
    _name = "VolumeAverageFactor"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=20,
            required=True,
            description="统计窗口大小"
        ),
        Field(
            name="side",
            label="方向",
            type=FieldType.RADIO,
            default="FLAT",
            choices=[("FLAT", "全部"), ("LONG", "上涨K线"), ("SHORT", "下跌K线")],
            choice_type=ChoiceType.STRING,
            required=True,
            description="统计方向"
        ),
        Field(
            name="type",
            label="类型",
            type=FieldType.RADIO,
            default="min_volume",
            choices=[("min_volume", "最小成交量"), ("max_volume", "最大成交量")],
            choice_type=ChoiceType.STRING,
            required=True,
            description="统计类型"
        ),
        Field(
            name="top",
            label="数量",
            type=FieldType.INT,
            default=20,
            required=True,
            description="取最小或最大成交量的周期数量"
        )
    ]
    
    def __init__(self, window=20, side="FLAT", type="min_volume", top=20, name=""):
        self.window = window
        self.side = side
        self.type = type
        self.top = top
        self._required_buffer_size = self.window + 10
        super().__init__()
        # 动态生成因子名称
        if name:
            self._factor_name = name
        else:
            self._factor_name = f"VolumeAverage_{self.window}_{self.side}_{self.type}_{self.top}"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量计算成交量平均值因子"""
        close = df['close'].values
        open_price = df['open'].values
        volume = df['volume'].values
        n = len(df)
        
        # 计算涨跌方向（1=上涨，-1=下跌，0=平盘）
        direction = np.zeros(n, dtype=np.int32)
        direction[close > open_price] = 1
        direction[close < open_price] = -1
        
        # 初始化结果数组
        result = np.full(n, np.nan, dtype=np.float64)
        top_count = min(self.top, self.window)
        
        # 滑动窗口计算
        for i in range(self.window - 1, n):
            start_idx = i - self.window + 1
            end_idx = i + 1
            
            vol_window = volume[start_idx:end_idx]
            dir_window = direction[start_idx:end_idx]
            
            # 根据 side 参数过滤
            if self.side == "LONG":
                mask = dir_window == 1
            elif self.side == "SHORT":
                mask = dir_window == -1
            else:
                mask = np.ones(len(vol_window), dtype=bool)
            
            valid_mask = ~np.isnan(vol_window) & mask
            valid_volumes = vol_window[valid_mask]
            
            if len(valid_volumes) < top_count:
                continue
            
            # 根据 type 参数选择成交量
            if self.type == "min_volume":
                selected_volumes = np.partition(valid_volumes, top_count - 1)[:top_count]
            else:
                selected_volumes = np.partition(valid_volumes, -top_count)[-top_count:]
            
            avg_volume = np.mean(selected_volumes)
            current_volume = volume[i]
            
            if current_volume > 0 and avg_volume > 0:
                result[i] = avg_volume / current_volume
        
        return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    def get_output_names(self) -> list:
        return [self._factor_name]
```

### 示例3：多输出因子 - 时间特征

```python
from typing import List
import numpy as np
import pandas as pd
from leek_core.models import Field, FieldType, KLine
from leek_core.ml.factors import DualModeFactor

class TimeFactor(DualModeFactor):
    """
    时间周期性因子
    
    提取时间相关的周期性特征：
    - hour: 小时 (0-23)
    - day_of_week: 星期几 (0=周一, 6=周日)
    - hour_sin/cos: 小时的周期性编码
    - dow_sin/cos: 星期的周期性编码
    """
    display_name = "TimeFeatures"
    _name = "Time"

    init_params = [
        Field(
            name="include_hour",
            label="包含小时特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含小时特征"
        ),
        Field(
            name="include_day_of_week",
            label="包含星期特征",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含星期几特征"
        ),
        Field(
            name="include_cyclic",
            label="包含周期性编码",
            type=FieldType.BOOLEAN,
            default=True,
            description="是否包含sin/cos周期性编码"
        ),
    ]
    
    def __init__(self, **kwargs):
        super().__init__()
        self.include_hour = kwargs.get("include_hour", True)
        self.include_day_of_week = kwargs.get("include_day_of_week", True)
        self.include_cyclic = kwargs.get("include_cyclic", True)
        
        # 构建因子名称列表
        self.factor_names = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称列表"""
        if self.include_hour:
            self.factor_names.append(f"{self.name}_hour")
            if self.include_cyclic:
                self.factor_names.append(f"{self.name}_hour_sin")
                self.factor_names.append(f"{self.name}_hour_cos")
        
        if self.include_day_of_week:
            self.factor_names.append(f"{self.name}_day_of_week")
            if self.include_cyclic:
                self.factor_names.append(f"{self.name}_dow_sin")
                self.factor_names.append(f"{self.name}_dow_cos")

    def update(self, kline: KLine) -> List[float]:
        """流式更新：从单条K线提取时间特征"""
        timestamp = getattr(kline, 'start_time', None)
        if timestamp is None:
            return [np.nan] * len(self.factor_names)
        
        dt = pd.to_datetime(timestamp, unit='ms')
        features = []
        
        if self.include_hour:
            hour = dt.hour
            features.append(float(hour))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * hour / 24)))
                features.append(float(np.cos(2 * np.pi * hour / 24)))
        
        if self.include_day_of_week:
            dow = dt.dayofweek
            features.append(float(dow))
            if self.include_cyclic:
                features.append(float(np.sin(2 * np.pi * dow / 7)))
                features.append(float(np.cos(2 * np.pi * dow / 7)))
        
        return features

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量计算：从DataFrame提取时间特征"""
        df["_datetime"] = pd.to_datetime(df['start_time'], unit='ms')
        dt = df["_datetime"].dt
        
        factor_results = {}
        
        if self.include_hour:
            hour = dt.hour
            factor_results[f"{self.name}_hour"] = hour
            if self.include_cyclic:
                factor_results[f"{self.name}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
                factor_results[f"{self.name}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        
        if self.include_day_of_week:
            dow = dt.dayofweek
            factor_results[f"{self.name}_day_of_week"] = dow
            if self.include_cyclic:
                factor_results[f"{self.name}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
                factor_results[f"{self.name}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        return pd.DataFrame(factor_results, index=df.index)

    def get_output_names(self) -> List[str]:
        return self.factor_names
```

### 示例4：复杂批量因子 - Alpha158

参考 Qlib 的 Alpha158 因子集，展示如何实现大规模因子计算：

```python
from typing import List
import numpy as np
import pandas as pd
from leek_core.models import Field, FieldType
from leek_core.ml.factors import DualModeFactor

class Alpha158Factor(DualModeFactor):
    """
    根据Qlib中发表的 Alpha158 因子
    
    标准Alpha158包含158个固定因子：
    1. kbar: K线特征（9个）- KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
    2. price: 原始价格特征（4个）- OPEN0, HIGH0, LOW0, VWAP0
    3. rolling: 滚动窗口特征（145个）- 29种算子 × 5个窗口[5,10,20,30,60]
    """
    display_name = "Alpha158"
    _required_buffer_size = 70  # 最大窗口60 + 余量
    _delta = 1e-20  # 除零保护
    
    init_params = [
        Field(name="include_kbar", label="包含KBar因子", type=FieldType.BOOLEAN, default=True),
        Field(name="include_price", label="包含价格因子", type=FieldType.BOOLEAN, default=True),
        Field(name="include_rolling", label="包含滚动因子", type=FieldType.BOOLEAN, default=True),
        Field(name="windows", label="滚动窗口", type=FieldType.STRING, default="5,10,20,30,60"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__()
        self.include_kbar = kwargs.get("include_kbar", True)
        self.include_price = kwargs.get("include_price", True)
        self.include_rolling = kwargs.get("include_rolling", True)
        self.windows_str = kwargs.get("windows", "5,10,20,30,60")
        
        # 解析窗口参数
        self.rolling_windows = []
        try:
            for part in self.windows_str.split(","):
                part = part.strip()
                if part:
                    self.rolling_windows.append(int(part))
        except Exception:
            self.rolling_windows = [5, 10, 20, 30, 60]
        
        # 预构建因子名称列表
        self.factor_names = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称"""
        if self.include_kbar:
            kbar_names = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
            self.factor_names.extend([f"{self.name}_{name}" for name in kbar_names])
        
        if self.include_price:
            price_names = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
            self.factor_names.extend([f"{self.name}_{name}" for name in price_names])
        
        if self.include_rolling:
            rolling_operators = ["ROC", "MA", "STD", "MAX", "MIN", "RANK", "RSV"]
            for op_name in rolling_operators:
                for window in self.rolling_windows:
                    self.factor_names.append(f"{self.name}_{op_name}{window}")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        factor_results = {}
        
        # 1. KBar 因子
        if self.include_kbar:
            factor_results[f"{self.name}_KMID"] = (close - open_price) / (open_price + self._delta)
            factor_results[f"{self.name}_KLEN"] = (high - low) / (open_price + self._delta)
            # ... 其他 KBar 因子
        
        # 2. Price 因子
        if self.include_price:
            close_eps = close + self._delta
            factor_results[f"{self.name}_OPEN0"] = open_price / close_eps
            factor_results[f"{self.name}_HIGH0"] = high / close_eps
            factor_results[f"{self.name}_LOW0"] = low / close_eps
            # ... 其他 Price 因子
        
        # 3. Rolling 因子
        if self.include_rolling:
            for window in self.rolling_windows:
                factor_results[f"{self.name}_ROC{window}"] = close.shift(window) / (close + self._delta)
                factor_results[f"{self.name}_MA{window}"] = close.rolling(window).mean() / (close + self._delta)
                factor_results[f"{self.name}_STD{window}"] = close.rolling(window).std() / (close + self._delta)
                # ... 其他滚动因子
        
        return pd.DataFrame(factor_results, index=df.index)

    def get_output_names(self) -> List[str]:
        return self.factor_names
```

## 内置因子库

系统提供了多个预置因子类：

| 因子类 | 说明 | 因子数量 |
|--------|------|----------|
| `MAFactor` | 移动平均线 | 1 |
| `RSIFactor` | 相对强弱指数 | 1 |
| `ATRFactor` | 平均真实波幅 | 1 |
| `TimeFactor` | 时间周期特征 | 可配置（5-30） |
| `DirectionFactor` | 涨跌K线比率 | 1 |
| `VolumeAverageFactor` | 成交量平均值 | 1 |
| `LongShortVolumeRatioFactor` | 涨跌成交量比率 | 1 |
| `Alpha101Factor` | WorldQuant 101因子 | 最多101 |
| `Alpha158Factor` | Qlib Alpha158 | 最多158 |
| `Alpha191Factor` | 国泰君安191因子 | 最多191 |
| `Alpha360Factor` | Qlib Alpha360 | 最多360 |

## 因子评价

使用 `FactorEvaluator` 评估因子有效性：

```python
from leek_core.ml.factors import FactorEvaluator

# 创建评价器
evaluator = FactorEvaluator(future_periods=1, quantile_count=5)

# 计算IC值
ic = evaluator.evaluate(df, factor_name='MA_20')
```

评价指标说明：

| 指标 | 说明 | 良好范围 |
|------|------|----------|
| IC | 信息系数（Spearman相关） | \|IC\| > 0.02 |
| IR | 信息比率（IC/IC_std） | \|IR\| > 0.5 |
| IC_decay | IC衰减 | 缓慢衰减 |

## 最佳实践

### 1. 命名规范

- **因子类名**：使用 `PascalCase`，以 `Factor` 结尾（如 `RSIFactor`）
- **display_name**：简洁的中文/英文名称（如 `"RSI"` 或 `"相对强弱指数"`）
- **_name**：用于列名前缀的短名称（如 `"RSI"`）
- **输出列名**：格式为 `{_name}_{参数}`（如 `RSI_14`）

### 2. 参数设计

- 所有可配置参数都应通过 `init_params` 定义
- 参数名应与 `__init__` 参数名一致
- 提供合理的默认值
- 参数描述应清晰说明作用

### 3. 数值稳定性

```python
# 使用小常量防止除零
_delta = 1e-20

# 正确方式
result = numerator / (denominator + self._delta)

# 避免
result = numerator / denominator  # 可能除零
```

### 4. 性能优化

```python
# 推荐：向量化计算
df['factor'] = df['close'].rolling(window).mean()

# 避免：逐行循环（性能差）
for i in range(len(df)):
    df.loc[i, 'factor'] = df['close'].iloc[max(0, i-window):i+1].mean()
```

### 5. 缓冲区设置

```python
# 根据最大窗口动态设置缓冲区
def __init__(self, window=60, **kwargs):
    self.window = window
    self._required_buffer_size = self.window + 10  # 添加余量
    super().__init__()
```

### 6. 返回格式

```python
def compute(self, df: pd.DataFrame) -> pd.DataFrame:
    # 正确：返回只包含因子列的新 DataFrame
    return pd.DataFrame({self._factor_name: result}, index=df.index)
    
    # 避免：直接修改并返回原 df
    # df[self._factor_name] = result
    # return df
```

### 7. NaN 处理

- 数据不足时返回 NaN 是正常的
- 不要用 0 或其他值替代 NaN
- 下游会正确处理 NaN 值

```python
# 流式计算中
def update(self, kline) -> float:
    if len(self._buffer) < self.window:
        return np.nan  # 数据不足，返回 NaN
    return float(self._calculate())
```

## 使用因子

### 在代码中使用

```python
from leek_core.ml.factors import MAFactor, Alpha158Factor

# 创建单因子
ma_factor = MAFactor(window=20)

# 流式计算
for kline in klines:
    value = ma_factor.update(kline)
    print(f"MA: {value}")

# 批量计算
result_df = ma_factor.compute(df)

# 创建多因子
alpha158 = Alpha158Factor(include_kbar=True, include_price=True, include_rolling=True)
features_df = alpha158.compute(df)
print(f"生成 {len(alpha158.get_output_names())} 个因子")
```

### 在策略中使用

因子通常通过 ML 模型与策略集成，详见策略模块文档。
