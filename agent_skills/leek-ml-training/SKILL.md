---
name: leek-ml-training
description: 训练 leek 机器学习模型。使用 TrainingEngine 完成数据加载、特征计算、标签生成、模型训练和评估的全流程。支持 XGBoost、GRU 等训练器和多种标签生成器。当用户要训练模型、配置训练参数、选择打标方式、选择训练器时使用。Use when training ML models, configuring training pipelines, selecting labels or trainers.
---

# Leek 模型训练

## 概述

模型训练采用**组件化流水线**设计，核心流程：

```
数据源 → 特征引擎(因子) → 标签生成器 → 训练器 → 评估器 → 模型文件
```

## 快速开始

### 通过配置字典训练

```python
from leek_core.ml.training_engine import training

config = {
    "symbols": ["BTC", "ETH"],
    "timeframes": ["5m"],
    "start_time": "2025-01-01",
    "end_time": "2025-06-01",
    "datasource_class": "model|KLineDatasource",
    "datasource_config": {},
    "datasource_extra": {
        "quote_currency": "USDT",
        "ins_type": "SWAP",
        "market": "okx"
    },
    "factors": [
        {
            "id": 1,
            "name": "Alpha158",
            "class_name": "leek_core.ml.factors.Alpha158Factor",
            "params": {"include_kbar": True, "include_price": True, "include_rolling": True}
        }
    ],
    "label_generator": {
        "class_name": "leek_core.ml.label.TripleBarrierLabel",
        "params": {"hold_periods": 10, "take_profit": 0.02, "stop_loss": 0.01, "side": "both"}
    },
    "trainer": {
        "class_name": "leek_core.ml.trainer.GRUTrainer",
        "params": {"task_type": "classification", "hidden_size": 64, "window_size": 60}
    },
    "train_split_ratio": 0.8,
    "save_model_path": "/path/to/model.joblib",
}

result = training(config)
```

### 使用 TrainingEngine

```python
from leek_core.ml import TrainingEngine, FeatureEngine, ModelEvaluator
from leek_core.ml.trainer import GRUTrainer
from leek_core.ml.label import TripleBarrierLabel
from leek_core.ml.factors import Alpha158Factor

# 创建组件
feature_engine = FeatureEngine([Alpha158Factor(include_kbar=True)])
label_generator = TripleBarrierLabel(hold_periods=10, take_profit=0.02, stop_loss=0.01)
trainer = GRUTrainer(task_type="classification", hidden_size=64)
evaluator = ModelEvaluator(task_type="classification")

engine = TrainingEngine(
    datasource=datasource,
    datasource_extra={"symbols": ["BTC"], "timeframes": ["5m"], ...},
    feature_engine=feature_engine,
    label_generator=label_generator,
    trainer=trainer,
    evaluator=evaluator,
)

engine.on_start()
result = engine.train()
engine.on_stop()
```

## 训练流水线

```
TrainingEngine.train() 执行流程：

1. _load_data()                  ← 从数据源加载 K 线（按 symbol×timeframe 并发）
2. _compute_features_and_labels() ← 特征引擎计算因子 + 标签生成器打标
3. _split_data()                 ← 按时间顺序切分训练集/验证集
4. _load_old_model()             ← 加载旧模型（可选，用于增量训练对比）
5. _evaluate_old_model()         ← 评估旧模型（可选）
6. _train_model()                ← 训练新模型
7. _evaluate_model()             ← 评估新模型（训练集+验证集）
8. _save_model()                 ← 保存模型文件
```

## 标签生成器

### 一览表

| 标签生成器 | 任务类型 | 适用场景 | 类名 |
|-----------|---------|---------|------|
| DirectionLabel | 分类 | 涨跌预测 | `leek_core.ml.label.DirectionLabel` |
| TripleBarrierLabel | 分类 | **短线择时（推荐）** | `leek_core.ml.label.TripleBarrierLabel` |
| EventLabel | 分类 | 事件驱动 | `leek_core.ml.label.EventLabel` |
| FutureReturnLabel | 回归 | 收益率预测 | `leek_core.ml.label.FutureReturnLabel` |
| RiskAdjustedReturnLabel | 回归 | 风险调整收益 | `leek_core.ml.label.RiskAdjustedReturnLabel` |
| ReversalStrengthLabel | 分类/回归 | 均值回归 | `leek_core.ml.label.ReversalStrengthLabel` |
| RankLabel | 排序 | 多因子打分 | `leek_core.ml.label.RankLabel` |
| MultiLabelFusion | 融合 | 多目标学习 | `leek_core.ml.label.MultiLabelFusion` |

### TripleBarrierLabel（三重屏障，推荐用于短线择时）

模拟真实交易：逐根 K 线检查止盈/止损/到期，**哪个先触发就按哪个结算**。

```python
from leek_core.ml.label import TripleBarrierLabel

label_gen = TripleBarrierLabel(
    hold_periods=10,       # 最大持仓 10 根 K 线
    take_profit=0.02,      # 止盈 2%
    stop_loss=0.01,        # 止损 1%
    side="both",           # "long"=做多, "short"=做空, "both"=双向
    num_classes=3,         # 3=三分类, 2=二分类
    conservative=True,     # 同一根K线止盈止损同时触发时假设止损先触发
)
```

三分类标签含义（side="both"）：

| 标签 | 含义 | 触发条件 |
|------|------|---------|
| 0 | 做空 | 做空止盈 或 做多止损 |
| 1 | 不操作 | 震荡/不确定 |
| 2 | 做多 | 做多止盈 或 做空止损 |

与 EventLabel 的区别：

```
EventLabel：独立检查最终收益和最大回撤（中间触发止损但最终回来也算好交易）
TripleBarrierLabel：按时间顺序逐根K线检查（触发止损就出局，不看后面）
```

### DirectionLabel（方向标签）

```python
from leek_core.ml.label import DirectionLabel

label_gen = DirectionLabel(
    periods=1,          # 未来 1 期
    threshold=0.01,     # 涨跌阈值 1%
    num_classes=3,      # 0=跌, 1=震荡, 2=涨
)
```

### FutureReturnLabel（未来收益率）

```python
from leek_core.ml.label import FutureReturnLabel

label_gen = FutureReturnLabel(
    periods=5,          # 未来 5 期收益率
    use_log=False,      # 是否用对数收益率
)
```

## 训练器

### 一览表

| 训练器 | 模型类型 | 适用场景 | 类名 |
|-------|---------|---------|------|
| XGBoostTrainer | 梯度提升树 | 快速验证、特征重要性分析 | `leek_core.ml.trainer.XGBoostTrainer` |
| GRUTrainer | 循环神经网络 | **时序建模（推荐）** | `leek_core.ml.trainer.GRUTrainer` |

### GRUTrainer

```python
from leek_core.ml.trainer import GRUTrainer

trainer = GRUTrainer(
    task_type="classification",  # "classification" 或 "regression"
    hidden_size=64,              # GRU 隐藏层大小
    num_layers=2,                # GRU 层数
    window_size=60,              # 滑动窗口大小（时序长度）
    dropout=0.2,                 # Dropout 比例
    bidirectional=False,         # 是否双向 GRU
    learning_rate=0.001,         # 学习率
    batch_size=32,               # 批大小
    epochs=100,                  # 最大训练轮数
    patience=10,                 # Early Stopping 耐心值
    device="cpu",                # "cpu", "cuda", "auto"
)
```

关键特点：
- 自动将扁平特征转换为时序窗口 `(batch, window_size, features)`
- 支持 Early Stopping 和学习率衰减
- 模型保存包含完整配置，加载后自动恢复

### XGBoostTrainer

```python
from leek_core.ml.trainer import XGBoostTrainer

trainer = XGBoostTrainer(
    task_type="classification",  # "classification" 或 "regression"
    max_depth=5,                 # 树最大深度
    learning_rate=0.05,          # 学习率
    n_estimators=500,            # 树的数量
    subsample=1.0,               # 子样本比例
)
```

## 策略集成

训练好的模型通过策略类在实盘/回测中使用：

### GRUStrategy（GRU 模型）

```python
from leek_core.strategy import GRUStrategy

strategy = GRUStrategy(
    model_config={
        "model_path": "/path/to/gru_model.joblib",
        "feature_config": {
            "factors": [...],          # 因子配置（与训练时一致）
            "encoder_classes": {...},   # 编码器配置
        }
    },
    mode="classification",       # 自动检测，或手动指定
    confidence_threshold=0.6,    # 置信度阈值
    return_threshold=0.02,       # 回归模式收益率阈值
    device="cpu",                # 推理设备
)
```

核心机制：内部维护滑动窗口 buffer，积累 `window_size` 根 K 线后开始推理。

### XGBoostStrategy（XGBoost 模型）

```python
from leek_core.strategy import XGBoostStrategy

strategy = XGBoostStrategy(
    model_config={
        "model_path": "/path/to/xgb_model.joblib",
        "feature_config": {
            "factors": [...],
            "encoder_classes": {...},
        }
    },
    mode="classification",
    confidence_threshold=0.6,
    return_threshold=0.02,
)
```

## 评估指标

### 分类任务

| 指标 | 含义 | 良好范围 |
|-----|------|---------|
| accuracy | 准确率 | > 0.5 |
| precision | 精确率 | > 0.5 |
| recall | 召回率 | > 0.5 |
| f1_score | F1 分数 | > 0.5 |
| roc_auc | AUC | > 0.6 |

### 回归任务

| 指标 | 含义 | 良好范围 |
|-----|------|---------|
| mse | 均方误差 | 越小越好 |
| mae | 平均绝对误差 | 越小越好 |
| r2 | R² 决定系数 | > 0 |

## 训练配置参考

### 配置字典完整字段

```python
config = {
    # === 数据配置 ===
    "symbols": ["BTC", "ETH"],              # 交易标的列表
    "timeframes": ["1m", "5m"],             # 时间框架列表
    "start_time": "2025-01-01",             # 开始时间
    "end_time": "2025-06-01",               # 结束时间
    
    "datasource_class": "model|KLineDatasource",
    "datasource_config": {},
    "datasource_extra": {
        "quote_currency": "USDT",
        "ins_type": "SWAP",
        "market": "okx"
    },
    
    # === 因子配置 ===
    "factors": [
        {
            "id": 1,
            "name": "factor_name",
            "class_name": "leek_core.ml.factors.Alpha158Factor",
            "params": {}
        }
    ],
    
    # === 标签配置 ===
    "label_generator": {
        "class_name": "leek_core.ml.label.TripleBarrierLabel",
        "params": {
            "hold_periods": 10,
            "take_profit": 0.02,
            "stop_loss": 0.01,
            "side": "both",
            "num_classes": 3,
        }
    },
    
    # === 训练器配置 ===
    "trainer": {
        "class_name": "leek_core.ml.trainer.GRUTrainer",
        "params": {
            "task_type": "classification",
            "hidden_size": 64,
            "num_layers": 2,
            "window_size": 60,
            "epochs": 100,
        }
    },
    
    # === 训练参数 ===
    "train_split_ratio": 0.8,               # 训练集占比
    "load_model_path": None,                # 加载旧模型路径（增量训练）
    "save_model_path": "/path/model.joblib", # 保存路径
    "mount_dirs": [],                        # 额外模块路径
}
```

## 推荐组合

### 短线择时（加密货币分钟级）

```python
# 标签：三重屏障，模拟真实交易
label_gen = TripleBarrierLabel(
    hold_periods=10,       # 10 根 K 线
    take_profit=0.015,     # 1.5% 止盈
    stop_loss=0.008,       # 0.8% 止损
    side="both",
    num_classes=3,
)

# 训练器：GRU 分类
trainer = GRUTrainer(
    task_type="classification",
    hidden_size=64,
    num_layers=2,
    window_size=60,        # 60 根 K 线作为输入
    epochs=100,
    patience=10,
)
```

### 趋势跟踪

```python
# 标签：方向标签
label_gen = DirectionLabel(periods=5, threshold=0.02, num_classes=3)

# 训练器：XGBoost（快速验证）
trainer = XGBoostTrainer(task_type="classification", n_estimators=500)
```

### 收益率预测

```python
# 标签：未来收益率
label_gen = FutureReturnLabel(periods=5)

# 训练器：GRU 回归
trainer = GRUTrainer(task_type="regression", window_size=60)
```

## 最佳实践

### 1. 数据量要求

```
XGBoost：至少 5000 条数据（较少数据也能工作）
GRU：至少 10000 条数据（时序模型需要更多数据）
     窗口大小 60 + 持仓期 10 = 至少 70 条连续数据才能生成 1 条样本
```

### 2. 特征与标签对齐

训练引擎自动处理对齐和 NaN 清理，无需手动处理。

### 3. 避免未来数据泄露

- 标签生成器使用 `shift(-periods)` 确保标签来自未来
- 切分数据按时间顺序（前 80% 训练，后 20% 验证）
- **不使用随机切分**

### 4. 模型保存与加载

```python
# 训练后保存
trainer.save_model(path="/path/to/model.joblib")

# 加载使用
trainer.load_model(path="/path/to/model.joblib")
result = trainer.predict(X_test)
```

### 5. 训练进度监控

```python
from multiprocessing import Queue

queue = Queue()
result = training(config, queue=queue)

# 读取进度
while True:
    msg = queue.get()
    print(f"Phase: {msg['phase']}, Status: {msg['status']}")
    if msg['status'] in ('completed', 'failed'):
        break
```
