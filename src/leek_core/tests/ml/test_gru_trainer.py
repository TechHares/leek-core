#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU 训练器测试用例

测试 GRUTrainer 的训练、预测、保存和加载功能。
"""
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from leek_core.ml.trainer import GRUTrainer
from leek_core.ml.label import TripleBarrierLabel, FutureReturnLabel


def load_data():
    """从 data.csv 加载数据"""
    data_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(data_path)
    # 确保列类型正确（TripleBarrierLabel 需要 float 类型）
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def generate_features(df: pd.DataFrame):
    """
    生成技术指标特征
    """
    # 价格相关特征（归一化）
    df["return_1"] = df["close"].pct_change()  # 收益率
    df["return_5"] = df["close"].pct_change(5)  # 5周期收益率
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]  # 振幅
    df["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]  # 实体
    
    # 均线特征
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    
    # 均线偏离度
    df["ma5_bias"] = (df["close"] - df["ma5"]) / df["ma5"]
    df["ma10_bias"] = (df["close"] - df["ma10"]) / df["ma10"]
    df["ma20_bias"] = (df["close"] - df["ma20"]) / df["ma20"]
    
    # 成交量特征
    df["volume_ma5"] = df["volume"].rolling(window=5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"]
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df


def generate_label(df: pd.DataFrame, hold_periods: int = 10, take_profit: float = 0.015, stop_loss: float = 0.008):
    """
    使用三重屏障标签（TripleBarrierLabel）生成标签
    
    模拟真实交易：逐根 K 线检查止盈/止损/到期，哪个先触发就按哪个结算。
    
    Args:
        df: DataFrame，需包含 open, high, low, close 列
        hold_periods: 最大持仓周期（K线根数）
        take_profit: 止盈比例（如0.015表示1.5%）
        stop_loss: 止损比例（如0.008表示0.8%）
    
    Returns:
        0 = 做空信号（做空止盈 或 做多止损）
        1 = 不操作（震荡/不确定）
        2 = 做多信号（做多止盈 或 做空止损）
    """
    label_gen = TripleBarrierLabel(
        hold_periods=hold_periods,
        take_profit=take_profit,
        stop_loss=stop_loss,
        side="both",
        num_classes=3,
        conservative=True,
    )
    
    labels = label_gen.generate(df)
    df["label"] = labels
    return df


def evaluate_model(y_test, y_pred, y_proba=None):
    """评估模型性能"""
    accuracy = accuracy_score(y_test, y_pred)
    
    # 多分类指标
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 打印评估结果
    print("=" * 60)
    print("GRU 模型评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision, weighted): {precision:.4f}")
    print(f"召回率 (Recall, weighted): {recall:.4f}")
    print(f"F1分数 (F1-Score, weighted): {f1:.4f}")
    
    print("\n混淆矩阵:")
    print(f"              预测")
    print(f"          做空  不操作  做多")
    for i, row_name in enumerate(['做空', '不操作', '做多']):
        row = "  ".join([f"{v:4d}" for v in cm[i]])
        print(f"实际 {row_name}  {row}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['做空', '不操作', '做多']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }


def test_gru_trainer():
    """测试 GRU 训练器"""
    print("\n" + "=" * 60)
    print("开始 GRU 训练器测试")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 正在加载数据...")
    df = load_data()
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 2. 生成特征和标签
    print("\n[2/6] 正在生成特征和标签（三重屏障标签）...")
    df = generate_features(df)
    df = generate_label(df, hold_periods=10, take_profit=0.01, stop_loss=0.015)
    print(f"  标签方法: TripleBarrierLabel (hold=10, tp=1.5%, sl=0.8%, side=both)")
    
    # 删除 NaN
    df = df.dropna()
    print(f"清理后数据量: {len(df)} 条记录")
    
    # 3. 准备特征和标签
    feature_cols = [
        "return_1", "return_5", "high_low_ratio", "close_open_ratio",
        "ma5_bias", "ma10_bias", "ma20_bias",
        "volume_ratio", "rsi"
    ]
    
    X = df[feature_cols].astype(np.float32)
    y = df["label"].astype(int)
    
    # 检查标签分布
    print(f"\n标签分布:")
    print(f"  做空 (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"  不操作 (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(f"  做多 (2): {(y == 2).sum()} ({(y == 2).mean():.1%})")
    
    # 4. 划分数据集（按时间顺序）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 再从训练集中划分验证集
    val_split_idx = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train_final)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")
    
    # 5. 创建并训练 GRU 模型
    print("\n[3/6] 正在训练 GRU 模型...")
    trainer = GRUTrainer(
        task_type="classification",
        window_size=20,           # 回看20根K线
        hidden_size=32,           # 较小的隐藏层（数据量不大）
        num_layers=1,             # 单层GRU
        dropout=0.2,
        bidirectional=False,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,                # 训练轮数
        early_stopping_patience=10,
        random_state=42,
        device="cpu",
    )
    
    # 定义进度回调
    def progress_callback(epoch, total, metrics):
        if epoch % 10 == 0 or epoch == 1:
            msg = f"  Epoch {epoch}/{total}"
            for k, v in metrics.items():
                if isinstance(v, float):
                    msg += f" - {k}: {v:.4f}"
            print(msg)
    
    trainer.train(
        X_train_final, 
        y_train_final, 
        X_val, 
        y_val,
        progress_callback=progress_callback
    )
    print("模型训练完成!")
    
    # 6. 在测试集上评估
    print("\n[4/6] 正在评估模型...")
    result = trainer.predict(X_test)
    
    y_pred = result['y_pred']
    y_proba = result.get('y_proba')
    
    # 对齐索引（GRU 会丢弃前 window_size 个样本）
    y_test_aligned = y_test.iloc[trainer.window_size:]
    
    print(f"\n预测结果数量: {len(y_pred)}")
    print(f"测试标签数量: {len(y_test_aligned)}")
    
    # 评估
    eval_results = evaluate_model(y_test_aligned, y_pred, y_proba)
    
    # 7. 测试模型保存和加载
    print("\n[5/6] 测试模型保存和加载...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "gru_model.pkl")
        
        # 保存
        trainer.save_model(model_path)
        print(f"  模型已保存到: {model_path}")
        
        # 加载
        new_trainer = GRUTrainer()
        new_trainer.load_model(path=model_path)
        print("  模型加载成功!")
        
        # 验证加载后的模型预测结果一致
        result_loaded = new_trainer.predict(X_test)
        y_pred_loaded = result_loaded['y_pred']
        
        # 比较预测结果
        match_rate = (y_pred.values == y_pred_loaded.values).mean()
        print(f"  加载后预测一致率: {match_rate:.2%}")
        
        assert match_rate > 0.99, "加载后的模型预测结果应与原模型一致"
    
    print("\n[6/6] 测试完成!")
    print("=" * 60)
    print("GRU 训练器测试通过!")
    print("=" * 60)
    
    return eval_results


def test_gru_trainer_regression():
    """测试 GRU 回归任务"""
    print("\n" + "=" * 60)
    print("开始 GRU 回归任务测试")
    print("=" * 60)
    
    # 加载数据
    df = load_data()
    df = generate_features(df)
    
    # 回归标签：使用 FutureReturnLabel 生成未来收益率
    label_gen = FutureReturnLabel(periods=3, use_log=False)
    df["label"] = label_gen.generate(df)
    df = df.dropna()
    print(f"  标签方法: FutureReturnLabel (periods=3)")
    
    feature_cols = [
        "return_1", "return_5", "high_low_ratio", "close_open_ratio",
        "ma5_bias", "ma10_bias", "ma20_bias",
        "volume_ratio", "rsi"
    ]
    
    X = df[feature_cols].astype(np.float32)
    y = df["label"].astype(np.float32)
    
    # 划分数据集
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split_idx = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    # 训练回归模型
    print("\n正在训练 GRU 回归模型...")
    trainer = GRUTrainer(
        task_type="regression",
        window_size=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=30,
        early_stopping_patience=10,
        random_state=42,
    )
    
    trainer.train(X_train_final, y_train_final, X_val, y_val)
    
    # 预测
    result = trainer.predict(X_test)
    y_pred = result['y_pred']
    
    # 对齐
    y_test_aligned = y_test.iloc[trainer.window_size:]
    
    # 计算回归指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test_aligned, y_pred)
    mae = mean_absolute_error(y_test_aligned, y_pred)
    r2 = r2_score(y_test_aligned, y_pred)
    
    print(f"\n回归评估结果:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.4f}")
    
    # 根据预测值生成交易信号
    threshold = 0.003
    signals = pd.Series(index=y_pred.index, dtype=int)
    signals[y_pred > threshold] = 2   # 做多
    signals[y_pred < -threshold] = 0  # 做空
    signals[(y_pred >= -threshold) & (y_pred <= threshold)] = 1  # 不操作
    
    print(f"\n根据回归预测生成的交易信号分布:")
    print(f"  做空 (0): {(signals == 0).sum()} ({(signals == 0).mean():.1%})")
    print(f"  不操作 (1): {(signals == 1).sum()} ({(signals == 1).mean():.1%})")
    print(f"  做多 (2): {(signals == 2).sum()} ({(signals == 2).mean():.1%})")
    
    print("\nGRU 回归任务测试完成!")


def test_gru_with_categorical_features():
    """
    测试 GRU 训练器的 categorical embedding 功能
    
    验证当存在 categorical 特征时，GRU 模型能正确：
    1. 接受 categorical_info 参数
    2. 创建 Embedding 层
    3. 训练和预测
    4. 保存和加载包含 categorical 信息的模型
    """
    print("\n" + "=" * 60)
    print("开始 GRU Categorical Embedding 测试")
    print("=" * 60)
    
    # 1. 加载数据并生成特征
    df = load_data()
    df = generate_features(df)
    
    # 添加 categorical 特征（模拟时间特征）
    if 'start_time' in df.columns:
        dt = pd.to_datetime(df['start_time'], unit='ms')
        df['hour'] = dt.dt.hour         # 0-23
        df['day_of_week'] = dt.dt.dayofweek  # 0-6
    else:
        # 如果没有 start_time，用随机数据模拟
        np.random.seed(42)
        df['hour'] = np.random.randint(0, 24, size=len(df))
        df['day_of_week'] = np.random.randint(0, 7, size=len(df))
    
    # 生成标签
    df = generate_label(df, hold_periods=10, take_profit=0.01, stop_loss=0.015)
    df = df.dropna()
    
    # 2. 准备特征（混合 numeric 和 categorical）
    feature_cols = [
        "return_1", "return_5", "high_low_ratio", "close_open_ratio",
        "ma5_bias", "ma10_bias", "ma20_bias",
        "volume_ratio", "rsi",
        "hour", "day_of_week",  # categorical 特征
    ]
    
    categorical_info = {
        "hour": 24,
        "day_of_week": 7,
    }
    
    X = df[feature_cols].astype(np.float32)
    y = df["label"].astype(int)
    
    print(f"特征数: {len(feature_cols)} (numeric: {len(feature_cols) - len(categorical_info)}, categorical: {len(categorical_info)})")
    print(f"Categorical 特征: {categorical_info}")
    
    # 3. 划分数据集
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split_idx = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"训练集: {len(X_train_final)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 4. 训练带 categorical embedding 的 GRU 模型
    print("\n正在训练带 Embedding 的 GRU 模型...")
    trainer = GRUTrainer(
        task_type="classification",
        window_size=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=30,
        early_stopping_patience=10,
        random_state=42,
        device="cpu",
    )
    
    trainer.train(
        X_train_final,
        y_train_final,
        X_val,
        y_val,
        categorical_info=categorical_info,
    )
    print("训练完成!")
    
    # 验证模型存储了 categorical 信息
    assert trainer._categorical_info == categorical_info, "Trainer 应存储 categorical_info"
    assert trainer._model.categorical_info == categorical_info, "Model 应存储 categorical_info"
    assert len(trainer._model.categorical_indices) == 2, "应有 2 个 categorical 特征索引"
    assert len(trainer._model.numeric_indices) == 9, "应有 9 个 numeric 特征索引"
    assert len(trainer._model.network.embeddings) == 2, "应有 2 个 Embedding 层"
    print("模型结构验证通过!")
    
    # 5. 预测
    result = trainer.predict(X_test)
    y_pred = result['y_pred']
    y_test_aligned = y_test.iloc[trainer.window_size:]
    
    accuracy = (y_pred.values == y_test_aligned.values).mean()
    print(f"预测准确率: {accuracy:.4f}")
    
    # 6. 测试保存和加载
    print("\n测试模型保存和加载（含 categorical 信息）...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "gru_cat_model.pkl")
        
        # 保存
        trainer.save_model(model_path)
        
        # 加载
        new_trainer = GRUTrainer(device="cpu")
        new_trainer.load_model(path=model_path)
        
        # 验证加载的 categorical 信息
        assert new_trainer._categorical_info == categorical_info, "加载后 categorical_info 应一致"
        assert new_trainer._model.categorical_info == categorical_info, "加载后 model.categorical_info 应一致"
        
        # 验证预测一致性
        result_loaded = new_trainer.predict(X_test)
        y_pred_loaded = result_loaded['y_pred']
        
        match_rate = (y_pred.values == y_pred_loaded.values).mean()
        print(f"  加载后预测一致率: {match_rate:.2%}")
        assert match_rate > 0.99, "加载后的模型预测结果应一致"
    
    print("\nGRU Categorical Embedding 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行分类测试
    test_gru_trainer()
    
    # 运行回归测试
    test_gru_trainer_regression()
    
    # 运行 categorical embedding 测试
    test_gru_with_categorical_features()
