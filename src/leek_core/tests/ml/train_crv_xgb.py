#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRV XGBoost 模型训练（精简版 ~121 特征）

加载 prepare_crv_data.py 生成的数据，训练 XGBoost 三分类模型（做空 / 不操作 / 做多）。

相比 GRU：
- 不需要滑动窗口，直接用扁平特征
- 不需要 StandardScaler（树模型对尺度不敏感）
- 训练速度快得多（分钟级）
- 天然利用所有 CPU 核心

用法:
    1. 先运行 prepare_crv_data.py 生成 crv_m1_prepared.csv
    2. 再运行本脚本:  python train_crv_xgb.py
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from leek_core.ml.trainer.xgboost import XGBoostTrainer

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "crv_m1_prepared.csv"
FEATURE_NAMES_FILE = "crv_m1_feature_names.txt"

# 不参与特征的列
NON_FEATURE_COLS = {
    "start_time", "open", "high", "low", "close",
    "volume", "amount", "label", "vwap",
}

# XGBoost 超参数
MAX_DEPTH = 6               # 树的最大深度
LEARNING_RATE = 0.05        # 学习率
N_ESTIMATORS = 1000         # 最大树数量（由 early stopping 控制）
SUBSAMPLE = 0.8             # 每棵树使用 80% 样本（防过拟合）
COLSAMPLE_BYTREE = 0.8      # 每棵树使用 80% 特征（防过拟合）
MIN_CHILD_WEIGHT = 5        # 叶子节点最小权重（防过拟合）
GAMMA = 0.1                 # 最小损失减少
REG_ALPHA = 0.1             # L1 正则化
REG_LAMBDA = 1.0            # L2 正则化
EARLY_STOPPING = 50         # 早停耐心

# 数据集划分比例
TRAIN_RATIO = 0.7           # 前 70% 训练
VAL_RATIO = 0.15            # 从训练集末尾切出 15% 验证
# 后 30% 测试


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """
    加载 prepare_crv_data.py 生成的数据

    特征列自动从 feature_names.txt 或 CSV header 推断
    """
    data_path = os.path.join(DATA_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            f"请先运行 prepare_crv_data.py 生成数据。"
        )

    # 尝试从特征名文件读取
    feature_list_path = os.path.join(DATA_DIR, FEATURE_NAMES_FILE)
    if os.path.exists(feature_list_path):
        with open(feature_list_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  从 {FEATURE_NAMES_FILE} 读取 {len(feature_cols)} 个特征名")
    else:
        # 降级: 从 CSV header 推断
        feature_cols = None

    df = pd.read_csv(data_path)

    # 如果没有 feature_names.txt，从列名推断
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        print(f"  从 CSV header 推断 {len(feature_cols)} 个特征列")

    # 验证所有特征列都存在
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 中缺少 {len(missing)} 个特征列: {missing[:5]}...")

    # 类型转换
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    df["label"] = df["label"].astype(int)

    # 替换 inf
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feature_cols + ["label"])
    if len(df) < before:
        print(f"  清理 NaN/inf: {before} -> {len(df)}")

    return df, feature_cols


# ============================================================
# 模型评估
# ============================================================
def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 70)
    print("XGBoost 模型评估结果")
    print("=" * 70)
    print(f"准确率 (Accuracy):           {accuracy:.4f}")
    print(f"精确率 (Precision, weighted): {precision:.4f}")
    print(f"召回率 (Recall, weighted):    {recall:.4f}")
    print(f"F1分数 (F1-Score, weighted):  {f1:.4f}")

    labels = ['做空', '不操作', '做多']
    print("\n混淆矩阵:")
    print(f"{'':>10s}  {'预测':>18s}")
    header = f"{'':>10s}  " + "  ".join([f"{l:>6s}" for l in labels])
    print(header)
    for i, row_name in enumerate(labels):
        if i < len(cm):
            row = "  ".join([f"{v:6d}" for v in cm[i]])
            print(f"实际 {row_name}  {row}")

    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # 各类别预测分布
    pred_series = pd.Series(y_pred)
    print("预测分布:")
    for v, name in [(0, "做空"), (1, "不操作"), (2, "做多")]:
        cnt = (pred_series == v).sum()
        print(f"  {name} ({v}): {cnt} ({cnt / len(pred_series):.1%})")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    print("=" * 70)
    print("CRV XGBoost 模型训练（精简版 ~121 特征）")
    print("=" * 70)

    # ---- 1. 加载数据 ----
    print("\n[1/4] 正在加载数据...")
    df, feature_cols = load_data()
    print(f"  数据量: {len(df)} 行, 特征数: {len(feature_cols)}")

    X = df[feature_cols].astype(np.float32)
    y = df["label"].astype(int)

    # 标签分布
    print(f"\n  标签分布:")
    for v, name in [(0, "做空"), (1, "不操作"), (2, "做多")]:
        cnt = (y == v).sum()
        print(f"    {name} ({v}): {cnt:>8d} ({cnt / len(y):.1%})")

    # ---- 2. 划分数据集（时间顺序） ----
    print(f"\n[2/4] 划分数据集...")
    n = len(df)
    train_end = int(n * TRAIN_RATIO)

    X_train_full, X_test = X[:train_end], X[train_end:]
    y_train_full, y_test = y[:train_end], y[train_end:]

    # 从训练集末尾切出验证集
    val_size = int(len(X_train_full) * VAL_RATIO)
    val_start = len(X_train_full) - val_size
    X_train, X_val = X_train_full[:val_start], X_train_full[val_start:]
    y_train, y_val = y_train_full[:val_start], y_train_full[val_start:]

    print(f"  训练集: {len(X_train):>8d} 行 ({len(X_train) / n:.1%})")
    print(f"  验证集: {len(X_val):>8d} 行 ({len(X_val) / n:.1%})")
    print(f"  测试集: {len(X_test):>8d} 行 ({len(X_test) / n:.1%})")

    # XGBoost 不需要 StandardScaler（树模型对特征尺度不敏感）

    # ---- 3. 训练 XGBoost 模型 ----
    print(f"\n[3/4] 正在训练 XGBoost 模型...")
    print(f"  超参数:")
    print(f"    max_depth          = {MAX_DEPTH}")
    print(f"    learning_rate      = {LEARNING_RATE}")
    print(f"    n_estimators       = {N_ESTIMATORS}")
    print(f"    subsample          = {SUBSAMPLE}")
    print(f"    colsample_bytree   = {COLSAMPLE_BYTREE}")
    print(f"    min_child_weight   = {MIN_CHILD_WEIGHT}")
    print(f"    gamma              = {GAMMA}")
    print(f"    early_stopping     = {EARLY_STOPPING}")
    print(f"    features           = {len(feature_cols)}")
    print()

    trainer = XGBoostTrainer(
        task_type="classification",
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING,
        eval_metric="mlogloss",
    )

    def progress_callback(epoch, total, metrics):
        if (epoch + 1) % 50 == 0 or epoch == 0:
            msg = f"  Tree {epoch + 1:4d}/{total}"
            if isinstance(metrics, dict):
                for k, v_dict in metrics.items():
                    for metric_name, values in v_dict.items():
                        if values:
                            msg += f"  {k}_{metric_name}: {values[-1]:.4f}"
            print(msg)

    # 类别不平衡时提高少数类（做多/做空）权重，避免模型只会预测多数类（不操作）
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    train_start = time.time()
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        progress_callback=progress_callback,
        sample_weight=sample_weight,
    )
    train_elapsed = time.time() - train_start
    print(f"\n  训练完成! 耗时: {train_elapsed:.1f}s ({train_elapsed/60:.1f} 分钟)")

    # 特征重要性 Top20
    if hasattr(trainer._model, 'feature_importances_'):
        importances = trainer._model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n  特征重要性 Top 20:")
        for rank, idx in enumerate(indices[:20]):
            print(f"    {rank + 1:2d}. {feature_cols[idx]:>40s}  {importances[idx]:.4f}")

    # ---- 4. 在测试集上评估 ----
    print(f"\n[4/4] 正在评估模型...")
    result = trainer.predict(X_test)
    y_pred = result['y_pred']

    print(f"  预测样本数: {len(y_pred)}")
    print(f"  测试标签数: {len(y_test)}")
    print()

    eval_results = evaluate_model(y_test, y_pred)

    # ---- 保存模型 ----
    print(f"\n保存模型...")
    model_path = os.path.join(DATA_DIR, "crv_xgb_model.pkl")
    trainer.save_model(model_path)
    model_size = os.path.getsize(model_path) / 1024
    print(f"  模型已保存到: {model_path}")
    print(f"  模型大小: {model_size:.1f} KB")

    # ---- 汇总 ----
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"训练完成!")
    print(f"  特征数:   {len(feature_cols)}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1-Score: {eval_results['f1']:.4f}")
    print(f"  总耗时:   {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
    print(f"{'=' * 70}")
