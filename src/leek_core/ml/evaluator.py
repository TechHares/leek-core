#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估器模块

统一计算模型评估指标，支持分类和回归任务。
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from leek_core.utils import get_logger

logger = get_logger(__name__)

# 采样配置：最多保留的预测值数量（用于减少存储空间）
MAX_PREDICTION_SAMPLES = 5000
# 直方图 bins 数量
HISTOGRAM_BINS = 30


def _sample_array(arr: list, max_samples: int = MAX_PREDICTION_SAMPLES) -> list:
    """
    对数组进行采样，减少数据量
    
    :param arr: 原始数组
    :param max_samples: 最大采样数量
    :return: 采样后的数组
    """
    if not arr or len(arr) <= max_samples:
        return arr
    
    # 随机采样，保持数据分布
    indices = np.random.choice(len(arr), size=max_samples, replace=False)
    indices = np.sort(indices)  # 保持顺序
    return [arr[i] for i in indices]


def _compute_histogram(data: np.ndarray, bins: int = HISTOGRAM_BINS) -> Dict[str, Any]:
    """
    计算直方图 bins 数据
    
    :param data: 输入数据数组
    :param bins: bins 数量
    :return: 包含 bins 边界和计数的字典
    """
    if data is None or len(data) == 0:
        return {'bins': [], 'counts': []}
    
    # 过滤无效值
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return {'bins': [], 'counts': []}
    
    # 计算直方图
    counts, bin_edges = np.histogram(valid_data, bins=bins)
    
    # 计算每个 bin 的中心值（用于 x 轴标签）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'bins': bin_centers.tolist(),  # bin 中心值
        'counts': counts.tolist(),     # 每个 bin 的计数
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data))
    }


class ModelEvaluator:
    """
    模型评估器
    
    统一计算模型评估指标，支持分类和回归任务。
    """
    
    def __init__(self, task_type: str = "classification"):
        """
        初始化评估器
        
        :param task_type: 任务类型，"classification"（分类）或 "regression"（回归）
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        self.task_type = task_type
    
    def evaluate(
        self,
        y_true: pd.Series,
        predict_result: Optional[Dict[str, Any]] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_proba: Optional[pd.Series] = None,
        target_names: Optional[list] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估模型性能
        
        支持两种调用方式：
        1. 直接传入 predict() 返回的字典：
           evaluate(y_true, predict_result=predict_result)
        2. 分别传入预测结果：
           evaluate(y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
        
        :param y_true: 真实标签
        :param y_pred: 预测类别（分类）或预测值（回归），如果提供了 predict_result 则忽略
        :param y_pred_proba: 预测概率（仅分类任务，可选），如果提供了 predict_result 则忽略
        :param predict_result: predict() 方法返回的字典，包含 'y_pred' 和可选的 'y_proba'
        :param target_names: 类别名称列表（仅分类任务，用于显示）
        :param verbose: 是否打印评估结果
        :return: 评估结果字典
        """
        # 如果提供了 predict_result，从中提取数据
        if predict_result is not None:
            y_pred = predict_result.get('y_pred')
            y_proba = predict_result.get('y_proba')
            # y_proba 可能是 Series 或 DataFrame，统一处理
            if y_proba is not None:
                if isinstance(y_proba, pd.DataFrame):
                    # 如果是 DataFrame，取第二列（正类概率）或第一列
                    if y_proba.shape[1] >= 2:
                        y_pred_proba = y_proba.iloc[:, 1]
                    else:
                        y_pred_proba = y_proba.iloc[:, 0]
                else:
                    y_pred_proba = y_proba
        
        if y_pred is None:
            raise ValueError("必须提供 y_pred 或 predict_result")
        
        if self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred, y_pred_proba, target_names, verbose)
        else:
            return self._evaluate_regression(y_true, y_pred, verbose)
    
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: Optional[pd.Series],
        target_names: Optional[list],
        verbose: bool
    ) -> Dict[str, Any]:
        """
        评估分类任务
        
        :param y_true: 真实标签
        :param y_pred: 预测类别
        :param y_pred_proba: 预测概率（可选）
        :param target_names: 类别名称列表（可选）
        :param verbose: 是否打印结果
        :return: 评估结果字典
        """
        # 计算基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        recall = recall_score(y_true, y_pred, zero_division=0, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        f1 = f1_score(y_true, y_pred, zero_division=0, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        
        # 计算 AUC（需要概率，且是二分类）
        auc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    auc = roc_auc_score(y_true, y_pred_proba)
                else:
                    # 多分类需要使用不同的方法
                    logger.warning("多分类任务的AUC计算需要特殊处理，当前仅支持二分类")
            except ValueError as e:
                logger.warning(f"无法计算AUC: {e}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        
        # 打印结果
        if verbose:
            self._print_classification_results(
                accuracy, precision, recall, f1, auc, cm, report, target_names
            )
        
        # 对预测数组进行采样，减少存储空间（用于散点图）
        y_pred_list = y_pred.tolist()
        y_pred_sampled = _sample_array(y_pred_list)
        
        y_pred_proba_list = None
        y_pred_proba_sampled = None
        y_pred_proba_histogram = None
        
        if y_pred_proba is not None:
            y_pred_proba_list = y_pred_proba.tolist()
            y_pred_proba_sampled = _sample_array(y_pred_proba_list)
            # 计算预测概率分布直方图（用于直方图展示）
            y_pred_proba_histogram = _compute_histogram(y_pred_proba.values)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'y_pred': y_pred_sampled,  # 采样后的数据，用于散点图
            'y_pred_proba': y_pred_proba_sampled,  # 采样后的数据，用于散点图
            'y_pred_proba_histogram': y_pred_proba_histogram,  # 直方图 bins 数据
        }
    
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        评估回归任务
        
        :param y_true: 真实值
        :param y_pred: 预测值
        :param verbose: 是否打印结果
        :return: 评估结果字典
        """
        # 计算回归指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算平均绝对百分比误差（MAPE）
        # 避免除以接近0的值导致MAPE异常大
        # 只有当真实值的绝对值大于阈值时才计算MAPE，否则使用MAE的相对值
        threshold = 1e-6
        valid_mask = np.abs(y_true) > threshold
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
        else:
            # 如果所有真实值都接近0，使用MAE作为替代指标
            mape = None
        
        # 打印结果
        if verbose:
            self._print_regression_results(mse, rmse, mae, r2, mape)
        
        # 对预测数组进行采样，减少存储空间（用于散点图）
        y_pred_list = y_pred.tolist()
        y_pred_sampled = _sample_array(y_pred_list)
        
        # 计算残差分布直方图（用于直方图展示）
        residuals = y_true.values - y_pred.values
        residual_histogram = _compute_histogram(residuals)
        
        result = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'y_pred': y_pred_sampled,  # 采样后的数据，用于散点图
            'residual_histogram': residual_histogram,  # 残差分布直方图 bins 数据
        }
        if mape is not None:
            result['mape'] = float(mape)
        return result
    
    def _print_classification_results(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc: Optional[float],
        cm: np.ndarray,
        report: Dict,
        target_names: Optional[list]
    ):
        """打印分类结果"""
        print("=" * 60)
        print("模型评估结果（分类）")
        print("=" * 60)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        if auc is not None:
            print(f"AUC-ROC: {auc:.4f}")
        else:
            print("AUC-ROC: 无法计算")
        
        # 打印混淆矩阵
        print("\n混淆矩阵:")
        if len(cm) == 2:
            print(f"                预测")
            labels = target_names if target_names else ['类别0', '类别1']
            print(f"              {labels[0]}  {labels[1]}")
            print(f"实际  {labels[0]}   {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"      {labels[1]}   {cm[1][0]:4d}  {cm[1][1]:4d}")
        else:
            print(cm)
        
        # 打印详细分类报告（从report字典格式化输出）
        print("\n详细分类报告:")
        # 打印每个类别的指标
        for key in sorted(report.keys()):
            if key in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(report[key], dict):
                metrics = report[key]
                print(f"{key}: precision={metrics.get('precision', 0):.4f}, recall={metrics.get('recall', 0):.4f}, "
                      f"f1-score={metrics.get('f1-score', 0):.4f}, support={metrics.get('support', 0)}")
        # 打印总体指标
        if 'accuracy' in report:
            print(f"\naccuracy: {report['accuracy']:.4f}")
        if 'macro avg' in report:
            avg = report['macro avg']
            print(f"macro avg: precision={avg.get('precision', 0):.4f}, recall={avg.get('recall', 0):.4f}, "
                  f"f1-score={avg.get('f1-score', 0):.4f}, support={avg.get('support', 0)}")
        if 'weighted avg' in report:
            avg = report['weighted avg']
            print(f"weighted avg: precision={avg.get('precision', 0):.4f}, recall={avg.get('recall', 0):.4f}, "
                  f"f1-score={avg.get('f1-score', 0):.4f}, support={avg.get('support', 0)}")
    
    def _print_regression_results(
        self,
        mse: float,
        rmse: float,
        mae: float,
        r2: float,
        mape: float
    ):
        """打印回归结果"""
        print("=" * 60)
        print("模型评估结果（回归）")
        print("=" * 60)
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"R²分数 (R²-Score): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

