#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost 训练器实现

支持分类和回归任务，提供丰富的超参数配置。
"""
from typing import Any, Callable, Dict, Optional

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback

from leek_core.models import Field, FieldType, ChoiceType

from .base import BaseTrainer
class ProgressCallback(TrainingCallback):
    def __init__(self, callback, total_iterations):
        self.callback = callback
        self.total_iterations = total_iterations
    
    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        self.callback(epoch, self.total_iterations, evals_log)
        return False

class XGBoostTrainer(BaseTrainer):
    """
    XGBoost 训练器
    
    支持分类和回归任务，通过 task_type 参数选择。
    """
    
    display_name = "XGBoost训练器"
    
    init_params = [
        Field(
            name="task_type",
            label="任务类型",
            type=FieldType.RADIO,
            default="classification",
            description="选择任务类型：classification（分类）或 regression（回归）",
            required=True,
            choices=[("classification", "分类"), ("regression", "回归")],
            choice_type=ChoiceType.STRING,
        ),
        Field(
            name="max_depth",
            label="最大深度",
            type=FieldType.INT,
            default=5,
            description="树的最大深度，范围 1-20",
            min=1,
            max=20,
            required=False,
        ),
        Field(
            name="learning_rate",
            label="学习率",
            type=FieldType.FLOAT,
            default=0.05,
            description="学习率（步长），范围 0.001-1.0",
            min=0.001,
            max=1.0,
            required=False,
        ),
        Field(
            name="n_estimators",
            label="树的数量",
            type=FieldType.INT,
            default=500,
            description="弱学习器（树）的数量，范围 1-10000",
            min=1,
            max=10000,
            required=False,
        ),
        Field(
            name="subsample",
            label="子样本比例",
            type=FieldType.FLOAT,
            default=1.0,
            description="每棵树使用的样本比例，范围 0.1-1.0",
            min=0.1,
            max=1.0,
            required=False,
        ),
        Field(
            name="colsample_bytree",
            label="特征采样比例",
            type=FieldType.FLOAT,
            default=1.0,
            description="每棵树使用的特征比例，范围 0.1-1.0",
            min=0.1,
            max=1.0,
            required=False,
        ),
        Field(
            name="min_child_weight",
            label="最小子节点权重",
            type=FieldType.INT,
            default=1,
            description="叶子节点最小权重和，范围 1-100",
            min=1,
            max=100,
            required=False,
        ),
        Field(
            name="gamma",
            label="最小损失减少",
            type=FieldType.FLOAT,
            default=0,
            description="节点分裂所需的最小损失减少，范围 0-100",
            min=0,
            max=100,
            required=False,
        ),
        Field(
            name="reg_alpha",
            label="L1正则化",
            type=FieldType.FLOAT,
            default=0,
            description="L1正则化系数，范围 0-10",
            min=0,
            max=10,
            required=False,
        ),
        Field(
            name="reg_lambda",
            label="L2正则化",
            type=FieldType.FLOAT,
            default=1,
            description="L2正则化系数，范围 0-10",
            min=0,
            max=10,
            required=False,
        ),
        Field(
            name="random_state",
            label="随机种子",
            type=FieldType.INT,
            default=None,
            description="随机种子，用于结果复现",
            required=False,
        ),
        Field(
            name="n_jobs",
            label="并行线程数",
            type=FieldType.INT,
            default=-1,
            description="并行线程数，-1表示使用所有CPU核心",
            min=-1,
            max=64,
            required=False,
        ),
        Field(
            name="early_stopping_rounds",
            label="早停轮数",
            type=FieldType.INT,
            default=None,
            description="早停轮数，如果验证集性能在指定轮数内没有提升则停止训练",
            min=1,
            max=1000,
            required=False,
        ),
        Field(
            name="eval_metric",
            label="评估指标",
            type=FieldType.STRING,
            default=None,
            description="评估指标，分类常用：logloss/auc，回归常用：rmse/mae。如果为None，会根据task_type自动选择",
            required=False,
        ),
    ]
    
    def __init__(
        self,
        task_type: str = "classification",
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
    ):
        """
        初始化 XGBoost 训练器
        
        :param task_type: 任务类型，"classification"（分类）或 "regression"（回归）
        :param max_depth: 树的最大深度，范围 1-20
        :param learning_rate: 学习率（步长），范围 0.001-1.0
        :param n_estimators: 弱学习器（树）的数量，范围 1-10000
        :param subsample: 每棵树使用的样本比例，范围 0.1-1.0
        :param colsample_bytree: 每棵树使用的特征比例，范围 0.1-1.0
        :param min_child_weight: 叶子节点最小权重和，范围 1-100
        :param gamma: 节点分裂所需的最小损失减少，范围 0-100
        :param reg_alpha: L1正则化系数，范围 0-10
        :param reg_lambda: L2正则化系数，范围 0-10
        :param random_state: 随机种子，用于结果复现
        :param n_jobs: 并行线程数，-1表示使用所有CPU核心
        :param early_stopping_rounds: 早停轮数，如果验证集性能在指定轮数内没有提升则停止训练
        :param eval_metric: 评估指标，分类常用：logloss/auc，回归常用：rmse/mae。如果为None，会根据task_type自动选择
        """
        super().__init__()
        
        # 验证任务类型
        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        self.task_type = task_type
        
        # 保存模型参数
        self.model_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        
        # 早停和评估指标
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        
        # 保存特征名称（用于后续使用）
        self._feature_names = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        sample_weight: Optional[Any] = None,
    ):
        """
        训练 XGBoost 模型
        
        如果 self._model 已存在，将在现有模型基础上继续训练（增量训练）。
        
        :param X_train: 训练集特征 DataFrame
        :param y_train: 训练集标签 Series
        :param X_val: 验证集特征 DataFrame（可选）
        :param y_val: 验证集标签 Series（可选）
        :param progress_callback: 进度回调函数，格式为 callback(current_iteration, total_iterations, metrics_dict)
        :param sample_weight: 样本权重（可选），用于类别不平衡时提高少数类权重，如 sklearn.utils.compute_sample_weight('balanced', y_train)
        """
        # 保存特征名称
        self._feature_names = list(X_train.columns)
        
        # 设置默认评估指标（如果尚未设置）
        if self.eval_metric is None:
            if self.task_type == "classification":
                self.eval_metric = "logloss"
            else:
                self.eval_metric = "rmse"
        
        # 检查是否存在已有模型，如果存在则使用继续训练模式
        use_continue_training = (
            self._model is not None and 
            isinstance(self._model, (XGBClassifier, XGBRegressor))
        )
        
        if use_continue_training:
            # 继续训练模式：使用已有模型
            model = self._model
        else:
            # 新建模型：将 eval_metric 和 early_stopping_rounds 添加到构造参数中
            model_params_with_eval = self.model_params.copy()
            if self.eval_metric is not None:
                model_params_with_eval["eval_metric"] = self.eval_metric
            if self.early_stopping_rounds is not None:
                model_params_with_eval["early_stopping_rounds"] = self.early_stopping_rounds
            
            if self.task_type == "classification":
                model = XGBClassifier(**model_params_with_eval)
            else:  # regression
                model = XGBRegressor(**model_params_with_eval)
        # 如果有进度回调，创建 XGBoost callback
        if progress_callback is not None:
            # 获取总迭代数：优先使用模型参数，如果没有则使用默认值
            if use_continue_training:
                # 继续训练模式：从已有模型获取 n_estimators，如果没有则使用默认值
                total_iterations = getattr(model, 'n_estimators', None) or self.model_params.get("n_estimators", 500)
            else:
                # 新建模型：从模型参数获取
                total_iterations = self.model_params.get("n_estimators", 500)
            model.callbacks = [ProgressCallback(progress_callback, total_iterations)]
        # 准备训练参数
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        
        # 如果有验证集，设置验证集
        # 注意：eval_metric 和 early_stopping_rounds 在构造函数中设置
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
        
        # 训练模型
        model.fit(X_train, y_train, **fit_params, verbose=False)
        
        # 更新内部模型引用
        self._model = model
    
    def predict(self, X_test: pd.DataFrame) -> dict:
        """
        预测结果
        
        根据任务类型返回相应的数据格式：
        - 分类任务：返回 {'y_pred': Series, 'y_proba': Series（正类概率）}
        - 回归任务：返回 {'y_pred': Series}
        
        :param X_test: 测试集特征 DataFrame
        :return: 预测结果字典
        """
        y_pred = self._model.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index)
        
        result = {'y_pred': y_pred_series}
        
        # 如果是分类任务，添加概率预测
        if self.task_type == "classification":
            proba = self._model.predict_proba(X_test)
            # 对于二分类，返回正类（第二列）的概率
            if proba.shape[1] == 2:
                result['y_proba'] = pd.Series(proba[:, 1], index=X_test.index)
            else:
                # 多分类，返回完整的概率 DataFrame
                result['y_proba'] = pd.DataFrame(proba, index=X_test.index)
        
        return result
    

