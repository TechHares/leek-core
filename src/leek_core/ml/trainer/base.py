#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器基类模块

提供模型训练的抽象接口，支持用户自定义训练逻辑。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import io

import joblib
import pandas as pd

from leek_core.base import LeekComponent


class BaseTrainer(LeekComponent, ABC):
    """
    训练器基类
    
    子类需要实现：
    - train(): 训练模型的具体逻辑
    - predict(): 预测结果，根据任务类型返回相应的数据格式
    """
    
    def __init__(self):
        """
        初始化训练器
        """
        super().__init__()
        self._model = None  # 训练好的模型
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ):
        """
        训练模型
        
        :param X_train: 训练集特征 DataFrame
        :param y_train: 训练集标签 Series
        :param X_val: 验证集特征 DataFrame（可选）
        :param y_val: 验证集标签 Series（可选）
        :param progress_callback: 进度回调函数，格式为 callback(current_iteration, total_iterations, metrics_dict)
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> dict:
        """
        预测结果
        
        子类根据任务类型返回相应的数据格式：
        - 分类任务：返回 {'y_pred': Series, 'y_proba': DataFrame或Series（可选）}
        - 回归任务：返回 {'y_pred': Series}
        
        :param X_test: 测试集特征 DataFrame
        :return: 预测结果字典
            - y_pred: 预测类别（分类）或预测值（回归）的 Series
            - y_proba: 预测概率（仅分类任务，可选），可以是 DataFrame（多列）或 Series（单列，如二分类的正类概率）
        """
        pass

    def save_model(self, path: Optional[str] = None) -> Optional[io.BytesIO]:
        """
        保存模型
        
        支持两种方式：
        1. 如果提供了 path，将模型保存到指定文件路径
        2. 如果未提供 path，返回包含模型数据的 BytesIO 对象
        
        :param path: 保存模型的文件路径（可选）
        :return: 如果未提供 path，返回 BytesIO 对象；否则返回 None
        """
        if self._model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        self._model.callbacks = None
        if path is not None:
            # 保存到文件
            joblib.dump(self._model, path)
            return None
        else:
            # 返回 BytesIO 对象
            model_io = io.BytesIO()
            joblib.dump(self._model, model_io)
            model_io.seek(0)  # 重置指针到开头
            return model_io

    def load_model(
        self, 
        path: Optional[str] = None, 
        model: Optional[Any] = None, 
        model_io: Optional[io.BytesIO] = None
    ):
        """
        加载模型
        
        支持三种方式（按优先级）：
        1. model: 直接传入模型对象
        2. model_io: 从 BytesIO 对象加载
        3. path: 从文件路径加载
        
        :param path: 模型文件路径（可选）
        :param model: 模型对象（可选）
        :param model_io: 包含模型数据的 BytesIO 对象（可选）
        """
        if model is not None:
            # 方式1: 直接使用传入的模型对象
            self._model = model
        elif model_io is not None:
            # 方式2: 从 BytesIO 加载
            model_io.seek(0)  # 确保指针在开头
            self._model = joblib.load(model_io)
        elif path is not None:
            # 方式3: 从文件路径加载
            self._model = joblib.load(path)
        else:
            raise ValueError(
                "No model source specified. Provide one of: path, model, or model_io"
            )