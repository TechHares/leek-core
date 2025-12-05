#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多标签融合（高级）

适用于：机器学习策略

融合多个标签，提供丰富信息，避免过拟合。
可以同时学习多个目标，提高模型的泛化能力。
"""
from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator

class MultiLabelFusion(LabelGenerator):
    """
    多标签融合（高级）
    
    融合多个标签生成器，同时生成多个标签列。
    提供丰富信息，避免过拟合，提高模型的泛化能力。
    
    参数:
        label_generators: 标签生成器列表，每个元素是一个 (LabelGenerator实例, 列名后缀) 的元组
        label_name: 基础标签列名前缀，默认"label"
        fusion_method: 融合方法，"concat"=拼接多个标签列，"weighted"=加权融合，默认"concat"
        weights: 权重列表（仅当fusion_method="weighted"时使用），默认None（等权重）
    
    适用策略:
        - 机器学习：提供丰富信息，避免过拟合
    
    示例:
        >>> from leek_core.ml.label import FutureReturnLabel, DirectionLabel, RankLabel
        >>> 
        >>> label_gen1 = FutureReturnLabel({"periods": 5})
        >>> label_gen2 = DirectionLabel({"periods": 5, "threshold": 0.01})
        >>> label_gen3 = RankLabel({"periods": 5})
        >>> 
        >>> fusion = MultiLabelFusion({
        ...     "label_generators": [
        ...         (label_gen1, "return"),
        ...         (label_gen2, "direction"),
        ...         (label_gen3, "rank")
        ...     ],
        ...     "fusion_method": "concat"
        ... })
        >>> df = fusion.generate(df_raw)
        >>> # 结果会有三列：label_return, label_direction, label_rank
    """
    display_name = "多标签融合"
    init_params = [
        Field(name="label_generators", label="标签生成器列表", type=FieldType.ARRAY, default=[], description="标签生成器列表，每个元素是一个 (LabelGenerator实例, 列名后缀) 的元组"),
        Field(name="fusion_method", label="融合方法", type=FieldType.RADIO, default="concat",
              choices=[("concat", "拼接多个标签列"), ("weighted", "加权融合")],
              description="融合方法"),
        Field(name="weights", label="权重列表", type=FieldType.ARRAY, default=None, description="权重列表（仅当fusion_method='weighted'时使用），默认None（等权重）"),
    ]
    
    def __init__(self, label_generators: List[tuple] = None,
                 fusion_method: str = "concat", weights: List[float] = None):
        super().__init__()
        self.label_generators: List[tuple] = label_generators if label_generators is not None else []
        self.fusion_method = fusion_method
        self.weights = weights
        
        if not self.label_generators:
            raise ValueError("label_generators cannot be empty")
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成融合标签
        
        :param df: 原始 DataFrame
        :return: 融合后的标签 Series
        """
        if self.fusion_method == "concat":
            # 拼接模式：多个标签加权平均（因为训练引擎只支持单标签）
            # 注意：如果需要多标签，需要修改训练引擎支持
            if self.weights is None:
                weights = [1.0 / len(self.label_generators)] * len(self.label_generators)
            else:
                weights = self.weights
            
            label_values = []
            for label_gen, _ in self.label_generators:
                label_series = label_gen.generate(df)
                label_values.append(label_series.values)
            
            # 加权平均
            fused_label = np.zeros(len(df))
            for i in range(len(self.label_generators)):
                fused_label += weights[i] * label_values[i]
            
            return pd.Series(fused_label, name=self.label_name)
        elif self.fusion_method == "weighted":
            # 加权融合模式：多个标签加权平均
            if self.weights is None:
                weights = [1.0 / len(self.label_generators)] * len(self.label_generators)
            else:
                weights = self.weights
            
            label_values = []
            for label_gen, _ in self.label_generators:
                label_series = label_gen.generate(df)
                label_values.append(label_series.values)
            
            # 加权平均
            fused_label = np.zeros(len(df))
            for i in range(len(self.label_generators)):
                fused_label += weights[i] * label_values[i]
            
            return pd.Series(fused_label, name=self.label_name)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
    
    def get_output_names(self) -> List[str]:
        """获取所有输出列名"""
        if self.fusion_method == "concat":
            return [f"{self.label_name}_{suffix}" for _, suffix in self.label_generators]
        else:
            return [self.label_name]

