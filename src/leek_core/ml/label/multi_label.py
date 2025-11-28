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
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.label_generators: List[tuple] = params.get("label_generators", [])
        self.fusion_method = params.get("fusion_method", "concat")
        self.weights = params.get("weights", None)
        
        if not self.label_generators:
            raise ValueError("label_generators cannot be empty")
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成融合标签
        
        :param df: 原始 DataFrame
        :return: 增加了多个 label 列的 DataFrame
        """
        if self.fusion_method == "concat":
            # 拼接模式：每个生成器生成一列
            for label_gen, suffix in self.label_generators:
                original_name = label_gen.label_name
                label_gen.label_name = f"{self.label_name}_{suffix}"
                df = label_gen.generate(df)
                label_gen.label_name = original_name  # 恢复原名
        elif self.fusion_method == "weighted":
            # 加权融合模式：多个标签加权平均
            if self.weights is None:
                weights = [1.0 / len(self.label_generators)] * len(self.label_generators)
            else:
                weights = self.weights
            
            label_values = []
            for label_gen, _ in self.label_generators:
                temp_df = df.copy()
                temp_df = label_gen.generate(temp_df)
                label_values.append(temp_df[label_gen.label_name].values)
            
            # 加权平均
            fused_label = np.zeros(len(df))
            for i, (label_gen, _) in enumerate(self.label_generators):
                fused_label += weights[i] * label_values[i]
            
            df[self.label_name] = fused_label
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        
        return df
    
    def get_output_names(self) -> List[str]:
        """获取所有输出列名"""
        if self.fusion_method == "concat":
            return [f"{self.label_name}_{suffix}" for _, suffix in self.label_generators]
        else:
            return [self.label_name]

