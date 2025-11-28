#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .feature_engine import FeatureEngine
from .label.base import LabelGenerator

class TrainingDataBuilder:
    """
    训练数据构建器
    组合特征工程和标签生成，生成完整的训练数据集
    """
    
    def __init__(
        self,
        feature_config: List[Dict[str, Any]],
        label_generator: LabelGenerator
    ):
        """
        初始化训练数据构建器
        
        :param feature_config: 特征配置（传给 FeatureEngine）
        :param label_generator: 标签生成器实例
        """
        self.feature_engine = FeatureEngine(feature_config)
        self.label_generator = label_generator
    
    def build(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        构建完整的训练数据集
        
        :param df: 原始 K线数据（包含 open, high, low, close, volume）
        :param drop_na: 是否删除包含 NaN 的行（通常是 label 为 NaN 的行，即最后几行）
        :return: 包含特征和标签的完整 DataFrame
        """
        # 1. 计算特征
        df = self.feature_engine.compute_all(df)
        
        # 2. 生成标签
        df = self.label_generator.generate(df)
        
        # 3. 删除 NaN（通常是最后几行，因为未来数据不存在）
        if drop_na:
            df = df.dropna()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取特征列名列表"""
        return self.feature_engine.get_feature_names()
    
    def get_label_name(self) -> str:
        """获取标签列名"""
        return self.label_generator.get_label_name()
    
    def split_features_labels(self, df: pd.DataFrame) -> tuple:
        """
        分离特征和标签
        
        :param df: 完整的训练数据 DataFrame
        :return: (X, y) 元组，X 是特征 DataFrame，y 是标签 Series
        """
        feature_names = self.get_feature_names()
        label_name = self.get_label_name()
        
        X = df[feature_names]
        y = df[label_name]
        
        return X, y

