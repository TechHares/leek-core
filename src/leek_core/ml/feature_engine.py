#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from leek_core.models import KLine

from .factors.base import DualModeFactor
from .factors.technical import ATRFactor, MAFactor, RSIFactor

class FeatureEngine:
    """
    特征工程引擎，支持流式和批量计算
    """
    
    # 简单的因子注册表
    FACTOR_REGISTRY = {
        "MA": MAFactor,
        "RSI": RSIFactor,
        "ATR": ATRFactor
    }

    def __init__(self, config: List[Dict[str, Any]]):
        """
        初始化特征引擎
        
        :param config: 因子配置列表
            [
                {"class": "MA", "params": {"window": 20}, "name": "MA_20"},
                {"class": "RSI", "params": {"window": 14}}
            ]
        """
        self.factors: List[DualModeFactor] = []
        self.feature_names: List[str] = []
        
        for item in config:
            cls_name = item.get("class")
            if cls_name not in self.FACTOR_REGISTRY:
                raise ValueError(f"Unknown factor class: {cls_name}")
            
            factor_cls = self.FACTOR_REGISTRY[cls_name]
            params = item.get("params", {})
            factor = factor_cls(params)
            
            self.factors.append(factor)
            # 兼容多因子模式
            self.feature_names.extend(factor.get_output_names())

    def update(self, kline: KLine) -> np.ndarray:
        """
        [实盘模式] 更新并返回当前特征向量
        """
        features = []
        for factor in self.factors:
            val = factor.update(kline)
            if isinstance(val, list):
                features.extend(val)
            else:
                features.append(val)
        return np.array(features).reshape(1, -1)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [训练模式] 批量计算所有特征
        """
        # 确保列名小写
        df = df.rename(columns={c: c.lower() for c in df.columns})
        
        # 链式调用，每个因子在 df 上增加列
        for factor in self.factors:
            df = factor.compute(df)
        
        return df

    def get_feature_names(self) -> List[str]:
        return self.feature_names
