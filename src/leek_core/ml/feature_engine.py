#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, List, Optional, Union
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from leek_core.models import KLine, TimeFrame

from .factors.base import DualModeFactor, FeatureSpec, FeatureType

class FeatureEngine:
    """
    特征工程引擎，支持流式和批量计算
    
    核心改进：
    - 使用 factor_id 作为特征名前缀，确保不同因子实例的输出名唯一
    - 存储 FeatureSpec 元数据，支持区分 NUMERIC 和 CATEGORICAL 特征
    - 自动处理 symbol 和 timeframe 的 label encoding
    """

    def __init__(
        self, 
        factors: List[DualModeFactor], 
        factor_ids: List[str]=None, 
        _call_back: Callable[[str, bool], None]=None,
        symbol_encoder: Optional[OneHotEncoder] = None,
        timeframe_encoder: Optional[LabelEncoder] = None,
        enable_symbol_timeframe_encoding: bool = True
    ):
        """
        :param factors: 因子列表
        :param factor_ids: 因子ID列表
        :param _call_back: 回调函数
        :param symbol_encoder: 预训练的 symbol OneHotEncoder（预测模式使用）
        :param timeframe_encoder: 预训练的 timeframe LabelEncoder（预测模式使用）
        :param enable_symbol_timeframe_encoding: 是否启用 symbol 和 timeframe 编码
        """
        self.factors: List[DualModeFactor] = factors
        self.feature_names: List[str] = []
        self.feature_specs: Dict[str, FeatureSpec] = {}  # full_name -> FeatureSpec
        
        for i in range(len(factors)):
            factor_id = factor_ids[i] if factor_ids is not None and len(factor_ids) >= len(factors) else f"{i}"
            setattr(self.factors[i], "_factor_id", factor_id)
            # 收集 specs 并加上 factor_id 前缀
            for spec in self.factors[i].get_output_specs():
                full_name = f"{factor_id}_{spec.name}"
                self.feature_names.append(full_name)
                self.feature_specs[full_name] = spec
        
        self._call_back = _call_back or (lambda factor_id, success: None)
        
        # Symbol 和 Timeframe 编码
        self.enable_symbol_timeframe_encoding = enable_symbol_timeframe_encoding
        if enable_symbol_timeframe_encoding:
            # Symbol 使用 OneHotEncoder
            if symbol_encoder is not None:
                self.symbol_encoder = symbol_encoder
                self._symbol_encoder_fitted = True
                # 生成 symbol one-hot 特征名称
                if hasattr(self.symbol_encoder, 'categories_') and len(self.symbol_encoder.categories_) > 0:
                    symbol_feature_names = [f'symbol_{cat}' for cat in self.symbol_encoder.categories_[0]]
                    self.feature_names.extend(symbol_feature_names)
                    for sfn in symbol_feature_names:
                        self.feature_specs[sfn] = FeatureSpec(name=sfn)
            else:
                # 训练模式：稍后通过 fit_encoders 拟合
                self.symbol_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self._symbol_encoder_fitted = False
            
            # Timeframe 使用 LabelEncoder（固定值）
            if timeframe_encoder is not None:
                self.timeframe_encoder = timeframe_encoder
                self._timeframe_encoder_fitted = True
            else:
                self.timeframe_encoder = LabelEncoder()
                self._timeframe_encoder_fitted = False
                # 在训练模式下，立即使用固定的 TimeFrame 枚举值拟合编码器
                all_timeframe_values = [tf.value for tf in TimeFrame]
                self.timeframe_encoder.fit(all_timeframe_values)
                self._timeframe_encoder_fitted = True
            
            # 添加 timeframe 编码特征名称
            self.feature_names.append('timeframe_encoded')
            self.feature_specs['timeframe_encoded'] = FeatureSpec(
                name='timeframe_encoded',
                type=FeatureType.CATEGORICAL,
                num_categories=len(TimeFrame),
            )

    def update(self, kline: KLine) -> np.ndarray:
        """
        [实盘模式] 更新并返回当前特征向量
        
        返回的特征向量顺序与 feature_names 完全一致，确保模型能正确识别每个特征
        """
        # 按照 feature_names 的顺序构建特征向量
        feature_dict = {}
        
        # 1. 收集因子特征（使用 factor_id 前缀作为 key）
        for factor in self.factors:
            val = factor.update(kline)
            factor_names = factor.get_output_names()
            factor_id = getattr(factor, "_factor_id")
            if isinstance(val, list):
                for i, v in enumerate(val):
                    full_name = f"{factor_id}_{factor_names[i]}"
                    feature_dict[full_name] = v
            else:
                for factor_name in factor_names:
                    full_name = f"{factor_id}_{factor_name}"
                    feature_dict[full_name] = None
        
        # 2. 添加 symbol 和 timeframe 编码特征
        if self.enable_symbol_timeframe_encoding:
            if not self._symbol_encoder_fitted:
                raise ValueError("Symbol encoder not fitted. Call fit_encoders() first or use pre-fitted encoder.")
            if not self._timeframe_encoder_fitted:
                raise ValueError("Timeframe encoder not fitted.")
            
            # Symbol: OneHotEncoder 返回数组
            symbol_encoded = self.symbol_encoder.transform([[kline.symbol]])[0]
            categories = self.symbol_encoder.categories_[0]
            for i, cat in enumerate(categories):
                feature_dict[f'symbol_{cat}'] = symbol_encoded[i]
            
            # Timeframe: LabelEncoder 返回单个值
            timeframe_encoded = self.timeframe_encoder.transform([kline.timeframe.value])[0]
            feature_dict['timeframe_encoded'] = timeframe_encoded
        
        # 3. 按照 feature_names 的顺序构建特征向量
        features = []
        for name in self.feature_names:
            if name not in feature_dict:
                raise ValueError(
                    f"Feature '{name}' not found in computed features. "
                    f"Available features: {list(feature_dict.keys())}"
                )
            val = feature_dict[name]
            # 处理 None 和 Decimal
            if val is None:
                features.append(np.nan)
            elif isinstance(val, Decimal):
                features.append(float(val))
            else:
                features.append(val)
        
        return np.asarray(features, dtype=np.float64).reshape(1, -1)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [训练模式] 批量计算所有特征
        
        返回的 DataFrame 列顺序与 feature_names 完全一致，确保模型能正确识别每个特征
        """
        # 1. 计算因子特征
        factor_dfs = []
        for factor in self.factors:
            factor_id = getattr(factor, "_factor_id")
            self.call_back(factor_id, False)
            factor_df = factor.compute(df)
            # 将因子原始列名重命名为带 factor_id 前缀的全名
            original_names = factor.get_output_names()
            rename_map = {name: f"{factor_id}_{name}" for name in original_names}
            factor_df = factor_df.rename(columns=rename_map)
            factor_dfs.append(factor_df)
            self.call_back(factor_id, True)
        
        result = pd.concat(factor_dfs, axis=1) if factor_dfs else pd.DataFrame(index=df.index)
        
        # 2. 添加 symbol 和 timeframe 编码特征
        if self.enable_symbol_timeframe_encoding:
            if not self._symbol_encoder_fitted:
                raise ValueError("Symbol encoder not fitted. Call fit_encoders() first.")
            if not self._timeframe_encoder_fitted:
                raise ValueError("Timeframe encoder not fitted.")
            
            # 确保 timeframe 是字符串格式
            timeframe_series = df['timeframe'].astype(str)
            
            # Symbol: OneHotEncoder 返回 DataFrame，每列对应一个类别
            symbol_encoded = self.symbol_encoder.transform(df[['symbol']].values)
            categories = self.symbol_encoder.categories_[0]
            symbol_df = pd.DataFrame(
                symbol_encoded,
                columns=[f'symbol_{cat}' for cat in categories],
                index=df.index
            )
            result = pd.concat([result, symbol_df], axis=1)
            
            # Timeframe: LabelEncoder 返回单个 Series
            result['timeframe_encoded'] = self.timeframe_encoder.transform(timeframe_series)
        
        # 3. 检查是否有重复列名
        if result.columns.duplicated().any():
            duplicated_cols = result.columns[result.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"发现重复的列名: {duplicated_cols}。"
                f"请检查因子配置，确保所有因子的输出列名都是唯一的。"
            )
        
        # 4. 按照 feature_names 的顺序选择列
        missing_cols = [name for name in self.feature_names if name not in result.columns]
        if missing_cols:
            raise ValueError(
                f"Missing features in result DataFrame: {missing_cols}. "
                f"Available columns: {list(result.columns)}"
            )
        
        return result[self.feature_names]

    def fit_encoders(self, symbols: List[str]):
        """
        预拟合编码器（训练模式）
        
        在开始处理数据之前，使用所有可能的 symbol 值拟合编码器
        timeframe 编码器已经在初始化时使用固定的 TimeFrame 枚举值拟合，无需再次拟合
        
        :param symbols: 所有可能的 symbol 列表
        """
        if not self.enable_symbol_timeframe_encoding:
            return
        
        if not self._symbol_encoder_fitted:
            # 对 symbols 进行排序，确保编码顺序一致
            sorted_symbols = sorted(symbols)
            # OneHotEncoder 需要二维输入
            self.symbol_encoder.fit(np.array(sorted_symbols).reshape(-1, 1))
            self._symbol_encoder_fitted = True
            # 按照 categories_ 的顺序添加 symbol 特征名称到 feature_names
            if hasattr(self.symbol_encoder, 'categories_') and len(self.symbol_encoder.categories_) > 0:
                symbol_feature_names = [f'symbol_{cat}' for cat in self.symbol_encoder.categories_[0]]
                # 移除旧的 symbol 特征名称（如果有），然后添加新的
                self.feature_names = [name for name in self.feature_names if not name.startswith('symbol_')]
                # 同时清理 feature_specs 中的旧 symbol 条目
                self.feature_specs = {k: v for k, v in self.feature_specs.items() if not k.startswith('symbol_')}
                # 找到 timeframe_encoded 的位置，在它之前插入 symbol 特征
                if 'timeframe_encoded' in self.feature_names:
                    tf_idx = self.feature_names.index('timeframe_encoded')
                    self.feature_names[tf_idx:tf_idx] = symbol_feature_names
                else:
                    self.feature_names.extend(symbol_feature_names)
                # 添加 symbol 特征的 specs
                for sfn in symbol_feature_names:
                    self.feature_specs[sfn] = FeatureSpec(name=sfn)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
    
    def get_feature_specs(self) -> Dict[str, FeatureSpec]:
        """返回所有特征的 spec 映射（full_name -> FeatureSpec）"""
        return self.feature_specs
    
    def get_categorical_info(self) -> Dict[str, int]:
        """
        返回 categorical 特征的信息映射
        
        :return: {full_feature_name: num_categories} 字典，供 GRU Embedding 层使用
        """
        return {
            name: spec.num_categories
            for name, spec in self.feature_specs.items()
            if spec.type == FeatureType.CATEGORICAL
        }
    
    def get_encoder_classes(self) -> Dict[str, List[str]]:
        """
        获取编码器的类别列表（用于序列化保存）
        
        :return: 包含 symbol_classes 和 timeframe_classes 的字典
        """
        if not self.enable_symbol_timeframe_encoding:
            return {
                "enable_symbol_timeframe_encoding": False
            }
        
        result = {}
        # Symbol: OneHotEncoder 使用 categories_
        if self._symbol_encoder_fitted and hasattr(self.symbol_encoder, 'categories_') and len(self.symbol_encoder.categories_) > 0:
            result['symbol_classes'] = self.symbol_encoder.categories_[0].tolist()
        # Timeframe: LabelEncoder 使用 classes_
        if self._timeframe_encoder_fitted and hasattr(self.timeframe_encoder, 'classes_'):
            result['timeframe_classes'] = self.timeframe_encoder.classes_.tolist()
        
        return result
    
    @classmethod
    def create_from_encoder_classes(
        cls,
        factors: List[DualModeFactor],
        factor_ids: List[str] = None,
        symbol_classes: Optional[List[str]] = None,
        timeframe_classes: Optional[List[str]] = None,
        enable_symbol_timeframe_encoding: bool = True
    ) -> 'FeatureEngine':
        """
        从编码器类别列表创建 FeatureEngine（用于预测模式）
        
        :param factors: 因子列表
        :param factor_ids: 因子ID列表
        :param symbol_classes: symbol 编码器的类别列表（用于 OneHotEncoder）
        :param timeframe_classes: timeframe 编码器的类别列表（用于 LabelEncoder）
        :param enable_symbol_timeframe_encoding: 是否启用 symbol 和 timeframe 编码
        :return: FeatureEngine 实例
        """
        symbol_encoder = None
        timeframe_encoder = None
        
        if enable_symbol_timeframe_encoding:
            if symbol_classes is not None:
                # Symbol 使用 OneHotEncoder
                sorted_symbol_classes = sorted(symbol_classes)
                symbol_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                symbol_encoder.fit(np.array(sorted_symbol_classes).reshape(-1, 1))
            
            if timeframe_classes is not None:
                # Timeframe 使用 LabelEncoder
                timeframe_encoder = LabelEncoder()
                timeframe_encoder.fit(timeframe_classes)
        
        engine = cls(
            factors=factors,
            factor_ids=factor_ids,
            symbol_encoder=symbol_encoder,
            timeframe_encoder=timeframe_encoder,
            enable_symbol_timeframe_encoding=enable_symbol_timeframe_encoding
        )
        return engine
    
    def call_back(self, factor_id: str, success: bool):
        if factor_id is not None and self._call_back is not None:
            self._call_back(factor_id, success)
