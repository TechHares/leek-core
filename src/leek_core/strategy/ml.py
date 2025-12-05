#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习策略基类

提供基于机器学习的策略框架，支持模型加载、特征计算、预测和信号生成。
"""
from abc import abstractmethod
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from leek_core.models import KLine, PositionSide, Position, Field, FieldType
from leek_core.strategy.cta import CTAStrategy, StrategyCommand
from leek_core.ml.feature_engine import FeatureEngine
from leek_core.base import create_component, load_class_from_str
from leek_core.utils import get_logger

logger = get_logger(__name__)


class MLStrategy(CTAStrategy):
    """
    机器学习策略基类
    
    功能：
    1. 加载训练好的模型（支持 joblib、onnx 等格式）
    2. 实时计算特征
    3. 模型预测
    4. 将预测结果转换为交易信号
    
    子类需要实现：
    - _load_model(): 加载模型（如果使用非标准格式）
    - _predict_to_signal(): 将模型预测结果转换为交易信号
    """
    
    display_name = "机器学习策略"
    
    init_params = [
        Field(
            name="model_config",
            label="模型",
            type=FieldType.MODEL,
            description="选择已训练好的模型",
            required=True,
        ),
        Field(
            name="confidence_threshold",
            label="置信度阈值",
            type=FieldType.FLOAT,
            default=0.5,
            description="置信度阈值，用于过滤低置信度信号。分类模式通常用概率值（0-1），回归模式可能用预测值绝对值。子类可自定义实现",
            required=False,
        ),
        Field(
            name="warmup_periods",
            label="预热期",
            type=FieldType.INT,
            default=0,
            description="预热期K线数量，等待指标稳定",
            required=False,
        ),
    ]
    
    def __init__(self, model_config: Dict[str, Any],
                 confidence_threshold: float = 0.5, warmup_periods: int = 0):
        """
        初始化ML策略
        
        :param model_config: 模型配置字典，必须包含：
            - model_path: 模型文件路径（必填）
            - feature_config: 特征配置列表（必填），格式：[{id, name, class_name, params}, ...]
            - model_id: 模型ID（可选，用于展示和关联）
        :param confidence_threshold: 置信度阈值（默认0.5）
        :param warmup_periods: 预热期（默认0）

        模型文件必须是使用 joblib.dump() 保存的 .joblib 或 .pkl 格式
        """
        super().__init__()
        self.model: Optional[Any] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.model_config = model_config
        self.model_path = model_config.get('model_path')
        feature_config_raw = model_config.get('feature_config', {})
        
        # 处理新格式：feature_config 是字典，包含 'factors' 和 'encoder_classes'
        self.feature_config = feature_config_raw.get('factors', [])
        self.encoder_classes = feature_config_raw.get('encoder_classes', {})
        
        self.confidence_threshold = confidence_threshold
        self.warmup_periods = warmup_periods
        
        # 运行时状态
        self._kline_count: int = 0
        self._current_signal: Optional[Dict[str, Any]] = None
        
        # 验证并初始化
        self._validate_and_init()
    
    def _validate_and_init(self):
        """验证参数并初始化模型和特征引擎"""
        # 验证模型配置
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config must be a dictionary")
        
        # 验证模型路径
        if not self.model_path:
            raise ValueError("model_config must contain 'model_path'")
        
        # 验证特征配置
        if not self.feature_config or not isinstance(self.feature_config, list):
            raise ValueError("model_config must contain 'feature_config' with 'factors' as a list")
        
        # 初始化模型
        self._load_model()
        
        # 从特征配置创建因子实例
        factor_instances = []
        factor_ids = []
        for factor_config in self.feature_config:
            if not isinstance(factor_config, dict):
                raise ValueError(f"Invalid feature_config item: {factor_config}, must be a dictionary")
            
            class_name = factor_config.get('class_name')
            if not class_name:
                raise ValueError(f"feature_config item must contain 'class_name': {factor_config}")
            
            try:
                factor_cls = load_class_from_str(class_name)
                factor_instance = create_component(factor_cls, **(factor_config.get('params', {})))
                factor_instances.append(factor_instance)
                factor_ids.append(str(factor_config.get('id', len(factor_instances) - 1)))
            except Exception as e:
                raise RuntimeError(f"Failed to create factor from config {factor_config}: {e}") from e
        
        self.feature_engine = FeatureEngine.create_from_encoder_classes(
            factors=factor_instances,
            factor_ids=factor_ids,
            symbol_classes=self.encoder_classes.get('symbol_classes', None) ,
            timeframe_classes=self.encoder_classes.get('timeframe_classes', None),
            enable_symbol_timeframe_encoding=self.encoder_classes.get('enable_symbol_timeframe_encoding', True)
        )
        logger.info(f"MLStrategy initialized: model_path={self.model_path}, features={len(self.feature_config)}")
    
    def _load_model(self):
        """
        加载模型
        
        从文件路径加载模型，默认支持 joblib 格式。
        子类可以重写以支持其他格式（如 onnx）
        """
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from file: {self.model_path}")
            
        except ModuleNotFoundError as e:
            # 捕获模块缺失错误，提供更友好的错误提示
            model_type = getattr(type(self.model).__module__, 'unknown', 'unknown') if self.model else 'unknown'
            raise RuntimeError(
                f"Failed to load model: Missing required library. "
                f"Model was saved using '{model_type}', but the corresponding library is not installed. "
                f"Please install: sklearn models → scikit-learn, XGBoost → xgboost, LightGBM → lightgbm. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def on_kline(self, kline: KLine):
        """
        处理K线数据
        
        :param kline: K线数据
        """
        self._kline_count += 1
        assert self.feature_engine is not None
        assert self.model is not None

        features = self.feature_engine.update(kline)
        
        if self._kline_count <= self.warmup_periods:
            logger.debug(f"Warmup period: {self._kline_count}/{self.warmup_periods}, skipping prediction")
            return
        
        # 检查特征是否有效（是否有 NaN）
        if np.isnan(features).any():
            logger.info("Features contain NaN, skipping prediction")
            return
        
        # 存储当前信号（供 should_open/should_close 使用）
        self._current_signal = self._predict(features)
    
    @abstractmethod
    def _predict(self, features: np.ndarray) -> Any:
        """
        模型预测
        
        模型应该在加载时已经验证过接口，这里直接使用。
        如果模型不符合要求，会在 _validate_model() 阶段就报错。
        
        :param features: 特征向量 (1, n_features)
        :return: 模型预测结果
            - 信号字典，包含 side, confidence 等信息，或 None（无信号）
        """
        raise NotImplementedError("Subclass must implement _predict")
    
    def should_open(self) -> Union[PositionSide, StrategyCommand, None]:
        """
        判断是否应该开仓
        
        基于模型预测结果决定是否开仓
        """
        if self._current_signal is None:
            return None
        
        signal = self._current_signal
        
        # 检查置信度
        confidence = signal.get('confidence', 0.0)
        if confidence < self.confidence_threshold:
            return None
        
        # 获取方向
        side = signal.get('side')
        if side is None:
            return None
        
        # 获取仓位比例（可选）
        ratio = signal.get('ratio', Decimal('1.0'))
        
        if ratio >= Decimal('1.0'):
            return side
        else:
            return StrategyCommand(side=side, ratio=ratio)
    
    def should_close(self, position_side: PositionSide) -> Union[bool, Decimal]:
        """
        判断是否应该平仓
        
        默认逻辑：如果当前信号方向与持仓方向相反，则平仓
        子类可以重写以实现更复杂的平仓逻辑
        """
        if not hasattr(self, '_current_signal') or self._current_signal is None:
            return False
        
        signal = self._current_signal
        signal_side = signal.get('side')
        
        if signal_side is None:
            return False
        
        # 如果信号方向与持仓方向相反，平仓
        if signal_side != position_side:
            return True
        
        return False
    