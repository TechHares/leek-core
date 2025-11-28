#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习策略基类

提供基于机器学习的策略框架，支持模型加载、特征计算、预测和信号生成。
"""
from abc import abstractmethod
import joblib
import numpy as np
import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from decimal import Decimal

from leek_core.models import KLine, PositionSide, Position, Field, FieldType
from leek_core.strategy.cta import CTAStrategy, StrategyCommand
from leek_core.ml.feature_engine import FeatureEngine
from leek_core.utils import get_logger

logger = get_logger(__name__)

# 全局模型加载器注册表（用于从平台存储加载模型）
_model_loaders: Dict[str, Callable[[str], Any]] = {}


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
            name="model_path",
            label="模型文件路径",
            type=FieldType.STRING,
            description="模型文件路径（.joblib格式），与model_data/model_id三选一",
            required=False,
        ),
        Field(
            name="model_data",
            label="模型数据(base64)",
            type=FieldType.STRING,
            description="base64编码的模型数据（适合gRPC传递），与model_path/model_id三选一",
            required=False,
        ),
        Field(
            name="model_id",
            label="模型ID",
            type=FieldType.STRING,
            description="平台模型ID（需注册加载器），与model_path/model_data三选一",
            required=False,
        ),
        Field(
            name="feature_config",
            label="特征配置",
            type=FieldType.ARRAY,
            description="特征配置列表（JSON格式），必须与训练时一致",
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
    
    def __init__(self, model_path: Optional[str] = None, model_data: Optional[str] = None,
     feature_config: Optional[List[Dict]] = None, confidence_threshold: float = 0.5, warmup_periods: int = 0):
        """
        初始化ML策略
        
        :param model_path: 模型文件路径（与model_data二选一）
        :param model_data: base64编码的模型数据（与model_path二选一）
        :param feature_config: 特征配置列表（必填）
        :param confidence_threshold: 置信度阈值（默认0.5）
        :param warmup_periods: 预热期（默认0）

        模型文件必须是使用 joblib.dump() 保存的 .joblib 或 .pkl 格式
        """
        super().__init__()
        self.model: Optional[Any] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.model_path = model_path
        self.model_data = model_data
        self.feature_config = feature_config
        self.confidence_threshold = confidence_threshold
        self.warmup_periods = warmup_periods
        
        # 运行时状态
        self._kline_count: int = 0
        self._current_signal: Optional[Dict[str, Any]] = None
        
        # 验证并初始化
        self._validate_and_init()
    
    def _validate_and_init(self):
        """验证参数并初始化模型和特征引擎"""
        # 验证模型源（至少提供一个）
        model_sources = sum([
            bool(self.model_path),
            bool(self.model_data),
        ])
        if model_sources == 0:
            raise ValueError("At least one of model_path or model_data is required")
        if model_sources > 1:
            logger.warning("Multiple model sources specified. Priority: model_data > model_path")
        
        # 验证特征配置
        if not self.feature_config:
            raise ValueError("feature_config is required")
        
        # 初始化模型和特征引擎
        self._load_model()
        self.feature_engine = FeatureEngine(self.feature_config)
        
        model_source = "base64_data" if self.model_data else f"file:{self.model_path}"
        logger.info(f"MLStrategy initialized: model_source={model_source}, features={len(self.feature_config)}")
    
    def _load_model(self):
        """
        加载模型
        
        支持两种方式（按优先级）：
        1. model_data: base64编码的模型数据（直接传递，适合gRPC）
        2. model_path: 文件路径（传统方式）
        
        默认支持 joblib 格式，子类可以重写以支持其他格式（如 onnx）
        """
        try:
            # 方式1: 从 base64 数据加载（最优先，适合gRPC传递）
            if self.model_data:
                model_bytes = base64.b64decode(self.model_data)
                self.model = joblib.load(io.BytesIO(model_bytes))
                logger.info("Model loaded successfully from base64 data")
            # 方式2: 从文件路径加载（传统方式）
            elif self.model_path:
                if not Path(self.model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from file: {self.model_path}")
            else:
                raise ValueError("No model source specified. Provide model_path or model_data")
            
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
            logger.debug("Features contain NaN, skipping prediction")
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
    