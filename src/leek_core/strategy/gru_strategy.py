#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU 策略实现

提供基于 GRU 的完整策略实现，支持分类和回归两种模式。
与 XGBoost 策略的关键区别：
1. 使用 GRUTrainer 加载 PyTorch 模型
2. 维护滑动窗口 buffer 以构建时序输入
3. 使用 torch 推理
"""
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from decimal import Decimal

from leek_core.models import KLine, PositionSide, Field, FieldType
from leek_core.strategy.ml import MLStrategy
from leek_core.utils import get_logger

logger = get_logger(__name__)

# 延迟导入 torch
torch = None


def _lazy_import_torch():
    """延迟导入 PyTorch"""
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GRUStrategy. "
                "Please install it via: pip install torch"
            )


class GRUStrategy(MLStrategy):
    """
    GRU 策略
    
    基于 GRU 模型的交易策略，支持分类和回归两种模式。
    
    与 XGBoostStrategy 的核心区别：
    - GRU 是时序模型，需要积累 window_size 根 K 线才能预测
    - 内部维护一个特征滑动窗口 buffer
    - 使用 PyTorch 进行推理
    """

    display_name = "GRU策略"

    init_params = MLStrategy.init_params + [
        Field(
            name="mode",
            label="模型模式",
            type=FieldType.RADIO,
            description="模型模式：'classification'（分类）或 'regression'（回归），默认自动检测",
            required=False,
            choices=[("classification", "分类"), ("regression", "回归")],
            choice_type=FieldType.STRING,
        ),
        Field(
            name="return_threshold",
            label="收益率阈值",
            type=FieldType.FLOAT,
            default=0.02,
            description="回归模式的收益率阈值（如0.02表示2%）",
            required=False,
        ),
        Field(
            name="device",
            label="计算设备",
            type=FieldType.RADIO,
            description="模型推理设备",
            required=False,
            choices=[("cpu", "CPU"), ("cuda", "CUDA"), ("auto", "自动")],
            choice_type=FieldType.STRING,
        ),
    ]

    def __init__(
        self,
        model_config: Dict[str, Any],
        confidence_threshold: float = 0.6,
        warmup_periods: int = 0,
        mode: Optional[str] = None,
        return_threshold: float = 0.02,
        device: str = "cpu",
    ):
        """
        初始化 GRU 策略
        
        :param model_config: 模型配置字典
        :param confidence_threshold: 置信度阈值（默认0.6）
        :param warmup_periods: 预热期（默认0，会自动设置为 >= window_size）
        :param mode: 模型模式，"classification"或"regression"（默认自动检测）
        :param return_threshold: 回归模式收益率阈值（默认0.02，即2%）
        :param device: 计算设备（默认 "cpu"）
        """
        self.mode = mode
        self.return_threshold = return_threshold
        self.device_config = device
        
        # GRU 特有的运行时状态
        self._feature_buffer: Optional[deque] = None  # 滑动窗口 buffer
        self._window_size: int = 0
        self._torch_device = None
        self._gru_model = None  # GRUModel 实例
        
        super().__init__(
            model_config=model_config,
            confidence_threshold=confidence_threshold,
            warmup_periods=warmup_periods,
        )

    def _load_model(self):
        """
        重写模型加载：使用 GRUTrainer 加载 PyTorch 模型
        """
        _lazy_import_torch()
        from pathlib import Path
        from leek_core.ml.trainer.gru import GRUTrainer

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 使用 GRUTrainer 加载模型
        trainer = GRUTrainer(device=self.device_config)
        trainer.load_model(path=self.model_path)

        # 提取模型和配置
        self._gru_model = trainer._model
        self._window_size = trainer.window_size
        self._torch_device = trainer._get_device()

        # 自动检测模式
        if self.mode is None:
            self.mode = trainer.task_type
            logger.info(f"Auto-detected mode: {self.mode}")

        # 将模型移到设备并设为评估模式
        self._gru_model.to(self._torch_device)
        self._gru_model.eval_mode()

        # 初始化滑动窗口 buffer
        self._feature_buffer = deque(maxlen=self._window_size)

        # 确保预热期 >= window_size
        self.warmup_periods = max(self.warmup_periods, self._window_size)

        # 设置 self.model 以通过父类校验
        self.model = self._gru_model

        logger.info(
            f"GRU model loaded: window_size={self._window_size}, "
            f"mode={self.mode}, device={self._torch_device}"
        )

    def _predict(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        重写预测方法：维护滑动窗口，窗口满后才预测
        
        :param features: 当前 K 线的特征向量 (n_features,)
        :return: 信号字典或 None
        """
        _lazy_import_torch()

        # 1. 将当前特征加入 buffer
        self._feature_buffer.append(features.copy())

        # 2. 窗口不够，不预测
        if len(self._feature_buffer) < self._window_size:
            return None

        # 3. 构建时序窗口 (1, window_size, n_features)
        window = np.array(list(self._feature_buffer), dtype=np.float32)
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self._torch_device)

        # 4. 模型推理
        with torch.no_grad():
            outputs = self._gru_model.network(window_tensor)

            if self.mode == "classification":
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                return self._classification_to_signal(proba)
            elif self.mode == "regression":
                pred_value = outputs.cpu().numpy().flatten()[0]
                return self._regression_to_signal(float(pred_value))
            else:
                logger.error(f"Unknown mode: {self.mode}")
                return None

    def _classification_to_signal(self, prediction: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        分类模式：将概率转换为信号
        
        二分类：[prob_down, prob_up]
        三分类：[prob_down, prob_neutral, prob_up]
        """
        if not isinstance(prediction, np.ndarray):
            return None

        if len(prediction) == 2:
            # 二分类
            prob_down, prob_up = float(prediction[0]), float(prediction[1])

            if prob_up > self.confidence_threshold:
                return {
                    'side': PositionSide.LONG,
                    'confidence': prob_up,
                }
            elif prob_down > self.confidence_threshold:
                return {
                    'side': PositionSide.SHORT,
                    'confidence': prob_down,
                }

        elif len(prediction) == 3:
            # 三分类：[down, neutral, up]
            prob_down, prob_neutral, prob_up = (
                float(prediction[0]),
                float(prediction[1]),
                float(prediction[2]),
            )

            max_prob = max(prob_down, prob_neutral, prob_up)

            if max_prob > self.confidence_threshold:
                if prob_up == max_prob:
                    return {
                        'side': PositionSide.LONG,
                        'confidence': prob_up,
                    }
                elif prob_down == max_prob:
                    return {
                        'side': PositionSide.SHORT,
                        'confidence': prob_down,
                    }
                # prob_neutral 最高时不交易

        return None

    def _regression_to_signal(self, predicted_return: float) -> Optional[Dict[str, Any]]:
        """
        回归模式：将预测收益率转换为信号
        
        :param predicted_return: 预测的未来收益率
        """
        if predicted_return > self.return_threshold:
            confidence = min(abs(predicted_return) / (self.return_threshold * 2), 1.0)
            return {
                'side': PositionSide.LONG,
                'confidence': confidence,
            }
        elif predicted_return < -self.return_threshold:
            confidence = min(abs(predicted_return) / (self.return_threshold * 2), 1.0)
            return {
                'side': PositionSide.SHORT,
                'confidence': confidence,
            }

        return None
