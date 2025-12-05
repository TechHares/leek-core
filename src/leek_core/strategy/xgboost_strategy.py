#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost 策略实现

提供基于 XGBoost 的完整策略实现，支持分类和回归两种模式。
"""
from typing import Any, Dict, Optional
import numpy as np
from decimal import Decimal

from leek_core.models import KLine, PositionSide, Field, FieldType
from leek_core.strategy.ml import MLStrategy
from leek_core.utils import get_logger

logger = get_logger(__name__)


class XGBoostStrategy(MLStrategy):
    """
    XGBoost 策略基类
    
    支持两种模式：
    1. 分类模式（Classification）：预测涨跌方向
    2. 回归模式（Regression）：预测未来收益率
    """

    display_name = "XGBoost策略"

    # 继承父类参数并添加XGBoost特定参数
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
        )
    ]

    def __init__(self, model_config: Dict[str, Any],
        confidence_threshold: float = 0.6, warmup_periods: int = 0, mode: Optional[str] = None, return_threshold: float = 0.02):
        """
        初始化XGBoost策略
        
        :param model_config: 模型配置字典，必须包含：
            - model_path: 模型文件路径（必填）
            - feature_config: 特征配置列表（必填）
            - model_id: 模型ID（可选）
        :param confidence_threshold: 置信度阈值（默认0.6）
        :param warmup_periods: 预热期（默认0）
        :param mode: 模型模式，"classification"或"regression"（默认自动检测）
        :param return_threshold: 收益率阈值（默认0.02，即2%）
        """
        super().__init__(
            model_config=model_config,
            confidence_threshold=confidence_threshold,
            warmup_periods=warmup_periods,
        )
        self.mode = mode  # "classification" 或 "regression"
        self.return_threshold = return_threshold

    def _predict(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        重写预测方法，根据模式调用不同的预测接口
        
        分类模式：使用 predict_proba() 获取概率
        回归模式：使用 predict() 获取预测值
        """
        if self.mode == "classification":
            # 分类模式：使用 predict_proba 获取概率
            proba = self.model.predict_proba(features)
            # predict_proba 返回 shape (1, n_classes)，取第一行
            if len(proba.shape) > 1 and proba.shape[0] > 0:
                prediction = proba[0]
            else:
                prediction = proba
            
            return self._classification_to_signal(prediction)
            
        elif self.mode == "regression":
            # 回归模式：使用 predict 获取预测值
            pred = self.model.predict(features)
            # predict 返回 shape (1,)，取第一个值
            if isinstance(pred, np.ndarray) and len(pred.shape) > 0:
                prediction = pred[0] if pred.shape[0] > 0 else pred
            else:
                prediction = pred
            
            return self._regression_to_signal(prediction)
        else:
            logger.error(f"Unknown mode: {self.mode}")
            return None

    def _classification_to_signal(self, prediction: Any) -> Optional[Dict[str, Any]]:
        """
        分类模式：将概率转换为信号
        
        假设二分类：
        - 类别 0: 下跌 -> SHORT
        - 类别 1: 上涨 -> LONG
        
        如果是多分类，需要根据具体业务逻辑调整
        """
        # 处理概率数组
        if isinstance(prediction, np.ndarray):
            if len(prediction) == 2:
                # 二分类：[prob_down, prob_up]
                prob_down, prob_up = float(prediction[0]), float(prediction[1])

                # 上涨概率高，做多
                if prob_up > self.confidence_threshold:
                    return {
                        'side': PositionSide.LONG,
                        'confidence': prob_up,
                    }
                # 下跌概率高，做空
                elif prob_down > self.confidence_threshold:
                    return {
                        'side': PositionSide.SHORT,
                        'confidence': prob_down,
                    }
            elif len(prediction) == 3:
                # 三分类：[prob_down, prob_neutral, prob_up]
                prob_down, prob_neutral, prob_up = (
                    float(prediction[0]),
                    float(prediction[1]),
                    float(prediction[2])
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
        else:
            # 如果返回的是类别索引（predict 而不是 predict_proba）
            logger.warning("Model returned class index instead of probabilities. "
                          "Consider using predict_proba for better control.")
            return None

        return None

    def _regression_to_signal(self, prediction: Any) -> Optional[Dict[str, Any]]:
        """
        回归模式：将预测收益率转换为信号
        
        :param prediction: 预测的未来收益率（float）
        """
        # 转换为 float
        if isinstance(prediction, np.ndarray):
            predicted_return = float(prediction[0]) if len(prediction) > 0 else 0.0
        elif isinstance(prediction, (float, np.floating)):
            predicted_return = float(prediction)
        else:
            logger.warning(f"Unexpected prediction type: {type(prediction)}")
            return None

        # 如果预测收益率超过阈值，做多
        if predicted_return > self.return_threshold:
            # 置信度可以根据预测值的绝对值来设定
            # 例如：预测 5% 收益率，置信度可以是 min(5% / 10%, 1.0) = 0.5
            confidence = min(abs(predicted_return) / (self.return_threshold * 2), 1.0)

            return {
                'side': PositionSide.LONG,
                'confidence': confidence,
            }
        # 如果预测负收益率超过阈值，做空
        elif predicted_return < -self.return_threshold:
            confidence = min(abs(predicted_return) / (self.return_threshold * 2), 1.0)

            return {
                'side': PositionSide.SHORT,
                'confidence': confidence,
            }

        return None
