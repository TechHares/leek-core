#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块

提供模型训练的抽象接口和具体实现。
"""
from .base import BaseTrainer
from .xgboost import XGBoostTrainer

__all__ = [
    "BaseTrainer",
    "XGBoostTrainer",
]

