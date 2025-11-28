#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from leek_core.base import LeekComponent

class LabelGenerator(LeekComponent, ABC):
    """
    Label 生成器基类
    负责从原始数据生成训练标签
    """
    
    def __init__(self, params: dict):
        self.params = params
        self.label_name = params.get("label_name", "label")
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成标签列并添加到 DataFrame
        
        :param df: 包含 open, high, low, close, volume 等原始数据的 DataFrame
        :return: 增加了 label 列的 DataFrame
        """
        pass
    
    def get_label_name(self) -> str:
        """获取标签列名"""
        return self.label_name

