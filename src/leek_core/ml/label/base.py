#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

import pandas as pd

from leek_core.base import LeekComponent

if TYPE_CHECKING:
    from leek_core.models import Field

class LabelGenerator(LeekComponent, ABC):
    """
    Label 生成器基类
    负责从原始数据生成训练标签
    """
    display_name: str = None
    init_params: List["Field"] = []
    
    def __init__(self):
        self.label_name = "label"
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成标签列
        
        :param df: 包含 open, high, low, close, volume 等原始数据的 DataFrame
        :return: 标签 Series，名称为 self.label_name
        """
        pass
    
    def get_label_name(self) -> str:
        """获取标签列名"""
        return self.label_name

