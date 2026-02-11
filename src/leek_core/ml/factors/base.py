#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from leek_core.base import LeekComponent
from leek_core.models import KLine
from leek_core.utils import get_logger

logger = get_logger(__name__)


class FeatureType(str, Enum):
    """特征类型枚举"""
    NUMERIC = "numeric"           # 连续数值特征
    CATEGORICAL = "categorical"   # 枚举/离散类别特征


@dataclass
class FeatureSpec:
    """
    特征元数据描述
    
    :param name: 特征名（因子自身的输出列名，不含 factor_id 前缀）
    :param type: 特征类型（NUMERIC 或 CATEGORICAL）
    :param num_categories: categorical 时的类别数（如 hour=24），NUMERIC 时为 0
    """
    name: str
    type: FeatureType = FeatureType.NUMERIC
    num_categories: int = 0


class DualModeFactor(LeekComponent, ABC):

    """
    双模因子基类
    同时支持：
    1. 流式计算 (Update) - 用于实盘/逐行回测
    2. 向量化计算 (Vectorized) - 用于离线批量计算
    """
    # 子类应定义此类变量作为显示名称
    display_name = None
    _name = None
    
    def __init__(self):
        super().__init__()
        # 子类可以覆盖此属性来指定所需的缓冲区大小
        # 默认360，足够大多数因子使用
        buffer_size = getattr(self, '_required_buffer_size', 360)
        # 使用 deque 缓存最近 N 条 K 线数据
        self._buffer = deque(maxlen=buffer_size)
        self._cached_output_specs = None  # 缓存输出 specs
        self._cached_output_names = None  # 缓存输出名称
    
    @property
    def name(self) -> str:
        """
        获取因子名称（兼容性属性）
        返回 display_name，如果未定义则返回类名
        """
        return self._name if self._name is not None else self.__class__.__name__

    def update(self, kline: KLine) -> Union[float, int, List[Union[float, int]], None]:
        """
        流式更新接口（默认实现）
        使用 deque 缓存最近 N 条数据，调用 compute 方法计算因子
        
        返回 float/int (单因子) 或 List[float|int] (多因子)
        categorical 特征返回 int，numeric 特征返回 float
        """
        # 将 KLine 转换为字典并添加到缓冲区
        kline_dict = {
            'start_time': kline.start_time,
            'symbol': kline.symbol,
            'open': float(kline.open) if kline.open is not None else np.nan,
            'high': float(kline.high) if kline.high is not None else np.nan,
            'low': float(kline.low) if kline.low is not None else np.nan,
            'close': float(kline.close) if kline.close is not None else np.nan,
            'volume': float(kline.volume) if kline.volume is not None else np.nan,
            'amount': float(kline.amount) if kline.amount is not None else np.nan,
        }
        if kline.is_finished:
            self._buffer.append(kline_dict)
            lst = list(self._buffer)
        else:
            lst = list(self._buffer) + [kline_dict]
        
        # 如果缓冲区数据不足，返回 None
        if len(self._buffer) < 2:
            return None
        
        # 将缓冲区转换为 DataFrame
        df = pd.DataFrame(list(lst))
        
        # 将 symbol 列转换为 category 类型（如果存在）
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype('category')
        
        # 调用 compute 方法计算因子
        try:
            result_df = self.compute(df.copy())
        except Exception as e:
            logger.error(f"Factor {self.name} compute failed: {e}", exc_info=True)
            return None
        
        # 获取输出 specs（缓存以提高性能）
        if self._cached_output_specs is None:
            self._cached_output_specs = self.get_output_specs()
        
        # 提取最后一行的因子值
        if not self._cached_output_specs:
            return None
        
        factor_values = []
        for spec in self._cached_output_specs:
            if spec.name in result_df.columns:
                value = result_df[spec.name].iloc[-1]
                # 处理 NaN 值
                if pd.isna(value):
                    factor_values.append(np.nan)
                elif spec.type == FeatureType.CATEGORICAL:
                    factor_values.append(int(value))
                else:
                    factor_values.append(float(value))
            else:
                factor_values.append(np.nan)
        
        if len(factor_values) == 1:
            return factor_values[0]
        else:
            return factor_values

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算接口
        在 df 上增加列（列名由 get_output_names() 指定），并返回 df
        :param df: 包含 open, high, low, close, volume 的 DataFrame
        :return: 修改后的 DataFrame（增加了特征列）
        """
        ...

    def get_output_specs(self) -> List[FeatureSpec]:
        """
        获取输出特征的元数据列表
        
        子类应覆写此方法以声明每个输出特征的名称和类型。
        默认返回单个 NUMERIC 特征，名称为 display_name 或类名。
        """
        name = self.display_name if self.display_name is not None else self.__class__.__name__
        return [FeatureSpec(name=name)]

    def get_output_names(self) -> List[str]:
        """
        获取输出列名列表（从 get_output_specs 派生）
        
        子类不应覆写此方法，应覆写 get_output_specs() 代替。
        """
        if self._cached_output_names is None:
            self._cached_output_names = [spec.name for spec in self.get_output_specs()]
        return self._cached_output_names

