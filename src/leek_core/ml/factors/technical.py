#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal

import numpy as np
import pandas as pd

from leek_core.indicators import ATR, MA, RSI
from leek_core.models import Field, FieldType

from .base import DualModeFactor

class MAFactor(DualModeFactor):
    display_name = "MA"
    _name = "MA"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=20,
            description="移动平均线的窗口大小"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 20))
        self._indicator = MA(self.window)
        # 动态生成因子名称
        self._factor_name = f"MA_{self.window}"
        super().__init__()

    def update(self, kline) -> float:
        val = self._indicator.update(kline)
        return float(val) if val is not None else np.nan

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # 在 df 上增加列并返回
        df[self._factor_name] = df['close'].rolling(window=self.window).mean()
        return df
    
    def get_output_names(self) -> list:
        return [self._factor_name]

class RSIFactor(DualModeFactor):
    display_name = "RSI"
    _name = "RSI"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=14,
            description="RSI计算的窗口大小"
        ),
        Field(
            name="name",
            label="因子名称",
            type=FieldType.STRING,
            default="",
            description="自定义因子名称，为空时自动生成"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 14))
        self._indicator = RSI(self.window)
        # 动态生成因子名称
        name = kwargs.get("name", "")
        self._factor_name = name if name else f"RSI_{self.window}"
        super().__init__()

    def update(self, kline) -> float:
        val = self._indicator.update(kline)
        return float(val) if val is not None else np.nan

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pandas 实现 RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        rs = gain / loss
        df[self._factor_name] = 100 - (100 / (1 + rs))
        return df
    
    def get_output_names(self) -> list:
        return [self._factor_name]

class ATRFactor(DualModeFactor):
    display_name = "ATR"
    _name = "ATR"
    
    init_params = [
        Field(
            name="window",
            label="窗口大小",
            type=FieldType.INT,
            default=14,
            description="ATR计算的窗口大小"
        ),
        Field(
            name="name",
            label="因子名称",
            type=FieldType.STRING,
            default="",
            description="自定义因子名称，为空时自动生成"
        )
    ]
    
    def __init__(self, **kwargs):
        self.window = int(kwargs.get("window", 14))
        self._indicator = ATR(self.window)
        # 动态生成因子名称
        name = kwargs.get("name", "")
        self._factor_name = name if name else f"ATR_{self.window}"
        super().__init__()

    def update(self, kline) -> float:
        val = self._indicator.update(kline)
        return float(val) if val is not None else np.nan

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[self._factor_name] = tr.rolling(window=self.window).mean()
        return df
    
    def get_output_names(self) -> list:
        return [self._factor_name]

