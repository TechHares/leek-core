#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal

import numpy as np
import pandas as pd

from leek_core.indicators import ATR, MA, RSI

from .base import DualModeFactor

class MAFactor(DualModeFactor):
    display_name = "MA"
    _name = "MA"
    
    def __init__(self, params: dict):
        super().__init__()
        self.window = int(params.get("window", 20))
        self._indicator = MA(self.window)
        # 动态生成因子名称
        self._factor_name = params.get("name", f"MA_{self.window}")

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
    
    def __init__(self, params: dict):
        super().__init__()
        self.window = int(params.get("window", 14))
        self._indicator = RSI(self.window)
        # 动态生成因子名称
        self._factor_name = params.get("name", f"RSI_{self.window}")

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
    
    def __init__(self, params: dict):
        super().__init__()
        self.window = int(params.get("window", 14))
        self._indicator = ATR(self.window)
        # 动态生成因子名称
        self._factor_name = params.get("name", f"ATR_{self.window}")

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

