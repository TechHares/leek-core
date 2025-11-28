from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor

class Alpha360Factor(DualModeFactor):

    """
    根据Qlib中发表的 Alpha360 因子
    6个基础因子 × 60个周期 = 360个因子
    
    6个基础因子：
    1. CLOSE (收盘价) - index 0-59
    2. OPEN (开盘价) - index 60-119
    3. HIGH (最高价) - index 120-179
    4. LOW (最低价) - index 180-239
    5. VWAP (成交量加权平均价) - index 240-299
    6. VOLUME (成交量) - index 300-359
    
    每个因子包含过去60天的归一化值：
    - 价格类因子（CLOSE/OPEN/HIGH/LOW/VWAP）：除以当前收盘价
    - 成交量因子（VOLUME）：除以当前成交量
    """

    display_name = "Alpha360"
    _delta = 1e-20
    
    # 默认缓冲区大小，会在 __init__ 中根据实际 windows 动态调整
    _required_buffer_size = 70

    init_params = [
        Field(
            name="fields", 
            label="包含字段", 
            type=FieldType.STRING, 
            default="CLOSE,OPEN,HIGH,LOW,VWAP,VOLUME", 
            description="包含的基础因子字段，逗号分隔"
        ),
        Field(
            name="windows", 
            label="包含窗口", 
            type=FieldType.STRING, 
            default="0-59", 
            description="包含的时间窗口，支持范围(如0-59)和单个数字(如5,10,15)，逗号分隔。可自由设置任意范围，如0-99表示100个周期"
        )
    ]
    
    def __init__(self, **kwargs):
        self.fields = kwargs.get("fields", "CLOSE,OPEN,HIGH,LOW,VWAP,VOLUME")
        self.windows = kwargs.get("windows", "0-59")
        
        # Parse fields
        self.selected_fields = [f.strip().upper() for f in self.fields.split(",") if f.strip()]
        
        # Parse windows
        self.selected_windows = []
        try:
            for part in self.windows.split(","):
                part = part.strip()
                if not part: continue
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    self.selected_windows.extend(range(start, end + 1))
                else:
                    self.selected_windows.append(int(part))
            # 去重并排序
            self.selected_windows = sorted(set(self.selected_windows))
        except Exception:
            # Fallback to default if parsing fails
            self.selected_windows = list(range(60))
        
        # 根据最大窗口值计算所需的缓冲区大小（加一些余量）
        max_window = max(self.selected_windows) if self.selected_windows else 59
        self._required_buffer_size = max_window + 10  # 加10作为余量
        
        # 调用父类初始化（此时 _required_buffer_size 已设置）
        super().__init__()

        # 预构建因子名称映射
        self.factor_names = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称映射"""
        # 6个基础因子名称
        base_names = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]
        
        # 根据实际选择的 windows 构建因子名称
        for base_name in base_names:
            if base_name in self.selected_fields:
                for i in self.selected_windows:
                    self.factor_names.append(f"{self.name}_{base_name}{i}")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有360个因子"""
        # 基础因子函数映射
        base_factor_map = {
            "CLOSE": self._base_factor_close,
            "OPEN": self._base_factor_open,
            "HIGH": self._base_factor_high,
            "LOW": self._base_factor_low,
            "VWAP": self._base_factor_vwap,
            "VOLUME": self._base_factor_volume,
        }
        
        # 先计算所有因子，收集到字典中
        factor_results = {}
        for factor_name in self.factor_names:
            # 解析因子名称：格式为 "Alpha360_CLOSE59" -> ("CLOSE", 59)
            # 去掉前缀 "Alpha360_"
            name_part = factor_name.replace(f"{self.name}_", "")
            
            # 提取基础因子类型和周期
            # 例如 "CLOSE59" -> ("CLOSE", 59), "VOLUME0" -> ("VOLUME", 0)
            for base_name in base_factor_map.keys():
                if name_part.startswith(base_name):
                    period_str = name_part[len(base_name):]
                    period = int(period_str)
                    factor_results[factor_name] = base_factor_map[base_name](df, period)
                    break
        
        return pd.DataFrame(factor_results, index=df.index)

    def _base_factor_close(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子1: CLOSE - Ref($close, period) / $close"""
        if period == 0:
            # 当前值：$close / $close = 1
            return pd.Series(1.0, index=df.index)
        else:
            # 历史值：Ref($close, period) / $close
            return df['close'].shift(period) / (df['close'] + self._delta)
    
    def _base_factor_open(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子2: OPEN - Ref($open, period) / $close"""
        if period == 0:
            # 当前值：$open / $close
            return df['open'] / (df['close'] + self._delta)
        else:
            # 历史值：Ref($open, period) / $close
            return df['open'].shift(period) / (df['close'] + self._delta)
    
    def _base_factor_high(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子3: HIGH - Ref($high, period) / $close"""
        if period == 0:
            # 当前值：$high / $close
            return df['high'] / (df['close'] + self._delta)
        else:
            # 历史值：Ref($high, period) / $close
            return df['high'].shift(period) / (df['close'] + self._delta)
    
    def _base_factor_low(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子4: LOW - Ref($low, period) / $close"""
        if period == 0:
            # 当前值：$low / $close
            return df['low'] / (df['close'] + self._delta)
        else:
            # 历史值：Ref($low, period) / $close
            return df['low'].shift(period) / (df['close'] + self._delta)
    
    def _base_factor_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子5: VWAP - Ref($vwap, period) / $close
        
        根据QLIB定义，VWAP = amount / volume（成交金额/成交量）
        如果没有amount字段，则使用 close * volume 作为成交金额的近似
        """
        # 计算 VWAP：优先使用 amount/volume，如果没有 amount 则用 close*volume 近似
        if 'amount' in df.columns:
            # 使用成交金额/成交量计算 VWAP（QLIB标准定义）
            vwap = df['amount'] / (df['volume'] + self._delta)
        elif 'vwap' in df.columns:
            # 如果已有 vwap 字段，直接使用
            vwap = df['vwap']
        else:
            # 如果没有 amount 和 vwap，使用 close 作为近似（不推荐，但保持兼容性）
            vwap = df['close']
        
        # 归一化：除以当前收盘价（与其他价格因子保持一致）
        if period == 0:
            # 当前值：$vwap / $close
            return vwap / (df['close'] + self._delta)
        else:
            # 历史值：Ref($vwap, period) / $close
            return vwap.shift(period) / (df['close'] + self._delta)
    
    def _base_factor_volume(self, df: pd.DataFrame, period: int) -> pd.Series:
        """基础因子6: VOLUME - Ref($volume, period) / ($volume + 1e-12)"""
        if period == 0:
            # 当前值：$volume / ($volume + 1e-12) = 1
            return df['volume'] / (df['volume'] + self._delta)
        else:
            # 历史值：Ref($volume, period) / ($volume + 1e-12)
            return df['volume'].shift(period) / (df['volume'] + self._delta)

    def get_output_names(self) -> List[str]:
        """返回所有因子的输出名称"""
        return self.factor_names
