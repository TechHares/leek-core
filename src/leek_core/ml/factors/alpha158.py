from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor, FeatureSpec

class Alpha158Factor(DualModeFactor):

    """
    根据Qlib中发表的 Alpha158 因子
    
    标准Alpha158包含158个固定因子：
    1. kbar: K线特征（9个）- KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
    2. price: 原始价格特征（4个）- OPEN0, HIGH0, LOW0, VWAP0
    3. rolling: 滚动窗口特征（145个）- 29种算子 × 5个窗口[5,10,20,30,60]
    
    共158个因子，固定结构，与QLIB标准实现一致。
    """
    display_name = "Alpha158"
    # Alpha158最大窗口为60，设置缓冲区大小为60+一些余量
    _required_buffer_size = 70

    _delta = 1e-20
    
    init_params = [
        Field(name="include_kbar", label="包含KBar因子", type=FieldType.BOOLEAN, default=True, description="是否包含K线形态因子(KMID, KLEN等)"),
        Field(name="include_price", label="包含价格因子", type=FieldType.BOOLEAN, default=True, description="是否包含原始价格因子(OPEN0, HIGH0等)"),
        Field(name="include_rolling", label="包含滚动因子", type=FieldType.BOOLEAN, default=True, description="是否包含滚动窗口因子(MA, STD等)"),
        Field(name="windows", label="滚动窗口", type=FieldType.STRING, default="5,10,20,30,60", description="滚动计算的时间窗口，逗号分隔")
    ]
    
    def __init__(self, **kwargs):
        super().__init__()
        self.include_kbar = kwargs.get("include_kbar", True)
        self.include_price = kwargs.get("include_price", True)
        self.include_rolling = kwargs.get("include_rolling", True)
        self.windows_str = kwargs.get("windows", "5,10,20,30,60")
        
        # Parse windows
        self.rolling_windows = []
        try:
            for part in self.windows_str.split(","):
                part = part.strip()
                if part:
                    self.rolling_windows.append(int(part))
        except Exception:
            self.rolling_windows = [5, 10, 20, 30, 60]
            
        # 预构建因子名称列表
        self.factor_names = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建固定的158个因子名称"""
        # 1. kbar 因子（9个）
        if self.include_kbar:
            kbar_names = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
            self.factor_names.extend([f"{self.name}_{name}" for name in kbar_names])
        
        # 2. price 因子（4个）- OPEN0, HIGH0, LOW0, VWAP0
        if self.include_price:
            price_names = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
            self.factor_names.extend([f"{self.name}_{name}" for name in price_names])
        
        # 3. rolling 因子（145个）- 29种算子 × 5个窗口
        if self.include_rolling:
            rolling_operators = [
                "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
                "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD",
                "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"
            ]
            for op_name in rolling_operators:
                for window in self.rolling_windows:
                    self.factor_names.append(f"{self.name}_{op_name}{window}")

    def _preprocess_dataframe(self, df: pd.DataFrame) -> tuple:
        """数据预处理：计算VWAP，处理NaN值
        
        返回: (vwap Series, 是否临时添加了vwap列)
        
        性能优化：不copy整个DataFrame，直接在原DataFrame上添加vwap列
        使用完后由调用者负责删除临时添加的列
        """
        vwap_added = False
        
        # 计算VWAP：优先使用amount/volume，如果没有amount则使用(high+low+close)/3
        if 'vwap' not in df.columns:
            if 'amount' in df.columns:
                vwap = df['amount'] / (df['volume'] + self._delta)
            else:
                # 降级方案：使用典型价格
                vwap = (df['high'] + df['low'] + df['close']) / 3.0
            # 直接添加到原DataFrame，不copy
            df['vwap'] = vwap
            vwap_added = True
        else:
            vwap = df['vwap']
        
        # 填充NaN值（直接修改原DataFrame，使用前向填充）
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for col in numeric_cols:
            if col in df.columns:
                # 使用新的API替代已弃用的fillna(method='ffill')
                df[col] = df[col].ffill().fillna(0)
        
        return vwap, vwap_added

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有158个因子"""
        # 预处理数据：直接修改原DataFrame，添加vwap列（如果不存在）
        vwap, vwap_added = self._preprocess_dataframe(df)
        
        # 预计算常用中间结果，避免重复计算
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        # vwap 已经从预处理中获取
        
        # 预计算shift结果
        close_shift1 = close.shift(1)
        volume_shift1 = volume.shift(1)
        
        # 预计算常用比率
        close_ratio = close / (close_shift1 + self._delta)
        volume_ratio = volume / (volume_shift1 + self._delta)
        log_volume = np.log(volume + 1)
        log_volume_ratio = np.log(volume_ratio + 1)
        
        # 预计算价格变化
        close_diff = close - close_shift1
        volume_diff = volume - volume_shift1
        
        # 收集所有因子结果
        factor_results = {}
        
        # 1. kbar 因子（9个）
        if self.include_kbar:
            factor_results[f"{self.name}_KMID"] = (close - open_price) / (open_price + self._delta)
            factor_results[f"{self.name}_KLEN"] = (high - low) / (open_price + self._delta)
            hl_diff = high - low + 1e-12
            factor_results[f"{self.name}_KMID2"] = (close - open_price) / hl_diff
            max_oc = pd.concat([open_price, close], axis=1).max(axis=1)
            min_oc = pd.concat([open_price, close], axis=1).min(axis=1)
            factor_results[f"{self.name}_KUP"] = (high - max_oc) / (open_price + self._delta)
            factor_results[f"{self.name}_KUP2"] = (high - max_oc) / hl_diff
            factor_results[f"{self.name}_KLOW"] = (min_oc - low) / (open_price + self._delta)
            factor_results[f"{self.name}_KLOW2"] = (min_oc - low) / hl_diff
            factor_results[f"{self.name}_KSFT"] = (2 * close - high - low) / (open_price + self._delta)
            factor_results[f"{self.name}_KSFT2"] = (2 * close - high - low) / hl_diff
        
        # 2. price 因子（4个）
        if self.include_price:
            close_eps = close + self._delta
            factor_results[f"{self.name}_OPEN0"] = open_price / close_eps
            factor_results[f"{self.name}_HIGH0"] = high / close_eps
            factor_results[f"{self.name}_LOW0"] = low / close_eps
            factor_results[f"{self.name}_VWAP0"] = vwap / close_eps
        
        # 3. rolling 因子（145个）- 批量计算以提高效率
        if self.include_rolling:
            close_eps = close + self._delta
            for window in self.rolling_windows:
                # ROC
                factor_results[f"{self.name}_ROC{window}"] = close.shift(window) / close_eps
                
                # MA
                factor_results[f"{self.name}_MA{window}"] = close.rolling(window).mean() / close_eps
                
                # STD
                factor_results[f"{self.name}_STD{window}"] = close.rolling(window).std() / close_eps
                
                # BETA, RSQR, RESI - 使用向量化线性回归
                beta, rsqr, resi = self._vectorized_linear_regression(close, window)
                factor_results[f"{self.name}_BETA{window}"] = beta / close_eps
                factor_results[f"{self.name}_RSQR{window}"] = rsqr
                factor_results[f"{self.name}_RESI{window}"] = resi / close_eps
                
                # MAX, MIN
                factor_results[f"{self.name}_MAX{window}"] = high.rolling(window).max() / close_eps
                factor_results[f"{self.name}_MIN{window}"] = low.rolling(window).min() / close_eps
                
                # QTLU, QTLD
                factor_results[f"{self.name}_QTLU{window}"] = close.rolling(window).quantile(0.8) / close_eps
                factor_results[f"{self.name}_QTLD{window}"] = close.rolling(window).quantile(0.2) / close_eps
                
                # RANK
                factor_results[f"{self.name}_RANK{window}"] = close.rolling(window).rank(pct=True)
                
                # RSV
                high_max = high.rolling(window).max()
                low_min = low.rolling(window).min()
                factor_results[f"{self.name}_RSV{window}"] = (close - low_min) / (high_max - low_min + 1e-12)
                
                # IMAX, IMIN, IMXD - 使用向量化argmax/argmin
                imax, imin = self._vectorized_argmaxmin(high, low, window)
                factor_results[f"{self.name}_IMAX{window}"] = imax / window
                factor_results[f"{self.name}_IMIN{window}"] = imin / window
                factor_results[f"{self.name}_IMXD{window}"] = (imax - imin) / window
                
                # CORR
                factor_results[f"{self.name}_CORR{window}"] = close.rolling(window).corr(log_volume)
                
                # CORD
                factor_results[f"{self.name}_CORD{window}"] = close_ratio.rolling(window).corr(log_volume_ratio)
                
                # CNTP, CNTN, CNTD
                close_up = (close > close_shift1).astype(float)
                close_down = (close < close_shift1).astype(float)
                factor_results[f"{self.name}_CNTP{window}"] = close_up.rolling(window).mean()
                factor_results[f"{self.name}_CNTN{window}"] = close_down.rolling(window).mean()
                factor_results[f"{self.name}_CNTD{window}"] = factor_results[f"{self.name}_CNTP{window}"] - factor_results[f"{self.name}_CNTN{window}"]
                
                # SUMP, SUMN, SUMD
                close_diff_abs = close_diff.abs()
                close_gain = close_diff.clip(lower=0)
                close_loss = (-close_diff).clip(lower=0)
                sum_gain = close_gain.rolling(window).sum()
                sum_loss = close_loss.rolling(window).sum()
                sum_total = close_diff_abs.rolling(window).sum()
                factor_results[f"{self.name}_SUMP{window}"] = sum_gain / (sum_total + self._delta)
                factor_results[f"{self.name}_SUMN{window}"] = sum_loss / (sum_total + self._delta)
                factor_results[f"{self.name}_SUMD{window}"] = (sum_gain - sum_loss) / (sum_total + self._delta)
                
                # VMA, VSTD
                volume_eps = volume + self._delta
                factor_results[f"{self.name}_VMA{window}"] = volume.rolling(window).mean() / volume_eps
                factor_results[f"{self.name}_VSTD{window}"] = volume.rolling(window).std() / volume_eps
                
                # WVMA
                abs_change = np.abs(close_ratio - 1) * volume
                factor_results[f"{self.name}_WVMA{window}"] = abs_change.rolling(window).std() / (abs_change.rolling(window).mean() + self._delta)
                
                # VSUMP, VSUMN, VSUMD
                volume_diff_abs = volume_diff.abs()
                volume_gain = volume_diff.clip(lower=0)
                volume_loss = (-volume_diff).clip(lower=0)
                vsum_gain = volume_gain.rolling(window).sum()
                vsum_loss = volume_loss.rolling(window).sum()
                vsum_total = volume_diff_abs.rolling(window).sum()
                factor_results[f"{self.name}_VSUMP{window}"] = vsum_gain / (vsum_total + self._delta)
                factor_results[f"{self.name}_VSUMN{window}"] = vsum_loss / (vsum_total + self._delta)
                factor_results[f"{self.name}_VSUMD{window}"] = (vsum_gain - vsum_loss) / (vsum_total + self._delta)
        
        return pd.DataFrame(factor_results, index=df.index)

    def _vectorized_linear_regression(self, series: pd.Series, window: int) -> tuple:
        """
        优化的线性回归计算：返回(beta, rsqr, resi)
        使用优化的apply方法，比纯循环更快
        """
        # 预计算x向量（时间索引）
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)
        
        # 预分配结果数组
        n = len(series)
        beta_arr = np.full(n, np.nan)
        rsqr_arr = np.full(n, np.nan)
        resi_arr = np.full(n, np.nan)
        
        values = series.values
        
        # 向量化计算（虽然仍有循环，但比apply快）
        for i in range(window - 1, n):
            y_window = values[i - window + 1:i + 1]
            
            if len(y_window) < 2:
                continue
            
            # 检查NaN和方差
            valid_mask = ~np.isnan(y_window)
            if np.sum(valid_mask) < 2:
                continue
            
            y_valid = y_window[valid_mask]
            if np.var(y_valid) == 0:
                continue
            
            y_mean = np.mean(y_valid)
            y_centered = y_window - y_mean
            
            # 计算斜率（beta）
            if x_var == 0:
                beta_val = np.nan
            else:
                # 只使用有效值计算
                x_valid = x_centered[valid_mask]
                y_valid_centered = y_centered[valid_mask]
                numerator = np.sum(x_valid * y_valid_centered)
                beta_val = numerator / x_var
            
            if np.isnan(beta_val):
                continue
            
            beta_arr[i] = beta_val
            
            # 计算R²
            y_pred = beta_val * x_centered + y_mean
            ss_res = np.sum((y_window - y_pred) ** 2)
            ss_tot = np.sum(y_centered ** 2)
            if ss_tot > 0:
                rsqr_arr[i] = 1 - (ss_res / ss_tot)
            
            # 残差（最后一个点）
            y_pred_last = beta_val * x_centered[-1] + y_mean
            resi_arr[i] = y_window[-1] - y_pred_last
        
        beta = pd.Series(beta_arr, index=series.index)
        rsqr = pd.Series(rsqr_arr, index=series.index)
        resi = pd.Series(resi_arr, index=series.index)
        
        return beta, rsqr, resi

    def _vectorized_argmaxmin(self, high: pd.Series, low: pd.Series, window: int) -> tuple:
        """
        优化的argmax/argmin计算
        使用向量化循环，比apply更快
        """
        n = len(high)
        imax_arr = np.full(n, np.nan)
        imin_arr = np.full(n, np.nan)
        
        high_values = high.values
        low_values = low.values
        
        for i in range(window - 1, n):
            high_window = high_values[i - window + 1:i + 1]
            low_window = low_values[i - window + 1:i + 1]
            
            if len(high_window) > 0:
                imax_arr[i] = len(high_window) - 1 - np.nanargmax(high_window)
            if len(low_window) > 0:
                imin_arr[i] = len(low_window) - 1 - np.nanargmin(low_window)
        
        imax = pd.Series(imax_arr, index=high.index)
        imin = pd.Series(imin_arr, index=low.index)
        
        return imax, imin

    def get_output_specs(self) -> List[FeatureSpec]:
        """返回所有158个因子的输出元数据"""
        return [FeatureSpec(name=name) for name in self.factor_names]
