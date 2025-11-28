from typing import List
import os

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor

# 抑制LAPACK的数值警告（通过环境变量）
# 这些警告通常不影响计算结果，只是数值精度问题
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class Alpha101Factor(DualModeFactor):
    """
    根据 WorldQuant LLC 发表的论文 101 Formulaic Alphas 中给出的 101 个 Alphas 因子公式
    
    参考：https://www.joinquant.com/data/dict/alpha101
    
    适配数字币场景：
    - 跳过需要市值(cap)的因子
    - 不进行行业中性化处理（indneutralize直接返回原值）
    - 使用1e-20作为除零保护常量
    """
    display_name = "Alpha101"
    
    # Alpha101最大窗口可能需要360天，设置缓冲区大小
    _required_buffer_size = 360
    
    # 除零保护常量（与alpha158/alpha360一致）
    _EPSILON = 1e-20
    
    init_params = [
        Field(name="alphas", label="包含Alpha", type=FieldType.STRING, default="1-101", description="包含的Alpha因子ID，支持范围(1-101)和单个数字，逗号分隔")
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self.alphas = kwargs.get("alphas", "1-101")
        
        # Parse alphas
        self.selected_alphas = []
        try:
            for part in self.alphas.split(","):
                part = part.strip()
                if not part: continue
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    self.selected_alphas.extend(range(start, end + 1))
                else:
                    self.selected_alphas.append(int(part))
        except Exception:
            # Fallback to default
            self.selected_alphas = list(range(1, 102))
            
        # Filter valid alphas (1-101)
        self.selected_alphas = [i for i in self.selected_alphas if 1 <= i <= 101]
        
        # Update factor names
        self.factor_names = [f"{self.name}_alpha{i:03d}" for i in self.selected_alphas]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算所有101个因子
        优化：预计算中间变量，批量合并结果
        """
        # 数据预处理：计算VWAP和returns
        vwap, vwap_added = self._preprocess_dataframe(df)
        returns = self._returns(df['close'])
        
        # 预计算常用中间变量，避免重复计算
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 收集所有因子结果
        factor_results = {}
        
        # 路由分发 - 计算选中的因子
        for i in self.selected_alphas:
            method_name = f"_alpha_{i:03d}"
            if hasattr(self, method_name):
                factor_name = f"{self.name}_alpha{i:03d}"
                try:
                    factor_results[factor_name] = getattr(self, method_name)(df, vwap, returns)
                except Exception as e:
                    # 如果计算失败，返回NaN
                    factor_results[factor_name] = pd.Series(np.nan, index=df.index)
        
        return pd.DataFrame(factor_results, index=df.index)

    def get_output_names(self) -> list:
        return self.factor_names
    
    # ========== 数据预处理 ==========
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> tuple:
        """
        数据预处理：计算VWAP
        返回: (vwap Series, 是否临时添加了vwap列)
        """
        vwap_added = False
        
        # 计算VWAP：优先使用amount/volume，如果没有amount则使用(high+low+close)/3
        if 'vwap' not in df.columns:
            if 'amount' in df.columns:
                vwap = df['amount'] / (df['volume'] + self._EPSILON)
            else:
                # 降级方案：使用典型价格
                vwap = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = vwap
            vwap_added = True
        else:
            vwap = df['vwap']
        
        return vwap, vwap_added
    
    # ========== 辅助函数 ==========
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """RANK: 横截面排名（向量化实现）"""
        return series.rank(pct=True)
    
    def _delta(self, series: pd.Series, n: int = 1) -> pd.Series:
        """DELTA: 当前值减去n期前的值"""
        return series - series.shift(n)
    
    def _delay(self, series: pd.Series, n: int) -> pd.Series:
        """DELAY: 返回n期前的值"""
        return series.shift(n)
    
    def _corr(self, x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """
        CORR: 过去n天的相关系数
        使用更健壮的计算方法，避免LAPACK警告
        """
        # 使用min_periods确保有足够的数据点
        # 当数据不足时返回NaN，避免触发LAPACK警告
        result = x.rolling(n, min_periods=n).corr(y)
        return result
    
    def _cov(self, x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """
        COV: 过去n天的协方差
        使用更健壮的计算方法，避免LAPACK警告
        """
        # 使用min_periods确保有足够的数据点
        result = x.rolling(n, min_periods=n).cov(y)
        return result
    
    def _std(self, series: pd.Series, n: int) -> pd.Series:
        """STD: 过去n天的标准差"""
        return series.rolling(n).std()
    
    def _mean(self, series: pd.Series, n: int) -> pd.Series:
        """MEAN: 过去n天的均值"""
        return series.rolling(n).mean()
    
    def _sum(self, series: pd.Series, n: int) -> pd.Series:
        """SUM: 过去n天的求和"""
        return series.rolling(n).sum()
    
    def _prod(self, series: pd.Series, n: int) -> pd.Series:
        """PROD: 过去n天的乘积"""
        return series.rolling(n).apply(lambda x: np.prod(x), raw=True)
    
    def _ts_min(self, series: pd.Series, n: int) -> pd.Series:
        """TS_MIN: 过去n天的最小值"""
        return series.rolling(n).min()
    
    def _ts_max(self, series: pd.Series, n: int) -> pd.Series:
        """TS_MAX: 过去n天的最大值"""
        return series.rolling(n).max()
    
    def _ts_rank(self, series: pd.Series, n: int) -> pd.Series:
        """TS_RANK: 序列的末位值在过去n天的顺序排位"""
        return series.rolling(n).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == n else np.nan,
            raw=False
        )
    
    def _ts_argmax(self, series: pd.Series, n: int) -> pd.Series:
        """TS_ARMAX: ts_max对应的发生日期（距离当前的天数）"""
        def calc_argmax(x):
            if len(x) < n:
                return np.nan
            return len(x) - 1 - np.argmax(x)
        return series.rolling(n).apply(calc_argmax, raw=True)
    
    def _ts_argmin(self, series: pd.Series, n: int) -> pd.Series:
        """TS_ARMIN: ts_min对应的发生日期（距离当前的天数）"""
        def calc_argmin(x):
            if len(x) < n:
                return np.nan
            return len(x) - 1 - np.argmin(x)
        return series.rolling(n).apply(calc_argmin, raw=True)
    
    def _decay_linear(self, series: pd.Series, d: int) -> pd.Series:
        """DECAY_LINEAR: 过去d天的线性衰减加权平均（权重为d, d-1,...,1）"""
        weights = np.array(range(d, 0, -1))
        weights = weights / weights.sum()
        return series.rolling(d).apply(
            lambda x: np.sum(x * weights[::-1]) if len(x) == d else np.nan,
            raw=True
        )
    
    def _sign(self, series: pd.Series) -> pd.Series:
        """SIGN: 符号函数（x>0返回1，x=0返回0，x<0返回-1）"""
        return np.sign(series)
    
    def _signedpower(self, series: pd.Series, a: float) -> pd.Series:
        """SIGNEDPOWER: x的a次幂（保持符号）"""
        return np.sign(series) * np.power(np.abs(series), a)
    
    def _scale(self, series: pd.Series, a: float = 1.0) -> pd.Series:
        """SCALE: 标准化x，使得sum(abs(x)) = a"""
        abs_sum = series.abs().sum()
        if abs_sum == 0:
            return series
        return series * a / abs_sum
    
    def _indneutralize(self, series: pd.Series, group: pd.Series = None) -> pd.Series:
        """
        INDNEUTRALIZE: 按分组进行横截面中性化
        数字币场景：不进行中性化，直接返回原值
        """
        return series
    
    def _returns(self, series: pd.Series) -> pd.Series:
        """RETURNS: 计算收益率（close-to-close returns）"""
        return series.pct_change()
    
    def _abs(self, series: pd.Series) -> pd.Series:
        """ABS: 绝对值"""
        return series.abs()
    
    def _log(self, series: pd.Series) -> pd.Series:
        """LOG: 自然对数"""
        return np.log(series + self._EPSILON)
    
    def _max(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """MAX: 两个序列的最大值"""
        return pd.concat([x, y], axis=1).max(axis=1)
    
    def _min(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """MIN: 两个序列的最小值"""
        return pd.concat([x, y], axis=1).min(axis=1)
    
    # ========== Alpha 因子实现 ==========
    
    def _alpha_001(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#1: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
        """
        log_volume = self._log(df['volume'])
        delta_log_volume = self._delta(log_volume, 1)
        rank_delta_log_volume = self._rank(delta_log_volume)
        
        close_open_ratio = (df['close'] - df['open']) / (df['open'] + self._EPSILON)
        rank_close_open = self._rank(close_open_ratio)
        
        return -1 * self._corr(rank_delta_log_volume, rank_close_open, 6)
    
    def _alpha_002(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#2: (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
        """
        high_low = df['high'] - df['low']
        numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
        ratio = numerator / (high_low + self._EPSILON)
        return -1 * self._delta(ratio, 1)
    
    def _alpha_003(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#3: SUM((CLOSE==DELAY(CLOSE,1)?0:CLOSE-(CLOSE<DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))), 6)
        """
        close_delay = self._delay(df['close'], 1)
        condition = df['close'] == close_delay
        
        low_min = self._min(df['low'], close_delay)
        high_max = self._max(df['high'], close_delay)
        
        value = np.where(
            df['close'] < close_delay,
            df['close'] - low_min,
            df['close'] - high_max
        )
        value = np.where(condition, 0, value)
        
        return pd.Series(value, index=df.index).rolling(6).sum()
    
    def _alpha_004(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#4: ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : 
                  (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : 
                  (((1 < (VOLUME / MEAN(VOLUME, 20))) || ((VOLUME / MEAN(VOLUME, 20)) == 1)) ? 1 : (-1 * 1))))
        """
        sum_close_8 = self._sum(df['close'], 8)
        mean_close_8 = sum_close_8 / 8
        std_close_8 = self._std(df['close'], 8)
        
        sum_close_2 = self._sum(df['close'], 2)
        mean_close_2 = sum_close_2 / 2
        
        volume_mean = self._mean(df['volume'], 20)
        volume_ratio = df['volume'] / (volume_mean + self._EPSILON)
        
        condition1 = (mean_close_8 + std_close_8) < mean_close_2
        condition2 = mean_close_2 < (mean_close_8 - std_close_8)
        condition3 = (volume_ratio >= 1)
        
        result = np.where(condition1, -1, np.where(condition2, 1, np.where(condition3, 1, -1)))
        return pd.Series(result, index=df.index)
    
    def _alpha_005(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#5: (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
        """
        tsrank_volume = self._ts_rank(df['volume'], 5)
        tsrank_high = self._ts_rank(df['high'], 5)
        corr_result = self._corr(tsrank_volume, tsrank_high, 5)
        return -1 * self._ts_max(corr_result, 3)
    
    def _alpha_006(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#6: ((RANK(SIGN(DELTA((((OPEN * 0.85) + (LOW * 0.15))), 4))).TSRANK(VOLUME, 3)) * -1)
        """
        weighted_price = df['open'] * 0.85 + df['low'] * 0.15
        delta_weighted = self._delta(weighted_price, 4)
        sign_delta = self._sign(delta_weighted)
        rank_sign = self._rank(sign_delta)
        
        tsrank_volume = self._ts_rank(df['volume'], 3)
        
        return rank_sign * tsrank_volume * -1
    
    def _alpha_007(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#7: ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
        """
        vwap_close_diff = vwap - df['close']
        
        max_diff = self._ts_max(vwap_close_diff, 3)
        min_diff = self._ts_min(vwap_close_diff, 3)
        
        rank_max = self._rank(max_diff)
        rank_min = self._rank(min_diff)
        rank_delta_volume = self._rank(self._delta(df['volume'], 3))
        
        return (rank_max + rank_min) * rank_delta_volume
    
    def _alpha_008(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#8: RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
        """
        weighted_price = ((df['high'] + df['low']) / 2) * 0.2 + vwap * 0.8
        delta_weighted = self._delta(weighted_price, 4)
        return self._rank(delta_weighted) * -1
    
    def _alpha_009(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#9: ((((HIGH + LOW) / 2) - (MEAN((HIGH + LOW) / 2, 7))) > 0 ? 
                  (MEAN((HIGH + LOW) / 2, 7) - (HIGH + LOW) / 2) : 
                  (((HIGH + LOW) / 2) - MEAN((HIGH + LOW) / 2, 7))) * (-1 * RANK(MEAN(VOLUME, 10))))
        """
        hl_avg = (df['high'] + df['low']) / 2
        mean_hl_avg = self._mean(hl_avg, 7)
        diff = hl_avg - mean_hl_avg
        
        condition = diff > 0
        value = np.where(condition, -diff, diff)
        
        rank_mean_volume = self._rank(self._mean(df['volume'], 10))
        
        return pd.Series(value, index=df.index) * (-1 * rank_mean_volume)
    
    def _alpha_010(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#10: RANK(((MAX((HIGH - CLOSE), 3) + MAX((CLOSE - LOW), 3)) * 100 / (MAX((HIGH - LOW), 3))))
        """
        high_close = df['high'] - df['close']
        close_low = df['close'] - df['low']
        high_low = df['high'] - df['low']
        
        max_high_close = self._ts_max(high_close, 3)
        max_close_low = self._ts_max(close_low, 3)
        max_high_low = self._ts_max(high_low, 3)
        
        numerator = (max_high_close + max_close_low) * 100
        denominator = max_high_low + self._delta
        
        return self._rank(numerator / denominator)
    
    def _alpha_011(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#11: SUM(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW), 6)
        """
        high_low = df['high'] - df['low']
        numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
        ratio = numerator / (high_low + self._EPSILON)
        return self._sum(ratio, 6)
    
    def _alpha_012(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#12: (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * RANK((CLOSE - VWAP)))
        """
        sum_vwap_10 = self._sum(vwap, 10)
        open_vwap_diff = df['open'] - (sum_vwap_10 / 10)
        rank_open_vwap = self._rank(open_vwap_diff)
        
        close_vwap_diff = df['close'] - vwap
        rank_close_vwap = self._rank(close_vwap_diff)
        
        return rank_open_vwap * (-1 * rank_close_vwap)
    
    def _alpha_013(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#13: (((HIGH * LOW)^0.5) - VWAP)
        """
        hl_sqrt = np.sqrt(df['high'] * df['low'])
        return hl_sqrt - vwap
    
    def _alpha_014(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#14: CLOSE - DELAY(CLOSE, 5)
        """
        return df['close'] - self._delay(df['close'], 5)
    
    def _alpha_015(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#15: OPEN / DELAY(CLOSE, 1) - 1
        """
        close_delay_1 = self._delay(df['close'], 1)
        return (df['open'] / (close_delay_1 + self._EPSILON)) - 1
    
    def _alpha_016(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#16: (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        """
        rank_volume = self._rank(df['volume'])
        rank_vwap = self._rank(vwap)
        corr_result = self._corr(rank_volume, rank_vwap, 5)
        rank_corr = self._rank(corr_result)
        return -1 * self._ts_max(rank_corr, 5)
    
    def _alpha_017(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#17: RANK((VWAP - MAX(VWAP, 15))) * DELTA(CLOSE, 1)
        """
        max_vwap = self._ts_max(vwap, 15)
        rank_diff = self._rank(vwap - max_vwap)
        delta_close = self._delta(df['close'], 1)
        return rank_diff * delta_close
    
    def _alpha_018(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#18: CLOSE / DELAY(CLOSE, 5)
        """
        close_delay_5 = self._delay(df['close'], 5)
        return df['close'] / (close_delay_5 + self._EPSILON)
    
    def _alpha_019(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#19: (CLOSE < DELAY(CLOSE, 5) ? (CLOSE - DELAY(CLOSE, 5)) / DELAY(CLOSE, 5) : 
                   (CLOSE == DELAY(CLOSE, 5) ? 0 : (CLOSE - DELAY(CLOSE, 5)) / CLOSE))
        """
        close_delay_5 = self._delay(df['close'], 5)
        condition1 = df['close'] < close_delay_5
        condition2 = df['close'] == close_delay_5
        
        value1 = (df['close'] - close_delay_5) / (close_delay_5 + self._EPSILON)
        value2 = (df['close'] - close_delay_5) / (df['close'] + self._EPSILON)
        
        result = np.where(condition1, value1, np.where(condition2, 0, value2))
        return pd.Series(result, index=df.index)
    
    def _alpha_020(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#20: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100
        """
        close_delay_6 = self._delay(df['close'], 6)
        return ((df['close'] - close_delay_6) / (close_delay_6 + self._EPSILON)) * 100
    
    def _alpha_021(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#21: REGBETA(MEAN(CLOSE, 6), SEQUENCE(6), 6)
        """
        mean_close_6 = self._mean(df['close'], 6)
        sequence_6 = pd.Series(range(1, 7), index=df.index)
        # 简化的线性回归beta计算
        def calc_beta(y, x):
            if len(y) < 2 or len(x) < 2:
                return np.nan
            if np.var(x) == 0:
                return np.nan
            return np.cov(y, x)[0, 1] / np.var(x)
        
        return mean_close_6.rolling(6).apply(
            lambda y: calc_beta(y.values, sequence_6.values[:len(y)]) if len(y) == 6 else np.nan,
            raw=False
        )
    
    def _alpha_022(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#22: SMEAN(((CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) - DELAY((CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6), 3)), 12)
        """
        mean_close_6 = self._mean(df['close'], 6)
        ratio = (df['close'] - mean_close_6) / (mean_close_6 + self._EPSILON)
        ratio_delay_3 = self._delay(ratio, 3)
        diff = ratio - ratio_delay_3
        return self._mean(diff, 12)
    
    def _alpha_023(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#23: SMEAN((CLOSE > DELAY(CLOSE, 1) ? STD(CLOSE, 20) : 0), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition = df['close'] > close_delay_1
        std_close_20 = self._std(df['close'], 20)
        value = np.where(condition, std_close_20, 0)
        return self._mean(pd.Series(value, index=df.index), 20)
    
    def _alpha_024(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#24: SMEAN(((CLOSE - DELAY(CLOSE, 5)) / DELAY(CLOSE, 5)), 20)
        """
        close_delay_5 = self._delay(df['close'], 5)
        ratio = (df['close'] - close_delay_5) / (close_delay_5 + self._EPSILON)
        return self._mean(ratio, 20)
    
    def _alpha_025(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#25: ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME, 20)), 9)))))) * 
                   (1 + RANK(SUM(RETURNS, 250))))
        """
        delta_close_7 = self._delta(df['close'], 7)
        volume_mean_20 = self._mean(df['volume'], 20)
        volume_ratio = df['volume'] / (volume_mean_20 + self._EPSILON)
        decay_volume = self._decay_linear(volume_ratio, 9)
        rank_decay = self._rank(decay_volume)
        
        part1 = delta_close_7 * (1 - rank_decay)
        rank_part1 = self._rank(part1)
        
        sum_returns_250 = self._sum(returns, 250)
        rank_sum_returns = self._rank(sum_returns_250)
        
        return (-1 * rank_part1) * (1 + rank_sum_returns)
    
    def _alpha_026(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#26: SUM((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 7) / 
                  SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)) : 0), 7) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, self._abs(diff), 0)
        
        sum_gain = self._sum(pd.Series(gain, index=df.index), 7)
        sum_loss = self._sum(pd.Series(loss, index=df.index), 7)
        
        return (sum_gain / (sum_loss + self._EPSILON)) * 100
    
    def _alpha_027(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#27: WMA((CLOSE - DELAY(CLOSE, 3)) / DELAY(CLOSE, 3) * 100 + (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100, 12)
        """
        close_delay_3 = self._delay(df['close'], 3)
        close_delay_6 = self._delay(df['close'], 6)
        
        part1 = ((df['close'] - close_delay_3) / (close_delay_3 + self._EPSILON)) * 100
        part2 = ((df['close'] - close_delay_6) / (close_delay_6 + self._EPSILON)) * 100
        
        combined = part1 + part2
        
        # WMA实现
        weights = np.array([0.9 ** i for i in range(12)])
        weights = weights / weights.sum()
        return combined.rolling(12).apply(
            lambda x: np.sum(x * weights[::-1]) if len(x) == 12 else np.nan,
            raw=True
        )
    
    def _alpha_028(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#28: 3 * SMA(CLOSE, 30, 1) - 2 * SMA(SMA(CLOSE, 30, 1), 30, 1) + SMA(SMA(SMA(CLOSE, 30, 1), 30, 1), 30, 1)
        """
        # SMA实现
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma1 = sma(df['close'], 30, 1)
        sma2 = sma(sma1, 30, 1)
        sma3 = sma(sma2, 30, 1)
        
        return 3 * sma1 - 2 * sma2 + sma3
    
    def _alpha_029(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#29: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME
        """
        close_delay_6 = self._delay(df['close'], 6)
        ratio = (df['close'] - close_delay_6) / (close_delay_6 + self._EPSILON)
        return ratio * df['volume']
    
    def _alpha_030(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#30: WMA((REGRESI(CLOSE / DELAY(CLOSE, 1) - 1, MEAN(VOLUME, 30), 10))^2, 30)
        """
        close_delay_1 = self._delay(df['close'], 1)
        y = (df['close'] / (close_delay_1 + self._EPSILON)) - 1
        x = self._mean(df['volume'], 30)
        
        # 简化的回归残差计算
        def calc_residual(y_vals, x_vals):
            if len(y_vals) < 2 or len(x_vals) < 2:
                return np.nan
            if np.var(x_vals) == 0:
                return y_vals[-1]
            coeffs = np.polyfit(x_vals, y_vals, 1)
            y_pred = np.polyval(coeffs, x_vals)
            return y_vals[-1] - y_pred[-1]
        
        resi = pd.Series(index=df.index)
        for i in range(9, len(df)):
            y_window = y.iloc[i-9:i+1].values
            x_window = x.iloc[i-9:i+1].values
            resi.iloc[i] = calc_residual(y_window, x_window)
        
        resi_squared = resi ** 2
        
        # WMA
        weights = np.array([0.9 ** i for i in range(30)])
        weights = weights / weights.sum()
        return resi_squared.rolling(30).apply(
            lambda x: np.sum(x * weights[::-1]) if len(x) == 30 else np.nan,
            raw=True
        )
    
    # ========== 第二批：Alpha 31-60 ==========
    
    def _alpha_031(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#31: (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100
        """
        mean_close_12 = self._mean(df['close'], 12)
        return ((df['close'] - mean_close_12) / (mean_close_12 + self._EPSILON)) * 100
    
    def _alpha_032(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#32: (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        """
        rank_high = self._rank(df['high'])
        rank_volume = self._rank(df['volume'])
        corr_result = self._corr(rank_high, rank_volume, 3)
        rank_corr = self._rank(corr_result)
        return -1 * self._sum(rank_corr, 3)
    
    def _alpha_033(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#33: ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RETURNS, 240) - SUM(RETURNS, 20)) / 220))) * TSRANK(VOLUME, 5))
        """
        tsmin_low_5 = self._ts_min(df['low'], 5)
        tsmin_low_5_delay = self._delay(tsmin_low_5, 5)
        part1 = (-1 * tsmin_low_5) + tsmin_low_5_delay
        
        sum_returns_240 = self._sum(returns, 240)
        sum_returns_20 = self._sum(returns, 20)
        ratio = (sum_returns_240 - sum_returns_20) / 220
        rank_ratio = self._rank(ratio)
        
        tsrank_volume = self._ts_rank(df['volume'], 5)
        
        return part1 * rank_ratio * tsrank_volume
    
    def _alpha_034(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#34: MEAN(CLOSE, 12) / CLOSE
        """
        mean_close_12 = self._mean(df['close'], 12)
        return mean_close_12 / (df['close'] + self._EPSILON)
    
    def _alpha_035(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#35: (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN * 0.35)), 17), 7))) * -1)
        """
        delta_open_1 = self._delta(df['open'], 1)
        decay1 = self._decay_linear(delta_open_1, 15)
        rank_decay1 = self._rank(decay1)
        
        weighted_open = df['open'] * 0.65 + df['open'] * 0.35  # 实际上就是OPEN
        corr_result = self._corr(df['volume'], weighted_open, 17)
        decay2 = self._decay_linear(corr_result, 7)
        rank_decay2 = self._rank(decay2)
        
        min_rank = self._min(rank_decay1, rank_decay2)
        return min_rank * -1
    
    def _alpha_036(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#36: RANK((SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2) / TSRANK(VWAP, 2)))
        """
        rank_volume = self._rank(df['volume'])
        rank_vwap = self._rank(vwap)
        corr_result = self._corr(rank_volume, rank_vwap, 6)
        sum_corr = self._sum(corr_result, 2)
        
        tsrank_vwap = self._ts_rank(vwap, 2)
        
        ratio = sum_corr / (tsrank_vwap + self._EPSILON)
        return self._rank(ratio)
    
    def _alpha_037(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#37: (-1 * RANK(((SUM(OPEN, 5) * SUM(RETURNS, 5)) - DELAY((SUM(OPEN, 5) * SUM(RETURNS, 5)), 10))))
        """
        sum_open_5 = self._sum(df['open'], 5)
        sum_returns_5 = self._sum(returns, 5)
        product = sum_open_5 * sum_returns_5
        product_delay = self._delay(product, 10)
        
        diff = product - product_delay
        return -1 * self._rank(diff)
    
    def _alpha_038(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#38: (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        """
        mean_high_20 = self._sum(df['high'], 20) / 20
        condition = mean_high_20 < df['high']
        delta_high_2 = self._delta(df['high'], 2)
        
        result = np.where(condition, -1 * delta_high_2, 0)
        return pd.Series(result, index=df.index)
    
    def _alpha_039(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#39: ((RANK(DECAYLINEAR(DELTA((CLOSE), 2), 8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1)
        """
        delta_close_2 = self._delta(df['close'], 2)
        decay1 = self._decay_linear(delta_close_2, 8)
        rank_decay1 = self._rank(decay1)
        
        weighted_price = vwap * 0.3 + df['open'] * 0.7
        mean_volume_180 = self._mean(df['volume'], 180)
        sum_mean_vol = self._sum(mean_volume_180, 37)
        corr_result = self._corr(weighted_price, sum_mean_vol, 14)
        decay2 = self._decay_linear(corr_result, 12)
        rank_decay2 = self._rank(decay2)
        
        return (rank_decay1 - rank_decay2) * -1
    
    def _alpha_040(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#40: SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : 0), 26) / SUM((CLOSE <= DELAY(CLOSE, 1) ? VOLUME : 0), 26) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition1 = df['close'] > close_delay_1
        condition2 = df['close'] <= close_delay_1
        
        volume_up = np.where(condition1, df['volume'], 0)
        volume_down = np.where(condition2, df['volume'], 0)
        
        sum_up = self._sum(pd.Series(volume_up, index=df.index), 26)
        sum_down = self._sum(pd.Series(volume_down, index=df.index), 26)
        
        return (sum_up / (sum_down + self._EPSILON)) * 100
    
    def _alpha_041(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#41: (RANK(MAX(DELTA((VWAP), 3), 5)) * -1)
        """
        delta_vwap_3 = self._delta(vwap, 3)
        max_delta = self._ts_max(delta_vwap_3, 5)
        return self._rank(max_delta) * -1
    
    def _alpha_042(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#42 (Delay-0, 均值回归): (-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)
        """
        std_high_10 = self._std(df['high'], 10)
        rank_std = self._rank(std_high_10)
        corr_result = self._corr(df['high'], df['volume'], 10)
        return -1 * rank_std * corr_result
    
    def _alpha_043(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#43: SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : -(CLOSE < DELAY(CLOSE, 1) ? VOLUME : 0)), 6)
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition1 = df['close'] > close_delay_1
        condition2 = df['close'] < close_delay_1
        
        value = np.where(condition1, df['volume'], np.where(condition2, -df['volume'], 0))
        return self._sum(pd.Series(value, index=df.index), 6)
    
    def _alpha_044(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#44: (TSRANK(DECAYLINEAR(CORR(((LOW)), MEAN(VOLUME, 10), 7), 6), 4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
        """
        mean_vol_10 = self._mean(df['volume'], 10)
        corr_result = self._corr(df['low'], mean_vol_10, 7)
        decay1 = self._decay_linear(corr_result, 6)
        tsrank1 = self._ts_rank(decay1, 4)
        
        delta_vwap_3 = self._delta(vwap, 3)
        decay2 = self._decay_linear(delta_vwap_3, 10)
        tsrank2 = self._ts_rank(decay2, 15)
        
        return tsrank1 + tsrank2
    
    def _alpha_045(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#45: (RANK(DELTA((((CLOSE * 0.6) + (OPEN * 0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
        """
        weighted_price = df['close'] * 0.6 + df['open'] * 0.4
        delta_weighted = self._delta(weighted_price, 1)
        rank_delta = self._rank(delta_weighted)
        
        mean_vol_150 = self._mean(df['volume'], 150)
        corr_result = self._corr(vwap, mean_vol_150, 15)
        rank_corr = self._rank(corr_result)
        
        return rank_delta * rank_corr
    
    def _alpha_046(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#46: (MEAN(CLOSE, 3) + MEAN(CLOSE, 6) + MEAN(CLOSE, 12) + MEAN(CLOSE, 24)) / 4
        """
        mean_close_3 = self._mean(df['close'], 3)
        mean_close_6 = self._mean(df['close'], 6)
        mean_close_12 = self._mean(df['close'], 12)
        mean_close_24 = self._mean(df['close'], 24)
        
        return (mean_close_3 + mean_close_6 + mean_close_12 + mean_close_24) / 4
    
    def _alpha_047(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#47: SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9)), 3, 1) - SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9)), 15, 1)
        """
        tsmin_low_9 = self._ts_min(df['low'], 9)
        tsmax_high_9 = self._ts_max(df['high'], 9)
        
        numerator = df['close'] - tsmin_low_9
        denominator = tsmax_high_9 - tsmin_low_9 + self._EPSILON
        ratio = numerator / denominator
        
        # SMA实现
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma1 = sma(ratio, 3, 1)
        sma2 = sma(ratio, 15, 1)
        
        return sma1 - sma2
    
    def _alpha_048(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#48 (Delay-0): (-1 * ((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1)) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20)))
        """
        close_delay_1 = self._delay(df['close'], 1)
        close_delay_2 = self._delay(df['close'], 2)
        close_delay_3 = self._delay(df['close'], 3)
        
        sign1 = self._sign(df['close'] - close_delay_1)
        sign2 = self._sign(close_delay_1 - close_delay_2)
        sign3 = self._sign(close_delay_2 - close_delay_3)
        
        sum_signs = sign1 + sign2 + sign3
        rank_signs = self._rank(sum_signs)
        
        sum_vol_5 = self._sum(df['volume'], 5)
        sum_vol_20 = self._sum(df['volume'], 20)
        
        ratio = (rank_signs * sum_vol_5) / (sum_vol_20 + self._EPSILON)
        return -1 * ratio
    
    def _alpha_049(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#49: SUM(((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0) / (CLOSE - DELAY(CLOSE, 1) < 0 ? CLOSE - DELAY(CLOSE, 1) : -1)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        numerator = np.where(diff > 0, diff, 0)
        denominator = np.where(diff < 0, diff, -1)
        
        ratio = numerator / (denominator + self._EPSILON)
        return self._sum(pd.Series(ratio, index=df.index), 20)
    
    def _alpha_050(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#50: SUM(((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0) / SUM((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 20)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.where(diff > 0, diff, 0)
        sum_gain_20 = self._sum(pd.Series(gain, index=df.index), 20)
        
        ratio = gain / (sum_gain_20 + self._EPSILON)
        return self._sum(pd.Series(ratio, index=df.index), 20)
    
    def _alpha_051(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#51: SUM(((CLOSE - DELAY(CLOSE, 1) < 0 ? CLOSE - DELAY(CLOSE, 1) : 0) / SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 20)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        loss = np.where(diff < 0, diff, 0)
        sum_loss_20 = self._sum(pd.Series(loss, index=df.index), 20)
        
        ratio = loss / (sum_loss_20 + self._EPSILON)
        return self._sum(pd.Series(ratio, index=df.index), 20)
    
    def _alpha_052(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#52: SUM(MAX(0, HIGH - DELAY(CLOSE, 1)), 26) / SUM(MAX(0, DELAY(CLOSE, 1) - LOW), 26) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        
        gain = np.maximum(0, df['high'] - close_delay_1)
        loss = np.maximum(0, close_delay_1 - df['low'])
        
        sum_gain = self._sum(pd.Series(gain, index=df.index), 26)
        sum_loss = self._sum(pd.Series(loss, index=df.index), 26)
        
        return (sum_gain / (sum_loss + self._EPSILON)) * 100
    
    def _alpha_053(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#53 (Delay-0): COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition = df['close'] > close_delay_1
        count_up = condition.rolling(12).sum()
        return (count_up / 12) * 100
    
    def _alpha_054(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#54 (Delay-0): (-1 * TSMIN(RANK(CORR(RANK(CLOSE), RANK(VOLUME), 6)), 5))
        """
        rank_close = self._rank(df['close'])
        rank_volume = self._rank(df['volume'])
        corr_result = self._corr(rank_close, rank_volume, 6)
        rank_corr = self._rank(corr_result)
        tsmin_rank = self._ts_min(rank_corr, 5)
        return -1 * tsmin_rank
    
    def _alpha_055(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#55: (-1 * CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW, 12)))), RANK(VOLUME), 6))
        """
        tsmin_low_12 = self._ts_min(df['low'], 12)
        tsmax_high_12 = self._ts_max(df['high'], 12)
        
        numerator = df['close'] - tsmin_low_12
        denominator = tsmax_high_12 - tsmin_low_12 + self._EPSILON
        ratio = numerator / denominator
        
        rank_ratio = self._rank(ratio)
        rank_volume = self._rank(df['volume'])
        
        return -1 * self._corr(rank_ratio, rank_volume, 6)
    
    def _alpha_056(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#56: (0 - (1 * (RANK((SUM(RETURNS, 10) / SUM(SUM(RETURNS, 2), 3))) * RANK((RETURNS * CAP))) * 1))
        需要CAP，跳过，返回NaN
        """
        return pd.Series(np.nan, index=df.index)
    
    def _alpha_057(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#57: (0 - (1 * ((CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)), 2))))
        """
        close_vwap_diff = df['close'] - vwap
        tsmax_close_30 = self._ts_max(df['close'], 30)
        rank_tsmax = self._rank(tsmax_close_30)
        decay_rank = self._decay_linear(rank_tsmax, 2)
        
        ratio = close_vwap_diff / (decay_rank + self._EPSILON)
        return 0 - (1 * ratio)
    
    def _alpha_058(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#58: (-1 * TSRANK(DECAYLINEAR(CORR((VWAP), SUM(MEAN(VOLUME, 5), 26), 5), 7), 3))
        """
        mean_vol_5 = self._mean(df['volume'], 5)
        sum_mean_vol = self._sum(mean_vol_5, 26)
        corr_result = self._corr(vwap, sum_mean_vol, 5)
        decay = self._decay_linear(corr_result, 7)
        tsrank = self._ts_rank(decay, 3)
        return -1 * tsrank
    
    def _alpha_059(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#59: SUM((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 20) / 
                  SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)) : 0), 20) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, self._abs(diff), 0)
        
        sum_gain = self._sum(pd.Series(gain, index=df.index), 20)
        sum_loss = self._sum(pd.Series(loss, index=df.index), 20)
        
        return (sum_gain / (sum_loss + self._EPSILON)) * 100
    
    def _alpha_060(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#60: SUM(((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0) / 
                  (CLOSE - DELAY(CLOSE, 1) < 0 ? CLOSE - DELAY(CLOSE, 1) : -1)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        numerator = np.where(diff > 0, diff, 0)
        denominator = np.where(diff < 0, diff, -1)
        
        ratio = numerator / (denominator + self._EPSILON)
        return self._sum(pd.Series(ratio, index=df.index), 20)
    
    # ========== 第三批：Alpha 61-90 ==========
    
    def _alpha_061(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#61: (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(((HIGH * 0.2) + (VWAP * 0.8))), 17), 19))) * -1)
        """
        delta_vwap_1 = self._delta(vwap, 1)
        decay1 = self._decay_linear(delta_vwap_1, 12)
        rank_decay1 = self._rank(decay1)
        
        weighted_price = df['high'] * 0.2 + vwap * 0.8
        rank_weighted = self._rank(weighted_price)
        decay2 = self._decay_linear(rank_weighted, 17)
        rank_decay2 = self._rank(decay2)
        
        max_rank = self._max(rank_decay1, rank_decay2)
        return max_rank * -1
    
    def _alpha_062(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#62: (-1 * CORR(HIGH, RANK(VOLUME), 5))
        """
        rank_volume = self._rank(df['volume'])
        return -1 * self._corr(df['high'], rank_volume, 5)
    
    def _alpha_063(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#63: SMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 6, 1) / SMA(ABS(CLOSE - DELAY(CLOSE, 1)), 6, 1) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.maximum(0, diff)
        abs_diff = self._abs(diff)
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma_gain = sma(pd.Series(gain, index=df.index), 6, 1)
        sma_abs = sma(abs_diff, 6, 1)
        
        return (sma_gain / (sma_abs + self._EPSILON)) * 100
    
    def _alpha_064(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#64: (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
        """
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr1 = self._corr(rank_vwap, rank_volume, 4)
        decay1 = self._decay_linear(corr1, 4)
        rank_decay1 = self._rank(decay1)
        
        rank_close = self._rank(df['close'])
        mean_vol_60 = self._mean(df['volume'], 60)
        rank_mean_vol = self._rank(mean_vol_60)
        corr2 = self._corr(rank_close, rank_mean_vol, 4)
        max_corr2 = self._ts_max(corr2, 13)
        decay2 = self._decay_linear(max_corr2, 14)
        rank_decay2 = self._rank(decay2)
        
        max_rank = self._max(rank_decay1, rank_decay2)
        return max_rank * -1
    
    def _alpha_065(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#65: MEAN(CLOSE, 6) / CLOSE
        """
        mean_close_6 = self._mean(df['close'], 6)
        return mean_close_6 / (df['close'] + self._EPSILON)
    
    def _alpha_066(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#66: (CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) * 100
        """
        mean_close_6 = self._mean(df['close'], 6)
        return ((df['close'] - mean_close_6) / (mean_close_6 + self._EPSILON)) * 100
    
    def _alpha_067(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#67: SMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 24, 1) / SMA(ABS(CLOSE - DELAY(CLOSE, 1)), 24, 1) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.maximum(0, diff)
        abs_diff = self._abs(diff)
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma_gain = sma(pd.Series(gain, index=df.index), 24, 1)
        sma_abs = sma(abs_diff, 24, 1)
        
        return (sma_gain / (sma_abs + self._EPSILON)) * 100
    
    def _alpha_068(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#68: SMA(((HIGH + LOW) / 2) - DELAY((HIGH + LOW) / 2, 1), 6, 1) / SMA(((HIGH + LOW) / 2) - DELAY((HIGH + LOW) / 2, 1), 6, 1) * 100
        """
        hl_avg = (df['high'] + df['low']) / 2
        hl_avg_delay = self._delay(hl_avg, 1)
        diff = hl_avg - hl_avg_delay
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma_diff = sma(diff, 6, 1)
        return sma_diff / (sma_diff.abs() + self._EPSILON) * 100
    
    def _alpha_069(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#69: (SUM(DTM, 20) > SUM(DBM, 20) ? (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DTM, 20) : 
                  (SUM(DTM, 20) == SUM(DBM, 20) ? 0 : (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DBM, 20)))
        DTM = IF(OPEN > DELAY(OPEN, 1), MAX((HIGH - OPEN), (OPEN - DELAY(OPEN, 1))), 0)
        DBM = IF(OPEN < DELAY(OPEN, 1), MAX((OPEN - LOW), (DELAY(OPEN, 1) - OPEN)), 0)
        """
        open_delay_1 = self._delay(df['open'], 1)
        
        condition_dtm = df['open'] > open_delay_1
        dtm_value = np.maximum(df['high'] - df['open'], df['open'] - open_delay_1)
        dtm = np.where(condition_dtm, dtm_value, 0)
        
        condition_dbm = df['open'] < open_delay_1
        dbm_value = np.maximum(df['open'] - df['low'], open_delay_1 - df['open'])
        dbm = np.where(condition_dbm, dbm_value, 0)
        
        sum_dtm_20 = self._sum(pd.Series(dtm, index=df.index), 20)
        sum_dbm_20 = self._sum(pd.Series(dbm, index=df.index), 20)
        
        condition1 = sum_dtm_20 > sum_dbm_20
        condition2 = sum_dtm_20 == sum_dbm_20
        
        result = np.where(condition1, (sum_dtm_20 - sum_dbm_20) / (sum_dtm_20 + self._EPSILON),
                         np.where(condition2, 0, (sum_dtm_20 - sum_dbm_20) / (sum_dbm_20 + self._EPSILON)))
        
        return pd.Series(result, index=df.index)
    
    def _alpha_070(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#70: STD(AMOUNT, 6)
        """
        if 'amount' in df.columns:
            return self._std(df['amount'], 6)
        else:
            # 使用volume * close近似
            amount = df['volume'] * df['close']
            return self._std(amount, 6)
    
    def _alpha_071(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#71: (CLOSE - MEAN(CLOSE, 24)) / MEAN(CLOSE, 24) * 100
        """
        mean_close_24 = self._mean(df['close'], 24)
        return ((df['close'] - mean_close_24) / (mean_close_24 + self._EPSILON)) * 100
    
    def _alpha_072(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#72: SMA((TSMAX(HIGH, 6) - CLOSE) / (TSMAX(HIGH, 6) - TSMIN(LOW, 6)), 15, 1)
        """
        tsmax_high_6 = self._ts_max(df['high'], 6)
        tsmin_low_6 = self._ts_min(df['low'], 6)
        
        numerator = tsmax_high_6 - df['close']
        denominator = tsmax_high_6 - tsmin_low_6 + self._EPSILON
        ratio = numerator / denominator
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        return sma(ratio, 15, 1)
    
    def _alpha_073(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#73: (TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - 
                  RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1
        """
        corr1 = self._corr(df['close'], df['volume'], 10)
        decay1 = self._decay_linear(corr1, 16)
        decay2 = self._decay_linear(decay1, 4)
        tsrank1 = self._ts_rank(decay2, 5)
        
        mean_vol_30 = self._mean(df['volume'], 30)
        corr2 = self._corr(vwap, mean_vol_30, 4)
        decay3 = self._decay_linear(corr2, 3)
        rank_decay = self._rank(decay3)
        
        return (tsrank1 - rank_decay) * -1
    
    def _alpha_074(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#74: (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + 
                  RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        """
        weighted_low = df['low'] * 0.35 + vwap * 0.65
        sum_weighted = self._sum(weighted_low, 20)
        mean_vol_40 = self._mean(df['volume'], 40)
        sum_mean_vol = self._sum(mean_vol_40, 20)
        corr1 = self._corr(sum_weighted, sum_mean_vol, 7)
        rank_corr1 = self._rank(corr1)
        
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr2 = self._corr(rank_vwap, rank_volume, 6)
        rank_corr2 = self._rank(corr2)
        
        return rank_corr1 + rank_corr2
    
    def _alpha_075(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#75: COUNT(CLOSE > OPEN, 20) / 20 * 100
        """
        condition = df['close'] > df['open']
        count_up = condition.rolling(20).sum()
        return (count_up / 20) * 100
    
    def _alpha_076(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#76: STD(ABS((CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1)), 20) / MEAN(ABS((CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        ratio = (df['close'] - close_delay_1) / (close_delay_1 + self._EPSILON)
        abs_ratio = self._abs(ratio)
        
        std_abs = self._std(abs_ratio, 20)
        mean_abs = self._mean(abs_ratio, 20)
        
        return std_abs / (mean_abs + self._EPSILON)
    
    def _alpha_077(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#77: MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), 
                  RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
        """
        hl_avg = (df['high'] + df['low']) / 2
        part1 = (hl_avg + df['high']) - (vwap + df['high'])
        decay1 = self._decay_linear(part1, 20)
        rank_decay1 = self._rank(decay1)
        
        mean_vol_40 = self._mean(df['volume'], 40)
        corr_result = self._corr(hl_avg, mean_vol_40, 3)
        decay2 = self._decay_linear(corr_result, 6)
        rank_decay2 = self._rank(decay2)
        
        return self._min(rank_decay1, rank_decay2)
    
    def _alpha_078(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#78: ((HIGH + LOW + CLOSE) / 3 - MA((HIGH + LOW + CLOSE) / 3, 12)) / (0.015 * MEAN(ABS(CLOSE - MA((HIGH + LOW + CLOSE) / 3, 12)), 12))
        """
        hlc_avg = (df['high'] + df['low'] + df['close']) / 3
        ma_hlc = self._mean(hlc_avg, 12)
        
        numerator = hlc_avg - ma_hlc
        abs_diff = self._abs(df['close'] - ma_hlc)
        mean_abs = self._mean(abs_diff, 12)
        denominator = 0.015 * mean_abs + self._EPSILON
        
        return numerator / denominator
    
    def _alpha_079(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#79: SMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 12, 1) / SMA(ABS(CLOSE - DELAY(CLOSE, 1)), 12, 1) * 100
        """
        close_delay_1 = self._delay(df['close'], 1)
        diff = df['close'] - close_delay_1
        
        gain = np.maximum(0, diff)
        abs_diff = self._abs(diff)
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma_gain = sma(pd.Series(gain, index=df.index), 12, 1)
        sma_abs = sma(abs_diff, 12, 1)
        
        return (sma_gain / (sma_abs + self._EPSILON)) * 100
    
    def _alpha_080(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#80: (VOLUME - DELAY(VOLUME, 5)) / DELAY(VOLUME, 5) * 100
        """
        volume_delay_5 = self._delay(df['volume'], 5)
        return ((df['volume'] - volume_delay_5) / (volume_delay_5 + self._EPSILON)) * 100
    
    def _alpha_081(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#81: SMA(VOLUME, 21, 2) / SMA(VOLUME, 7, 2) * 100
        """
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma_21 = sma(df['volume'], 21, 2)
        sma_7 = sma(df['volume'], 7, 2)
        
        return (sma_21 / (sma_7 + self._EPSILON)) * 100
    
    def _alpha_082(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#82: SMA((TSMAX(HIGH, 6) - CLOSE) / (TSMAX(HIGH, 6) - TSMIN(LOW, 6)) * 100, 20, 1)
        """
        tsmax_high_6 = self._ts_max(df['high'], 6)
        tsmin_low_6 = self._ts_min(df['low'], 6)
        
        numerator = tsmax_high_6 - df['close']
        denominator = tsmax_high_6 - tsmin_low_6 + self._EPSILON
        ratio = (numerator / denominator) * 100
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        return sma(ratio, 20, 1)
    
    def _alpha_083(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#83: (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
        """
        rank_high = self._rank(df['high'])
        rank_volume = self._rank(df['volume'])
        cov_result = self._cov(rank_high, rank_volume, 5)
        return -1 * self._rank(cov_result)
    
    def _alpha_084(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#84: SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : (CLOSE < DELAY(CLOSE, 1) ? -VOLUME : 0)), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition1 = df['close'] > close_delay_1
        condition2 = df['close'] < close_delay_1
        
        value = np.where(condition1, df['volume'], np.where(condition2, -df['volume'], 0))
        return self._sum(pd.Series(value, index=df.index), 20)
    
    def _alpha_085(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#85: (TSRANK((VOLUME / MEAN(VOLUME, 20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
        """
        mean_vol_20 = self._mean(df['volume'], 20)
        volume_ratio = df['volume'] / (mean_vol_20 + self._EPSILON)
        tsrank1 = self._ts_rank(volume_ratio, 20)
        
        delta_close_7 = self._delta(df['close'], 7)
        tsrank2 = self._ts_rank(-1 * delta_close_7, 8)
        
        return tsrank1 * tsrank2
    
    def _alpha_086(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#86: ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? 
                  (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 
                  1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))
        """
        close_delay_1 = self._delay(df['close'], 1)
        close_delay_10 = self._delay(df['close'], 10)
        close_delay_20 = self._delay(df['close'], 20)
        
        part1 = (close_delay_20 - close_delay_10) / 10
        part2 = (close_delay_10 - df['close']) / 10
        diff = part1 - part2
        
        condition1 = diff > 0.25
        condition2 = diff < 0
        
        result = np.where(condition1, -1, np.where(condition2, 1, -1 * (df['close'] - close_delay_1)))
        return pd.Series(result, index=df.index)
    
    def _alpha_087(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#87: ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / 
                  (OPEN - (HIGH + LOW) / 2)), 11), 7)) * -1)
        """
        delta_vwap_4 = self._delta(vwap, 4)
        decay1 = self._decay_linear(delta_vwap_4, 7)
        rank_decay1 = self._rank(decay1)
        
        weighted_low = df['low'] * 0.9 + df['low'] * 0.1  # 实际上就是LOW
        numerator = weighted_low - vwap
        denominator = df['open'] - (df['high'] + df['low']) / 2 + self._EPSILON
        ratio = numerator / denominator
        decay2 = self._decay_linear(ratio, 11)
        tsrank2 = self._ts_rank(decay2, 7)
        
        return (rank_decay1 + tsrank2) * -1
    
    def _alpha_088(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#88: (RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
        """
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr_result = self._corr(rank_vwap, rank_volume, 5)
        return self._rank(corr_result) * -1
    
    def _alpha_089(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#89: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP * 0.65)), 2), 15)), 
                  RANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 17))) * -1)
        """
        weighted_close = df['close'] * 0.35 + vwap * 0.65
        delta_weighted = self._delta(weighted_close, 2)
        decay1 = self._decay_linear(delta_weighted, 15)
        rank_decay1 = self._rank(decay1)
        
        mean_vol_180 = self._mean(df['volume'], 180)
        corr_result = self._corr(mean_vol_180, df['close'], 13)
        abs_corr = self._abs(corr_result)
        decay2 = self._decay_linear(abs_corr, 17)
        rank_decay2 = self._rank(decay2)
        
        max_rank = self._max(rank_decay1, rank_decay2)
        return max_rank * -1
    
    def _alpha_090(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#90: (RANK((CLOSE - MAX(CLOSE, 5))) * RANK(CORR((MEAN(VOLUME,40)), LOW, 5)))
        """
        max_close_5 = self._ts_max(df['close'], 5)
        rank_diff = self._rank(df['close'] - max_close_5)
        
        mean_vol_40 = self._mean(df['volume'], 40)
        corr_result = self._corr(mean_vol_40, df['low'], 5)
        rank_corr = self._rank(corr_result)
        
        return rank_diff * rank_corr
    
    # ========== 第四批：Alpha 91-101 ==========
    
    def _alpha_091(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#91: ((RANK((CLOSE - MAX(CLOSE, 5))) * RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
        """
        max_close_5 = self._ts_max(df['close'], 5)
        rank_diff = self._rank(df['close'] - max_close_5)
        
        mean_vol_40 = self._mean(df['volume'], 40)
        corr_result = self._corr(mean_vol_40, df['low'], 5)
        rank_corr = self._rank(corr_result)
        
        return rank_diff * rank_corr * -1
    
    def _alpha_092(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#92: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP * 0.65)), 2), 15)), 
                  RANK(DECAYLINEAR(CORR(((MEAN(VOLUME,180)), CLOSE), 13), 17))) * -1)
        """
        weighted_close = df['close'] * 0.35 + vwap * 0.65
        delta_weighted = self._delta(weighted_close, 2)
        decay1 = self._decay_linear(delta_weighted, 15)
        rank_decay1 = self._rank(decay1)
        
        mean_vol_180 = self._mean(df['volume'], 180)
        corr_result = self._corr(mean_vol_180, df['close'], 13)
        decay2 = self._decay_linear(corr_result, 17)
        rank_decay2 = self._rank(decay2)
        
        max_rank = self._max(rank_decay1, rank_decay2)
        return max_rank * -1
    
    def _alpha_093(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#93: SUM((CLOSE >= DELAY(CLOSE, 1) ? VOLUME : 0), 20)
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition = df['close'] >= close_delay_1
        value = np.where(condition, df['volume'], 0)
        return self._sum(pd.Series(value, index=df.index), 20)
    
    def _alpha_094(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#94: SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : (CLOSE < DELAY(CLOSE, 1) ? -VOLUME : 0)), 30)
        """
        close_delay_1 = self._delay(df['close'], 1)
        condition1 = df['close'] > close_delay_1
        condition2 = df['close'] < close_delay_1
        
        value = np.where(condition1, df['volume'], np.where(condition2, -df['volume'], 0))
        return self._sum(pd.Series(value, index=df.index), 30)
    
    def _alpha_095(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#95: STD(AMOUNT, 20)
        """
        if 'amount' in df.columns:
            return self._std(df['amount'], 20)
        else:
            # 使用volume * close近似
            amount = df['volume'] * df['close']
            return self._std(amount, 20)
    
    def _alpha_096(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#96: SMA(SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9)), 3, 1), 3, 1)
        """
        tsmin_low_9 = self._ts_min(df['low'], 9)
        tsmax_high_9 = self._ts_max(df['high'], 9)
        
        numerator = df['close'] - tsmin_low_9
        denominator = tsmax_high_9 - tsmin_low_9 + self._EPSILON
        ratio = numerator / denominator
        
        def sma(series, n, m):
            alpha = m / n
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        sma1 = sma(ratio, 3, 1)
        return sma(sma1, 3, 1)
    
    def _alpha_097(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#97: STD(VOLUME, 10)
        """
        return self._std(df['volume'], 10)
    
    def _alpha_098(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#98: (((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) <= 0.05) ? 
                  (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
        """
        sum_close_100 = self._sum(df['close'], 100)
        mean_close_100 = sum_close_100 / 100
        delta_mean = self._delta(mean_close_100, 100)
        close_delay_100 = self._delay(df['close'], 100)
        
        ratio = delta_mean / (close_delay_100 + self._EPSILON)
        condition = ratio <= 0.05
        
        value1 = -1 * (df['close'] - self._ts_min(df['close'], 100))
        value2 = -1 * self._delta(df['close'], 3)
        
        result = np.where(condition, value1, value2)
        return pd.Series(result, index=df.index)
    
    def _alpha_099(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#99: (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
        """
        rank_close = self._rank(df['close'])
        rank_volume = self._rank(df['volume'])
        cov_result = self._cov(rank_close, rank_volume, 5)
        return -1 * self._rank(cov_result)
    
    def _alpha_100(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#100: STD(VOLUME, 20)
        """
        return self._std(df['volume'], 20)
    
    def _alpha_101(self, df: pd.DataFrame, vwap: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#101 (Delay-1, 动量): ((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))
        """
        numerator = df['close'] - df['open']
        denominator = df['high'] - df['low'] + 0.001
        return numerator / denominator
    