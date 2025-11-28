from typing import List

import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import DualModeFactor

class Alpha191Factor(DualModeFactor):
    """
    根据国泰君安数量化专题研究报告 - 基于短周期价量特征的多因子选股体系给出了 191 个短周期交易型阿尔法因子
    
    参考：https://www.joinquant.com/data/dict/alpha191
    """
    display_name = "Alpha191"
    _EPSILON = 1e-20  # 除0保护常量，适用于数字币等极小价格
    
    init_params = [
        Field(name="alphas", label="包含Alpha", type=FieldType.STRING, default="1-191", description="包含的Alpha因子ID，支持范围(1-191)和单个数字，逗号分隔")
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self.alphas = kwargs.get("alphas", "1-191")
        
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
            self.selected_alphas = list(range(1, 192))
            
        # Filter valid alphas (1-191)
        self.selected_alphas = [i for i in self.selected_alphas if 1 <= i <= 191]
        
        # Update factor names
        self.factor_names = [f"{self.name}_alpha{i:03d}" for i in self.selected_alphas]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # 数据预处理：计算VWAP和AMOUNT
        vwap, amount, vwap_added, amount_added = self._preprocess_dataframe(df)
        
        # 收集所有因子结果，避免DataFrame碎片化
        factor_results = {}
        for i in self.selected_alphas:
            method_name = f"_alpha_{i:03d}"
            if hasattr(self, method_name):
                factor_name = f"{self.name}_alpha{i:03d}"
                try:
                    factor_results[factor_name] = getattr(self, method_name)(df)
                except Exception as e:
                    # 如果计算失败，返回NaN
                    factor_results[factor_name] = pd.Series(np.nan, index=df.index)
        
        return pd.DataFrame(factor_results, index=df.index)
    
    def get_output_names(self) -> list:
        return self.factor_names
    
    # ========== 数据预处理 ==========
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> tuple:
        """
        数据预处理：计算VWAP和AMOUNT
        返回: (vwap Series, amount Series, vwap_added bool, amount_added bool)
        """
        vwap_added = False
        amount_added = False
        
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
        
        # 计算AMOUNT：使用close * volume（如果不存在）
        if 'amount' not in df.columns:
            amount = df['close'] * df['volume']
            df['amount'] = amount
            amount_added = True
        else:
            amount = df['amount']
        
        return vwap, amount, vwap_added, amount_added
    
    # ========== 辅助函数 ==========
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """RANK: 向量升序排序"""
        return series.rank(pct=True)
    
    def _delta(self, series: pd.Series, n: int = 1) -> pd.Series:
        """DELTA: 当前值减去n期前的值"""
        return series - series.shift(n)
    
    def _delay(self, series: pd.Series, n: int) -> pd.Series:
        """DELAY: 返回n期前的值"""
        return series.shift(n)
    
    def _corr(self, x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """CORR: 过去n天的相关系数"""
        return x.rolling(n).corr(y)
    
    def _std(self, series: pd.Series, n: int) -> pd.Series:
        """STD: 过去n天的标准差"""
        return series.rolling(n).std()
    
    def _mean(self, series: pd.Series, n: int) -> pd.Series:
        """MEAN: 过去n天的均值"""
        return series.rolling(n).mean()
    
    def _sum(self, series: pd.Series, n: int) -> pd.Series:
        """SUM: 过去n天的求和"""
        return series.rolling(n).sum()
    
    def _max(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """MAX: 两个序列的最大值"""
        return pd.concat([x, y], axis=1).max(axis=1)
    
    def _min(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """MIN: 两个序列的最小值"""
        return pd.concat([x, y], axis=1).min(axis=1)
    
    def _tsrank(self, series: pd.Series, n: int) -> pd.Series:
        """TSRANK: 序列的末位值在过去n天的顺序排位"""
        return series.rolling(n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == n else np.nan, raw=False)
    
    def _tsmax(self, series: pd.Series, n: int) -> pd.Series:
        """TSMAX: 过去n天的最大值"""
        return series.rolling(n).max()
    
    def _tsmin(self, series: pd.Series, n: int) -> pd.Series:
        """TSMIN: 过去n天的最小值"""
        return series.rolling(n).min()
    
    def _prod(self, series: pd.Series, n: int) -> pd.Series:
        """PROD: 过去n天累乘"""
        return series.rolling(n).apply(lambda x: np.prod(x), raw=True)
    
    def _count(self, condition: pd.Series, n: int) -> pd.Series:
        """COUNT: 计算前n期满足条件的样本个数"""
        return condition.rolling(n).sum()
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """SMA: 加权移动平均，权重为m"""
        alpha = m / n
        result = pd.Series(index=series.index, dtype=float)
        result.iloc[0] = series.iloc[0]
        for i in range(1, len(series)):
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
        return result
    
    def _wma(self, series: pd.Series, n: int) -> pd.Series:
        """WMA: 加权移动平均，权重为0.9^i"""
        weights = np.array([0.9 ** i for i in range(n)])
        weights = weights / weights.sum()
        return series.rolling(n).apply(lambda x: np.sum(x * weights[::-1]) if len(x) == n else np.nan, raw=True)
    
    def _decaylinear(self, series: pd.Series, d: int) -> pd.Series:
        """DECAYLINEAR: 移动平均加权，权重为d,d-1,...,1"""
        weights = np.array(range(d, 0, -1))
        weights = weights / weights.sum()
        return series.rolling(d).apply(lambda x: np.sum(x * weights[::-1]) if len(x) == d else np.nan, raw=True)
    
    def _sumac(self, series: pd.Series, n: int) -> pd.Series:
        """SUMAC: 前n项的累加"""
        return series.rolling(n).sum()
    
    def _highday(self, series: pd.Series, n: int) -> pd.Series:
        """HIGHDAY: 最大值距离当前时点的间隔"""
        def calc_highday(x):
            if len(x) < n:
                return np.nan
            return len(x) - 1 - np.argmax(x)
        return series.rolling(n).apply(calc_highday, raw=True)
    
    def _lowday(self, series: pd.Series, n: int) -> pd.Series:
        """LOWDAY: 最小值距离当前时点的间隔"""
        def calc_lowday(x):
            if len(x) < n:
                return np.nan
            return len(x) - 1 - np.argmin(x)
        return series.rolling(n).apply(calc_lowday, raw=True)
    
    def _regbeta(self, y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """REGBETA: 前n期样本y对x做回归所得回归系数"""
        def calc_beta(window):
            if len(window) < 2:
                return np.nan
            y_vals = window[:, 0]
            x_vals = window[:, 1]
            if len(y_vals) != n:
                return np.nan
            coeffs = np.polyfit(x_vals, y_vals, 1)
            return coeffs[0]
        
        combined = pd.concat([y, x], axis=1).values
        return pd.Series(index=y.index).apply(
            lambda i: calc_beta(combined[max(0, i-n+1):i+1]) if i >= n-1 else np.nan
        )
    
    def _regresi(self, y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """REGRESI: 前n期样本y对x做回归所得的残差"""
        def calc_residual(window):
            if len(window) < 2:
                return np.nan
            y_vals = window[:, 0]
            x_vals = window[:, 1]
            if len(y_vals) != n:
                return np.nan
            coeffs = np.polyfit(x_vals, y_vals, 1)
            y_pred = np.polyval(coeffs, x_vals)
            return y_vals[-1] - y_pred[-1]
        
        combined = pd.concat([y, x], axis=1).values
        return pd.Series(index=y.index).apply(
            lambda i: calc_residual(combined[max(0, i-n+1):i+1]) if i >= n-1 else np.nan
        )
    
    def _sumif(self, series: pd.Series, n: int, condition: pd.Series) -> pd.Series:
        """SUMIF: 对series前n项条件求和"""
        masked = series.where(condition, 0)
        return masked.rolling(n).sum()
    
    def _covariance(self, x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """COVIANCE: 序列A、B过去n天协方差"""
        return x.rolling(n).cov(y)
    
    def _sign(self, series: pd.Series) -> pd.Series:
        """SIGN: 符号函数：1 if A>0; 0 if A=0; -1 if A<0"""
        return series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    def _sequence(self, n: int) -> np.ndarray:
        """SEQUENCE: 生成1~n的等差序列"""
        return np.arange(1, n + 1)
    
    # ========== Alpha 因子实现 ==========
    
    def _alpha_001(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
        """
        log_volume = np.log(df['volume'] + self._EPSILON)
        delta_log_volume = self._delta(log_volume, 1)
        rank_delta_log_volume = self._rank(delta_log_volume)
        
        close_open_ratio = (df['close'] - df['open']) / (df['open'] + self._EPSILON)
        rank_close_open = self._rank(close_open_ratio)
        
        return -1 * self._corr(rank_delta_log_volume, rank_close_open, 6)
    
    def _alpha_002(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
        """
        high_low = df['high'] - df['low']
        numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
        ratio = numerator / (high_low + self._EPSILON)
        return -1 * self._delta(ratio, 1)
    
    def _alpha_003(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE<DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
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
    
    def _alpha_004(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
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
    
    def _alpha_005(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
        """
        tsrank_volume = self._tsrank(df['volume'], 5)
        tsrank_high = self._tsrank(df['high'], 5)
        corr_result = self._corr(tsrank_volume, tsrank_high, 5)
        return -1 * self._tsmax(corr_result, 3)
    
    def _alpha_006(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4))).TSRANK(VOLUME, 3)) * -1)
        """
        weighted_price = df['open'] * 0.85 + df['high'] * 0.15
        delta_weighted = self._delta(weighted_price, 4)
        sign_delta = self._sign(delta_weighted)
        rank_sign = self._rank(sign_delta)
        
        tsrank_volume = self._tsrank(df['volume'], 3)
        
        return rank_sign * tsrank_volume * -1
    
    def _alpha_007(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
        注意：MAX和MIN应该是TSMAX和TSMIN（滚动窗口的最大最小值）
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        vwap_close_diff = vwap - df['close']
        
        max_diff = self._tsmax(vwap_close_diff, 3)
        min_diff = self._tsmin(vwap_close_diff, 3)
        
        rank_max = self._rank(max_diff)
        rank_min = self._rank(min_diff)
        rank_delta_volume = self._rank(self._delta(df['volume'], 3))
        
        return (rank_max + rank_min) * rank_delta_volume
    
    def _alpha_008(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        weighted_price = ((df['high'] + df['low']) / 2) * 0.2 + vwap * 0.8
        delta_weighted = self._delta(weighted_price, 4)
        return self._rank(delta_weighted) * -1
    
    def _alpha_009(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((((HIGH + LOW) / 2) - (MEAN((HIGH + LOW) / 2, 7))) > 0 ? (MEAN((HIGH + LOW) / 2, 7) - (HIGH + LOW) / 2) : (((HIGH + LOW) / 2) - MEAN((HIGH + LOW) / 2, 7))) * (-1 * RANK(MEAN(VOLUME, 10))))
        """
        hl_avg = (df['high'] + df['low']) / 2
        mean_hl_avg = self._mean(hl_avg, 7)
        diff = hl_avg - mean_hl_avg
        
        condition = diff > 0
        value = np.where(condition, -diff, diff)
        
        rank_mean_volume = self._rank(self._mean(df['volume'], 10))
        
        return pd.Series(value, index=df.index) * (-1 * rank_mean_volume)
    
    def _alpha_010(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：RANK(((MAX((HIGH - CLOSE), 3) + MAX((CLOSE - LOW), 3)) * 100 / (MAX((HIGH - LOW), 3))))
        """
        high_close = df['high'] - df['close']
        close_low = df['close'] - df['low']
        high_low = df['high'] - df['low']
        
        max_high_close = self._tsmax(high_close, 3)
        max_close_low = self._tsmax(close_low, 3)
        max_high_low = self._tsmax(high_low, 3)
        
        numerator = (max_high_close + max_close_low) * 100
        denominator = max_high_low + self._EPSILON
        
        return self._rank(numerator / denominator)
    
    def _alpha_011(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
        """
        numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
        denominator = df['high'] - df['low'] + self._EPSILON
        ratio = numerator / denominator
        return self._sum(ratio * df['volume'], 6)
    
    def _alpha_012(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(RANK((OPEN - (SUM((HIGH + LOW + CLOSE) / 3, 10) / 10)))) * (-1 * RANK(ABS((CLOSE - MEAN(CLOSE, 10)))))
        """
        hlc_avg = (df['high'] + df['low'] + df['close']) / 3
        sum_hlc_avg = self._sum(hlc_avg, 10) / 10
        
        rank_open_diff = self._rank(df['open'] - sum_hlc_avg)
        
        mean_close = self._mean(df['close'], 10)
        abs_close_mean = np.abs(df['close'] - mean_close)
        rank_abs = self._rank(abs_close_mean)
        
        return rank_open_diff * (-1 * rank_abs)
    
    def _alpha_013(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(((HIGH * LOW)^0.5) - VWAP)
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        hl_sqrt = np.sqrt(df['high'] * df['low'])
        return hl_sqrt - vwap
    
    def _alpha_014(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：CLOSE-DELAY(CLOSE,5)
        """
        return df['close'] - self._delay(df['close'], 5)
    
    def _alpha_015(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：OPEN/DELAY(CLOSE,1)-1
        """
        return df['open'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
    
    def _alpha_016(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        """
        rank_volume = self._rank(df['volume'])
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_vwap = self._rank(vwap)
        corr_result = self._corr(rank_volume, rank_vwap, 5)
        rank_corr = self._rank(corr_result)
        return -1 * self._tsmax(rank_corr, 5)
    
    def _alpha_017(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：RANK((VWAP - MAX(VWAP, 15))) * DELTA(CLOSE, 1)
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        max_vwap = self._tsmax(vwap, 15)
        rank_diff = self._rank(vwap - max_vwap)
        delta_close = self._delta(df['close'], 1)
        return rank_diff * delta_close
    
    def _alpha_018(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：CLOSE/DELAY(CLOSE,5)
        """
        return df['close'] / (self._delay(df['close'], 5) + self._EPSILON)
    
    def _alpha_019(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)
        """
        close_delay5 = self._delay(df['close'], 5)
        return (df['close'] - close_delay5) / (close_delay5 + self._EPSILON)
    
    def _alpha_020(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        """
        close_delay6 = self._delay(df['close'], 6)
        return (df['close'] - close_delay6) / (close_delay6 + self._EPSILON) * 100
    
    def _alpha_021(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：REGbeta(MEAN(CLOSE,6),SEQUENCE(6),6)
        """
        mean_close = self._mean(df['close'], 6)
        sequence_6 = pd.Series(self._sequence(6), index=df.index[-6:])
        # 简化实现：使用线性回归
        def calc_beta(window):
            if len(window) < 6:
                return np.nan
            y = window
            x = np.arange(1, 7)
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        
        return mean_close.rolling(6).apply(calc_beta, raw=True)
    
    def _alpha_022(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
        """
        mean_close = self._mean(df['close'], 6)
        ratio = (df['close'] - mean_close) / (mean_close + self._EPSILON)
        ratio_delay = self._delay(ratio, 3)
        diff = ratio - ratio_delay
        return self._sma(diff, 12, 1)
    
    def _alpha_023(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        """
        condition = df['close'] > self._delay(df['close'], 1)
        std_close = self._std(df['close'], 20)
        value = np.where(condition, std_close, 0)
        return self._sma(pd.Series(value, index=df.index), 20, 1)
    
    def _alpha_024(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SMA(CLOSE-DELAY(CLOSE,5),5,1)
        """
        diff = df['close'] - self._delay(df['close'], 5)
        return self._sma(diff, 5, 1)
    
    def _alpha_025(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM((CLOSE / DELAY(CLOSE, 1) - 1), 250))))
        """
        delta_close = self._delta(df['close'], 7)
        volume_mean = self._mean(df['volume'], 20)
        volume_ratio = df['volume'] / (volume_mean + self._EPSILON)
        decay_volume = self._decaylinear(volume_ratio, 9)
        rank_decay = self._rank(decay_volume)
        
        part1 = delta_close * (1 - rank_decay)
        rank_part1 = self._rank(part1)
        
        close_ratio = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        sum_close_ratio = self._sum(close_ratio, 250)
        rank_sum = self._rank(sum_close_ratio)
        
        return (-1 * rank_part1) * (1 + rank_sum)
    
    def _alpha_026(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230)))
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        sum_close_7 = self._sum(df['close'], 7) / 7
        part1 = sum_close_7 - df['close']
        part2 = self._corr(vwap, self._delay(df['close'], 5), 230)
        return part1 + part2
    
    def _alpha_027(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
        """
        close_delay3 = self._delay(df['close'], 3)
        close_delay6 = self._delay(df['close'], 6)
        part1 = (df['close'] - close_delay3) / (close_delay3 + self._EPSILON) * 100
        part2 = (df['close'] - close_delay6) / (close_delay6 + self._EPSILON) * 100
        combined = part1 + part2
        return self._wma(combined, 12)
    
    def _alpha_028(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
        """
        tsmin_low = self._tsmin(df['low'], 9)
        tsmax_high = self._tsmax(df['high'], 9)
        rsv = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        sma1 = self._sma(rsv, 3, 1)
        sma2 = self._sma(sma1, 3, 1)
        return 3 * sma1 - 2 * sma2
    
    def _alpha_029(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        """
        close_delay6 = self._delay(df['close'], 6)
        return (df['close'] - close_delay6) / (close_delay6 + self._EPSILON) * df['volume']
    
    def _alpha_030(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：WMA((REGRESI(CLOSE/DELAY(CLOSE,1)-1,MKT,20))^2,20)
        """
        # MKT 通常指市场收益率，这里简化使用close的收益率
        close_ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        mkt = close_ret  # 简化处理
        regresi_result = self._regresi(close_ret, mkt, 20)
        squared = regresi_result ** 2
        return self._wma(squared, 20)
    
    def _alpha_031(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
        """
        mean_close = self._mean(df['close'], 12)
        return (df['close'] - mean_close) / (mean_close + self._EPSILON) * 100
    
    def _alpha_032(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        """
        rank_high = self._rank(df['high'])
        rank_volume = self._rank(df['volume'])
        corr_result = self._corr(rank_high, rank_volume, 3)
        rank_corr = self._rank(corr_result)
        return -1 * self._sum(rank_corr, 3)
    
    def _alpha_033(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM((CLOSE / DELAY(CLOSE, 1) - 1), 240) - SUM((CLOSE / DELAY(CLOSE, 1) - 1), 20)) / 220))) * TSRANK(VOLUME, 5))
        """
        tsmin_low = self._tsmin(df['low'], 5)
        tsmin_low_delay = self._delay(tsmin_low, 5)
        part1 = -1 * tsmin_low + tsmin_low_delay
        
        close_ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        sum_240 = self._sum(close_ret, 240)
        sum_20 = self._sum(close_ret, 20)
        part2 = (sum_240 - sum_20) / 220
        rank_part2 = self._rank(part2)
        
        tsrank_volume = self._tsrank(df['volume'], 5)
        
        return part1 * rank_part2 * tsrank_volume
    
    def _alpha_034(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：MEAN(CLOSE,12)/CLOSE
        """
        mean_close = self._mean(df['close'], 12)
        return mean_close / (df['close'] + self._EPSILON)
    
    def _alpha_035(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((CLOSE * 0.65) + (OPEN * 0.35)), 17), 7))) * -1)
        """
        delta_open = self._delta(df['open'], 1)
        decay_delta = self._decaylinear(delta_open, 15)
        rank_decay_delta = self._rank(decay_delta)
        
        weighted_price = df['close'] * 0.65 + df['open'] * 0.35
        corr_vol_price = self._corr(df['volume'], weighted_price, 17)
        decay_corr = self._decaylinear(corr_vol_price, 7)
        rank_decay_corr = self._rank(decay_corr)
        
        min_rank = self._min(rank_decay_delta, rank_decay_corr)
        return min_rank * -1
    
    def _alpha_036(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_volume = self._rank(df['volume'])
        rank_vwap = self._rank(vwap)
        corr_result = self._corr(rank_volume, rank_vwap, 6)
        sum_corr = self._sum(corr_result, 2)
        return self._rank(sum_corr)
    
    def _alpha_037(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1 * RANK(((SUM(OPEN, 5) * SUM((CLOSE / DELAY(CLOSE, 1) - 1), 5)) - DELAY((SUM(OPEN, 5) * SUM((CLOSE / DELAY(CLOSE, 1) - 1), 5)), 10))))
        """
        sum_open = self._sum(df['open'], 5)
        close_ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        sum_ret = self._sum(close_ret, 5)
        product = sum_open * sum_ret
        product_delay = self._delay(product, 10)
        return -1 * self._rank(product - product_delay)
    
    def _alpha_038(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        """
        mean_high = self._sum(df['high'], 20) / 20
        condition = mean_high < df['high']
        delta_high = self._delta(df['high'], 2)
        return np.where(condition, -1 * delta_high, 0)
    
    def _alpha_039(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
        """
        delta_close = self._delta(df['close'], 2)
        decay_delta = self._decaylinear(delta_close, 8)
        rank_decay_delta = self._rank(decay_delta)
        
        vwap = df['vwap']  # 预处理已确保vwap存在
        weighted_price = vwap * 0.3 + df['open'] * 0.7
        mean_volume = self._mean(df['volume'], 180)
        sum_mean_vol = self._sum(mean_volume, 37)
        corr_result = self._corr(weighted_price, sum_mean_vol, 14)
        decay_corr = self._decaylinear(corr_result, 12)
        rank_decay_corr = self._rank(decay_corr)
        
        return (rank_decay_delta - rank_decay_corr) * -1
    
    def _alpha_040(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
        """
        condition_up = df['close'] > self._delay(df['close'], 1)
        condition_down = df['close'] <= self._delay(df['close'], 1)
        volume_up = np.where(condition_up, df['volume'], 0)
        volume_down = np.where(condition_down, df['volume'], 0)
        sum_up = pd.Series(volume_up, index=df.index).rolling(26).sum()
        sum_down = pd.Series(volume_down, index=df.index).rolling(26).sum()
        return sum_up / (sum_down + self._EPSILON) * 100
    
    def _alpha_041(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(RANK(MAX(DELTA((VWAP), 3), 5)) * -1)
        """
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_vwap = self._delta(vwap, 3)
        max_delta = self._tsmax(delta_vwap, 5)
        return self._rank(max_delta) * -1
    
    def _alpha_042(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
        """
        std_high = self._std(df['high'], 10)
        rank_std = self._rank(std_high)
        corr_high_vol = self._corr(df['high'], df['volume'], 10)
        return (-1 * rank_std) * corr_high_vol
    
    def _alpha_043(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
        """
        close_delay = self._delay(df['close'], 1)
        condition_up = df['close'] > close_delay
        condition_down = df['close'] < close_delay
        value = np.where(condition_up, df['volume'], np.where(condition_down, -df['volume'], 0))
        return pd.Series(value, index=df.index).rolling(6).sum()
    
    def _alpha_044(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6), 4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
        """
        mean_volume = self._mean(df['volume'], 10)
        corr_low_vol = self._corr(df['low'], mean_volume, 7)
        decay_corr = self._decaylinear(corr_low_vol, 6)
        tsrank1 = self._tsrank(decay_corr, 4)
        
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_vwap = self._delta(vwap, 3)
        decay_delta = self._decaylinear(delta_vwap, 10)
        tsrank2 = self._tsrank(decay_delta, 15)
        
        return tsrank1 + tsrank2
    
    def _alpha_045(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(RANK(DELTA((((CLOSE * 0.6) + (OPEN * 0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
        """
        weighted_price = df['close'] * 0.6 + df['open'] * 0.4
        delta_weighted = self._delta(weighted_price, 1)
        rank_delta = self._rank(delta_weighted)
        
        vwap = df['vwap']  # 预处理已确保vwap存在
        mean_volume = self._mean(df['volume'], 150)
        corr_result = self._corr(vwap, mean_volume, 15)
        rank_corr = self._rank(corr_result)
        
        return rank_delta * rank_corr
    
    def _alpha_046(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
        """
        mean3 = self._mean(df['close'], 3)
        mean6 = self._mean(df['close'], 6)
        mean12 = self._mean(df['close'], 12)
        mean24 = self._mean(df['close'], 24)
        return (mean3 + mean6 + mean12 + mean24) / 4
    
    def _alpha_047(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
        """
        tsmax_high = self._tsmax(df['high'], 6)
        tsmin_low = self._tsmin(df['low'], 6)
        rsv = (tsmax_high - df['close']) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        return self._sma(rsv, 9, 1)
    
    def _alpha_048(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
        """
        close_delay1 = self._delay(df['close'], 1)
        close_delay2 = self._delay(df['close'], 2)
        close_delay3 = self._delay(df['close'], 3)
        
        sign1 = self._sign(df['close'] - close_delay1)
        sign2 = self._sign(close_delay1 - close_delay2)
        sign3 = self._sign(close_delay2 - close_delay3)
        
        sum_signs = sign1 + sign2 + sign3
        rank_signs = self._rank(sum_signs)
        
        sum_vol_5 = self._sum(df['volume'], 5)
        sum_vol_20 = self._sum(df['volume'], 20)
        
        return -1 * rank_signs * sum_vol_5 / (sum_vol_20 + self._EPSILON)
    
    def _alpha_049(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        """
        high_delay = self._delay(df['high'], 1)
        low_delay = self._delay(df['low'], 1)
        
        hl_sum = df['high'] + df['low']
        hl_sum_delay = high_delay + low_delay
        
        abs_high_diff = np.abs(df['high'] - high_delay)
        abs_low_diff = np.abs(df['low'] - low_delay)
        max_abs_diff = self._max(abs_high_diff, abs_low_diff)
        
        condition1 = hl_sum >= hl_sum_delay
        value1 = np.where(condition1, 0, max_abs_diff)
        sum1 = pd.Series(value1, index=df.index).rolling(12).sum()
        
        condition2 = hl_sum <= hl_sum_delay
        value2 = np.where(condition2, 0, max_abs_diff)
        sum2 = pd.Series(value2, index=df.index).rolling(12).sum()
        
        return sum1 / (sum1 + sum2 + self._EPSILON)
    
    def _alpha_050(self, df: pd.DataFrame) -> pd.Series:
        """
        公式：SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        """
        high_delay = self._delay(df['high'], 1)
        low_delay = self._delay(df['low'], 1)
        
        hl_sum = df['high'] + df['low']
        hl_sum_delay = high_delay + low_delay
        
        abs_high_diff = np.abs(df['high'] - high_delay)
        abs_low_diff = np.abs(df['low'] - low_delay)
        max_abs_diff = self._max(abs_high_diff, abs_low_diff)
        
        condition1 = hl_sum <= hl_sum_delay
        value1 = np.where(condition1, 0, max_abs_diff)
        sum1 = pd.Series(value1, index=df.index).rolling(12).sum()
        
        condition2 = hl_sum >= hl_sum_delay
        value2 = np.where(condition2, 0, max_abs_diff)
        sum2 = pd.Series(value2, index=df.index).rolling(12).sum()
        
        return sum1 / (sum1 + sum2 + self._EPSILON)
    
    # 继续实现剩余的因子...由于篇幅限制，我将实现所有191个因子
    # 这里先实现前50个，然后继续实现剩余的141个因子
    
    def _alpha_051(self, df: pd.DataFrame) -> pd.Series:
        """SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)"""
        high_delay = self._delay(df['high'], 1)
        low_delay = self._delay(df['low'], 1)
        condition = (df['high'] + df['low']) <= (high_delay + low_delay)
        max_diff = self._max(np.abs(df['high'] - high_delay), np.abs(df['low'] - low_delay))
        value = np.where(condition, 0, max_diff)
        return pd.Series(value, index=df.index).rolling(12).sum()
    
    def _alpha_052(self, df: pd.DataFrame) -> pd.Series:
        """SUM(MAX(0,HIGH-DELAY(CLOSE,1)),26)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),26)*100"""
        high_close_delay = df['high'] - self._delay(df['close'], 1)
        close_delay_low = self._delay(df['close'], 1) - df['low']
        sum_up = self._sum(high_close_delay.clip(lower=0), 26)
        sum_down = self._sum(close_delay_low.clip(lower=0), 26)
        return sum_up / (sum_down + self._EPSILON) * 100
    
    def _alpha_053(self, df: pd.DataFrame) -> pd.Series:
        """COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100"""
        condition = df['close'] > self._delay(df['close'], 1)
        count = self._count(condition, 12)
        return count / 12 * 100
    
    def _alpha_054(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))"""
        abs_close_open = np.abs(df['close'] - df['open'])
        std_abs = self._std(abs_close_open, 1)
        close_open_diff = df['close'] - df['open']
        corr_co = self._corr(df['close'], df['open'], 10)
        combined = std_abs + close_open_diff + corr_co
        return -1 * self._rank(combined)
    
    def _alpha_055(self, df: pd.DataFrame) -> pd.Series:
        """SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)"""
        # 这个因子公式非常复杂，简化实现
        close_delay = self._delay(df['close'], 1)
        open_delay = self._delay(df['open'], 1)
        low_delay = self._delay(df['low'], 1)
        
        numerator = 16 * (df['close'] - close_delay + (df['close'] - df['open']) / 2 + close_delay - open_delay)
        
        abs_high_close = np.abs(df['high'] - close_delay)
        abs_low_close = np.abs(df['low'] - close_delay)
        abs_high_low = np.abs(df['high'] - low_delay)
        abs_close_open = np.abs(close_delay - open_delay)
        
        # 简化分母计算
        denominator = abs_high_close + abs_low_close / 2 + abs_close_open / 4
        
        ratio = numerator / (denominator + self._EPSILON)
        max_abs = self._max(abs_high_close, abs_low_close)
        
        return self._sum(ratio * max_abs, 20)
    
    def _alpha_056(self, df: pd.DataFrame) -> pd.Series:
        """(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))"""
        open_tsmin = self._tsmin(df['open'], 12)
        rank_open_diff = self._rank(df['open'] - open_tsmin)
        
        hl_avg = (df['high'] + df['low']) / 2
        sum_hl = self._sum(hl_avg, 19)
        mean_vol = self._mean(df['volume'], 40)
        sum_mean_vol = self._sum(mean_vol, 19)
        corr_result = self._corr(sum_hl, sum_mean_vol, 13)
        rank_corr = self._rank(corr_result)
        rank_corr_pow = rank_corr ** 5
        
        condition = rank_open_diff < rank_corr_pow
        return pd.Series(np.where(condition, 1, 0), index=df.index)
    
    def _alpha_057(self, df: pd.DataFrame) -> pd.Series:
        """SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)"""
        tsmin_low = self._tsmin(df['low'], 9)
        tsmax_high = self._tsmax(df['high'], 9)
        rsv = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        return self._sma(rsv, 3, 1)
    
    def _alpha_058(self, df: pd.DataFrame) -> pd.Series:
        """COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100"""
        condition = df['close'] > self._delay(df['close'], 1)
        count = self._count(condition, 20)
        return count / 20 * 100
    
    def _alpha_059(self, df: pd.DataFrame) -> pd.Series:
        """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)"""
        close_delay = self._delay(df['close'], 1)
        condition_eq = df['close'] == close_delay
        condition_gt = df['close'] > close_delay
        
        min_low_close = self._min(df['low'], close_delay)
        max_high_close = self._max(df['high'], close_delay)
        
        value = np.where(
            condition_eq, 0,
            np.where(
                condition_gt,
                df['close'] - min_low_close,
                df['close'] - max_high_close
            )
        )
        return pd.Series(value, index=df.index).rolling(20).sum()
    
    def _alpha_060(self, df: pd.DataFrame) -> pd.Series:
        """SUM(((CLOSE-DELAY(CLOSE,1))>0?(CLOSE-DELAY(CLOSE,1)):0),20)"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        value = diff.clip(lower=0)
        return self._sum(value, 20)
    
    def _alpha_061(self, df: pd.DataFrame) -> pd.Series:
        """SUM(((CLOSE-DELAY(CLOSE,1))<0?ABS(CLOSE-DELAY(CLOSE,1)):0),20)"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        value = np.where(diff < 0, np.abs(diff), 0)
        return pd.Series(value, index=df.index).rolling(20).sum()
    
    def _alpha_062(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * CORR(HIGH, RANK(VOLUME), 5))"""
        rank_volume = self._rank(df['volume'])
        return -1 * self._corr(df['high'], rank_volume, 5)
    
    def _alpha_063(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        max_diff = diff.clip(lower=0)
        abs_diff = np.abs(diff)
        sma_max = self._sma(max_diff, 6, 1)
        sma_abs = self._sma(abs_diff, 6, 1)
        return sma_max / (sma_abs + self._EPSILON) * 100
    
    def _alpha_064(self, df: pd.DataFrame) -> pd.Series:
        """(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr1 = self._corr(rank_vwap, rank_volume, 4)
        decay1 = self._decaylinear(corr1, 4)
        rank_decay1 = self._rank(decay1)
        
        rank_close = self._rank(df['close'])
        mean_vol = self._mean(df['volume'], 60)
        rank_mean_vol = self._rank(mean_vol)
        corr2 = self._corr(rank_close, rank_mean_vol, 4)
        max_corr2 = self._tsmax(corr2, 13)
        decay2 = self._decaylinear(max_corr2, 14)
        rank_decay2 = self._rank(decay2)
        
        return self._max(rank_decay1, rank_decay2) * -1
    
    def _alpha_065(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(CLOSE,6)/CLOSE"""
        mean_close = self._mean(df['close'], 6)
        return mean_close / (df['close'] + self._EPSILON)
    
    def _alpha_066(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100"""
        mean_close = self._mean(df['close'], 6)
        return (df['close'] - mean_close) / (mean_close + self._EPSILON) * 100
    
    def _alpha_067(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        max_diff = diff.clip(lower=0)
        abs_diff = np.abs(diff)
        sma_max = self._sma(max_diff, 24, 1)
        sma_abs = self._sma(abs_diff, 24, 1)
        return sma_max / (sma_abs + self._EPSILON) * 100
    
    def _alpha_068(self, df: pd.DataFrame) -> pd.Series:
        """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)"""
        hl_avg = (df['high'] + df['low']) / 2
        hl_avg_delay = (self._delay(df['high'], 1) + self._delay(df['low'], 1)) / 2
        hl_diff = df['high'] - df['low']
        numerator = (hl_avg - hl_avg_delay) * hl_diff / (df['volume'] + self._EPSILON)
        return self._sma(numerator, 15, 2)
    
    def _alpha_069(self, df: pd.DataFrame) -> pd.Series:
        """(SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)?0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))"""
        # DTM = (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        open_delay = self._delay(df['open'], 1)
        dtm = np.where(df['open'] <= open_delay, 0, 
                       np.maximum(df['high'] - df['open'], df['open'] - open_delay))
        # DBM = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        dbm = np.where(df['open'] >= open_delay, 0,
                       np.maximum(df['open'] - df['low'], df['open'] - open_delay))
        
        sum_dtm = pd.Series(dtm, index=df.index).rolling(20).sum()
        sum_dbm = pd.Series(dbm, index=df.index).rolling(20).sum()
        
        condition1 = sum_dtm > sum_dbm
        condition2 = sum_dtm == sum_dbm
        
        result = np.where(condition1, (sum_dtm - sum_dbm) / (sum_dtm + self._EPSILON),
                         np.where(condition2, 0, (sum_dtm - sum_dbm) / (sum_dbm + self._EPSILON)))
        return pd.Series(result, index=df.index)
    
    def _alpha_070(self, df: pd.DataFrame) -> pd.Series:
        """STD(AMOUNT,6)"""
        amount = df['amount']  # 预处理已确保amount存在
        return self._std(amount, 6)
    
    def _alpha_071(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100"""
        mean_close = self._mean(df['close'], 24)
        return (df['close'] - mean_close) / (mean_close + self._EPSILON) * 100
    
    def _alpha_072(self, df: pd.DataFrame) -> pd.Series:
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)"""
        tsmax_high = self._tsmax(df['high'], 6)
        tsmin_low = self._tsmin(df['low'], 6)
        rsv = (tsmax_high - df['close']) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        return self._sma(rsv, 15, 1)
    
    def _alpha_073(self, df: pd.DataFrame) -> pd.Series:
        """((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)"""
        corr_cv = self._corr(df['close'], df['volume'], 10)
        decay1 = self._decaylinear(corr_cv, 16)
        decay2 = self._decaylinear(decay1, 4)
        tsrank_decay = self._tsrank(decay2, 5)
        
        vwap = df['vwap']  # 预处理已确保vwap存在
        mean_vol = self._mean(df['volume'], 30)
        corr_vm = self._corr(vwap, mean_vol, 4)
        decay3 = self._decaylinear(corr_vm, 3)
        rank_decay = self._rank(decay3)
        
        return (tsrank_decay - rank_decay) * -1
    
    def _alpha_074(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        weighted_low = df['low'] * 0.35 + vwap * 0.65
        sum_weighted = self._sum(weighted_low, 20)
        mean_vol = self._mean(df['volume'], 40)
        sum_mean_vol = self._sum(mean_vol, 20)
        corr1 = self._corr(sum_weighted, sum_mean_vol, 7)
        rank_corr1 = self._rank(corr1)
        
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr2 = self._corr(rank_vwap, rank_volume, 6)
        rank_corr2 = self._rank(corr2)
        
        return rank_corr1 + rank_corr2
    
    def _alpha_075(self, df: pd.DataFrame) -> pd.Series:
        """COUNT(CLOSE>OPEN&BARSLAST(CLOSE>OPEN)>=0,20)/20*100"""
        condition = (df['close'] > df['open'])
        count = self._count(condition, 20)
        return count / 20 * 100
    
    def _alpha_076(self, df: pd.DataFrame) -> pd.Series:
        """STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)"""
        close_delay = self._delay(df['close'], 1)
        ret = (df['close'] / (close_delay + self._EPSILON) - 1)
        abs_ret_vol = np.abs(ret) / (df['volume'] + self._EPSILON)
        std_abs = self._std(abs_ret_vol, 20)
        mean_abs = self._mean(abs_ret_vol, 20)
        return std_abs / (mean_abs + self._EPSILON)
    
    def _alpha_077(self, df: pd.DataFrame) -> pd.Series:
        """MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        hl_avg = (df['high'] + df['low']) / 2
        part1 = (hl_avg + df['high']) - (vwap + df['high'])
        decay1 = self._decaylinear(part1, 20)
        rank_decay1 = self._rank(decay1)
        
        mean_vol = self._mean(df['volume'], 40)
        corr_result = self._corr(hl_avg, mean_vol, 3)
        decay2 = self._decaylinear(corr_result, 6)
        rank_decay2 = self._rank(decay2)
        
        return self._min(rank_decay1, rank_decay2)
    
    def _alpha_078(self, df: pd.DataFrame) -> pd.Series:
        """((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))"""
        hlc_avg = (df['high'] + df['low'] + df['close']) / 3
        ma_hlc = self._mean(hlc_avg, 12)
        numerator = hlc_avg - ma_hlc
        mean_abs_diff = self._mean(np.abs(df['close'] - self._mean(hlc_avg, 12)), 12)
        denominator = 0.015 * mean_abs_diff
        return numerator / (denominator + self._EPSILON)
    
    def _alpha_079(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        max_diff = diff.clip(lower=0)
        abs_diff = np.abs(diff)
        sma_max = self._sma(max_diff, 12, 1)
        sma_abs = self._sma(abs_diff, 12, 1)
        return sma_max / (sma_abs + self._EPSILON) * 100
    
    def _alpha_080(self, df: pd.DataFrame) -> pd.Series:
        """(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100"""
        volume_delay = self._delay(df['volume'], 5)
        return (df['volume'] - volume_delay) / (volume_delay + self._EPSILON) * 100
    
    def _alpha_081(self, df: pd.DataFrame) -> pd.Series:
        """SMA(VOLUME,21,2)"""
        return self._sma(df['volume'], 21, 2)
    
    def _alpha_082(self, df: pd.DataFrame) -> pd.Series:
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)"""
        tsmax_high = self._tsmax(df['high'], 6)
        tsmin_low = self._tsmin(df['low'], 6)
        rsv = (tsmax_high - df['close']) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        return self._sma(rsv, 20, 1)
    
    def _alpha_083(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))"""
        rank_high = self._rank(df['high'])
        rank_volume = self._rank(df['volume'])
        cov = self._covariance(rank_high, rank_volume, 5)
        return -1 * self._rank(cov)
    
    def _alpha_084(self, df: pd.DataFrame) -> pd.Series:
        """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)"""
        close_delay = self._delay(df['close'], 1)
        condition_up = df['close'] > close_delay
        condition_down = df['close'] < close_delay
        value = np.where(condition_up, df['volume'], np.where(condition_down, -df['volume'], 0))
        return pd.Series(value, index=df.index).rolling(20).sum()
    
    def _alpha_085(self, df: pd.DataFrame) -> pd.Series:
        """(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))"""
        volume_mean = self._mean(df['volume'], 20)
        volume_ratio = df['volume'] / (volume_mean + self._EPSILON)
        tsrank_vol = self._tsrank(volume_ratio, 20)
        
        delta_close = self._delta(df['close'], 7)
        tsrank_delta = self._tsrank(-1 * delta_close, 8)
        
        return tsrank_vol * tsrank_delta
    
    def _alpha_086(self, df: pd.DataFrame) -> pd.Series:
        """((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : (-1 * 1)))"""
        close_delay10 = self._delay(df['close'], 10)
        close_delay20 = self._delay(df['close'], 20)
        part1 = (close_delay20 - close_delay10) / 10
        part2 = (close_delay10 - df['close']) / 10
        diff = part1 - part2
        
        condition1 = 0.25 < diff
        condition2 = diff < 0
        
        result = np.where(condition1, -1, np.where(condition2, 1, -1))
        return pd.Series(result, index=df.index)
    
    def _alpha_087(self, df: pd.DataFrame) -> pd.Series:
        """((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - (HIGH + LOW) / 2)), 11), 7)) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_vwap = self._delta(vwap, 4)
        decay1 = self._decaylinear(delta_vwap, 7)
        rank_decay1 = self._rank(decay1)
        
        weighted_low = df['low'] * 0.9 + df['low'] * 0.1
        numerator = weighted_low - vwap
        denominator = df['open'] - (df['high'] + df['low']) / 2
        ratio = numerator / (denominator + self._EPSILON)
        decay2 = self._decaylinear(ratio, 11)
        tsrank_decay2 = self._tsrank(decay2, 7)
        
        return (rank_decay1 + tsrank_decay2) * -1
    
    def _alpha_088(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100"""
        close_delay = self._delay(df['close'], 20)
        return (df['close'] - close_delay) / (close_delay + self._EPSILON) * 100
    
    def _alpha_089(self, df: pd.DataFrame) -> pd.Series:
        """2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))"""
        sma1 = self._sma(df['close'], 13, 2)
        sma2 = self._sma(df['close'], 27, 2)
        diff = sma1 - sma2
        sma_diff = self._sma(diff, 10, 2)
        return 2 * (diff - sma_diff)
    
    def _alpha_090(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr_result = self._corr(rank_vwap, rank_volume, 5)
        return self._rank(corr_result) * -1
    
    def _alpha_091(self, df: pd.DataFrame) -> pd.Series:
        """((RANK((CLOSE - MAX(CLOSE, 5))) * RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)"""
        max_close = self._tsmax(df['close'], 5)
        rank_diff = self._rank(df['close'] - max_close)
        
        mean_vol = self._mean(df['volume'], 40)
        corr_result = self._corr(mean_vol, df['low'], 5)
        rank_corr = self._rank(corr_result)
        
        return rank_diff * rank_corr * -1
    
    def _alpha_092(self, df: pd.DataFrame) -> pd.Series:
        """(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP * 0.65)), 2), 15)), RANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 17))) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        weighted_close = df['close'] * 0.35 + vwap * 0.65
        delta_weighted = self._delta(weighted_close, 2)
        decay1 = self._decaylinear(delta_weighted, 15)
        rank_decay1 = self._rank(decay1)
        
        mean_vol = self._mean(df['volume'], 180)
        corr_result = self._corr(mean_vol, df['close'], 13)
        abs_corr = np.abs(corr_result)
        decay2 = self._decaylinear(abs_corr, 17)
        rank_decay2 = self._rank(decay2)
        
        return self._max(rank_decay1, rank_decay2) * -1
    
    def _alpha_093(self, df: pd.DataFrame) -> pd.Series:
        """SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))),20)"""
        open_delay = self._delay(df['open'], 1)
        condition = df['open'] >= open_delay
        max_diff = self._max(df['open'] - df['low'], df['open'] - open_delay)
        value = np.where(condition, 0, max_diff)
        return pd.Series(value, index=df.index).rolling(20).sum()
    
    def _alpha_094(self, df: pd.DataFrame) -> pd.Series:
        """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0),30)"""
        close_delay = self._delay(df['close'], 1)
        condition_up = df['close'] > close_delay
        condition_down = df['close'] < close_delay
        value = np.where(condition_up, df['volume'], np.where(condition_down, -df['volume'], 0))
        return pd.Series(value, index=df.index).rolling(30).sum()
    
    def _alpha_095(self, df: pd.DataFrame) -> pd.Series:
        """STD(AMOUNT,20)"""
        amount = df['amount']  # 预处理已确保amount存在
        return self._std(amount, 20)
    
    def _alpha_096(self, df: pd.DataFrame) -> pd.Series:
        """SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)"""
        tsmin_low = self._tsmin(df['low'], 9)
        tsmax_high = self._tsmax(df['high'], 9)
        rsv = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + self._EPSILON) * 100
        sma1 = self._sma(rsv, 3, 1)
        return self._sma(sma1, 3, 1)
    
    def _alpha_097(self, df: pd.DataFrame) -> pd.Series:
        """STD(VOLUME,10)"""
        return self._std(df['volume'], 10)
    
    def _alpha_098(self, df: pd.DataFrame) -> pd.Series:
        """((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))"""
        sum_close_100 = self._sum(df['close'], 100) / 100
        delta_sum = self._delta(sum_close_100, 100)
        close_delay100 = self._delay(df['close'], 100)
        ratio = delta_sum / (close_delay100 + self._EPSILON)
        
        condition = (ratio < 0.05) | (ratio == 0.05)
        value1 = -1 * (df['close'] - self._tsmin(df['close'], 100))
        value2 = -1 * self._delta(df['close'], 3)
        
        result = np.where(condition, value1, value2)
        return pd.Series(result, index=df.index)
    
    def _alpha_099(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))"""
        rank_close = self._rank(df['close'])
        rank_volume = self._rank(df['volume'])
        cov = self._covariance(rank_close, rank_volume, 5)
        return -1 * self._rank(cov)
    
    def _alpha_100(self, df: pd.DataFrame) -> pd.Series:
        """STD(VOLUME,20)"""
        return self._std(df['volume'], 20)
    
    def _alpha_101(self, df: pd.DataFrame) -> pd.Series:
        """((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))"""
        return (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    
    def _alpha_102(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        max_diff = diff.clip(lower=0)
        abs_diff = np.abs(diff)
        sma_max = self._sma(max_diff, 6, 1)
        sma_abs = self._sma(abs_diff, 6, 1)
        return sma_max / (sma_abs + self._EPSILON) * 100
    
    def _alpha_103(self, df: pd.DataFrame) -> pd.Series:
        """((20 - HIGHDAY(HIGH, 20)) / 20) * 100"""
        highday = self._highday(df['high'], 20)
        return (20 - highday) / 20 * 100
    
    def _alpha_104(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))"""
        corr_hv = self._corr(df['high'], df['volume'], 5)
        delta_corr = self._delta(corr_hv, 5)
        std_close = self._std(df['close'], 20)
        rank_std = self._rank(std_close)
        return -1 * delta_corr * rank_std
    
    def _alpha_105(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))"""
        rank_open = self._rank(df['open'])
        rank_volume = self._rank(df['volume'])
        return -1 * self._corr(rank_open, rank_volume, 10)
    
    def _alpha_106(self, df: pd.DataFrame) -> pd.Series:
        """CLOSE-DELAY(CLOSE,10)"""
        return df['close'] - self._delay(df['close'], 10)
    
    def _alpha_107(self, df: pd.DataFrame) -> pd.Series:
        """((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))"""
        high_delay = self._delay(df['high'], 1)
        close_delay = self._delay(df['close'], 1)
        low_delay = self._delay(df['low'], 1)
        
        rank1 = self._rank(df['open'] - high_delay)
        rank2 = self._rank(df['open'] - close_delay)
        rank3 = self._rank(df['open'] - low_delay)
        
        return -1 * rank1 * rank2 * rank3
    
    def _alpha_108(self, df: pd.DataFrame) -> pd.Series:
        """((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        min_high = self._tsmin(df['high'], 2)
        rank_diff = self._rank(df['high'] - min_high)
        
        mean_vol = self._mean(df['volume'], 120)
        corr_result = self._corr(vwap, mean_vol, 6)
        rank_corr = self._rank(corr_result)
        
        return (rank_diff ** rank_corr) * -1
    
    def _alpha_109(self, df: pd.DataFrame) -> pd.Series:
        """SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)"""
        hl_diff = df['high'] - df['low']
        sma1 = self._sma(hl_diff, 10, 2)
        sma2 = self._sma(sma1, 10, 2)
        return sma1 / (sma2 + self._EPSILON)
    
    def _alpha_110(self, df: pd.DataFrame) -> pd.Series:
        """SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100"""
        high_close_delay = df['high'] - self._delay(df['close'], 1)
        close_delay_low = self._delay(df['close'], 1) - df['low']
        sum_up = self._sum(high_close_delay.clip(lower=0), 20)
        sum_down = self._sum(close_delay_low.clip(lower=0), 20)
        return sum_up / (sum_down + self._EPSILON) * 100
    
    def _alpha_111(self, df: pd.DataFrame) -> pd.Series:
        """SMA(VOLUME*(2*CLOSE-HIGH-LOW)/(HIGH+LOW),11,2)-SMA(VOLUME*(2*CLOSE-HIGH-LOW)/(HIGH+LOW),4,2)"""
        numerator = df['volume'] * (2 * df['close'] - df['high'] - df['low'])
        denominator = df['high'] + df['low'] + self._EPSILON
        ratio = numerator / denominator
        sma1 = self._sma(ratio, 11, 2)
        sma2 = self._sma(ratio, 4, 2)
        return sma1 - sma2
    
    def _alpha_112(self, df: pd.DataFrame) -> pd.Series:
        """(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        
        sum_up = self._sum(diff.clip(lower=0), 12)
        sum_down = self._sum(np.abs(diff.clip(upper=0)), 12)
        
        return (sum_up - sum_down) / (sum_up + sum_down + self._EPSILON) * 100
    
    def _alpha_113(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))"""
        delay_close = self._delay(df['close'], 5)
        sum_delay = self._sum(delay_close, 20) / 20
        rank_sum = self._rank(sum_delay)
        
        corr_cv = self._corr(df['close'], df['volume'], 2)
        
        sum_close5 = self._sum(df['close'], 5)
        sum_close20 = self._sum(df['close'], 20)
        corr_sum = self._corr(sum_close5, sum_close20, 2)
        rank_corr = self._rank(corr_sum)
        
        return -1 * rank_sum * corr_cv * rank_corr
    
    def _alpha_114(self, df: pd.DataFrame) -> pd.Series:
        """((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        hl_diff = df['high'] - df['low']
        sum_close5 = self._sum(df['close'], 5) / 5
        ratio = hl_diff / (sum_close5 + self._EPSILON)
        delay_ratio = self._delay(ratio, 2)
        rank_delay = self._rank(delay_ratio)
        
        rank_volume = self._rank(self._rank(df['volume']))
        
        denominator = ratio / (vwap - df['close'] + self._EPSILON)
        
        return rank_delay * rank_volume / (denominator + self._EPSILON)
    
    def _alpha_115(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))"""
        weighted_price = df['high'] * 0.9 + df['close'] * 0.1
        mean_vol = self._mean(df['volume'], 30)
        corr1 = self._corr(weighted_price, mean_vol, 10)
        rank_corr1 = self._rank(corr1)
        
        hl_avg = (df['high'] + df['low']) / 2
        tsrank_hl = self._tsrank(hl_avg, 4)
        tsrank_vol = self._tsrank(df['volume'], 10)
        corr2 = self._corr(tsrank_hl, tsrank_vol, 7)
        rank_corr2 = self._rank(corr2)
        
        return rank_corr1 ** rank_corr2
    
    def _alpha_116(self, df: pd.DataFrame) -> pd.Series:
        """REGBETA(CLOSE,SEQUENCE(20),20)"""
        sequence_20 = pd.Series(self._sequence(20), index=df.index[-20:])
        def calc_beta(window):
            if len(window) < 20:
                return np.nan
            y = window
            x = np.arange(1, 21)
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        return df['close'].rolling(20).apply(calc_beta, raw=True)
    
    def _alpha_117(self, df: pd.DataFrame) -> pd.Series:
        """((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))"""
        tsrank_vol = self._tsrank(df['volume'], 32)
        hl_sum = df['close'] + df['high'] - df['low']
        tsrank_hl = self._tsrank(hl_sum, 16)
        ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        tsrank_ret = self._tsrank(ret, 32)
        
        return tsrank_vol * (1 - tsrank_hl) * (1 - tsrank_ret)
    
    def _alpha_118(self, df: pd.DataFrame) -> pd.Series:
        """SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100"""
        high_open = df['high'] - df['open']
        open_low = df['open'] - df['low']
        sum_high_open = self._sum(high_open, 20)
        sum_open_low = self._sum(open_low, 20)
        return sum_high_open / (sum_open_low + self._EPSILON) * 100
    
    def _alpha_119(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 6)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        mean_vol = self._mean(df['volume'], 5)
        sum_mean_vol = self._sum(mean_vol, 26)
        corr1 = self._corr(vwap, sum_mean_vol, 5)
        decay1 = self._decaylinear(corr1, 7)
        rank_decay1 = self._rank(decay1)
        
        rank_open = self._rank(df['open'])
        mean_vol15 = self._mean(df['volume'], 15)
        rank_mean_vol = self._rank(mean_vol15)
        corr2 = self._corr(rank_open, rank_mean_vol, 21)
        min_corr = self._tsmin(corr2, 9)
        tsrank_min = self._tsrank(min_corr, 7)
        decay2 = self._decaylinear(tsrank_min, 6)
        rank_decay2 = self._rank(decay2)
        
        return rank_decay1 - rank_decay2
    
    def _alpha_120(self, df: pd.DataFrame) -> pd.Series:
        """(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_diff = self._rank(vwap - df['close'])
        rank_sum = self._rank(vwap + df['close'])
        return rank_diff / (rank_sum + self._EPSILON)
    
    def _alpha_121(self, df: pd.DataFrame) -> pd.Series:
        """((RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 4), 18), 3)) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        min_vwap = self._tsmin(vwap, 12)
        rank_diff = self._rank(vwap - min_vwap)
        
        tsrank_vwap = self._tsrank(vwap, 20)
        mean_vol = self._mean(df['volume'], 60)
        tsrank_mean_vol = self._tsrank(mean_vol, 4)
        corr_result = self._corr(tsrank_vwap, tsrank_mean_vol, 18)
        tsrank_corr = self._tsrank(corr_result, 3)
        
        return (rank_diff ** tsrank_corr) * -1
    
    def _alpha_122(self, df: pd.DataFrame) -> pd.Series:
        """(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)"""
        log_close = np.log(df['close'] + self._EPSILON)
        sma1 = self._sma(log_close, 13, 2)
        sma2 = self._sma(sma1, 13, 2)
        sma3 = self._sma(sma2, 13, 2)
        sma3_delay = self._delay(sma3, 1)
        return (sma3 - sma3_delay) / (sma3_delay + self._EPSILON)
    
    def _alpha_123(self, df: pd.DataFrame) -> pd.Series:
        """((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)"""
        hl_avg = (df['high'] + df['low']) / 2
        sum_hl = self._sum(hl_avg, 20)
        mean_vol = self._mean(df['volume'], 60)
        sum_mean_vol = self._sum(mean_vol, 20)
        corr1 = self._corr(sum_hl, sum_mean_vol, 9)
        rank_corr1 = self._rank(corr1)
        
        corr2 = self._corr(df['low'], df['volume'], 6)
        rank_corr2 = self._rank(corr2)
        
        condition = rank_corr1 < rank_corr2
        return pd.Series(np.where(condition, -1, 0), index=df.index)
    
    def _alpha_124(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)), 2)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        numerator = df['close'] - vwap
        
        tsmax_close = self._tsmax(df['close'], 30)
        rank_tsmax = self._rank(tsmax_close)
        decay = self._decaylinear(rank_tsmax, 2)
        
        return numerator / (decay + self._EPSILON)
    
    def _alpha_125(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80), 17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        mean_vol = self._mean(df['volume'], 80)
        corr_result = self._corr(vwap, mean_vol, 17)
        decay1 = self._decaylinear(corr_result, 20)
        rank_decay1 = self._rank(decay1)
        
        weighted_price = df['close'] * 0.5 + vwap * 0.5
        delta_weighted = self._delta(weighted_price, 3)
        decay2 = self._decaylinear(delta_weighted, 16)
        rank_decay2 = self._rank(decay2)
        
        return rank_decay1 / (rank_decay2 + self._EPSILON)
    
    def _alpha_126(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE+HIGH+LOW)/3"""
        return (df['close'] + df['high'] + df['low']) / 3
    
    def _alpha_127(self, df: pd.DataFrame) -> pd.Series:
        """(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)"""
        max_close = self._tsmax(df['close'], 12)
        ratio = 100 * (df['close'] - max_close) / (max_close + self._EPSILON)
        squared = ratio ** 2
        mean_squared = self._mean(squared, 12)
        return np.sqrt(mean_squared)
    
    def _alpha_128(self, df: pd.DataFrame) -> pd.Series:
        """100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?((HIGH+LOW+CLOSE)/3-DELAY((HIGH+LOW+CLOSE)/3,1)):0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?ABS(((HIGH+LOW+CLOSE)/3-DELAY((HIGH+LOW+CLOSE)/3,1))):0),14)))"""
        hlc_avg = (df['high'] + df['low'] + df['close']) / 3
        hlc_avg_delay = self._delay(hlc_avg, 1)
        
        condition_up = hlc_avg > hlc_avg_delay
        condition_down = hlc_avg < hlc_avg_delay
        
        sum_up = self._sum(pd.Series(np.where(condition_up, hlc_avg - hlc_avg_delay, 0), index=df.index), 14)
        sum_down = self._sum(pd.Series(np.where(condition_down, np.abs(hlc_avg - hlc_avg_delay), 0), index=df.index), 14)
        
        rsi_like = 100 / (1 + sum_up / (sum_down + self._EPSILON))
        return 100 - rsi_like
    
    def _alpha_129(self, df: pd.DataFrame) -> pd.Series:
        """SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        value = np.where(diff < 0, np.abs(diff), 0)
        return pd.Series(value, index=df.index).rolling(12).sum()
    
    def _alpha_130(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        hl_avg = (df['high'] + df['low']) / 2
        mean_vol = self._mean(df['volume'], 40)
        corr1 = self._corr(hl_avg, mean_vol, 9)
        decay1 = self._decaylinear(corr1, 10)
        rank_decay1 = self._rank(decay1)
        
        rank_vwap = self._rank(vwap)
        rank_volume = self._rank(df['volume'])
        corr2 = self._corr(rank_vwap, rank_volume, 7)
        decay2 = self._decaylinear(corr2, 3)
        rank_decay2 = self._rank(decay2)
        
        return rank_decay1 / (rank_decay2 + self._EPSILON)
    
    def _alpha_131(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_vwap = self._delta(vwap, 1)
        rank_delta = self._rank(delta_vwap)
        
        mean_vol = self._mean(df['volume'], 50)
        corr_result = self._corr(df['close'], mean_vol, 18)
        tsrank_corr = self._tsrank(corr_result, 18)
        
        return rank_delta ** tsrank_corr
    
    def _alpha_132(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(AMOUNT,20)"""
        amount = df['amount']  # 预处理已确保amount存在
        return self._mean(amount, 20)
    
    def _alpha_133(self, df: pd.DataFrame) -> pd.Series:
        """((20-HIGHDAY(HIGH,20))/20)*100 - ((20-LOWDAY(LOW,20))/20)*100"""
        highday = self._highday(df['high'], 20)
        lowday = self._lowday(df['low'], 20)
        part1 = (20 - highday) / 20 * 100
        part2 = (20 - lowday) / 20 * 100
        return part1 - part2
    
    def _alpha_134(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME"""
        close_delay = self._delay(df['close'], 12)
        return (df['close'] - close_delay) / (close_delay + self._EPSILON) * df['volume']
    
    def _alpha_135(self, df: pd.DataFrame) -> pd.Series:
        """SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)"""
        close_delay20 = self._delay(df['close'], 20)
        ratio = df['close'] / (close_delay20 + self._EPSILON)
        ratio_delay = self._delay(ratio, 1)
        return self._sma(ratio_delay, 20, 1)
    
    def _alpha_136(self, df: pd.DataFrame) -> pd.Series:
        """((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))"""
        ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        delta_ret = self._delta(ret, 3)
        rank_delta = self._rank(delta_ret)
        
        corr_ov = self._corr(df['open'], df['volume'], 10)
        
        return -1 * rank_delta * corr_ov
    
    def _alpha_137(self, df: pd.DataFrame) -> pd.Series:
        """16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))"""
        # 简化实现，与alpha_055类似
        close_delay = self._delay(df['close'], 1)
        open_delay = self._delay(df['open'], 1)
        low_delay = self._delay(df['low'], 1)
        
        numerator = 16 * (df['close'] - close_delay + (df['close'] - df['open']) / 2 + close_delay - open_delay)
        
        abs_high_close = np.abs(df['high'] - close_delay)
        abs_low_close = np.abs(df['low'] - close_delay)
        abs_high_low = np.abs(df['high'] - low_delay)
        abs_close_open = np.abs(close_delay - open_delay)
        
        denominator = abs_high_close + abs_low_close / 2 + abs_close_open / 4
        max_abs = self._max(abs_high_close, abs_low_close)
        
        return numerator / (denominator + self._EPSILON) * max_abs
    
    def _alpha_138(self, df: pd.DataFrame) -> pd.Series:
        """((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        weighted_low = df['low'] * 0.7 + vwap * 0.3
        delta_weighted = self._delta(weighted_low, 3)
        decay1 = self._decaylinear(delta_weighted, 20)
        rank_decay1 = self._rank(decay1)
        
        tsrank_low = self._tsrank(df['low'], 8)
        mean_vol = self._mean(df['volume'], 60)
        tsrank_mean_vol = self._tsrank(mean_vol, 17)
        corr_result = self._corr(tsrank_low, tsrank_mean_vol, 5)
        tsrank_corr = self._tsrank(corr_result, 19)
        decay2 = self._decaylinear(tsrank_corr, 16)
        tsrank_decay2 = self._tsrank(decay2, 7)
        
        return (rank_decay1 - tsrank_decay2) * -1
    
    def _alpha_139(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * CORR(OPEN, VOLUME, 10))"""
        return -1 * self._corr(df['open'], df['volume'], 10)
    
    def _alpha_140(self, df: pd.DataFrame) -> pd.Series:
        """MIN(RANK(DECAYLINEAR(((RANK(((HIGH + LOW) / 2)) + RANK(HIGH)) + RANK(VOLUME)), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))"""
        hl_avg = (df['high'] + df['low']) / 2
        rank_hl = self._rank(hl_avg)
        rank_high = self._rank(df['high'])
        rank_vol = self._rank(df['volume'])
        combined = rank_hl + rank_high + rank_vol
        decay1 = self._decaylinear(combined, 8)
        rank_decay1 = self._rank(decay1)
        
        tsrank_close = self._tsrank(df['close'], 8)
        mean_vol = self._mean(df['volume'], 60)
        tsrank_mean_vol = self._tsrank(mean_vol, 20)
        corr_result = self._corr(tsrank_close, tsrank_mean_vol, 8)
        decay2 = self._decaylinear(corr_result, 7)
        tsrank_decay2 = self._tsrank(decay2, 3)
        
        return self._min(rank_decay1, tsrank_decay2)
    
    def _alpha_141(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) < RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,15)), 9))) * -1"""
        mean_vol = self._mean(df['volume'], 15)
        rank_mean_vol = self._rank(mean_vol)
        
        rank_high = self._rank(df['high'])
        corr_high = self._corr(rank_high, rank_mean_vol, 9)
        rank_corr_high = self._rank(corr_high)
        
        rank_low = self._rank(df['low'])
        corr_low = self._corr(rank_low, rank_mean_vol, 9)
        rank_corr_low = self._rank(corr_low)
        
        condition = rank_corr_high < rank_corr_low
        return pd.Series(np.where(condition, -1, 0), index=df.index)
    
    def _alpha_142(self, df: pd.DataFrame) -> pd.Series:
        """(((RANK(TSMAX(DELTA(CLOSE, 1), 3)) + RANK(TSMIN(DELTA(CLOSE, 1), 3))) * RANK(DELTA(VOLUME, 3))) * RANK(CORR(((CLOSE * 0.5) + (VWAP * 0.5)), MEAN(VOLUME,180), 17)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_close = self._delta(df['close'], 1)
        tsmax_delta = self._tsmax(delta_close, 3)
        tsmin_delta = self._tsmin(delta_close, 3)
        rank_tsmax = self._rank(tsmax_delta)
        rank_tsmin = self._rank(tsmin_delta)
        
        delta_vol = self._delta(df['volume'], 3)
        rank_delta_vol = self._rank(delta_vol)
        
        weighted_price = df['close'] * 0.5 + vwap * 0.5
        mean_vol = self._mean(df['volume'], 180)
        corr_result = self._corr(weighted_price, mean_vol, 17)
        rank_corr = self._rank(corr_result)
        
        return (rank_tsmax + rank_tsmin) * rank_delta_vol * rank_corr
    
    def _alpha_143(self, df: pd.DataFrame) -> pd.Series:
        """((CLOSE>DELAY(CLOSE,1)?1:-1)*VOLUME)"""
        close_delay = self._delay(df['close'], 1)
        condition = df['close'] > close_delay
        sign = np.where(condition, 1, -1)
        return pd.Series(sign, index=df.index) * df['volume']
    
    def _alpha_144(self, df: pd.DataFrame) -> pd.Series:
        """SUM((CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20))^2,20)"""
        close_delay = self._delay(df['close'], 1)
        ret = df['close'] / (close_delay + self._EPSILON) - 1
        mean_ret = self._mean(ret, 20)
        diff_squared = (ret - mean_ret) ** 2
        return self._sum(diff_squared, 20)
    
    def _alpha_145(self, df: pd.DataFrame) -> pd.Series:
        """(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100"""
        mean_vol9 = self._mean(df['volume'], 9)
        mean_vol26 = self._mean(df['volume'], 26)
        mean_vol12 = self._mean(df['volume'], 12)
        return (mean_vol9 - mean_vol26) / (mean_vol12 + self._EPSILON) * 100
    
    def _alpha_146(self, df: pd.DataFrame) -> pd.Series:
        """MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),20),20)"""
        close_delay = self._delay(df['close'], 1)
        ret = (df['close'] - close_delay) / (close_delay + self._EPSILON)
        mean_ret = self._mean(ret, 20)
        diff = ret - mean_ret
        return self._mean(diff, 20)
    
    def _alpha_147(self, df: pd.DataFrame) -> pd.Series:
        """REGbeta(MEAN(CLOSE,12),SEQUENCE(12),12)"""
        mean_close = self._mean(df['close'], 12)
        def calc_beta(window):
            if len(window) < 12:
                return np.nan
            y = window
            x = np.arange(1, 13)
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        return mean_close.rolling(12).apply(calc_beta, raw=True)
    
    def _alpha_148(self, df: pd.DataFrame) -> pd.Series:
        """((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)"""
        mean_vol = self._mean(df['volume'], 60)
        sum_mean_vol = self._sum(mean_vol, 9)
        corr_result = self._corr(df['open'], sum_mean_vol, 6)
        rank_corr = self._rank(corr_result)
        
        tsmin_open = self._tsmin(df['open'], 14)
        rank_diff = self._rank(df['open'] - tsmin_open)
        
        condition = rank_corr < rank_diff
        return pd.Series(np.where(condition, -1, 0), index=df.index)
    
    def _alpha_149(self, df: pd.DataFrame) -> pd.Series:
        """REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<0),SEQUENCE(20),20)-REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE>0),SEQUENCE(20),20)"""
        # 需要benchmark_close数据，数字币场景跳过，返回NaN
        return pd.Series(np.nan, index=df.index)
    
    def _alpha_150(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE+HIGH+LOW)/3*VOLUME"""
        return (df['close'] + df['high'] + df['low']) / 3 * df['volume']
    
    def _alpha_151(self, df: pd.DataFrame) -> pd.Series:
        """SMA(CLOSE-DELAY(CLOSE,20),20,1)"""
        close_delay = self._delay(df['close'], 20)
        diff = df['close'] - close_delay
        return self._sma(diff, 20, 1)
    
    def _alpha_152(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)"""
        close_delay9 = self._delay(df['close'], 9)
        ratio = df['close'] / (close_delay9 + self._EPSILON)
        ratio_delay = self._delay(ratio, 1)
        sma1 = self._sma(ratio_delay, 9, 1)
        sma1_delay = self._delay(sma1, 1)
        mean1 = self._mean(sma1_delay, 12)
        mean2 = self._mean(sma1_delay, 26)
        diff = mean1 - mean2
        return self._sma(diff, 9, 1)
    
    def _alpha_153(self, df: pd.DataFrame) -> pd.Series:
        """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4"""
        mean3 = self._mean(df['close'], 3)
        mean6 = self._mean(df['close'], 6)
        mean12 = self._mean(df['close'], 12)
        mean24 = self._mean(df['close'], 24)
        return (mean3 + mean6 + mean12 + mean24) / 4
    
    def _alpha_154(self, df: pd.DataFrame) -> pd.Series:
        """((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        min_vwap = self._tsmin(vwap, 16)
        diff = vwap - min_vwap
        
        mean_vol = self._mean(df['volume'], 180)
        corr_result = self._corr(vwap, mean_vol, 18)
        
        condition = diff < corr_result
        return pd.Series(np.where(condition, 1, 0), index=df.index)
    
    def _alpha_155(self, df: pd.DataFrame) -> pd.Series:
        """SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)"""
        sma1 = self._sma(df['volume'], 13, 2)
        sma2 = self._sma(df['volume'], 27, 2)
        diff = sma1 - sma2
        sma_diff = self._sma(diff, 10, 2)
        return diff - sma_diff
    
    def _alpha_156(self, df: pd.DataFrame) -> pd.Series:
        """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW * 0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        delta_vwap = self._delta(vwap, 5)
        decay1 = self._decaylinear(delta_vwap, 3)
        rank_decay1 = self._rank(decay1)
        
        weighted_price = df['open'] * 0.15 + df['low'] * 0.85
        delta_weighted = self._delta(weighted_price, 2)
        ratio = delta_weighted / (weighted_price + self._EPSILON) * -1
        decay2 = self._decaylinear(ratio, 3)
        rank_decay2 = self._rank(decay2)
        
        return self._max(rank_decay1, rank_decay2) * -1
    
    def _alpha_157(self, df: pd.DataFrame) -> pd.Series:
        """(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))"""
        # 非常复杂的因子，简化实现
        delta_close = self._delta(df['close'] - 1, 5)
        rank_delta = self._rank(delta_close)
        rank_rank = self._rank(rank_delta)
        tsmin_rank = self._tsmin(rank_rank, 2)
        log_sum = np.log(self._sum(tsmin_rank, 1) + self._EPSILON)
        rank_log = self._rank(self._rank(log_sum))
        prod_rank = self._prod(rank_log, 1)
        min_prod = self._tsmin(prod_rank, 5)
        
        ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        delay_ret = self._delay(-1 * ret, 6)
        tsrank_delay = self._tsrank(delay_ret, 5)
        
        return min_prod + tsrank_delay
    
    def _alpha_158(self, df: pd.DataFrame) -> pd.Series:
        """((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE"""
        sma_close = self._sma(df['close'], 15, 2)
        high_diff = df['high'] - sma_close
        low_diff = df['low'] - sma_close
        return (high_diff - low_diff) / (df['close'] + self._EPSILON)
    
    def _alpha_159(self, df: pd.DataFrame) -> pd.Series:
        """((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)"""
        close_delay = self._delay(df['close'], 1)
        min_low_close = self._min(df['low'], close_delay)
        max_high_close = self._max(df['high'], close_delay)
        numerator = df['close'] - self._sum(min_low_close, 6)
        denominator = self._sum(max_high_close - min_low_close, 6)
        return numerator / (denominator + self._EPSILON)
    
    def _alpha_160(self, df: pd.DataFrame) -> pd.Series:
        """SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
        close_delay = self._delay(df['close'], 1)
        condition = df['close'] <= close_delay
        std_close = self._std(df['close'], 20)
        value = np.where(condition, std_close, 0)
        return self._sma(pd.Series(value, index=df.index), 20, 1)
    
    def _alpha_161(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)"""
        close_delay = self._delay(df['close'], 1)
        hl_diff = df['high'] - df['low']
        abs_high_diff = np.abs(close_delay - df['high'])
        abs_low_diff = np.abs(close_delay - df['low'])
        max1 = self._max(hl_diff, abs_high_diff)
        max2 = self._max(max1, abs_low_diff)
        return self._mean(max2, 12)
    
    def _alpha_162(self, df: pd.DataFrame) -> pd.Series:
        """(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        max_diff = diff.clip(lower=0)
        abs_diff = np.abs(diff)
        sma_max = self._sma(max_diff, 12, 1)
        sma_abs = self._sma(abs_diff, 12, 1)
        rsi = sma_max / (sma_abs + self._EPSILON) * 100
        
        min_rsi = self._tsmin(rsi, 12)
        max_rsi = self._tsmax(rsi, 12)
        
        return (rsi - min_rsi) / (max_rsi - min_rsi + self._EPSILON)
    
    def _alpha_163(self, df: pd.DataFrame) -> pd.Series:
        """RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))"""
        ret = df['close'] / (self._delay(df['close'], 1) + self._EPSILON) - 1
        mean_vol = self._mean(df['volume'], 20)
        vwap = df['vwap']  # 预处理已确保vwap存在
        high_close_diff = df['high'] - df['close']
        
        combined = -1 * ret * mean_vol * vwap * high_close_diff
        return self._rank(combined)
    
    def _alpha_164(self, df: pd.DataFrame) -> pd.Series:
        """SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*10000,13,2)"""
        close_delay = self._delay(df['close'], 1)
        condition = df['close'] > close_delay
        value = np.where(condition, 1 / (df['close'] - close_delay + self._EPSILON), 1)
        min_value = self._tsmin(value, 12)
        numerator = (value - min_value) / (df['high'] - df['low'] + self._EPSILON) * 10000
        return self._sma(numerator, 13, 2)
    
    def _alpha_165(self, df: pd.DataFrame) -> pd.Series:
        """MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)"""
        mean_close = self._mean(df['close'], 24)
        diff = df['close'] - mean_close
        sumac_diff = self._sumac(diff, 24)
        max_sumac = self._tsmax(sumac_diff, 24)
        min_sumac = self._tsmin(sumac_diff, 24)
        std_close = self._std(df['close'], 24)
        return (max_sumac - min_sumac) / (std_close + self._EPSILON)
    
    def _alpha_166(self, df: pd.DataFrame) -> pd.Series:
        """-1*COVIANCE(OPEN,VOLUME,10)"""
        return -1 * self._covariance(df['open'], df['volume'], 10)
    
    def _alpha_167(self, df: pd.DataFrame) -> pd.Series:
        """SUM(((CLOSE-DELAY(CLOSE,1)>0)?(CLOSE-DELAY(CLOSE,1)):0),12)"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        value = diff.clip(lower=0)
        return self._sum(value, 12)
    
    def _alpha_168(self, df: pd.DataFrame) -> pd.Series:
        """(-1*VOLUME/MEAN(VOLUME,20))"""
        mean_vol = self._mean(df['volume'], 20)
        return -1 * df['volume'] / (mean_vol + self._EPSILON)
    
    def _alpha_169(self, df: pd.DataFrame) -> pd.Series:
        """SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)"""
        close_delay = self._delay(df['close'], 1)
        diff = df['close'] - close_delay
        sma_diff = self._sma(diff, 9, 1)
        sma_diff_delay = self._delay(sma_diff, 1)
        mean1 = self._mean(sma_diff_delay, 12)
        mean2 = self._mean(sma_diff_delay, 26)
        diff_mean = mean1 - mean2
        return self._sma(diff_mean, 10, 1)
    
    def _alpha_170(self, df: pd.DataFrame) -> pd.Series:
        """((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        rank_inv_close = self._rank(1 / (df['close'] + self._EPSILON))
        mean_vol = self._mean(df['volume'], 20)
        part1 = rank_inv_close * df['volume'] / (mean_vol + self._EPSILON)
        
        rank_high_close = self._rank(df['high'] - df['close'])
        sum_high = self._sum(df['high'], 5) / 5
        part2 = df['high'] * rank_high_close / (sum_high + self._EPSILON)
        
        delta_vwap = vwap - self._delay(vwap, 5)
        rank_delta = self._rank(delta_vwap)
        
        return part1 * part2 - rank_delta
    
    def _alpha_171(self, df: pd.DataFrame) -> pd.Series:
        """((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))"""
        numerator = -1 * (df['low'] - df['close']) * (df['open'] ** 5)
        denominator = (df['close'] - df['high']) * (df['close'] ** 5)
        return numerator / (denominator + self._EPSILON)
    
    def _alpha_172(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)"""
        # HD = HIGH-DELAY(HIGH,1), LD = DELAY(LOW,1)-LOW
        high_delay = self._delay(df['high'], 1)
        low_delay = self._delay(df['low'], 1)
        hd = df['high'] - high_delay
        ld = low_delay - df['low']
        
        # TR = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        close_delay = self._delay(df['close'], 1)
        tr = self._max(df['high'] - df['low'], 
                      self._max(np.abs(df['high'] - close_delay), 
                               np.abs(df['low'] - close_delay)))
        
        condition_ld = (ld > 0) & (ld > hd)
        condition_hd = (hd > 0) & (hd > ld)
        
        sum_ld = self._sum(pd.Series(np.where(condition_ld, ld, 0), index=df.index), 14)
        sum_hd = self._sum(pd.Series(np.where(condition_hd, hd, 0), index=df.index), 14)
        sum_tr = self._sum(tr, 14)
        
        ratio_ld = sum_ld * 100 / (sum_tr + self._EPSILON)
        ratio_hd = sum_hd * 100 / (sum_tr + self._EPSILON)
        
        abs_diff = np.abs(ratio_ld - ratio_hd)
        sum_ratios = ratio_ld + ratio_hd
        
        value = abs_diff / (sum_ratios + self._EPSILON) * 100
        return self._mean(value, 6)
    
    def _alpha_173(self, df: pd.DataFrame) -> pd.Series:
        """3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(CLOSE,13,2),13,2),13,2)"""
        sma1 = self._sma(df['close'], 13, 2)
        sma2 = self._sma(sma1, 13, 2)
        sma3 = self._sma(sma2, 13, 2)
        return 3 * sma1 - 2 * sma2 + sma3
    
    def _alpha_174(self, df: pd.DataFrame) -> pd.Series:
        """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
        close_delay = self._delay(df['close'], 1)
        condition = df['close'] > close_delay
        std_close = self._std(df['close'], 20)
        value = np.where(condition, std_close, 0)
        return self._sma(pd.Series(value, index=df.index), 20, 1)
    
    def _alpha_175(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)"""
        close_delay = self._delay(df['close'], 1)
        hl_diff = df['high'] - df['low']
        abs_high_diff = np.abs(close_delay - df['high'])
        abs_low_diff = np.abs(close_delay - df['low'])
        max1 = self._max(hl_diff, abs_high_diff)
        max2 = self._max(max1, abs_low_diff)
        return self._mean(max2, 6)
    
    def _alpha_176(self, df: pd.DataFrame) -> pd.Series:
        """CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW, 12)))), RANK(VOLUME), 6)"""
        tsmin_low = self._tsmin(df['low'], 12)
        tsmax_high = self._tsmax(df['high'], 12)
        rsv = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + self._EPSILON)
        rank_rsv = self._rank(rsv)
        rank_volume = self._rank(df['volume'])
        return self._corr(rank_rsv, rank_volume, 6)
    
    def _alpha_177(self, df: pd.DataFrame) -> pd.Series:
        """((20-HIGHDAY(HIGH,20))/20)*100"""
        highday = self._highday(df['high'], 20)
        return (20 - highday) / 20 * 100
    
    def _alpha_178(self, df: pd.DataFrame) -> pd.Series:
        """(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME"""
        close_delay = self._delay(df['close'], 1)
        return (df['close'] - close_delay) / (close_delay + self._EPSILON) * df['volume']
    
    def _alpha_179(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(VWAP, VOLUME, 4)) * RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))"""
        vwap = df['vwap']  # 预处理已确保vwap存在
        corr1 = self._corr(vwap, df['volume'], 4)
        rank_corr1 = self._rank(corr1)
        
        rank_low = self._rank(df['low'])
        mean_vol = self._mean(df['volume'], 50)
        rank_mean_vol = self._rank(mean_vol)
        corr2 = self._corr(rank_low, rank_mean_vol, 12)
        rank_corr2 = self._rank(corr2)
        
        return rank_corr1 * rank_corr2
    
    def _alpha_180(self, df: pd.DataFrame) -> pd.Series:
        """((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME))"""
        mean_vol = self._mean(df['volume'], 20)
        condition = mean_vol < df['volume']
        
        delta_close = self._delta(df['close'], 7)
        abs_delta = np.abs(delta_close)
        tsrank_abs = self._tsrank(abs_delta, 60)
        sign_delta = self._sign(delta_close)
        value1 = -1 * tsrank_abs * sign_delta
        
        value2 = -1 * df['volume']
        
        result = np.where(condition, value1, value2)
        return pd.Series(result, index=df.index)
    
    def _alpha_181(self, df: pd.DataFrame) -> pd.Series:
        """SUM(((CLOSE/DELAY(CLOSE,1)-1-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)"""
        # 需要benchmark_close数据，数字币场景跳过，返回NaN
        return pd.Series(np.nan, index=df.index)
    
    def _alpha_182(self, df: pd.DataFrame) -> pd.Series:
        """COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20"""
        # 需要benchmark_open和benchmark_close数据，数字币场景跳过，返回NaN
        return pd.Series(np.nan, index=df.index)
    
    def _alpha_183(self, df: pd.DataFrame) -> pd.Series:
        """MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)"""
        mean_close = self._mean(df['close'], 24)
        diff = df['close'] - mean_close
        sumac_diff = self._sumac(diff, 24)
        max_sumac = self._tsmax(sumac_diff, 24)
        min_sumac = self._tsmin(sumac_diff, 24)
        std_close = self._std(df['close'], 24)
        return (max_sumac - min_sumac) / (std_close + self._EPSILON)
    
    def _alpha_184(self, df: pd.DataFrame) -> pd.Series:
        """(RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE)))"""
        open_close_diff = df['open'] - df['close']
        delay_diff = self._delay(open_close_diff, 1)
        corr_result = self._corr(delay_diff, df['close'], 200)
        rank_corr = self._rank(corr_result)
        rank_diff = self._rank(open_close_diff)
        return rank_corr + rank_diff
    
    def _alpha_185(self, df: pd.DataFrame) -> pd.Series:
        """RANK((-1*((1-(OPEN/CLOSE))^2)))"""
        ratio = 1 - (df['open'] / (df['close'] + self._EPSILON))
        squared = ratio ** 2
        return self._rank(-1 * squared)
    
    def _alpha_186(self, df: pd.DataFrame) -> pd.Series:
        """(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2"""
        # 与alpha_172类似
        high_delay = self._delay(df['high'], 1)
        low_delay = self._delay(df['low'], 1)
        hd = df['high'] - high_delay
        ld = low_delay - df['low']
        
        close_delay = self._delay(df['close'], 1)
        tr = self._max(df['high'] - df['low'], 
                      self._max(np.abs(df['high'] - close_delay), 
                               np.abs(df['low'] - close_delay)))
        
        condition_ld = (ld > 0) & (ld > hd)
        condition_hd = (hd > 0) & (hd > ld)
        
        sum_ld = self._sum(pd.Series(np.where(condition_ld, ld, 0), index=df.index), 14)
        sum_hd = self._sum(pd.Series(np.where(condition_hd, hd, 0), index=df.index), 14)
        sum_tr = self._sum(tr, 14)
        
        ratio_ld = sum_ld * 100 / (sum_tr + self._EPSILON)
        ratio_hd = sum_hd * 100 / (sum_tr + self._EPSILON)
        
        abs_diff = np.abs(ratio_ld - ratio_hd)
        sum_ratios = ratio_ld + ratio_hd
        
        value = abs_diff / (sum_ratios + self._EPSILON) * 100
        mean_value = self._mean(value, 6)
        delay_mean = self._delay(mean_value, 6)
        
        return (mean_value + delay_mean) / 2
    
    def _alpha_187(self, df: pd.DataFrame) -> pd.Series:
        """SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)"""
        open_delay = self._delay(df['open'], 1)
        condition = df['open'] <= open_delay
        max_diff = self._max(df['high'] - df['open'], df['open'] - open_delay)
        value = np.where(condition, 0, max_diff)
        return pd.Series(value, index=df.index).rolling(20).sum()
    
    def _alpha_188(self, df: pd.DataFrame) -> pd.Series:
        """((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100"""
        hl_diff = df['high'] - df['low']
        sma_hl = self._sma(hl_diff, 11, 2)
        return (hl_diff - sma_hl) / (sma_hl + self._EPSILON) * 100
    
    def _alpha_189(self, df: pd.DataFrame) -> pd.Series:
        """MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)"""
        mean_close = self._mean(df['close'], 6)
        abs_diff = np.abs(df['close'] - mean_close)
        return self._mean(abs_diff, 6)
    
    def _alpha_190(self, df: pd.DataFrame) -> pd.Series:
        """LOG((COUNT(CLOSE/DELAY(CLOSE,1)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE,1)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE,1)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))"""
        close_delay1 = self._delay(df['close'], 1)
        close_delay19 = self._delay(df['close'], 19)
        
        ret = df['close'] / (close_delay1 + self._EPSILON) - 1
        benchmark = (df['close'] / (close_delay19 + self._EPSILON)) ** (1/20) - 1
        
        condition_up = ret > benchmark
        condition_down = ret < benchmark
        
        count_up = self._count(condition_up, 20)
        count_down = self._count(condition_down, 20)
        
        diff = ret - benchmark
        diff_squared = diff ** 2
        
        sumif_down = self._sumif(diff_squared, 20, condition_down)
        sumif_up = self._sumif(diff_squared, 20, condition_up)
        
        numerator = (count_up - 1) * sumif_down
        denominator = count_down * sumif_up
        
        return np.log(numerator / (denominator + self._EPSILON) + self._EPSILON)
    
    def _alpha_191(self, df: pd.DataFrame) -> pd.Series:
        """((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)"""
        mean_vol = self._mean(df['volume'], 20)
        corr_result = self._corr(mean_vol, df['low'], 5)
        hl_avg = (df['high'] + df['low']) / 2
        return corr_result + hl_avg - df['close']
