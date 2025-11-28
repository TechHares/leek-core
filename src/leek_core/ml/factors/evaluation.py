#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子评价模块

用于计算因子的IC、IR等评价指标，分析因子有效性。

使用成熟的统计库（scipy.stats, pandas）进行计算，确保计算结果的准确性和稳定性。
参考了 alphalens 等成熟因子分析库的实现思路。
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew, spearmanr

from leek_core.utils import get_logger

logger = get_logger(__name__)


class FactorEvaluator:
    """
    因子评价器
    
    用于计算因子的IC、IR等评价指标，分析因子有效性。
    """
    
    def __init__(self, future_periods: int = 1, quantile_count: int = 5):
        """
        初始化因子评价器
        
        Args:
            future_periods: 未来收益期数（用于计算未来收益）
            quantile_count: 分位数数量（用于收益分析）
        """
        self.future_periods = future_periods
        self.quantile_count = quantile_count
    
    def _calculate_future_return(self, df: pd.DataFrame) -> pd.Series:
        """
        计算未来收益率
        
        Args:
            df: 包含 close 列的 DataFrame
        """
        if "future_return" in df.columns:
            return df['future_return']
        if 'close' not in df.columns:
            raise ValueError("DataFrame 中必须包含 close 列")
        close = df['close']
        future_close = close.shift(-self.future_periods)
        df['future_return'] = (future_close - close) / close
        return df['future_return']
    
    def _calculate_ic(self, data: pd.DataFrame, factor_name: str) -> float:
        """
        计算Rank IC（使用Spearman相关系数）
        Args:
            data: 数据
            factor_name: 因子名称
        
        Returns:
            IC值（Spearman相关系数）
        """
        correlation, _ = spearmanr(data[factor_name], data['future_return'], nan_policy='omit')
        return float(correlation) if not np.isnan(correlation) else np.nan
    
    def _calculate_cumulative_ic(
        self, 
        data: pd.DataFrame, 
        factor_name: str,
        min_periods: int
    ) -> pd.Series:
        """计算累积IC（使用所有历史数据）"""
        # 预过滤NaN：预先过滤掉无效数据，避免循环中重复检查
        valid_mask = ~(data[factor_name].isna() | data['future_return'].isna())
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < min_periods:
            return pd.Series(index=data.index, dtype=float)
        
        # 使用预过滤后的数据计算秩，避免NaN干扰
        factor_values = data[factor_name].values[valid_indices]
        return_values = data['future_return'].values[valid_indices]
        
        # 预先计算秩（只对有效数据）
        factor_ranks = pd.Series(factor_values).rank(method='average').values
        return_ranks = pd.Series(return_values).rank(method='average').values
        
        ic_series = pd.Series(index=data.index, dtype=float)
        
        # 使用预计算的秩进行累积相关系数计算
        # Spearman相关系数是秩的Pearson相关系数
        # 使用numpy的corrcoef替代scipy的pearsonr，性能更好
        for i in range(min_periods, len(valid_indices) + 1):
            # 获取当前窗口的秩（已经是有效数据）
            window_factor_ranks = factor_ranks[:i]
            window_return_ranks = return_ranks[:i]
            
            if len(window_factor_ranks) < min_periods:
                continue
            
            # 使用numpy的corrcoef计算相关系数（比scipy的pearsonr快）
            # corrcoef返回2x2矩阵，[0,1]或[1,0]位置是相关系数
            corr_matrix = np.corrcoef(window_factor_ranks, window_return_ranks)
            correlation = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else np.nan
            
            # 找到对应的原始索引
            idx = data.index[valid_indices[i-1]]
            ic_series[idx] = float(correlation) if not np.isnan(correlation) else np.nan
        
        return ic_series.dropna()
    
    def _calculate_rolling_ic_fixed_window(
        self, 
        data: pd.DataFrame, 
        factor_name: str,
        window_size: int,
        min_periods: int,
        return_ranks: pd.Series | None = None
    ) -> pd.Series:
        """使用固定窗口大小计算滚动IC"""
        # 预先计算秩，提高性能
        factor_ranks = data[factor_name].rank(method='average')
        # 如果提供了预计算的return_ranks，使用它；否则重新计算
        if return_ranks is None:
            return_ranks = data['future_return'].rank(method='average')
        else:
            # 确保return_ranks的索引与data对齐
            return_ranks = return_ranks.reindex(data.index, method=None)
        
        # 使用rolling.corr计算秩的相关系数（即Spearman IC）
        # 这是向量化操作，性能远优于apply
        ic_series = factor_ranks.rolling(
            window=window_size,
            min_periods=min_periods
        ).corr(return_ranks)
        
        return ic_series.rename('ic')
    
    def _calculate_ic_series(
        self, 
        data: pd.DataFrame, 
        factor_name: str,
        window: int |None = None,
        min_periods: int = 10,
        return_ranks: pd.Series | None = None
    ) -> pd.Series:
        """
        优化版：计算IC时序序列
        
        Args:
            data: 数据，必须包含 time, factor_name, future_return 列
            factor_name: 因子名称
            window: 滚动窗口大小
                - None: 逐日计算（使用所有历史数据）
                - int: 固定窗口大小（如60）
            min_periods: 计算IC所需的最小样本数
            return_ranks: 预计算的future_return的rank（可选，用于批量计算优化）
            
        Returns:
            IC时序序列，索引为时间
        """
        # 验证必要的列是否存在
        if factor_name not in data.columns:
            raise ValueError(f"数据中不存在因子列: {factor_name}")
        if 'future_return' not in data.columns:
            raise ValueError("数据中不存在 future_return 列，请先调用 _calculate_future_return")
        
        # 处理数据边界：移除最后 future_periods 行（未来收益为NaN）
        # 使用视图而非copy，避免不必要的内存复制
        if self.future_periods > 0:
            valid_end_idx = len(data) - self.future_periods
            if valid_end_idx <= 0:
                return pd.Series(index=data.index, dtype=float)
            # 使用视图，不复制数据
            valid_data = data.iloc[:valid_end_idx]
            # 如果提供了预计算的return_ranks，说明它已经处理过边界，不需要再次切片
        else:
            valid_data = data
        
        if len(valid_data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        # 情况1：无窗口（使用所有历史数据计算逐日IC）
        if window is None:
            return self._calculate_cumulative_ic(valid_data, factor_name, min_periods)
        
        # 情况2：固定窗口大小（整数）
        if isinstance(window, int):
            return self._calculate_rolling_ic_fixed_window(
                valid_data, factor_name, window, min_periods, return_ranks
            )
        raise ValueError(f"不支持的 window 类型: {type(window)}")
    
    def _calculate_ir(self, ic_series: pd.Series) -> float:
        """
        计算IR (Information Ratio) = IC均值 / IC标准差
        
        Args:
            ic_series: IC时序序列
            
        Returns:
            IR值
        """
        ic_series_clean = ic_series.dropna()
        if len(ic_series_clean) == 0:
            return np.nan
        
        ic_mean = ic_series_clean.mean()
        ic_std = ic_series_clean.std()
        
        if ic_std == 0 or np.isnan(ic_std):
            return np.nan
        
        return float(ic_mean / ic_std)
    
    def _calculate_ic_win_rate(self, ic_series: pd.Series) -> float:
        """
        计算IC胜率（IC>0的时间点占比）
        
        Args:
            ic_series: IC时序序列
            
        Returns:
            IC胜率
        """
        ic_series_clean = ic_series.dropna()
        if len(ic_series_clean) == 0:
            return 0.0
        
        positive_count = (ic_series_clean > 0).sum()
        return float(positive_count / len(ic_series_clean))
    
    def _calculate_ic_skewness(self, ic_series: pd.Series) -> float:
        """
        计算IC偏度
        
        Args:
            ic_series: IC时序序列
            
        Returns:
            IC偏度
        """
        ic_series_clean = ic_series.dropna()
        if len(ic_series_clean) < 3:
            return 0.0
        
        try:
            return float(skew(ic_series_clean))
        except Exception:
            return 0.0
    
    def _calculate_factor_returns(
        self, 
        data: pd.DataFrame,
        factor_name: str,
    ) -> Dict[str, float]:
        """
        按因子分位数计算收益分布
        
        Args:
            data: DataFrame（包含 factor_name 和 'future_return' 列）或因子值序列
            factor_name: 因子列名（当 data 是 DataFrame 时必需）
            future_returns: 未来收益率序列（当 data 是 Series 时必需）
            
        Returns:
            分位数收益字典，key为分位数名称，value为平均收益
        """
        # 支持两种调用方式：
        # 1. data 是 DataFrame，factor_name 指定因子列名
        # 2. data 是 Series（因子值），future_returns 是 Series（未来收益）
        aligned_data = data[[factor_name, 'future_return']].dropna()
        if len(aligned_data) == 0:
            return {}
        
        # 计算分位数
        quantiles = pd.qcut(
            aligned_data[factor_name], 
            q=self.quantile_count, 
            labels=False, 
            duplicates='drop'
        )
        
        quantile_returns = {}
        for q in range(self.quantile_count):
            mask = quantiles == q
            if mask.sum() > 0:
                avg_return = aligned_data.loc[mask, 'future_return'].mean()
                quantile_returns[f'Q{q+1}'] = float(avg_return)
        
        return quantile_returns
    
    def _calculate_long_short_return(
        self, 
        data: pd.DataFrame | pd.Series,
        factor_name: str | None = None,
        future_returns: pd.Series | None = None
    ) -> float:
        """
        计算多空收益（最高分位数 - 最低分位数）
        
        Args:
            data: DataFrame（包含 factor_name 和 'future_return' 列）或因子值序列
            factor_name: 因子列名（当 data 是 DataFrame 时必需）
            future_returns: 未来收益率序列（当 data 是 Series 时必需）
            
        Returns:
            多空收益
        """
        quantile_returns = self._calculate_factor_returns(data, factor_name)
        
        if len(quantile_returns) < 2:
            return 0.0
        
        # 获取最高和最低分位数的收益
        returns_list = list(quantile_returns.values())
        max_return = max(returns_list)
        min_return = min(returns_list)
        
        return float(max_return - min_return)
    
    def _calculate_factor_correlation(
        self, 
        factor_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算多因子相关性矩阵
        
        Args:
            factor_data: 因子数据字典，key为因子名称，value为因子值序列
            
        Returns:
            相关性矩阵DataFrame
        """
        # 计算相关性矩阵
        correlation_matrix = factor_data.corr(method='pearson')
        return correlation_matrix
    
    def evaluate_factor(
        self,
        data: pd.DataFrame,
        factor_name: str,
        ic_window: Optional[int] = None,
        return_ranks: pd.Series | None = None
    ) -> Dict[str, Any]:
        """
        评价单个因子
        
        Args:
            data: 因子数据DataFrame，必须包含start_time列（毫秒时间戳）
            factor_name: 因子名称
            ic_window: IC计算窗口大小（如果为None，则逐日计算）
            return_ranks: 预计算的future_return的rank（可选，用于批量计算优化）
            
        Returns:
            评价结果字典，包含ic_series和ic_times（时间戳列表）
        """
        self._calculate_future_return(data)
        # 计算IC时序序列
        ic_series = self._calculate_ic_series(data, factor_name, ic_window, return_ranks=return_ranks)
        
        # 提取时间戳：根据IC序列的index从DataFrame的start_time列获取对应的时间戳
        ic_times = []
        if 'start_time' not in data.columns:
            raise ValueError("DataFrame必须包含start_time列")
        
        for idx in ic_series.index:
            if idx in data.index:
                start_time = data.loc[idx, 'start_time']
                if pd.notna(start_time):
                    ic_times.append(int(start_time))
        
        # 保持ic_series和ic_times一一对应
        ic_values = []
        ic_times_filtered = []
        for ic_val, ic_time in zip(ic_series.tolist(), ic_times):
            if ic_time is not None and pd.notna(ic_val):
                ic_values.append(ic_val)
                ic_times_filtered.append(ic_time)
        
        # 计算各项指标（使用过滤后的IC序列）
        ic_series_filtered = pd.Series(ic_values)
        ic_mean = float(ic_series_filtered.mean()) if len(ic_series_filtered) > 0 else 0.0
        ic_std = float(ic_series_filtered.std()) if len(ic_series_filtered) > 0 else 0.0
        ir = self._calculate_ir(ic_series_filtered)
        ic_win_rate = self._calculate_ic_win_rate(ic_series_filtered)
        ic_skewness = self._calculate_ic_skewness(ic_series_filtered)
        
        # 计算收益分析（直接传入 DataFrame，避免不必要的拆分和合并）
        quantile_returns = self._calculate_factor_returns(data, factor_name)
        long_short_return = self._calculate_long_short_return(data, factor_name)
        
        return {
            'factor_name': factor_name,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_win_rate': ic_win_rate,
            'ic_skewness': ic_skewness,
            'ic_series': ic_values,  # 过滤后的IC值列表
            'ic_times': ic_times_filtered,  # 对应的时间戳列表（毫秒）
            'quantile_returns': quantile_returns,
            'long_short_return': long_short_return,
        }
    
    def evaluate_factors(
        self,
        factor_data: pd.DataFrame,
        factor_names: List[str],
        ic_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        评价多个因子
        
        Args:
            factor_data: 因子数据DataFrame，包含所有因子列和'future_return'列
            factor_names: 因子名称列表
            ic_window: IC计算窗口大小（如果为None，则逐日计算）
            
        Returns:
            评价结果字典，包含每个因子的评价结果和因子相关性矩阵
        """
        if ic_window is not None and ic_window <= 0:
            ic_window = None
        # 批量优化：预先计算future_return（所有因子共享）
        self._calculate_future_return(factor_data)
        
        # 批量优化：如果使用固定窗口，预先计算future_return的rank（所有因子共享）
        # 这样可以避免在每个因子计算时重复计算
        shared_return_ranks = None
        if ic_window is not None and isinstance(ic_window, int):
            # 处理数据边界
            if self.future_periods > 0:
                valid_end_idx = len(factor_data) - self.future_periods
                if valid_end_idx > 0:
                    shared_return_ranks = factor_data['future_return'].iloc[:valid_end_idx].rank(method='average')
            else:
                shared_return_ranks = factor_data['future_return'].rank(method='average')
        
        results = {}
        
        # 评价每个因子
        for factor_name in factor_names:
            try:
                # 传递预计算的return_ranks，避免重复计算
                result = self.evaluate_factor(
                    factor_data, 
                    factor_name, 
                    ic_window,
                    return_ranks=shared_return_ranks
                )
                results[factor_name] = result
            except Exception as e:
                logger.error(f"评价因子 {factor_name} 失败: {e}")
                continue
        
        # 计算因子相关性矩阵
        correlation_matrix = self._calculate_factor_correlation(factor_data[factor_names])
        
        return {
            'factor_results': results,
            'correlation_matrix': correlation_matrix.to_dict(),
        }

