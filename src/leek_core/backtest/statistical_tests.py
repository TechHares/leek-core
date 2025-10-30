#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计检验模块

提供统计显著性检验功能，帮助判断策略是否真的有效而非随机运气。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats


def t_test_returns(returns: np.ndarray) -> Tuple[float, float]:
    """
    单样本t检验：检验策略收益率是否显著大于0
    
    Args:
        returns: 收益率序列
        
    Returns:
        (t_statistic, p_value)
    """
    if len(returns) < 2:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)
    # 单边检验：检验是否显著大于0
    if t_stat > 0:
        p_value = p_value / 2.0  # 单边检验p值是双边的一半
    else:
        p_value = 1.0 - p_value / 2.0
    
    return float(t_stat), float(p_value)


def paired_t_test(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
    """
    配对t检验：检验策略是否显著优于基准
    
    Args:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        (t_statistic, p_value)
    """
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
        return 0.0, 1.0
    
    diff = strategy_returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(diff, 0.0)
    # 单边检验：检验策略是否显著优于基准
    if t_stat > 0:
        p_value = p_value / 2.0
    else:
        p_value = 1.0 - p_value / 2.0
    
    return float(t_stat), float(p_value)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Bootstrap置信区间估计
    
    Args:
        data: 原始数据
        statistic_func: 统计量计算函数（如np.mean, np.std等）
        n_bootstrap: Bootstrap重采样次数
        confidence_level: 置信水平（默认0.95）
        
    Returns:
        (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return 0.0, 0.0
    
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        stat_value = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat_value)
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    
    return lower, upper


def win_rate_binomial_test(win_trades: int, total_trades: int) -> float:
    """
    胜率二项检验：检验胜率是否显著大于50%
    
    Args:
        win_trades: 盈利交易数
        total_trades: 总交易数
        
    Returns:
        p_value
    """
    if total_trades < 1:
        return 1.0
    
    # 二项分布检验：H0: p=0.5, H1: p>0.5
    p_value = stats.binomtest(win_trades, total_trades, p=0.5, alternative='greater').pvalue
    return float(p_value)


def regression_analysis(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> Dict[str, float]:
    """
    回归分析：计算alpha、beta、R²，并检验alpha是否显著大于0
    
    Args:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        {
            'alpha': float,
            'beta': float,
            'r_squared': float,
            'alpha_pvalue': float
        }
    """
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
        return {
            'alpha': 0.0,
            'beta': 0.0,
            'r_squared': 0.0,
            'alpha_pvalue': 1.0
        }
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        benchmark_returns, strategy_returns
    )
    
    # alpha = intercept（截距）
    # beta = slope（斜率）
    # R² = r_value²
    alpha = float(intercept)
    beta = float(slope)
    r_squared = float(r_value ** 2)
    
    # 检验alpha是否显著大于0
    # 使用t检验：t = alpha / std_err
    if std_err > 0:
        t_alpha = intercept / std_err
        # 单边检验p值
        alpha_pvalue = 1.0 - stats.t.cdf(t_alpha, len(strategy_returns) - 2)
        if alpha <= 0:
            alpha_pvalue = 1.0 - alpha_pvalue
    else:
        alpha_pvalue = 1.0
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'alpha_pvalue': float(alpha_pvalue)
    }


def calculate_statistical_tests(
    equity_curve: List[float],
    benchmark_curve: Optional[List[float]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
    n_bootstrap: int = 1000
) -> Dict[str, float]:
    """
    计算所有统计检验结果
    
    Args:
        equity_curve: 净值曲线
        benchmark_curve: 基准曲线（可选）
        trades: 交易记录（可选，用于胜率检验）
        n_bootstrap: Bootstrap重采样次数
        
    Returns:
        包含所有统计检验结果的字典
    """
    result = {
        't_statistic': 0.0,
        't_pvalue': 1.0,
        'paired_t_statistic': 0.0,
        'paired_t_pvalue': 1.0,
        'bootstrap_sharpe_ci_lower': 0.0,
        'bootstrap_sharpe_ci_upper': 0.0,
        'bootstrap_annual_return_ci_lower': 0.0,
        'bootstrap_annual_return_ci_upper': 0.0,
        'win_rate_pvalue': 1.0,
        'alpha_pvalue': 1.0,
    }
    
    if len(equity_curve) < 2:
        return result
    
    # 计算收益率序列
    equity_array = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # 单样本t检验
    t_stat, t_pvalue = t_test_returns(returns)
    result['t_statistic'] = t_stat
    result['t_pvalue'] = t_pvalue
    
    # 如果有基准曲线，进行配对t检验和回归分析
    if benchmark_curve and len(benchmark_curve) >= 2:
        benchmark_array = np.array(benchmark_curve, dtype=np.float64)
        benchmark_returns = np.diff(benchmark_array) / benchmark_array[:-1]
        
        # 对齐长度
        min_len = min(len(returns), len(benchmark_returns))
        returns_aligned = returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 配对t检验
        paired_t_stat, paired_t_pvalue = paired_t_test(returns_aligned, benchmark_returns_aligned)
        result['paired_t_statistic'] = paired_t_stat
        result['paired_t_pvalue'] = paired_t_pvalue
        
        # 回归分析
        reg_result = regression_analysis(returns_aligned, benchmark_returns_aligned)
        result['alpha_pvalue'] = reg_result['alpha_pvalue']
    
    # Bootstrap置信区间
    # Sharpe比率
    def sharpe_statistic(returns_sample: np.ndarray) -> float:
        if len(returns_sample) < 2 or np.std(returns_sample) == 0:
            return 0.0
        return np.mean(returns_sample) / np.std(returns_sample)
    
    sharpe_lower, sharpe_upper = bootstrap_confidence_interval(
        returns, sharpe_statistic, n_bootstrap=n_bootstrap
    )
    result['bootstrap_sharpe_ci_lower'] = sharpe_lower
    result['bootstrap_sharpe_ci_upper'] = sharpe_upper
    
    # 年化收益率
    def annual_return_statistic(returns_sample: np.ndarray) -> float:
        if len(returns_sample) < 1:
            return 0.0
        total_return = np.prod(1 + returns_sample) - 1.0
        # 假设252个交易日
        periods = len(returns_sample) / 252.0 if len(returns_sample) > 0 else 1.0
        return (1.0 + total_return) ** (1.0 / periods) - 1.0 if periods > 0 else 0.0
    
    annual_lower, annual_upper = bootstrap_confidence_interval(
        returns, annual_return_statistic, n_bootstrap=n_bootstrap
    )
    result['bootstrap_annual_return_ci_lower'] = annual_lower
    result['bootstrap_annual_return_ci_upper'] = annual_upper
    
    # 胜率二项检验
    if trades and len(trades) > 0:
        win_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_trades = len(trades)
        result['win_rate_pvalue'] = win_rate_binomial_test(win_trades, total_trades)
    
    return result

