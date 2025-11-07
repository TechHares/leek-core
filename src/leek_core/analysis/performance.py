#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能指标计算模块

提供常用的金融性能指标计算功能：
- 年化收益率 (Annualized Return)
- 最大回撤 (Maximum Drawdown)
- 波动率 (Volatility)
- 夏普比率 (Sharpe Ratio)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from decimal import Decimal
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化性能指标计算器
        
        Args:
            risk_free_rate: 无风险利率，默认2%
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_annualized_return(
        self, 
        returns: List[float], 
        periods_per_year: int = 365  # 日数据，一年365天
    ) -> float:
        """
        计算年化收益率
        
        Args:
            returns: 收益率序列
            periods_per_year: 一年的期数，默认365（日数据）
            
        Returns:
            年化收益率
        """
        if not returns:
            return 0.0
        
        try:
            # 转换为numpy数组
            returns_array = np.array(returns, dtype=float)
            
            # 检查是否全为零
            if np.all(returns_array == 0):
                return 0.0
            
            # 计算总收益率
            total_return = np.prod(1 + returns_array) - 1
            
            # 计算年化收益率
            n_periods = len(returns_array)
            if n_periods == 0:
                return 0.0
            
            # 防止溢出
            if total_return <= -1:
                return -1.0
            
            # 限制指数范围，防止溢出
            exponent = periods_per_year / n_periods
            
            # 如果期数太少（少于30期），使用简单年化避免复利放大问题
            # 如果指数太大（>100），也使用简单年化
            if n_periods < 30 or exponent > 100:
                # 简单年化：总收益率 * (periods_per_year / n_periods)
                annualized_return = total_return * exponent
            else:
                # 复利年化：(1 + 总收益率) ^ exponent - 1
                annualized_return = (1 + total_return) ** exponent - 1
            
            # 检查结果是否有效
            if not np.isfinite(annualized_return):
                # 如果结果无效，回退到简单年化
                annualized_return = total_return * exponent
            
            return float(annualized_return)
            
        except Exception as e:
            logger.error(f"计算年化收益率失败: {e}")
            return 0.0
    
    def calculate_maximum_drawdown(self, equity_curve: List[float]) -> Dict[str, float]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线（累计净值）
            
        Returns:
            包含最大回撤和回撤持续期的字典
        """
        if not equity_curve or len(equity_curve) < 2:
            return {"max_drawdown": 0.0, "drawdown_duration": 0}
        
        try:
            equity_array = np.array(equity_curve, dtype=float)
            
            # 检查是否全为零
            if np.all(equity_array == 0):
                return {"max_drawdown": 0.0, "drawdown_duration": 0}
            
            # 计算历史最高点
            running_max = np.maximum.accumulate(equity_array)
            
            # 计算回撤，避免除零错误
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = np.where(running_max != 0, 
                                   (equity_array - running_max) / running_max, 
                                   0.0)
            
            # 最大回撤
            max_drawdown = float(np.min(drawdown))
            
            # 检查结果是否有效
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                max_drawdown = 0.0
            
            # 计算最大回撤的持续期
            max_dd_idx = np.argmin(drawdown)
            peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
            drawdown_duration = max_dd_idx - peak_idx
            
            return {
                "max_drawdown": max_drawdown,
                "drawdown_duration": int(drawdown_duration)
            }
            
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return {"max_drawdown": 0.0, "drawdown_duration": 0}
    
    def calculate_volatility(
        self, 
        returns: List[float], 
        periods_per_year: int = 365
    ) -> float:
        """
        计算年化波动率
        
        Args:
            returns: 收益率序列
            periods_per_year: 一年的期数，默认365（日数据）
            
        Returns:
            年化波动率
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns, dtype=float)
            
            # 计算收益率的标准差
            std_return = np.std(returns_array, ddof=1)
            
            # 年化波动率
            annualized_volatility = std_return * np.sqrt(periods_per_year)
            
            return float(annualized_volatility)
            
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(
        self, 
        returns: List[float], 
        periods_per_year: int = 365
    ) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 一年的期数，默认365（日数据）
            
        Returns:
            夏普比率
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            returns_array = np.array(returns, dtype=float)
            
            # 计算平均收益率
            mean_return = np.mean(returns_array)
            
            # 计算收益率的标准差
            std_return = np.std(returns_array, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            # 计算夏普比率
            sharpe_ratio = (mean_return - self.risk_free_rate / periods_per_year) / std_return
            
            # 年化夏普比率
            annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
            
            return float(annualized_sharpe)
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    def calculate_all_metrics(
        self, 
        equity_curve: List[float], 
        periods_per_year: int = 365
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        计算所有性能指标
        
        Args:
            equity_curve: 权益曲线（累计净值）
            periods_per_year: 一年的期数，默认365（日数据）
            
        Returns:
            包含所有性能指标的字典
        """
        if not equity_curve or len(equity_curve) < 2:
            return {
                "annualized_return": 0.0,
                "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }
        
        try:
            # 计算收益率序列
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] != 0:
                    ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                else:
                    ret = 0.0
                returns.append(ret)
            
            # 计算各项指标
            annualized_return = self.calculate_annualized_return(returns, periods_per_year)
            max_drawdown_info = self.calculate_maximum_drawdown(equity_curve)
            volatility = self.calculate_volatility(returns, periods_per_year)
            sharpe_ratio = self.calculate_sharpe_ratio(returns, periods_per_year)
            
            return {
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown_info,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return {
                "annualized_return": 0.0,
                "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }
    
    def calculate_period_comparison(
        self, 
        current_data: List[float], 
        previous_data: List[float],
        periods_per_year: int = 365
    ) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """
        计算两个时期的性能指标对比
        
        Args:
            current_data: 当前时期权益曲线
            previous_data: 上一时期权益曲线
            periods_per_year: 一年的期数，默认365（日数据）
            
        Returns:
            包含两个时期性能指标的字典
        """
        try:
            current_metrics = self.calculate_all_metrics(current_data, periods_per_year)
            previous_metrics = self.calculate_all_metrics(previous_data, periods_per_year)
            
            return {
                "current": current_metrics,
                "previous": previous_metrics
            }
            
        except Exception as e:
            logger.error(f"计算时期对比失败: {e}")
            return {
                "current": {
                    "annualized_return": 0.0,
                    "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0
                },
                "previous": {
                    "annualized_return": 0.0,
                    "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0
                }
            }


def calculate_performance_from_values(
    values: List[Union[float, Decimal]], 
    periods_per_year: int = 365
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    从数值列表计算性能指标
    
    Args:
        values: 数值列表（如资产总额序列）
        periods_per_year: 一年的期数，默认365（日数据）
        
    Returns:
        性能指标字典
    """
    if not values or len(values) < 2:
        return {
            "annualized_return": 0.0,
            "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
            "volatility": 0.0,
            "sharpe_ratio": 0.0
        }
    
    try:
        # 转换为float列表
        equity_curve = [float(value) for value in values]
        
        # 计算性能指标
        calculator = PerformanceMetrics()
        return calculator.calculate_all_metrics(equity_curve, periods_per_year)
        
    except Exception as e:
        logger.error(f"从数值列表计算性能指标失败: {e}")
        return {
            "annualized_return": 0.0,
            "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
            "volatility": 0.0,
            "sharpe_ratio": 0.0
        }


def calculate_period_comparison_from_values(
    current_values: List[Union[float, Decimal]],
    previous_values: List[Union[float, Decimal]],
    periods_per_year: int = 365
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """
    从数值列表计算两个时期的性能指标对比
    
    Args:
        current_values: 当前时期数值列表
        previous_values: 上一时期数值列表
        periods_per_year: 一年的期数，默认365（日数据）
        
    Returns:
        包含两个时期性能指标的字典
    """
    try:
        # 转换为float列表
        current_equity = [float(v) for v in current_values]
        previous_equity = [float(v) for v in previous_values]
        
        # 计算性能指标对比
        calculator = PerformanceMetrics()
        return calculator.calculate_period_comparison(
            current_equity, previous_equity, periods_per_year
        )
        
    except Exception as e:
        logger.error(f"从数值列表计算时期对比失败: {e}")
        return {
            "current": {
                "annualized_return": 0.0,
                "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            },
            "previous": {
                "annualized_return": 0.0,
                "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }
        }


 