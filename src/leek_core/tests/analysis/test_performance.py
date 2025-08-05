#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能指标计算模块测试
"""

import pytest
import numpy as np
from leek_core.analysis.performance import (
    PerformanceMetrics,
    calculate_performance_from_values,
    calculate_period_comparison_from_values
)


class TestPerformanceMetrics:
    """性能指标计算器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.calculator = PerformanceMetrics()
    
    def test_calculate_annualized_return(self):
        """测试年化收益率计算"""
        # 测试数据：简单的收益率序列
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        result = self.calculator.calculate_annualized_return(returns, periods_per_year=8760)
        
        assert isinstance(result, float)
        assert result != 0.0  # 应该有计算结果
    
    def test_calculate_maximum_drawdown(self):
        """测试最大回撤计算"""
        # 测试数据：权益曲线
        equity_curve = [100, 105, 110, 95, 100, 108, 102, 115]
        
        result = self.calculator.calculate_maximum_drawdown(equity_curve)
        
        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert "drawdown_duration" in result
        assert result["max_drawdown"] <= 0  # 回撤应该是负数或零
    
    def test_calculate_volatility(self):
        """测试波动率计算"""
        # 测试数据：收益率序列
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01]
        
        result = self.calculator.calculate_volatility(returns, periods_per_year=8760)
        
        assert isinstance(result, float)
        assert result >= 0  # 波动率应该是非负数
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        # 测试数据：收益率序列
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01]
        
        result = self.calculator.calculate_sharpe_ratio(returns, periods_per_year=8760)
        
        assert isinstance(result, float)
    
    def test_calculate_all_metrics(self):
        """测试所有指标计算"""
        # 测试数据：权益曲线
        equity_curve = [100, 105, 110, 95, 100, 108, 102, 115, 120, 118]
        
        result = self.calculator.calculate_all_metrics(equity_curve, periods_per_year=8760)
        
        assert isinstance(result, dict)
        assert "annualized_return" in result
        assert "max_drawdown" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
    
    def test_empty_data(self):
        """测试空数据处理"""
        # 测试空收益率序列
        result = self.calculator.calculate_annualized_return([])
        assert result == 0.0
        
        # 测试空权益曲线
        result = self.calculator.calculate_maximum_drawdown([])
        assert result == {"max_drawdown": 0.0, "drawdown_duration": 0}
        
        # 测试单个数据点
        result = self.calculator.calculate_maximum_drawdown([100])
        assert result == {"max_drawdown": 0.0, "drawdown_duration": 0}


class TestValueFunctions:
    """数值列表相关函数测试"""
    
    def test_calculate_performance_from_values(self):
        """测试从数值列表计算性能指标"""
        # 测试数据：数值列表
        values = [100.0, 105.0, 110.0, 95.0, 100.0, 108.0]
        
        result = calculate_performance_from_values(values)
        
        assert isinstance(result, dict)
        assert "annualized_return" in result
        assert "max_drawdown" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
    
    def test_calculate_period_comparison_from_values(self):
        """测试从数值列表计算时期对比"""
        # 测试数据：当前时期和上一时期的数值列表
        current_values = [100.0, 105.0, 110.0]
        previous_values = [95.0, 98.0, 102.0]
        
        result = calculate_period_comparison_from_values(current_values, previous_values)
        
        assert isinstance(result, dict)
        assert "current" in result
        assert "previous" in result
        assert "annualized_return" in result["current"]
        assert "annualized_return" in result["previous"]
    
    def test_empty_values(self):
        """测试空数值列表处理"""
        # 测试空数值列表
        result = calculate_performance_from_values([])
        assert result == {
            "annualized_return": 0.0,
            "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
            "volatility": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # 测试单个数值
        result = calculate_performance_from_values([100.0])
        assert result == {
            "annualized_return": 0.0,
            "max_drawdown": {"max_drawdown": 0.0, "drawdown_duration": 0},
            "volatility": 0.0,
            "sharpe_ratio": 0.0
        }


class TestEdgeCases:
    """边界情况测试"""
    
    def test_zero_values(self):
        """测试零值处理"""
        calculator = PerformanceMetrics()
        
        # 测试全零收益率
        returns = [0.0, 0.0, 0.0, 0.0]
        result = calculator.calculate_annualized_return(returns)
        assert result == 0.0
        
        # 测试全零权益曲线
        equity_curve = [0.0, 0.0, 0.0, 0.0]
        result = calculator.calculate_maximum_drawdown(equity_curve)
        assert result == {"max_drawdown": 0.0, "drawdown_duration": 0}
    
    def test_negative_values(self):
        """测试负值处理"""
        calculator = PerformanceMetrics()
        
        # 测试负收益率
        returns = [-0.01, -0.02, -0.01, -0.03]
        result = calculator.calculate_annualized_return(returns)
        assert result < 0  # 应该为负值
        
        # 测试递减权益曲线
        equity_curve = [100, 95, 90, 85, 80]
        result = calculator.calculate_maximum_drawdown(equity_curve)
        assert result["max_drawdown"] < 0  # 应该有回撤
    
    def test_large_numbers(self):
        """测试大数值处理"""
        calculator = PerformanceMetrics()
        
        # 测试大数值
        returns = [0.1, 0.2, -0.1, 0.3, 0.1]
        result = calculator.calculate_annualized_return(returns)
        assert isinstance(result, float)
        assert not np.isnan(result)  # 不应该为NaN
        assert not np.isinf(result)  # 不应该为无穷大


if __name__ == "__main__":
    pytest.main([__file__]) 