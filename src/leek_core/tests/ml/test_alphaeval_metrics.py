#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AlphaEval 评估指标测试用例
"""

import os
import unittest
import numpy as np
import pandas as pd
from scipy.stats import entropy

from leek_core.ml.factors.evaluation import FactorEvaluator


class TestAlphaEvalMetrics(unittest.TestCase):
    """AlphaEval 评估指标测试类"""

    def setUp(self):
        """设置测试数据"""
        self.evaluator = FactorEvaluator(future_periods=1, quantile_count=5)
        
        # 生成模拟数据
        np.random.seed(42)
        n_samples = 200
        n_assets = 10
        
        # 生成时间序列数据（模拟多个时间点，每个时间点有多个资产）
        self.test_data = pd.DataFrame({
            'start_time': np.repeat(np.arange(n_samples), n_assets),
            'symbol': np.tile([f'ASSET_{i}' for i in range(n_assets)], n_samples),
            'close': np.random.randn(n_samples * n_assets) * 10 + 100,
            'open': np.random.randn(n_samples * n_assets) * 10 + 100,
            'high': np.random.randn(n_samples * n_assets) * 10 + 105,
            'low': np.random.randn(n_samples * n_assets) * 10 + 95,
            'volume': np.random.randn(n_samples * n_assets) * 1000 + 5000,
        })
        
        # 生成因子值（带有一定预测能力）
        # 稳定因子：排序变化小
        self.test_data['stable_factor'] = self.test_data.groupby('start_time')['close'].rank(pct=True) + \
                                           np.random.randn(len(self.test_data)) * 0.05
        
        # 不稳定因子：排序变化大
        self.test_data['unstable_factor'] = np.random.randn(len(self.test_data))
        
        # 计算未来收益
        self.test_data['future_return'] = self.test_data.groupby('symbol')['close'].pct_change().shift(-1)

    def test_temporal_stability_stable_factor(self):
        """测试时间稳定性计算 - 稳定因子"""
        result = self.evaluator.calculate_temporal_stability(
            self.test_data, 
            'stable_factor'
        )
        
        # 验证返回格式
        self.assertIn('rre_score', result)
        self.assertIn('estimated_turnover', result)
        
        # 验证数值范围
        self.assertGreaterEqual(result['rre_score'], 0.0)
        self.assertLessEqual(result['rre_score'], 1.0)
        self.assertGreaterEqual(result['estimated_turnover'], 0.0)
        
        # 稳定因子应该有较高的稳定性得分
        self.assertGreater(result['rre_score'], 0.5)
        
        # 高稳定性应该对应低换手率
        self.assertLess(result['estimated_turnover'], 15)
        
        print(f"稳定因子 - RRE: {result['rre_score']:.3f}, 预估换手率: {result['estimated_turnover']:.2f}")

    def test_temporal_stability_unstable_factor(self):
        """测试时间稳定性计算 - 不稳定因子"""
        result = self.evaluator.calculate_temporal_stability(
            self.test_data, 
            'unstable_factor'
        )
        
        # 验证返回格式
        self.assertIn('rre_score', result)
        self.assertIn('estimated_turnover', result)
        
        # 不稳定因子应该有较低的稳定性得分
        self.assertLess(result['rre_score'], 0.7)
        
        # 低稳定性应该对应高换手率
        self.assertGreater(result['estimated_turnover'], 5)
        
        print(f"不稳定因子 - RRE: {result['rre_score']:.3f}, 预估换手率: {result['estimated_turnover']:.2f}")

    def test_robustness_disabled(self):
        """测试鲁棒性计算 - 未启用"""
        result = self.evaluator.calculate_robustness(
            self.test_data,
            'stable_factor',
            enable_robustness=False
        )
        
        # 未启用时应返回默认值
        self.assertEqual(result['pfs_gaussian'], 1.0)
        self.assertEqual(result['pfs_t_dist'], 1.0)
        self.assertEqual(result['pfs_min'], 1.0)
        self.assertFalse(result['enabled'])

    def test_robustness_enabled_simplified(self):
        """测试鲁棒性计算 - 简化版本（直接对因子值添加噪声）"""
        result = self.evaluator.calculate_robustness(
            self.test_data,
            'stable_factor',
            factor_compute_func=None,  # 简化版本
            noise_level=0.05,
            n_trials=5,
            enable_robustness=True
        )
        
        # 验证返回格式
        self.assertIn('pfs_gaussian', result)
        self.assertIn('pfs_t_dist', result)
        self.assertIn('pfs_min', result)
        self.assertTrue(result['enabled'])
        
        # 验证数值范围
        self.assertGreaterEqual(result['pfs_gaussian'], 0.0)
        self.assertLessEqual(result['pfs_gaussian'], 1.0)
        self.assertGreaterEqual(result['pfs_t_dist'], 0.0)
        self.assertLessEqual(result['pfs_t_dist'], 1.0)
        
        # pfs_min 应该是两者的最小值
        self.assertEqual(result['pfs_min'], min(result['pfs_gaussian'], result['pfs_t_dist']))
        
        # 鲁棒性得分应该较高（因为是在因子值上直接添加小噪声）
        self.assertGreater(result['pfs_gaussian'], 0.7)
        
        print(f"鲁棒性 - Gaussian: {result['pfs_gaussian']:.3f}, t-dist: {result['pfs_t_dist']:.3f}, Min: {result['pfs_min']:.3f}")

    def test_diversity_entropy_independent_factors(self):
        """测试多样性熵 - 独立因子"""
        # 创建3个完全独立的因子（正交）
        n = 300
        factor1 = np.random.randn(n)
        factor2 = np.random.randn(n)
        factor3 = np.random.randn(n)
        
        factor_df = pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'factor3': factor3
        })
        
        # 计算相关性矩阵
        corr_matrix = factor_df.corr()
        
        # 计算多样性熵
        diversity = self.evaluator.calculate_diversity_entropy(corr_matrix)
        
        # 独立因子的多样性应该很高（接近1）
        self.assertGreater(diversity, 0.8)
        self.assertLessEqual(diversity, 1.0)
        
        print(f"独立因子多样性: {diversity:.3f}")

    def test_diversity_entropy_correlated_factors(self):
        """测试多样性熵 - 高度相关因子"""
        # 创建3个高度相关的因子
        n = 300
        base_factor = np.random.randn(n)
        
        factor_df = pd.DataFrame({
            'factor1': base_factor + np.random.randn(n) * 0.1,
            'factor2': base_factor + np.random.randn(n) * 0.1,
            'factor3': base_factor + np.random.randn(n) * 0.1
        })
        
        # 计算相关性矩阵
        corr_matrix = factor_df.corr()
        
        # 计算多样性熵
        diversity = self.evaluator.calculate_diversity_entropy(corr_matrix)
        
        # 高度相关因子的多样性应该很低
        self.assertLess(diversity, 0.5)
        self.assertGreaterEqual(diversity, 0.0)
        
        print(f"相关因子多样性: {diversity:.3f}")

    def test_diversity_entropy_single_factor(self):
        """测试多样性熵 - 单因子"""
        # 单因子应该返回0（无多样性）
        factor_df = pd.DataFrame({
            'factor1': np.random.randn(100)
        })
        
        corr_matrix = factor_df.corr()
        diversity = self.evaluator.calculate_diversity_entropy(corr_matrix)
        
        self.assertEqual(diversity, 0.0)
        print(f"单因子多样性: {diversity:.3f}")

    def test_diversity_entropy_empty_matrix(self):
        """测试多样性熵 - 空矩阵"""
        diversity = self.evaluator.calculate_diversity_entropy(pd.DataFrame())
        self.assertEqual(diversity, 0.0)

    def test_temporal_stability_edge_cases(self):
        """测试时间稳定性 - 边界情况"""
        # 测试数据不足的情况
        small_data = self.test_data.head(5)
        result = self.evaluator.calculate_temporal_stability(small_data, 'stable_factor')
        
        # 数据不足时应该返回默认值
        self.assertGreaterEqual(result['rre_score'], 0.0)
        self.assertGreaterEqual(result['estimated_turnover'], 0.0)

    def test_robustness_with_different_noise_levels(self):
        """测试鲁棒性 - 不同噪声水平"""
        # 低噪声
        result_low = self.evaluator.calculate_robustness(
            self.test_data,
            'stable_factor',
            noise_level=0.01,
            n_trials=3,
            enable_robustness=True
        )
        
        # 高噪声
        result_high = self.evaluator.calculate_robustness(
            self.test_data,
            'stable_factor',
            noise_level=0.2,
            n_trials=3,
            enable_robustness=True
        )
        
        # 低噪声下的鲁棒性应该更高
        self.assertGreaterEqual(result_low['pfs_min'], result_high['pfs_min'])
        
        print(f"低噪声鲁棒性: {result_low['pfs_min']:.3f}")
        print(f"高噪声鲁棒性: {result_high['pfs_min']:.3f}")

    def test_integration_alphaeval_metrics(self):
        """集成测试：完整的 AlphaEval 评估流程"""
        # 计算因子的所有 AlphaEval 指标
        
        # 1. 预测能力（使用现有evaluate_factor方法）
        eval_result = self.evaluator.evaluate_factor(
            self.test_data,
            'stable_factor',
            ic_window=20
        )
        
        self.assertIn('ic_mean', eval_result)
        self.assertIn('ir', eval_result)
        
        # 2. 时间稳定性
        stability_result = self.evaluator.calculate_temporal_stability(
            self.test_data,
            'stable_factor'
        )
        
        self.assertIn('rre_score', stability_result)
        
        # 3. 鲁棒性
        robustness_result = self.evaluator.calculate_robustness(
            self.test_data,
            'stable_factor',
            enable_robustness=True,
            n_trials=3
        )
        
        self.assertIn('pfs_min', robustness_result)
        
        # 4. 构建完整的评估报告
        full_report = {
            'factor_name': 'stable_factor',
            'ic_mean': eval_result['ic_mean'],
            'ir': eval_result['ir'],
            'ic_win_rate': eval_result['ic_win_rate'],
            'temporal_stability': stability_result['rre_score'],
            'estimated_turnover': stability_result['estimated_turnover'],
            'robustness_min': robustness_result['pfs_min'],
        }
        
        print("\n===== AlphaEval 完整评估报告 =====")
        for key, value in full_report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("===================================\n")
        
        # 验证所有关键指标都已计算
        self.assertIsNotNone(full_report['temporal_stability'])
        self.assertIsNotNone(full_report['robustness_min'])

    def test_correlation_matrix_to_diversity(self):
        """测试从相关性矩阵计算多样性的完整流程"""
        # 创建多个因子
        n = 300
        factor_data = pd.DataFrame({
            'factor1': np.random.randn(n),
            'factor2': np.random.randn(n) * 0.5 + np.random.randn(n) * 0.5,
            'factor3': np.random.randn(n),
            'factor4': np.random.randn(n) * 0.7 + np.random.randn(n) * 0.3,
        })
        
        # 计算相关性矩阵
        corr_matrix = self.evaluator._calculate_factor_correlation(factor_data)
        
        # 验证相关性矩阵格式
        self.assertEqual(corr_matrix.shape[0], 4)
        self.assertEqual(corr_matrix.shape[1], 4)
        
        # 计算多样性
        diversity = self.evaluator.calculate_diversity_entropy(corr_matrix)
        
        # 验证多样性在合理范围内
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
        
        # 打印相关性矩阵和多样性得分
        print("\n相关性矩阵:")
        print(corr_matrix)
        print(f"\n多样性得分: {diversity:.3f}\n")


class TestAlphaEvalCompositeScore(unittest.TestCase):
    """AlphaEval 综合评分测试"""
    
    def test_score_calculation(self):
        """测试综合评分计算"""
        from leek_manager.app.service.factor_evaluation_service import calculate_alpha_eval_score
        
        # 测试优秀因子
        metrics_excellent = {
            'ic_mean': 0.08,
            'temporal_stability': 0.85,
            'robustness_min': 0.92
        }
        
        score = calculate_alpha_eval_score(metrics_excellent)
        
        # 验证得分范围
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # 优秀因子应该有高分
        self.assertGreater(score, 0.7)
        
        print(f"优秀因子综合得分: {score:.3f}")
        
        # 测试差因子
        metrics_poor = {
            'ic_mean': 0.01,
            'temporal_stability': 0.3,
            'robustness_min': 0.5
        }
        
        score_poor = calculate_alpha_eval_score(metrics_poor)
        
        # 差因子应该有低分
        self.assertLess(score_poor, 0.5)
        
        print(f"差因子综合得分: {score_poor:.3f}")

    def test_custom_weights(self):
        """测试自定义权重"""
        from leek_manager.app.service.factor_evaluation_service import calculate_alpha_eval_score
        
        metrics = {
            'ic_mean': 0.05,
            'temporal_stability': 0.7,
            'robustness_min': 0.8
        }
        
        # 默认权重
        score_default = calculate_alpha_eval_score(metrics)
        
        # 强调预测能力的权重
        score_predictive = calculate_alpha_eval_score(metrics, {
            'predictive_power': 0.6,
            'temporal_stability': 0.2,
            'robustness': 0.2
        })
        
        # 强调稳定性的权重
        score_stability = calculate_alpha_eval_score(metrics, {
            'predictive_power': 0.2,
            'temporal_stability': 0.6,
            'robustness': 0.2
        })
        
        # 验证不同权重产生不同结果
        self.assertNotEqual(score_default, score_predictive)
        self.assertNotEqual(score_default, score_stability)
        
        print(f"默认权重得分: {score_default:.3f}")
        print(f"强调预测得分: {score_predictive:.3f}")
        print(f"强调稳定得分: {score_stability:.3f}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
