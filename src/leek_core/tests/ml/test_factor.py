#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
因子测试用例
"""

import os
import unittest
import logging
import time
from unittest.mock import patch, MagicMock
from io import StringIO
from leek_core.ml.factors.alpha101 import Alpha101Factor
from leek_core.ml.factors.alpha191 import Alpha191Factor
from leek_core.ml.factors.base import FeatureSpec, FeatureType
import pandas as pd

from leek_core.ml.factors.alpha158 import Alpha158Factor
from leek_core.ml.factors.alpha360 import Alpha360Factor
from leek_core.ml.factors.volume import LongShortVolumeRatioFactor, VolumeAverageFactor
from leek_core.ml.factors.direction import DirectionFactor
from leek_core.ml.factors.time import TimeFactor
from leek_core.ml.factors.price_spike import PriceSpikeFeaturesFactor, SimplePriceSpikeDetector
from leek_core.utils.decorator import retry


class TestFactor(unittest.TestCase):
    """因子测试类"""

    def _load_test_data(self):
        """加载测试数据"""
        return pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))

    def test_alpha158(self):
        """测试Alpha158因子"""
        factor = Alpha158Factor()
        df = self._load_test_data()
        result = factor.compute(df)
        print(factor.get_output_names())
        print(result.columns)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_alpha360(self):
        """测试Alpha360因子"""
        factor = Alpha360Factor()
        df = self._load_test_data()
        result = factor.compute(df)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))
        
    def test_alpha360_2(self):
        """测试Alpha360因子"""
        # {'fields': 'CLOSE,OPEN,HIGH,LOW,VWAP,VOLUME', 'windows': '5,10,20,30,60,120'}
        factor = Alpha360Factor(fields='CLOSE,OPEN,HIGH,LOW,VWAP,VOLUME', windows='5,10,20,30,60,120')
        df = self._load_test_data()
        result = factor.compute(df)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_time_factor_default(self):
        """测试时间因子（默认配置）"""
        factor = TimeFactor()
        df = self._load_test_data()
        result = factor.compute(df)
        
        # 默认配置应该包含：hour, hour_sin, hour_cos, day_of_week, dow_sin, dow_cos, day_of_month
        # 以及7个布尔特征 = 3 + 3 + 1 + 7 = 14个因子
        expected_features = len(factor.get_output_names())
        time_features = [col for col in result.columns if col.startswith("Time_")]
        self.assertEqual(len(time_features), expected_features)
        
        # 验证特征名称
        self.assertIn("Time_hour", result.columns)
        self.assertIn("Time_hour_sin", result.columns)
        self.assertIn("Time_hour_cos", result.columns)
        self.assertIn("Time_day_of_week", result.columns)
        self.assertIn("Time_dow_sin", result.columns)
        self.assertIn("Time_dow_cos", result.columns)
        self.assertIn("Time_day_of_month", result.columns)
        self.assertIn("Time_is_weekend", result.columns)
        
        # 验证数值范围
        self.assertTrue((result["Time_hour"] >= 0).all() and (result["Time_hour"] <= 23).all())
        self.assertTrue((result["Time_day_of_week"] >= 0).all() and (result["Time_day_of_week"] <= 6).all())
        self.assertTrue((result["Time_is_weekend"] >= 0).all() and (result["Time_is_weekend"] <= 1).all())

    def test_time_factor_extended(self):
        """测试时间因子（扩展配置）"""
        factor = TimeFactor(
            include_day_of_year=True,
            include_week_of_year=True,
            include_month=True,
            include_quarter=True,
            include_days_since=True
        )
        df = self._load_test_data()
        result = factor.compute(df)
        
        # 验证扩展特征存在
        self.assertIn("Time_day_of_year", result.columns)
        self.assertIn("Time_week_of_year", result.columns)
        self.assertIn("Time_month", result.columns)
        self.assertIn("Time_month_sin", result.columns)
        self.assertIn("Time_month_cos", result.columns)
        self.assertIn("Time_quarter", result.columns)
        self.assertIn("Time_quarter_sin", result.columns)
        self.assertIn("Time_quarter_cos", result.columns)
        self.assertIn("Time_days_since_month_start", result.columns)
        self.assertIn("Time_days_since_quarter_start", result.columns)
        self.assertIn("Time_days_since_year_start", result.columns)
        
        # 验证数值范围
        self.assertTrue((result["Time_month"] >= 1).all() and (result["Time_month"] <= 12).all())
        self.assertTrue((result["Time_quarter"] >= 1).all() and (result["Time_quarter"] <= 4).all())
        self.assertTrue((result["Time_days_since_month_start"] >= 0).all())

    def test_time_factor_minute(self):
        """测试时间因子（包含分钟）"""
        factor = TimeFactor(include_minute=True)
        df = self._load_test_data()
        result = factor.compute(df)
        
        # 验证分钟特征存在
        self.assertIn("Time_minute", result.columns)
        self.assertIn("Time_minute_sin", result.columns)
        self.assertIn("Time_minute_cos", result.columns)
        
        # 验证数值范围
        self.assertTrue((result["Time_minute"] >= 0).all() and (result["Time_minute"] <= 59).all())

    def test_time_factor_no_time_column(self):
        """测试时间因子（没有时间列）"""
        factor = TimeFactor()
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 2000, 3000]
        })
        result = factor.compute(df)
        
        # 如果没有时间列，应该返回原始df，不添加任何列
        self.assertEqual(len(result.columns), len(df.columns))
        self.assertListEqual(list(result.columns), list(df.columns))

    def test_alpha101(self):
        """测试Alpha101因子"""
        factor = Alpha101Factor()
        df = self._load_test_data()
        result = factor.compute(df)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_alpha191(self):
        """测试Alpha191因子"""
        factor = Alpha191Factor()
        df = self._load_test_data()
        result = factor.compute(df)
        print(len(factor.get_output_names()))
        self.assertEqual(len(result.columns), len(factor.get_output_names()))
        
    def test_LongShortVolumeRatioFactor(self):
        """测试成交量比率因子"""
        factor = LongShortVolumeRatioFactor(100, 3)
        df = self._load_test_data()
        df = df.iloc[:100]
        pct = df['close'].pct_change()
        max_pct_row = df.loc[pct == pct.max()]
        min_pct_row = df.loc[pct == pct.min()]

        # 获取成交量
        max_pct_volume = max_pct_row['volume'].values[0]
        min_pct_volume = min_pct_row['volume'].values[0]

        print(f"最大涨跌幅: {pct.max():.4f}, 对应成交量: {max_pct_volume}")
        print(f"最小涨跌幅: {pct.min():.4f}, 对应成交量: {min_pct_volume}")
        result = factor.compute(df)
        print(result[factor.get_output_names()[0]].values)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_DirectionFactor(self):
        """测试涨跌K线之比因子"""
        factor = DirectionFactor(100, "count")
        df = self._load_test_data()
        df = df.iloc[:100]
        pct = df['close'].pct_change()
        positive_days = (pct >= 0).sum()
        negative_days = (pct < 0).sum()
        # 下跌幅度之和（所有负涨跌幅的绝对值之和）
        down_sum = pct[pct < 0].abs().sum()
        up_sum = pct[pct >= 0].sum()
        print(f"上涨K线数量: {positive_days}, 下跌K线数量: {negative_days}")
        print(f"上涨幅度之和: {up_sum}, 下跌幅度之和: {down_sum}")
        result = factor.compute(df)
        print(result[factor.get_output_names()[0]].values)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_VolumeAverageFactor(self):
        """测试成交量平均值因子"""
        factor = VolumeAverageFactor(100, "FLAT", "min_volume", 1)
        df = self._load_test_data()
        df = df.iloc[:100]
        volume = df['volume'].values
        sorted_volume = sorted(volume)
        print(f"最小成交量: {sorted_volume[0]/df['volume'].values[-1]}")
        result = factor.compute(df)
        print(result[factor.get_output_names()[0]].values)
        self.assertEqual(len(result.columns), len(factor.get_output_names()))

    def test_feature_spec_api(self):
        """测试 get_output_specs() API 返回正确的 FeatureSpec"""
        # TimeFactor 应标记 categorical 特征
        factor = TimeFactor()
        specs = factor.get_output_specs()
        names = factor.get_output_names()
        
        # get_output_names 应从 specs 派生且长度一致
        self.assertEqual(len(specs), len(names))
        for spec, name in zip(specs, names):
            self.assertEqual(spec.name, name)
        
        # hour 应为 CATEGORICAL
        hour_specs = [s for s in specs if s.name == "Time_hour"]
        self.assertEqual(len(hour_specs), 1)
        self.assertEqual(hour_specs[0].type, FeatureType.CATEGORICAL)
        self.assertEqual(hour_specs[0].num_categories, 24)
        
        # day_of_week 应为 CATEGORICAL
        dow_specs = [s for s in specs if s.name == "Time_day_of_week"]
        self.assertEqual(len(dow_specs), 1)
        self.assertEqual(dow_specs[0].type, FeatureType.CATEGORICAL)
        self.assertEqual(dow_specs[0].num_categories, 7)
        
        # hour_sin 应为 NUMERIC（默认）
        sin_specs = [s for s in specs if s.name == "Time_hour_sin"]
        self.assertEqual(len(sin_specs), 1)
        self.assertEqual(sin_specs[0].type, FeatureType.NUMERIC)
        self.assertEqual(sin_specs[0].num_categories, 0)

    def test_price_spike_specs(self):
        """测试 PriceSpikeFeaturesFactor 的 specs 标记 is_spike 为 categorical"""
        factor = PriceSpikeFeaturesFactor()
        specs = factor.get_output_specs()
        names = factor.get_output_names()
        
        self.assertEqual(len(specs), len(names))
        
        # is_spike 应为 CATEGORICAL
        spike_specs = [s for s in specs if "is_spike" in s.name]
        self.assertEqual(len(spike_specs), 1)
        self.assertEqual(spike_specs[0].type, FeatureType.CATEGORICAL)
        self.assertEqual(spike_specs[0].num_categories, 2)
        
        # diff 应为 NUMERIC
        diff_specs = [s for s in specs if s.name.endswith("_diff") and "pos" not in s.name and "neg" not in s.name]
        self.assertEqual(len(diff_specs), 1)
        self.assertEqual(diff_specs[0].type, FeatureType.NUMERIC)

    def test_alpha_specs_all_numeric(self):
        """测试 Alpha 因子的 specs 全部为 NUMERIC"""
        factor = Alpha158Factor()
        specs = factor.get_output_specs()
        
        for spec in specs:
            self.assertEqual(spec.type, FeatureType.NUMERIC, 
                           f"Alpha158 spec {spec.name} should be NUMERIC")
            self.assertEqual(spec.num_categories, 0)
    

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 