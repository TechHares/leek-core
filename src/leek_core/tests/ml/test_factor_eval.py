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
from leek_core.ml.factors.evaluation import FactorEvaluator
import pandas as pd

from leek_core.ml.factors.alpha158 import Alpha158Factor
from leek_core.ml.factors.alpha360 import Alpha360Factor
from leek_core.ml.factors.time import TimeFactor
from leek_core.utils.decorator import retry


class TestFactor(unittest.TestCase):
    """因子测试类"""

    def _load_test_data(self):
        """加载测试数据"""
        return pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))

    def test_alpha158(self):
        """测试Alpha158因子"""
        factor = Alpha158Factor(include_kbar=False, include_price=False, include_rolling=True, windows="5,10")
        df = self._load_test_data()
        print(factor.get_output_names())
        result = factor.compute(df)
        print(len(result.columns))
        evaluator = FactorEvaluator(future_periods=1, quantile_count=5)
        future_returns = evaluator.evaluate_factors(result, factor.get_output_names(), ic_window=20)
        print(future_returns)
    

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 