#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC
from typing import Tuple

from leek_core.models import Data, KLine


class StrategyMode(ABC):
    """策略运行模式。strategy context会根据Key的不同，来决定是否需要创建新的strategy实例。"""
    @abstractmethod
    def build_instance_key(self, data: Data) -> Tuple:
        """
        构建实例Key，用于区分不同的实例。

        参数:
            dt: 数据类型
            data: 数据
        返回:
            实例Key
        """
        ...


class Single(StrategyMode):
    """ 只有一个实例 """
    def build_instance_key(self, data: KLine) -> str:
        return "default",

class KlineSimple(StrategyMode):
    """根据K线交易对 资产类型 和 时间周期 来创建策略实例"""
    def build_instance_key(self, data: KLine) -> str:
        return data.row_key