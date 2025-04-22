#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理器基类模块，定义处理器的基本接口。
"""

from abc import ABC, abstractmethod
from typing import List, Any

from models import DataType, Component, Data


class Processor(Component, ABC):
    """处理器基类，定义处理器的基本接口"""
    priority = 10  # 优先级，数字越小优先级越高
    process_data_type = DataType.KLINE  # 处理器支持的数据类型
    display_name = "数据处理器"  # 处理器的显示名称
    
    @abstractmethod
    def process(self, data: List[Data]) -> List[Data]:
        """
        处理数据
        
        Args:
            data: 输入数据

        Returns:
            处理后的数据列表
        """
        ...