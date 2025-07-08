#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decimal 工具模块，处理金融数值精度问题。
"""

from decimal import *
from typing import Any


class DecimalEncoder:
    """处理Decimal类型的序列化和反序列化"""
    
    @staticmethod
    def encode(obj: Any) -> Any:
        """
        将对象中的Decimal转换为字符串
        
        参数:
            obj: 要编码的对象
            
        返回:
            编码后的对象
        """
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: DecimalEncoder.encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DecimalEncoder.encode(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(DecimalEncoder.encode(item) for item in obj)
        return obj
    
    @staticmethod
    def decode(obj: Any) -> Any:
        """
        将对象中的字符串转换为Decimal（如果它看起来像数字）
        
        参数:
            obj: 要解码的对象
            
        返回:
            解码后的对象
        """
        if isinstance(obj, str):
            # 尝试将字符串转换为Decimal，如果失败则保持不变
            try:
                # 检查字符串是否符合数字格式
                if obj.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return Decimal(obj)
            except:
                pass
            return obj
        elif isinstance(obj, dict):
            return {k: DecimalEncoder.decode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DecimalEncoder.decode(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(DecimalEncoder.decode(item) for item in tuple)
        return obj

def decimal_quantize(d, n=2, rounding=2):
    """
    decimal 精度处理
    :param d: 待处理decimal
    :param n: 小数位数
    :param rounding: 保留方式 0 四舍五入 1 进一法 2 舍弃
    :return:
    """
    if isinstance(d, float):
        d = Decimal(str(d))
    elif isinstance(d, int):
        d = Decimal(str(d))
    elif isinstance(d, str):
        d = Decimal(d)
    if d is None:
        return None
    r = ROUND_HALF_DOWN
    if rounding == 1:
        r = ROUND_UP
    elif rounding == 2:
        r = ROUND_DOWN

    p = "0"
    if n > 0:
        p = "0." + "0" * n
    return d.quantize(Decimal(p), rounding=r)