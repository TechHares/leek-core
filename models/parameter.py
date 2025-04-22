#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class FieldType(Enum):
    """字段类型"""
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    RADIO = "radio"
    SELECT = "select"

class ChoiceType(Enum):
    """可选值类型"""
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"

@dataclass
class Field:
    """
    用于表示参数的类。
    """
    name: str  # 字段名称
    label: str = None  # 显示名称，默认为name
    description: str = "" # 字段描述
    type: FieldType = FieldType.STR  # 入参类型，可以是 "str", "int", "float", "bool", "datetime", "radio", "select"
    default: Any = None  # 默认值
    length: int = None  # 字段长度，仅适用于str
    min: float = None  # 最小值，仅适用于 int、float和datetime
    max: float = None  # 最大值，仅适用于 int、float和datetime
    required: bool = False  # 是否必传
    choices: list = field(default_factory=list)  # 可选值列表，仅适用于radio和select
    choice_type: ChoiceType = None  # 可选值类型，仅适用于radio和select，如 "str", "int", "float", "bool", "datetime"

if __name__ == '__main__':
    pass
