#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from leek_core.utils import DateTimeUtils


class FieldType(Enum):
    """字段类型"""
    PASSWORD = "password"
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    RADIO = "radio"
    SELECT = "select"
    ARRAY = "array"

class ChoiceType(Enum):
    """可选值类型"""
    STRING = "string"
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
    type: FieldType = FieldType.STRING  # 入参类型，可以是 "string", "int", "float", "bool", "datetime", "radio", "select"
    default: Any = None  # 默认值
    length: int = None  # 字段长度，仅适用于str
    min: float = None  # 最小值，仅适用于 int、float和datetime
    max: float = None  # 最大值，仅适用于 int、float和datetime
    required: bool = False  # 是否必传
    choices: list = field(default_factory=list)  # 可选值列表，仅适用于radio和select
    choice_type: ChoiceType = None  # 可选值类型，仅适用于radio和select，如 "str", "int", "float", "bool", "datetime"

    def covert(self, value: Any) -> Any:
        """
        将字符串值转换为指定类型。
        """
        if self.type in [FieldType.RADIO, FieldType.SELECT, FieldType.ARRAY]:
            if self.choice_type is None:
                return value
            return self.covert_value(FieldType(self.choice_type.value), value)
        return self.covert_value(self.type, value)

    @staticmethod
    def covert_value(tp: FieldType, value: Any) -> Any:
        """
        将字符串值转换为指定类型。
        """
        if value is None:
            return None
        if isinstance(value, Enum):
            return Field.covert_value(tp, value.value)

        if tp == FieldType.STRING or tp == FieldType.PASSWORD:
            return str(value)

        if tp == FieldType.INT:
            return int(value)

        if tp == FieldType.FLOAT:
            return Decimal(value)

        if tp == FieldType.BOOLEAN:
            try:
                return bool(value)
            except Exception:
                return str(value).lower() in ("true", 'on', 'open', '1')

        if tp == FieldType.DATETIME:
            if isinstance(value, datetime):
                return value
            if isinstance(value, int):
                return DateTimeUtils.to_datetime(value)
            if isinstance(value, str):
                return DateTimeUtils.to_datetime(DateTimeUtils.to_timestamp(value))
            raise ValueError(f"Invalid datetime value: {value}")


if __name__ == '__main__':
    pass
