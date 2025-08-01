#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
序列化工具模块，提供对象状态序列化和反序列化功能。
"""

from decimal import Decimal
from time import time
from typing import Any, Dict, Set
from enum import Enum
import json
from datetime import datetime
from dataclasses import is_dataclass, asdict
import time as time_module

class StrategyStateSerializer:
    """
    策略状态序列化器，用于序列化和反序列化策略的运行状态。
    """
    
    @staticmethod
    def is_basic_type(value: Any) -> bool:
        """
        判断是否为支持的基础类型
        
        参数:
            value: 要检查的值
            
        返回:
            是否为支持的基础类型
        """
        if value is None:
            return True
        return isinstance(value, (int, float, bool, Decimal)) or (
            hasattr(value, '__class__') and 
            hasattr(value.__class__, '__bases__') and 
            any(base.__name__ == 'Enum' for base in value.__class__.__bases__)
        )
    
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """
        序列化单个值
        
        参数:
            value: 要序列化的值
            
        返回:
            序列化后的值
        """
        if value is None:
            return None
        elif isinstance(value, Decimal):
            return str(value)
        elif hasattr(value, '__class__') and hasattr(value.__class__, '__bases__'):
            # 处理枚举类型
            if any(base.__name__ == 'Enum' for base in value.__class__.__bases__):
                return value.value
        return value
    
    @staticmethod
    def deserialize_value(value: Any, field_type: str) -> Any:
        """
        反序列化单个值
        
        参数:
            value: 要反序列化的值
            field_type: 字段类型字符串
            
        返回:
            反序列化后的值
        """
        if value is None:
            return None
        elif field_type == 'Decimal':
            return Decimal(str(value))
        elif field_type == 'int':
            return int(value)
        elif field_type == 'float':
            return float(value)
        elif field_type == 'bool':
            return bool(value)
        elif field_type == 'datetime':
            # 将毫秒时间戳转换为datetime对象
            if isinstance(value, (int, float)):
                # 如果是毫秒时间戳，需要除以1000
                if value > 1e10:  # 毫秒时间戳通常大于1e10
                    return datetime.fromtimestamp(value / 1000)
                else:  # 秒时间戳
                    return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                try:
                    # 尝试解析为时间戳
                    timestamp = float(value)
                    if timestamp > 1e10:  # 毫秒时间戳
                        return datetime.fromtimestamp(timestamp / 1000)
                    else:  # 秒时间戳
                        return datetime.fromtimestamp(timestamp)
                except ValueError:
                    # 如果不是时间戳，尝试解析为ISO格式
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
            return value
        elif 'Enum' in field_type:
            # 自动转换models包下的枚举类型
            return StrategyStateSerializer._deserialize_enum_value(value, field_type)
        return value
    
    @staticmethod
    def _deserialize_enum_value(value: Any, field_type: str) -> Any:
        """
        反序列化枚举值
        
        参数:
            value: 要反序列化的值
            field_type: 字段类型字符串，格式为 "Enum(EnumClassName)"
            
        返回:
            反序列化后的枚举值
        """
        try:
            # 从field_type中提取枚举类名
            if field_type.startswith('Enum(') and field_type.endswith(')'):
                enum_class_name = field_type[5:-1]  # 去掉 "Enum(" 和 ")"
                
                # 获取当前模块的所有枚举类
                import leek_core.models as models_module
                enum_classes = {}
                
                for attr_name in dir(models_module):
                    attr = getattr(models_module, attr_name)
                    # 检查是否为枚举类（排除非枚举类）
                    if (hasattr(attr, '__class__') and 
                        hasattr(attr.__class__, '__bases__') and
                        any(base.__name__ == 'Enum' for base in attr.__class__.__bases__)):
                        enum_classes[attr_name] = attr.__class__
                
                # 获取枚举类
                enum_class = enum_classes.get(enum_class_name)
                
                if enum_class:
                    # 尝试通过值创建枚举实例
                    return enum_class(value)
                else:
                    # 如果找不到对应的枚举类，返回原始值
                    return value
            else:
                return value
        except (ValueError, TypeError):
            # 如果转换失败，返回原始值
            return value
    
    @staticmethod
    def get_serializable_fields(obj: Any, init_param_names: Set[str]) -> Dict[str, Any]:
        """
        获取可序列化的字段
        
        参数:
            obj: 要序列化的对象
            init_param_names: init_params中定义的字段名称集合
            
        返回:
            字段名到字段值的字典
        """
        serializable_fields = {}
        # 获取当前实例的所有属性
        for attr_name, attr_value in obj.__dict__.items():
            # 跳过以下情况：
            # 1. 以_开头的私有属性
            # 2. init_params中定义的字段
            # 3. 不是基础类型的字段
            if (not attr_name.startswith('_') and 
                attr_name not in init_param_names and 
                StrategyStateSerializer.is_basic_type(attr_value)):
                
                serializable_fields[attr_name] = attr_value
        
        return serializable_fields
    
    @staticmethod
    def serialize(obj: Any, init_param_names: Set[str]) -> Dict[str, Any]:
        """
        序列化对象状态
        
        参数:
            obj: 要序列化的对象
            init_param_names: init_params中定义的字段名称集合
            
        返回:
            包含对象状态和字段类型信息的字典
        """
        serializable_fields = StrategyStateSerializer.get_serializable_fields(obj, init_param_names)
        # 序列化字段值
        serialized_state = {}
        field_extra = {}
        
        for field_name, field_value in serializable_fields.items():
            # 序列化值
            serialized_value = StrategyStateSerializer.serialize_value(field_value)
            serialized_state[field_name] = serialized_value
            
            # 记录字段类型
            if field_value is None:
                field_extra[field_name] = 'None'
            elif isinstance(field_value, Decimal):
                field_extra[field_name] = 'Decimal'
            elif isinstance(field_value, int):
                field_extra[field_name] = 'int'
            elif isinstance(field_value, float):
                field_extra[field_name] = 'float'
            elif isinstance(field_value, bool):
                field_extra[field_name] = 'bool'
            elif hasattr(field_value, '__class__') and hasattr(field_value.__class__, '__bases__'):
                if any(base.__name__ == 'Enum' for base in field_value.__class__.__bases__):
                    field_extra[field_name] = f'Enum({field_value.__class__.__name__})'
                else:
                    field_extra[field_name] = field_value.__class__.__name__
            else:
                field_extra[field_name] = field_value.__class__.__name__
        
        # 添加字段类型信息
        serialized_state['field_extra'] = field_extra
        
        return serialized_state
    
    @staticmethod
    def deserialize(obj: Any, state: Dict[str, Any]):
        """
        反序列化对象状态
        
        参数:
            obj: 要反序列化的对象
            state: 状态字典，包含field_extra字段记录字段类型信息
        """
        if not state:
            return
            
        # 获取字段类型信息
        field_extra = state.get('field_extra', {})
        
        for field_name, field_value in state.items():
            # 跳过field_extra字段本身
            if field_name == 'field_extra':
                continue
                
            # 获取字段类型
            field_type = field_extra.get(field_name, 'unknown')
            
            # 反序列化值
            deserialized_value = StrategyStateSerializer.deserialize_value(field_value, field_type)
            
            # 设置到实例属性
            if hasattr(obj, field_name):
                setattr(obj, field_name, deserialized_value) 

class LeekJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Leek objects.
    Handles Enum, Decimal, datetime, and dataclass types.
    """
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return int(obj.timestamp() * 1000)  # 转换为毫秒时间戳
        # 检查是否为 time.struct_time 对象
        if isinstance(obj, time_module.struct_time):
            return int(time_module.mktime(obj) * 1000)  # 转换为毫秒时间戳
        if is_dataclass(obj):
            return asdict(obj)
        # 处理类对象，返回 modelname|classname 格式
        if isinstance(obj, type):
            module_name = obj.__module__
            class_name = obj.__name__
            return f"{module_name}|{class_name}"
        return super().default(obj)

class LeekJSONDecoder:
    """
    Custom JSON decoder for Leek objects.
    Handles basic type conversion and special value restoration.
    """
    
    @staticmethod
    def decode_basic_type(value_str: str):
        """
        解码基础类型字符串
        
        参数:
            value_str: 序列化的值字符串
            
        返回:
            解码后的Python对象
        """
        if value_str == "true":
            return True
        elif value_str == "false":
            return False
        elif value_str == "null" or value_str == "None":
            return None
        elif value_str.startswith('"') and value_str.endswith('"'):
            # 字符串类型，去掉引号
            return value_str[1:-1]
        else:
            # 尝试解析为数字
            try:
                if '.' in value_str:
                    return float(value_str)
                else:
                    return int(value_str)
            except ValueError:
                # 如果都不是，返回原始字符串
                return value_str
    
    @staticmethod
    def loads(json_str: str):
        """
        解析JSON字符串，处理基础类型的特殊转换
        
        参数:
            json_str: JSON字符串
            
        返回:
            解析后的Python对象
        """
        if not json_str:
            return None
            
        # 如果是基础类型的字符串表示，直接解码
        if json_str in ["true", "false", "null", "None"]:
            return LeekJSONDecoder.decode_basic_type(json_str)
        
        # 如果是数字字符串
        if json_str.replace('.', '').replace('-', '').isdigit():
            return LeekJSONDecoder.decode_basic_type(json_str)
        
        # 如果是带引号的字符串
        if json_str.startswith('"') and json_str.endswith('"'):
            return LeekJSONDecoder.decode_basic_type(json_str)
        
        # 否则使用标准JSON解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回原始字符串
            return json_str