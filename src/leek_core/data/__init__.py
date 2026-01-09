#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据模块，用于处理多个数据源、数据对齐和处理。
"""
from .base import DataSource
from .binance_source import BinanceDataSource
from .clickhouse_source import ClickHouseKlineDataSource
from .context import DataSourceContext
from .gate_source import GateDataSource
from .okx_source import OkxDataSource
from .websocket import WebSocketDataSource
from .redis_clickhouse_source import RedisClickHouseDataSource
__all__ = [
    'DataSourceContext',
    'DataSource',
    'WebSocketDataSource',
    'ClickHouseKlineDataSource',
    'BinanceDataSource',
    'GateDataSource',
    'OkxDataSource',
    'RedisClickHouseDataSource',
]