#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易平台的数据源抽象定义。
该模块提供了不同数据源的基类和接口。
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Any, Callable, Iterator

from leek_core.base import LeekComponent
from leek_core.models import DataType, Field, AssetType, Data


class DataSource(LeekComponent, ABC):
    # 声明支持的数据类型
    supported_data_type: DataType = DataType.KLINE
    # 声明支持的资产类型
    supported_asset_type: DataType = AssetType.STOCK
    # 是否支持回测
    backtest_supported: bool = False
    # 声明显示名称
    display_name = "X数据源"
    """
    所有数据源的抽象基类。

    该类定义了所有数据源必须实现的接口，
    无论其具体实现细节（REST API、WebSocket、基于文件、数据库等）。
    """

    def __init__(self):
        """
        初始化数据源。

        参数:
            instance_id: 数据源实例ID，用于跟踪数据流向
            name: 数据源的名称
        """
        super().__init__()

        self.callback: Callable | None = None

    def send_data(self, data: Data):
        self.callback(data)

    @abstractmethod
    def parse_row_key(self, **kwargs) -> List[tuple]:
        """
        解析行键为参数。
        参数:
            kwargs: 参数
        返回:
            tuple 数据键 和 Data定义保持一致 会用于subscribe和unsubscribe
        """
        pass

    def subscribe(self, row_key: str):
        """
        订阅实时K线/蜡烛图更新。

        参数:
            row_key: 参数

        返回:
            如果不支持实时订阅，则抛出 NotImplementedError 异常
        """
        raise NotImplementedError("数据源不支持实时订阅")

    def unsubscribe(self, row_key: str):
        """
        取消订阅实时K线/蜡烛图更新。

        参数:
            row_key: 参数

        返回:
            如果不支持实时订阅，则抛出 NotImplementedError 异常
        """
        raise NotImplementedError("数据源不支持订阅")

    @abstractmethod
    def get_history_data(self, row_key: str, start_time: datetime | int = None, end_time: datetime | int = None, limit: int = None) -> Iterator[Any]:
        """
        获取历史K线/蜡烛图数据。

        参数:
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            kwargs: 其它参数

        返回:
            Iterator[Any]: 数据迭代器
        """
        pass

    @abstractmethod
    def get_supported_parameters(self) -> List[Field]:
        """
        获取数据源订阅支持的参数。
        返回:
            List[Field]: 参数定义
        """
        pass
