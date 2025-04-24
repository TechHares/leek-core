#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易平台的数据源抽象定义。
该模块提供了不同数据源的基类和接口。
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Any, Callable, Iterator

from models import DataType, Field, AssetType, Component, Data
from utils import EventSource


class DataSource(Component, ABC):
    # 声明支持的数据类型
    supported_data_type: DataType = DataType.KLINE
    # 声明支持的资产类型
    supported_asset_type: DataType = AssetType.STOCK
    backtest_supported: bool = False
    # 声明显示名称
    verbose_name = "OKX K线"
    """
    所有数据源的抽象基类。

    该类定义了所有数据源必须实现的接口，
    无论其具体实现细节（REST API、WebSocket、基于文件、数据库等）。
    """

    def __init__(self, instance_id: str = None, name: str = None):
        """
        初始化数据源。

        参数:
            instance_id: 数据源实例ID，用于跟踪数据流向
            name: 数据源的名称
        """
        super().__init__(instance_id, name if name else self.__class__.verbose_name)

        self.is_connected = False
        self.callback: Callable|None = None
        self.params_list = None

    def set_callback(self, callback: Callable):
        """
        设置数据源的回调函数。
        参数:
            callback: 回调函数
        """
        self.callback = callback

    def _callback(self, data: Data):
        if self.params_list is None:
            self.params_list = [p.name for p in  self.get_supported_parameters()]

        if self.callback:
            self.callback(EventSource(
                instance_id=self.instance_id,
                name=self.name,
                cls=self.__class__.__name__,
                extra={"params": self.params_list}
            ), data)

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与数据源的连接。

        返回:
            bool: 连接成功返回True，否则返回False
        """
        ...

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与数据源的连接。

        返回:
            bool: 断开连接成功返回True，否则返回False
        """
        ...

    def on_start(self):
        """
        启动数据源。
        """
        if self.is_connected:
            return
        self.is_connected = self.connect()

    def on_stop(self):
        """
        停止数据源。
        """
        if not self.is_connected:
            return
        self.disconnect()
        self.is_connected = False

    def subscribe(self, **kwargs):
        """
        订阅实时K线/蜡烛图更新。

        参数:
            kwargs: 参数

        返回:
            如果不支持实时订阅，则抛出 NotImplementedError 异常
        """
        raise NotImplementedError("数据源不支持实时订阅")

    def unsubscribe(self, **kwargs):
        """
        取消订阅实时K线/蜡烛图更新。

        参数:
            kwargs: 参数
            callback: 回调函数

        返回:
            如果不支持实时订阅，则抛出 NotImplementedError 异常
        """
        raise NotImplementedError("数据源不支持订阅")

    @abstractmethod
    def get_history_data(self,start_time: datetime|int = None, end_time: datetime|int = None, limit: int = None,
                         **kwargs) -> Iterator[Any]:
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
        获取数据源支持的参数。
        返回:
            List[Field]: 参数定义
        """
        pass

    def __str__(self) -> str:
        """数据源的字符串表示。"""
        return f"DataSource({self.name}-{self.instance_id})"

    def __repr__(self) -> str:
        """数据源的详细表示。"""
        return f"DataSource(name={self.name}-{self.instance_id}, connected={self.is_connected})"
