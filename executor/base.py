#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易执行抽象基类
"""

from abc import abstractmethod, ABC
import asyncio
import websockets
from enum import Enum, auto

from models import Component, SubOrder, Order
from utils import get_logger

logger = get_logger(__name__)

class WSStatus(Enum):
    INIT = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    DISCONNECTED = auto()
    RECONNECTING = auto()

class Executor(Component, ABC):
    """
    交易执行抽象基类
    """

    def __init__(self, callback, instance_id: str=None, name: str=None, **kwargs):
        """
        初始化交易器
        
        参数:
            callback: 回调函数，用于处理订单状态变化等信息
        """
        super().__init__(instance_id, name)
        self.callback = callback
        self.orders: dict[str, SubOrder] = {}

    def check_order(self, order: Order) -> bool:
        """
        检查订单是否可执行

        参数:
            order: 订单信息
        返回:
            bool: True 表示可执行，False 表示不可执行
        """
        return True

    @abstractmethod
    def send_order(self, order: SubOrder):
        """
        下单

        参数:
            order: 订单信息
        """
        raise NotImplementedError()


    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str, **kwargs):
        """
        撤单接口，子类需实现。
        :param order_id: 订单ID
        :param symbol: 交易对
        """
        raise NotImplementedError()

    def get_order(self, order_id: str) -> 'SubOrder|None':
        """
        根据订单ID获取订单对象
        """
        return self.orders.get(order_id)

    def update_order(self, order: 'SubOrder'):
        """
        新增或更新订单对象
        """
        self.orders[order.order_id] = order

    def remove_order(self, order_id: str):
        """
        删除已完成（或撤销）订单
        """
        if order_id in self.orders:
            del self.orders[order_id]

    def _trade_callback(self, order):
        """
        交易回调，反馈成交详细等信息。
        若订单已完成或撤销，则自动删除。
        """
        # 用户自定义回调
        if self.callback:
            self.callback(order)
        # 自动清理已完成订单
        if hasattr(order, 'order_id') and hasattr(order, 'state'):
            if order.state in ("filled", "canceled"):  # 成交或撤单即删除
                self.remove_order(order.order_id)


class WebSocketExecutor(Executor, ABC):
    """
    基于websockets实现的异步WebSocket执行器抽象基类。
    具备自动重连、心跳、详细状态、异常处理等功能。
    子类只需实现消息处理、心跳内容等方法。
    """
    def __init__(self, callback, ws_url: str, instance_id: str=None, name: str=None,
                 heartbeat_interval: float=-1, reconnect_interval: float=5, max_retries: int=5, **kwargs):
        super().__init__(callback, instance_id, name, **kwargs)
        self.ws_url = ws_url
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self._ws = None
        self._status = WSStatus.INIT
        self._loop = None
        self._main_task = None
        self._stop_event = asyncio.Event()
        self._current_retries = 0

    @property
    def status(self):
        return self._status

    def on_start(self):
        """
        启动WebSocket连接主循环（同步接口，自动调度异步任务）
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # 启动异步主任务
        self._main_task = loop.create_task(self._run())

    def on_stop(self):
        """
        停止WebSocket连接（同步接口）
        """
        async def _stop_async():
            self._stop_event.set()
            if self._main_task:
                await self._main_task
            await self._close_ws()
            self._status = WSStatus.DISCONNECTED
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(_stop_async())

    async def _run(self):
        self._status = WSStatus.CONNECTING
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    self._ws = ws
                    self._status = WSStatus.CONNECTED
                    self._current_retries = 0
                    await self.on_open()
                    await asyncio.gather(
                        self._recv_loop(),
                        self._heartbeat_loop(),
                        return_exceptions=True
                    )
            except Exception as e:
                await self.on_error(e)
                self._status = WSStatus.RECONNECTING
                self._current_retries += 1
                if self._current_retries > self.max_retries:
                    self._status = WSStatus.DISCONNECTED
                    break
                await asyncio.sleep(self.reconnect_interval)
            finally:
                await self._close_ws()

    async def _recv_loop(self):
        while not self._stop_event.is_set() and self._ws:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat_interval+5)
                await self.on_message(msg)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                await self.on_error(e)
                break

    async def _heartbeat_loop(self):
        if self.heartbeat_interval is None or self.heartbeat_interval < 0:
            # 心跳间隔<0则不发心跳
            return
        while not self._stop_event.is_set() and self._ws:
            try:
                await self.send_heartbeat()
            except Exception as e:
                await self.on_error(e)
            await asyncio.sleep(self.heartbeat_interval)

    async def _close_ws(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            await self.on_close()
            self._ws = None

    async def send(self, msg):
        if self._ws and self._status == WSStatus.CONNECTED:
            await self._ws.send(msg)

    async def send_heartbeat(self):
        """
        发送心跳包，子类可重写实现具体心跳内容。
        """
        ...

    async def on_message(self, msg):
        """
        收到消息时的回调，需由子类实现。
        """
        ...

    async def on_open(self):
        """
        连接建立时回调，子类可选实现。
        """
        ...

    async def on_close(self):
        """
        连接关闭时回调，子类可选实现。
        """
        ...

    async def on_error(self, error):
        """
        错误处理回调，默认打印日志，子类可重写。
        """
        ...
