#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易执行抽象基类
"""
import asyncio
import threading
from abc import abstractmethod, ABC
from enum import Enum, auto
from typing import List

import websockets

from leek_core.base import LeekComponent
from leek_core.models import Order, Field, FieldType, OrderUpdateMessage
from leek_core.utils import get_logger

logger = get_logger(__name__)

class WSStatus(Enum):
    INIT = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    DISCONNECTED = auto()
    RECONNECTING = auto()

class Executor(LeekComponent, ABC):
    """
    交易执行抽象基类
    """
    just_backtest = False  # 仅用于回测，不实际执行

    def __init__(self):
        """
        初始化交易器
        
        参数:
            callback: 回调函数，用于处理订单状态变化等信息
        """
        self.instance_id = None
        self.callback = None

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
    def send_order(self, order: Order|List[Order]):
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

    def _trade_callback(self, order_update_message: OrderUpdateMessage):
        """
        交易回调，反馈成交详细等信息。
        若订单已完成或撤销，则自动删除。
        """
        # 用户自定义回调
        if self.callback:
            self.callback(order_update_message)


class WebSocketExecutor(Executor, ABC):
    """
    基于websockets实现的异步WebSocket执行器抽象基类。
    具备自动重连、心跳、详细状态、异常处理等功能。
    子类只需实现消息处理、心跳内容等方法。
    """
    init_params = [
        Field(name="ws_url", label="WebSocket URL", type=FieldType.STRING, required=True, description="WebSocket连接URL"),
        Field(name="heartbeat_interval", label="心跳间隔", type=FieldType.FLOAT, default=-1, min=-1, description="心跳间隔，单位秒，-1表示不发送心跳"),
        Field(name="reconnect_interval", label="重连间隔", type=FieldType.FLOAT, default=5, min=1, description="重连间隔，单位秒"),
        Field(name="max_retries", label="最大重连次数", type=FieldType.INT, default=5, min=0, description="最大重连次数")
    ]

    def __init__(self, ws_url: str, heartbeat_interval: float=-1, reconnect_interval: float=5, max_retries: int=5, **kwargs):
        super().__init__()
        self.ws_url = ws_url
        self.heartbeat_interval = float(heartbeat_interval)
        self.reconnect_interval = float(reconnect_interval)
        self.max_retries = max_retries
        self._ws = None
        self._status = WSStatus.INIT
        self._loop = None
        self._loop_thread = None
        self._stop_event = asyncio.Event()
        self._current_retries = 0
        self._conn_event = None

    @property
    def status(self):
        return self._status

    def on_start(self):
        """
        启动WebSocket连接主循环（同步接口，自动调度异步任务）
        """
        self._conn_event = threading.Event()
        self._loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._loop_thread.start()
        # 等待连接完成或超时
        if not self._conn_event.wait(timeout=15.0):
            raise TimeoutError(f"WebSocket数据源'{self.ws_url}'连接超时")

    def _run_async_loop(self):
        """在后台线程中运行异步事件循环"""
        try:
            logger.info(f"[WebSocketExecutor] on_start: 启动WebSocket主循环，目标URL: {self.ws_url}")
            # 创建异步事件循环和事件
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._loop.run_until_complete(self._run())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket异步循环出错: {e}", exc_info=True)

    def on_stop(self):
        """
        停止WebSocket连接（同步接口）
        """
        logger.warning("[WebSocketExecutor] on_stop: 停止WebSocket主循环")
        async def _stop_async():
            self._stop_event.set()
            await self._close_ws()
            self._status = WSStatus.DISCONNECTED

        asyncio.run_coroutine_threadsafe(_stop_async(), self._loop).result()

    async def _run(self):
        self._status = WSStatus.CONNECTING
        logger.info(f"[WebSocketExecutor] _run: 尝试连接 {self.ws_url}")
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    self._ws = ws
                    self._status = WSStatus.CONNECTED
                    self._current_retries = 0
                    logger.info(f"[WebSocketExecutor] _run: 连接成功 {self.ws_url}")
                    self._conn_event.set()
                    try:
                        await asyncio.gather(
                            self.on_open(),
                            self._recv_loop(),
                            self._heartbeat_loop()
                        )
                    except Exception as e:
                        logger.warning(f"[WebSocketExecutor] _run: 任务异常 {e}, 当前重试次数: {self._current_retries}")
                        await self.on_error(e)
                        raise  # 重新抛出异常以触发重连
            except Exception as e:
                logger.warning(f"[WebSocketExecutor] _run: 连接异常 {e}, 当前重试次数: {self._current_retries}")
                await self.on_error(e)
                self._status = WSStatus.RECONNECTING
                self._current_retries += 1
                if self._current_retries > self.max_retries:
                    logger.error(f"[WebSocketExecutor] _run: 超过最大重连次数({self.max_retries})，断开连接")
                    self._status = WSStatus.DISCONNECTED
                    break
                await asyncio.sleep(self.reconnect_interval)
            finally:
                await self._close_ws()

    async def _recv_loop(self):
        logger.debug("[WebSocketExecutor] _recv_loop: 启动消息接收循环")
        while not self._stop_event.is_set() and self._ws:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat_interval+5)
                logger.debug(f"[WebSocketExecutor] _recv_loop: 收到消息 {msg}")
                await self.on_message(msg)
            except asyncio.TimeoutError:
                logger.debug("[WebSocketExecutor] _recv_loop: 接收超时，继续监听")
                continue
            except websockets.ConnectionClosed as e:
                logger.info(f"[WebSocketExecutor] _recv_loop: WebSocket连接关闭: {e}")
                break
            except Exception as e:
                logger.error(f"[WebSocketExecutor] _recv_loop: 接收异常: {e}", exc_info=True)
                await self.on_error(e)
                break

    async def _heartbeat_loop(self):
        if self.heartbeat_interval is None or self.heartbeat_interval < 0:
            # 心跳间隔<0则不发心跳
            logger.warning("[WebSocketExecutor] _heartbeat_loop: 心跳功能关闭")
            return
        logger.debug(f"[WebSocketExecutor] _heartbeat_loop: 启动心跳循环，间隔 {self.heartbeat_interval}s")
        while not self._stop_event.is_set() and self._ws:
            await self.send_heartbeat()
            logger.debug("[WebSocketExecutor] _heartbeat_loop: 已发送心跳")
            await asyncio.sleep(self.heartbeat_interval)

    async def _close_ws(self):
        if self._ws:
            try:
                await self._ws.close()
                logger.warning("[WebSocketExecutor] _close_ws: WebSocket已关闭")
            except Exception as e:
                logger.error(f"[WebSocketExecutor] _close_ws: 关闭异常: {e}")
            await self.on_close()
            self._ws = None

    async def send(self, msg):
        if self._ws and self._status == WSStatus.CONNECTED:
            logger.debug(f"[WebSocketExecutor] send: 发送消息 {msg}")
            await self._ws.send(msg)

    async def send_heartbeat(self):
        """
        发送心跳包，子类可重写实现具体心跳内容。
        """
        logger.debug("[WebSocketExecutor] send_heartbeat: 默认心跳，未实现")
        ...

    async def on_message(self, msg):
        """
        收到消息时的回调，需由子类实现。
        """
        logger.debug(f"[WebSocketExecutor] on_message: 收到消息 {msg}")
        ...

    async def on_open(self):
        """
        连接建立时回调，子类可选实现。
        """
        logger.info("[WebSocketExecutor] on_open: 连接建立")
        ...

    async def on_close(self):
        """
        连接关闭时回调，子类可选实现。
        """
        logger.info("[WebSocketExecutor] on_close: 连接关闭")
        ...

    async def on_error(self, error):
        """
        错误处理回调，默认打印日志，子类可重写。
        """
        logger.error(f"[WebSocketExecutor] on_error: {error}")
        ...

    def async_send(self, message: str) -> bool:
        """
        从同步上下文发送消息到WebSocket服务器。

        参数:
            message: 要发送的消息

        返回:
            bool: 发送成功返回True，否则返回False
        """
        if self._loop is None:
            logger.error("WebSocket未连接，无法发送消息")
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.send(message),
                self._loop
            )
            return future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"异步发送WebSocket消息时出错: {e}", exc_info=True)
            return False
