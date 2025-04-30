#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读数据源的公共实现， 比如WebSocket和文件等。
"""
import asyncio
import threading
from abc import ABC, abstractmethod
import time
from typing import List

import websockets

from models import Field, FieldType
from .base import DataSource
from utils import get_logger

logger = get_logger(__name__)


class WebSocketDataSource(DataSource, ABC):
    """
    WebSocket数据源基类。

    专注于管理WebSocket连接的生命周期和通讯，
    子类需要实现on_message等方法来处理具体的业务逻辑。
    """
    init_params: List[Field] = [
        Field(name="ws_url",
              type=FieldType.STRING,
              required=True,
              default=None,
              description="WebSocket服务器地址"),
    ]

    def __init__(self, ws_url: str):
        """
        初始化WebSocket数据源。

        参数:
            name: 数据源名称
            ws_url: WebSocket服务器地址
            instance_id: 数据源实例ID，用于跟踪数据流向
        """
        super().__init__()
        self.ws_url = ws_url
        self._connection = None
        self._listener_task = None
        self._loop = None
        self._thread = None
        self._conn_event = None
        self._lock = threading.RLock()

    def on_start(self):
        """
        连接到WebSocket服务器。

        返回:
            bool: 连接成功返回True，否则返回False
        """
        try:
            # 创建异步事件循环和事件
            self._loop = asyncio.new_event_loop()
            self._conn_event = threading.Event()

            # 启动后台线程运行异步连接
            self._thread = threading.Thread(target=self._run_async_loop)
            self._thread.daemon = True
            self._thread.start()

            # 等待连接完成或超时
            if not self._conn_event.wait(timeout=10.0):
                logger.error(f"WebSocket数据源'{self.ws_url}'连接超时")
                self._cleanup_resources()
                return False

        except Exception as e:
            logger.error(f"WebSocket数据源'{self.ws_url}'连接失败: {e}", exc_info=True)
            self._cleanup_resources()

    def _run_async_loop(self):
        """在后台线程中运行异步事件循环"""
        try:
            self._loop.run_until_complete(self._async_connect())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket异步循环出错: {e}")
        finally:
            self._cleanup_loop()

    async def _async_connect(self):
        """异步连接实现"""
        try:
            self._connection = await websockets.connect(self.ws_url)
            # 启动监听任务
            self._listener_task = asyncio.create_task(self._listen())
            logger.info(f"WebSocket数据源'{self.ws_url}'连接成功: {self.ws_url}")
            self.on_connect()
        except Exception as e:
            logger.error(f"WebSocket数据源'{self.ws_url}'异步连接失败: {e}")
            self._connection = None
        finally:
            # 通知连接过程已完成
            self._conn_event.set()

    def on_stop(self):
        """
        断开与WebSocket服务器的连接。

        返回:
            bool: 断开连接成功返回True，否则返回False
        """
        try:
            # 先调用子类的断开处理
            self.on_disconnect()
            logger.debug(f"WebSocket数据源'{self.ws_url}'开始断开连接...")

            # 直接在主线程中进行资源清理，不依赖异步操作
            # 这样可以避免事件循环死锁或超时问题
            self._cleanup_resources()
            logger.info(f"WebSocket数据源'{self.ws_url}'已断开连接")
        except Exception as e:
            logger.error(f"WebSocket数据源'{self.ws_url}'断开连接失败: {e}", exc_info=True)
            # 强制清理资源
            try:
                self._cleanup_resources()
            except Exception as cleanup_error:
                logger.error(f"清理资源时出错: {cleanup_error}", exc_info=True)

    def _cleanup_resources(self):
        """清理资源"""
        # 取消监听任务
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            logger.debug(f"已取消WebSocket监听任务: {self.ws_url}")

        # 尝试关闭WebSocket连接
        if self._connection:
            try:
                # 在主线程中无法使用await，但可以检查是否已关闭
                if hasattr(self._connection, "closed") and not self._connection.closed:
                    logger.debug(f"WebSocket连接未关闭，将在事件循环停止时自动关闭: {self.ws_url}")
            except Exception as e:
                logger.error(f"检查WebSocket连接状态时出错: {e}", exc_info=True)

        # 停止事件循环
        if self._loop and self._loop.is_running():
            try:
                # 在事件循环中安排取消所有任务
                if self._loop.is_running():
                    try:
                        # 必须使用call_soon_threadsafe，因为我们从不同的线程调用
                        self._loop.call_soon_threadsafe(
                            lambda: asyncio.ensure_future(self._cancel_all_tasks(), loop=self._loop)
                        )
                        logger.debug(f"已安排取消所有任务: {self.ws_url}")
                    except Exception as e:
                        logger.error(f"安排取消任务时出错: {e}", exc_info=True)

                # 延迟一小段时间，确保取消操作有时间执行
                time.sleep(0.1)

                # 停止事件循环（必须使用call_soon_threadsafe）
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.on_stop)
                    logger.debug(f"已安排停止事件循环: {self.ws_url}")
            except Exception as e:
                logger.error(f"停止事件循环失败: {e}", exc_info=True)

        # 等待线程结束
        if self._thread and self._thread.is_alive():
            # 给线程一个合理的时间来清理
            join_timeout = 5.0
            logger.debug(f"等待WebSocket线程结束，超时时间: {join_timeout}秒")
            self._thread.join(timeout=join_timeout)

            if self._thread.is_alive():
                logger.error(f"WebSocket线程'{self.ws_url}'未能在{join_timeout}秒内结束，可能导致资源泄漏")
                # 最后的紧急手段 - 强制结束线程（可选，取决于您的风险承受能力）
                # import ctypes
                # ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._thread.ident),
                #                                         ctypes.py_object(SystemExit))
            else:
                logger.debug(f"WebSocket线程'{self.ws_url}'已成功结束")

        # 清理状态
        self._connection = None
        self._loop = None
        self._thread = None
        self._conn_event = None
        self._listener_task = None

    async def _cancel_all_tasks(self):
        """取消事件循环中的所有任务"""
        try:
            # 获取事件循环中的所有任务
            tasks = [task for task in asyncio.all_tasks(self._loop)
                     if task is not asyncio.current_task(self._loop)]

            if not tasks:
                return

            logger.debug(f"正在取消{len(tasks)}个任务")

            # 关闭WebSocket连接（如果存在）
            if self._connection and hasattr(self._connection, "close"):
                try:
                    await self._connection.close()
                    logger.debug(f"已关闭WebSocket连接: {self.ws_url}")
                except Exception as e:
                    logger.error(f"关闭WebSocket连接时出错: {e}", exc_info=True)

            # 取消所有任务
            for task in tasks:
                task.cancel()

            # 等待所有任务取消完成
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"所有任务已取消: {self.ws_url}")
        except Exception as e:
            logger.error(f"取消任务时出错: {e}", exc_info=True)

    def _cleanup_loop(self):
        """清理事件循环"""
        # 关闭所有未完成的任务
        try:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()

            # 关闭事件循环
            self._loop.close()
        except Exception as e:
            logger.error(f"清理事件循环时出错: {e}")

    async def _listen(self):
        """监听WebSocket消息"""
        if not self._connection:
            logger.error("WebSocket连接不存在，无法监听消息")
            return

        try:
            # 使用websockets推荐的方式接收消息
            while True:
                try:
                    message = await self._connection.recv()
                    try:
                        # 将消息交给子类处理
                        await self.on_message(message)
                    except Exception as e:
                        logger.error(f"处理WebSocket消息时出错: {e}", exc_info=True)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket连接已关闭: {e}")
                    await self.on_connection_closed(e)
                    break
        except Exception as e:
            logger.error(f"WebSocket监听循环中发生未知错误: {e}", exc_info=True)

    @abstractmethod
    async def on_message(self, message: str):
        """
        处理收到的WebSocket消息。
        子类必须实现此方法。

        参数:
            message: 收到的消息
        """
        raise NotImplementedError("子类必须实现on_message方法")

    def on_connect(self):
        """
        连接成功后调用。
        子类可以覆盖此方法实现自己的连接后处理逻辑。
        """
        pass

    def on_disconnect(self):
        """
        断开连接前调用。
        子类可以覆盖此方法实现自己的断开连接前处理逻辑。
        """
        pass

    async def on_connection_closed(self, exception: Exception):
        """
        连接关闭时调用。
        子类可以覆盖此方法实现自己的连接关闭处理逻辑。

        参数:
            exception: 连接关闭的异常
        """
        pass

    async def send(self, message: str) -> bool:
        """
        发送消息到WebSocket服务器。

        参数:
            message: 要发送的消息

        返回:
            bool: 发送成功返回True，否则返回False
        """
        if not self._connection:
            logger.error("WebSocket未连接，无法发送消息")
            return False

        try:
            await self._connection.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.error("发送消息失败，连接已关闭")
            return False
        except Exception as e:
            logger.error(f"发送WebSocket消息时出错: {e}")
            return False

    def async_send(self, message: str) -> bool:
        """
        从同步上下文发送消息到WebSocket服务器。

        参数:
            message: 要发送的消息

        返回:
            bool: 发送成功返回True，否则返回False
        """
        if self._loop:
            logger.error("WebSocket未连接，无法发送消息")
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.send(message),
                self._loop
            )
            return future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"异步发送WebSocket消息时出错: {e}")
            return False




if __name__ == '__main__':
    pass
