#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import asyncio
import time
import os
import sys
from queue import Queue
from typing import Dict, Any, List, Set, Optional, Callable
from multiprocessing import Process
from decimal import Decimal
from leek_core.utils.serialization import LeekJSONEncoder, LeekJSONDecoder
from leek_core.utils import get_logger, generate
from leek_core.engine.base import Engine
from leek_core.event import Event, EventType, EventSource
from leek_core.models import LeekComponentConfig, PositionConfig
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.event import EventBus
from leek_core.data import DataSourceContext
from leek_core.strategy import StrategyContext
from leek_core.executor import ExecutorContext
from leek_core.utils import setup_logging, set_worker_id
from leek_core.base import load_class_from_str
from leek_core.alarm import alarm_manager, ErrorAlarmHandler
from concurrent import futures  
import uuid
logger = get_logger(__name__)

try:
    import grpc
    from .grpc.engine_pb2_grpc import *
    from .grpc.engine_pb2 import *
    from .grpc.grpc_server import EngineServiceServicer
except Exception as e:
    logger.error(f"导入grpc模块失败: {e}")

options = [
    # Keepalive: 每 30 秒发送一次 ping，适用于空闲连接保活
    ('grpc.keepalive_time_ms', 30000),

    # Keepalive 响应超时：等待 ping 响应最多 10 秒
    ('grpc.keepalive_timeout_ms', 10000),

    # 允许在没有活跃调用时发送 keepalive ping
    ('grpc.keepalive_permit_without_calls', 1),

    # 允许最多 5 次连续的 ping（即使没有数据流量）
    ('grpc.http2.max_pings_without_data', 5),

    # 最大接收消息大小：100MB（默认通常 4MB，太小）
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),

    # 最大发送消息大小：100MB
    ('grpc.max_send_message_length', 100 * 1024 * 1024),

    # 连接空闲超时（单位 ms），设为 0 表示不超时（可选）
    ('grpc.client_idle_timeout_ms', 600000),  # 10 分钟

    # HTTP/2 连接最大并发流（避免压垮服务）
    ('grpc.http2.max_concurrent_streams', 100),

    # 启用 HTTP/2 ping 的频率控制（可选）
    ('grpc.http2.min_time_between_pings_ms', 10000),
]
class GrpcEngine(Engine):
    """
    独立进程中的交易引擎实现。
    通过gRPC与主进程通信，接收指令并执行。
    主进程挂掉时，gRPC连接断开，子进程自动退出。
    """

    def __init__(self, instance_id: str, name: str, position_config: PositionConfig = None):
        super().__init__(instance_id, name, position_config)
        
        self.engine_server = None
        self.engine_port = None

        self._event_queue = asyncio.Queue()

        try:
            self._event_loop = asyncio.get_event_loop()
        except (RuntimeError, Exception):
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        


    def _handle_event(self, event: Event):
        """处理事件，推送到主进程"""
        try:
            # 创建EventMessage对象
            event_msg = EventMessage(
                project_id=self.instance_id,
                event_type=event.event_type.value if event.event_type else "",
                data_json=json.dumps(event.data, cls=LeekJSONEncoder) if event.data else "{}",
                timestamp=generate(),
                source=json.dumps(event.source, cls=LeekJSONEncoder) if event.source else "{}"
            )
            if self._event_loop.is_running():
                # 如果事件循环正在运行，使用 create_task
                self._event_loop.create_task(self._event_queue.put(event_msg))
            else:
                # 如果事件循环没有运行，使用 run_until_complete
                self._event_loop.run_until_complete(self._event_queue.put(event_msg))
            logger.debug(f"事件已放入队列: {event.event_type.value}")
            
        except Exception as e:
            logger.error(f"处理事件异常: {e}", exc_info=True)

    async def start_engine_server(self) -> int:
        """启动引擎gRPC服务器，返回端口号"""
        logger.info(f"创建异步服务器: {self.instance_id}")
        self.engine_server = grpc.aio.server(options=options)
        add_EngineServiceServicer_to_server(EngineServiceServicer(self), self.engine_server)
        
        # 启动服务器
        server_address = f'[::]:{self.engine_port}'
        self.engine_server.add_insecure_port(server_address)
        logger.info(f"启动服务器: {server_address}")
        
        # 获取事件循环并启动服务器
        await self.engine_server.start()
        
        # 保持服务器运行
        logger.info(f"gRPC服务器已启动，保持运行状态: {self.instance_id}")
        self.on_start()

        # 启动周期性组件检查任务
        self._check_task = None
        
        async def periodic_component_check():
            try:
                while True:
                    await asyncio.sleep(60)  # 每60秒检查一次
                    component_status = self.check_component()
                    logger.info(f"周期性组件状态检查: {component_status}")
            except asyncio.CancelledError:
                logger.info("周期性组件检查任务被取消")
            except Exception as e:
                logger.error(f"周期性组件状态检查失败: {e}", exc_info=True)
        
        # 创建周期性检查任务
        self._check_task = asyncio.create_task(periodic_component_check())

        try:
            await self.engine_server.wait_for_termination()
        except asyncio.exceptions.CancelledError:
            logger.info("收到中断信号，正在关闭服务器...")
        except Exception as e:
            logger.error(f"服务异常: {e}", exc_info=True)
        finally:
            try:
                if self._check_task and not self._check_task.done():
                    self._check_task.cancel()
                    try:
                        await self._check_task
                    except asyncio.CancelledError:
                        pass
                await self.engine_server.stop(grace=5)
                self.on_stop()
            except Exception:
                ...
        

    def start(self) -> None:
        try:
            asyncio.run(self.start_engine_server())
        except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
            logger.info("收到中断信号，正在关闭服务器...")
        except Exception as e:
            logger.error(f"服务异常: {e}", exc_info=True)


    def shutdown(self):
        """关闭引擎"""
        logger.info(f"收到关闭请求: {self.instance_id} {self.name}")
        try:
            self._event_loop.run_until_complete(self.engine_server.stop(grace=5))
        except Exception as e:
            logger.error(f"停止引擎服务器时出错: {e}")
        finally:
            self._event_loop.close()
        self.on_stop()

class GrpcEngineClient():
    """主进程的 gRPC 客户端"""

    def __init__(self, instance_id: str, name: str, config=None):
        super().__init__()
        self.instance_id = instance_id
        self.name = name
        self.config = config or {}
        
        self.process = None
        self.engine_channel = None
        self.engine_stub = None
        self.engine_port = None
        self._running = False
        self._handlers: Dict[EventType, Callable] = {}
        
        # 事件监听相关
        self._event_listener_thread = None
        self._event_loop = None

    def register_handler(self, event_type: EventType, handler: Callable):
        """注册动作处理器"""
        # 将 action 转为大写，方便回调
        self._handlers[event_type] = handler

    def unregister_handler(self, event_type: EventType):
        """注销动作处理器"""
        # 将 action 转为大写
        if event_type in self._handlers:
            del self._handlers[event_type]

    def call_handler(self, action: str, *args, **kwargs):
        """调用注册的处理器"""
        # 将 action 转为大写
        action_upper = action.upper()
        if action_upper in self._handlers:
            try:
                return self._handlers[action_upper](*args, **kwargs)
            except Exception as e:
                logger.error(f"调用处理器 {action_upper} 失败: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"未找到处理器: {action_upper}")
            return None

    async def invoke(self, action: str, *args, **kwargs):
        """调用引擎"""
        if self.engine_stub is None:
            raise BaseException(f"引擎未连接: {self.instance_id}")
        logger.debug(f"调用引擎: {action} {args} {kwargs}")
        req = ActionRequest(
            project_id=self.instance_id,
            action=action.upper(),
            args_json=json.dumps(args, cls=LeekJSONEncoder),
            kwargs_json=json.dumps(kwargs, cls=LeekJSONEncoder),
            request_id=str(uuid.uuid4())
        )
        response = await self.engine_stub.ExecuteAction(req)
        logger.debug(f"收到引擎响应: {response}")
        if not response.success:
            raise BaseException(f"调用引擎失败: {response.error}")
        
        # 使用LeekJSONDecoder解包基础类型
        return LeekJSONDecoder.loads(response.result_json)


    async def start(self):
        """启动子进程"""
        if self.process and self.process.is_alive():
            return
        
        # 主进程先找一个可用端口
        self._find_available_port()
        if not self.engine_port:
            logger.error(f"无法找到可用端口: {self.instance_id}")
            return False
        
        # 启动子进程，传入端口号
        self.process = Process(
            target=self._start_engine,
            args=(
                self.instance_id,
                self.name,
                self.config,
                self.engine_port  # 传入端口号
            ),
            name=f"{self.name}-{self.instance_id}",
            daemon=True
        )
        
        self.process.start()
        time.sleep(2)
        self.engine_channel = grpc.aio.insecure_channel(f'localhost:{self.engine_port}')
        self.engine_stub = EngineServiceStub(self.engine_channel)
        # 等待子进程启动并尝试连接
        await self.invoke("ping", instance_id=self.instance_id)
        self._running = True
        
        # 启动事件监听线程
        self._start_event_listener()
        logger.info(f"子进程启动成功: {self.instance_id} (端口: {self.engine_port})")
        return True

    def _find_available_port(self, start_port: int = 50052, max_attempts: int = 100):
        """主进程查找可用端口"""
        import socket
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    self.engine_port = port
            except OSError:
                continue

    async def stop(self):
        """停止客户端（同步包装器）"""
        self._running = False
        
        # 停止事件监听任务
        if hasattr(self, '_event_listener_task') and not self._event_listener_task.done():
            logger.info(f"停止事件监听任务: {self.instance_id}")
            self._event_listener_task.cancel()
            try:
                await self._event_listener_task
            except asyncio.CancelledError:
                pass
        
        # 关闭异步 gRPC 连接
        logger.info(f"关闭异步 gRPC 连接: {self.instance_id}")
        try:
            await self.invoke("shutdown")
        except Exception as e:
            logger.warning(f"调用 shutdown 失败，可能连接已断开: {e.message}")
        
        try:
            if self.engine_channel:
                await self.engine_channel.close()
                self.engine_channel = None
                self.engine_stub = None
                logger.info(f"关闭 gRPC 连接: {self.instance_id}")
        except Exception as e:
            logger.warning(f"关闭 gRPC 连接时出错: {e}")
        
        # 停止子进程
        if self.process and self.process.is_alive():
            logger.info(f"停止子进程: {self.instance_id} (PID: {self.process.pid})")
            try:
                self.process.terminate()
                # 等待进程结束
                self.process.join(timeout=5)
                if self.process.is_alive():
                    logger.warning(f"子进程未在5秒内结束，强制杀死: {self.instance_id}")
                    self.process.kill()
                    self.process.join()
            except Exception as e:
                logger.error(f"停止子进程异常: {e}")
        
        logger.info(f"客户端已停止: {self.instance_id}")

    def is_alive(self):
        """检查子进程是否存活"""
        return self.process and self.process.is_alive()

    # 便捷方法
    async def add_strategy(self, config: LeekComponentConfig):
        return await self.invoke("add_strategy", config)

    async def update_strategy(self, config: LeekComponentConfig):
        return await self.invoke("update_strategy", config)

    async def remove_strategy(self, instance_id):
        return await self.invoke("remove_strategy", instance_id)

    async def add_executor(self, config: LeekComponentConfig):
        return await self.invoke("add_executor", config)

    async def update_executor(self, config: LeekComponentConfig):
        return await self.invoke("update_executor", config)

    async def remove_executor(self, instance_id):
        return await self.invoke("remove_executor", instance_id)

    async def add_data_source(self, config: LeekComponentConfig):
        return await self.invoke("add_data_source", config)

    async def update_data_source(self, config: LeekComponentConfig):
        return await self.invoke("update_data_source", config)

    async def remove_data_source(self, instance_id):
        return await self.invoke("remove_data_source", instance_id)

    async def update_position_config(self, position_config, data=None) -> None:
        return await self.invoke("update_position_config", position_config, data)

    async def get_status(self):
        """获取引擎状态"""
        return await self.invoke("engine_state")

    def _start_event_listener(self):
        """启动事件监听"""
        if hasattr(self, '_event_listener_task') and not self._event_listener_task.done():
            return
        
        # 在主线程的事件循环中启动监听任务
        self._event_listener_task = asyncio.create_task(self._listen_events())
        logger.info(f"事件监听任务已启动: {self.instance_id}")

    async def _listen_events(self):
        """监听事件流"""
        try:
            logger.info(f"开始监听引擎事件流: {self.instance_id} {self.name}")
            # 创建监听请求
            listen_request = ListenRequest(
                project_id=self.instance_id
            )
            
            # 调用ListenEvents流
            async for event_msg in self.engine_stub.ListenEvents(listen_request):
                try:
                    # 处理接收到的事件
                    await self._handle_received_event(event_msg)
                except Exception as e:
                    logger.error(f"处理接收到的事件失败: {e}", exc_info=True)
                    
        except Exception as e:
            # 如果连接断开，尝试重连
            await asyncio.sleep(1)
            if self._running:
                logger.error(f"监听事件流异常: {e}", exc_info=True)
                logger.info(f"尝试重新连接事件流: {self.instance_id}")
                await asyncio.sleep(5)  # 等待5秒后重试
                await self._listen_events()

    async def _handle_received_event(self, event_msg: EventMessage):
        """处理接收到的事件"""
        try:
            project_id = event_msg.project_id
            event_type = event_msg.event_type
            data_json = event_msg.data_json
            timestamp = event_msg.timestamp
            source = event_msg.source
            # 跳过心跳事件
            if event_type == "heartbeat":
                logger.debug(f"收到心跳事件: {project_id}")
                return
            logger.info(f"收到子进程事件[{project_id}-{timestamp}]:  type={event_type} source={source} data={data_json}")
            # 解析事件数据
            try:
                data = json.loads(data_json) if data_json else {}
                source = json.loads(source) if source and source != "{}" else None
            except json.JSONDecodeError as e:
                logger.error(f"事件数据解析失败: {e}")
                return
            
            # 检查事件类型是否有效
            if event_type not in [et.value for et in EventType]:
                logger.warning(f"未知的事件类型: {event_type}")
                return
            # 创建事件源对象
            event_source = None
            if source:
                event_source = EventSource(
                    instance_id=source.get('instance_id', ''),
                    name=source.get('name', ''),
                    cls=source.get('cls', ''),
                    extra=source.get('extra', {})
                )
            
            event = Event(
                event_type=EventType(event_type),
                data=data,
                source=event_source,
            )
            await self._handle_event(event)
        except Exception as e:
            logger.error(f"处理接收到的事件异常: {e}", exc_info=True)

    async def _handle_event(self, event: Event):
        """处理事件（可被子类重写）"""
        try:
            if event.event_type in self._handlers:
                self._handlers[event.event_type](int(self.instance_id), event)
                logger.info(f"处理事件完成: {event.event_type} - {event.data} - {event.source}")
        except Exception as e:
            logger.error(f"处理事件异常: {event.event_type} - {event.data} - {event.source}: {e}", exc_info=True)
            raise e

    @staticmethod
    def _start_engine(instance_id: str, name: str, config: dict, port: int):
        """在子进程中启动引擎"""
        try:
            # 设置日志格式
            log_level = config.get('log_level', 'INFO')
            log_format = config.get('log_format', 'json')
            log_alarm = config.get('log_alarm', False)
            set_worker_id(int(instance_id))

            # 准备 external_handlers
            external_handlers = []
            if log_alarm:
                external_handlers.append(ErrorAlarmHandler())

            # 调用 setup_logging
            try:
                setup_logging(
                    level=log_level,
                    format_type=log_format,
                    external_handlers=external_handlers if external_handlers else None
                )
            except Exception as e:
                logger.error(f"设置日志失败: {e}")
            
            # 设置报警器
            alert_config = config.get('alert_config', [])
            for alert_item in alert_config:
                if alert_item.get('enabled'):
                    alert_class = alert_item.get('class_name')
                    alarm_manager.register_cls(load_class_from_str(alert_class), alert_item.get('config', {}))
                    
            # 加载 mount_dirs
            mount_dirs = config.get('mount_dirs', [])
            for dir_path in mount_dirs:
                if dir_path == "default":
                    continue
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    if dir_path not in sys.path:
                        sys.path.append(dir_path)
                else:
                    logger.error(f"目录不存在或无法访问: {dir_path}")
            
            # 设置仓位配置
            position_setting = config.get('position_setting', {})
            position_setting['data'] = config.get('position_data', None)
            try:
                # 检查 risk_policies 是否存在，如果不存在则设为空列表
                risk_policies = position_setting.get('risk_policies', [])
                position_setting['risk_policies'] = [LeekComponentConfig(
                    instance_id=instance_id,
                    name=policy.get('name'),
                    cls=load_class_from_str(policy.get('class_name')),
                    config=policy.get('params')) for policy in risk_policies if policy.get('enabled')]
                
                # 为 PositionConfig 提供默认值
                position_setting.setdefault('init_amount', Decimal('100000'))
                position_setting.setdefault('max_strategy_amount', Decimal('50000'))
                position_setting.setdefault('max_strategy_ratio', Decimal('0.5'))
                position_setting.setdefault('max_symbol_amount', Decimal('25000'))
                position_setting.setdefault('max_symbol_ratio', Decimal('0.25'))
                position_setting.setdefault('max_amount', Decimal('10000'))
                position_setting.setdefault('max_ratio', Decimal('0.1'))
                
                position_config = PositionConfig(**position_setting)
            except Exception as e:
                logger.error(f"设置仓位配置时出错: {e}", exc_info=True)
                return
            
            # 创建引擎
            engine = GrpcEngine(
                instance_id=instance_id,
                name=name,
                position_config=position_config
            )
            
            # 设置端口号
            engine.engine_port = port
            
            # 运行引擎（这会处理所有异步操作，包括启动 gRPC 服务器）
            engine.start()
        except Exception as e:
            logger.error(f"子进程启动失败: {e}", exc_info=True)