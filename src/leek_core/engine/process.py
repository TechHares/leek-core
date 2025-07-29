from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import time
from typing import Optional, Any, Dict, List, Callable, Set
import uuid
import inspect
import psutil
from threading import Lock
from functools import wraps
from decimal import Decimal

from leek_core.alarm import ErrorAlarmHandler
from .base import Engine
from leek_core.data import DataSource, DataSourceContext
from leek_core.event import EventBus, Event, EventType
from leek_core.executor import Executor, ExecutorContext
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.models import LeekComponentConfig, StrategyConfig, PositionConfig
from leek_core.strategy import Strategy, StrategyContext
from leek_core.utils import get_logger, run_func_timeout
from leek_core.alarm import AlarmSender, alarm_manager
from leek_core.base import create_component, load_class_from_str
import asyncio
from leek_core.utils import setup_logging, set_worker_id
import sys
import os
import pickle

logger = get_logger(__name__)

class ProcessEngine(Engine):
    """
    独立进程中的交易引擎实现。
    通过Pipe与主进程通信，接收指令并执行。
    主进程挂掉时，Pipe断开，子进程自动退出。
    """

    def __init__(self, conn: Connection, instance_id: str, name: str, position_config: PositionConfig = None, event_hook: Set[EventType] = None):
        super().__init__()
        self.conn = conn  # 用于收发主进程消息
        self.running = True
        self.instance_id = instance_id
        self.name = name
        self.position_config = position_config
        self.response_handlers: Dict[str, Callable] = {}  # 存储响应处理器，按消息ID索引
        self.__send_lock = Lock()

        self.event_bus = EventBus()
        self.event_hook = event_hook or set()
        self.data_source_manager: DataManager = DataManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-数据源管理",
                cls=DataSourceContext,
                config=None
            ))
        self.data_source_manager.on_start()
        self.strategy_manager: StrategyManager = StrategyManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-策略管理",
                cls=StrategyContext,
                config=None
            ))
        self.strategy_manager.on_start()
        self.position_manager: PositionManager = PositionManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-仓位管理",
                cls=None,
                config=position_config,
                data=position_config.data
            ))
        self.position_manager.on_start()
        self.executor_manager: ExecutorManager = ExecutorManager(
            self.event_bus, LeekComponentConfig(
                instance_id=instance_id,
                name=name + "-执行器管理",
                cls=ExecutorContext,
                config=None
            ))
        self.executor_manager.on_start()

    def handle_event(self, event: Event):
        if event.event_type in self.event_hook:
            self.send_msg("event", event=event)

    def start(self) -> None:
        pass

    def run(self):
        """
        启动引擎主循环，监听主进程指令。
        收到'shutdown'或管道断开时自动退出。
        """
        self.running = True
        self.event_bus.subscribe_event(None, self.handle_event)
        self.start()
        try:
            while self.running:
                if self.conn.poll(5):  # 1秒超时，非阻塞检查
                    msg = self.conn.recv()
                    self.handle_message(msg)
        except (EOFError, KeyboardInterrupt) as e:
            logger.warning(f"引擎退出: {e}")
        except Exception as e:
            logger.error(f"引擎运行时出错: {e}", exc_info=True)
        
        self.on_stop()

    def position_image(self):
        state = self.position_manager.get_state()
        self.send_msg("position_image", data=state)
        
    def storage_postion(self):
        """
        存储仓位
        """
        return self.position_manager.get_state()

    def storage_strategy(self):
        """
        存储策略
        """
        for instance_id, strategy in self.strategy_manager.components.items():
            state = strategy.get_state()
            self.send_msg("strategy_data", strategy_id=instance_id, data=state)
    
    def update_strategy_state(self, instance_id: str, state: Dict):
        """
        更新策略
        """
        self.strategy_manager.update_state(instance_id, state)

    def handle_message(self, msg: Dict):
        """
        处理主进程发来的消息，自动分发到对应方法
        """
        if not isinstance(msg, dict) or "action" not in msg:
            logger.error(f"未知消息: {msg}")
            return
            
        action = msg["action"]
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})
        msg_id = msg.get("msg_id")
        
        # 反序列化参数
        args = [arg for arg in args]
        kwargs = {k: v for k, v in kwargs.items()}
        
        if hasattr(self, action):
            method = getattr(self, action)
            try:
                result = method(*args, **kwargs)
                # 如果是invoke请求（有msg_id），则发送响应
                if msg_id:
                    self.send_msg("response", msg_id=msg_id, result=result)
            except Exception as e:
                logger.error(f"处理命令 {action} 时出错，参数{args} {kwargs}，错误信息: {e}", exc_info=True)
                if msg_id:
                    self.send_msg("response", msg_id=msg_id, error=str(e))
        else:
            logger.error(f"未知action: {action}")
            if msg_id:
                self.send_msg("response", msg_id=msg_id, error=f"未知action: {action}")

    def engine_state(self):
        """
        返回引擎当前状态，包括数据源、策略、仓位、执行器等
        """
        # 获取系统信息
        process = psutil.Process(os.getpid())
        process_id = process.pid

        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0)

        # 获取内存使用情况
        memory = psutil.virtual_memory()
        mem_used = round(memory.used / (1024 ** 3), 1)  # GB
        mem_total = round(memory.total / (1024 ** 3), 1)  # GB
        mem_percent = round((memory.used / memory.total) * 100, 1)

        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_used = round(disk.used / (1024 ** 3), 1)  # GB
        disk_total = round(disk.total / (1024 ** 3), 1)  # GB
        disk_percent = disk.percent
        return {
            "state": {
                "process_id": process_id,
                "data_source_count": len(self.data_source_manager),
                "strategy_count": len(self.strategy_manager),
                "executor_count": len(self.executor_manager),
            },
            "resources": {
                "cpu": {
                    "percent": cpu_percent,
                    "value": f"{cpu_percent}%",
                    "status": "success" if cpu_percent < 60 else "warning" if cpu_percent < 85 else "error"
                },
                "memory": {
                    "percent": mem_percent,
                    "value": f"{mem_used}G/{mem_total}G",
                    "status": "success" if mem_percent < 60 else "warning" if mem_percent < 85 else "error"
                },
                "disk": {
                    "percent": disk_percent,
                    "value": f"{disk_used}G/{disk_total}G",
                    "status": "success" if disk_percent < 60 else "warning" if disk_percent < 85 else "error"
                }
            }
        }

    def on_stop(self):
        """
        清理资源，安全退出。
        """
        if not self.running:
            return
        self.running = False
        try:
            self.conn.close()
        except Exception as e:
            logger.error(f"关闭管道时出错: {e}")
        logger.warning(f"[ProcessEngine] ({self.instance_id}, {self.name}) Cleaning up and exiting.")

    # 下面实现所有Engine抽象方法（这里只做简单打印/存储，实际可接入管理器）
    def add_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        logger.info(f"添加策略: {config}")
        self.strategy_manager.add(config)

    def update_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        logger.info(f"更新策略: {config}")
        self.strategy_manager.update(config)

    def remove_strategy(self, instance_id):
        logger.info(f"移除策略: {instance_id}")
        self.strategy_manager.remove(instance_id)

    def add_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        logger.info(f"添加执行器: {config}")
        self.executor_manager.add(config)

    def update_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        logger.info(f"更新执行器: {config}")
        self.executor_manager.update(config)

    def remove_executor(self, instance_id):
        logger.info(f"移除执行器: {instance_id}")
        self.executor_manager.remove(instance_id)

    def add_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        logger.info(f"添加数据源: {config}")
        self.data_source_manager.add(config)

    def update_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        logger.info(f"更新数据源: {config}")
        self.data_source_manager.update(config)

    def remove_data_source(self, instance_id):
        logger.info(f"移除数据源: {instance_id}")
        self.data_source_manager.remove(instance_id)

    def update_position_config(self, position_config: PositionConfig) -> None:
        logger.info(f"更新仓位配置: {position_config}")
        self.position_config = position_config

    def ping(self, instance_id: str):
        """响应主进程的 ping 消息，回复 pong"""
        if instance_id == self.instance_id:
            self.send_msg("pong", instance_id=instance_id)

    def send_msg(self, action: str, *args, **kwargs):
        """发送消息到主进程"""
        msg = {"action": action, "args": args, "kwargs": kwargs}
        try:
            self.conn.send(msg) 
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    def close_position(self, position_id: str):
        position = self.position_manager.get_position(position_id)
        if position:
            self.strategy_manager.close_position(position)
            return True
        return False

    def reset_position_state(self):
        """
        重置仓位状态
        """
        self.position_manager.reset_position_state()


class ProcessEngineClient(Engine):
    """
    主进程中的引擎客户端。
    负责启动/关闭引擎子进程，发送指令，接收结果。
    """

    def __init__(self, instance_id: str, name: str, config = None, event_hook: Set[EventType] = None):
        super().__init__()
        # 创建双向管道，parent_conn 用于主进程发送消息，child_conn 用于子进程接收消息
        self.parent_conn, self.child_conn = Pipe()
        self.process: Optional[Process] = None
        self.instance_id = instance_id
        self.name = name
        self.config = config
        self.event_hook = event_hook or set()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}  # 存储invoke响应的Future对象

    def register_handler(self, action: str, handler: Callable):
        """注册消息处理器"""
        if action not in self.message_handlers:
            self.message_handlers[action] = []
        self.message_handlers[action].append(handler)
        logger.info(f"注册消息处理器: {action} -> {handler.__name__}")

    def unregister_handler(self, action: str, handler: Callable):
        """删除消息处理器"""
        if action in self.message_handlers:
            self.message_handlers[action].remove(handler)
            if not self.message_handlers[action]:
                del self.message_handlers[action]
            logger.debug(f"删除消息处理器: {action} -> {handler.__name__}")

    def start(self):
        """
        启动引擎子进程。
        """
        if self.process is None or not self.process.is_alive():
            # 只传 child_conn 给子进程
            self.process = Process(target=self._start_engine,
                                 args=(self.child_conn, self.instance_id, self.name, self.config, self.event_hook), name=f"ProcessEngine-{self.instance_id}-{self.name}")
            self.process.start()
            logger.info(f"[ProcessEngineClient] Engine process started. ({self.instance_id}, {self.name})")

    def _start_engine(self, conn: Connection, instance_id: str, name: str, config=None, event_hook: Set[EventType] = None):
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
            
            engine = ProcessEngine(conn, instance_id, name, PositionConfig(**position_setting), event_hook)
        except Exception as e:
            logger.error(f"设置仓位配置时出错: {e}", exc_info=True)
            return
        engine.run()

    def send_action(self, action: str, *args, **kwargs):
        """
        发送消息到引擎进程，不等待响应
        """
        msg = {"action": action, "args": args, "kwargs": kwargs}
        try:
            self.parent_conn.send(msg)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    async def invoke(self, action: str, *args, **kwargs) -> Any:
        """
        发送消息到引擎进程并等待响应，类似函数调用
        
        Args:
            action: 要执行的动作
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            引擎进程返回的结果
            
        Raises:
            Exception: 如果动作执行失败或超时
        """
        msg_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.response_futures[msg_id] = future
        
        try:
            msg = {
                "action": action,
                "args": args,
                "kwargs": kwargs,
                "msg_id": msg_id
            }
            self.parent_conn.send(msg)
            # 等待响应，设置30秒超时
            try:
                result = await asyncio.wait_for(future, timeout=30.0)
                if isinstance(result, dict) and "error" in result:
                    raise Exception(result["error"])
                return result
            except asyncio.TimeoutError:
                raise Exception(f"调用超时: {action}")
        finally:
            self.response_futures.pop(msg_id, None)

    def stop(self):
        """
        发送关闭指令并等待子进程退出。
        """
        self.send_action("on_stop")
        self.message_handlers = {}
        if self.process and self.process.is_alive():
            self.process.join()
            logger.info(f"[ProcessEngineClient] Engine process stopped. ({self.instance_id}, {self.name})")

    # 下面实现所有Engine抽象方法，全部转发到子进程
    def add_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        self.send_action("add_strategy", config)

    def update_strategy(self, config: LeekComponentConfig[Strategy, StrategyConfig]):
        self.send_action("update_strategy", config)

    def remove_strategy(self, instance_id):
        self.send_action("remove_strategy", instance_id)

    def add_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        self.send_action("add_executor", config)

    def update_executor(self, config: LeekComponentConfig[Executor, Dict[str, Any]]):
        self.send_action("update_executor", config)

    def remove_executor(self, instance_id):
        self.send_action("remove_executor", instance_id)

    def add_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        self.send_action("add_data_source", config)

    def update_data_source(self, config: LeekComponentConfig[DataSource, Dict[str, Any]]):
        self.send_action("update_data_source", config)

    def remove_data_source(self, instance_id):
        self.send_action("remove_data_source", instance_id)

    def update_position_config(self, position_config, data=None) -> None:
        # 检查 risk_policies 是否存在，如果不存在则设为空列表
        risk_policies = position_config.get('risk_policies', [])
        position_config['risk_policies'] = [LeekComponentConfig(
                instance_id=self.instance_id,
                name=policy.get('name'),
                cls=load_class_from_str(policy.get('class_name')),
                config=policy.get('params')) for policy in risk_policies if policy.get('enabled')]
        position_config['data'] = data
        self.send_action("update_position_config", PositionConfig(**position_config))

    def listen(self, poll_interval=0.2):
        """
        异步轮询接收子进程消息，收到消息时调用 self.on_message。
        """
        retry_count = 0
        max_retries = 3
        
        while self.process and self.process.is_alive():
            try:
                has_msg = self.parent_conn.poll(poll_interval)
                if has_msg:
                    try:
                        msg = self.parent_conn.recv()
                        self.on_message(msg)
                        retry_count = 0  # 重置重试计数
                    except (EOFError, ConnectionError) as e:
                        logger.error(f"管道连接错误: {e}", exc_info=True)
                        break
                    except (pickle.UnpicklingError, UnicodeDecodeError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"接收消息失败，已达到最大重试次数: {e}")
                            break
                        logger.warning(f"接收消息失败，正在重试 ({retry_count}/{max_retries}): {e}")
                        continue
                    except Exception as e:
                        logger.error(f"接收消息时出错: {e}", exc_info=True)
                        break
            except Exception as e:
                logger.error(f"轮询时出错: {e}", exc_info=True)
                break
                
            time.sleep(poll_interval)

    def on_message(self, msg):
        """处理引擎进程消息，路由到对应的处理器"""
        if not isinstance(msg, dict) or "action" not in msg:
            logger.error(f"未知消息: {msg}")
            return

        action = msg["action"]
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})

        # 反序列化参数
        args = [arg for arg in args]
        kwargs = {k: v for k, v in kwargs.items()}

        # 处理响应消息
        if action == "response":
            msg_id = kwargs.get("msg_id")
            if msg_id in self.response_futures:
                future = self.response_futures[msg_id]
                if "error" in kwargs:
                    future.set_exception(Exception(kwargs["error"]))
                else:
                    future.set_result(kwargs.get("result"))
                return

        # 处理普通消息
        if action in self.message_handlers:
            for handler in self.message_handlers[action]:
                try:
                    if "project_id" in inspect.signature(handler).parameters:
                        handler(project_id=int(self.instance_id), *args, **kwargs)
                    else:
                        handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"处理消息 {action} 时出错: {e}", exc_info=True)
        else:
            logger.info(f"未注册的消息处理器: {action}")
