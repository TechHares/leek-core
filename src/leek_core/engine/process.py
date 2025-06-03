from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Optional, Any, Dict, List, Callable
import uuid
import psutil

from leek_core.alarm import ErrorAlarmHandler
from .base import Engine
from leek_core.data import DataSource, DataSourceContext
from leek_core.event import EventBus
from leek_core.executor import Executor, ExecutorContext
from leek_core.manager import DataManager, StrategyManager, PositionManager, ExecutorManager
from leek_core.models import LeekComponentConfig, StrategyConfig, PositionConfig
from leek_core.strategy import Strategy, StrategyContext
from leek_core.utils import get_logger, run_func_timeout
from leek_core.alarm import AlarmSender, alarm_manager
from leek_core.base import create_component, load_class_from_str
import asyncio
from leek_core.utils import setup_logging
import sys
import os

logger = get_logger(__name__)

class ProcessEngine(Engine):
    """
    独立进程中的交易引擎实现。
    通过Pipe与主进程通信，接收指令并执行。
    主进程挂掉时，Pipe断开，子进程自动退出。
    """

    def __init__(self, conn: Connection, instance_id: str, name: str, position_config: PositionConfig = None):
        super().__init__()
        self.conn = conn  # 用于收发主进程消息
        self.running = True
        self.instance_id = instance_id
        self.name = name
        self.position_config = position_config
        self.response_handlers: Dict[str, Callable] = {}  # 存储响应处理器，按消息ID索引

        self.event_bus = EventBus()
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
                config=position_config
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

    def start(self) -> None:
        pass

    def run(self):
        """
        启动引擎主循环，监听主进程指令。
        收到'shutdown'或管道断开时自动退出。
        """
        self.running = True
        self.start()
        try:
            while self.running:
                if self.conn.poll(5):  # 1秒超时，非阻塞检查
                    msg = self.conn.recv()
                    self.handle_message(msg)
        except (EOFError, KeyboardInterrupt):
            ...
        except Exception as e:
            logger.error(f"引擎运行时出错: {e}")
        
        self.on_stop()

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
        
        if hasattr(self, action):
            method = getattr(self, action)
            try:
                result = method(*args, **kwargs)
                # 如果是invoke请求（有msg_id），则发送响应
                if msg_id:
                    self.send_msg("response", msg_id=msg_id, result=result)
            except Exception as e:
                logger.error(f"处理命令 {method.__name__} 时出错: {e}", exc_info=True)
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
        print(f"更新仓位配置: {position_config}")
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


class ProcessEngineClient(Engine):
    """
    主进程中的引擎客户端。
    负责启动/关闭引擎子进程，发送指令，接收结果。
    """

    def __init__(self, instance_id: str, name: str, config = None):
        super().__init__()
        # 创建双向管道，parent_conn 用于主进程发送消息，child_conn 用于子进程接收消息
        self.parent_conn, self.child_conn = Pipe()
        self.process: Optional[Process] = None
        self.instance_id = instance_id
        self.name = name
        self.config = config
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}  # 存储invoke响应的Future对象

    def register_handler(self, action: str, handler: Callable):
        """注册消息处理器"""
        if action not in self.message_handlers:
            self.message_handlers[action] = []
        self.message_handlers[action].append(handler)
        logger.debug(f"注册消息处理器: {action} -> {handler.__name__}")

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
                                 args=(self.child_conn, self.instance_id, self.name, self.config), name=f"ProcessEngine-{self.instance_id}-{self.name}")
            self.process.start()
            logger.info(f"[ProcessEngineClient] Engine process started. ({self.instance_id}, {self.name})")

    def _start_engine(self, conn: Connection, instance_id: str, name: str, config=None):
        # 设置日志格式
        log_level = config.get('log_level', 'INFO')
        log_format = config.get('log_format', 'json')
        log_alarm = config.get('log_alarm', False)

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
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                if dir_path not in sys.path:
                    sys.path.append(dir_path)
            else:
                logger.error(f"目录不存在或无法访问: {dir_path}")

        # 设置仓位配置
        position_setting = config.get('position_setting', {})
        try:
            position_setting['risk_policies'] = [LeekComponentConfig(
                instance_id=instance_id,
                name=policy.get('name'),
                cls=load_class_from_str(policy.get('class_name')),
                config=policy.get('params')) for policy in position_setting['risk_policies'] if policy.get('enabled')]
            engine = ProcessEngine(conn, instance_id, name, PositionConfig(**position_setting))
        except Exception as e:
            logger.error(f"设置仓位配置时出错: {e}")
            return
        engine.run()

    def send_action(self, action: str, *args, **kwargs):
        """
        发送消息到引擎进程，不等待响应
        """
        msg = {"action": action, "args": args, "kwargs": kwargs}
        self.parent_conn.send(msg)

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

    def update_position_config(self, position_config) -> None:
        position_config['risk_policies'] = [LeekComponentConfig(
                instance_id=self.instance_id,
                name=policy.get('name'),
                cls=load_class_from_str(policy.get('class_name')),
                config=policy.get('params')) for policy in position_config['risk_policies'] if policy.get('enabled')]
        self.send_action("update_position_config", PositionConfig(**position_config))

    async def listen(self, poll_interval=0.2):
        """
        异步轮询接收子进程消息，收到消息时调用 self.on_message。
        """
        while self.process and self.process.is_alive():
            has_msg = await asyncio.to_thread(self.parent_conn.poll, poll_interval)
            if has_msg:
                try:
                    msg = await asyncio.to_thread(self.parent_conn.recv)
                    self.on_message(msg)
                except EOFError:
                    logger.error("管道已断开")
                    break
                except Exception as e:
                    logger.error(f"接收消息时出错: {e}")
                    break
            await asyncio.sleep(poll_interval)

    def on_message(self, msg):
        """处理引擎进程消息，路由到对应的处理器"""
        if not isinstance(msg, dict) or "action" not in msg:
            logger.error(f"未知消息: {msg}")
            return

        action = msg["action"]
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})

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
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"处理消息 {action} 时出错: {e}")
        else:
            logger.info(f"未注册的消息处理器: {action}")
