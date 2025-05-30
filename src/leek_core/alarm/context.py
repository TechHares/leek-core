import logging
from typing import Any, Dict, List, Tuple, Type

from leek_core.base import create_component
from .base import AlarmLevel, AlarmSender


class ErrorAlarmHandler(logging.Handler):
    """
    日志错误报警处理器：凡是 error 及以上级别的日志，自动触发报警
    """

    def emit(self, record):
        try:
            if record.levelno >= logging.ERROR:
                msg = self.format(record)
                alarm_manager.send_alarm(AlarmLevel.ERROR, msg)
        except Exception:
            # 避免报警异常导致主程序崩溃
            pass


class AlarmManager:
    def __init__(self, alarm_list: List[Tuple[Type[AlarmSender], Dict[str, Any]]]=None):
        self.senders = [create_component(cls, **config) for cls, config in alarm_list] if alarm_list else []

    def register(self, sender: AlarmSender):
        self.senders.append(sender)

    def register_cls(self, cls: Type[AlarmSender], config):
        self.register(create_component(cls, **config))

    def send_alarm(self, level, message, **kwargs):
        for sender in self.senders:
            sender.send(level, message, **kwargs)


# 单例全局报警管理器
alarm_manager = AlarmManager()