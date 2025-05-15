import json
import logging

import requests


class AlarmLevel:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlarmSender:
    def send(self, level, message, **kwargs):
        raise NotImplementedError

class DingDingAlarmSender(AlarmSender):
    def __init__(self, alert_token):
        self.alert_token = alert_token

    def send(self, level, message, **kwargs):
        if level == AlarmLevel.CRITICAL or level == AlarmLevel.ERROR:
            requests.post(
                "https://oapi.dingtalk.com/robot/send?access_token=" + self.alert_token,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"msgtype": "text", "text": {"content": f"{message}"}})
            )


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
    def __init__(self):
        self.senders = []

    def register(self, sender: AlarmSender):
        self.senders.append(sender)

    def send_alarm(self, level, message, **kwargs):
        for sender in self.senders:
            sender.send(level, message, **kwargs)

# 单例全局报警管理器
alarm_manager = AlarmManager()
