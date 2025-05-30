from abc import abstractmethod

from leek_core.base import LeekComponent


class AlarmLevel:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlarmSender(LeekComponent):

    @abstractmethod
    def send(self, level, message, **kwargs):
        raise NotImplementedError