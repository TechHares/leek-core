import hashlib
import json
import logging
import time

import requests

from leek_core.models import Field, FieldType
from .base import AlarmSender, AlarmLevel


class FeishuAlarmSender(AlarmSender):
    display_name = "飞书机器人"
    init_params = [
        Field(name="webhook_url", required=True, type=FieldType.STRING, description="飞书机器人的webhook地址"),
        Field(name="secret", type=FieldType.STRING, description="飞书机器人的加签密钥（可选）"),
    ]

    def __init__(self, webhook_url, secret=None):
        super().__init__()
        self.webhook_url = webhook_url
        self.secret = secret
    def _get_feishu_sign(self):
        """生成飞书加签"""
        if not self.secret:
            return None, None
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self.secret}"
        h = hashlib.sha256(string_to_sign.encode("utf-8"))
        sign = h.hexdigest()
        return timestamp, sign

    def send(self, level, message, **kwargs):
        if level == AlarmLevel.CRITICAL or level == AlarmLevel.ERROR:
            timestamp, sign = self._get_feishu_sign()
            payload = {
                "msg_type": "text",
                "content": {
                    "text": message
                }
            }
            if timestamp and sign:
                payload["timestamp"] = timestamp
                payload["sign"] = sign
            try:
                response = requests.post(self.webhook_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
                response.raise_for_status()
            except requests.RequestException as e:
                logging.error(f"Feishu alarm send failed: {e}")