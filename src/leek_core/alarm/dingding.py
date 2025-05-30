import base64
import hashlib
import hmac
import json
import logging
import time
from urllib.parse import quote_plus  # 导入 quote 方法

import requests

from leek_core.models import Field, FieldType
from .base import AlarmSender, AlarmLevel


class DingDingAlarmSender(AlarmSender):
    display_name = "钉钉机器人"
    init_params = [
        Field(name="webhook_url", required=True, type=FieldType.STRING, description="钉钉机器人的webhook地址"),
        Field(name="secret", type=FieldType.STRING, description="钉钉机器人的加签密钥（可选）"),
    ]

    def __init__(self, webhook_url, secret=None):
        super().__init__()
        self.webhook_url = webhook_url
        self.secret = secret

    def _get_dingtalk_sign(self):
        """生成钉钉加签"""
        if not self.secret:
            return None, None
        timestamp = str(round(time.time() * 1000))
        secret_enc = self.secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, self.secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = quote_plus(base64.b64encode(hmac_code).decode())
        return timestamp, sign

    def send(self, level, message, **kwargs):
        if level == AlarmLevel.CRITICAL or level == AlarmLevel.ERROR:
            timestamp, sign = self._get_dingtalk_sign()
            url = self.webhook_url
            if timestamp and sign:
                url = f"{url}&timestamp={timestamp}&sign={sign}"
            payload = {
                "msgtype": "text",
                "text": {
                    "content": message
                }
            }
            try:
                response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
                response.raise_for_status()
            except requests.RequestException as e:
                logging.error(f"DingTalk alarm send failed: {e}")
