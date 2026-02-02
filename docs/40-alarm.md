# 40 告警系统

## 概述

告警系统提供实时消息通知能力，支持将交易信号、风控触发、系统异常等重要事件推送到钉钉、飞书等即时通讯工具。告警系统采用插件化设计，方便扩展新的通知渠道。

## 核心组件

### 组件层次结构

```text
AlarmSender (告警发送器基类)
├── DingDingAlarmSender   # 钉钉机器人
└── FeishuAlarmSender     # 飞书机器人
```

### `AlarmLevel` - 告警级别

```python
class AlarmLevel:
    INFO = "info"         # 信息
    WARNING = "warning"   # 警告
    ERROR = "error"       # 错误
    CRITICAL = "critical" # 严重
```

### `AlarmSender` - 告警发送器基类

```python
class AlarmSender(LeekComponent):
    """告警发送器抽象基类"""
    
    @abstractmethod
    def send(self, level: str, message: str, **kwargs):
        """
        发送告警
        
        参数:
            level: 告警级别 (AlarmLevel)
            message: 告警消息内容
            **kwargs: 额外参数
        """
        raise NotImplementedError
```

## 内置告警渠道

### 1. 钉钉机器人

```python
class DingDingAlarmSender(AlarmSender):
    """钉钉机器人告警"""
    
    display_name = "钉钉机器人"
    init_params = [
        Field(name="webhook_url", required=True, type=FieldType.STRING, 
              description="钉钉机器人的webhook地址"),
        Field(name="secret", type=FieldType.STRING, 
              description="钉钉机器人的加签密钥（可选）"),
    ]
    
    def __init__(self, webhook_url: str, secret: str = None):
        self.webhook_url = webhook_url
        self.secret = secret
```

**配置步骤：**

1. 在钉钉群中添加自定义机器人
2. 获取 Webhook URL
3. （可选）启用加签并获取 Secret

**使用示例：**

```python
from leek_core.alarm import DingDingAlarmSender, AlarmLevel

# 创建告警发送器
alarm = DingDingAlarmSender(
    webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx",
    secret="SECxxx"  # 可选
)

# 发送告警
alarm.send(AlarmLevel.ERROR, "策略触发止损，已平仓 BTC_USDT")
alarm.send(AlarmLevel.WARNING, "系统内存使用率超过80%")
alarm.send(AlarmLevel.CRITICAL, "交易所连接断开")
```

### 2. 飞书机器人

```python
class FeishuAlarmSender(AlarmSender):
    """飞书机器人告警"""
    
    display_name = "飞书机器人"
    init_params = [
        Field(name="webhook_url", required=True, type=FieldType.STRING, 
              description="飞书机器人的webhook地址"),
        Field(name="secret", type=FieldType.STRING, 
              description="飞书机器人的加签密钥（可选）"),
    ]
    
    def __init__(self, webhook_url: str, secret: str = None):
        self.webhook_url = webhook_url
        self.secret = secret
```

**使用示例：**

```python
from leek_core.alarm import FeishuAlarmSender, AlarmLevel

alarm = FeishuAlarmSender(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
    secret="xxx"  # 可选
)

alarm.send(AlarmLevel.ERROR, "订单执行失败：余额不足")
```

## 告警级别说明

| 级别 | 说明 | 触发条件示例 |
|------|------|-------------|
| `INFO` | 一般信息 | 策略启动、定时报告 |
| `WARNING` | 警告 | 资源使用率较高、网络延迟 |
| `ERROR` | 错误 | 订单失败、连接中断 |
| `CRITICAL` | 严重 | 系统崩溃、资金异常 |

**注意：** 默认实现中，只有 `ERROR` 和 `CRITICAL` 级别会实际发送消息，以避免告警轰炸。

## 自定义告警渠道

### 实现新的发送器

```python
from leek_core.alarm import AlarmSender, AlarmLevel
from leek_core.models import Field, FieldType
import requests

class SlackAlarmSender(AlarmSender):
    """Slack 告警发送器"""
    
    display_name = "Slack 机器人"
    init_params = [
        Field(name="webhook_url", required=True, type=FieldType.STRING,
              description="Slack Incoming Webhook URL"),
        Field(name="channel", type=FieldType.STRING, default="#alerts",
              description="目标频道"),
    ]
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        super().__init__()
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, level: str, message: str, **kwargs):
        # 根据级别选择是否发送
        if level not in [AlarmLevel.ERROR, AlarmLevel.CRITICAL]:
            return
        
        # 根据级别设置颜色
        color_map = {
            AlarmLevel.WARNING: "warning",
            AlarmLevel.ERROR: "danger",
            AlarmLevel.CRITICAL: "danger",
        }
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": color_map.get(level, "good"),
                "title": f"[{level.upper()}] 交易告警",
                "text": message,
                "ts": int(time.time())
            }]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Slack 告警发送失败: {e}")


class EmailAlarmSender(AlarmSender):
    """邮件告警发送器"""
    
    display_name = "邮件告警"
    init_params = [
        Field(name="smtp_host", required=True, type=FieldType.STRING),
        Field(name="smtp_port", type=FieldType.INT, default=587),
        Field(name="username", required=True, type=FieldType.STRING),
        Field(name="password", required=True, type=FieldType.PASSWORD),
        Field(name="from_addr", required=True, type=FieldType.STRING),
        Field(name="to_addrs", required=True, type=FieldType.STRING,
              description="收件人邮箱，多个用逗号分隔"),
    ]
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_addr: str, to_addrs: str):
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = [addr.strip() for addr in to_addrs.split(",")]
    
    def send(self, level: str, message: str, **kwargs):
        if level not in [AlarmLevel.ERROR, AlarmLevel.CRITICAL]:
            return
        
        import smtplib
        from email.mime.text import MIMEText
        
        subject = f"[{level.upper()}] Leek 交易告警"
        msg = MIMEText(message, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")
```

## 在引擎中集成告警

### 方式1：监听事件发送告警

```python
from leek_core.event import EventType
from leek_core.alarm import DingDingAlarmSender, AlarmLevel

alarm = DingDingAlarmSender(webhook_url="https://...")

def on_risk_triggered(event):
    """风控触发时发送告警"""
    data = event.data
    message = f"风控触发\n" \
              f"策略: {data['strategy_id']}\n" \
              f"原因: {data['policy_name']}\n" \
              f"动作: {data['action']}"
    alarm.send(AlarmLevel.WARNING, message)

def on_order_error(event):
    """订单失败时发送告警"""
    order = event.data
    if order.order_status.is_failed:
        message = f"订单失败\n" \
                  f"订单ID: {order.order_id}\n" \
                  f"交易对: {order.symbol}\n" \
                  f"状态: {order.order_status.name}"
        alarm.send(AlarmLevel.ERROR, message)

# 订阅事件
engine.event_bus.subscribe_event(EventType.RISK_TRIGGERED, on_risk_triggered)
engine.event_bus.subscribe_event(EventType.ORDER_UPDATED, on_order_error)
```

### 方式2：封装告警管理器

```python
class AlarmManager:
    """告警管理器"""
    
    def __init__(self, senders: list[AlarmSender]):
        self.senders = senders
    
    def send_all(self, level: str, message: str, **kwargs):
        """发送到所有渠道"""
        for sender in self.senders:
            try:
                sender.send(level, message, **kwargs)
            except Exception as e:
                logger.error(f"告警发送失败 [{sender.__class__.__name__}]: {e}")
    
    def send_trade_signal(self, signal):
        """发送交易信号通知"""
        assets_info = "\n".join([
            f"  - {a.symbol}: {'开仓' if a.is_open else '平仓'} {a.side.name}"
            for a in signal.assets
        ])
        message = f"交易信号\n" \
                  f"策略: {signal.strategy_id}\n" \
                  f"资产:\n{assets_info}"
        self.send_all(AlarmLevel.INFO, message)
    
    def send_position_update(self, position, pnl_pct: float):
        """发送仓位更新通知"""
        if abs(pnl_pct) > 0.05:  # 盈亏超过5%才通知
            level = AlarmLevel.WARNING if pnl_pct < 0 else AlarmLevel.INFO
            message = f"仓位更新\n" \
                      f"交易对: {position.symbol}\n" \
                      f"方向: {position.side.name}\n" \
                      f"盈亏: {pnl_pct:.2%}"
            self.send_all(level, message)


# 使用
alarm_manager = AlarmManager([
    DingDingAlarmSender(webhook_url="https://..."),
    FeishuAlarmSender(webhook_url="https://..."),
])
```

## 最佳实践

### 1. 告警去重

避免同一问题重复告警：

```python
from functools import lru_cache
import time

class DedupAlarmSender(AlarmSender):
    def __init__(self, sender: AlarmSender, cooldown: int = 300):
        self.sender = sender
        self.cooldown = cooldown
        self._last_alerts = {}
    
    def send(self, level: str, message: str, **kwargs):
        key = f"{level}:{hash(message)}"
        now = time.time()
        
        if key in self._last_alerts:
            if now - self._last_alerts[key] < self.cooldown:
                return  # 冷却期内，跳过
        
        self._last_alerts[key] = now
        self.sender.send(level, message, **kwargs)
```

### 2. 告警聚合

批量发送减少打扰：

```python
import threading
from collections import defaultdict

class AggregatedAlarmSender(AlarmSender):
    def __init__(self, sender: AlarmSender, interval: int = 60):
        self.sender = sender
        self.interval = interval
        self._buffer = defaultdict(list)
        self._lock = threading.Lock()
        self._start_timer()
    
    def send(self, level: str, message: str, **kwargs):
        with self._lock:
            self._buffer[level].append(message)
    
    def _flush(self):
        with self._lock:
            for level, messages in self._buffer.items():
                if messages:
                    combined = f"[{len(messages)}条告警]\n" + "\n---\n".join(messages)
                    self.sender.send(level, combined)
            self._buffer.clear()
        self._start_timer()
    
    def _start_timer(self):
        timer = threading.Timer(self.interval, self._flush)
        timer.daemon = True
        timer.start()
```

### 3. 分级发送

不同级别发送到不同渠道：

```python
class LevelRoutedAlarmSender(AlarmSender):
    def __init__(self, routes: dict[str, AlarmSender]):
        """
        routes: {level: sender}
        """
        self.routes = routes
        self.default_sender = routes.get("default")
    
    def send(self, level: str, message: str, **kwargs):
        sender = self.routes.get(level, self.default_sender)
        if sender:
            sender.send(level, message, **kwargs)


# 使用
routed_alarm = LevelRoutedAlarmSender({
    AlarmLevel.INFO: FeishuAlarmSender(webhook_url="..."),  # 信息发飞书
    AlarmLevel.ERROR: DingDingAlarmSender(webhook_url="..."),  # 错误发钉钉
    AlarmLevel.CRITICAL: DingDingAlarmSender(webhook_url="..."),  # 严重也发钉钉
})
```

## 相关模块

- [事件总线](22-event-bus.md) - 事件订阅
- [风险控制](04-risk.md) - 风控触发告警
- [引擎架构](20-engine.md) - 系统集成
