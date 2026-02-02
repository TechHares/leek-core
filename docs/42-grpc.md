# 42 gRPC 服务

## 概述

gRPC 服务模块提供远程过程调用能力，支持将引擎作为独立进程运行，并通过 gRPC 协议与主进程通信。适用于多项目管理、进程隔离、远程部署等场景。

## 核心组件

### 协议定义

```protobuf
// engine.proto
syntax = "proto3";
package engine;

// 引擎服务 - 提供调用方法和stream监听
service EngineService {
  // 主进程调用子进程的方法
  rpc ExecuteAction(ActionRequest) returns (ActionResponse);
  
  // 主进程监听子进程事件的stream
  rpc ListenEvents(ListenRequest) returns (stream EventMessage);
}

// 动作请求
message ActionRequest {
  string project_id = 1;
  string action = 2;
  string args_json = 3;
  string kwargs_json = 4;
  string request_id = 5;
}

// 动作响应
message ActionResponse {
  string request_id = 1;
  bool success = 2;
  string result_json = 3;
  string error = 4;
}

// 事件消息
message EventMessage {
  string project_id = 1;
  string event_type = 2;
  string data_json = 3;
  int64 timestamp = 4;
  string source = 5;
}

// 监听请求
message ListenRequest {
  string project_id = 1;
}
```

### `EngineServiceServicer` - gRPC 服务实现

```python
class EngineServiceServicer(EngineServiceServicer):
    """子进程的 gRPC 服务实现（异步版本）"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def ExecuteAction(self, request, context) -> ActionResponse:
        """处理主进程发送的动作"""
        ...
    
    async def ListenEvents(self, request, context) -> AsyncIterator[EventMessage]:
        """处理事件监听请求 - 主进程监听子进程事件"""
        ...
```

## 架构设计

### 进程通信模型

```text
┌─────────────────────────────────────────────────────────────┐
│                        主进程 (Manager)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  gRPC Client                          │  │
│  │  - ExecuteAction() 调用子进程方法                      │  │
│  │  - ListenEvents() 订阅子进程事件                       │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
└──────────────────────────│──────────────────────────────────┘
                           │ gRPC (HTTP/2)
┌──────────────────────────│──────────────────────────────────┐
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │                  gRPC Server                          │  │
│  │  - EngineServiceServicer                              │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │                  SimpleEngine                         │  │
│  │  - 策略执行                                           │  │
│  │  - 订单处理                                           │  │
│  │  - 事件队列                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│                        子进程 (Engine)                       │
└──────────────────────────────────────────────────────────────┘
```

### 事件流

```text
子进程 Engine
    │
    ▼
EventBus.publish_event()
    │
    ▼
_event_queue.put(EventMessage)
    │
    ▼
gRPC Server (ListenEvents stream)
    │
    ▼
主进程 Client 接收事件
```

## 使用示例

### 启动 gRPC 服务端

```python
import grpc
from concurrent import futures
from leek_core.engine import SimpleEngine
from leek_core.engine.grpc import EngineServiceServicer, add_EngineServiceServicer_to_server

# 创建引擎
engine = SimpleEngine(
    instance_id="engine-001",
    name="gRPC引擎",
    position_config=position_config,
)

# 创建 gRPC 服务器
server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

# 注册服务
servicer = EngineServiceServicer(engine)
add_EngineServiceServicer_to_server(servicer, server)

# 监听端口
server.add_insecure_port('[::]:50051')

# 启动
await server.start()
await server.wait_for_termination()
```

### gRPC 客户端调用

```python
import grpc
import json
from leek_core.engine.grpc import EngineServiceStub, ActionRequest, ListenRequest

# 创建通道
channel = grpc.insecure_channel('localhost:50051')
stub = EngineServiceStub(channel)

# 调用引擎方法
def call_engine_action(action: str, *args, **kwargs):
    request = ActionRequest(
        project_id="project-001",
        action=action,
        args_json=json.dumps(args),
        kwargs_json=json.dumps(kwargs),
        request_id=str(uuid.uuid4()),
    )
    
    response = stub.ExecuteAction(request)
    
    if response.success:
        return json.loads(response.result_json) if response.result_json else None
    else:
        raise Exception(response.error)

# 示例：添加策略
call_engine_action('add_strategy', {
    "id": "strategy-001",
    "name": "双均线策略",
    "class_name": "my_module.MyStrategy",
    "params": {"period": 20},
})

# 示例：获取状态
state = call_engine_action('engine_state')
print(state)

# 示例：关闭仓位
call_engine_action('close_position', "position-001")
```

### 监听事件流

```python
import asyncio

async def listen_events():
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = EngineServiceStub(channel)
    
    request = ListenRequest(project_id="project-001")
    
    async for event in stub.ListenEvents(request):
        if event.event_type == "heartbeat":
            continue
        
        print(f"事件: {event.event_type}")
        print(f"数据: {event.data_json}")
        print(f"时间: {event.timestamp}")
        
        # 处理不同类型的事件
        if event.event_type == "ORDER_UPDATED":
            data = json.loads(event.data_json)
            print(f"订单更新: {data['order_id']} -> {data['status']}")
        
        elif event.event_type == "POSITION_UPDATE":
            data = json.loads(event.data_json)
            print(f"仓位更新: {data['symbol']} PnL: {data['pnl']}")

asyncio.run(listen_events())
```

## 支持的动作

gRPC 服务支持调用引擎的所有公开方法：

| 动作 | 说明 | 参数 |
|------|------|------|
| `add_strategy` | 添加策略 | `config: dict` |
| `update_strategy` | 更新策略 | `config: dict` |
| `remove_strategy` | 移除策略 | `instance_id: str` |
| `add_executor` | 添加执行器 | `config: dict` |
| `update_executor` | 更新执行器 | `config: dict` |
| `remove_executor` | 移除执行器 | `instance_id: str` |
| `add_data_source` | 添加数据源 | `config: dict` |
| `update_data_source` | 更新数据源 | `config: dict` |
| `remove_data_source` | 移除数据源 | `instance_id: str` |
| `engine_state` | 获取引擎状态 | - |
| `get_position_state` | 获取仓位状态 | - |
| `get_strategy_state` | 获取策略状态 | - |
| `close_position` | 平仓 | `position_id: str` |
| `update_position_config` | 更新仓位配置 | `config: dict` |
| `on_start` | 启动引擎 | - |
| `on_stop` | 停止引擎 | - |

## 事件类型

通过 `ListenEvents` 可以接收以下事件：

| 事件类型 | 说明 |
|----------|------|
| `DATA_RECEIVED` | 接收到市场数据 |
| `STRATEGY_SIGNAL` | 策略产生信号 |
| `ORDER_CREATED` | 订单创建 |
| `ORDER_UPDATED` | 订单状态更新 |
| `POSITION_UPDATE` | 仓位更新 |
| `RISK_TRIGGERED` | 风控触发 |
| `TRANSACTION` | 资金流水 |
| `heartbeat` | 心跳（30秒无事件时发送） |

## 配置说明

### 服务端配置

```python
# 线程池大小
server = grpc.aio.server(
    futures.ThreadPoolExecutor(max_workers=10)
)

# 最大消息大小
server = grpc.aio.server(
    options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ]
)

# TLS 加密（生产环境推荐）
with open('server.key', 'rb') as f:
    private_key = f.read()
with open('server.crt', 'rb') as f:
    certificate_chain = f.read()

credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
server.add_secure_port('[::]:50051', credentials)
```

### 客户端配置

```python
# 超时设置
response = stub.ExecuteAction(request, timeout=30)

# TLS 加密
with open('ca.crt', 'rb') as f:
    trusted_certs = f.read()

credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
channel = grpc.secure_channel('localhost:50051', credentials)

# 重试策略
import json

retry_policy = json.dumps({
    "methodConfig": [{
        "name": [{"service": "engine.EngineService"}],
        "retryPolicy": {
            "maxAttempts": 5,
            "initialBackoff": "0.1s",
            "maxBackoff": "10s",
            "backoffMultiplier": 2,
            "retryableStatusCodes": ["UNAVAILABLE"],
        },
    }]
})

channel = grpc.insecure_channel(
    'localhost:50051',
    options=[('grpc.service_config', retry_policy)]
)
```

## 错误处理

```python
import grpc

try:
    response = stub.ExecuteAction(request)
    if not response.success:
        print(f"执行失败: {response.error}")
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        print("服务不可用")
    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("请求超时")
    elif e.code() == grpc.StatusCode.INTERNAL:
        print(f"服务器内部错误: {e.details()}")
    else:
        print(f"RPC错误: {e.code()} - {e.details()}")
```

## 最佳实践

### 1. 连接池管理

```python
class GrpcConnectionPool:
    def __init__(self, address: str, pool_size: int = 5):
        self.address = address
        self.pool_size = pool_size
        self._channels = []
        self._index = 0
    
    def get_channel(self):
        if len(self._channels) < self.pool_size:
            channel = grpc.insecure_channel(self.address)
            self._channels.append(channel)
        
        channel = self._channels[self._index % len(self._channels)]
        self._index += 1
        return channel
    
    def close(self):
        for channel in self._channels:
            channel.close()
```

### 2. 健康检查

```python
from grpc_health.v1 import health_pb2, health_pb2_grpc

class HealthServicer(health_pb2_grpc.HealthServicer):
    def __init__(self, engine):
        self.engine = engine
    
    def Check(self, request, context):
        if self.engine.running:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

# 添加健康检查服务
health_pb2_grpc.add_HealthServicer_to_server(
    HealthServicer(engine), server
)
```

### 3. 日志追踪

```python
import grpc

class LoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        logger.info(f"gRPC调用: {handler_call_details.method}")
        return continuation(handler_call_details)

server = grpc.aio.server(
    interceptors=[LoggingInterceptor()]
)
```

## 相关模块

- [引擎架构](20-engine.md) - SimpleEngine 详情
- [事件总线](22-event-bus.md) - 事件系统
