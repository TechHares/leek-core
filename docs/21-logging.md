# Leek Core 日志系统

Leek Core提供了功能全面、易于使用且高度可配置的日志系统，可以满足从开发调试到生产环境的各种需求。本文档详细介绍了日志系统的功能、配置方法和使用示例。

## 特性

- **多级别日志** - 支持DEBUG、INFO、WARNING、ERROR、CRITICAL五个标准日志级别
- **多种输出格式** - 支持文本（易读）、JSON（结构化）和简单（紧凑）三种输出格式
- **灵活的输出目标** - 可同时输出到控制台、文件、HTTP或TCP服务器
- **自动日志装饰器** - 为函数和方法提供自动日志记录功能，简化代码
- **结构化上下文** - 支持添加结构化数据到日志记录，方便后期分析
- **高性能设计** - 批处理、异步发送等优化，减少日志对主要代码的性能影响
- **完整的类型标注** - 所有API都有完整的类型标注，提供更好的IDE支持

## 基本用法

### 获取日志器

```python
from utils import get_logger

# 创建带命名空间的日志器
logger = get_logger("app.module")

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 记录结构化数据

```python
# 添加额外上下文信息
logger.info("订单创建", extra={
    "order_id": "12345",
    "amount": 100.50,
    "user_id": 42
})
```

## 配置

### 全局配置

可以在应用启动时设置全局日志配置：

```python
from utils import setup_logging

# 基本配置
setup_logging(
    level="INFO",           # 日志级别
    console=True,           # 输出到控制台
    file=False,             # 默认不输出到文件
    format_type="TEXT"      # 使用文本格式
)

# 完整配置示例
setup_logging(
    level="DEBUG",          # 更详细的日志级别
    console=True,           # 控制台输出
    file=True,              # 文件输出
    file_path="logs/app.log", # 日志文件路径
    format_type="JSON",     # JSON格式，便于机器处理
    max_bytes=10485760,     # 10MB文件大小限制
    backup_count=5          # 保留5个备份文件
)
```

### 环境变量配置

日志系统支持通过环境变量进行配置，便于在不同环境中轻松调整：

```bash
# 设置日志级别
export LEEK_LOG_LEVEL=DEBUG

# 控制输出目标
export LEEK_LOG_CONSOLE=true
export LEEK_LOG_FILE=false

# 设置日志格式
export LEEK_LOG_FORMAT=TEXT  # 可选: TEXT, JSON, SIMPLE

# 自定义文件路径
export LEEK_LOG_FILE_PATH=/var/log/leek/app.log
```

## 装饰器

日志系统提供了强大的装饰器功能，可以自动记录函数和方法的调用、参数、结果和异常：

### 函数日志装饰器

```python
from utils import log_function

@log_function()
def process_data(data):
    """处理数据的函数"""
    # 函数执行时会自动记录入参、结果和性能数据
    return processed_result
```

记录示例：
```
INFO - 调用函数 process_data - 来自 module.py:25 - 参数: data=<Data object>
INFO - 函数 process_data 执行完成 - 用时: 1.23ms - 返回: <ProcessedData object>
```

如果发生异常：
```
ERROR - 函数 process_data 执行异常: ValueError: Invalid data format - 用时: 0.45ms
```

### 方法日志装饰器

专为类方法设计的装饰器，会包含类和实例的信息：

```python
from utils import log_method

class DataService:
    @log_method(level="DEBUG")
    def update(self, item_id, data):
        # 方法执行会记录类名、实例信息、参数和返回值
        return updated_item
```

记录示例：
```
DEBUG - 调用方法 DataService.update [对象: DataService(connection='db1')] - 参数: item_id=42, data={'name': 'New Value'}
DEBUG - 方法 DataService.update 执行完成 - 用时: 2.54ms - 返回: {'id': 42, 'name': 'New Value', 'updated_at': '2023-04-03T12:34:56'}
```

### 交易日志装饰器

专为交易操作设计的装饰器，自动记录交易相关的重要信息：

```python
from utils import log_trade

@log_trade()
def place_order(symbol, quantity, price, side="BUY"):
    # 交易执行...
    return order_result
```

记录示例：
```
INFO - 交易操作开始: place_order - 参数: symbol='BTC/USDT', quantity=0.1, price=50000, side='BUY'
INFO - 交易操作完成: place_order - 结果: {'order_id': '123456', 'status': 'FILLED'}
```

## 输出格式

日志系统支持三种主要的输出格式：

### 文本格式（默认）

人类可读的格式，包含丰富的上下文信息：

```
2025-04-03 19:39:57.f [INFO] [app.module] [MainThread:8421182528] [MainProcess:57192] module.function:172 - 日志消息
```

### JSON格式

结构化格式，适合机器处理和分析：

```json
{
  "timestamp": "2025-04-03T19:39:57.447835",
  "level": "INFO",
  "logger": "app.module",
  "message": "日志消息",
  "module": "module",
  "function": "function",
  "line": 172,
  "env": {
    "hostname": "hostname"
  },
  "thread": {
    "name": "MainThread",
    "id": 8421182528
  },
  "process": {
    "name": "MainProcess",
    "id": 57192
  }
}
```

### 简单格式

紧凑的输出格式，适用于资源有限的环境：

```
I 日志消息
```

## 高级功能

### 外部系统集成

日志系统可以与各种外部服务集成，支持将日志发送到HTTP端点或TCP服务器：

```python
from utils import setup_logging
from utils.logging.handlers import HttpHandler, TcpSocketHandler

# 发送日志到HTTP服务
http_handler = HttpHandler(
    url="http://log-server/api/logs",
    batch_size=10,
    flush_interval=5.0
)

# 发送日志到TCP服务
tcp_handler = TcpSocketHandler(
    host="log-server.example.com",
    port=5000
)

# 配置日志系统使用这些外部处理器
setup_logging(
    external_handlers=[http_handler, tcp_handler]
)
```

### 临时调整日志级别

在特定代码块中临时调整日志级别：

```python
import logging
from utils import get_logger

logger = get_logger("app")

# 正常级别日志
logger.debug("这条不会显示，因为默认级别是INFO")

# 临时调整日志级别
original_level = logger.level
logger.setLevel(logging.DEBUG)
try:
    logger.debug("这条会显示，因为我们临时设置了DEBUG级别")
    # 执行需要详细日志的代码
finally:
    logger.setLevel(original_level)

# 恢复到原有级别
logger.debug("这条又不会显示了")
```

## 最佳实践

1. **使用命名空间** - 创建日志器时使用有意义的命名空间，如`app.module.submodule`，方便过滤和配置
2. **适当的日志级别** - 按照不同信息的重要性选择合适的日志级别
   - DEBUG: 调试信息，只在开发环境使用
   - INFO: 正常操作信息，记录重要流程节点
   - WARNING: 潜在问题，但不影响当前操作
   - ERROR: 错误信息，操作失败但程序可以继续运行
   - CRITICAL: 严重错误，可能导致程序无法继续运行
3. **结构化数据** - 使用`extra`参数添加结构化数据，而不是在消息中嵌入
4. **使用装饰器** - 对关键函数使用日志装饰器，自动记录参数和结果
5. **配置文件归档** - 在生产环境中启用文件滚动和备份，防止日志文件过大
6. **异常记录** - 使用`logger.exception()`或`logger.error(exc_info=True)`记录完整堆栈信息

## 性能考虑

日志系统设计时考虑了性能因素：

1. **懒加载** - 只有在需要时才初始化日志组件
2. **级别过滤** - 低于当前级别的日志调用会快速返回，几乎没有开销
3. **批处理** - 外部处理器支持批量发送日志，减少网络开销
4. **异步处理** - HTTP和TCP处理器支持异步发送，不阻塞主线程
5. **缓冲溢出保护** - 当内存缓冲接近限制时，会自动丢弃老旧日志，防止内存泄漏

## 自定义扩展

日志系统设计为可扩展的，可以自定义处理器和格式化器：

```python
import logging
from utils import setup_logging

# 自定义处理器
class MyCustomHandler(logging.Handler):
    def emit(self, record):
        # 自定义日志处理逻辑
        formatted_message = self.format(record)
        # 处理日志...

# 使用自定义处理器
setup_logging(
    external_handlers=[MyCustomHandler()]
)
```

## 常见问题

**Q: 如何在不同环境使用不同的日志配置?**

A: 使用环境变量或者创建环境特定的配置函数:

```python
def setup_dev_logging():
    setup_logging(level="DEBUG", console=True, file=False)

def setup_prod_logging():
    setup_logging(
        level="INFO", 
        console=False, 
        file=True, 
        file_path="/var/log/app.log"
    )

# 根据环境选择
if os.environ.get("ENV") == "production":
    setup_prod_logging()
else:
    setup_dev_logging()
```

**Q: 日志记录是否会影响性能?**

A: 日志系统设计时考虑了性能因素。对于低于当前级别的日志调用，几乎没有开销。在高性能要求的循环中，可以预先检查日志级别:

```python
# 高效的日志记录
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"计算结果: {expensive_calculation()}")
```

**Q: 如何同时输出到控制台和文件?**

A: 默认配置支持同时输出到多个目标:

```python
setup_logging(
    console=True,
    file=True,
    file_path="logs/app.log"
)
```

**Q: 如何保持日志文件大小可控?**

A: 日志系统已内置了文件滚动功能:

```python
setup_logging(
    file=True,
    file_path="logs/app.log",
    max_bytes=10485760,  # 10MB
    backup_count=5       # 保留5个备份文件
)
```