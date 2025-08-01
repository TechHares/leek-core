import asyncio
import signal
import grpc
from leek_core.engine.grpc.engine_pb2_grpc import add_EngineServiceServicer_to_server
from leek_core.engine.grpc.grpc_server import EngineServiceServicer
from leek_core.utils import get_logger

logger = get_logger(__name__)

async def start_server():
    server = grpc.aio.server()
    add_EngineServiceServicer_to_server(EngineServiceServicer(None), server)
    
    # 启动服务器
    server_address = f'[::]:{51023}'
    server.add_insecure_port(server_address)
    logger.info(f"启动服务器: {server_address}")
    
    # 获取事件循环并启动服务器
    await server.start()
    
    # 保持服务器运行
    logger.info(f"gRPC服务器已启动，保持运行状态")
    
    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        logger.info("收到中断信号，正在关闭服务器...")
    finally:
        # 确保服务器优雅关闭
        await server.stop(grace=5)
        logger.info("服务器已关闭")

if __name__ == "__main__":
    # 注册信号处理器
    try:
        asyncio.run(start_server())
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        logger.info("收到中断信号，正在关闭服务器...")