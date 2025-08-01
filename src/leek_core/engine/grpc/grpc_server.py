#!/usr/bin/env python
# -*- coding: utf-8 -*-

import grpc
import json
import asyncio
import threading
import time
from concurrent import futures
from typing import Dict, Set, Callable, Optional, AsyncIterator
from leek_core.utils.serialization import LeekJSONEncoder
from leek_core.utils import get_logger

logger = get_logger(__name__)

# 导入 protobuf 生成的代码
from .engine_pb2 import *
from .engine_pb2_grpc import *


class EngineServiceServicer(EngineServiceServicer):
    """子进程的 gRPC 服务实现（异步版本）"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def ExecuteAction(self, request, context):
        """处理主进程发送的动作（异步）"""
        try:
            project_id = request.project_id
            action = request.action.lower()
            args_json = request.args_json
            kwargs_json = request.kwargs_json
            request_id = request.request_id
            
            logger.debug(f"收到动作请求: {project_id} {action} {request_id}")
            try:
                args = json.loads(args_json) if args_json else []
                kwargs = json.loads(kwargs_json) if kwargs_json else {}
                
                # 调用引擎的动作处理器
                if not hasattr(self.engine, action):
                    raise Exception(f"引擎不支持动作: {action}")

                method = getattr(self.engine, action)
                result = method(*args, **kwargs)
                result_json = json.dumps(result, cls=LeekJSONEncoder) if result is not None else None
                return ActionResponse(
                    request_id=request_id,
                    success=True,
                    result_json=result_json
                )
                
            except Exception as e:
                logger.error(f"执行动作失败: {action} {e}", exc_info=True)
                return ActionResponse(
                    request_id=request_id,
                    success=False,
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"处理动作请求失败: {e}", exc_info=True)
            return ActionResponse(
                request_id=request_id,
                success=False,
                error=str(e)
            )

    async def ListenEvents(self, request: ListenRequest, context) -> AsyncIterator[EventMessage]:
        """处理事件监听请求 - 主进程监听子进程事件（异步）"""
        try:
            project_id = request.project_id
            try:
                # 从引擎的事件队列中监听事件
                while self.engine.running:
                    try:
                        # 从异步队列中获取事件，设置超时
                        event = await asyncio.wait_for(
                            self.engine._event_queue.get(),
                            timeout=30.0
                        )
                        yield event
                        
                    except asyncio.TimeoutError:
                        # 发送心跳事件保持连接
                        heartbeat_event = EventMessage(
                            project_id=project_id,
                            event_type="heartbeat",
                            data_json="{}",
                            timestamp=int(time.time() * 1000),
                            source="heartbeat"
                        )
                        yield heartbeat_event
                        
            except Exception as e:
                logger.error(f"事件监听流异常: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"处理事件监听请求失败: {e}", exc_info=True)
        finally:
            logger.info(f"主进程停止监听事件: {project_id}") 