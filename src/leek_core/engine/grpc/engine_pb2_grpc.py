# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import engine_pb2 as engine__pb2

GRPC_GENERATED_VERSION = '1.74.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in engine_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class EngineServiceStub(object):
    """引擎服务 - 提供调用方法和stream监听
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ExecuteAction = channel.unary_unary(
                '/engine.EngineService/ExecuteAction',
                request_serializer=engine__pb2.ActionRequest.SerializeToString,
                response_deserializer=engine__pb2.ActionResponse.FromString,
                _registered_method=True)
        self.ListenEvents = channel.unary_stream(
                '/engine.EngineService/ListenEvents',
                request_serializer=engine__pb2.ListenRequest.SerializeToString,
                response_deserializer=engine__pb2.EventMessage.FromString,
                _registered_method=True)


class EngineServiceServicer(object):
    """引擎服务 - 提供调用方法和stream监听
    """

    def ExecuteAction(self, request, context):
        """主进程调用子进程的方法
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListenEvents(self, request, context):
        """主进程监听子进程事件的stream
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EngineServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ExecuteAction': grpc.unary_unary_rpc_method_handler(
                    servicer.ExecuteAction,
                    request_deserializer=engine__pb2.ActionRequest.FromString,
                    response_serializer=engine__pb2.ActionResponse.SerializeToString,
            ),
            'ListenEvents': grpc.unary_stream_rpc_method_handler(
                    servicer.ListenEvents,
                    request_deserializer=engine__pb2.ListenRequest.FromString,
                    response_serializer=engine__pb2.EventMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'engine.EngineService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('engine.EngineService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class EngineService(object):
    """引擎服务 - 提供调用方法和stream监听
    """

    @staticmethod
    def ExecuteAction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/engine.EngineService/ExecuteAction',
            engine__pb2.ActionRequest.SerializeToString,
            engine__pb2.ActionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ListenEvents(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/engine.EngineService/ListenEvents',
            engine__pb2.ListenRequest.SerializeToString,
            engine__pb2.EventMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
