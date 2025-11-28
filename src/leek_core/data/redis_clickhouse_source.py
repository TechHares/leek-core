#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
共享内存缓存模块：
1. 跨进程数据共享
2. 数据序列化和反序列化
3. 内存管理和清理
4. 线程安全访问
"""

from datetime import datetime
from typing import Any, Iterator, List
from decimal import Decimal
import os
import pickle
import time

import msgpack
from redis import Redis
from redis_lock import Lock as RedisLock

from leek_core.models import (
    AssetType,
    ChoiceType,
    DataType,
    Field,
    FieldType,
    KLine,
    TimeFrame,
    TradeInsType,
)
from leek_core.utils import DateTimeUtils, get_logger

from .base import DataSource
from .clickhouse_source import ClickHouseKlineDataSource

logger = get_logger(__name__)
class RedisClickHouseDataSource(DataSource):
    supported_data_type: DataType = DataType.KLINE
    display_name = "ClickHouse K线(Redis缓存)"
    # 声明支持的资产类型
    supported_asset_type: DataType = {AssetType.STOCK, AssetType.CRYPTO}
    just_backtest: bool = True
    init_params: List[Field] = [
        Field(name="host", label="ClickHouse主机", type=FieldType.STRING, required=True, default="127.0.0.1"),
        Field(name="port", label="ClickHouse端口", type=FieldType.INT, required=True, default=9000),
        Field(name="user", label="ClickHouse用户名", type=FieldType.STRING, required=True, default="default"),
        Field(name="password", label="ClickHouse密码", type=FieldType.STRING, default="default"),
        Field(name="database", label="ClickHouse数据库名", type=FieldType.STRING, default="default"),
        Field(name="redis_host", label="Redis主机", type=FieldType.STRING, required=True, default="127.0.0.1"),
        Field(name="redis_port", label="Redis端口", type=FieldType.INT, required=True, default=6379),
        Field(name="redis_password", label="Redis密码", type=FieldType.STRING, default=""),
        Field(name="redis_db", label="Redis数据库", type=FieldType.INT, default=0),
    ]
    """缓存管理器"""
    
    def __init__(self, host: str="127.0.0.1", port: int = 9000, user: str = 'default', password: str = '', database: str = 'default',
     redis_host: str="127.0.0.1", redis_port: int = 6379, redis_password: str = '', redis_db: int = 0):
        super().__init__()
        self.data_source = ClickHouseKlineDataSource(host, port, user, password, database)
        self.redis_client = Redis(host=redis_host, port=redis_port, password=redis_password, db=redis_db)
        self.init = False

    def parse_row_key(self, **kwargs) -> List[tuple]:
        return self.data_source.parse_row_key(**kwargs)

    def subscribe(self, row_key: str):
        return self.data_source.subscribe(row_key)

    def unsubscribe(self, row_key: str):
        return self.data_source.unsubscribe(row_key)
    
    def get_supported_parameters(self) -> List[Field]:
        return self.data_source.get_supported_parameters()

    def on_stop(self):
        if self.init:
            self.data_source.on_stop()
        self.redis_client.close()

    @staticmethod
    def _serialize_kline(kline: KLine) -> bytes:
        """将 KLine 序列化为 MessagePack bytes"""
        # 转换枚举为值，Decimal 转为字符串（避免精度丢失）
        data = {
            'symbol': kline.symbol,
            'market': kline.market,
            'quote_currency': kline.quote_currency,
            'ins_type': kline.ins_type.value if isinstance(kline.ins_type, TradeInsType) else kline.ins_type,
            'asset_type': kline.asset_type.value,
            'open': str(kline.open) if kline.open is not None else None,
            'close': str(kline.close) if kline.close is not None else None,
            'high': str(kline.high) if kline.high is not None else None,
            'low': str(kline.low) if kline.low is not None else None,
            'volume': str(kline.volume) if kline.volume is not None else None,
            'amount': str(kline.amount) if kline.amount is not None else None,
            'start_time': kline.start_time,
            'end_time': kline.end_time,
            'current_time': kline.current_time,
            'timeframe': kline.timeframe.value if isinstance(kline.timeframe, TimeFrame) else kline.timeframe,
            'is_finished': kline.is_finished,
            'data_source_id': kline.data_source_id,
            'dynamic_attrs': kline.dynamic_attrs,
            'metadata': kline.metadata,
        }
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def _deserialize_kline(data: bytes) -> KLine:
        """从 MessagePack bytes 反序列化为 KLine"""
        raw = msgpack.unpackb(data, raw=False)
        # 枚举还原
        ins_type = TradeInsType(raw['ins_type'])
        timeframe = TimeFrame(raw['timeframe'])
        asset_type = AssetType(raw['asset_type'])

        return KLine(
            symbol=raw['symbol'],
            market=raw['market'],
            data_type=DataType.KLINE,
            quote_currency=raw['quote_currency'],
            ins_type=ins_type,
            asset_type=asset_type,
            open=Decimal(raw['open']) if raw['open'] is not None else None,
            close=Decimal(raw['close']) if raw['close'] is not None else None,
            high=Decimal(raw['high']) if raw['high'] is not None else None,
            low=Decimal(raw['low']) if raw['low'] is not None else None,
            volume=Decimal(raw['volume']) if raw['volume'] is not None else None,
            amount=Decimal(raw['amount']) if raw['amount'] is not None else None,
            start_time=raw['start_time'],
            end_time=raw['end_time'],
            current_time=raw['current_time'],
            timeframe=timeframe,
            is_finished=raw['is_finished'],
            data_source_id=raw.get('data_source_id'),
            dynamic_attrs=raw.get('dynamic_attrs', {}),
            metadata=raw.get('metadata', {}),
        )

    def save_klines_batch(self, key: str, klines: List[KLine]) -> None:
        """批量保存 KLine（使用 pipeline 提升性能）"""
        if not klines:
            return
        mapping = {}
        ct = 0
        for kline in klines:
            if kline is None or kline.start_time is None:
                continue  # 跳过无效数据
            member = self._serialize_kline(kline)
            score = kline.start_time
            mapping[member] = score
            ct += 1
        if not mapping:
            return

        self.redis_client.zadd(key, mapping)
        self.redis_client.expire(key, 60 * 60 * 24)
        return ct
    
    
    def get_history_data(self, row_key: str, start_time: datetime | int = None, end_time: datetime | int = None, limit: int = None,
                        pre_load_start_time: datetime | int = None, pre_load_end_time: datetime | int = None, **kwargs) -> Iterator[Any]:
        # 转换datetime参数为int，确保可哈希
        assert start_time is not None and end_time is not None and row_key is not None
        normalized_start = DateTimeUtils.datetime_to_timestamp(start_time) if isinstance(start_time, datetime) else start_time
        normalized_end = DateTimeUtils.datetime_to_timestamp(end_time) if isinstance(end_time, datetime) else end_time
        normalized_pre_load_start = normalized_start
        if pre_load_start_time is not None:
            normalized_pre_load_start = DateTimeUtils.datetime_to_timestamp(pre_load_start_time) if isinstance(pre_load_start_time, datetime) else pre_load_start_time
        normalized_pre_load_end = normalized_end
        if pre_load_end_time is not None:
            normalized_pre_load_end = DateTimeUtils.datetime_to_timestamp(pre_load_end_time) if isinstance(pre_load_end_time, datetime) else pre_load_end_time
        
        key = f"{row_key}:{normalized_pre_load_start}-{normalized_pre_load_end}"
        if self.redis_client.exists(key):
            return self.get_klines_by_time_range(key, normalized_start, normalized_end)
        logger.info(f"get_history_data: row_key={row_key}, start_time={start_time}, end_time={end_time}, limit={limit}, **kwargs={kwargs}")
        with RedisLock(self.redis_client, f"lock:{key}", expire=60, auto_renewal=True):
            if not self.redis_client.exists(key):
                result_list = self._get_history_data(row_key, normalized_pre_load_start, normalized_pre_load_end, limit, **kwargs)
                ct = self.save_klines_batch(key, result_list)
                logger.info(f"get_history_data: row_key={row_key}, count={ct}")
        return self.get_klines_by_time_range(key, normalized_start, normalized_end)

    def get_klines_by_time_range(self, key: str, start_timestamp: int, end_timestamp: int) -> Iterator[KLine]:
        """
        按时间范围查询 KLine（闭区间 [start, end]）
        :param key: key名称
        :param start_timestamp: 毫秒时间戳
        :param end_timestamp: 毫秒时间戳
        :param with_scores: 是否返回时间戳（一般不需要，KLine 自带）
        :return: KLine 列表（按时间升序）
        """
        for mem in self.redis_client.zrangebyscore(key, min=start_timestamp, max=end_timestamp, withscores=False):
            yield self._deserialize_kline(mem)

    def _get_history_data(self, row_key: str, start_time: int | None = None,
                          end_time: int | None = None, limit: int = None, **kwargs) -> Iterator[Any]:
        logger.info(f"Cache miss for row_key={row_key}, start={start_time}, end={end_time}, limit={limit}")
        if not self.init:
            self.data_source.on_start()
            self.init = True
        return self.data_source.get_history_data(row_key=row_key, start_time=start_time, end_time=end_time, limit=limit, **kwargs)

    