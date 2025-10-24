# -*- coding: utf-8 -*-
"""
分布式顺序唯一数字ID生成工具（Snowflake简化版，单例模式）。
支持多台机器传入机器号（worker_id），生成全局唯一且递增的数字ID。
同一worker_id在单进程内只允许有一个实例，线程安全。
"""
import threading
import time


class IdGenerator:
    """
    分布式数字ID生成器（雪花算法简化版，线程安全，单例模式）。
    生成ID结构：
        41位时间戳（毫秒） | 10位机器号 | 12位序列号
    同一worker_id在单进程内只允许有一个实例。
    """
    _epoch = 1704067200000  # 可自定义起始时间戳（毫秒），如2024-01-01
    _timestamp_bits = 41
    _worker_id_bits = 10
    _sequence_bits = 12

    _max_worker_id = (1 << _worker_id_bits) - 1
    _max_sequence = (1 << _sequence_bits) - 1

    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, worker_id: int):
        if not (0 <= worker_id <= cls._max_worker_id):
            raise ValueError(f"worker_id 必须在0~{cls._max_worker_id}之间")
        if worker_id in cls._instances:
            return cls._instances[worker_id]
        with cls._instances_lock:
            if worker_id not in cls._instances:
                instance = super().__new__(cls)
                instance._init_instance(worker_id)
                cls._instances[worker_id] = instance
            return cls._instances[worker_id]

    def _init_instance(self, worker_id: int):
        self.worker_id = worker_id
        self._lock = threading.Lock()
        self._last_timestamp = -1
        self._sequence = 0
        # 以单调时钟构造逻辑时间，避免系统时钟回拨导致倒退
        self._base_wall_ms = int(time.time() * 1000)
        try:
            self._base_mono_ms = time.monotonic_ns() // 1_000_000
            self._get_mono_ms = lambda: time.monotonic_ns() // 1_000_000
        except Exception:
            # 兼容环境：退回到 monotonic()
            self._base_mono_ms = int(time.monotonic() * 1000)
            self._get_mono_ms = lambda: int(time.monotonic() * 1000)

    def _current_millis(self):
        # 使用单调时钟推导逻辑毫秒时间，避免回拨
        mono_ms = self._get_mono_ms()
        logical_ms = self._base_wall_ms + (mono_ms - self._base_mono_ms)
        # 夹逼以确保非递减
        if self._last_timestamp >= 0 and logical_ms < self._last_timestamp:
            return self._last_timestamp
        return int(logical_ms)

    def next_id(self) -> int:
        """
        获取一个全局唯一且递增的分布式数字ID。
        :return: int
        """
        with self._lock:
            timestamp = self._current_millis()
            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self._max_sequence
                if self._sequence == 0:
                    while True:
                        timestamp = self._current_millis()
                        if timestamp > self._last_timestamp:
                            break
                        # 避免忙等占满CPU
                        time.sleep(0.0001)
            else:
                self._sequence = 0
            self._last_timestamp = timestamp
            
            # 确保时间戳部分不超过41位
            timestamp_bits = (timestamp - self._epoch) & ((1 << self._timestamp_bits) - 1)
            
            # 组合ID
            id_ = (timestamp_bits << (self._worker_id_bits + self._sequence_bits)) | \
                  (self.worker_id << self._sequence_bits) | \
                  self._sequence
            return id_


_id_generator = IdGenerator(worker_id=0)
def set_worker_id(worker_id=0):
    global _id_generator
    _id_generator = IdGenerator(worker_id=worker_id)

def generate_str(worker_id=None):
    return str(generate(worker_id))


def generate(worker_id=0):
    if worker_id is None:
        return _id_generator.next_id()

    return IdGenerator(worker_id=worker_id).next_id()
