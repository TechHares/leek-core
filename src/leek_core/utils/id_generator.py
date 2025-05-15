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

    def _current_millis(self):
        return int(time.time() * 1000)

    def next_id(self) -> int:
        """
        获取一个全局唯一且递增的分布式数字ID。
        :return: int
        """
        with self._lock:
            timestamp = self._current_millis()
            if timestamp < self._last_timestamp:
                raise Exception("系统时钟回拨，ID生成异常")
            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self._max_sequence
                if self._sequence == 0:
                    while True:
                        timestamp = self._current_millis()
                        if timestamp > self._last_timestamp:
                            break
            else:
                self._sequence = 0
            self._last_timestamp = timestamp
            id_ = ((timestamp - self._epoch) << (self._worker_id_bits + self._sequence_bits)) | \
                  (self.worker_id << self._sequence_bits) | \
                  self._sequence
            return id_


# 用法示例：
id_gen = IdGenerator(worker_id=0)


def generate_str():
    return str(id_gen.next_id())


def generate():
    return id_gen.next_id()
