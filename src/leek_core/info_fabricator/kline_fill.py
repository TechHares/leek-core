#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K线数据填充处理器模块，用于处理缺失的K线数据。
"""

from collections import deque
from decimal import Decimal
from typing import List, Dict, Tuple

from leek_core.models import DataType, KLine, FieldType, Field, ChoiceType
from leek_core.utils import get_logger
from .base import Fabricator

logger = get_logger(__name__)


class KLineFillFabricator(Fabricator):
    """
    K线数据填充处理器，用于处理缺失的K线数据。

    本插件主要用于自动检测和填补K线（蜡烛图）数据流中的缺失区间，保证下游策略和风控模块能够收到连续、完整的K线数据序列。

    主要功能与原理：
    1. 自动记录每个交易对的历史K线数据，维护时间序列连续性。
    2. 检测到相邻K线之间存在时间缺口时，自动插入补全K线。
    3. 填充的K线数据会采用前后K线的价格均值、成交量等信息进行合理估算，尽量还原真实走势。

    典型应用场景：
    - 数据源推送不稳定、偶发丢包时，保障策略回测/实盘的K线连续性。
    - 多数据源拼接、历史数据回补等场景下，自动修复时间断档。

    注意事项：
    - 填充K线为估算数据，仅用于辅助分析和保持数据流完整，不能作为真实成交依据。
    """
    priority = -100
    process_data_type = {DataType.KLINE}
    display_name = "缺失K线填充"
    init_params: List[Field] = [
        Field(name="fill_method", label="填充方法", type=FieldType.RADIO, default="okx",
              choices=[
                  ("okx", "OKX市场数据"),
                  ("algorithm", "算法填充"),
              ],
              choice_type=ChoiceType.STRING),
    ]

    def __init__(self, fill_method: str = "okx"):
        """
        初始化K线填充处理器
        """
        super().__init__()
        self.fill_method = fill_method
        # 使用字典存储不同交易对的历史数据
        # key: (data_source_id, symbol, quote_currency, ins_type, timeframe)
        # value: deque of KLine
        self.history: Dict[Tuple, deque] = {}
        if self.fill_method == "okx":
            from leek_core.data import OkxDataSource
            self.okx_data_source = OkxDataSource()

    def _get_history(self, kline: KLine) -> deque:
        """
        获取指定交易对的历史数据队列
        
        参数:
            kline: K线数据
            
        返回:
            历史数据队列
        """
        key = kline.row_key
        if key not in self.history:
            self.history[key] = deque(maxlen=5)
        return self.history[key]

    def process(self, kline: List[KLine]) -> List[KLine]:
        """
        处理K线数据，返回填充后的K线列表
        
        参数:
            kline: 输入的K线数据
            
        返回:
            处理后的K线列表，如果不需要填充则只包含输入的K线
        """
        for k in kline:
            for nk in self._fill(k):
                yield nk

    def _fill(self, kline: KLine) -> List[KLine]:
        last_kline = self._check_kline_seq(kline)
        if last_kline is None:
            return [kline]
        logger.info(f"填充K线[{self.fill_method}]: {last_kline} - {kline}")
        if self.fill_method == "okx":
            return self._okx_fill(last_kline, kline)
        return self._algorithm_fill(last_kline,kline)
    
    def _check_kline_seq(self, kline: KLine) -> bool:
        history = self._get_history(kline)
        if len(history) == 0:
            history.append(kline)
            return None

        last_kline = history[-1]
        if not history[-1].is_finished:
            history.pop()
        # 计算时间间隔（毫秒）
        time_diff = kline.start_time - last_kline.start_time

        # 历史数据为空，或者时间间隔为0， 或者 新K线开启
        if time_diff == 0 or (time_diff <= kline.timeframe.milliseconds and last_kline.is_finished):
            history.append(kline)
            return None
        return last_kline

    def _okx_fill(self, last_kline: KLine, kline: KLine) -> List[KLine]:
        """
        填充OKX的K线数据
        """
        last_start_time = last_kline.start_time
        if not last_kline.is_finished:
            last_start_time = last_kline.start_time - kline.timeframe.milliseconds
        
        klines = list(self.okx_data_source.get_history_data(
            row_key=kline.row_key,
            start_time=last_start_time,
            end_time=kline.end_time,
            limit=100,
        ))
        r = []
        for k in klines:
            if kline.start_time >= k.start_time >= last_start_time:
                if k.start_time == last_start_time and last_kline.is_finished:
                    continue
                k.metadata = {"is_filled": True}
                r.append(k)
        if r[-1].start_time < kline.start_time:
            r.append(kline)
        for k in r:
            self._get_history(kline).append(k)
        return r

    def _algorithm_fill(self, last_kline: KLine, kline: KLine) -> List[KLine]:
        history = self._get_history(kline)
        filled_kline = KLine(
            symbol=kline.symbol,
            market=kline.market,
            open=last_kline.close,
            high=last_kline.high,
            low=last_kline.low,
            target_instance_id=kline.target_instance_id,
            close=(last_kline.high + last_kline.low) / 2,
            volume=sum(k.volume for k in history) / len(history),
            amount=sum(k.amount for k in history) / len(history),
            start_time=last_kline.end_time,
            end_time=last_kline.end_time + kline.timeframe.milliseconds,
            current_time=kline.current_time,
            timeframe=kline.timeframe,
            ins_type=kline.ins_type,
            quote_currency=kline.quote_currency,
            is_finished=True,  # 填充的K线都是已完成的,
            data_source_id=kline.data_source_id,
            data_type=kline.data_type,
            metadata={"is_filled": True}  # 标记这是填充的K线
        )
        filled_kline.asset_type = kline.asset_type
        if not last_kline.is_finished:
            r = Decimal(
                int(kline.timeframe.milliseconds * 100 / (last_kline.current_time - last_kline.start_time + 1)) / 100)
            filled_kline.volume = last_kline.volume * r
            filled_kline.amount = last_kline.amount * r
            filled_kline.start_time = last_kline.start_time
            filled_kline.end_time = last_kline.end_time

        if filled_kline.end_time == kline.start_time:
            filled_kline.close = kline.open
        filled_kline.high = max(filled_kline.open, filled_kline.close, filled_kline.high)
        filled_kline.low = min(filled_kline.open, filled_kline.close, filled_kline.low)
        # 添加当前K线到历史数据
        history.append(kline)
        if filled_kline.end_time >= kline.start_time:
            return [filled_kline, kline]
        return [filled_kline] + self._algorithm_fill(filled_kline, kline)
