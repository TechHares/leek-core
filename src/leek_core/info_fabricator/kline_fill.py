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

    填充优先级：
    1. 优先使用K线的market字段对应的数据源填充（如okx、gate、binance）
    2. 如果market不支持，则使用配置的fill_method指定的数据源
    3. 如果市场数据填充失败，则使用算法填充

    典型应用场景：
    - 数据源推送不稳定、偶发丢包时，保障策略回测/实盘的K线连续性。
    - 多数据源拼接、历史数据回补等场景下，自动修复时间断档。

    注意事项：
    - 填充K线为估算数据，仅用于辅助分析和保持数据流完整，不能作为真实成交依据。
    """
    priority = -100
    process_data_type = {DataType.KLINE}
    display_name = "缺失K线填充"
    # 支持的市场数据源
    SUPPORTED_MARKETS = {"okx", "gate", "binance"}
    
    init_params: List[Field] = [
        Field(name="fill_method", label="备用填充方法", type=FieldType.RADIO, default="okx",
              choices=[
                  ("okx", "OKX市场数据"),
                  ("gate", "Gate市场数据"),
                  ("binance", "Binance市场数据"),
                  ("algorithm", "算法填充"),
              ],
              choice_type=ChoiceType.STRING,
              description="当K线的market字段不支持时使用的备用填充方法"),
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
        # 缓存数据源实例，避免重复创建
        self._data_sources: Dict[str, object] = {}
    
    def _get_data_source(self, market: str):
        """
        获取或创建指定市场的数据源实例
        
        参数:
            market: 市场标识（okx、gate、binance）
            
        返回:
            数据源实例，如果不支持则返回None
        """
        if market not in self.SUPPORTED_MARKETS:
            return None
        
        if market not in self._data_sources:
            if market == "okx":
                from leek_core.data import OkxDataSource
                self._data_sources[market] = OkxDataSource()
            elif market == "gate":
                from leek_core.data import GateDataSource
                self._data_sources[market] = GateDataSource()
            elif market == "binance":
                from leek_core.data import BinanceDataSource
                self._data_sources[market] = BinanceDataSource()
        
        return self._data_sources.get(market)

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
        
        # 确定使用的市场数据源：优先使用K线的market，否则使用配置的fill_method
        market = kline.market
        if market not in self.SUPPORTED_MARKETS:
            market = self.fill_method
        
        logger.info(f"填充K线[{market}]: {last_kline} - {kline}")
        
        # 尝试使用市场数据填充
        if market in self.SUPPORTED_MARKETS:
            result = self._market_data_fill(last_kline, kline, market)
            if result is not None:
                return result
            # 市场数据填充失败，使用算法填充
            logger.warning(f"市场数据填充失败，切换到算法填充: {kline.row_key}")
        
        return self._algorithm_fill(last_kline, kline)
    
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

    def _market_data_fill(self, last_kline: KLine, kline: KLine, market: str) -> List[KLine]:
        """
        使用市场数据源填充K线数据（支持OKX、Gate、Binance）
        
        参数:
            last_kline: 上一根K线
            kline: 当前K线
            market: 市场标识（okx、gate、binance）
            
        返回:
            填充后的K线列表，如果填充失败返回None
        """
        data_source = self._get_data_source(market)
        if data_source is None:
            return None
        
        last_start_time = last_kline.start_time
        if not last_kline.is_finished:
            last_start_time = last_kline.start_time - kline.timeframe.milliseconds
        
        try:
            klines = list(data_source.get_history_data(
                row_key=kline.row_key,
                start_time=last_start_time,
                end_time=kline.end_time,
                limit=100,
            ))
        except Exception as e:
            logger.error(f"获取{market}历史K线数据异常: {e}", exc_info=True)
            return None
        
        r = []
        for k in klines:
            if kline.start_time >= k.start_time >= last_start_time:
                if k.start_time == last_start_time and last_kline.is_finished:
                    continue
                k.metadata = {"is_filled": True}
                r.append(k)
        if len(r) == 0:
            logger.error(f"填充{market}的K线数据失败: 最后K线时间{last_start_time}, 当前K线时间{kline.start_time}, k线时间间隔{kline.timeframe.milliseconds}, 查询结果klines={klines}")
            return None
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
