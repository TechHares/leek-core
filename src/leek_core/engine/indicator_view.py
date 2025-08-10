#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

from leek_core.models.data import KLine
from .base import Engine, LeekComponent

from leek_core.indicators import T, MERGE
from leek_core.data import DataSource, ClickHouseKlineDataSource
from leek_core.models import TimeFrame, TradeInsType
from leek_core.utils import get_logger, DateTimeUtils

logger = get_logger(__name__)
colors = [
    '#1f77b4',  # 蓝色系 - 深蓝
    '#ff7f0e',  # 橙色系 - 鲜橙
    '#2ca02c',  # 绿色系 - 绿
    '#d62728',  # 红色系 - 暗红
    '#9467bd',  # 紫色系 - 紫罗兰
    '#8c564b',  # 棕色系 - 深棕
    '#e377c2',  # 粉色系 - 玫瑰粉
    '#7f7f7f',  # 灰色系 - 中灰
    '#bcbd22',  # 黄绿色系 - 橄榄绿
    '#17becf'   # 青色系 - 宝石蓝
]

class IndicatorView(LeekComponent):
    def __init__(self, t: T|list[T], t_count=None,symbol: str = "ETH", start_time: datetime = datetime.now() - timedelta(days=10),
                 end_time: datetime = datetime.now(),
                 timeframe: TimeFrame = TimeFrame.M5, market: str = "okx", quote_currency: str = "USDT",
                 ins_type: TradeInsType = TradeInsType.SWAP, data_source: DataSource = ClickHouseKlineDataSource()):
        super().__init__()
        self.data_source = data_source
        self.t = t if isinstance(t, list) else [t]
        self.t_count = t_count
        if not t_count:
            self.t_count = [1 for i in range(len(self.t))]

        assert len(self.t_count) == len(self.t)
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.timeframe = timeframe
        self.market = market
        self.quote_currency = quote_currency
        self.ins_type = ins_type

        self.color_index = 0

    def get_color(self, color=None):
        if color is not None:
            return color
        color = colors[self.color_index % len(colors)]
        self.color_index += 1
        return color

    def start(self, *args, show_kline=True, show_volume=False, **kwargs):
        self.data_source.connect()
        count = 0
        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "time": []
        }
        for i in range(len(args)):
            data["%s" % i] = []
        if len(self.t) == 1 and self.t_count[0] == 1: # 只有一个指标的是时候省略指标数目的参数
            self.t_count[0] = len(args)
        
        for kline in self.data_source.get_history_data(start_time=self.start_time, end_time=self.end_time,
                                                       row_key=KLine.pack_row_key(self.symbol, self.quote_currency, self.ins_type, self.timeframe),
                                                       market=self.market):
            count += 1
            data["open"].append(kline.open)
            data["high"].append(kline.high)
            data["low"].append(kline.low)
            data["close"].append(kline.close)
            data["volume"].append(kline.volume)
            data["time"].append(DateTimeUtils.to_datetime(kline.start_time))
            _idx = 0
            for i in range(len(self.t)):
                indicators = self.t[i].update(kline)
                ct = self.t_count[i]
                for j in range(ct):
                    try:
                        if indicators is None:
                            data["%s" % _idx].append(None)
                        elif isinstance(indicators, list|tuple):
                            data["%s" % _idx].append(indicators[j] if indicators and len(indicators) > j else None)
                        elif j == 0:
                            data["%s" % _idx].append(indicators)
                        else:
                            raise Exception("indicators is not list or tuple")
                    finally:
                        _idx += 1
        self.data_source.disconnect()
        logger.info(f"数据执行完成，共{count}条")
        self.draw(data, *args, show_kline=show_kline, show_volume=show_volume, **kwargs)

    def draw(self, data, *args, show_kline=True, show_volume=False, **kwargs) -> None:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = pd.DataFrame(data)
        row_count = max((1 if show_kline else 0) + (1 if show_volume else 0), max([arg.get("row", 1) for arg in args]) if args else 0)
        fig = make_subplots(rows=row_count, cols=1, shared_xaxes=True)
        if show_kline:
            fig.add_trace(go.Candlestick(x=df['time'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name=self.symbol),
                          row=1, col=1)
        if show_volume:
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume'),
                          row=2 if show_kline else 1, col=1)

        for i in range(len(args)):
            arg = args[i]
            name = arg.get("name", "%s" % i)
            type = arg.get("type", "line")
            row = arg.get("row", 1)
            color = arg.get("color", None)
            key = "%s" % i
            if type == "line":
                fig.add_trace(go.Scatter(x=df['time'], y=df[key], mode='lines', name=name,
                                         line=dict(color=self.get_color(color), width=arg.get("width", 1))),
                              row=row, col=1)
            if type == "dash_line":
                fig.add_trace(go.Scatter(x=df['time'], y=df[key], mode='lines', name=name,
                                         line=dict(color=self.get_color(color), width=arg.get("width", 1), dash='dash')),
                              row=row, col=1)
            elif type == "bar":
                fig.add_trace(go.Bar(x=df['time'], y=df[key], name=name, marker_color=self.get_color(color)),
                              row=row, col=1)

            elif type == "mark":
                fig.add_trace(go.Scatter(x=df['time'], y=df[key], mode='markers', name=name,
                                         marker=dict(color=self.get_color(color), size=arg.get("size", 5))),
                              row=row, col=1)

        fig.update_layout(height=kwargs.get("height", 600))
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.show()


if __name__ == '__main__':
    pass
