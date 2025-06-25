#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
from leek_core.indicators import *
from leek_core.models import TimeFrame, TradeInsType
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from leek_core.utils import DateTimeUtils


class TestIndicatorChan(unittest.TestCase):
    def test_TR(self):
        view = IndicatorView([MA()], symbol="ETH",timeframe=TimeFrame.M1,
                             start_time=DateTimeUtils.to_timestamp("2025-03-01 23:10"),
                             end_time=DateTimeUtils.to_timestamp("2025-03-25 20:00"))
        chan = Chan(bi_zs=True, seg=True, seg_zs=True, dr=True, dr_zs=True, exclude_equal=False, zs_max_level=2,
                    allow_similar_zs=True)
        view.data_source.connect()
        data = view.data_source.get_history_data(start_time=view.start_time, end_time=view.end_time,
                                                       symbol=view.symbol, timeframe=view.timeframe,
                                                       market=view.market, quote_currency=view.quote_currency,
                                                       ins_type=view.ins_type)
        data = list(data)
        for kline in data:
            chan.update(kline)
        chan.mark_on_data()

        df = pd.DataFrame([{
            "start_time": x.start_time,
            "symbol": x.symbol,
            "open": x.open,
            "high": x.high,
            "low": x.low,
            "close": x.close,
            "volume": x.volume,
            "amount": x.amount,
            "interval": x.interval,
            "finish": x.is_finished,
            "dif": x.dif,
            "dea": x.dea,
            "m": x.m,
            "chan_high": x.chan_high,
            "chan_low": x.chan_low,
            "chan_open": x.chan_open,
            "chan_close": x.chan_close,
            "ck_idx": x.ck_idx,
            "bi": x.bi,
            "bi_value": x.bi_value,
            "bi_idx": x.bi_idx,
            "seg": x.seg,
            "seg_value": x.seg_value,
            "seg_idx": x.seg_idx,
            "dr": x.dr,
            "lower_ps": x.lower_ps,
            "lower_pst": x.lower_pst,
            "current_pst": x.current_pst,
            "dr_": x.dr_,
            "seg_": x.seg_,
            "bi_": x.bi_,
        } for x in data])
        df['Datetime'] = pd.to_datetime(df['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        if "seg" in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Datetime'], y=df["seg_value"], mode='text', text=df["seg_idx"], name='seg_idx'),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['seg'], mode='lines', line=dict(color='blue', width=2),
                                     name='segment', connectgaps=True), row=1, col=1)
        if "seg_" in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Datetime'], y=df['seg_'], mode='lines', line=dict(color='blue', width=2, dash='dash'),
                           name='segment', connectgaps=True), row=1, col=1)

        if "bi" in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df["bi_value"], mode='text', text=df["bi_idx"], name='bi_idx'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bi'], mode='lines', line=dict(color='black', width=1),
                                     name='chan b', connectgaps=True), row=1, col=1)
        if "bi_" in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Datetime'], y=df['bi_'], mode='lines', line=dict(color='black', width=1, dash='dash'),
                           name='chan b', connectgaps=True), row=1, col=1)

        if "dr" in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['dr'], mode='lines', line=dict(color='darkblue', width=3),
                                     name='chan dr', connectgaps=True), row=1, col=1)
        if "dr_" in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Datetime'], y=df['dr_'], mode='lines',
                           line=dict(color='darkblue', width=3, dash='dash'),
                           name='chan dr', connectgaps=True), row=1, col=1)

        # zs
        colors = ["orange", "skyblue", "lightgreen", "gainsboro", "darkblue"]
        if chan.seg_zs:
            for zs in chan.zs_manager.zs_list:
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=zs.level + 1),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
            if chan.zs_manager.cur_zs is not None and chan.zs_manager.cur_zs.is_satisfy:
                zs = chan.zs_manager.cur_zs
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=zs.level + 1, dash='dash'),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
        if chan.dr_zs:
            for zs in chan.drzs_manager.zs_list:
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=zs.level + 3),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
            if chan.drzs_manager.cur_zs is not None and chan.drzs_manager.cur_zs.is_satisfy:
                zs = chan.drzs_manager.cur_zs
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=zs.level + 3, dash='dash'),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
        if chan.bi_zs:
            for zs in chan.bizs_manager.zs_list:
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=1),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
            if chan.bizs_manager.cur_zs is not None and chan.bizs_manager.cur_zs.is_satisfy:
                zs = chan.bizs_manager.cur_zs
                fig.add_shape(
                    type='rect',
                    x0=pd.to_datetime([zs.start_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.down_line,
                    x1=pd.to_datetime([zs.end_timestamp + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.up_line,
                    line=dict(color=colors[zs.level], width=1, dash='dash'),
                    fillcolor=None,  # 透明填充，只显示边框
                    name='Highlight Area'
                )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

        fig.add_trace(go.Candlestick(x=df['Datetime'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name="CRV"),
                      row=1, col=1)
        fig.show()



if __name__ == '__main__':
    unittest.main()