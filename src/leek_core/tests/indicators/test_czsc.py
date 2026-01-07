#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from decimal import Decimal

import numpy as np
from leek_core.indicators import *
from leek_core.indicators.czsc.czsc import CZSC
from leek_core.indicators.czsc.d import D
from leek_core.indicators.czsc.bi import BI
from leek_core.indicators.czsc.zs import ZS
from leek_core.models import PositionSide, TimeFrame, TradeInsType, KLine
from leek_core.engine import IndicatorView
from leek_core.data import DataSource, ClickHouseKlineDataSource, RedisClickHouseDataSource
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from leek_core.utils import DateTimeUtils
from leek_core.engine import StrategyDebugView
from leek_core.models import TimeFrame

import plotly.graph_objs as go

class TestCzsc(unittest.TestCase):
    def _to_pdf(self, data):
        return pd.DataFrame([{
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
            "bi_dr": x.bi_dr,
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

    def test_czsc(self):
        view = IndicatorView([MA()], symbol="CRV", timeframe=TimeFrame.M1,
                             start_time=DateTimeUtils.to_timestamp("2025-12-01 23:10"),
                             end_time=DateTimeUtils.to_timestamp("2025-12-12 20:00"),
                             data_source=RedisClickHouseDataSource(password="default"))
        key = KLine.pack_row_key(view.symbol, view.quote_currency, view.ins_type, view.timeframe)
        data = view.data_source.get_history_data(start_time=view.start_time, end_time=view.end_time,
                                                 row_key=key, timeframe=view.timeframe,
                                                 market=view.market, quote_currency=view.quote_currency,
                                                 ins_type=view.ins_type)
        chan = Chan(bi_zs=True, seg=True, seg_zs=True, dr=False, dr_zs=False, exclude_equal=False, zs_max_level=1, allow_similar_zs=False)
        czsc = CZSC(debug=True)
        data = list(data)
        for kline in data:
            chan.update(kline)
            czsc.update(kline)
        chan.mark_on_data()
        df = self._to_pdf(data)
        df['Datetime'] = pd.to_datetime(df['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(go.Candlestick(x=df['Datetime'],
                                     open=df['chan_open'],
                                     high=df['chan_high'],
                                     low=df['chan_low'],
                                     close=df['chan_close'],
                                     name="CRV"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bi'], mode='lines', line=dict(color='black', width=1),
                                     name='chan b', connectgaps=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['seg'], mode='lines', line=dict(color='blue', width=2),
                                     name='chan seg', connectgaps=True), row=1, col=1)
        df2 = pd.DataFrame([{
            "start_time": x.start_time,
            "open": x.low if x.is_up else x.high,
            "close": x.high if x.is_up else x.low,
            "high": x.high,
            "low": x.low,
        } for x in czsc.ks])
        df2['Datetime'] = pd.to_datetime(df2['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        fig.add_trace(go.Candlestick(x=df2['Datetime'],
                                     open=df2['open'],
                                     high=df2['high'],
                                     low=df2['low'],
                                     close=df2['close'],
                                     name="CRV"),
                      row=2, col=1)
        df3 = pd.DataFrame([point for x in czsc.bis for point in [
            {"start_time": x.start_time, "value": x.low if x.is_up else x.high},
            {"start_time": x.end_time, "value": x.high if x.is_up else x.low}
        ]])
        print(f"旧版笔数: {len(chan.bi_manager)}，新版笔数: {len(czsc.bis)}")
        df3['Datetime'] = pd.to_datetime(df3['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        fig.add_trace(go.Scatter(x=df3['Datetime'], y=df3['value'], mode='lines', line=dict(color='black', width=1),
                                     name='bi', connectgaps=True), row=2, col=1)

        df4 = pd.DataFrame([point for x in czsc.ds for point in [
            {"start_time": x.start_time, "value": x.start_value},
            {"start_time": x.end_time, "value": x.end_value}
        ]])
        print(f"旧版段数: {len(chan.seg_manager)}，新版段数: {len(czsc.ds)}")
        df4['Datetime'] = pd.to_datetime(df4['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        fig.add_trace(go.Scatter(x=df4['Datetime'], y=df4['value'], mode='lines', line=dict(color='blue', width=2),
                                     name='d', connectgaps=True), row=2, col=1)


        # fig.add_trace(go.Scatter(x=df3['Datetime'], y=df3['value'], mode='lines', line=dict(color='black', width=1),
        #                              name='bi', connectgaps=True), row=3, col=1)
        # start = False
        # ds = []
        # for bi in czsc.bis:
        #     if not start:
        #         if bi == czsc.ds[0].eles[0]:
        #             ds.append(D(bi))
        #             start = True
        #         continue
        #     res = ds[-1].update(bi, debug=True)
        #     for rd in res or []:
        #         ds.append(rd)
        # print(f"旧版段数: {len(chan.seg_manager)}，新版段数: {len(ds)}")
        # df4 = pd.DataFrame([point for x in ds for point in [
        #     {"start_time": x.start_time, "value": x.start_value},
        #     # {"start_time": x.end_time, "value": x.end_value}
        # ]])
        # if ds[-1].peak_idx != 0:
        #     df4 = pd.concat([df4, pd.DataFrame([{
        #         "start_time": ds[-1].eles[ds[-1].peak_idx].end_time,
        #         "value": ds[-1].eles[ds[-1].peak_idx].end_value,
        #     }])], ignore_index=True)
        # df4['Datetime'] = pd.to_datetime(df4['start_time'] + 8 * 60 * 60 * 1000, unit='ms')
        # fig.add_trace(go.Scatter(x=df4['Datetime'], y=df4['value'], mode='lines', line=dict(color='blue', width=2),
        #                              name='d', connectgaps=True), row=3, col=1)
        print(f"中枢数: {len(czsc.zs_list)}")
        for zs in czsc.zs_list:
            fig.add_shape(
                type='rect',
                x0=pd.to_datetime([zs.start_time + 8 * 60 * 60 * 1000], unit="ms")[0], y0=zs.low,
                x1=pd.to_datetime([zs.end_time + 8 * 60 * 60 * 1000], unit="ms")[0], y1=zs.high,
                line=dict(color='orange', width=1),
                fillcolor=None,  # 透明填充，只显示边框
                name='Highlight Area',
                row=2, col=1
            )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        fig.show()


if __name__ == '__main__':
    unittest.main()
