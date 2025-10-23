#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque

from leek_core.models import KLine, TimeFrame


class T:
    def __init__(self, max_cache=100):
        self.cache = deque(maxlen=max_cache)

    def update(self, data: KLine):
        pass

    def last(self, n=100):
        n = min(len(self.cache), n)
        return list(self.cache)[-n:]


class MERGE(T):
    """
    K线合并(周期升级)
    """
    def __init__(self, window=9, max_cache=100):
        T.__init__(self, max_cache)
        self.window = window
        self.q = deque(maxlen=window)

        self.not_start = True

    def update(self, data: KLine):
        if self.not_start:
            if  data.start_time % int(self.window * data.timeframe.milliseconds) != 0:
                return None
            self.not_start = False

        if len(self.q) > 0 and not self.q[-1].is_finished:
            self.q.pop()

        self.q.append(data)
        ls = list(self.q)
        r = KLine(symbol=data.symbol,
              market=data.market,
              timeframe=TimeFrame.from_milliseconds(data.timeframe.milliseconds * self.window),
              start_time=ls[0].start_time,
              end_time=ls[-1].end_time,
              current_time=data.current_time or data.end_time,
              open=ls[0].open,
              high=max(d.high for d in ls),
              low=min(d.low for d in ls),
              close=data.close,
              volume=sum(d.volume for d in ls),
              amount=sum(d.amount for d in ls),
              quote_currency=data.quote_currency,
              ins_type=data.ins_type,
              is_finished=False,
              )
        r.merge = True
        r.end_time = r.end_time + r.timeframe.milliseconds
        if len(ls) == self.window and data.is_finished:
            r.is_finished = True
            self.q.clear()
            self.cache.append(r)

        return r

