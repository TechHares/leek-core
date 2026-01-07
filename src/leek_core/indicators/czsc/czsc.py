#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅基础组件
"""
from typing import List
from collections import deque

from leek_core.indicators.chan import zs
from leek_core.models.parameter import DateTimeUtils
from .k import CK
from .bi import BI
from .d import D
from .zs import ZS
from leek_core.models import KLine, PositionSide


class CZSC():
    """
    缠中说禅
    """
    def __init__(self, debug: bool = False):
        super().__init__()

        self.ks = deque[CK](maxlen=20000 if debug else 200)
        self.bis = deque[BI](maxlen=20000 if debug else 50)
        self.ds = deque[D](maxlen=20000 if debug else 30)
        self.zs_list = deque[ZS](maxlen=20000 if debug else 20)
        self.zs = ZS()
        self.tmp_bi_list: List[BI] = []

    def update(self, k: KLine):
        if len(self.ks) == 0:
            self.ks.append(CK(k, PositionSide.LONG))
            return
        
        rk = self.ks[-1].update(k)
        if rk is not None:
            self.ks.append(rk)
        if len(self.bis) == 0:
            self.bis.append(BI())
            return
        rb = self.bis[-1].update(rk)
        if rb:
            self.bis.append(rb)
        if len(self.ds) == 0:
            self.try_confirm_first_seg(rb)
        else:
            rds = self.ds[-1].update(rb)
            for rd in rds or []:
                self.ds.append(rd)
            self.update_zs(rds)

    def update_zs(self, rds):
        if rds is None or len(rds) == 0:
            rds = [None]
                    
        for rd in rds:
            self.zs.update(rd)
            if self.zs.is_zs():
                if len(self.zs_list) == 0 or self.zs.start_time > self.zs_list[-1].start_time:
                    self.zs_list.append(self.zs.to_model())
                else:
                    self.zs_list[-1] = self.zs.to_model()
    
    def try_confirm_first_seg(self, bi: BI):
        if bi:
            self.tmp_bi_list.append(bi)

        if len(self.tmp_bi_list) < 10:
            return
        if self.tmp_bi_list[0].is_up:
            high_bi = max(self.tmp_bi_list, key=lambda x: x.end_value)
            high_bi_idx = self.tmp_bi_list.index(high_bi) + 1
            low_bi = min(self.tmp_bi_list, key=lambda x: x.start_value)
            low_bi_idx = self.tmp_bi_list.index(low_bi)
        else:
            high_bi = max(self.tmp_bi_list, key=lambda x: x.start_value)
            high_bi_idx = self.tmp_bi_list.index(high_bi)
            low_bi = min(self.tmp_bi_list, key=lambda x: x.end_value)
            low_bi_idx = self.tmp_bi_list.index(low_bi) + 1

        idx = min(high_bi_idx, low_bi_idx) # 出现在同一笔内 向后取  否则取先比
        if abs(high_bi_idx - low_bi_idx) == 1:
            idx = max(high_bi_idx, low_bi_idx)
        if idx + 5 > len(self.tmp_bi_list): # 后面至少空余5笔
            return
        d = D(self.tmp_bi_list[idx])
        self.ds.append(d)
        for bi in self.tmp_bi_list[idx + 1:]:
            rds = self.ds[-1].update(bi)
            for rd in rds or []:
                self.ds.append(rd)
        self.tmp_bi_list = []
    
    
