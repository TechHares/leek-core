#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅基础组件
"""
from collections import deque
from time import sleep

from leek_core.utils import DateTimeUtils

from .k import CK
from .base import Component

from leek_core.models import KLine, PositionSide


class BI(Component):
    """
    缠中说禅基础组件定义
    """
    def __init__(self, direction: PositionSide | None = None):
        super().__init__()
        self.direction = direction
        self.candidate_end_point: int = 0
    
    @property
    def start_time(self):
        if len(self.eles) < 2:
            return 0
        return self.eles[1].min_start_time() if self.is_up else self.eles[1].max_start_time()

    @property
    def end_time(self):
        if len(self.eles) < 2:
            return 0
        return self.eles[-2].max_start_time() if self.is_up else self.eles[-2].min_start_time()

    def ensure_direction(self):
        if self.direction is not None:
            return
        eles = [e for e in self.eles if e.is_finished]
        if len(eles) < 3:
            return
        if eles[-1].high < eles[-2].high > eles[-3].high:
            self.high = eles[-2].high
            self.low = self.min
            self.direction = PositionSide.SHORT
        if eles[-1].low > eles[-2].low < eles[-3].low:
            self.low = eles[-2].low
            self.high = self.max
            self.direction = PositionSide.LONG
        unfinished = [e for e in self.eles if not e.is_finished]
        self.eles.clear()
        for e in eles[-3:] + unfinished:
            self.eles.append(e)

    def update(self, k: CK | None):
        if k:
            self.eles.append(k)
        # 确认方向
        self.ensure_direction()
        if self.direction is None:
            return
        
        if len(self.eles) < 6:  # 长度不够
            return
        
        if self.candidate_end_point == 0:  # 没有候选结束点, 需要先找疑似结束的地方
            if self.is_up and self.low > self.min or self.is_down and self.high < self.max: # 新低或新高, 需要更新候选结束点
                self.direction = None
                return
            for i in range(5, len(self.eles)):
                if self.is_strong_enough(self.is_up, 1, i):
                    self.candidate_end_point = i
                    break
        else: # 有候选结束点, 确认是否可以正常结束
            if self.candidate_end_point != len(self.eles) - 1: # 出现新低或新高点更新候选结束点
                if self.is_up and self.eles[-1].high > self.eles[self.candidate_end_point].high or \
                                    self.is_down and self.eles[-1].low < self.eles[self.candidate_end_point].low: 
                    if self.is_up:
                        self.high = self.eles[-1].high
                    else:
                        self.low = self.eles[-1].low
                    self.candidate_end_point = len(self.eles) - 1

            if self.is_strong_enough(self.is_down, self.candidate_end_point, len(self.eles)-1):
                return self.split_bi()
    
    def is_strong_enough(self, is_up: bool, start_idx: int, end_idx: int) -> bool:
        # 顶分型与底分型经过包含处理后，不允许共用 K 线，也就是不能有一 K 线分别属于顶分型与底分型 + 必须有趋势K
        # idx = f"[{start_idx},{end_idx}]{'↑' if is_up else '↓'}({DateTimeUtils.to_datetime(self.eles[start_idx].start_time)} - {DateTimeUtils.to_datetime(self.eles[end_idx].start_time)})"
        if end_idx - start_idx < 4 or self.eles[start_idx].include(self.eles[end_idx]):
            # print(f"1. 必须有趋势K or include {idx} : {end_idx - start_idx < 4} or {self.eles[start_idx].include(self.eles[end_idx])}")
            return False
        if is_up: 
            # 1. 极点成笔 - 候选顶点必须是这段内的最高点
            if self.eles[end_idx].high < max([self.eles[x].high for x in range(start_idx, end_idx+1)]):
                # print(f"1. 极点成笔-{idx}: {self.eles[end_idx].high} < {max([self.eles[x].high for x in range(start_idx, end_idx+1)])}")
                return False
            if self.eles[start_idx].low > min([self.eles[x].low for x in range(start_idx, end_idx+1)]):
                # print(f"2. 极点成笔-{idx}: {self.eles[start_idx].low} > {min([self.eles[x].low for x in range(start_idx, end_idx+1)])}")
                return False
            if max(self.eles[start_idx].high, self.eles[start_idx+1].high) > self.eles[end_idx].high:
                # print(f"3. 力度确认-{idx}: {max(self.eles[start_idx].high, self.eles[start_idx+1].high) > self.eles[end_idx].high}")
                return False
            return True
        else:
            # 1. 极点成笔 - 候选底点必须是这段内的最低点
            if self.eles[end_idx].low > min([self.eles[x].low for x in range(start_idx, end_idx+1)]):
                # print(f"1. 极点成笔-{idx}: {self.eles[end_idx].low} > {min([self.eles[x].low for x in range(start_idx, end_idx+1)])}")
                return False
            if self.eles[start_idx].high < max([self.eles[x].high for x in range(start_idx, end_idx+1)]):
                # print(f"2. 极点成笔-{idx}: {self.eles[start_idx].high} < {max([self.eles[x].high for x in range(start_idx, end_idx+1)])}")
                return False
            
            if min(self.eles[start_idx].low, self.eles[start_idx+1].low) < self.eles[end_idx].low:
                # print(f"3. 力度确认-{idx}: {min(self.eles[start_idx].low, self.eles[start_idx+1].low) > self.eles[end_idx].low}")
                return False
            return True
                
    def split_bi(self):
        eles = [self.eles[i] for i in range(self.candidate_end_point+2)]
        new_eles = [self.eles[i] for i in range(self.candidate_end_point-1, len(self.eles))]
        self.eles.clear()
        for e in eles:
            self.eles.append(e)
        self.high = self.eles[-2].high if self.is_up else self.eles[1].high
        self.low = self.eles[1].low if self.is_up else self.eles[-2].low

        self.is_finished = True
        bi = BI(self.direction.switch())
        for e in new_eles:
            bi.eles.append(e)
        if bi.is_down:
            bi.high = self.high
            bi.low = bi.min
        if bi.is_up:
            bi.low = self.low
            bi.high = bi.max
        bi.candidate_end_point = len(new_eles) - 1
        return bi
    
    @property
    def start_value(self):
        return self.eles[1].high if self.is_down else self.eles[1].low
    
    @property
    def end_value(self):
        return self.eles[-2].high if self.is_up else self.eles[-2].low