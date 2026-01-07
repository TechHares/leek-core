#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠中说禅组件
"""
from typing import List

from .base import Component
from .bi import BI


class D(Component):
    """
    缠中说禅 段
    没缺口 分型就结束
    有缺口 无法判断结束， 只能下一段判断被破坏， 然后往回传导

    不使用线段分型判断（因为目前根据个人解读原版的分型划分规则，一个线段结束之后， 后续笔无法继续画的情况， 所以暂时不使用线段分型判断）
    情况如下图所示：
        i点出现之后线段 a-d 结束（bc和de之间无缺口， 有缺口的话根据定义可得a~n是一段）， 从d开始下降段，向下段开始d~k可以合并为 dejk, 随后价格一路新高，再也无法成段
        故在没有新的理解能解决该问题之前， 没有缺口的情况下，使用破坏判断逻辑，即划分为a~d, d~k, k~n 三段
        

    |                                                                                                                         ● n
    |                                                                                                              l         .
    |                                                                                                              ●        .
    |                                                                                                             .   .    .
    |                                                                                                            .      ●         
    |                                         d                                                                 .       m  
    |                                        ●                                                                 .          
    |                                       .  .                   f                                           .             
    |                                      .    .                  ●                                         .               
    |                                     .      .                .  .          h                           .                 
    |                                    .        .              .    .         ●                j         .            
    |                                   .          .             .     .      .   .              ●        .                   
    |                                  .            .           .       .   .       .         .   .      .                         
    |                                 .              .         .         ●            .    .       .    .                    
    |                                .                .       .          g              ●           .  .                      
    |                               .                  .      .                         i            ●                      
    |            b ●               .                    .    .                                       k                        
    |             .   .           .                      .  .                                                                 
    |            .     .         .                        ●                                                             
    |           .        .      .                         e                                                                  
    |          .           .   .                                                                                            
    |         .              ●                                                                                           
    |        .               c                                                                                            
    |       .                                                                                                             
    |      .                                                                                                              
    |     .                                                                                                               
    |    ● a                                                                                                                 
    |--------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self, bi: BI):
        super().__init__()
        self.direction = bi.direction
        self.eles.append(bi)
        self.high = bi.high
        self.low = bi.low

        # 状态信息
        self.peak_idx = 0 # 极点索引
        self.has_gap = False # 是否有缺口
        self.left_feature = None # 左特征
        self.right_feature = None # 右特征
        self.second_peak_idx = 0 # 第二极点索引(有缺口时用)
        self.is_finished = False # 是否结束


    def update(self, bi: BI):
        if bi:
            self.eles.append(bi)
            self.high = max(self.high, bi.high)
            self.low = min(self.low, bi.low)
        if len(self.eles) < 3:
            return

        # 尝试更新极点
        for i in range(self.peak_idx+2, len(self.eles), 2):
            if self.peak_idx == 0:
                if self.is_up and self.eles[i].high > self.eles[i-2].high or \
                    self.is_down and self.eles[i].low < self.eles[i-2].low:
                    self.update_peak(i)
                continue
            if self.is_up and self.eles[i].high > self.eles[self.peak_idx].high or \
                        self.is_down and self.eles[i].low < self.eles[self.peak_idx].low:
                self.update_peak(i)

        # 尝试更新缺口
        if self.left_feature is None or len(self.eles) <= self.peak_idx + 3:
            return
        if self.is_up:
            self.has_gap = self.eles[self.peak_idx+1].low > self.left_feature
        if self.is_down:
            self.has_gap = self.eles[self.peak_idx+1].high < self.left_feature

        # 看看线段是否破坏
        for i in range(self.peak_idx+3, len(self.eles), 2):
            if self.is_up and self.eles[i].low < self.eles[i-2].low:
                self.second_peak_idx = i
            if self.is_down and self.eles[i].high > self.eles[i-2].high:
                self.second_peak_idx = i

        if self.second_peak_idx == 0:
            return

        if not self.has_gap:
            # 没缺口 分型就结束
            return self._split_d()

        # 有缺口 无法判断结束， 只能下一段判断被破坏， 然后往回传导
        # if len(self.eles) <= self.second_peak_idx + 3:
            # return
        for i in range(self.second_peak_idx + 3, len(self.eles), 2):
            if self.is_up and self.eles[i].high > self.eles[i-2].high or \
                        self.is_down and self.eles[i].low < self.eles[i-2].low:
                return self._split_d()


    def update_peak(self, idx: int):
        self.reset()
        self.peak_idx = idx
        self.left_feature = self.eles[0].end_value
        for i in range(0, idx, 2):
            if self.is_up:
                self.left_feature = max(self.left_feature, self.eles[i].end_value)
            else:
                self.left_feature = min(self.left_feature, self.eles[i].end_value)

    def reset(self):
        self.peak_idx = 0 # 极点索引
        self.has_gap = False # 是否有缺口
        self.left_feature = None # 左特征
        self.right_feature = None # 右特征
        self.second_peak_idx = 0 # 第二极点索引(有缺口时用)
        self.is_finished = False # 是否结束

    @property
    def start_value(self):
        return self.eles[0].start_value
    
    @property
    def end_value(self):
        return self.eles[-1].end_value

    def _split_d(self):
        current_d = list(self.eles)[:self.peak_idx+1]
        self.high = max([ele.high for ele in current_d])
        self.low = min([ele.low for ele in current_d])
        eles = list(self.eles)
        self.is_finished = True
        self.eles.clear()
        for ele in current_d:
            self.eles.append(ele)
        if self.has_gap:
            second_d = eles[self.peak_idx+1:self.second_peak_idx+1]
            last_d = eles[self.second_peak_idx+1:]
            return [D._new_d(second_d, is_finished=True), D._new_d(last_d)]

        last_d = eles[self.peak_idx+1:]
        return [D._new_d(last_d)]
    
    @classmethod
    def _new_d(cls, eles: List[BI], is_finished: bool = False):
        d = D(eles[0])
        for ele in eles[1:]:
            d.update(ele)
        d.is_finished = is_finished
        return d
