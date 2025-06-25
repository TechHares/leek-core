#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : fx.py
# @Software: PyCharm
from collections import deque

from .comm import ChanUnion
from .enums import ChanFX


class ChanFXManager:
    """
    分型
    """

    def __init__(self):
        self.__chan_list = deque(maxlen=3)

    def __getitem__(self, index: int) -> ChanUnion:
        return list(self.__chan_list)[index]

    @property
    def high(self):
        assert self.fx.is_top
        return self.point.high

    @property
    def low(self):
        assert self.fx.is_bottom
        return self.point.low

    @property
    def left(self):
        assert len(self.__chan_list) >= 1
        return self[0]

    @property
    def point(self):
        assert len(self.__chan_list) >= 2
        return self[1]

    @property
    def peak_value(self):
        assert self.fx is not None
        return self.high if self.fx.is_top else self.low

    @property
    def lst(self):
        assert len(self.__chan_list) == 3
        return list(self.__chan_list)

    @property
    def right(self):
        assert len(self.__chan_list) == 3
        return self[2]

    def next(self, k: ChanUnion) -> ChanFX:
        if len(self.__chan_list) == 0 or self[-1].idx < k.idx:
            self.__chan_list.append(k)
        else:
            self.__chan_list[-1]= k
        return self.fx

    @property
    def fx(self) -> ChanFX:
        if len(self.__chan_list) < 3:
            return None
        if self.left.high <= self.point.high and self.point.high >= self.right.high:
            return ChanFX.TOP

        if self.left.low >= self.point.low and self.point.low <= self.right.low:
            return ChanFX.BOTTOM

    @property
    def gap(self) -> bool:
        assert self.fx is not None
        return (self.fx.is_top and self.left.high < self.point.low) or (self.fx.is_bottom and self.left.low > self.point.high)

    @property
    def score(self) -> int:
        """
        分型强度打分
        :return:
        """
        """
        第一根K线的高点，被卖分力阻击后，出现回落，这个回落，出现在第一根K线的上影部分或者第二根K线的下影部分，
        而在第二根K线，出现一个更高的高点，但这个高点，显然与第一根K线的高点中出现的买的分力，一定在小级别上出现力度背驰，从而至少制造了第二根K线的上影部分。
        最后，第三根K线，会再次继续一次买的分力的攻击，但这个攻击，完全被卖的分力击败，从而不能成为一个新高点，在小级别上，大致出现一种第二类卖点的走势。
        
        首先，一个完全没有包含关系的分型结构， 意味着市场双方都是直截了当，没有太多犹豫。
        包含关系（只要不是直接把阳线以长阴线吃掉）意味着一种犹豫，一种不确定的观望等，一般在小级别上，都会有中枢延伸、扩展之类的东西
        其次，还是用没有包含关系的顶分型为例子。如果第一K线是一长阳线，而第二、三都是小阴、小阳，那么这个分型结构的意义就不大了，
        在小级别上，一定显现出小级别中枢上移后小级别新中枢的形成，一般来说，这种顶分型，成为真正顶的可能性很小，绝大多数都是中继的。

        但，如果第二根K线是长上影甚至就是直接的长阴，而第三根K线不能以阳线收在第二根K线区间的一半之上，那么该顶分型的力度就比较大，最终要延续成笔的可能性就极大了。
        直接把阳线以长阴线吃掉，是最坏的一种包含关系。
        
        一般来说，非包含关系处理后的顶分型中，第三根K线如果跌破第一根K线的底而且不能高收到第一根K线区间的一半之上，属于最弱的一种，也就是说这顶分型有着较强的杀伤力。
        """
        # 基础一分
        score = 1

        # 三元素很直接 + 分
        if self.left.size == 1:
            score += 1
        if self.point.size == 1:
            score += 1
        if self.right.size == 1:
            score += 1
        # size = sum([self.left.size, self.point.size, self.right.size])
        # size 共4分
        # 一元素和二元素力度减弱
        if self.left.high - self.left.low < self.point.high - self.point.low:
            score += 1
        # 二元素和三素力度加强
        if self.point.high - self.point.low < self.right.high - self.right.low:
            score += 1

        # 破左元素
        if (self.fx.is_top and self.right.low < self.left.low) or (self.fx.is_bottom and self.right.high > self.left.high):
            score += 1
            # 破左元素 30% 以上
            if self.left.high - self.left.low > 0:
                if ((self.fx.is_top and (self.left.low - self.right.low) / (self.left.high - self.left.low) > 0.3 )
                        or (self.fx.is_bottom and (self.right.high - self.left.high) / (self.left.high - self.left.low) > 0.3)):
                    score += 1

        # 二三元素之间交集很少
        cross = min(self.point.high, self.right.high) - max(self.point.low, self.right.low)
        if cross < 0 or (cross / (self.point.high - self.point.low)) < 0.1: # 跳空或者10%以内
            score += 1

        if self.point.size == 1:
            from leek.t.chan.k import ChanK
            if isinstance(self.point, ChanK):
                k = self.point.klines[0]
                if (self.fx.is_top and k.close > k.open) or (self.fx.is_bottom and k.close < k.open): # 阴线
                    score += 1
                body = abs(k.open - k.close)
                if (self.fx.is_top and (k.high - k.close) > body) or (self.fx.is_bottom and (k.close - k.low) > body): # 长引线
                    score += 1

        return 1


if __name__ == '__main__':
    pass
