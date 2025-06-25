#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : chan.py
# @Software: PyCharm
from .bi import ChanBIManager
from .dr import ChanDRManager
from .seg import ChanSegmentManager
from .zs import ChanZSManager
from ..t import T


class Chan(T):
    def __init__(self, bi_zs=True, seg=True, seg_zs=True, dr=True, dr_zs=True, exclude_equal=False, zs_max_level=2, allow_similar_zs=True):
        """
        缠论：递归计算一次
        :param bi_zs: 计算笔中枢
        :param seg: 计算线段
        :param seg_zs: 计算线段中枢
        :param dr: 计算走势
        :param dr_zs: 计算走势中枢
        :param exclude_equal: 合并时是否排除相等的元素
        :param zs_max_level: 最大中枢级别
        :param allow_similar_zs: 是否允许类中枢
        """
        super().__init__()
        self.bi_zs = bi_zs
        self.seg = seg
        self.seg_zs = seg_zs
        self.dr = dr
        self.dr_zs = dr_zs

        self.bi_manager = ChanBIManager(exclude_equal)
        if seg:
            self.seg_manager = ChanSegmentManager()
        if bi_zs:
            self.bizs_manager = ChanZSManager(max_level=zs_max_level, allow_similar_zs=True)
        if seg_zs:
            self.zs_manager = ChanZSManager(max_level=zs_max_level, allow_similar_zs=allow_similar_zs)
        if dr:
            self._zs_manager = ChanZSManager(max_level=zs_max_level, allow_similar_zs=True)
            self.dr_manager = ChanDRManager()
        if dr_zs:
            self.drzs_manager = ChanZSManager(max_level=zs_max_level, allow_similar_zs=allow_similar_zs)

        self.tmp_zs = None
        self.tmp_dr = None
        self.tmp_bi = None

    def update(self, data):
        bi = self.bi_manager.update(data)
        if bi is None:
            return
        if self.bi_zs:
            self.bizs_manager.update(bi)
        if not self.seg:
            return
        seg = self.seg_manager.update_bi(bi)
        if seg:
            if not self.seg_zs:
                return
            self.zs_manager.update(seg)
            self._zs_manager.update(seg)
            if not self.dr:
                return
            for zs in self._zs_manager.zs_list:
                if self.tmp_zs is None or zs.idx >= self.tmp_zs.idx:
                    self.dr_manager.update(zs)
                    self.tmp_zs = zs
            if self._zs_manager.cur_zs:
                self.dr_manager.update(self._zs_manager.cur_zs)
                self.tmp_zs = self._zs_manager.cur_zs
                if not self.dr_zs:
                    return
                for dr in self.dr_manager.dr_list:
                    if self.tmp_dr is None or dr.idx >= self.tmp_dr.idx:
                        self.drzs_manager.update(dr)
                        self.tmp_dr = dr

    @property
    def lower_near_zs(self):
        """
        :return: 次级别最近的中枢
        """
        if self.bi_zs:
            return self.bizs_manager.cur_zs

    @property
    def current_near_zs(self):
        """
        :return: 当前级别最近的中枢
        """
        if self.seg_zs:
            return self.zs_manager.cur_zs

    @property
    def higher_near_zs(self):
        """
        :return: 高级别最近的中枢
        """
        if self.bi_zs:
            return self.drzs_manager.cur_zs


    def mark_on_data(self):
        for bi in self.bi_manager:
            bi.mark_on_data()
            for ck in bi.chan_k_list:
                ck.mark_on_data()

        if self.seg:
            for seg in self.seg_manager:
                seg.mark_on_data()


        if self.seg_zs:
            for zs in self.zs_manager.zs_list:
                zs.mark_on_data()
        if self.dr:
            for dr in self.dr_manager.dr_list:
                dr.mark_on_data()
        if self.dr_zs:
            for zs in self.drzs_manager.zs_list:
                zs.mark_on_data()


if __name__ == '__main__':
    pass
