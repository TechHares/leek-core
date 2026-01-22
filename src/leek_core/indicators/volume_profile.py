#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : volume_profile.py
# @Software: PyCharm
from collections import deque
import numpy as np
import pandas as pd

from .t import T


class VolumeProfile(T):
    """
    成交量轮廓指标
    
    计算并返回:
    - POC (Point of Control): 成交量最高的价格水平距离百分比
    - is_in_hvn: 当前价格是否在高成交量节点 (1/0)
    - is_in_lvn: 当前价格是否在低成交量节点 (1/0)
    """

    def __init__(self, lookback=90, bins=20, hvn_percentile=0.80, lvn_percentile=0.20, max_cache=100):
        """
        参数:
        - lookback: 回溯窗口期数，默认90
        - bins: 价格区间数量，默认20
        - hvn_percentile: 高成交量节点阈值（百分位数），默认0.80
        - lvn_percentile: 低成交量节点阈值（百分位数），默认0.20
        - max_cache: 最大缓存数量
        """
        T.__init__(self, max_cache)
        self.lookback = lookback
        self.bins = bins
        self.hvn_percentile = hvn_percentile
        self.lvn_percentile = lvn_percentile
        
        # 存储历史数据
        self.history = deque(maxlen=lookback)

    def update(self, data):
        """
        更新指标
        
        返回: (poc_distance_pct, is_in_hvn, is_in_lvn)
        """
        result = (None, 0, 0)
        
        try:
            # 如果数据完成，添加到历史记录
            if data.is_finished:
                self.history.append({
                    'high': float(data.high),
                    'low': float(data.low),
                    'close': float(data.close),
                    'volume': float(data.volume)
                })
            
            # 需要足够的历史数据才能计算
            if len(self.history) < self.lookback:
                return result
            
            # 转换为DataFrame便于计算
            df = pd.DataFrame(list(self.history))
            
            # 1. 创建价格区间
            min_low = df['low'].min()
            max_high = df['high'].max()
            
            if pd.isna(min_low) or min_low == max_high:
                return result
            
            # 创建价格bins
            price_bins = np.linspace(min_low, max_high, self.bins + 1)
            bin_width = (max_high - min_low) / self.bins
            
            # 2. 将每个K线的价格中心分配到一个区间，并汇总成交量
            price_centers = (df['high'] + df['low']) / 2
            binned_prices = pd.cut(price_centers, bins=price_bins, include_lowest=True)
            volume_profile = df.groupby(binned_prices, observed=False)['volume'].sum()
            
            if volume_profile.empty or volume_profile.sum() == 0:
                return result
            
            # 3. 识别关键水平
            # POC: 成交量最高的价格区间
            poc_interval = volume_profile.idxmax()
            poc_price = (poc_interval.left + poc_interval.right) / 2
            
            # 高成交量节点 (HVN)
            hvn_threshold = volume_profile.quantile(self.hvn_percentile)
            hvns = volume_profile[volume_profile >= hvn_threshold].index
            
            # 低成交量节点 (LVN)
            lvn_threshold = volume_profile.quantile(self.lvn_percentile)
            lvns = volume_profile[volume_profile <= lvn_threshold].index
            
            # 4. 计算当前K线的特征
            current_price = float(data.close)
            
            # POC距离百分比
            poc_distance_pct = (current_price - poc_price) / poc_price if poc_price != 0 else 0
            
            # 检查当前价格是否在HVN或LVN中
            is_in_hvn = 0
            is_in_lvn = 0
            
            for interval in hvns:
                if interval.left <= current_price <= interval.right:
                    is_in_hvn = 1
                    break
            
            for interval in lvns:
                if interval.left <= current_price <= interval.right:
                    is_in_lvn = 1
                    break
            
            result = (poc_distance_pct, is_in_hvn, is_in_lvn)
            return result
            
        except Exception as e:
            # 如果计算出错，返回默认值
            return result
        finally:
            if data.is_finished:
                self.cache.append(result)


if __name__ == '__main__':
    pass
