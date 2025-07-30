#!/usr/bin/env python
# -*- coding: utf-8 -*-
from decimal import Decimal

from leek_core.indicators import DMI
from leek_core.strategy import CTAStrategy, StrategyCommand
from leek_core.utils import get_logger
from leek_core.models import Field, FieldType, KLine, PositionSide
logger = get_logger(__name__)


class DMIStrategy(CTAStrategy):
    
    """
    DMI (Directional Movement Index) 趋势跟踪策略(示例策略)
    
    策略说明:
        该策略基于DMI指标进行趋势跟踪交易，通过ADX指标确认趋势强度，DI指标判断趋势方向。
        策略包含趋势确认、趋势反转和趋势退出三个主要部分。
    
    参数说明:
        adx_smoothing: ADX指标的平滑周期，用于计算ADX值
        di_length: DI指标的周期，用于计算+DI和-DI值
        adx_threshold: ADX趋势确认阈值，当ADX超过此值时认为趋势成立
        adx_peak_threshold: ADX峰值阈值，用于判断趋势强度达到极限
        adx_fallback_threshold: ADX回撤阈值，用于判断趋势是否开始减弱
    
    交易逻辑:
        开仓条件:
            1. ADX突破阈值，确认趋势成立
            2. +DI和-DI交叉，确认趋势方向
            3. 多头：+DI > -DI 且 -DI开始下降
            4. 空头：+DI < -DI 且 +DI开始下降
        
        平仓条件:
            1. 趋势反转：DI指标交叉且ADX继续上升
            2. 趋势减弱：ADX从峰值回落
            3. 趋势结束：ADX跌破阈值
    
    风险控制:
        1. 使用ADX阈值过滤弱趋势
        2. 通过ADX峰值阈值控制持仓时间
        3. 通过ADX回撤阈值及时止盈
    """
    display_name: str = "示例(DMI)"
    init_params = [
        Field(name="adx_smoothing", label="adx平滑(DMI)", type=FieldType.INT, default=6, required=True),
        Field(name="di_length", label="di窗口(DMI)", type=FieldType.INT, default=14, required=True),
        Field(name="adx_threshold", label="adx阈值", type=FieldType.INT, default=25, required=True),
        Field(name="adx_peak_threshold", label="adx峰值阈值", type=FieldType.INT, default=70, required=True),
        Field(name="adx_fallback_threshold", label="adx回撤阈值", type=FieldType.INT, default=12, required=True),
    ]

    def __init__(self, adx_threshold=25, adx_peak_threshold=70, adx_fallback_threshold=12, adx_smoothing=6, di_length=14):
        """
        初始化DMI策略
        
        Args:
            adx_threshold: ADX趋势确认阈值
            adx_peak_threshold: ADX峰值阈值
            adx_fallback_threshold: ADX回撤阈值
            adx_smoothing: ADX平滑周期
            di_length: DI计算周期
        """
        super().__init__()
        self.adx_threshold = adx_threshold  # adx 趋势确认
        self.adx_peak_threshold = adx_peak_threshold  # adx 极限反转阈值
        self.adx_fallback_threshold = adx_fallback_threshold  # adx高点回撤

        self.dmi = DMI(adx_smoothing=adx_smoothing, di_length=di_length)
        self.symbol = None
        self.pre_adx = None
        self.pre_up_di = None
        self.pre_down_di = None
        self.pre_adxr = None
        self.pre_k = None
        self.k = None
        self.high_adx = None

    def on_kline(self, kline: KLine):
        """
        处理K线数据，更新DMI指标
        
        Args:
            kline: 当前K线数据
        """
        self.k = kline
        last = self.dmi.last(1)
        if len(last) > 0:
            self.pre_adx = last[0][0]
            self.pre_up_di = last[0][1]
            self.pre_down_di = last[0][2]
            self.pre_adxr = last[0][3]
        adx, up_di, down_di, adxr = self.dmi.update(kline)
        if adx is None or up_di is None or down_di is None:
            return

        # 赋值画图
        kline.adx, kline.up_di, kline.down_di, kline.adxr = adx, up_di, down_di, adxr
        if self.pre_adx is None:
            self.pre_adx = adx
            self.pre_up_di = up_di
            self.pre_down_di = down_di
            self.pre_adxr = adxr
        if kline.is_finished:
            self.pre_k = kline

    def should_open(self) -> PositionSide | StrategyCommand:
        """
        判断是否应该开仓
        
        Returns:
            PositionSide: 开仓方向，多头或空头
            None: 不开仓
        """
        # 检查必要的数据是否存在
        if (self.pre_k is None or self.k is None or 
            self.k.adx is None or self.k.adxr is None or 
            self.pre_adxr is None or self.k.up_di is None or 
            self.k.down_di is None or self.pre_up_di is None or 
            self.pre_down_di is None):
            return None
            
        if self.high_adx:
            self.high_adx = max(self.high_adx, self.k.adx)
        adx_last = [x[0] for x in self.dmi.last(10)]
        if not self.k.is_finished:
            adx_last.append(self.k.adx)
        adx_cross = self.pre_adxr < self.adx_threshold < self.k.adxr < self.k.adx
        logger.debug(f"DMI,{adx_cross} finish:{self.k.is_finished}, close:{self.k.close}, adxr:{self.k.adxr} {self.pre_adxr}"
                     f"pdi:{self.k.up_di} {self.pre_up_di}, mdi:{self.k.down_di} {self.pre_down_di}, adx:{self.k.adx} {self.pre_adx},"
                     f" rsi:{self.k.rsi_k} {self.k.rsi_d}")

        if not adx_cross:
            return None
        self.high_adx = self.k.adx
        logger.debug(f"CROSS, 多头条件:{self.k.up_di > self.k.down_di} and {self.k.down_di <= self.pre_down_di}, "
                     f"空头条件:{self.k.up_di < self.k.down_di} and {self.k.up_di <= self.pre_up_di}")
        if self.k.up_di > self.k.down_di and self.k.down_di <= self.pre_down_di: # 多头
            logger.info(f"DMI多头开仓, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
            return PositionSide.LONG

        if self.k.up_di < self.k.down_di and self.k.up_di <= self.pre_up_di :  # 空头
            logger.info(f"DMI空头开仓, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
            return PositionSide.SHORT

    def should_close(self, position_side: PositionSide) -> bool | Decimal:
        """
        判断是否应该平仓
        
        Args:
            position_side: 当前持仓方向
            
        Returns:
            bool: True表示应该平仓，False表示继续持仓
            Decimal: 平仓价格（可选）
        """
        # 检查必要的数据是否存在
        if (self.k is None or self.pre_k is None or 
            self.k.up_di is None or self.k.down_di is None or 
            self.k.adx is None or self.pre_adx is None or
            self.k.adxr is None or self.pre_adxr is None):
            return False
            
        # 退出条件
        if position_side.is_long and self.k and self.pre_k:
            if self.k.up_di < self.k.down_di and self.k.adx > self.adx_threshold and self.k.adx > self.pre_adx:  # 趋势反转
                logger.info(f"DMI多头趋势反转平仓, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
                return True
        else:
            if self.k.up_di > self.k.down_di and self.k.adx > self.adx_threshold and self.k.adx > self.pre_adx:  # 趋势反转
                logger.info(f"DMI空头趋势反转平仓, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
                return True

        # adx 极限退出
        if self.adx_peak_threshold < self.k.adx < self.pre_adx:
            logger.info(f"DMI极限阈值止盈, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
            return True
        if self.k.adx < self.k.adxr < self.pre_adxr and self.k.adx < self.pre_adx:
            logger.info(f"DMI回撤止盈, 趋势成立, adx:{self.k.adx}, up_di:{self.k.up_di}, down_di:{self.k.down_di}")
            return True


if __name__ == '__main__':
    pass
