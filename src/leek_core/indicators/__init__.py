#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Software: PyCharm

__all__ = ["T", "TR", "ATR", "RSRS", "SAR", "KDJ", "MA", "EMA", "BollBand", "DK", "LLT", "KAMA", "FRAMA", "StochRSI", "RSI", "BBI",
           "ChanKManager", "ChanK", "ChanUnion", "ChanBIManager", "ChanBI", "ChanFeature", "ChanSegment",
           "ChanSegmentManager", "BiFXValidMethod", "ChanFX", "ChanDirection", "ChanZSManager", "ChanZS", "Chan",
           "ChanBSPoint", "ChanFXManager",
           "MACD", "MERGE", "SuperSmoother", "UltimateOscillator", "Reflex", "TrendFlex", "BiasRatio",
           "DeMarker", "TDSequence", "TDTrendLine", "DMI", "ADX", "WMA", "HMA", "IchimokuCloud", "WR", "CCI", "CCIV2", "Divergence",
           "HurstExponent", "Extreme", "ChaikinVolatility", "SuperTrend", "ZScore"]

from.t import T
from .boll import BollBand
from .atr import ATR, TR
from .supertrend import SuperTrend
from .cci import CCI, CCIV2
from .chan.k import ChanKManager, ChanK, ChanUnion
from .chan.seg import ChanFeature, ChanSegment, ChanSegmentManager
from .chan.bi import ChanBI, ChanBIManager
from .chan.bsp import ChanBSPoint
from .chan.enums import BiFXValidMethod, ChanFX, ChanDirection
from .chan.chan import Chan
from .chan.fx import ChanFXManager
from .chan.zs import ChanZSManager, ChanZS
from .de_mark import DeMarker, TDSequence, TDTrendLine
from .dk import DK
from .dm import DMI, ADX
from .dsp import Reflex, TrendFlex
from .ichimoku_cloud import IchimokuCloud
from .kdj import KDJ
from .ma import MA, EMA, LLT, KAMA, FRAMA, SuperSmoother, UltimateOscillator, WMA, HMA
from .macd import MACD, Divergence
from .rsi import StochRSI, RSI
from .rsrs import RSRS
from .sar import SAR
from .t import MERGE
from .hurst import HurstExponent
from .extreme import Extreme
from .wr import WR
from .bias import BiasRatio
from .bbi import BBI
from .chaikin_volatility import ChaikinVolatility
from .zscore import ZScore

MA_TYPE = {
    "MA": MA,
    "EMA": EMA,
    "LLT": LLT,
    "KAMA": KAMA,
    "FRAMA": FRAMA,
    "SSM": SuperSmoother,
    "UO": UltimateOscillator,
    "WMA": WMA,
    "HMA": HMA
}

if __name__ == '__main__':
    pass
