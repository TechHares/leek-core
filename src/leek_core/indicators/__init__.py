#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Software: PyCharm

__all__ = ["TR", "ATR", "RSRS", "SAR", "KDJ", "MA", "EMA", "BollBand", "DK", "LLT", "KAMA", "FRAMA", "StochRSI", "RSI",
           "ChanKManager", "ChanK", "ChanUnion", "ChanBIManager", "ChanBI", "ChanFeature", "ChanSegment",
           "ChanSegmentManager", "BiFXValidMethod", "ChanFX", "ChanDirection", "ChanZSManager", "ChanZS", "Chan",
           "ChanBSPoint", "ChanFXManager", "MACD", "MERGE", "SuperSmoother", "UltimateOscillator", "Reflex", "TrendFlex",
           "DeMarker", "TDSequence", "TDTrendLine", "DMI", "WMA", "HMA", "IchimokuCloud", "WR", "CCI", "CCIV2", "Divergence"]

from indicators.atr import ATR, TR
from leek_core.indicators.boll import BollBand
from indicators.cci import CCI, CCIV2
from indicators.chan.bi import ChanBI, ChanBIManager
from indicators.chan.bsp import ChanBSPoint
from indicators.chan.chan import Chan
from indicators.chan.zs import ChanZSManager, ChanZS
from indicators.de_mark import DeMarker, TDSequence, TDTrendLine
from indicators.dk import DK
from indicators.dm import DMI
from indicators.dsp import Reflex, TrendFlex
from indicators.ichimoku_cloud import IchimokuCloud
from indicators.kdj import KDJ
from indicators.ma import MA, EMA, LLT, KAMA, FRAMA, SuperSmoother, UltimateOscillator, WMA, HMA
from indicators.macd import MACD, Divergence
from indicators.rsi import StochRSI, RSI
from indicators.rsrs import RSRS
from indicators.sar import SAR
from indicators.t import MERGE
from indicators.wr import WR

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
