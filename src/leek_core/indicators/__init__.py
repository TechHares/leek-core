#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Software: PyCharm

# __all__ = ["TR", "ATR", "RSRS", "SAR", "KDJ", "MA", "EMA", "BollBand", "DK", "LLT", "KAMA", "FRAMA", "StochRSI", "RSI",
#            "ChanKManager", "ChanK", "ChanUnion", "ChanBIManager", "ChanBI", "ChanFeature", "ChanSegment",
#            "ChanSegmentManager", "BiFXValidMethod", "ChanFX", "ChanDirection", "ChanZSManager", "ChanZS", "Chan",
#            "ChanBSPoint", "ChanFXManager", "MACD", "MERGE", "SuperSmoother", "UltimateOscillator", "Reflex", "TrendFlex",
#            "DeMarker", "TDSequence", "TDTrendLine", "DMI", "WMA", "HMA", "IchimokuCloud", "WR", "CCI", "CCIV2", "Divergence"]
#
# from leek_core.indicators.boll import BollBand
# from .atr import ATR, TR
# from .cci import CCI, CCIV2
# from .chan.bi import ChanBI, ChanBIManager
# from .chan.bsp import ChanBSPoint
# from .chan.chan import Chan
# from .chan.zs import ChanZSManager, ChanZS
# from .de_mark import DeMarker, TDSequence, TDTrendLine
# from .dk import DK
# from .dm import DMI
# from .dsp import Reflex, TrendFlex
# from .ichimoku_cloud import IchimokuCloud
# from .kdj import KDJ
# from .ma import MA, EMA, LLT, KAMA, FRAMA, SuperSmoother, UltimateOscillator, WMA, HMA
# from .macd import MACD, Divergence
# from .rsi import StochRSI, RSI
# from .rsrs import RSRS
# from .sar import SAR
# from .t import MERGE
# from .wr import WR
__all__ = ["TR", "ATR", "DMI", "MA", "EMA", "LLT", "KAMA", "FRAMA", "SuperSmoother", "UltimateOscillator", "WMA", "HMA"]

from .atr import ATR, TR
from .dm import DMI
from .ma import MA, EMA, LLT, KAMA, FRAMA, SuperSmoother, UltimateOscillator, WMA, HMA
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
