"""
标签生成器模块

提供多种标签生成器，适用于不同的量化策略：

1. FutureReturnLabel - 未来收益率标签（回归任务）
2. DirectionLabel - 方向标签（分类任务）
3. RankLabel - 分位数排名标签（排序任务，多因子打分）
4. EventLabel - 事件驱动标签（高频交易）
5. RiskAdjustedReturnLabel - 风险调整收益率标签（趋势跟踪）
6. ReversalStrengthLabel - 反转强度标签（均值回归）
7. MultiLabelFusion - 多标签融合（机器学习）

推荐使用策略：
- 多因子打分 -> RankLabel
- 趋势跟踪 -> RiskAdjustedReturnLabel
- 均值回归 -> ReversalStrengthLabel
- 高频交易 -> EventLabel
- 机器学习 -> MultiLabelFusion
"""
from .base import LabelGenerator
from .future_return import FutureReturnLabel
from .direction import DirectionLabel
from .rank import RankLabel
from .event import EventLabel
from .risk_adjusted import RiskAdjustedReturnLabel
from .reversal import ReversalStrengthLabel
from .multi_label import MultiLabelFusion

__all__ = [
    "LabelGenerator",
    "FutureReturnLabel",
    "DirectionLabel",
    "RankLabel",
    "EventLabel",
    "RiskAdjustedReturnLabel",
    "ReversalStrengthLabel",
    "MultiLabelFusion",
]
