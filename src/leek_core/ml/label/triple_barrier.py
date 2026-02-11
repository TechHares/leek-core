#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三重屏障标签（Triple Barrier Method）

适用于：短线择时策略、高频交易策略

核心逻辑（双向模式 side="both"）：
1. 先看止损：未来 N 根 K 线内，做多/做空方向是否触发止损
2. 两个方向都止损 → 不操作（行情太乱，两边都亏）
3. 没被止损的方向 → 看是否在 N 根内达到目标收益
4. 达到目标 → 该方向信号；都没达到 → 不操作

使用向量化计算，50 万行数据约 1-2 秒完成打标。

参考：Marcos Lopez de Prado, "Advances in Financial Machine Learning"
"""
import numpy as np
import pandas as pd

from leek_core.models import Field, FieldType

from .base import LabelGenerator


class TripleBarrierLabel(LabelGenerator):
    """
    三重屏障标签

    在未来 hold_periods 根 K 线内，用 high/low 检查三个屏障：

    1. 上屏障（止盈）：价格涨幅触及 take_profit
    2. 下屏障（止损）：价格跌幅触及 stop_loss
    3. 时间屏障（到期）：持仓期内未触发止盈/止损

    关键特点：
    - 先检查止损，再检查止盈（止损优先，模拟真实交易风控）
    - 同一根 K 线内同时触及止盈和止损时，保守模式假设止损先触发
    - 支持做多（long）、做空（short）、双向（both）标注
    - 向量化实现，速度快

    参数:
        hold_periods: 最大持仓周期（K线根数），默认10
        take_profit: 止盈比例（小数形式，如0.02表示2%），默认0.02
        stop_loss: 止损比例（小数形式，如0.01表示1%），默认0.01
        side: 交易方向，"long"=做多，"short"=做空，"both"=双向，默认"both"
        num_classes: 分类数，2=二分类，3=三分类，默认3
        conservative: 同一根K线内止盈止损同时触发时，是否假设止损先触发，默认True

    标签含义:
        三分类 (num_classes=3, side="both"):
            0 = 做空信号（做空达到止盈 且 做多未达到）
            1 = 不操作（两方向都止损 / 都达标 / 都未达标）
            2 = 做多信号（做多达到止盈 且 做空未达到）

        三分类 (num_classes=3, side="long" 或 "short"):
            0 = 止损出局
            1 = 到期（未触发止盈止损）
            2 = 止盈出局

        二分类 (num_classes=2):
            0 = 坏交易/做空
            1 = 好交易/做多

    示例:
        >>> label_gen = TripleBarrierLabel(
        ...     hold_periods=10,
        ...     take_profit=0.02,
        ...     stop_loss=0.01,
        ...     side="both",
        ...     num_classes=3,
        ... )
        >>> labels = label_gen.generate(df)
    """
    display_name = "三重屏障标签"
    init_params = [
        Field(
            name="hold_periods",
            label="最大持仓周期",
            type=FieldType.INT,
            default=10,
            description="最大持仓K线根数",
        ),
        Field(
            name="take_profit",
            label="止盈比例",
            type=FieldType.FLOAT,
            default=0.02,
            description="止盈比例（如0.02表示2%）",
        ),
        Field(
            name="stop_loss",
            label="止损比例",
            type=FieldType.FLOAT,
            default=0.01,
            description="止损比例（如0.01表示1%）",
        ),
        Field(
            name="side",
            label="交易方向",
            type=FieldType.RADIO,
            default="both",
            choices=[("long", "做多"), ("short", "做空"), ("both", "双向")],
            description="交易方向",
        ),
        Field(
            name="num_classes",
            label="分类数",
            type=FieldType.INT,
            default=3,
            description="2=二分类（好/坏交易），3=三分类（止盈/持平/止损）",
        ),
        Field(
            name="conservative",
            label="保守模式",
            type=FieldType.BOOLEAN,
            default=True,
            description="同一根K线内止盈止损同时触发时，假设止损先触发",
        ),
    ]

    def __init__(
        self,
        hold_periods: int = 10,
        take_profit: float = 0.02,
        stop_loss: float = 0.01,
        side: str = "both",
        num_classes: int = 3,
        conservative: bool = True,
    ):
        super().__init__()
        self.hold_periods = hold_periods
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.side = side
        self.num_classes = num_classes
        self.conservative = conservative

    def _find_first_events(self, df: pd.DataFrame, n: int):
        """
        向量化查找未来 hold_periods 根 K 线内，做多/做空止盈止损首次触发的 bar 编号

        对每个样本 i，逐步检查 i+1, i+2, ..., i+hold_periods 的 high/low，
        记录做多/做空的止盈和止损各自第一次被触发的时间点。

        :return: (long_sl_bar, long_tp_bar, short_sl_bar, short_tp_bar, INF)
            bar 值为 1~hold_periods 表示第几根 K 线触发
            bar 值为 INF (hold_periods+1) 表示持仓期内未触发
        """
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64) if 'high' in df.columns else close.copy()
        low = df['low'].values.astype(np.float64) if 'low' in df.columns else close.copy()

        tp = self.take_profit
        sl = self.stop_loss
        hp = self.hold_periods
        INF = hp + 1  # 哨兵值：表示持仓期内未触发

        long_sl_bar = np.full(n, INF, dtype=np.int32)
        long_tp_bar = np.full(n, INF, dtype=np.int32)
        short_sl_bar = np.full(n, INF, dtype=np.int32)
        short_tp_bar = np.full(n, INF, dtype=np.int32)

        for j in range(1, hp + 1):
            end = n - j
            if end <= 0:
                break

            idx = np.arange(end)
            entry = close[idx]
            fut_high = high[idx + j]
            fut_low = low[idx + j]

            # 做多止损：低点跌破 entry * (1 - sl)
            mask = (fut_low <= entry * (1 - sl)) & (long_sl_bar[idx] == INF)
            long_sl_bar[idx[mask]] = j

            # 做多止盈：高点突破 entry * (1 + tp)
            mask = (fut_high >= entry * (1 + tp)) & (long_tp_bar[idx] == INF)
            long_tp_bar[idx[mask]] = j

            # 做空止损：高点突破 entry * (1 + sl)
            mask = (fut_high >= entry * (1 + sl)) & (short_sl_bar[idx] == INF)
            short_sl_bar[idx[mask]] = j

            # 做空止盈：低点跌破 entry * (1 - tp)
            mask = (fut_low <= entry * (1 - tp)) & (short_tp_bar[idx] == INF)
            short_tp_bar[idx[mask]] = j

        return long_sl_bar, long_tp_bar, short_sl_bar, short_tp_bar, INF

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        生成三重屏障标签

        :param df: 包含 close 列（建议包含 high, low）的 DataFrame
        :return: 标签 Series
        """
        n = len(df)

        if self.side == "both":
            labels = self._generate_both(df, n)
        elif self.side == "long":
            labels = self._generate_single_side(df, n, side="long")
        elif self.side == "short":
            labels = self._generate_single_side(df, n, side="short")
        else:
            raise ValueError(f"Unknown side: {self.side}, must be 'long', 'short', or 'both'")

        return pd.Series(labels.astype(int), name=self.label_name)

    def _generate_single_side(self, df: pd.DataFrame, n: int, side: str) -> np.ndarray:
        """
        单方向打标（仅做多 或 仅做空）

        逻辑：
        1. 先查止损是否在止盈之前触发
        2. 止损先触发 → 止损出局
        3. 止盈先触发 → 止盈出局
        4. 都未触发 → 不操作（三分类）/ 按最终收益判断（二分类）

        三分类: 0=止损, 1=不操作, 2=止盈
        二分类: 0=坏交易, 1=好交易
        """
        long_sl, long_tp, short_sl, short_tp, INF = self._find_first_events(df, n)

        if side == "long":
            sl_bar, tp_bar = long_sl, long_tp
        else:
            sl_bar, tp_bar = short_sl, short_tp

        # 止损优先判定
        if self.conservative:
            stopped = sl_bar <= tp_bar  # 同 bar 时止损优先
        else:
            stopped = sl_bar < tp_bar

        # 止盈达标：未被止损 且 止盈在持仓期内触发
        wins = (~stopped) & (tp_bar < INF)

        if self.num_classes == 3:
            labels = np.full(n, 1, dtype=int)  # 默认：不操作
            labels[wins] = 2      # 止盈
            labels[stopped] = 0   # 止损
        else:
            # 二分类
            labels = np.zeros(n, dtype=int)
            labels[wins] = 1      # 好交易（止盈）
            labels[stopped] = 0   # 坏交易（止损）

            # 都未触发 → 按最终收益方向判断
            neutral = ~wins & ~stopped
            if np.any(neutral):
                close = df['close'].values.astype(np.float64)
                end_idx = np.minimum(np.arange(n) + self.hold_periods, n - 1)
                if side == "long":
                    final_ret = (close[end_idx] - close) / (close + 1e-20)
                else:
                    final_ret = (close - close[end_idx]) / (close + 1e-20)
                labels[neutral & (final_ret > 0)] = 1
                # final_ret <= 0 保持默认 0

        return labels

    def _generate_both(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """
        双向打标（向量化实现）

        核心逻辑：
        1. 先看止损：做多/做空方向是否在持仓期内触发止损（且止损先于止盈）
        2. 两个方向都被止损 → 不操作（行情太乱，两边都亏）
        3. 没被止损的方向 → 看是否达到止盈目标
        4. 达到目标 → 该方向信号；都达到/都未达到 → 不操作

        三分类: 0=做空, 1=不操作, 2=做多
        二分类: 0=做空, 1=做多
        """
        long_sl, long_tp, short_sl, short_tp, INF = self._find_first_events(df, n)

        # 判定止损：止损在止盈之前触发（conservative 时同 bar 也算止损优先）
        if self.conservative:
            long_stopped = long_sl <= long_tp
            short_stopped = short_sl <= short_tp
        else:
            long_stopped = long_sl < long_tp
            short_stopped = short_sl < short_tp

        # 达到目标：未被止损 且 止盈在持仓期内触发
        long_wins = (~long_stopped) & (long_tp < INF)
        short_wins = (~short_stopped) & (short_tp < INF)

        if self.num_classes == 3:
            labels = np.full(n, 1, dtype=int)  # 默认：不操作

            # 只做多赢 → 做多信号
            labels[long_wins & ~short_wins] = 2
            # 只做空赢 → 做空信号
            labels[short_wins & ~long_wins] = 0
            # 两个都赢 / 两个都停 / 都没信号 → 保持不操作
        else:
            # 二分类：做多(1) vs 做空(0)
            labels = np.zeros(n, dtype=int)

            # 只做多赢 → 做多
            labels[long_wins & ~short_wins] = 1
            # 只做空赢 → 做空
            labels[short_wins & ~long_wins] = 0

            # 两个都赢 → 看谁的止盈先到
            both_win = long_wins & short_wins
            labels[both_win & (long_tp <= short_tp)] = 1
            labels[both_win & (short_tp < long_tp)] = 0

            # 都没赢 → 按最终收益方向判断
            neither_win = ~long_wins & ~short_wins
            if np.any(neither_win):
                close = df['close'].values.astype(np.float64)
                end_idx = np.minimum(np.arange(n) + self.hold_periods, n - 1)
                final_ret = (close[end_idx] - close) / (close + 1e-20)
                labels[neither_win & (final_ret >= 0)] = 1
                labels[neither_win & (final_ret < 0)] = 0

        return labels
