#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRV 3分钟线数据准备（精简版 ~121 特征）

1. 从 ClickHouse 获取 CRV 最近一年 3分钟线数据
2. 计算 ~121 维特征因子（同 M1 版本）
3. 使用三重屏障法打标（参数按 3 分钟周期调整）
4. 保存为 CSV

用法:
    python prepare_crv_m3_data.py
"""
import os
import time

import numpy as np
import pandas as pd

from leek_core.data import ClickHouseKlineDataSource
from leek_core.models import KLine, TimeFrame, TradeInsType
from leek_core.utils import DateTimeUtils
from leek_core.ml.label import TripleBarrierLabel
from leek_core.ml.factors import (
    RSIFactor,
    AccelerationMomentumFactor,
    SimplifiedDMOMFactor,
    SimpleARFactor,
    PriceSpikeFeaturesFactor,
)
from leek_core.ml.factors.alpha158 import Alpha158Factor

# ============================================================
# 配置
# ============================================================
SYMBOL = "CRV"
QUOTE_CURRENCY = "USDT"
INS_TYPE = TradeInsType.SWAP
TIMEFRAME = TimeFrame.M3               # 3 分钟线
MARKET = "okx"
START_DATE = "2025-02-10"
END_DATE = "2026-02-10"

# 三重屏障参数（按「机会」定义，不按目标分布反推）
# 逻辑：大部分时间空仓，只有明确波动才算做多/做空，其余=不操作
# 3 分钟线单根波动更大，TP/SL 比 M1 稍宽
HOLD_PERIODS = 20       # 20 根 K线 = 60 分钟持仓期
TAKE_PROFIT = 0.015     # 1.5% 止盈（只有明确上涨才算做多机会）
STOP_LOSS = 0.009       # 0.9% 止损（只有明确下跌才算做空机会）

# 不参与特征的列名
NON_FEATURE_COLS = {
    "start_time", "open", "high", "low", "close",
    "volume", "amount", "label", "vwap",
}

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 数据获取
# ============================================================
def fetch_data() -> pd.DataFrame:
    """从 ClickHouse 获取 CRV 3分钟线数据"""
    print(f"连接 ClickHouse...")
    ds = ClickHouseKlineDataSource(password="default")
    ds.connect()

    row_key = KLine.pack_row_key(SYMBOL, QUOTE_CURRENCY, INS_TYPE, TIMEFRAME)
    start_time = DateTimeUtils.to_timestamp(START_DATE)
    end_time = DateTimeUtils.to_timestamp(END_DATE)

    print(f"获取 {SYMBOL}/{QUOTE_CURRENCY} {TIMEFRAME.value} K线数据...")
    print(f"  时间范围: {START_DATE} ~ {END_DATE}")

    klines = ds.get_history_data(
        start_time=start_time,
        end_time=end_time,
        row_key=row_key,
        market=MARKET,
    )

    df = pd.DataFrame([{
        "start_time": x.start_time,
        "open": float(x.open),
        "high": float(x.high),
        "low": float(x.low),
        "close": float(x.close),
        "volume": float(x.volume),
        "amount": float(x.amount),
    } for x in klines])

    ds.disconnect()
    return df


# ============================================================
# 特征计算
# ============================================================
def compute_features(df: pd.DataFrame):
    """
    计算全部 ~121 维特征

    返回:
        df: 带有特征列的 DataFrame
        feature_names: 特征列名列表（有序）
    """
    feature_names = []

    # ================================================================
    # Group A+B: Alpha158 精简 (100 features)
    # KBar(9) + Price(4) + Rolling(29 ops × 3 windows = 87)
    # 窗口 [5,20,60] 去掉 10,30 冗余窗口
    # ================================================================
    print("  [1/7] Alpha158 精简 (100 features)...")
    t = time.time()
    alpha158 = Alpha158Factor(
        include_kbar=True,
        include_price=True,
        include_rolling=True,
        windows="5,20,60",
    )
    alpha158_result = alpha158.compute(df)
    for col in alpha158_result.columns:
        df[col] = alpha158_result[col]
    feature_names.extend(alpha158.factor_names)
    print(f"         完成 ({len(alpha158.factor_names)} features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group C: RSI 多尺度 (2 features)
    # ================================================================
    print("  [2/7] RSI 多尺度 (2 features)...")
    t = time.time()
    for w in [7, 14]:
        rsi = RSIFactor(window=w)
        rsi.compute(df)
        col_name = f"RSI_{w}"
        df[col_name] = df[col_name] / 100.0
        feature_names.append(col_name)
    print(f"         完成 (2 features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group D: 价格尖峰特征 (8 features)
    # ================================================================
    print("  [3/7] PriceSpike 价格尖峰 (8 features)...")
    t = time.time()
    spike = PriceSpikeFeaturesFactor(window=30, spike_window=3)
    spike_result = spike.compute(df)
    for col in spike_result.columns:
        df[col] = spike_result[col]
    feature_names.extend(spike.factor_names)
    print(f"         完成 ({len(spike.factor_names)} features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group E: 加速度动量 (4 features)
    # ================================================================
    print("  [4/7] AccelerationMomentum 加速度动量 (4 features)...")
    t = time.time()
    acc = AccelerationMomentumFactor(window=15)
    acc_result = acc.compute(df)
    for col in acc_result.columns:
        df[col] = acc_result[col]
    feature_names.extend(acc.factor_names)
    print(f"         完成 ({len(acc.factor_names)} features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group F: 简化方向动量 (1 feature)
    # ================================================================
    print("  [5/7] SimplifiedDMOM 方向动量 (1 feature)...")
    t = time.time()
    dmom = SimplifiedDMOMFactor(vol_window=20, max_duration=12)
    dmom_result = dmom.compute(df)
    for col in dmom_result.columns:
        df[col] = dmom_result[col]
    feature_names.extend(dmom.factor_names)
    print(f"         完成 ({len(dmom.factor_names)} features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group G: 简化 AR 价差 (2 features)
    # ================================================================
    print("  [6/7] SimpleAR 价差 (2 features)...")
    t = time.time()
    for w in [20, 60]:
        ar = SimpleARFactor(window=w)
        ar_result = ar.compute(df)
        for col in ar_result.columns:
            df[col] = ar_result[col]
        feature_names.extend([f"SimpleAR_{w}"])
    print(f"         完成 (2 features, {time.time() - t:.1f}s)")

    # ================================================================
    # Group H: 时间周期特征 (4 features)
    # ================================================================
    print("  [7/7] 时间周期特征 (4 features)...")
    t = time.time()
    if "start_time" in df.columns:
        dt = pd.to_datetime(df["start_time"], unit="ms")
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        feature_names.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    print(f"         完成 (4 features, {time.time() - t:.1f}s)")

    return df, feature_names


# ============================================================
# 标签生成
# ============================================================
def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用三重屏障法打标

    标签含义（三分类, side=both）:
        0 = 做空信号
        1 = 不操作（震荡）
        2 = 做多信号
    """
    label_gen = TripleBarrierLabel(
        hold_periods=HOLD_PERIODS,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        side="both",
        num_classes=3,
        conservative=True,
    )
    labels = label_gen.generate(df)
    df["label"] = labels
    return df


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    print("=" * 70)
    print("CRV 3分钟线数据准备（精简版 ~121 特征）")
    print("=" * 70)
    print(f"  品种: {SYMBOL}/{QUOTE_CURRENCY}  周期: {TIMEFRAME.value}")
    print(f"  时间: {START_DATE} ~ {END_DATE}")
    print(f"  打标: TripleBarrier (TP={TAKE_PROFIT:.1%}, SL={STOP_LOSS:.1%}, hold={HOLD_PERIODS})")
    print()

    # 1. 获取数据
    df = fetch_data()
    print(f"  获取到 {len(df)} 条 K 线数据\n")
    if len(df) == 0:
        print("错误: 未获取到数据，请检查 ClickHouse 连接和数据。")
        exit(1)

    # 2. 计算因子
    print(f"正在计算特征因子...")
    t1 = time.time()
    df, feature_names = compute_features(df)
    print(f"\n  因子计算总耗时: {time.time() - t1:.1f}s")
    print(f"  特征总数: {len(feature_names)}")

    # 3. 打标
    print(f"\n正在生成三重屏障标签...")
    t2 = time.time()
    df = generate_labels(df)
    print(f"  打标完成 ({time.time() - t2:.1f}s)")

    # 4. 清理 NaN
    before = len(df)
    df = df.dropna(subset=feature_names + ["label"])
    print(f"\n清理 NaN: {before} -> {len(df)} (丢弃 {before - len(df)} 行)")

    # 5. 替换 inf 为 NaN 再清理
    for col in feature_names:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    before2 = len(df)
    df = df.dropna(subset=feature_names)
    if before2 != len(df):
        print(f"清理 inf: {before2} -> {len(df)} (丢弃 {before2 - len(df)} 行)")

    # 6. 检查标签分布
    print(f"\n标签分布:")
    for label_val, label_name in [(0, "做空"), (1, "不操作"), (2, "做多")]:
        count = (df["label"] == label_val).sum()
        ratio = count / len(df) if len(df) > 0 else 0
        print(f"  {label_name} ({label_val}): {count:>8d} ({ratio:.1%})")

    # 7. 特征统计摘要
    print(f"\n特征统计摘要 (前20个):")
    for i, col in enumerate(feature_names[:20]):
        vals = df[col]
        print(f"  {col:>40s}: mean={vals.mean():>10.4f}  std={vals.std():>10.4f}")
    if len(feature_names) > 20:
        print(f"  ... 还有 {len(feature_names) - 20} 个特征")

    # 8. 保存
    save_cols = ["start_time", "open", "high", "low", "close", "volume", "amount"]
    save_cols += feature_names
    save_cols += ["label"]
    df_save = df[save_cols].copy()

    output_path = os.path.join(OUTPUT_DIR, "crv_m3_prepared.csv")
    df_save.to_csv(output_path, index=False)

    # 同时保存特征名列表
    feature_list_path = os.path.join(OUTPUT_DIR, "crv_m3_feature_names.txt")
    with open(feature_list_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n{'=' * 70}")
    print(f"数据已保存到: {output_path}")
    print(f"特征名列表: {feature_list_path}")
    print(f"  数据量: {len(df_save)} 行 × {len(save_cols)} 列")
    print(f"  特征数: {len(feature_names)}")
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
    print(f"{'=' * 70}")
