#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import joblib
from leek_core.indicators import *
from leek_core.models import TimeFrame, KLine
from leek_core.engine import IndicatorView
from leek_core.data import ClickHouseKlineDataSource
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from leek_core.utils import DateTimeUtils
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np

def load_data():
    view = IndicatorView([MA()], symbol="CRV", timeframe=TimeFrame.M15,
                             start_time=DateTimeUtils.to_timestamp("2025-01-01"),
                            #  end_time=DateTimeUtils.to_timestamp("2025-10-15 20:00"), data_source=ClickHouseKlineDataSource(password="default"))
                             end_time=DateTimeUtils.to_timestamp("2025-12-01 20:00"), data_source=ClickHouseKlineDataSource(password="default"))
    view.data_source.connect()
    key = KLine.pack_row_key(view.symbol, view.quote_currency, view.ins_type, view.timeframe)
    data = view.data_source.get_history_data(start_time=view.start_time, end_time=view.end_time,
                                                row_key=key, timeframe=view.timeframe,
                                                market=view.market, quote_currency=view.quote_currency,
                                                ins_type=view.ins_type)
    return pd.DataFrame([{
            "start_time": x.start_time,
            "symbol": x.symbol,  # 保留为字符串（不作为特征）
            "open": float(x.open),
            "high": float(x.high),
            "low": float(x.low),
            "close": float(x.close),
            "volume": float(x.volume),
            "amount": float(x.amount)} for x in data])

def generate_features(df: pd.DataFrame, use_time_features=False):
    """
    生成技术指标特征
    
    Args:
        df: 包含OHLCV数据的DataFrame
        use_time_features: 是否添加时间周期性特征（小时、星期等）
    """
    # 技术指标
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma100"] = df["close"].rolling(window=100).mean()
    
    # 可选：时间周期性特征（如果需要捕捉日内/周内模式）
    if use_time_features and "start_time" in df.columns:
        # 将时间戳转换为datetime
        df["datetime"] = pd.to_datetime(df["start_time"], unit='s')
        df["hour"] = df["datetime"].dt.hour  # 小时 (0-23)
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 星期几 (0=周一, 6=周日)
        df["day_of_month"] = df["datetime"].dt.day  # 月内第几天
        # 使用sin/cos编码周期性特征（使0和23相邻）
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df = df.drop(["datetime"], axis=1)  # 删除临时列
    
    return df

def generate_label(df: pd.DataFrame):
    df["label"] = df["close"].shift(-1) > df["close"]
    return df

def train_model(X_train, y_train):
    model = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=500, objective="binary:logistic")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC（需要处理全为同一类的情况）
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 打印评估结果
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC: {auc:.4f}")
    else:
        print("AUC-ROC: 无法计算（测试集可能只有单一类别）")
    
    print("\n混淆矩阵:")
    print(f"                预测")
    print(f"              下跌  上涨")
    print(f"实际  下跌   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      上涨   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['下跌', '上涨']))
    
    # 特征重要性
    print("\n特征重要性:")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def visualize_results(df_test, y_test, y_pred, y_pred_proba):
    """可视化预测结果"""
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('价格走势与预测', '预测概率分布', '累计收益（假设策略）'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 图1: 价格走势和预测点
    df_plot = df_test.copy()
    df_plot['pred'] = y_pred
    df_plot['pred_proba'] = y_pred_proba
    
    # 绘制收盘价
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['close'], 
                  name='收盘价', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # 标记预测上涨的点（y_pred为1表示上涨）
    up_points = df_plot[df_plot['pred'] == 1]
    fig.add_trace(
        go.Scatter(x=up_points.index, y=up_points['close'],
                  mode='markers', name='预测上涨',
                  marker=dict(color='green', size=6, symbol='triangle-up')),
        row=1, col=1
    )
    
    # 标记预测下跌的点（y_pred为0表示下跌）
    down_points = df_plot[df_plot['pred'] == 0]
    fig.add_trace(
        go.Scatter(x=down_points.index, y=down_points['close'],
                  mode='markers', name='预测下跌',
                  marker=dict(color='red', size=6, symbol='triangle-down')),
        row=1, col=1
    )
    
    # 图2: 预测概率分布
    fig.add_trace(
        go.Histogram(x=y_pred_proba, nbinsx=30, name='预测概率分布',
                    marker_color='lightblue'),
        row=2, col=1
    )
    
    # 图3: 累计收益（简单策略：预测上涨时买入，预测下跌时卖出）
    df_plot['returns'] = df_plot['close'].pct_change()
    # 预测上涨(1)时持有，预测下跌(0)时卖出（不持有）
    df_plot['strategy_returns'] = df_plot['returns'] * df_plot['pred'].shift(1).fillna(0)
    df_plot['cumulative_returns'] = (1 + df_plot['strategy_returns']).cumprod()
    df_plot['buy_hold_returns'] = (1 + df_plot['returns']).cumprod()
    
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['cumulative_returns'],
                  name='策略累计收益', line=dict(color='green')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['buy_hold_returns'],
                  name='买入持有收益', line=dict(color='gray', dash='dash')),
        row=3, col=1
    )
    
    fig.update_layout(height=900, title_text="模型预测结果可视化", showlegend=True)
    fig.update_xaxes(title_text="时间索引", row=3, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="频数", row=2, col=1)
    fig.update_yaxes(title_text="累计收益", row=3, col=1)
    
    fig.write_html("model_evaluation.html")
    print("\n可视化结果已保存到 model_evaluation.html")

if __name__ == "__main__":
    # 加载数据
    print("正在加载数据...")
    df = load_data()
    print(f"数据加载完成，共 {len(df)} 条记录")
    df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)
    
    # 生成特征和标签
    print("正在生成特征和标签...")
    # use_time_features=False: 不使用时间特征（推荐，因为时间戳本身不包含预测信息）
    # 如果需要捕捉日内/周内模式，可以设置为True
    df = generate_features(df, use_time_features=False)
    df = generate_label(df)
    
    # 删除包含NaN的行（由于rolling和shift操作产生）
    df = df.dropna()
    print(f"清理后数据量: {len(df)} 条记录")
    
    # 准备特征和标签
    feature_cols = ["open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "ma50", "ma100"]
    X = df[feature_cols]
    y = df["label"].astype(int)  # 转换为整数类型
    
    # 时间序列数据：按时间顺序划分训练集和测试集（前80%训练，后20%测试）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集中上涨比例: {y_train.mean():.2%}")
    print(f"测试集中上涨比例: {y_test.mean():.2%}")
    
    # 训练模型
    print("\n正在训练模型...")
    model = train_model(X_train, y_train)
    print("模型训练完成")
    
    # 评估模型
    print("\n正在评估模型...")
    eval_results = evaluate_model(model, X_test, y_test)
    
    # 可视化结果
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)