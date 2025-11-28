#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子评价模块

用于封装单次因子评价任务，支持多进程并行执行。
"""
from concurrent.futures import as_completed
from multiprocessing import Manager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib.externals.loky import ProcessPoolExecutor

from leek_core.base import create_component, load_class_from_str
from leek_core.data import DataSource
from leek_core.ml import DualModeFactor, FactorEvaluator
from leek_core.models import KLine, TimeFrame, TradeInsType
from leek_core.utils import get_logger

from .types import FactorEvaluationConfig

logger = get_logger(__name__)


def run_symbol_timeframe_evaluation(config: Dict[str, Any], status_queue, cancel_event=None) -> Optional[Dict[str, Any]]:
    """
    运行单个 symbol×timeframe 的所有因子评价
    
    该函数设计为可序列化，用于多进程并行执行。
    一次性处理一个 symbol×timeframe 组合的所有因子，可以提前计算相关性矩阵。
    
    Args:
        config: 配置字典，包含以下字段：
            - symbol: 交易标的
            - timeframe: 时间框架字符串
            - factor_configs: 因子配置列表，每个元素包含：
                - factor_id: 因子ID
                - factor_class_name: 因子类名
                - factor_params: 因子参数字典
            - data_source_class_name: 数据源类名
            - data_source_config: 数据源配置字典
            - market: 市场名称
            - quote_currency: 计价货币
            - ins_type: 合约类型字符串
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - future_periods: 未来收益期数
            - quantile_count: 分位数数量
            - ic_window_size: IC窗口大小
        status_queue: Manager().Queue() 对象，用于通知任务状态
        cancel_event: Manager().Event() 对象，用于取消任务（如果有任务失败）
    
    Returns:
        包含所有因子评价结果和相关性矩阵的字典，格式：
        {
            'symbol': str,
            'timeframe': str,
            'factor_results': Dict[str, Dict],  # key: factor_id_output_name, value: 评价结果
            'correlation_matrix': Dict[str, Dict],  # 因子相关性矩阵
        }
        如果失败或被取消，返回 None
    """
    # 检查是否有任务已经失败，如果有则直接返回
    if cancel_event is not None and cancel_event.is_set():
        return None
    
    data_source = None
    try:
        # 解析配置
        symbol = config['symbol']
        timeframe_str = config['timeframe']
        factor_configs = config['factor_configs']  # 因子配置列表
        
        # 提取因子ID列表用于通知
        factor_ids = [fc['factor_id'] for fc in factor_configs]
        
        # 通知任务开始执行（在进程中真正开始执行时）
        if status_queue is not None:
            try:
                status_queue.put(('running', symbol, timeframe_str, factor_ids))
            except Exception as e:
                logger.warning(f"Failed to send running status notification: {e}")
        data_source_class = config['data_source_class_name']
        data_source_config = config.get('data_source_config', {})
        market = config.get('market', 'okx')
        quote_currency = config.get('quote_currency', 'USDT')
        ins_type_str = config.get('ins_type')
        start_time = config['start_time']
        end_time = config['end_time']
        future_periods = config.get('future_periods', 1)
        quantile_count = config.get('quantile_count', 5)
        ic_window_size = config.get('ic_window_size', 0)
        
        if not factor_configs:
            raise ValueError(f"No factors configured for {symbol} {timeframe_str}")
        
        # 创建数据源
        datasource_class = load_class_from_str(data_source_class)
        data_source: DataSource = create_component(datasource_class, **data_source_config)
        data_source.on_start()
        
        try:
            # 解析时间框架和合约类型
            timeframe = TimeFrame(timeframe_str)
            ins_type = TradeInsType(ins_type_str) if ins_type_str else TradeInsType.SWAP
            
            # 构建 row_key
            row_key = KLine.pack_row_key(symbol, quote_currency, ins_type, timeframe)
            # 获取K线数据
            klines = list(data_source.get_history_data(
                row_key=row_key,
                start_time=start_time,
                end_time=end_time,
                market=market
            ))
            
            if not klines:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame([{
                'start_time': k.start_time,
                'open': float(k.open) if k.open else np.nan,
                'high': float(k.high) if k.high else np.nan,
                'low': float(k.low) if k.low else np.nan,
                'close': float(k.close) if k.close else np.nan,
                'volume': float(k.volume) if k.volume else np.nan,
                'amount': float(k.amount) if k.amount else np.nan,
            } for k in klines])
            
            if len(df) < future_periods + 1:
                return None
            
            # 计算所有因子的因子值
            factor_output_names_map = {}  # factor_id -> [output_names]
            factor_values = None
            for factor_config in factor_configs:
                factor_id = factor_config['factor_id']
                factor_class_name = factor_config['factor_class_name']
                factor_params = factor_config.get('factor_params', {})
                
                # 创建因子实例并计算因子值
                factor_cls = load_class_from_str(factor_class_name)
                factor_instance: DualModeFactor = create_component(factor_cls, **factor_params)
                
                # 计算因子值
                result = factor_instance.compute(df.copy() if len(factor_configs) > 1 else df)
                factor_values = pd.concat([factor_values, result], axis=1) if factor_values is not None else result
                
                # 获取因子输出名称
                output_names = factor_instance.get_output_names()
                factor_output_names_map[factor_id] = output_names
            if factor_values is None or len(factor_values.columns) == 0:
                return None
            # 创建评价器
            evaluator = FactorEvaluator(
                future_periods=future_periods,
                quantile_count=quantile_count
            )
            
            # 计算未来收益（所有因子共享）
            evaluator._calculate_future_return(df)
            
            # 计算相关性矩阵（在评价之前，只需要因子值数据）
            correlation_matrix = evaluator._calculate_factor_correlation(factor_values)
            
            # 评价每个因子
            factor_results = {}
            df = pd.concat([df, factor_values], axis=1)
            for factor_config in factor_configs:
                factor_id = factor_config['factor_id']
                factor_class_name = factor_config['factor_class_name']
                output_names = factor_output_names_map.get(factor_id, [])
                # 为每个输出评价因子
                for output_name in output_names:
                    result = evaluator.evaluate_factor(df, output_name, ic_window_size if ic_window_size > 0 else None)
                    if result:
                        result['factor_id'] = factor_id
                        result['factor_class_name'] = factor_class_name
                        result['output_name'] = output_name
                        factor_results[output_name] = result
            
            return {
                'symbol': symbol,
                'timeframe': timeframe_str,
                'factor_results': factor_results,
                'correlation_matrix': correlation_matrix.to_dict() if correlation_matrix is not None else {},
            }
        finally:
            # 清理数据源
            try:
                data_source.on_stop()
            except Exception as e:
                ...
    
    except Exception as e:
        logger.error(f"Failed to evaluate factors for {config.get('symbol')} {config.get('timeframe')}: {e}", exc_info=True)
        # 设置取消事件，通知其他任务停止执行
        if cancel_event is not None:
            try:
                cancel_event.set()
                logger.warning(f"Set cancel event due to failure in {config.get('symbol')} {config.get('timeframe')}")
            except Exception:
                ...
        
        # 通知任务失败
        if status_queue is not None:
            try:
                symbol = config.get('symbol', '')
                timeframe_str = config.get('timeframe', '')
                factor_configs = config.get('factor_configs', [])
                factor_ids = [fc['factor_id'] for fc in factor_configs]
                status_queue.put(('failed', symbol, timeframe_str, factor_ids))
            except Exception:
                ...


class FactorEvaluatorExecutor:
    def __init__(self, config: FactorEvaluationConfig, _progress_callback: Optional[Callable[[str, str, List[int], str], None]] = None, _subphase_callback: Optional[Callable[[str, str], None]] = None):
        self.config = config
        self._progress_callback: Optional[Callable[[str, str, List[int], str], None]] = _progress_callback
        self._subphase_callback: Optional[Callable[[str, str], None]] = _subphase_callback  # (phase_name, status)
        self.executor = ProcessPoolExecutor(max_workers=config.max_workers)
        # 创建状态队列和取消事件用于进程间通信（使用 Manager() 可以序列化）
        # 即使没有回调函数也创建队列，保持接口一致性
        self.manager = Manager()
        self.status_queue = self.manager.Queue()
        self.cancel_event = self.manager.Event()  # 用于通知其他任务取消执行

    def _merge_ic_series_by_time(
        self,
        ic_series_list: List[List[float]],
        ic_times_list: List[List[int]],
        symbols: List[str],
        timeframes: List[str]
    ) -> Tuple[List[float], List[int], Dict[str, Dict[str, Any]]]:
        """
        按时间对齐合并IC序列（使用pandas优化）
        
        Args:
            ic_series_list: IC值列表的列表，每个元素是一个symbol×timeframe的IC序列
            ic_times_list: 时间戳列表的列表，每个元素是对应的时间戳序列
            symbols: symbol列表，与ic_series_list一一对应
            timeframes: timeframe列表，与ic_series_list一一对应
            
        Returns:
            (merged_ic_series, merged_ic_times, ic_series_by_st)
            - merged_ic_series: 合并后的IC序列
            - merged_ic_times: 合并后的时间戳序列
            - ic_series_by_st: 按symbol×timeframe保存的IC序列和时间戳
        """
        # 保存每个symbol×timeframe的IC序列和时间戳
        ic_series_by_st = {}
        dfs = []
        
        for ic_series, ic_times, symbol, timeframe in zip(ic_series_list, ic_times_list, symbols, timeframes):
            if len(ic_series) > 0 and len(ic_times) > 0:
                st_key = f"{symbol}_{timeframe}"
                ic_series_by_st[st_key] = {
                    'ic_series': ic_series,
                    'ic_times': ic_times,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                
                # 转换为Series，使用时间戳作为索引
                series = pd.Series(ic_series, index=pd.Index(ic_times, name='time'), name=st_key)
                dfs.append(series)
        
        if not dfs:
            return [], [], ic_series_by_st
        
        # 使用pandas的concat按时间对齐所有序列（outer join）
        # 将所有Series合并到一个DataFrame，每列代表一个symbol×timeframe的IC值
        merged_df = pd.concat(dfs, axis=1, join='outer')
        
        # 计算每行的平均值（忽略NaN值）
        merged_ic_series = merged_df.mean(axis=1, skipna=True)
        
        # 过滤掉全为NaN的行
        valid_mask = ~merged_ic_series.isna()
        merged_ic_series = merged_ic_series[valid_mask].astype(float).tolist()
        merged_ic_times = merged_df.index[valid_mask].astype(int).tolist()
        
        return merged_ic_series, merged_ic_times, ic_series_by_st
    
    def _merge_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """合并所有 symbol×timeframe 的结果，按时间对齐合并IC序列"""
        factor_results_buffer: Dict[str, Dict[str, Any]] = {}
        
        for result in all_results:
            if not result:
                continue
            
            symbol = result.get('symbol', '')
            timeframe = result.get('timeframe', '')
            factor_results = result.get('factor_results', {})
            
            for factor_key, factor_result in factor_results.items():
                if factor_key not in factor_results_buffer:
                    factor_results_buffer[factor_key] = {
                        'ic_series_list': [],
                        'ic_times_list': [],
                        'symbols': [],
                        'timeframes': [],
                        'quantile_returns_list': [],
                        'long_short_return_list': [],
                        'factor_id': factor_result.get('factor_id'),
                        'factor_class_name': factor_result.get('factor_class_name'),
                        'output_name': factor_result.get('output_name'),
                    }
                
                buffer = factor_results_buffer[factor_key]
                buffer['ic_series_list'].append(factor_result.get('ic_series', []))
                buffer['ic_times_list'].append(factor_result.get('ic_times', []))
                buffer['symbols'].append(symbol)
                buffer['timeframes'].append(timeframe)
                buffer['quantile_returns_list'].append(factor_result.get('quantile_returns', {}))
                buffer['long_short_return_list'].append(factor_result.get('long_short_return', 0.0))
        
        # 修复逻辑：按时间对齐合并IC序列
        evaluation_results = {}
        factor_metrics = []
        
        for factor_key, buffer in factor_results_buffer.items():
            # 按时间对齐合并IC序列
            merged_ic_series, merged_ic_times, ic_series_by_st = self._merge_ic_series_by_time(
                buffer['ic_series_list'],
                buffer['ic_times_list'],
                buffer['symbols'],
                buffer['timeframes']
            )
            
            # 从合并后的IC序列计算整体指标
            merged_ic_series_pd = pd.Series(merged_ic_series)
            ic_mean = float(merged_ic_series_pd.mean()) if len(merged_ic_series_pd) > 0 else 0.0
            ic_std = float(merged_ic_series_pd.std()) if len(merged_ic_series_pd) > 0 else 0.0
            
            # 计算IR = IC均值 / IC标准差
            ir = float(ic_mean / ic_std) if ic_std > 0 and not np.isnan(ic_std) else np.nan
            
            # 计算IC胜率
            ic_win_rate = float((merged_ic_series_pd > 0).sum() / len(merged_ic_series_pd)) if len(merged_ic_series_pd) > 0 else 0.0
            
            # 计算IC偏度
            ic_skewness = 0.0
            if len(merged_ic_series_pd) >= 3:
                try:
                    from scipy.stats import skew
                    ic_skewness = float(skew(merged_ic_series_pd))
                except Exception:
                    ic_skewness = 0.0
            
            # 合并分位数收益（取平均值）
            quantile_returns_merged = {}
            if buffer['quantile_returns_list']:
                quantile_dicts = buffer['quantile_returns_list']
                all_keys = set()
                for qd in quantile_dicts:
                    all_keys.update(qd.keys())
                
                for key in all_keys:
                    values = [qd.get(key, 0.0) for qd in quantile_dicts if key in qd]
                    if values:
                        quantile_returns_merged[key] = float(np.mean(values))
            
            # 合并多空收益（取平均值）
            long_short_return = float(np.mean([v for v in buffer['long_short_return_list'] if not np.isnan(v)])) if buffer['long_short_return_list'] else 0.0
            
            factor_id = buffer.get('factor_id')
            output_name = buffer.get('output_name', factor_key.split('_', 1)[1] if '_' in factor_key else factor_key)
            
            # evaluation_results 包含合并后的IC序列和按symbol×timeframe保存的数据
            evaluation_results[factor_key] = {
                'factor_id': factor_id,
                'output_name': output_name,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                'ic_win_rate': ic_win_rate,
                'ic_skewness': ic_skewness,
                'ic_series': merged_ic_series,  # 合并后的IC序列
                'ic_times': merged_ic_times,    # 合并后的时间戳
                'ic_series_by_st': ic_series_by_st,  # 按symbol×timeframe保存
                'quantile_returns': quantile_returns_merged,
                'long_short_return': long_short_return,
            }
            
            # factor_metrics 不包含 ic_series（减少存储大小）
            factor_metrics.append({
                'factor_id': factor_id,
                'output_name': output_name,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                'ic_win_rate': ic_win_rate,
                'ic_skewness': ic_skewness,
                'quantile_returns': quantile_returns_merged,
                'long_short_return': long_short_return,
            })
        
        return evaluation_results, factor_metrics
    
    def _merge_correlation_matrices(self, correlation_matrices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个相关性矩阵（取平均值）"""
        if not correlation_matrices:
            return {}
        
        if len(correlation_matrices) == 1:
            return correlation_matrices[0]
        
        # 收集所有矩阵的键
        all_keys = set()
        for matrix in correlation_matrices:
            if isinstance(matrix, dict):
                all_keys.update(matrix.keys())
        
        # 计算平均值
        merged = {}
        for key in all_keys:
            values = []
            for matrix in correlation_matrices:
                if isinstance(matrix, dict) and key in matrix:
                    row = matrix[key]
                    if isinstance(row, dict):
                        values.append(row)
            
            if values:
                # 合并行（取平均值）
                row_keys = set()
                for row in values:
                    row_keys.update(row.keys())
                
                merged_row = {}
                for row_key in row_keys:
                    row_values = [row.get(row_key, 0.0) for row in values if row_key in row]
                    if row_values:
                        merged_row[row_key] = float(np.mean(row_values))
                
                merged[key] = merged_row
        
        return merged
    
    def _generate_summary(self, factor_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总指标"""
        if not factor_metrics:
            return {
                'ic_mean': 0.0,
                'ir': 0.0,
                'ic_win_rate': 0.0,
                'factor_count': 0,
            }
        
        ic_means = [m['ic_mean'] for m in factor_metrics if not np.isnan(m.get('ic_mean', np.nan))]
        irs = [m['ir'] for m in factor_metrics if not np.isnan(m.get('ir', np.nan))]
        ic_win_rates = [m['ic_win_rate'] for m in factor_metrics]
        
        summary = {
            'ic_mean': float(np.mean(ic_means)) if ic_means else 0.0,
            'ir': float(np.mean(irs)) if irs else 0.0,
            'ic_win_rate': float(np.mean(ic_win_rates)) if ic_win_rates else 0.0,
            'factor_count': len(factor_metrics),
        }
        
        return summary
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行因子评价并返回最终结果
        
        Returns:
            包含 summary, metrics, charts 的字典
        """
        import threading
        status_thread = None
        stop_event = None
        
        try:
            futures = []
            correlation_matrices = []
            
            # 按 symbol×timeframe 分组提交任务
            future_info_map = {}  # future -> (symbol, timeframe, factor_ids)
            for symbol in self.config.symbols:
                for timeframe_str in self.config.timeframes:
                    if not self.config.factor_classes or not self.config.factor_params:
                        logger.warning("No factors configured")
                        continue
                    
                    # 构建该 symbol×timeframe 的所有因子配置
                    factor_configs = []
                    factor_ids = []
                    for factor_id, factor_class_name in self.config.factor_classes.items():
                        factor_params = self.config.factor_params.get(factor_id, {})
                        factor_configs.append({
                            'factor_id': factor_id,
                            'factor_class_name': factor_class_name,
                            'factor_params': factor_params or {},
                        })
                        factor_ids.append(factor_id)
                    
                    task_config = {
                        'symbol': symbol,
                        'timeframe': timeframe_str,
                        'factor_configs': factor_configs,
                            'data_source_class_name': self.config.data_source_class,
                            'data_source_config': self.config.data_source_config or {},
                            'market': self.config.market,
                            'quote_currency': self.config.quote_currency,
                            'ins_type': self.config.ins_type,
                            'start_time': self.config.start_time,
                            'end_time': self.config.end_time,
                            'future_periods': self.config.future_periods,
                            'quantile_count': self.config.quantile_count,
                            'ic_window_size': self.config.ic_window if self.config.ic_window is not None else 0,
                        }
                    # 使用带状态队列和取消事件的包装函数
                    future = self.executor.submit(run_symbol_timeframe_evaluation, task_config, self.status_queue, self.cancel_event)
                    futures.append(future)
                    future_info_map[future] = (symbol, timeframe_str, factor_ids)
            
            logger.info(f"Starting {len(futures)} symbol×timeframe tasks")
            
            # 启动状态监听线程（如果有回调函数）
            if self._progress_callback:
                stop_event = threading.Event()
                
                def status_listener():
                    while not stop_event.is_set():
                        try:
                            # 设置超时，避免无限等待
                            status_info = self.status_queue.get(timeout=1.0)
                            if status_info is None:  # 结束信号
                                break
                            status, symbol, timeframe_str, factor_ids = status_info
                            self._progress_callback(symbol, timeframe_str, factor_ids, status)
                        except Exception:
                            # 超时或其他异常，继续监听
                            continue
                
                status_thread = threading.Thread(target=status_listener, daemon=True)
                status_thread.start()
            
            # 收集所有结果
            all_results = []
            correlation_matrices = []
            total_results = 0
            for future in as_completed(futures):
                symbol, timeframe_str, factor_ids = future_info_map.get(future, (None, None, []))
                try:
                    result = future.result()
                    if not result:
                        self._progress_callback(symbol, timeframe_str, factor_ids, 'NoResult')
                        continue
                    all_results.append(result)
                    correlation_matrices.append(result.get('correlation_matrix', {}))
                    total_results += 1
                    
                    # 通知任务完成（通过回调，不通过队列，因为已经在主进程）
                    if self._progress_callback:
                        self._progress_callback(symbol, timeframe_str, factor_ids, 'completed')
                except Exception:
                    if self._progress_callback:
                        self._progress_callback(symbol, timeframe_str, factor_ids, 'failed')
                    raise
            
            # 合并结果
            if self._subphase_callback:
                self._subphase_callback('data_merge', 'running')
            evaluation_results, factor_metrics = self._merge_results(all_results)
            if self._subphase_callback:
                self._subphase_callback('data_merge', 'completed')
            
            # 合并相关性矩阵
            if self._subphase_callback:
                self._subphase_callback('correlation', 'running')
            correlation_matrix = self._merge_correlation_matrices(correlation_matrices)
            if self._subphase_callback:
                self._subphase_callback('correlation', 'completed')
            
            # 生成汇总指标
            if self._subphase_callback:
                self._subphase_callback('summary', 'running')
            summary = self._generate_summary(factor_metrics)
            if self._subphase_callback:
                self._subphase_callback('summary', 'completed')
            
            # 返回结果（charts 的生成需要压缩工具，在 Service 层完成）
            return {
                'summary': summary,
                'metrics': factor_metrics,
                'evaluation_results': evaluation_results,  # 包含 ic_series，用于生成 charts
                'correlation_matrix': correlation_matrix,
            }
        finally:
            # 停止状态监听线程
            if stop_event:
                stop_event.set()
            if status_thread and status_thread.is_alive():
                # 等待线程结束（最多等待2秒）
                status_thread.join(timeout=2.0)
            # 关闭状态队列和 Manager
            if self.status_queue:
                try:
                    self.status_queue.put(None)  # 发送结束信号
                except Exception:
                    pass
            if self.manager:
                try:
                    self.manager.shutdown()
                except Exception:
                    pass
            self.executor.shutdown(wait=False, kill_workers=True)
    
