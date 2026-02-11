#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Queue
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import numpy as np
import pandas as pd

from leek_core.base import LeekComponent, create_component, load_class_from_str
from leek_core.data import DataSource
from leek_core.models import KLine, TimeFrame, TradeInsType
from leek_core.utils import DateTimeUtils, get_logger

from .evaluator import ModelEvaluator
from .feature_engine import FeatureEngine
from .label.base import LabelGenerator
from .trainer.base import BaseTrainer

logger = get_logger(__name__)


def training(config: Dict[str, Any], queue: Queue = None):
    """
    训练函数，支持进程独立运行和多线程并发处理
    
    {
        "id": 1,
        "name": "test2",
        "symbols": ["CRV", "FIL", "PENGU", "INJ", "ATOM", "ALGO"], 
        "timeframes": ["1m"], 
        "start_time": "2025-11-25", 
        "end_time": "2025-12-02", 
        "datasource_class": "model|KLineDatasource",
        "datasource_config": {},
        "datasource_extra": {
            "quote_currency": "USDT",
            "ins_type": "SWAP",
            "market": "okx"
        },
        "factors": [
            {
                "id": 1,
                "name": "factor1",
                "class_name": "model|Factor",
                "params": {}
            }
        ],
        "label_generator": {
            "id": 1,
            "name": "label_generator1",
            "class_name": "model|LabelGenerator",
            "params": {}
        },
        "trainer": {
            "id": 1,
            "name": "trainer1",
            "class_name": "model|Trainer",
            "params": {}
        },
        "mount_dirs": ["/path/to/mount/dir"],
        "load_model_path": "/path/to/model/dir",
        "save_model_path": "/path/to/model/dir",
        "chunk_size": 100000,  # 可选，分批处理大小
        "train_split_ratio": 0.8,
    }
    """
    engine = None
    try:
        # 发送开始消息
        if queue is not None:
            queue.put({
                'phase': 'initializing',
                'status': 'running'
            })
        
        # 加载 mount_dirs
        mount_dirs = config.get('mount_dirs', [])
        for path in mount_dirs:
            if path and path not in sys.path:
                sys.path.insert(0, path)
        
        # 创建组件
        datasource_cls = load_class_from_str(config['datasource_class'])
        datasource = create_component(datasource_cls, **(config.get('datasource_config', {})))
        
        # 创建特征引擎
        factors = config.get('factors', [])
        factor_instances = []
        factor_ids = []
        for factor_config in factors:
            factor_cls = load_class_from_str(factor_config['class_name'])
            factor_instance = create_component(factor_cls, **(factor_config.get('params', {})))
            factor_instances.append(factor_instance)
            factor_ids.append(str(factor_config.get('id', len(factor_instances) - 1)))
        
        enable_symbol_timeframe_encoding = config.get('enable_symbol_timeframe_encoding', True)
        feature_engine = FeatureEngine(
            factor_instances, 
            factor_ids=factor_ids,
            enable_symbol_timeframe_encoding=enable_symbol_timeframe_encoding
        )
        
        # 创建标签生成器
        label_gen_config = config.get('label_generator', {})
        label_gen_cls = load_class_from_str(label_gen_config['class_name'])
        label_generator = create_component(label_gen_cls, **(label_gen_config.get('params', {})))
        
        # 创建训练器
        trainer_config = config.get('trainer', {})
        trainer_cls = load_class_from_str(trainer_config['class_name'])
        trainer = create_component(trainer_cls, **(trainer_config.get('params', {})))
        
        # 创建评估器（从训练器获取任务类型）
        task_type = getattr(trainer, 'task_type', 'classification')
        evaluator = ModelEvaluator(task_type=task_type)
        
        def _call_back(progress_info):
            if queue is not None:
                queue.put(progress_info)
        
        # 构建 datasource_extra 配置
        datasource_extra = {
            'symbols': config.get('symbols', []),
            'timeframes': config.get('timeframes', []),
            'start_time': config.get('start_time'),
            'end_time': config.get('end_time'),
            'market': config.get('market', 'okx'),
            'quote_currency': config.get('quote_currency', 'USDT'),
            'ins_type': config.get('ins_type', 'SWAP'),
        }
        
        # 创建训练引擎
        engine = TrainingEngine(
            datasource=datasource,
            datasource_extra=datasource_extra,
            feature_engine=feature_engine,
            label_generator=label_generator,
            trainer=trainer,
            evaluator=evaluator,
            train_split_ratio=config.get('train_split_ratio', 0.8),
            load_model_path=config.get('load_model_path'),
            save_model_path=config.get('save_model_path'),
            _call_back=_call_back
        )
        
        engine.on_start()
        # 执行训练
        result = engine.train()
        
        # 发送完成消息（包含结果）
        if queue is not None:
            queue.put({
                'phase': 'completed',
                'status': 'completed',
                'result': result
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if queue is not None:
            queue.put({
                'phase': 'failed',
                'status': 'failed',
                'error': str(e)
            })
        raise
    finally:
        if engine is not None:
            engine.on_stop()


class TrainingEngine(LeekComponent):
    """
    训练引擎
    
    训练流程：
    1. 加载数据（分批加载，保留 symbol/timeframe）
    2. 计算特征：symbol X timeframe 的数据
    3. 生成标签：symbol X timeframe 的数据
    4. 切分数据：symbol X timeframe 每个数据集按 train_split_ratio 切分
    5. 加载旧模型（如果存在）
    6. 评估旧模型（如果存在）
    7. 混合所有数据
    8. 训练模型
    9. 评估模型
    10. 保存模型
    """
    
    def __init__(
        self, 
        datasource: DataSource,
        datasource_extra: Dict[str, Any],
        feature_engine: FeatureEngine, 
        label_generator: LabelGenerator, 
        trainer: BaseTrainer, 
        evaluator: ModelEvaluator, 
        train_split_ratio: float = 0.8,
        load_model_path: Optional[str] = None,
        save_model_path: Optional[str] = None,
        _call_back: Callable[[Any], None] = None,
    ):
        self.datasource = datasource
        self.datasource_extra = datasource_extra
        self.feature_engine = feature_engine
        self.label_generator = label_generator
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_split_ratio = train_split_ratio
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self._call_back = _call_back or (lambda x: None)
        
        # 存储分组数据
        self.processed_data_by_group: Dict[str, pd.DataFrame] = {}
    
    def _send_progress(
        self, 
        phase: str, 
        status: str,
        symbol: Optional[str] = None, 
        timeframe: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """
        发送进度更新
        
        :param phase: 阶段名称 ('loading_data', 'computing_features', 'generating_labels', 'splitting_data', etc.)
        :param status: 状态 ('running', 'completed', 'failed')
        :param symbol: 可选的标的符号（用于子任务）
        :param timeframe: 可选的时间周期（用于子任务）
        :param error_message: 可选的错误消息（当 status 为 'failed' 时使用）
        """
        progress_info = {
            'phase': phase,
            'status': status
        }
        
        if symbol is not None:
            progress_info['symbol'] = symbol
        if timeframe is not None:
            progress_info['timeframe'] = timeframe
        if error_message is not None:
            progress_info['error_message'] = error_message
        
        self._call_back(progress_info)
    
    def on_start(self):
        self.datasource.on_start()
    
    def on_stop(self):
        self.datasource.on_stop()
    
    def _load_data(self) -> Generator[Tuple[str, str, pd.DataFrame], None, None]:
        """
        并发加载数据，保留 symbol 和 timeframe 列
        
        返回: Generator[Tuple[str, str, pd.DataFrame], None, None]
        """
        symbols = self.datasource_extra.get('symbols', [])
        timeframes = self.datasource_extra.get('timeframes', [])
        start_time = DateTimeUtils.to_timestamp(self.datasource_extra.get('start_time'))
        end_time = DateTimeUtils.to_timestamp(self.datasource_extra.get('end_time'))
        market = self.datasource_extra.get('market', 'okx')
        quote_currency = self.datasource_extra.get('quote_currency', 'USDT')
        ins_type_str = self.datasource_extra.get('ins_type', 'SWAP')
        ins_type = TradeInsType(ins_type_str) if isinstance(ins_type_str, str) else ins_type_str
        
        def load_single_group(symbol: str, timeframe_str: str) -> pd.DataFrame:
            """加载单个 symbol×timeframe 的数据"""
            # 每个线程创建独立的数据源实例
            timeframe = TimeFrame(timeframe_str)
            row_key = KLine.pack_row_key(symbol, quote_currency, ins_type, timeframe)
            self._send_progress('loading_data', 'running', symbol=symbol, timeframe=timeframe_str)
            # 获取K线数据
            klines = list(self.datasource.get_history_data(
                row_key=row_key,
                start_time=start_time,
                end_time=end_time,
                market=market
            ))
            
            if not klines:
                logger.warning(f"No data found for {symbol}_{timeframe_str}")
                return None
            
            # 转换为DataFrame格式
            all_data = []
            for k in klines:
                all_data.append({
                    'symbol': symbol,
                    'timeframe': timeframe_str,
                    'start_time': k.start_time,
                    'open': float(k.open) if k.open else np.nan,
                    'high': float(k.high) if k.high else np.nan,
                    'low': float(k.low) if k.low else np.nan,
                    'close': float(k.close) if k.close else np.nan,
                    'volume': float(k.volume) if k.volume else np.nan,
                    'amount': float(k.amount) if k.amount else np.nan,
                })
            
            df = pd.DataFrame(all_data)
            
            # 数据类型优化
            df['open'] = df['open'].astype('float32')
            df['high'] = df['high'].astype('float32')
            df['low'] = df['low'].astype('float32')
            df['close'] = df['close'].astype('float32')
            df['volume'] = df['volume'].astype('float32')
            df['amount'] = df['amount'].astype('float32')
            df['symbol'] = df['symbol'].astype('category')
            df['timeframe'] = df['timeframe'].astype('category')
            
            # 按时间排序
            if not df.empty:
                df = df.sort_values('start_time').reset_index(drop=True)
            self._send_progress('loading_data', 'completed', symbol=symbol, timeframe=timeframe_str)
            return df
        # 创建所有任务
        tasks = [(symbol, timeframe_str) for symbol in symbols for timeframe_str in timeframes]
        
        total_groups = len(tasks)
        completed_groups = 0
        phase_completed_sent = False
        
        # 发送阶段开始回调
        if total_groups > 0:
            self._send_progress('loading_data', 'running')
        
        # 使用线程池并发加载数据
        with ThreadPoolExecutor(max_workers=1) as executor:  # 恢复并发数
            futures = {
                executor.submit(load_single_group, symbol, timeframe_str): (symbol, timeframe_str)
                for symbol, timeframe_str in tasks
            }
            
            # 主线程通过 as_completed 获取加载完成的数据
            for future in as_completed(futures):
                symbol, timeframe_str = futures[future]
                try:
                    df = future.result()
                    completed_groups += 1
                    if df is not None:
                        yield symbol, timeframe_str, df
                except Exception as e:
                    logger.error(f"Failed to load data for {symbol}_{timeframe_str}: {e}", exc_info=True)
                    completed_groups += 1
                    executor.shutdown(wait=False, cancel_futures=True)
                    # 发送失败回调
                    self._send_progress('loading_data', 'failed', symbol=symbol, timeframe=timeframe_str)
                    # 即使失败也继续处理其他数据
                
                # 所有任务完成后发送阶段完成回调
                if completed_groups >= total_groups and not phase_completed_sent:
                    self._send_progress('loading_data', 'completed')
                    phase_completed_sent = True
        
    def _compute_features_and_labels(
        self,
        data_generator: Generator[Tuple[str, str, pd.DataFrame], None, None],
    ) -> Generator[Tuple[str, str, pd.DataFrame, pd.Series], None, None]:
        """
        流式计算特征和标签（pandas计算不支持并发）
        
        返回: Generator[Tuple[str, str, pd.DataFrame, pd.Series], None, None]，包含特征和标签的完整数据
        """
        # 发送 computing_features 阶段开始回调
        self._send_progress('computing_features', 'running')
        self._send_progress('generating_labels', 'running')
        
        # 流式处理每个组合（pandas计算不支持并发） 
        for symbol, timeframe_str, df in data_generator:
            if df is None or df.empty:
                continue
                
            try:
                # 发送开始计算特征的回调
                self._send_progress('computing_features', 'running', symbol=symbol, timeframe=timeframe_str)
                
                feature_df = self.feature_engine.compute_all(df)
                # 将 start_time 添加到 feature_df 中（作为元数据列，用于排序，不用于训练）
                # 确保索引对齐
                if 'start_time' in df.columns:
                    feature_df['start_time'] = df['start_time'].values
                # 发送特征计算完成回调
                self._send_progress('computing_features', 'completed', symbol=symbol, timeframe=timeframe_str)
                
                # 发送开始生成标签的回调
                self._send_progress('generating_labels', 'running', symbol=symbol, timeframe=timeframe_str)
                
                # 生成标签
                label_series = self.label_generator.generate(df)
                
                # 发送标签生成完成回调
                self._send_progress('generating_labels', 'completed', symbol=symbol, timeframe=timeframe_str)
                
                yield symbol, timeframe_str, feature_df, label_series
                
            except Exception as e:
                error_msg = f"Failed to process {symbol}_{timeframe_str}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                # 发送失败回调
                self._send_progress('computing_features', 'failed', symbol=symbol, timeframe=timeframe_str, error_message=error_msg)
                self._send_progress('generating_labels', 'failed', symbol=symbol, timeframe=timeframe_str, error_message=error_msg)
                # 继续处理下一个组合，而不是中断整个流程
                # 如果希望严格失败，可以取消下面的注释并启用 raise
                # raise
                continue
        
        # 发送 computing_features 阶段完成回调
        self._send_progress('computing_features', 'completed')
        self._send_progress('generating_labels', 'completed')
    
    def _split_data(
        self,
        processed_data_generator: Generator[Tuple[str, str, pd.DataFrame, pd.Series], None, None],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        按时间顺序切分数据（流式处理）
        
        返回: (X_train, y_train, X_val, y_val)
        """
        # 发送阶段开始回调
        self._send_progress('splitting_data', 'running')
        
        train_chunks = []
        val_chunks = []
        
        completed_groups = 0
        feature_names = self.feature_engine.get_feature_names()
        label_name = self.label_generator.get_label_name()
        
        for symbol, timeframe_str, feature_df, label_series in processed_data_generator:
            self._send_progress('splitting_data', 'running', symbol=symbol, timeframe=timeframe_str)
            if feature_df is None or feature_df.empty:
                logger.warning(f"Skipping {symbol}_{timeframe_str}: empty dataframe")
                continue
            
            # 将标签合并到特征DataFrame中，确保索引对齐
            df_with_label = feature_df.copy()
            df_with_label[label_name] = label_series
            
            # 删除 NaN（通常是 label 为 NaN 的行）
            df_clean = df_with_label.dropna()
            if df_clean.empty:
                logger.warning(f"Skipping {symbol}_{timeframe_str}: all rows contain NaN after dropna")
                continue
            
            # 按时间顺序切分
            split_idx = int(len(df_clean) * self.train_split_ratio)
            train_df = df_clean.iloc[:split_idx]
            val_df = df_clean.iloc[split_idx:]
            # 如果训练集或验证集为空，跳过这个数据集
            if train_df.empty or val_df.empty:
                logger.warning(
                    f"Skipping {symbol}_{timeframe_str}: train_df empty={train_df.empty}, "
                    f"val_df empty={val_df.empty}"
                )
                continue
            
            train_chunks.append(train_df)
            val_chunks.append(val_df)
            
            completed_groups += 1
            logger.info(f"Successfully split {symbol}_{timeframe_str}: train={len(train_df)}, val={len(val_df)}")
            # 发送单个任务完成回调
            self._send_progress('splitting_data', 'completed', symbol=symbol, timeframe=timeframe_str)
        
        if not train_chunks or not val_chunks:
            logger.error(
                f"No valid data after splitting. "
                f"Processed {completed_groups} groups, "
                f"train_chunks: {len(train_chunks)}, val_chunks: {len(val_chunks)}"
            )
            raise ValueError(
                f"No valid data after splitting. "
                f"All datasets were either empty after dropna() or resulted in empty train/val splits. "
                f"Processed {completed_groups} groups."
            )
        
        self._send_progress('splitting_data', 'completed')
        self._send_progress('merging_data', 'running')
        # 合并所有训练集和验证集
        train_df_all = pd.concat(train_chunks, ignore_index=True)
        val_df_all = pd.concat(val_chunks, ignore_index=True)
        
        # 按时间排序
        train_df_all = train_df_all.sort_values('start_time').reset_index(drop=True)
        val_df_all = val_df_all.sort_values('start_time').reset_index(drop=True)
        
        # 分离特征和标签
        X_train = train_df_all[feature_names]
        y_train = train_df_all[label_name]
        X_val = val_df_all[feature_names]
        y_val = val_df_all[label_name]
        self._send_progress('merging_data', 'completed')
        return X_train, y_train, X_val, y_val
    
    def _load_old_model(self) -> bool:
        """加载旧模型"""
        if not self.load_model_path or not os.path.exists(self.load_model_path):
            return False
        
        self._send_progress('loading_old_model', 'running')
        
        try:
            self.trainer.load_model(path=self.load_model_path)
            self._send_progress('loading_old_model', 'completed')
            return True
        except Exception as e:
            error_msg = f"Failed to load old model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_progress('loading_old_model', 'failed', error_message=error_msg)
            return False
    
    def _evaluate_old_model(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """评估旧模型"""
        if self.trainer._model is None:
            return None
        
        self._send_progress('evaluating_old_model', 'running')
        
        try:
            predict_result = self.trainer.predict(X_val)
            old_metrics = self.evaluator.evaluate(
                y_true=y_val,
                predict_result=predict_result,
                verbose=False
            )
            
            self._send_progress('evaluating_old_model', 'completed')
            return old_metrics
        except Exception as e:
            error_msg = f"Failed to evaluate old model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_progress('evaluating_old_model', 'failed', error_message=error_msg)
            return None
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """训练模型"""
        self._send_progress('training', 'running')
        
        try:
            # 创建训练进度回调
            def training_progress_callback(current_iteration: int, total_iterations: int, metrics: Dict[str, Any]):
                """训练进度回调"""
                progress = current_iteration / total_iterations if total_iterations > 0 else 0.0
                progress_info = {
                    'phase': 'training',
                    'status': 'running',
                    'progress': progress,
                    'current_iteration': current_iteration,
                    'total_iterations': total_iterations,
                    'metrics': metrics
                }
                self._call_back(progress_info)
            
            # 调用训练器，传入进度回调
            # 如果训练器支持 categorical_info（如 GRUTrainer），传递 FeatureEngine 的 categorical 信息
            train_kwargs = {}
            categorical_info = self.feature_engine.get_categorical_info()
            if categorical_info and hasattr(self.trainer, '_categorical_info'):
                train_kwargs['categorical_info'] = categorical_info
            
            self.trainer.train(
                X_train, 
                y_train, 
                X_val, 
                y_val,
                progress_callback=training_progress_callback,
                **train_kwargs
            )
            self._send_progress('training', 'completed')
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_progress('training', 'failed', error_message=error_msg)
            raise
    
    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """评估模型"""
        self._send_progress('evaluating', 'running')
        
        try:
            # 评估训练集
            train_predict = self.trainer.predict(X_train)
            train_metrics = self.evaluator.evaluate(
                y_true=y_train,
                predict_result=train_predict,
                verbose=False
            )
            
            # 评估验证集
            val_predict = self.trainer.predict(X_val)
            val_metrics = self.evaluator.evaluate(
                y_true=y_val,
                predict_result=val_predict,
                verbose=False
            )
            
            metrics = {
                'train': train_metrics,
                'validation': val_metrics
            }
            
            self._send_progress('evaluating', 'completed')
            
            return metrics
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_progress('evaluating', 'failed', error_message=error_msg)
            raise
    
    def _save_model(self) -> Optional[str]:
        """保存模型"""
        if not self.save_model_path:
            return None
        
        self._send_progress('saving_model', 'running')
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
            
            self.trainer.save_model(path=self.save_model_path)
            
            self._send_progress('saving_model', 'completed')
            
            return self.save_model_path
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_progress('saving_model', 'failed', error_message=error_msg)
            return None
    
    def train(self) -> Dict[str, Any]:
        """
        主训练流程
        
        返回: {
            'old_model_metrics': {...},  # 如果存在旧模型
            'new_model_metrics': {
                'train': {...},
                'validation': {...}
            },
            'model_path': str  # 保存的模型路径
            'encoder_classes': {
                'symbol_classes': list,
                'timeframe_classes': list
            }
        }
        """
        try:
            # 1. 加载数据（流式处理）
            data_generator = self._load_data()
            
            # 计算总任务数（用于进度显示）
            symbols = self.datasource_extra.get('symbols', [])
            
            # 1.5. 预拟合编码器（只拟合 symbol，timeframe 已在 FeatureEngine 初始化时固定拟合）
            # 这样可以确保编码器能够处理所有数据，避免后续出现 "previously unseen labels" 错误
            if self.feature_engine.enable_symbol_timeframe_encoding:
                # timeframe 编码器已经在 FeatureEngine 初始化时使用固定的 TimeFrame 枚举值拟合
                # 这里只需要拟合 symbol 编码器
                self.feature_engine.fit_encoders(symbols=symbols)
            
            # 2-3. 计算特征和生成标签（流式处理）
            processed_data_generator = self._compute_features_and_labels(data_generator)
            
            # 4. 切分数据（流式处理）
            X_train, y_train, X_val, y_val = self._split_data(processed_data_generator)
            
            # 5. 加载旧模型
            has_old_model = self._load_old_model()
            
            # 6. 评估旧模型（使用已编码的特征，因为新模型包含编码特征）
            old_model_metrics = None
            if has_old_model:
                old_model_metrics = self._evaluate_old_model(X_val, y_val)
            
            # 8. 训练模型
            self._train_model(X_train, y_train, X_val, y_val)
            
            # 9. 评估模型
            new_model_metrics = self._evaluate_model(X_train, y_train, X_val, y_val)
            
            # 10. 保存模型
            model_path = self._save_model()
            
            # 11. 获取编码器信息（用于保存到 model_config）
            result = {
                'new_model_metrics': new_model_metrics,
                'model_path': model_path,
                'encoder_classes': self.feature_engine.get_encoder_classes()  # 保存编码器类别信息
            }
            
            if old_model_metrics is not None:
                result['old_model_metrics'] = old_model_metrics
            
            return result
            
        except Exception as e:
            # 只传递错误消息，不包含堆栈跟踪
            error_msg = str(e)
            logger.error(f"Training failed: {error_msg}", exc_info=True)
            # 限制错误消息长度，避免过长
            error_msg_limited = error_msg[:2000] if len(error_msg) > 2000 else error_msg
            self._send_progress('failed', 'failed', error_message=error_msg_limited)
            raise
    