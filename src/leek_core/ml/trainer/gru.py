#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU 训练器实现

基于 PyTorch 的 GRU 模型，支持分类和回归任务。
自动将扁平特征转换为时序窗口格式。

依赖：需要安装 torch，可通过 `pip install torch` 或 `poetry add torch` 安装
"""
import io
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from leek_core.models import ChoiceType, Field, FieldType
from leek_core.utils import get_logger

from .base import BaseTrainer

logger = get_logger(__name__)

# 延迟导入 PyTorch，避免未安装时报错
torch = None
nn = None


def _lazy_import_torch():
    """延迟导入 PyTorch"""
    global torch, nn
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
            torch = _torch
            nn = _nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for GRUTrainer. "
                "Please install it via: pip install torch or poetry add torch"
            )


class GRUModel(object):
    """
    GRU 模型封装类
    
    包含模型结构和相关配置，便于序列化保存
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        task_type: str,
        window_size: int,
        feature_names: List[str],
    ):
        _lazy_import_torch()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.task_type = task_type
        self.window_size = window_size
        self.feature_names = feature_names
        
        # 构建网络
        self.network = self._build_network()
        self.callbacks = None  # 兼容基类的 save_model
    
    def _build_network(self):
        """构建 GRU 网络"""
        _lazy_import_torch()
        
        class GRUNetwork(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                num_classes: int,
                dropout: float,
                bidirectional: bool,
                task_type: str,
            ):
                super().__init__()
                self.task_type = task_type
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                self.num_directions = 2 if bidirectional else 1
                
                # GRU 层
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                )
                
                # 输出层
                fc_input_size = hidden_size * self.num_directions
                
                if task_type == "classification":
                    self.fc = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(fc_input_size, fc_input_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout / 2),
                        nn.Linear(fc_input_size // 2, num_classes),
                    )
                else:  # regression
                    self.fc = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(fc_input_size, fc_input_size // 2),
                        nn.ReLU(),
                        nn.Linear(fc_input_size // 2, 1),
                    )
            
            def forward(self, x):
                # x: (batch, seq_len, input_size)
                # GRU 输出
                gru_out, _ = self.gru(x)
                # 取最后一个时间步的输出
                # gru_out: (batch, seq_len, hidden_size * num_directions)
                out = gru_out[:, -1, :]
                # 全连接层
                out = self.fc(out)
                return out
        
        return GRUNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            task_type=self.task_type,
        )
    
    def to(self, device):
        """移动模型到指定设备"""
        self.network = self.network.to(device)
        return self
    
    def train_mode(self):
        """设置为训练模式"""
        self.network.train()
    
    def eval_mode(self):
        """设置为评估模式"""
        self.network.eval()
    
    def parameters(self):
        """获取模型参数"""
        return self.network.parameters()
    
    def state_dict(self):
        """获取模型状态字典"""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        self.network.load_state_dict(state_dict)
    
    def __call__(self, x):
        """前向传播"""
        return self.network(x)


class GRUTrainer(BaseTrainer):
    """
    GRU 训练器
    
    基于 PyTorch 的 GRU 模型，支持分类和回归任务。
    自动将扁平特征转换为时序窗口格式进行训练。
    
    特点：
    - 支持时序建模，适合短线择时策略
    - 自动处理时序窗口，无需修改 FeatureEngine
    - 支持 GPU 加速（如果可用）
    - 支持早停、学习率调度等训练技巧
    """
    
    display_name = "GRU训练器"
    
    init_params = [
        Field(
            name="task_type",
            label="任务类型",
            type=FieldType.RADIO,
            default="classification",
            description="选择任务类型：classification（分类）或 regression（回归）",
            required=True,
            choices=[("classification", "分类"), ("regression", "回归")],
            choice_type=ChoiceType.STRING,
        ),
        Field(
            name="window_size",
            label="时序窗口大小",
            type=FieldType.INT,
            default=30,
            description="用于预测的历史K线数量，范围 5-200",
            min=5,
            max=200,
            required=True,
        ),
        Field(
            name="hidden_size",
            label="隐藏层大小",
            type=FieldType.INT,
            default=64,
            description="GRU 隐藏层的神经元数量，范围 16-512",
            min=16,
            max=512,
            required=False,
        ),
        Field(
            name="num_layers",
            label="GRU层数",
            type=FieldType.INT,
            default=2,
            description="GRU 层的数量，范围 1-4",
            min=1,
            max=4,
            required=False,
        ),
        Field(
            name="dropout",
            label="Dropout比例",
            type=FieldType.FLOAT,
            default=0.2,
            description="Dropout 正则化比例，范围 0-0.5",
            min=0,
            max=0.5,
            required=False,
        ),
        Field(
            name="bidirectional",
            label="双向GRU",
            type=FieldType.BOOLEAN,
            default=False,
            description="是否使用双向 GRU（增强特征提取，但推理速度会降低）",
            required=False,
        ),
        Field(
            name="learning_rate",
            label="学习率",
            type=FieldType.FLOAT,
            default=0.001,
            description="优化器学习率，范围 0.0001-0.1",
            min=0.0001,
            max=0.1,
            required=False,
        ),
        Field(
            name="batch_size",
            label="批次大小",
            type=FieldType.INT,
            default=64,
            description="每批次训练样本数，范围 16-512",
            min=16,
            max=512,
            required=False,
        ),
        Field(
            name="epochs",
            label="训练轮数",
            type=FieldType.INT,
            default=100,
            description="最大训练轮数，范围 10-500",
            min=10,
            max=500,
            required=False,
        ),
        Field(
            name="early_stopping_patience",
            label="早停耐心值",
            type=FieldType.INT,
            default=10,
            description="验证集性能连续多少轮没有提升则停止训练，设为0禁用早停",
            min=0,
            max=50,
            required=False,
        ),
        Field(
            name="random_state",
            label="随机种子",
            type=FieldType.INT,
            default=None,
            description="随机种子，用于结果复现",
            required=False,
        ),
        Field(
            name="device",
            label="计算设备",
            type=FieldType.RADIO,
            default="auto",
            description="选择计算设备：auto（自动选择GPU/CPU）、cuda（强制GPU）、cpu（强制CPU）",
            choices=[("auto", "自动"), ("cuda", "GPU"), ("cpu", "CPU")],
            choice_type=ChoiceType.STRING,
            required=False,
        ),
    ]
    
    def __init__(
        self,
        task_type: str = "classification",
        window_size: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        random_state: Optional[int] = None,
        device: str = "auto",
    ):
        """
        初始化 GRU 训练器
        
        :param task_type: 任务类型，"classification"（分类）或 "regression"（回归）
        :param window_size: 时序窗口大小，用于预测的历史K线数量
        :param hidden_size: GRU 隐藏层大小
        :param num_layers: GRU 层数
        :param dropout: Dropout 比例
        :param bidirectional: 是否使用双向 GRU
        :param learning_rate: 学习率
        :param batch_size: 批次大小
        :param epochs: 最大训练轮数
        :param early_stopping_patience: 早停耐心值，0表示禁用
        :param random_state: 随机种子
        :param device: 计算设备 ("auto", "cuda", "cpu")
        """
        super().__init__()
        
        # 验证任务类型
        if task_type not in ["classification", "regression"]:
            raise ValueError(
                f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'"
            )
        
        self.task_type = task_type
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device_config = device
        
        # 运行时属性
        self._device = None
        self._feature_names = None
        self._num_classes = None
    
    def _get_device(self):
        """获取计算设备"""
        _lazy_import_torch()
        
        if self._device is not None:
            return self._device
        
        if self.device_config == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_config == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Using device: {self._device}")
        return self._device
    
    def _set_random_seed(self):
        """设置随机种子"""
        if self.random_state is not None:
            _lazy_import_torch()
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
    
    def _create_sequences(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将扁平特征转换为时序窗口格式
        
        :param X: 特征 DataFrame (N, features)
        :param y: 标签 Series (N,)
        :return: (X_seq, y_seq) 其中 X_seq 形状为 (N-window_size, window_size, features)
        """
        X_values = X.values.astype(np.float32)
        y_values = y.values
        
        X_seq, y_seq = [], []
        for i in range(self.window_size, len(X_values)):
            X_seq.append(X_values[i - self.window_size:i])
            y_seq.append(y_values[i])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        shuffle: bool = True
    ):
        """创建 DataLoader"""
        _lazy_import_torch()
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.FloatTensor(X)
        
        if self.task_type == "classification":
            y_tensor = torch.LongTensor(y)
        else:
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            drop_last=False,
        )
        
        return dataloader
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ):
        """
        训练 GRU 模型
        
        :param X_train: 训练集特征 DataFrame（扁平格式，会自动转换为时序窗口）
        :param y_train: 训练集标签 Series
        :param X_val: 验证集特征 DataFrame（可选）
        :param y_val: 验证集标签 Series（可选）
        :param progress_callback: 进度回调函数
        """
        _lazy_import_torch()
        
        # 设置随机种子
        self._set_random_seed()
        
        # 获取设备
        device = self._get_device()
        
        # 保存特征名称
        self._feature_names = list(X_train.columns)
        input_size = len(self._feature_names)
        
        # 转换为时序格式
        logger.info(f"Creating sequences with window_size={self.window_size}...")
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        logger.info(f"Training sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            logger.info(f"Validation sequences shape: X={X_val_seq.shape}, y={y_val_seq.shape}")
        else:
            X_val_seq, y_val_seq = None, None
        
        # 确定类别数（分类任务）
        if self.task_type == "classification":
            self._num_classes = len(np.unique(y_train_seq))
            logger.info(f"Number of classes: {self._num_classes}")
        else:
            self._num_classes = 1
        
        # 创建模型
        self._model = GRUModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self._num_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            task_type=self.task_type,
            window_size=self.window_size,
            feature_names=self._feature_names,
        ).to(device)
        
        # 创建 DataLoader
        train_loader = self._create_dataloader(X_train_seq, y_train_seq, shuffle=True)
        val_loader = None
        if X_val_seq is not None:
            val_loader = self._create_dataloader(X_val_seq, y_val_seq, shuffle=False)
        
        # 定义损失函数
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # 定义优化器
        optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=self.learning_rate
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
        )
        
        # 早停相关
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 训练循环
        logger.info("Starting training...")
        for epoch in range(self.epochs):
            # 训练阶段
            self._model.train_mode()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
                if self.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # 验证阶段
            val_loss = None
            val_metrics = {}
            
            if val_loader is not None:
                self._model.eval_mode()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        outputs = self._model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * batch_X.size(0)
                        
                        if self.task_type == "classification":
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()
                
                val_loss = val_loss / len(val_loader.dataset)
                
                if self.task_type == "classification":
                    val_metrics['val_accuracy'] = val_correct / val_total
                val_metrics['val_loss'] = val_loss
                
                # 学习率调度
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # 构建指标
            metrics = {'train_loss': train_loss}
            if self.task_type == "classification":
                metrics['train_accuracy'] = train_correct / train_total
            metrics.update(val_metrics)
            
            # 进度回调
            if progress_callback is not None:
                progress_callback(epoch + 1, self.epochs, metrics)
            
            # 日志
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch [{epoch + 1}/{self.epochs}] - train_loss: {train_loss:.4f}"
                if self.task_type == "classification":
                    log_msg += f", train_acc: {train_correct / train_total:.4f}"
                if val_loss is not None:
                    log_msg += f", val_loss: {val_loss:.4f}"
                    if self.task_type == "classification":
                        log_msg += f", val_acc: {val_metrics.get('val_accuracy', 0):.4f}"
                logger.info(log_msg)
        
        # 恢复最佳模型
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with val_loss: {best_val_loss:.4f}")
        
        logger.info("Training completed!")
    
    def predict(self, X_test: pd.DataFrame) -> dict:
        """
        预测结果
        
        :param X_test: 测试集特征 DataFrame（扁平格式，会自动转换为时序窗口）
        :return: 预测结果字典
            - y_pred: 预测类别（分类）或预测值（回归）
            - y_proba: 预测概率（仅分类任务）
        """
        _lazy_import_torch()
        
        if self._model is None:
            raise ValueError("Model not trained. Please call train() first.")
        
        device = self._get_device()
        self._model.to(device)
        self._model.eval_mode()
        
        # 转换为时序格式
        # 注意：预测时需要创建一个假的标签
        dummy_y = pd.Series(np.zeros(len(X_test)), index=X_test.index)
        X_test_seq, _ = self._create_sequences(X_test, dummy_y)
        
        # 调整索引（因为窗口化会丢失前 window_size 个样本）
        valid_indices = X_test.index[self.window_size:]
        
        # 创建 tensor
        X_tensor = torch.FloatTensor(X_test_seq).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = self._model(X_tensor)
            
            if self.task_type == "classification":
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(proba, axis=1)
                
                result = {
                    'y_pred': pd.Series(y_pred, index=valid_indices),
                }
                
                # 返回概率
                if proba.shape[1] == 2:
                    # 二分类，返回正类概率
                    result['y_proba'] = pd.Series(proba[:, 1], index=valid_indices)
                else:
                    # 多分类，返回完整概率矩阵
                    result['y_proba'] = pd.DataFrame(proba, index=valid_indices)
            else:
                y_pred = outputs.cpu().numpy().flatten()
                result = {
                    'y_pred': pd.Series(y_pred, index=valid_indices),
                }
        
        return result
    
    def save_model(self, path: Optional[str] = None) -> Optional[io.BytesIO]:
        """
        保存模型
        
        :param path: 保存路径（可选）
        :return: 如果未提供路径，返回 BytesIO 对象
        """
        if self._model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # 准备保存的数据
        save_data = {
            'model_state_dict': self._model.state_dict(),
            'model_config': {
                'input_size': self._model.input_size,
                'hidden_size': self._model.hidden_size,
                'num_layers': self._model.num_layers,
                'num_classes': self._model.num_classes,
                'dropout': self._model.dropout,
                'bidirectional': self._model.bidirectional,
                'task_type': self._model.task_type,
                'window_size': self._model.window_size,
                'feature_names': self._model.feature_names,
            },
            'trainer_config': {
                'task_type': self.task_type,
                'window_size': self.window_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'early_stopping_patience': self.early_stopping_patience,
            },
        }
        
        if path is not None:
            joblib.dump(save_data, path)
            return None
        else:
            model_io = io.BytesIO()
            joblib.dump(save_data, model_io)
            model_io.seek(0)
            return model_io
    
    def load_model(
        self, 
        path: Optional[str] = None, 
        model: Optional[Any] = None, 
        model_io: Optional[io.BytesIO] = None
    ):
        """
        加载模型
        
        :param path: 模型文件路径
        :param model: 模型对象（直接传入）
        :param model_io: BytesIO 对象
        """
        _lazy_import_torch()
        
        # 加载数据
        if model is not None:
            save_data = model
        elif model_io is not None:
            model_io.seek(0)
            save_data = joblib.load(model_io)
        elif path is not None:
            save_data = joblib.load(path)
        else:
            raise ValueError(
                "No model source specified. Provide one of: path, model, or model_io"
            )
        
        # 恢复模型配置
        model_config = save_data['model_config']
        self._feature_names = model_config['feature_names']
        self._num_classes = model_config['num_classes']
        
        # 同步 trainer 配置（重要：确保 predict 时使用正确的参数）
        self.window_size = model_config['window_size']
        self.task_type = model_config['task_type']
        
        # 重建模型
        self._model = GRUModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional'],
            task_type=model_config['task_type'],
            window_size=model_config['window_size'],
            feature_names=model_config['feature_names'],
        )
        
        # 加载权重
        self._model.load_state_dict(save_data['model_state_dict'])
        
        # 移动到设备
        device = self._get_device()
        self._model.to(device)
        
        logger.info(f"Model loaded successfully. Window size: {model_config['window_size']}")
