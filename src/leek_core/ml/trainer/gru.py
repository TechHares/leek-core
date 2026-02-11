#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU 训练器实现

基于 PyTorch 的 GRU 模型，支持分类和回归任务。
自动将扁平特征转换为时序窗口格式。
支持 categorical 特征的 Embedding 层。

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


def _compute_embed_dim(num_categories: int) -> int:
    """计算 Embedding 维度的启发式规则"""
    return min(50, max(2, (num_categories + 1) // 2))


class GRUModel(object):
    """
    GRU 模型封装类
    
    包含模型结构和相关配置，便于序列化保存。
    支持 categorical 特征的 Embedding 层。
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
        categorical_info: Optional[Dict[str, int]] = None,
    ):
        """
        :param input_size: 数值特征数量（不含 categorical）
        :param hidden_size: GRU 隐藏层大小
        :param num_layers: GRU 层数
        :param num_classes: 输出类别数
        :param dropout: Dropout 比例
        :param bidirectional: 是否双向
        :param task_type: 任务类型
        :param window_size: 时序窗口大小
        :param feature_names: 全部特征名称列表
        :param categorical_info: categorical 特征信息 {feature_name: num_categories}
        """
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
        self.categorical_info = categorical_info or {}
        
        # 计算 categorical/numeric 特征的列索引
        self.categorical_indices = []
        self.numeric_indices = []
        self.embed_configs = []  # [(num_categories, embed_dim), ...]
        
        for i, name in enumerate(feature_names):
            if name in self.categorical_info:
                self.categorical_indices.append(i)
                num_cat = self.categorical_info[name]
                embed_dim = _compute_embed_dim(num_cat)
                self.embed_configs.append((num_cat, embed_dim))
            else:
                self.numeric_indices.append(i)
        
        self.total_embed_dim = sum(ed for _, ed in self.embed_configs)
        
        # 构建网络
        self.network = self._build_network()
        self.callbacks = None  # 兼容基类的 save_model
    
    def _build_network(self):
        """构建 GRU 网络（支持 Embedding）"""
        _lazy_import_torch()
        
        cat_indices = self.categorical_indices
        num_indices = self.numeric_indices
        embed_configs = self.embed_configs
        num_numeric = len(num_indices)
        total_embed_dim = self.total_embed_dim
        
        class GRUNetwork(nn.Module):
            def __init__(
                self,
                numeric_size: int,
                embed_configs_inner: list,
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
                self.cat_indices = cat_indices
                self.num_indices = num_indices
                
                # 缓存索引为 buffer，避免每次 forward 重新创建 Tensor
                if num_indices:
                    self.register_buffer(
                        '_num_idx', torch.tensor(num_indices, dtype=torch.long)
                    )
                else:
                    self._num_idx = None
                if cat_indices:
                    self.register_buffer(
                        '_cat_idx', torch.tensor(cat_indices, dtype=torch.long)
                    )
                else:
                    self._cat_idx = None
                
                # Embedding 层（每个 categorical 特征一个）
                self.embeddings = nn.ModuleList()
                self._total_embed_dim = 0
                for num_cat, embed_dim in embed_configs_inner:
                    self.embeddings.append(nn.Embedding(num_cat + 1, embed_dim, padding_idx=0))
                    self._total_embed_dim += embed_dim
                
                # GRU 输入大小 = 数值特征数 + embedding 总维度
                gru_input_size = numeric_size + self._total_embed_dim
                
                # GRU 层
                self.gru = nn.GRU(
                    input_size=gru_input_size,
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
                # x: (batch, seq_len, total_features) - all features as float
                parts = []
                
                # 1. 提取数值特征（使用缓存的 buffer 索引）
                if self._num_idx is not None:
                    numeric_features = x.index_select(2, self._num_idx)
                    parts.append(numeric_features)
                
                # 2. 提取 categorical 特征并通过 Embedding
                if self._cat_idx is not None:
                    cat_features = x.index_select(2, self._cat_idx).long()  # (batch, seq_len, num_cat)
                    
                    embed_parts = []
                    for j, embed_layer in enumerate(self.embeddings):
                        # 取第 j 个 categorical 特征
                        cat_col = cat_features[:, :, j]  # (batch, seq_len)
                        embedded = embed_layer(cat_col)    # (batch, seq_len, embed_dim)
                        embed_parts.append(embedded)
                    
                    if embed_parts:
                        parts.append(torch.cat(embed_parts, dim=2))
                
                # 3. 拼接所有特征
                gru_input = torch.cat(parts, dim=2)  # (batch, seq_len, gru_input_size)
                
                # 4. GRU 输出
                gru_out, _ = self.gru(gru_input)
                out = gru_out[:, -1, :]
                out = self.fc(out)
                return out
        
        return GRUNetwork(
            numeric_size=num_numeric,
            embed_configs_inner=embed_configs,
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
    支持 categorical 特征通过 Embedding 层编码。
    
    特点：
    - 支持时序建模，适合短线择时策略
    - 自动处理时序窗口，无需修改 FeatureEngine
    - 支持 categorical 特征 Embedding（如时间特征 hour/day_of_week 等）
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
            description="选择计算设备：auto（自动选择GPU/CPU）、cuda（NVIDIA GPU）、mps（Apple Silicon GPU）、cpu（CPU）",
            choices=[("auto", "自动"), ("cuda", "NVIDIA GPU"), ("mps", "Apple GPU"), ("cpu", "CPU")],
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
        super().__init__()
        
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
        self._categorical_info = None  # {feature_name: num_categories}
    
    def _get_device(self):
        """获取计算设备（支持 CUDA / MPS / CPU）"""
        _lazy_import_torch()
        
        if self._device is not None:
            return self._device
        
        if self.device_config == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        elif self.device_config == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("cuda")
        elif self.device_config == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS not available, falling back to CPU")
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("mps")
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
        将扁平特征转换为时序窗口格式（向量化实现，零拷贝滑动窗口）
        
        所有特征统一为 float32（categorical 特征在网络内部转为 LongTensor）
        
        :param X: 特征 DataFrame (N, features)
        :param y: 标签 Series (N,)
        :return: (X_seq, y_seq) 其中 X_seq 形状为 (N-window_size, window_size, features)
        """
        X_values = np.ascontiguousarray(X.values, dtype=np.float32)
        y_values = y.values
        
        # 使用 stride_tricks 实现零拷贝滑动窗口，比 Python for 循环快 100x+
        n_samples, n_features = X_values.shape
        stride_sample, stride_feature = X_values.strides
        X_seq = np.lib.stride_tricks.as_strided(
            X_values,
            shape=(n_samples - self.window_size, self.window_size, n_features),
            strides=(stride_sample, stride_sample, stride_feature),
        )
        # as_strided 返回视图，需要 copy 以避免后续操作的内存问题
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = y_values[self.window_size:]
        
        return X_seq, y_seq
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        shuffle: bool = True
    ):
        """创建 DataLoader（数据预先搬到目标设备，避免逐 batch 传输）"""
        _lazy_import_torch()
        from torch.utils.data import DataLoader, TensorDataset
        
        device = self._get_device()
        
        # 直接在目标设备上创建 Tensor，避免训练循环中反复 .to(device)
        # 对 MPS/CUDA 设备尤为重要：消除每 batch 的 CPU→GPU 传输延迟
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        if self.task_type == "classification":
            y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        
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
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        categorical_info: Optional[Dict[str, int]] = None,
    ):
        """
        训练 GRU 模型
        
        :param X_train: 训练集特征 DataFrame（扁平格式，会自动转换为时序窗口）
        :param y_train: 训练集标签 Series
        :param X_val: 验证集特征 DataFrame（可选）
        :param y_val: 验证集标签 Series（可选）
        :param progress_callback: 进度回调函数
        :param categorical_info: categorical 特征信息 {feature_name: num_categories}，
               来自 FeatureEngine.get_categorical_info()
        """
        _lazy_import_torch()
        
        # 设置随机种子
        self._set_random_seed()
        
        # 获取设备
        device = self._get_device()
        
        # 保存特征名称和 categorical 信息
        self._feature_names = list(X_train.columns)
        self._categorical_info = categorical_info or {}
        
        # 只保留在特征列中实际存在的 categorical 信息
        self._categorical_info = {
            k: v for k, v in self._categorical_info.items()
            if k in self._feature_names
        }
        
        num_numeric = len(self._feature_names) - len(self._categorical_info)
        
        if self._categorical_info:
            cat_embed_dims = sum(
                _compute_embed_dim(nc) for nc in self._categorical_info.values()
            )
            logger.info(
                f"Categorical features: {len(self._categorical_info)}, "
                f"total embed dim: {cat_embed_dims}"
            )
        
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
        
        # 创建模型（input_size 传 numeric 特征数，embedding 在网络内部处理）
        self._model = GRUModel(
            input_size=num_numeric,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self._num_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            task_type=self.task_type,
            window_size=self.window_size,
            feature_names=self._feature_names,
            categorical_info=self._categorical_info,
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
                # 数据已在目标设备上，无需 .to(device)
                optimizer.zero_grad(set_to_none=True)
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
                        # 数据已在目标设备上，无需 .to(device)
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
        
        # 释放训练阶段的 GPU 显存（优化器、梯度、DataLoader 等），为预测腾出空间
        del train_loader, optimizer, scheduler, criterion
        if val_loader is not None:
            del val_loader
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Training completed!")
    
    def predict(self, X_test: pd.DataFrame) -> dict:
        """
        预测结果（分批推理，避免 GPU OOM）
        
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
        dummy_y = pd.Series(np.zeros(len(X_test)), index=X_test.index)
        X_test_seq, _ = self._create_sequences(X_test, dummy_y)
        
        # 调整索引（因为窗口化会丢失前 window_size 个样本）
        valid_indices = X_test.index[self.window_size:]
        
        # 分批推理，每批推完立即搬回 CPU，避免 GPU OOM
        all_outputs = []
        with torch.no_grad():
            for i in range(0, len(X_test_seq), self.batch_size):
                batch = torch.tensor(
                    X_test_seq[i:i + self.batch_size],
                    dtype=torch.float32, device=device,
                )
                out = self._model(batch)
                all_outputs.append(out.cpu())
            
            outputs = torch.cat(all_outputs, dim=0)
            
            if self.task_type == "classification":
                proba = torch.softmax(outputs, dim=1).numpy()
                y_pred = np.argmax(proba, axis=1)
                
                result = {
                    'y_pred': pd.Series(y_pred, index=valid_indices),
                }
                
                # 返回概率
                if proba.shape[1] == 2:
                    result['y_proba'] = pd.Series(proba[:, 1], index=valid_indices)
                else:
                    result['y_proba'] = pd.DataFrame(proba, index=valid_indices)
            else:
                y_pred = outputs.numpy().flatten()
                result = {
                    'y_pred': pd.Series(y_pred, index=valid_indices),
                }
        
        return result
    
    def save_model(self, path: Optional[str] = None) -> Optional[io.BytesIO]:
        """保存模型"""
        if self._model is None:
            raise ValueError("No model to save. Please train the model first.")
        
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
                'categorical_info': self._model.categorical_info,
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
        """加载模型"""
        _lazy_import_torch()
        
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
        
        model_config = save_data['model_config']
        self._feature_names = model_config['feature_names']
        self._num_classes = model_config['num_classes']
        self._categorical_info = model_config.get('categorical_info', {})
        
        self.window_size = model_config['window_size']
        self.task_type = model_config['task_type']
        
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
            categorical_info=model_config.get('categorical_info', {}),
        )
        
        self._model.load_state_dict(save_data['model_state_dict'])
        
        device = self._get_device()
        self._model.to(device)
        
        cat_count = len(self._categorical_info)
        logger.info(
            f"Model loaded successfully. Window size: {model_config['window_size']}, "
            f"categorical features: {cat_count}"
        )
