"""
CrossGRU-2: 双分支时序 + 截面交互的端到端模型

参考：华创证券《CrossGRU-2：双分支时序模块的端到端选股模型》

================================================================================
一、核心思想
================================================================================

时序数据在不同尺度下的特征模式可能不同：
- 分钟级：高频噪声 + 短期动量
- 日级：中期趋势
- 周/月级：长期趋势

传统 GRU 只能处理单一时间尺度，CrossGRU-2 通过双分支结构同时捕捉多尺度信息。

================================================================================
二、模型架构
================================================================================

整体流程：
                    原始输入 X ∈ R^(l, m)
                    │  l=时间序列长度, m=特征数
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│   分支1 (高频)  │       │   分支2 (低频)  │
│  小 Patch Size │       │  大 Patch Size │
│  kernel_size=5 │       │ kernel_size=20 │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  1D Conv       │       │  1D Conv       │
│  (Patch Embed) │       │  (Patch Embed) │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
    X_hf ∈ R^(l1, d)       X_lf ∈ R^(l2, d)
    l1 > l2 (高频token多)   l2 < l1 (低频token少)
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│    GRU-1      │       │    GRU-2      │
│   (高频分支)   │       │   (低频分支)   │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
  S_hf ∈ R^(1, d)         S_lf ∈ R^(l2, d)
  (取最后时间步)           (全部时间步)
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Cross Attn   │
            │  Q=S_hf       │
            │  K,V=S_lf     │
            └───────┬───────┘
                    │
                    ▼
            S_ts = S_attn + S_hf  (残差连接)
                    │
                    ▼
            ┌───────────────┐
            │  截面交互模块   │  (可选)
            │ Cross Section │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   FFN + BN    │
            │  特征交互模块   │
            └───────┬───────┘
                    │
                    ▼
                  输出

================================================================================
三、关键组件详解
================================================================================

1. Patch Embedding (时序分块嵌入)
----------------------------------------------------------------------
目的：将连续时点聚合成子序列，减少序列长度，增加单个token信息量

实现：使用 1D 卷积，kernel_size 决定 patch 大小

原始序列: [K1, K2, K3, K4, K5, K6, ..., K60]  shape=(60, features)

Patch=5 后: [patch1, patch2, ..., patch12]  shape=(12, embed_dim)
            每个 patch 包含 5 个时间步的信息

代码思路:
    # patch_size=5, embed_dim=64
    self.patch_embed = nn.Conv1d(
        in_channels=num_features,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size  # 不重叠
    )
    # 或 stride < patch_size 实现重叠 patch

2. 双分支 GRU
----------------------------------------------------------------------
两个分支使用不同的 patch size：
- 高频分支：小 patch (如 5)，保留更多时间细节
- 低频分支：大 patch (如 20)，捕捉长期趋势

class DualBranchGRU(nn.Module):
    def __init__(self, ...):
        # 高频分支
        self.patch_embed_hf = nn.Conv1d(..., kernel_size=patch_size_hf)
        self.gru_hf = nn.GRU(...)
        
        # 低频分支
        self.patch_embed_lf = nn.Conv1d(..., kernel_size=patch_size_lf)
        self.gru_lf = nn.GRU(...)

3. 交叉注意力 (Cross Attention)
----------------------------------------------------------------------
融合高频和低频两个分支的信息

公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

- Q 来自高频分支 GRU 最后一个时间步: S_hf ∈ R^(1, d)
- K, V 来自低频分支 GRU 全部输出: S_lf ∈ R^(l', d)

含义：让"短期信号"去查询"长期上下文"，获得多尺度融合表征

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key_value):
        # query: (1, batch, d) 高频分支最后时间步
        # key_value: (seq_len, batch, d) 低频分支全部输出
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return attn_output

4. 截面交互模块 (Cross Section Attention) - 可选
----------------------------------------------------------------------
目的：让不同股票之间进行信息交互（适用于多股票同时预测）

机制：
1. 初始化 c 个可学习的"市场隐状态" R ∈ R^(c, d)
2. 第一次交叉注意力：R 作为 Q，股票表征作为 K,V → 聚合市场信息
3. 第二次交叉注意力：股票表征作为 Q，市场信息作为 K,V → 分发给每只股票
4. 门控残差连接

注意：对于单股票预测场景，此模块可省略

5. 特征交互模块 (FFN)
----------------------------------------------------------------------
标准的 Transformer FFN 结构：

S_out = BatchNorm(S_cs + MLP(ReLU(MLP(S_cs))))

class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm = nn.BatchNorm1d(embed_dim)
    
    def forward(self, x):
        return self.norm(x + self.mlp(x))

================================================================================
四、超参数建议
================================================================================

根据论文实验：

| 参数 | 30D数据集 | 90D数据集 | 30min数据集 |
|------|----------|----------|-------------|
| 高频 patch_size | 5 | 5 | 5 |
| 低频 patch_size | 10 | 30 | 10 |
| embed_dim | 64 | 64 | 64 |
| GRU hidden_size | 64 | 64 | 64 |
| GRU num_layers | 2 | 2 | 2 |
| attention heads | 4 | 4 | 4 |
| dropout | 0.1 | 0.1 | 0.1 |

================================================================================
五、实现优先级
================================================================================

建议分步实现：

Phase 1: 单分支 + Patch（在现有 GRUTrainer 基础上）
-------------------------------------------------
- 在 GRUTrainer 中增加 patch_size 参数
- 使用 1D 卷积实现 Patch Embedding
- 验证 Patch 对性能的影响

Phase 2: 双分支结构
-------------------------------------------------
- 实现 DualBranchGRU
- 实现 CrossAttention
- 双分支输出融合

Phase 3: 截面交互（可选）
-------------------------------------------------
- 实现 CrossSectionAttention
- 适用于多股票联合预测场景

================================================================================
六、与现有 GRUTrainer 的关系
================================================================================

方案A：扩展 GRUTrainer
- 增加 use_patch, patch_size_hf, patch_size_lf 等参数
- 向后兼容，默认关闭新功能

方案B：新建 CrossGRUTrainer（推荐）
- 独立实现，不影响原有 GRUTrainer
- 更清晰的代码结构

================================================================================
七、注意事项
================================================================================

1. 序列长度要求：
   - 需要 seq_len >= max(patch_size_hf, patch_size_lf) * 2
   - 建议 seq_len 为 patch_size 的整数倍

2. 计算开销：
   - 双分支 + 注意力会增加计算量
   - 但 Patch 减少了序列长度，部分抵消

3. 适用场景：
   - 论文验证了在 A 股日线/分钟线的有效性
   - 加密货币高频场景需要单独验证

4. 过拟合风险：
   - 模型更复杂，需要更多数据
   - 建议使用 dropout、early stopping

================================================================================
八、参考代码结构（不完整实现）
================================================================================
"""

# 以下为类结构骨架，具体实现待补充

from typing import Optional, Tuple
# import torch
# import torch.nn as nn


class PatchEmbedding:
    """
    将时间序列分块并嵌入到高维空间
    
    输入: (batch, seq_len, num_features)
    输出: (batch, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        patch_size: int = 5,
        stride: Optional[int] = None  # None 表示不重叠
    ):
        """
        Args:
            num_features: 输入特征数（如 OHLCV = 5）
            embed_dim: 嵌入维度
            patch_size: 每个 patch 包含的时间步数
            stride: 滑动步长，None 时等于 patch_size
        """
        pass
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # return: (batch, num_patches, embed_dim)
        pass


class CrossAttention:
    """
    交叉注意力模块
    
    用于融合高频分支和低频分支的信息
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        pass
    
    def forward(self, query, key_value):
        """
        Args:
            query: 高频分支输出 (batch, 1, embed_dim)
            key_value: 低频分支输出 (batch, seq_len, embed_dim)
        
        Returns:
            融合后的表征 (batch, 1, embed_dim)
        """
        pass


class DualBranchTemporalModule:
    """
    双分支时序模块
    
    包含高频和低频两个分支，通过交叉注意力融合
    """
    
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        patch_size_hf: int = 5,   # 高频分支 patch 大小
        patch_size_lf: int = 20,  # 低频分支 patch 大小
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        初始化双分支结构
        
        高频分支：小 patch，捕捉短期模式
        低频分支：大 patch，捕捉长期趋势
        """
        pass
    
    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch, seq_len, num_features)
        
        Returns:
            时序特征 (batch, embed_dim)
        
        流程:
            1. x → PatchEmbed_hf → GRU_hf → S_hf (取最后时间步)
            2. x → PatchEmbed_lf → GRU_lf → S_lf (全部时间步)
            3. CrossAttn(Q=S_hf, K=S_lf, V=S_lf) → S_attn
            4. S_ts = S_attn + S_hf (残差连接)
        """
        pass


class CrossSectionModule:
    """
    截面交互模块（可选）
    
    实现不同股票之间的信息交互
    适用于多股票联合预测场景
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_market_states: int = 8,  # 市场隐状态数量
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_market_states: 可学习的市场隐状态数量 (论文中的 c)
            num_heads: 注意力头数
        """
        pass
    
    def forward(self, x):
        """
        Args:
            x: 股票表征序列 (num_stocks, embed_dim)
        
        Returns:
            交互后的表征 (num_stocks, embed_dim)
        
        流程:
            1. R (市场隐状态) 作为 Q，x 作为 K,V → B
            2. x 作为 Q，B 作为 K,V → S_cs_attn
            3. 门控连接：S_cs = x + gate * S_cs_attn
        """
        pass


class FeatureInteractionModule:
    """
    特征交互模块 (FFN)
    
    实现特征维度的非线性变换
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        pass
    
    def forward(self, x):
        """
        S_out = BatchNorm(x + MLP(ReLU(MLP(x))))
        """
        pass


class CrossGRUModel:
    """
    CrossGRU-2 完整模型
    
    整合时序模块、截面模块、特征模块
    """
    
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        patch_size_hf: int = 5,
        patch_size_lf: int = 20,
        num_heads: int = 4,
        num_classes: int = 1,  # 1 表示回归，>1 表示分类
        use_cross_section: bool = False,  # 是否使用截面模块
        dropout: float = 0.1
    ):
        """
        完整模型初始化
        """
        pass
    
    def forward(self, x):
        """
        Args:
            x: 输入 (batch, seq_len, num_features)
               或多股票 (batch, num_stocks, seq_len, num_features)
        
        Returns:
            预测结果 (batch, num_classes) 或 (batch, num_stocks, num_classes)
        """
        pass


class CrossGRUTrainer:
    """
    CrossGRU-2 训练器
    
    继承 BaseTrainer 接口，实现 train/predict/save/load
    
    TODO: 具体实现参考 GRUTrainer
    """
    
    # 可配置参数
    init_params = [
        # Field("embed_dim", FieldType.INT, "嵌入维度", default=64),
        # Field("hidden_size", FieldType.INT, "GRU隐藏层大小", default=64),
        # Field("num_layers", FieldType.INT, "GRU层数", default=2),
        # Field("patch_size_hf", FieldType.INT, "高频分支Patch大小", default=5),
        # Field("patch_size_lf", FieldType.INT, "低频分支Patch大小", default=20),
        # Field("num_heads", FieldType.INT, "注意力头数", default=4),
        # Field("use_cross_section", FieldType.BOOL, "是否使用截面模块", default=False),
        # Field("dropout", FieldType.FLOAT, "Dropout比例", default=0.1),
        # Field("learning_rate", FieldType.FLOAT, "学习率", default=0.001),
        # Field("batch_size", FieldType.INT, "批大小", default=32),
        # Field("epochs", FieldType.INT, "训练轮数", default=100),
        # Field("window_size", FieldType.INT, "滑动窗口大小", default=60),
    ]
    
    def train(self, X, y, validation_data=None):
        """训练模型"""
        pass
    
    def predict(self, X):
        """预测"""
        pass
    
    def save_model(self, path):
        """保存模型"""
        pass
    
    def load_model(self, path):
        """加载模型"""
        pass


# ============================================================================
# 实现备忘
# ============================================================================
"""
1. 数据预处理注意事项：
   - 确保 seq_len >= max(patch_size_hf, patch_size_lf) * 2
   - 特征标准化很重要
   - 考虑使用 LayerNorm 而不是 BatchNorm（时序数据）

2. 训练技巧：
   - 使用 AdamW 优化器
   - 学习率 warmup + cosine decay
   - gradient clipping
   - early stopping

3. 调参优先级：
   - patch_size_hf, patch_size_lf（最重要）
   - embed_dim, hidden_size
   - num_layers, num_heads
   - dropout, learning_rate

4. 评估指标：
   - 分类：准确率、F1、RankIC
   - 回归：MSE、MAE、IC、RankIC

5. 消融实验顺序：
   - 基础 GRU（baseline）
   - GRU + Patch（单分支）
   - GRU + 双分支（无交叉注意力）
   - GRU + 双分支 + 交叉注意力
   - 完整 CrossGRU-2
"""
