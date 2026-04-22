"""
encoders.py —— 可插拔分子编码器

消融实验 1
  GATEncoder         : 图注意力网络，输出节点嵌入 + 图嵌入
  FingerprintEncoder : Morgan 指纹 + MLP，仅输出图嵌入（无节点概念）

统一输出接口 EncoderOutput，下游模块无感知切换。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from config.config import ModelConfig


# ══════════════════════════════════════════════════════════════════════
#  统一输出容器
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EncoderOutput:
    """
    所有 Encoder 的统一输出格式

    graph_emb     : [B, node_dim]            图级别嵌入（必有）
    dense_nodes   : [B, max_nodes, node_dim] 节点嵌入 Dense Batch（GAT有，指纹为零）
    node_pad_mask : [B, max_nodes]           True=padding（GAT有，指纹全False）
    has_nodes     : bool                     是否有有效节点嵌入（消融时判断）
    """
    graph_emb     : torch.Tensor
    dense_nodes   : torch.Tensor
    node_pad_mask : torch.Tensor
    has_nodes     : bool = True


# ══════════════════════════════════════════════════════════════════════
#  抽象基类
# ══════════════════════════════════════════════════════════════════════

class MoleculeEncoderBase(ABC, nn.Module):
    """所有分子编码器的基类，强制统一输出接口"""

    @abstractmethod
    def forward(self, **kwargs) -> EncoderOutput:
        """子类实现各自的编码逻辑，统一返回 EncoderOutput"""
        ...

    @property
    @abstractmethod
    def node_dim(self) -> int:
        """输出的节点/图嵌入维度"""
        ...


# ══════════════════════════════════════════════════════════════════════
#  实现 1：GAT 图编码器（主方案）
# ══════════════════════════════════════════════════════════════════════

class GATEncoder(MoleculeEncoderBase):
    """
    图注意力网络编码器

    输入: x [N, node_in_dim], edge_index [2, E], batch [N]
    输出: EncoderOutput
      - graph_emb     [B, node_dim]
      - dense_nodes   [B, max_nodes, node_dim]
      - node_pad_mask [B, max_nodes]  True=padding
      - has_nodes     = True
    """
    def __init__(
        self,
        node_in_dim : int,
        node_dim    : int,
        num_layers  : int = 4,
        gat_heads   : int = 4,
    ):
        super().__init__()
        self._node_dim  = node_dim
        self.node_proj  = nn.Linear(node_in_dim, node_dim)
        self.convs       = nn.ModuleList([
            GATConv(node_dim, node_dim, heads=gat_heads, concat=False)
            for _ in range(num_layers)
        ])
        self.graph_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )

    @property
    def node_dim(self) -> int:
        return self._node_dim

    def forward(
        self,
        x          : torch.Tensor,   # [N, node_in_dim]
        edge_index : torch.Tensor,   # [2, E]
        batch      : torch.Tensor,   # [N]
        **kwargs,
    ) -> EncoderOutput:
        h = self.node_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h          # 残差

        graph_emb   = self.graph_pool(global_mean_pool(h, batch))  # [B, node_dim]
        dense_nodes, node_mask = to_dense_batch(h, batch)          # [B, max_N, D], [B, max_N]
        node_pad_mask = ~node_mask                                  # True = padding

        return EncoderOutput(
            graph_emb=graph_emb,
            dense_nodes=dense_nodes,
            node_pad_mask=node_pad_mask,
            has_nodes=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  实现 2：分子指纹编码器（消融对照组）
# ══════════════════════════════════════════════════════════════════════

class FingerprintEncoder(MoleculeEncoderBase):
    """
    分子指纹编码器（消融实验对照组）

    使用预计算的 Morgan/ECFP 指纹（由 Dataset 提供），
    通过 MLP 映射到与 GATEncoder 相同的 node_dim 空间。

    输入: fingerprint [B, fp_dim]  (float, 0/1 二值或连续)
    输出: EncoderOutput
      - graph_emb     [B, node_dim]
      - dense_nodes   [B, 1, node_dim]  零占位（Pointer 退化）
      - node_pad_mask [B, 1]            全 True（Pointer 全部 mask）
      - has_nodes     = False           下游可据此跳过 Pointer 任务
    """
    def __init__(
        self,
        fp_dim   : int,
        node_dim : int,
    ):
        super().__init__()
        self._node_dim = node_dim
        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, node_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.LayerNorm(node_dim),
        )

    @property
    def node_dim(self) -> int:
        return self._node_dim

    def forward(
        self,
        fingerprint : torch.Tensor,   # [B, fp_dim]
        **kwargs,
    ) -> EncoderOutput:
        B          = fingerprint.size(0)
        graph_emb  = self.mlp(fingerprint)                          # [B, node_dim]

        # 占位节点：Pointer Network 在 has_nodes=False 时应跳过
        dense_nodes   = torch.zeros(B, 1, self._node_dim, device=fingerprint.device)
        node_pad_mask = torch.ones(B, 1, dtype=torch.bool, device=fingerprint.device)

        return EncoderOutput(
            graph_emb=graph_emb,
            dense_nodes=dense_nodes,
            node_pad_mask=node_pad_mask,
            has_nodes=False,
        )


# ══════════════════════════════════════════════════════════════════════
#  工厂函数：根据 config 自动选择 Encoder
# ══════════════════════════════════════════════════════════════════════

# encoders.py 的工厂函数同步修改
def build_encoder(cfg: ModelConfig):
    if cfg.encoder_type == "gat":
        return GATEncoder(
            node_in_dim = cfg.node_in_dim,
            node_dim    = cfg.node_dim,
            num_layers  = cfg.gat_layers,
            gat_heads   = cfg.gat_heads,
        )
    elif cfg.encoder_type == "fingerprint":
        return FingerprintEncoder(fp_dim=cfg.fp_dim, node_dim=cfg.node_dim)
