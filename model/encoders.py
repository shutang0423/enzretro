from __future__ import annotations
from typing import NamedTuple

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, GCNConv, GINConv, SAGEConv, TransformerConv,
    global_mean_pool, global_add_pool,
)
from torch_geometric.utils import to_dense_batch   # ← 关键工具函数
from config.config import MODEL_CFG


# ══════════════════════════════════════════════════════════════════════
#  返回值类型（完整版，供 pointer_network 使用）
# ══════════════════════════════════════════════════════════════════════

class EncoderOutput(NamedTuple):
    node_emb     : torch.Tensor   # [N, node_dim]       稀疏节点表示
    graph_emb    : torch.Tensor   # [B, node_dim]       图级表示
    dense_nodes  : torch.Tensor   # [B, max_atoms, d]   稠密节点矩阵（含 padding）
    node_pad_mask: torch.Tensor   # [B, max_atoms]      True = padding 位（无效）
    has_nodes    : torch.Tensor   # [B]  bool           该图是否有节点


# ══════════════════════════════════════════════════════════════════════
#  GNN 基类
# ══════════════════════════════════════════════════════════════════════

class BaseGNNEncoder(nn.Module):
    """
    所有 GNN 编码器的公共基类。
    forward 返回完整 EncoderOutput，包含：
      - 稀疏节点表示 node_emb      [N, d]
      - 图级表示     graph_emb     [B, d]
      - 稠密节点矩阵 dense_nodes   [B, max_atoms, d]
      - padding 掩码 node_pad_mask [B, max_atoms]  True=padding
      - 有效标志     has_nodes     [B] bool
    """

    def __init__(
        self,
        node_in_dim : int,
        node_dim    : int,
        num_layers  : int,
        dropout     : float = 0.1,
        residual    : bool  = True,
        pooling     : str   = "mean",
    ):
        super().__init__()
        self.residual = residual
        self.dropout  = nn.Dropout(dropout)
        self.pool_fn  = global_mean_pool if pooling == "mean" else global_add_pool

        self.input_proj = (
            nn.Linear(node_in_dim, node_dim)
            if node_in_dim != node_dim else nn.Identity()
        )
        self.convs = self._build_conv_layers(node_dim, num_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        self.acts  = nn.ModuleList([nn.GELU()              for _ in range(num_layers)])

    def _build_conv_layers(self, node_dim: int, num_layers: int) -> nn.ModuleList:
        raise NotImplementedError

    def _conv_forward(self, conv: nn.Module, x: torch.Tensor,
                      edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x          : torch.Tensor,   # [N, node_in_dim]
        edge_index : torch.Tensor,   # [2, E]
        batch      : torch.Tensor,   # [N]  节点→图映射
    ) -> EncoderOutput:
        # ── 1. GNN 消息传递 ──────────────────────────────────────────
        x = self.input_proj(x)
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            h = self._conv_forward(conv, x, edge_index)
            h = norm(h)
            h = act(h)
            h = self.dropout(h)
            x = (x + h) if self.residual else h

        node_emb  = x                           # [N, node_dim]
        graph_emb = self.pool_fn(x, batch)      # [B, node_dim]

        # ── 2. 稀疏 → 稠密，供指针网络使用 ──────────────────────────
        # to_dense_batch: [N,d] + batch → [B, max_nodes, d], mask [B, max_nodes]
        # mask: True = 有效节点，False = padding
        dense_nodes, valid_mask = to_dense_batch(node_emb, batch)  # valid=True 有效

        # pointer_network 习惯用 True=padding，取反
        node_pad_mask = ~valid_mask              # [B, max_atoms]  True=padding

        # ── 3. has_nodes：每个图至少有 1 个有效节点 ──────────────────
        has_nodes = valid_mask.any(dim=-1)       # [B]  bool

        return EncoderOutput(
            node_emb      = node_emb,
            graph_emb     = graph_emb,
            dense_nodes   = dense_nodes,
            node_pad_mask = node_pad_mask,
            has_nodes     = has_nodes,
        )


# ══════════════════════════════════════════════════════════════════════
#  具体 GNN 编码器（子类只需实现两个方法，forward 完全继承）
# ══════════════════════════════════════════════════════════════════════

class GATEncoder(BaseGNNEncoder):
    def __init__(self, node_in_dim, node_dim, num_layers, heads=4, **kwargs):
        assert node_dim % heads == 0, f"node_dim({node_dim}) 须整除 heads({heads})"
        self._heads = heads
        super().__init__(node_in_dim, node_dim, num_layers, **kwargs)

    def _build_conv_layers(self, node_dim, num_layers):
        head_dim = node_dim // self._heads
        return nn.ModuleList([
            GATConv(node_dim, head_dim, heads=self._heads, concat=True)
            for _ in range(num_layers)
        ])

    def _conv_forward(self, conv, x, edge_index):
        return conv(x, edge_index)


class GCNEncoder(BaseGNNEncoder):
    def _build_conv_layers(self, node_dim, num_layers):
        return nn.ModuleList([
            GCNConv(node_dim, node_dim) for _ in range(num_layers)
        ])

    def _conv_forward(self, conv, x, edge_index):
        return conv(x, edge_index)


class GINEncoder(BaseGNNEncoder):
    def _build_conv_layers(self, node_dim, num_layers):
        def _mlp():
            return nn.Sequential(
                nn.Linear(node_dim, node_dim * 2),
                nn.BatchNorm1d(node_dim * 2),
                nn.GELU(),
                nn.Linear(node_dim * 2, node_dim),
            )
        return nn.ModuleList([
            GINConv(_mlp(), train_eps=True) for _ in range(num_layers)
        ])

    def _conv_forward(self, conv, x, edge_index):
        return conv(x, edge_index)


class GraphSAGEEncoder(BaseGNNEncoder):
    def _build_conv_layers(self, node_dim, num_layers):
        return nn.ModuleList([
            SAGEConv(node_dim, node_dim, normalize=True)
            for _ in range(num_layers)
        ])

    def _conv_forward(self, conv, x, edge_index):
        return conv(x, edge_index)


class GraphTransformerEncoder(BaseGNNEncoder):
    def __init__(self, node_in_dim, node_dim, num_layers, heads=4, **kwargs):
        assert node_dim % heads == 0
        self._heads = heads
        super().__init__(node_in_dim, node_dim, num_layers, **kwargs)

    def _build_conv_layers(self, node_dim, num_layers):
        head_dim = node_dim // self._heads
        return nn.ModuleList([
            TransformerConv(node_dim, head_dim, heads=self._heads, concat=True)
            for _ in range(num_layers)
        ])

    def _conv_forward(self, conv, x, edge_index):
        return conv(x, edge_index)


# ══════════════════════════════════════════════════════════════════════
#  工厂函数
# ══════════════════════════════════════════════════════════════════════

_ENCODER_REGISTRY: dict[str, type] = {
    "gat"        : GATEncoder,
    "gcn"        : GCNEncoder,
    "gin"        : GINEncoder,
    "sage"       : GraphSAGEEncoder,
    "transformer": GraphTransformerEncoder,
}


def build_encoder(cfg=MODEL_CFG) -> BaseGNNEncoder:
    t = cfg.encoder_type.lower()
    if t not in _ENCODER_REGISTRY:
        raise ValueError(
            f"未知 encoder_type='{t}'，可选: {list(_ENCODER_REGISTRY.keys())}"
        )
    common = dict(
        node_in_dim = cfg.node_in_dim,
        node_dim    = cfg.node_dim,
        num_layers  = cfg.num_layers,
        dropout     = cfg.gnn_dropout,
        residual    = cfg.gnn_residual,
        pooling     = cfg.gnn_pooling,
    )
    if t in ("gat", "transformer"):
        common["heads"] = cfg.gnn_heads

    return _ENCODER_REGISTRY[t](**common)