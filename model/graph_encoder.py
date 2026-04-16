import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from config.config import MODEL_CONFIG as C


class GraphEncoder(nn.Module):
    """产物图编码器 (基于 GAT)"""
    def __init__(self, node_in_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch):
        h = self.node_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h
        node_embeddings = h
        graph_emb = self.graph_pool(global_mean_pool(node_embeddings, batch))
        return node_embeddings, graph_emb
