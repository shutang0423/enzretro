"""Module 4+5: Critic Network (独立于Actor)

V(s)   : 状态价值 → GAE 优势估计
Q(s,a) : 动作价值 → 评估复合动作质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class CriticEncoder(nn.Module):
    """Critic 专用轻量图编码器"""

    def __init__(self, node_in_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch):
        h = self.proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h
        return global_mean_pool(h, batch)


class CriticNetwork(nn.Module):
    """独立 Critic: V(s) + Q(s,a) 双头"""

    def __init__(self, cfg: dict):
        super().__init__()
        hd = cfg["hidden_dim"]
        na = cfg["num_actions"]
        pad = cfg["pad_action_id"]

        self.encoder = CriticEncoder(cfg["node_in_dim"], hd, num_layers=3)

        # 历史编码 (GRU)
        self.hist_emb = nn.Embedding(na + 1, hd, padding_idx=pad)
        self.hist_gru = nn.GRU(hd, hd, batch_first=True)

        # V(s) 头
        self.value_head = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Linear(hd // 2, 1),
        )

        # Q(s,a) 头
        self.action_emb = nn.Embedding(na, hd)
        self.q_head = nn.Sequential(
            nn.Linear(hd * 2, hd), nn.ReLU(), nn.Linear(hd, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.value_head, self.q_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)

    def _encode_state(self, x, edge_index, batch, history_actions):
        graph_emb = self.encoder(x, edge_index, batch)
        if history_actions is not None:
            hist_seq = self.hist_emb(history_actions)
            h0 = graph_emb.unsqueeze(0)
            _, hn = self.hist_gru(hist_seq, h0)
            return hn.squeeze(0) + graph_emb  # 残差
        return graph_emb

    def get_value(self, x, edge_index, batch, history_actions=None):
        state = self._encode_state(x, edge_index, batch, history_actions)
        return self.value_head(state)

    def get_q_value(self, x, edge_index, batch, action_type, history_actions=None):
        state = self._encode_state(x, edge_index, batch, history_actions)
        act = self.action_emb(action_type)
        return self.q_head(torch.cat([state, act], dim=-1))

    def forward(self, x, edge_index, batch, action_type=None, history_actions=None):
        state = self._encode_state(x, edge_index, batch, history_actions)
        v = self.value_head(state)
        q = None
        if action_type is not None:
            act = self.action_emb(action_type)
            q = self.q_head(torch.cat([state, act], dim=-1))
        return v, q