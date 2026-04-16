"""Module 2: State Tracker — 维护编辑历史的时序状态"""

import torch
import torch.nn as nn


class GRUStateTracker(nn.Module):
    """基于 GRU 的状态追踪器

    核心改进: 用 GRU 替代简单求和，保留动作的时序顺序信息。

    输入:
        graph_embedding : [B, hidden_dim]   图级嵌入(已投影)
        history_actions : [B, T]            历史动作序列
    输出:
        state : [B, hidden_dim]
    """

    def __init__(self, hidden_dim: int, num_actions: int,
                 pad_action_id: int, num_layers: int = 2):
        super().__init__()
        self.act_emb = nn.Embedding(
            num_actions + 1, hidden_dim, padding_idx=pad_action_id
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, history_actions, graph_embedding):
        B = graph_embedding.size(0)
        act_seq = self.act_emb(history_actions)                # [B, T, h]

        # 用 graph_embedding 初始化 GRU 隐状态
        h0 = graph_embedding.unsqueeze(0).expand(
            self.gru.num_layers, B, -1
        ).contiguous()                                         # [layers, B, h]

        output, _ = self.gru(act_seq, h0)                     # [B, T, h]
        last_hidden = output[:, -1, :]                         # [B, h]
        return self.out_proj(last_hidden) + graph_embedding    # 残差连接


class SimpleStateTracker(nn.Module):
    """简单求和版本 (向后兼容)"""

    def __init__(self, hidden_dim: int, num_actions: int, pad_action_id: int):
        super().__init__()
        self.act_emb = nn.Embedding(
            num_actions + 1, hidden_dim, padding_idx=pad_action_id
        )

    def forward(self, history_actions, graph_embedding):
        return graph_embedding + self.act_emb(history_actions).sum(dim=1)