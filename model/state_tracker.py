"""
state_tracker.py —— 历史状态追踪器

模块职责：
  HistoryEncoder : 将单步动作 (action, src, tgt, label_seq) 编码为固定维向量
  StateTracker   : GRU 聚合历史序列 + 融合图特征 → decoder_state

数据流：
  history = [e_0, ..., e_{t-1}]
    每步 e_i = (action_type, src_idx, tgt_idx, label_seq)
      → HistoryEncoder → step_repr [B, T, H]
      → GRU            → hist_context [B, H]
    + graph_emb [B, H]
      → fusion_proj    → decoder_state [B, H]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
#  数据容器
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HistoryBatch:
    """
    历史动作序列（已执行的编辑步骤）

    形状约定（T = max_hist_len, L = max_label_len）：
      actions    : [B, T]     action_type ids，padding = pad_action_id
      src_idxs   : [B, T]     src 原子索引，无效步 = -1
      tgt_idxs   : [B, T]     tgt 原子索引，无效步 = -1
      label_seqs : [B, T, L]  label token ids，padding = pad_token_id

    第 0 步（无历史）：T=0 的空张量，StateTracker 自动处理。
    """
    actions    : torch.Tensor   # [B, T]
    src_idxs   : torch.Tensor   # [B, T]
    tgt_idxs   : torch.Tensor   # [B, T]
    label_seqs : torch.Tensor   # [B, T, L]

    @classmethod
    def empty(cls, batch_size: int, label_len: int, device: torch.device) -> "HistoryBatch":
        """构造第 0 步的空历史（T=0）"""
        return cls(
            actions    = torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            src_idxs   = torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            tgt_idxs   = torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            label_seqs = torch.zeros(batch_size, 0, label_len, dtype=torch.long, device=device),
        )


# ══════════════════════════════════════════════════════════════════════
#  HistoryEncoder：单步动作多模态编码
# ══════════════════════════════════════════════════════════════════════

class HistoryEncoder(nn.Module):
    """
    将单步动作 (action, src, tgt, label_seq) 编码为固定维向量

    编码策略：
      action_type → Embedding                    [B, T, H]
      src_idx     → Embedding (clamp -1 → pad)  [B, T, H]
      tgt_idx     → Embedding (clamp -1 → pad)  [B, T, H]  共享 atom_emb
      label_seq   → TokenEmb + MaskedMeanPool   [B, T, H]
      四路拼接 [B, T, 4H] → Linear + ReLU + LayerNorm → [B, T, H]
    """
    def __init__(
        self,
        hidden_dim    : int,
        num_actions   : int,
        max_atoms     : int,
        vocab_size    : int,
        pad_action_id : int,
        pad_atom_id   : int,
        pad_token_id  : int,
    ):
        super().__init__()
        self.pad_atom_id  = pad_atom_id
        self.pad_token_id = pad_token_id

        self.action_emb      = nn.Embedding(num_actions + 1, hidden_dim,
                                            padding_idx=pad_action_id)
        # src / tgt 共享 atom_emb（语义相同，节省参数）
        self.atom_emb        = nn.Embedding(max_atoms + 1, hidden_dim,
                                            padding_idx=pad_atom_id)
        self.label_token_emb = nn.Embedding(vocab_size, hidden_dim,
                                            padding_idx=pad_token_id)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        actions    : torch.Tensor,   # [B, T]
        src_idxs   : torch.Tensor,   # [B, T]
        tgt_idxs   : torch.Tensor,   # [B, T]
        label_seqs : torch.Tensor,   # [B, T, L]
    ) -> torch.Tensor:               # → [B, T, H]

        # ── 原子索引：将 -1 (无效) 替换为 pad_atom_id ──────────────
        src_safe = src_idxs.clamp(min=0).masked_fill(src_idxs < 0, self.pad_atom_id)
        tgt_safe = tgt_idxs.clamp(min=0).masked_fill(tgt_idxs < 0, self.pad_atom_id)

        act_e = self.action_emb(actions)    # [B, T, H]
        src_e = self.atom_emb(src_safe)     # [B, T, H]
        tgt_e = self.atom_emb(tgt_safe)     # [B, T, H]

        # ── label_seq：Masked Mean Pooling ──────────────────────────
        B, T, L   = label_seqs.shape
        lbl_flat  = label_seqs.view(B * T, L)                            # [B*T, L]
        lbl_emb   = self.label_token_emb(lbl_flat)                       # [B*T, L, H]
        lbl_mask  = (lbl_flat != self.pad_token_id).unsqueeze(-1).float()# [B*T, L, 1]
        lbl_e     = ((lbl_emb * lbl_mask).sum(1)
                     / lbl_mask.sum(1).clamp(min=1)).view(B, T, -1)     # [B, T, H]

        # ── 四路融合 ────────────────────────────────────────────────
        return self.fusion(
            torch.cat([act_e, src_e, tgt_e, lbl_e], dim=-1)             # [B, T, 4H]
        )                                                                 # [B, T, H]


# ══════════════════════════════════════════════════════════════════════
#  StateTracker：GRU 聚合 + 图特征融合
# ══════════════════════════════════════════════════════════════════════

class StateTracker(nn.Module):
    """
    状态追踪器

    训练模式：history 含完整 T 步，一次性 GRU 前向
    推理模式：每步传入单步 history（T=1）+ gru_hidden，增量更新

    返回：
      decoder_state : [B, H]
      gru_hidden    : [1, B, H]  推理时传给下一步
    """
    def __init__(
        self,
        hidden_dim    : int,
        num_actions   : int,
        max_atoms     : int,
        vocab_size    : int,
        pad_action_id : int,
        pad_atom_id   : int,
        pad_token_id  : int,
    ):
        super().__init__()
        self.history_encoder = HistoryEncoder(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            max_atoms=max_atoms,
            vocab_size=vocab_size,
            pad_action_id=pad_action_id,
            pad_atom_id=pad_atom_id,
            pad_token_id=pad_token_id,
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # 图特征 + 历史上下文 → decoder_state
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        graph_emb  : torch.Tensor,                    # [B, H]
        history    : HistoryBatch,
        gru_hidden : Optional[torch.Tensor] = None,   # [1, B, H]
    ):
        B = graph_emb.size(0)
        T = history.actions.size(1)

        if T == 0:
            # 第 0 步：无历史，hist_context 为零向量
            hist_context = torch.zeros_like(graph_emb)
            gru_hidden   = torch.zeros(1, B, graph_emb.size(-1),
                                       device=graph_emb.device)
        else:
            step_repr  = self.history_encoder(
                history.actions, history.src_idxs,
                history.tgt_idxs, history.label_seqs,
            )                                          # [B, T, H]
            gru_out, gru_hidden = self.gru(step_repr, gru_hidden)
            hist_context = gru_out[:, -1, :]          # [B, H] 最后有效步

        decoder_state = self.fusion_proj(
            torch.cat([graph_emb, hist_context], dim=-1)  # [B, 2H]
        )
        return decoder_state, gru_hidden