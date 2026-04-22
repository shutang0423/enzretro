"""
prediction_heads.py —— 三大预测头

ActionTypePredictor : decoder_state → action_logits [B, num_actions]
PointerNetwork      : decoder_state + nodes → src_logits, tgt_logits [B, max_nodes]
LabelDecoder        : decoder_state + action + tgt_seq → label_logits [B, L, vocab_size]

消融说明：
  FingerprintEncoder 时 has_nodes=False，PointerNetwork 的 forward 返回全 -inf logits，
  训练时对应 loss 权重应设为 0（由 loss_strategy 控制）。
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
#  Action Type Predictor
# ══════════════════════════════════════════════════════════════════════

class ActionTypePredictor(nn.Module):
    """
    动作类型分类头（7分类 MLP）
    输入: decoder_state [B, H]
    输出: action_logits [B, num_actions]
    """
    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, decoder_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(decoder_state)   # [B, num_actions]


# ══════════════════════════════════════════════════════════════════════
#  Pointer Network
# ══════════════════════════════════════════════════════════════════════

class PointerNetwork(nn.Module):
    """
    注意力指针网络：预测操作的 src / tgt 原子索引

    数据流：
      state + action_emb → Q_src → 与 K_src 点积 → src_logits [B, max_nodes]
      state + action_emb + src_emb → Q_tgt → 与 K_tgt 点积 → tgt_logits [B, max_nodes]

    Teacher Forcing：训练时传 target_src_idx，推理时用 argmax
    消融兼容：has_nodes=False 时直接返回全 -inf（不参与梯度）
    """
    def __init__(
        self,
        hidden_dim  : int,
        node_dim    : int,
        num_actions : int,
        max_atoms   : int,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.src_emb    = nn.Embedding(max_atoms,   hidden_dim)
        self.node_proj  = nn.Linear(node_dim, hidden_dim)

        self.q_src = nn.Linear(hidden_dim, hidden_dim)
        self.k_src = nn.Linear(hidden_dim, hidden_dim)
        self.q_tgt = nn.Linear(hidden_dim, hidden_dim)
        self.k_tgt = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        decoder_state   : torch.Tensor,             # [B, H]
        dense_nodes     : torch.Tensor,             # [B, max_nodes, node_dim]
        action_type     : torch.Tensor,             # [B]
        target_src_idx  : Optional[torch.Tensor] = None,  # [B] Teacher Forcing
        node_mask       : Optional[torch.Tensor] = None,  # [B, max_nodes] True=padding
        has_nodes       : bool = True,
    ):
        """
        返回: src_logits [B, max_nodes], tgt_logits [B, max_nodes]
        has_nodes=False 时返回全 -inf（FingerprintEncoder 消融用）
        """
        if not has_nodes:
            # 指纹模式：无节点概念，返回占位 logits（不参与有效 loss 计算）
            dummy = torch.full(
                (decoder_state.size(0), 1), -1e9,
                device=decoder_state.device
            )
            return dummy, dummy

        act_e      = self.action_emb(action_type)            # [B, H]
        nodes_proj = self.node_proj(dense_nodes)             # [B, max_nodes, H]

        # ── src ──────────────────────────────────────────────────────
        q_src      = self.q_src(decoder_state + act_e).unsqueeze(1)  # [B, 1, H]
        k_src      = self.k_src(nodes_proj)                          # [B, max_nodes, H]
        src_logits = torch.bmm(q_src, k_src.transpose(1, 2)).squeeze(1)  # [B, max_nodes]
        if node_mask is not None:
            src_logits = src_logits.masked_fill(node_mask, -1e9)

        # Teacher Forcing / 推理 argmax
        src_idx = (target_src_idx if target_src_idx is not None
                   else src_logits.argmax(dim=-1))
        src_idx = src_idx.clamp(0, self.src_emb.num_embeddings - 1)
        src_e   = self.src_emb(src_idx)                              # [B, H]

        # ── tgt ──────────────────────────────────────────────────────
        q_tgt      = self.q_tgt(decoder_state + act_e + src_e).unsqueeze(1)
        k_tgt      = self.k_tgt(nodes_proj)
        tgt_logits = torch.bmm(q_tgt, k_tgt.transpose(1, 2)).squeeze(1)
        if node_mask is not None:
            tgt_logits = tgt_logits.masked_fill(node_mask, -1e9)

        return src_logits, tgt_logits


# ══════════════════════════════════════════════════════════════════════
#  Label Decoder
# ══════════════════════════════════════════════════════════════════════

class LabelDecoder(nn.Module):
    """
    标签序列解码器（Transformer Decoder，自回归）

    训练：tgt_seq 为完整序列（含 BOS），Teacher Forcing
    推理：greedy_decode() 逐 token 生成直到 EOS 或 max_len

    输入: decoder_state [B, H], action_type [B], tgt_seq [B, L]
    输出: label_logits  [B, L, vocab_size]
    """
    def __init__(
        self,
        vocab_size  : int,
        hidden_dim  : int,
        num_actions : int,
        max_pos_enc : int,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.action_emb   = nn.Embedding(num_actions, hidden_dim)
        self.token_emb    = nn.Embedding(vocab_size,  hidden_dim)
        self.pos_enc      = nn.Embedding(max_pos_enc, hidden_dim)
        decoder_layer     = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1,
        )
        self.transformer  = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc_out       = nn.Linear(hidden_dim, vocab_size)

    def _build_memory(
        self,
        decoder_state : torch.Tensor,   # [B, H]
        action_type   : torch.Tensor,   # [B]
    ) -> torch.Tensor:                  # → [B, 1, H]
        """构建 Transformer Decoder 的 memory（状态 + 动作类型）"""
        act_e  = self.action_emb(action_type).unsqueeze(1)   # [B, 1, H]
        return decoder_state.unsqueeze(1) + act_e            # [B, 1, H]

    def forward(
        self,
        decoder_state : torch.Tensor,   # [B, H]
        action_type   : torch.Tensor,   # [B]
        tgt_seq       : torch.Tensor,   # [B, L]  含 BOS，不含 EOS
    ) -> torch.Tensor:                  # → [B, L, vocab_size]
        """训练前向（Teacher Forcing）"""
        B, L   = tgt_seq.shape
        memory = self._build_memory(decoder_state, action_type)  # [B, 1, H]
        pos    = torch.arange(L, device=tgt_seq.device).unsqueeze(0).expand(B, L)
        tgt_e  = self.token_emb(tgt_seq) + self.pos_enc(pos)     # [B, L, H]
        causal = nn.Transformer.generate_square_subsequent_mask(
            L, device=tgt_seq.device
        )
        out    = self.transformer(tgt=tgt_e, memory=memory, tgt_mask=causal)
        return self.fc_out(out)                                   # [B, L, vocab_size]

    @torch.no_grad()
    def greedy_decode(
        self,
        decoder_state : torch.Tensor,   # [B, H]
        action_type   : torch.Tensor,   # [B]
        bos_token_id  : int,
        max_len       : int = 32,
    ) -> torch.Tensor:                  # → [B, max_len]  含 BOS，遇 EOS 后补 pad
        """
        推理：贪心自回归解码
        逐 token 生成，遇到 EOS 停止
        """
        B      = decoder_state.size(0)
        memory = self._build_memory(decoder_state, action_type)   # [B, 1, H]
        # 初始化：只有 BOS
        generated = torch.full((B, 1), bos_token_id,
                                dtype=torch.long, device=decoder_state.device)
        finished  = torch.zeros(B, dtype=torch.bool, device=decoder_state.device)

        for step in range(max_len - 1):
            L   = generated.size(1)
            pos = torch.arange(L, device=generated.device).unsqueeze(0).expand(B, L)
            tgt_e  = self.token_emb(generated) + self.pos_enc(pos)
            causal = nn.Transformer.generate_square_subsequent_mask(
                L, device=generated.device
            )
            out    = self.transformer(tgt=tgt_e, memory=memory, tgt_mask=causal)
            logits = self.fc_out(out[:, -1, :])                    # [B, vocab_size]
            next_token = logits.argmax(dim=-1, keepdim=True)       # [B, 1]

            # 已结束的序列继续输出 EOS（不影响后续 mask）
            next_token = next_token.masked_fill(
                finished.unsqueeze(-1), self.eos_token_id
            )
            generated = torch.cat([generated, next_token], dim=1)  # [B, step+2]
            finished  = finished | (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break

        return generated   # [B, actual_len]