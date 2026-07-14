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
from typing import Optional, Tuple

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
    多头边感知指针网络：预测操作的 src / tgt 原子索引

    改进点（相对原始单头版本）：
      1. 多 Head 注意力：hidden_dim 拆为 num_heads 组，并行捕捉不同化学上下文
      2. 边感知偏置：邻接矩阵作为注意力偏置，键合原子获得更高 logit
      3. 双线性 tgt 打分：Bilinear(src_node_feat, candidate_node_feat)
         替代简单的 src_emb Embedding，显式建模原子对交互，减少级联错误
      4. Scheduled Sampling（可选）：训练时以概率 p 用预测 src 替代 GT src

    Teacher Forcing：训练时传 target_src_idx，推理时用 argmax
    消融兼容：has_nodes=False 时直接返回全 -inf（不参与梯度）

    越界防御：
      所有 Embedding 查表前 clamp 到合法范围，
      避免数据噪声触发 CUDA indexSelectLargeIndex assert
    """

    def __init__(
        self,
        hidden_dim  : int,
        node_dim    : int,
        num_actions : int,
        max_atoms   : int,
        num_heads   : int   = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim({hidden_dim}) 须整除 num_heads({num_heads})"
        self.num_actions = num_actions
        self.max_atoms   = max_atoms
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self._scale      = self.head_dim ** -0.5   # 按 head_dim 缩放

        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.src_emb    = nn.Embedding(max_atoms,   hidden_dim)
        self.node_proj  = nn.Linear(node_dim, hidden_dim)

        # ── 多 Head Q/K 投影 ─────────────────────────────────────────
        # src: query = [decoder_state; action_emb]
        self.q_src = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_src = nn.Linear(hidden_dim,     hidden_dim)

        # tgt: query = [decoder_state; action_emb]（双线性版本不再拼接 src_emb）
        self.q_tgt = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_tgt = nn.Linear(hidden_dim,     hidden_dim)

        # ── 双线性 src→tgt 交互矩阵 ──────────────────────────────────
        self.bilinear_src_tgt = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # ── 边偏置可学习缩放因子 ─────────────────────────────────────
        self.edge_bias_scale = nn.Parameter(torch.tensor(1.0))

        # ── Dropout ──────────────────────────────────────────────────
        self.attn_dropout = nn.Dropout(dropout)

    def _multi_head_attention(
        self,
        q_proj      : torch.Tensor,   # [B, H]  已通过 q_src/q_tgt 投影的 query
        k_proj      : torch.Tensor,   # [B, N, H]  已通过 k_src/k_tgt 投影的 key
        node_mask   : Optional[torch.Tensor] = None,  # [B, N] True=padding
        adj_bias    : Optional[torch.Tensor] = None,  # [B, N, N] 边偏置（仅 src 使用全矩阵，tgt 传入 [B, N] 行切片）
        is_tgt_bias : bool = False,  # True 时 adj_bias 是 [B, N] 行向量而非 [B, N, N] 矩阵
    ) -> torch.Tensor:                 # → [B, N]
        """
        多 Head 缩放点积注意力 + 可选边偏置

        Args:
          q_proj   : 已投影的 query 向量 [B, H]
          k_proj   : 已投影的 key 矩阵 [B, N, H]
          node_mask: True=padding [B, N]
          adj_bias : 边偏置，src 时 [B, N, N]，tgt 时 [B, N]
          is_tgt_bias: True 时 adj_bias 以 additive 方式加到 logits 上
        """
        B, N, H = k_proj.shape

        # 重塑为多头格式
        q_mh = q_proj.view(B, self.num_heads, self.head_dim)        # [B, h, d]
        k_mh = k_proj.view(B, N, self.num_heads, self.head_dim)     # [B, N, h, d]
        k_mh = k_mh.transpose(1, 2)                                  # [B, h, N, d]

        # 缩放点积
        attn = torch.matmul(
            q_mh.unsqueeze(2), k_mh.transpose(-1, -2)               # [B, h, 1, N]
        ) * self._scale  # [B, h, 1, N]

        # ── 边感知偏置 ──────────────────────────────────────────────
        if adj_bias is not None:
            if is_tgt_bias:
                # tgt: adj_bias 是 [B, N]，加到每个 head
                bias = adj_bias.unsqueeze(1).unsqueeze(2) * self.edge_bias_scale  # [B, 1, 1, N]
            else:
                # src: adj_bias 是 [B, N, N]，取 query 维度切片
                bias = adj_bias.unsqueeze(1)[:, :, :1, :] * self.edge_bias_scale  # [B, 1, 1, N]
            attn = attn + bias

        # ── Mask padding ────────────────────────────────────────────
        if node_mask is not None:
            attn = attn.masked_fill(
                node_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = self.attn_dropout(attn)

        # ── 多头平均聚合 ────────────────────────────────────────────
        logits = attn.mean(dim=1).squeeze(1)  # [B, N]
        return logits

    def forward(
        self,
        decoder_state            : torch.Tensor,                       # [B, H]
        dense_nodes              : torch.Tensor,                       # [B, max_nodes, node_dim]
        action_type              : torch.Tensor,                       # [B]
        target_src_idx           : Optional[torch.Tensor] = None,     # [B] Teacher Forcing
        node_mask                : Optional[torch.Tensor] = None,     # [B, max_nodes] True=padding
        adj_matrix               : Optional[torch.Tensor] = None,     # [B, max_nodes, max_nodes]
        has_nodes                : bool = True,
        scheduled_sampling_prob  : float = 0.0,  # Scheduled Sampling 概率（0.0=纯Teacher Forcing）
    ):
        """
        返回: src_logits [B, max_nodes], tgt_logits [B, max_nodes]
        has_nodes=False 时返回全 -inf（FingerprintEncoder 消融用）
        """
        B = decoder_state.size(0)

        # ── 兼容 has_nodes 是 Tensor 还是 bool ───────────────────────
        if isinstance(has_nodes, torch.Tensor):
            _has_nodes = has_nodes.any().item()
        else:
            _has_nodes = bool(has_nodes)

        if not _has_nodes:
            inf_logits = torch.full(
                (B, self.max_atoms), float("-inf"),
                device=decoder_state.device,
            )
            return inf_logits, inf_logits

        # ── 越界防御 ────────────────────────────────────────────────
        action_type_safe = action_type.clamp(0, self.num_actions - 1)
        a_emb = self.action_emb(action_type_safe)    # [B, H]

        # ── 节点投影 ────────────────────────────────────────────────
        node_feat = self.node_proj(dense_nodes)       # [B, N, H]

        # ══════════════════════════════════════════════════════════════
        #  src 预测（多头 + 边偏置）
        # ══════════════════════════════════════════════════════════════
        q_src = self.q_src(torch.cat([decoder_state, a_emb], dim=-1))  # [B, H]
        k_src = self.k_src(node_feat)                                    # [B, N, H]

        src_logits = self._multi_head_attention(
            q_proj    = q_src,
            k_proj    = k_src,
            node_mask = node_mask,
            adj_bias  = adj_matrix,   # [B, N, N] 全矩阵，键合原子对获得偏置
            is_tgt_bias = False,
        )

        # ══════════════════════════════════════════════════════════════
        #  tgt 预测（双线性打分 + 边偏置 + Scheduled Sampling）
        # ══════════════════════════════════════════════════════════════
        if target_src_idx is not None and self.training and scheduled_sampling_prob > 0:
            # Scheduled Sampling：以概率 p 使用预测的 src（而非 GT）
            use_pred = torch.rand(B, device=decoder_state.device) < scheduled_sampling_prob
            pred_src = src_logits.argmax(dim=-1).clamp(0, self.max_atoms - 1)
            src_idx_safe = torch.where(
                use_pred,
                pred_src,
                target_src_idx.clamp(0, self.max_atoms - 1),
            )
        elif target_src_idx is not None:
            src_idx_safe = target_src_idx.clamp(0, self.max_atoms - 1)
        else:
            src_idx_safe = src_logits.argmax(dim=-1).clamp(0, self.max_atoms - 1)

        # ── 双线性 src-tgt 交互打分 ──────────────────────────────────
        # 取出 src 对应的节点特征
        src_node_feat = node_feat[
            torch.arange(B, device=node_feat.device), src_idx_safe
        ]  # [B, H]

        # Bilinear(src_node, candidate_node): [B*N, H] × [B*N, H] → [B*N, H]
        N = node_feat.size(1)
        src_expanded = src_node_feat.unsqueeze(1).expand(-1, N, -1).reshape(-1, node_feat.size(-1))  # [B*N, H]
        candidates_flat = node_feat.reshape(-1, node_feat.size(-1))                                   # [B*N, H]
        bilinear_out = self.bilinear_src_tgt(src_expanded, candidates_flat)  # [B*N, H]
        bilinear_out = bilinear_out.view(B, N, -1)                            # [B, N, H]

        # 与 query 交互得到最终 tgt logits
        q_tgt = self.q_tgt(torch.cat([decoder_state, a_emb], dim=-1))  # [B, H]
        tgt_logits = (q_tgt.unsqueeze(1) * bilinear_out).sum(dim=-1)    # [B, N]
        tgt_logits = tgt_logits * self._scale

        # ── 边偏置：src 原子的邻居获得额外 logit ─────────────────────
        if adj_matrix is not None:
            tgt_edge_bias = adj_matrix[
                torch.arange(B, device=adj_matrix.device), src_idx_safe, :
            ] * self.edge_bias_scale  # [B, N]
            tgt_logits = tgt_logits + tgt_edge_bias

        # ── Mask padding ────────────────────────────────────────────
        if node_mask is not None:
            tgt_logits = tgt_logits.masked_fill(node_mask, float("-inf"))

        return src_logits, tgt_logits   # [B, max_nodes], [B, max_nodes]


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

    @torch.no_grad()
    def sample_decode(
        self,
        decoder_state : torch.Tensor,   # [B, H]
        action_type   : torch.Tensor,   # [B]
        bos_token_id  : int,
        max_len       : int,
        temperature   : float = 1.0,
        top_k         : int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自回归采样解码 label 序列（用于 Top-N 生成）
        
        Args:
            decoder_state: 解码器状态 [B, H]
            action_type: 动作类型 [B]
            bos_token_id: 起始 token ID
            max_len: 最大生成长度
            temperature: 采样温度 (0.5=保守, 1.0=正常, 1.5=激进)
            top_k: 只从前 k 个候选中采样
        
        Returns:
            tokens: [B, L] 生成的 token 序列
            total_log_prob: [B] 序列总对数概率（用于排序）
        """
        B = decoder_state.size(0)
        device = decoder_state.device
        
        # 初始化：只有 BOS token
        tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        total_log_prob = torch.zeros(B, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # 前向传播（复用 forward 方法）
            logits = self.forward(decoder_state, action_type, tokens)[:, -1, :]  # [B, vocab]
            
            # 温度缩放
            logits = logits / temperature
            
            # Top-K 过滤
            top_k_val = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k_val, dim=-1)
            
            # 计算概率并采样
            probs = torch.softmax(top_k_logits, dim=-1)
            sampled_idx_in_topk = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            next_token = top_k_indices.gather(-1, sampled_idx_in_topk.unsqueeze(-1)).squeeze(-1)
            
            # 累积对数概率（只累加未结束样本的概率）
            log_probs = torch.log_softmax(top_k_logits, dim=-1)
            token_log_prob = log_probs.gather(-1, sampled_idx_in_topk.unsqueeze(-1)).squeeze(-1)
            total_log_prob += token_log_prob * (~finished).float()
            
            # 拼接 token
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            
            # 检查 EOS（提前终止）
            finished = finished | (next_token == self.eos_token_id)
            if finished.all():
                break

        return tokens, total_log_prob
