"""
actor_network.py —— Actor Network 顶层组装

职责：
  - 组装 Encoder / StateTracker / 三大预测头
  - 提供统一 forward 接口（训练模式，Teacher Forcing）
  - 提供 predict_step / generate 接口（推理模式）
  - 不含 Loss 计算（由 pretrain.py + loss_strategy 负责）

消融实验切换（只改 ModelConfig）：
  MODEL_CFG.encoder_type = "gat" | "fingerprint"

数据流（训练）：
  encoder_kwargs → Encoder → EncoderOutput
  history        → StateTracker → decoder_state [B, H]
  decoder_state  → ActionPredictor  → action_logits [B, num_actions]
                 → PointerNetwork   → src_logits, tgt_logits [B, max_atoms]
                 → LabelDecoder     → label_logits [B, L, vocab_size]

数据流（推理）：
  generate() 循环调用 predict_step()
    → greedy action / src / tgt
    → greedy_decode label
    → _append_history() 更新历史
    → 直到 STOP 或 max_steps
"""

from __future__ import annotations
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config.config import MODEL_CFG
from model.encoders         import build_encoder
from model.state_tracker    import StateTracker, HistoryBatch
from model.prediction_heads import ActionTypePredictor, PointerNetwork, LabelDecoder


# ══════════════════════════════════════════════════════════════════════
#  数据容器
# ══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class TeacherForcingTargets:
    """
    训练时的 Teacher Forcing 目标值
      action    : [B]     目标 action_type
      src       : [B]     目标 src_idx
      label_seq : [B, L]  目标 label token 序列（含 BOS 和 EOS）
    """
    action    : torch.Tensor
    src       : torch.Tensor
    label_seq : torch.Tensor


@dataclass
class EditStep:
    """
    单步推理结果（完整的一步编辑动作）
      action_type  : [B]
      src_idx      : [B]
      tgt_idx      : [B]
      label_tokens : [B, L]
    """
    action_type  : torch.Tensor
    src_idx      : torch.Tensor
    tgt_idx      : torch.Tensor
    label_tokens : torch.Tensor


# ══════════════════════════════════════════════════════════════════════
#  ActorNetwork
# ══════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """
    Actor Network 完整实现

    训练模式 forward()：
      Teacher Forcing，一次性前向，返回四组 logits + gru_hidden

    推理模式 generate()：
      自回归逐步生成完整编辑序列，直到预测 STOP 或达到 max_steps
    """

    def __init__(
        self,
        vocab_size : int,
        cfg        = MODEL_CFG,
    ):
        super().__init__()

        # ── 统一使用 ModelConfig dataclass ──────────────────────────
        self.cfg = cfg or MODEL_CFG

        # ── 推理时需要的常量 ─────────────────────────────────────────
        self.stop_action_id = cfg.stop_action_id
        self.bos_token_id   = cfg.bos_token_id
        self.label_max_len  = cfg.max_seq_len    # 推理时 label 最大长度

        # ── 子模块实例化（全部从 cfg 属性读取，无硬编码）────────────
        self.encoder = build_encoder(cfg)

        # node_dim → hidden_dim 维度对齐投影
        self.state_proj = nn.Linear(cfg.node_dim, cfg.hidden_dim)

        self.state_tracker = StateTracker(
            hidden_dim    = cfg.hidden_dim,
            num_actions   = cfg.num_actions,
            max_atoms     = cfg.max_atoms,
            vocab_size    = vocab_size,
            pad_action_id = cfg.pad_action_id,
            pad_atom_id   = cfg.pad_atom_id,
            pad_token_id  = cfg.pad_token_id,
        )
        self.action_predictor = ActionTypePredictor(
            hidden_dim  = cfg.hidden_dim,
            num_actions = cfg.num_actions,
        )
        self.pointer_network = PointerNetwork(
            hidden_dim  = cfg.hidden_dim,
            node_dim    = cfg.node_dim,
            num_actions = cfg.num_actions,
            max_atoms   = cfg.max_atoms,
        )
        self.label_decoder = LabelDecoder(
            vocab_size   = vocab_size,
            hidden_dim   = cfg.hidden_dim,
            num_actions  = cfg.num_actions,
            max_pos_enc  = cfg.max_pos_enc,
            eos_token_id = cfg.eos_token_id,
        )

    # ══════════════════════════════════════════════════════════════════
    #  内部工具
    # ══════════════════════════════════════════════════════════════════

    def _encode(self, **encoder_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一编码入口：调用可插拔 Encoder，并将 graph_emb 投影到 hidden_dim
        返回:
          enc_out       : EncoderOutput（含 dense_nodes / node_pad_mask / has_nodes）
          graph_emb_proj: [B, hidden_dim]
        """
        enc_out        = self.encoder(**encoder_kwargs)
        graph_emb_proj = self.state_proj(enc_out.graph_emb)   # [B, hidden_dim]
        return enc_out, graph_emb_proj

    def _append_history(
        self,
        history   : HistoryBatch,
        edit      : EditStep,
        label_len : int,
    ) -> HistoryBatch:
        """
        将当前步的 EditStep 追加到 HistoryBatch，供下一步 StateTracker 使用。

        label_tokens 可能比 label_len 短（greedy_decode 提前结束），
        需要 pad 到统一长度 label_len。
        """
        B      = edit.action_type.size(0)
        device = edit.action_type.device

        # ── label_tokens pad / 截断到 label_len ─────────────────────
        L_cur = edit.label_tokens.size(1)
        if L_cur < label_len:
            pad = torch.full(
                (B, label_len - L_cur),
                self.cfg.pad_token_id,
                dtype=torch.long, device=device,
            )
            label_padded = torch.cat([edit.label_tokens, pad], dim=1)  # [B, label_len]
        else:
            label_padded = edit.label_tokens[:, :label_len]            # [B, label_len]

        # ── 拼接到已有历史（在 T 维度 cat）──────────────────────────
        return HistoryBatch(
            actions    = torch.cat(
                [history.actions,    edit.action_type.unsqueeze(1)], dim=1
            ),  # [B, T+1]
            src_idxs   = torch.cat(
                [history.src_idxs,   edit.src_idx.unsqueeze(1)],    dim=1
            ),  # [B, T+1]
            tgt_idxs   = torch.cat(
                [history.tgt_idxs,   edit.tgt_idx.unsqueeze(1)],    dim=1
            ),  # [B, T+1]
            label_seqs = torch.cat(
                [history.label_seqs, label_padded.unsqueeze(1)],     dim=1
            ),  # [B, T+1, label_len]
        )

    # ══════════════════════════════════════════════════════════════════
    #  训练接口
    # ══════════════════════════════════════════════════════════════════

    def forward(
        self,
        history    : HistoryBatch,
        tf         : TeacherForcingTargets,
        gru_hidden : Optional[torch.Tensor] = None,
        **encoder_kwargs,                        # x/edge_index/batch 或 fingerprint
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练前向（Teacher Forcing）

        Args:
          history       : 完整历史动作序列
          tf            : Teacher Forcing 目标值
          gru_hidden    : [1, B, H]，推理增量更新时传入；训练时传 None
          encoder_kwargs: 图数据（GAT）或指纹数据（Fingerprint）

        Returns:
          action_logits : [B, num_actions]
          src_logits    : [B, max_atoms]
          tgt_logits    : [B, max_atoms]
          label_logits  : [B, L, vocab_size]
          gru_hidden    : [1, B, H]
        """
        # Step 1: 图编码
        enc_out, graph_emb = self._encode(**encoder_kwargs)

        # Step 2: 状态追踪（历史 + 图特征 → decoder_state）
        decoder_state, gru_hidden = self.state_tracker(
            graph_emb, history, gru_hidden
        )

        # Step 3a: 动作类型预测
        action_logits = self.action_predictor(decoder_state)

        # Step 3b: 指针预测（Teacher Forcing：传入 target src）
        src_logits, tgt_logits = self.pointer_network(
            decoder_state  = decoder_state,
            dense_nodes    = enc_out.dense_nodes,
            action_type    = tf.action,
            target_src_idx = tf.src,
            node_mask      = enc_out.node_pad_mask,
            has_nodes      = enc_out.has_nodes,
        )

        # Step 3c: 标签序列预测（输入去掉最后一个 token，即 EOS）
        label_logits = self.label_decoder(
            decoder_state = decoder_state,
            action_type   = tf.action,
            tgt_seq       = tf.label_seq[:, :-1],
        )

        return action_logits, src_logits, tgt_logits, label_logits, gru_hidden

    # ══════════════════════════════════════════════════════════════════
    #  推理接口：单步
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict_step(
        self,
        history       : HistoryBatch,
        gru_hidden    : Optional[torch.Tensor] = None,
        label_max_len : Optional[int] = None,
        **encoder_kwargs,
    ) -> Tuple[EditStep, torch.Tensor]:
        """
        单步推理：贪心解码，返回完整 EditStep 和更新后的 gru_hidden

        Args:
          history       : 当前步之前的历史（T 步）
          gru_hidden    : [1, B, H]，上一步的 GRU 隐状态
          label_max_len : label 自回归最大长度，默认取 cfg.max_seq_len
          encoder_kwargs: 图数据或指纹数据

        Returns:
          edit       : EditStep（action_type, src_idx, tgt_idx, label_tokens）
          gru_hidden : [1, B, H]，供下一步使用
        """
        if label_max_len is None:
            label_max_len = self.label_max_len

        enc_out, graph_emb = self._encode(**encoder_kwargs)

        decoder_state, gru_hidden = self.state_tracker(
            graph_emb, history, gru_hidden
        )

        # 贪心预测 action
        pred_action = self.action_predictor(decoder_state).argmax(dim=-1)  # [B]

        # 贪心预测 src / tgt（不传 target_src_idx → 推理模式）
        src_logits, tgt_logits = self.pointer_network(
            decoder_state  = decoder_state,
            dense_nodes    = enc_out.dense_nodes,
            action_type    = pred_action,
            target_src_idx = None,
            node_mask      = enc_out.node_pad_mask,
            has_nodes      = enc_out.has_nodes,
        )
        pred_src = src_logits.argmax(dim=-1)   # [B]
        pred_tgt = tgt_logits.argmax(dim=-1)   # [B]

        # 自回归贪心解码 label
        pred_label = self.label_decoder.greedy_decode(
            decoder_state = decoder_state,
            action_type   = pred_action,
            bos_token_id  = self.bos_token_id,
            max_len       = label_max_len,
        )                                      # [B, actual_len ≤ label_max_len]

        edit = EditStep(
            action_type  = pred_action,
            src_idx      = pred_src,
            tgt_idx      = pred_tgt,
            label_tokens = pred_label,
        )
        return edit, gru_hidden

    # ══════════════════════════════════════════════════════════════════
    #  推理接口：完整序列生成
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        label_len     : Optional[int] = None,
        max_steps     : int = 10,
        label_max_len : Optional[int] = None,
        **encoder_kwargs,
    ) -> List[EditStep]:
        """
        自回归生成完整编辑序列

        循环调用 predict_step，直到所有样本预测 STOP 或达到 max_steps。

        Args:
          label_len     : HistoryBatch.label_seqs 的 L 维度
                          默认取 cfg.max_seq_len
          max_steps     : 最大生成步数，默认取 cfg.max_hist_len
          label_max_len : 每步 label 自回归最大长度，默认取 cfg.max_seq_len
          encoder_kwargs: 传给 encoder 的图数据或指纹数据

        Returns:
          edit_steps : List[EditStep]，长度 ≤ max_steps
        """
        # ── 默认值从 cfg 读取，不硬编码 ─────────────────────────────
        if label_len is None:
            label_len = self.cfg.max_seq_len
        if label_max_len is None:
            label_max_len = self.cfg.max_seq_len
        if max_steps is None:
            max_steps = self.cfg.max_hist_len

        # ── 推断 batch_size & device ─────────────────────────────────
        first_val  = next(iter(encoder_kwargs.values()))
        batch_size = 1 # first_val.size(0)
        device     = first_val.device

        # ── 初始化空历史（第 0 步）───────────────────────────────────
        history    = HistoryBatch.empty(batch_size, label_len, device)
        gru_hidden = None
        edit_steps : List[EditStep] = []
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            edit, gru_hidden = self.predict_step(
                history       = history,
                gru_hidden    = gru_hidden,
                label_max_len = label_max_len,
                **encoder_kwargs,
            )
            edit_steps.append(edit)

            # 检查终止条件
            finished = finished | (edit.action_type == self.stop_action_id)
            if finished.all():
                break

            # 将当前步追加到历史，供下一步 StateTracker 使用
            history = self._append_history(history, edit, label_len)

        return edit_steps