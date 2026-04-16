"""Policy: 整合 Actor + Critic + StateTracker 的统一 RL 接口"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple

from models.actor import ActorNetwork
from models.critic import CriticNetwork
from models.state_tracker import GRUStateTracker


class RetroSynthesisPolicy(nn.Module):
    """逆合成编辑策略

    职责:
      1. 管理 Actor / Critic / StateTracker 生命周期
      2. 提供 RL 训练接口: sample_action / evaluate_actions
      3. 加载预训练 Actor 并注入 LoRA
    """

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        hd, na = cfg["hidden_dim"], cfg["num_actions"]

        self.actor  = ActorNetwork(vocab_size, cfg)
        self.critic = CriticNetwork(cfg)
        self.state_tracker = GRUStateTracker(
            hd, na, cfg["pad_action_id"], cfg.get("gru_layers", 2)
        )

    # ─── 预训练加载 & LoRA ────────────────────────────────
    def load_pretrained_actor(self, ckpt_path: str, strict: bool = False):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        actor_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "").replace("actor.", "")
            actor_dict[clean_k] = v
        return self.actor.load_state_dict(actor_dict, strict=strict)

    def setup_lora(self, rank: int = 8, alpha: float = 16.0):
        n = self.actor.inject_lora(rank, alpha)
        self.actor.freeze_pretrained()
        return n

    def freeze_encoder(self):
        for p in self.actor.graph_encoder.parameters():
            p.requires_grad_(False)

    # ─── 状态计算 (单次编码，共享结果) ────────────────────
    def _get_decoder_state(self, x, edge_index, batch, history_actions=None):
        """Returns: dense_nodes [B,N,d], pad_mask [B,N], state [B,h]"""
        dense_nodes, pad_mask, graph_state = self.actor.encode_graph(
            x, edge_index, batch
        )
        if history_actions is not None:
            state = self.state_tracker(history_actions, graph_state)
        else:
            state = graph_state
        return dense_nodes, pad_mask, state

    # ─── RL: 采样 ────────────────────────────────────────
    @torch.no_grad()
    def sample_action(self, x, edge_index, batch,
                      history_actions=None, action_mask=None,
                      temperature: float = 1.0):
        """采样一步动作 (rollout 用)

        Returns: action_type[B], src[B], tgt[B], log_prob[B], value[B]
        """
        dense_nodes, pad_mask, state = self._get_decoder_state(
            x, edge_index, batch, history_actions
        )

        # action type
        act_logits = self.actor.action_predictor(state) / max(temperature, 1e-8)
        if action_mask is not None:
            act_logits = act_logits.masked_fill(~action_mask, -1e9)
        act_dist = Categorical(logits=act_logits)
        action_type = act_dist.sample()
        act_lp = act_dist.log_prob(action_type)

        # src, tgt (分层: src先采样，tgt条件于src)
        src_logits, tgt_logits = self.actor.pointer_network(
            state, dense_nodes, action_type, src_idx=None, node_mask=pad_mask
        )
        src_dist = Categorical(logits=src_logits / max(temperature, 1e-8))
        src_idx = src_dist.sample()
        src_lp = src_dist.log_prob(src_idx)

        tgt_dist = Categorical(logits=tgt_logits / max(temperature, 1e-8))
        tgt_idx = tgt_dist.sample()
        tgt_lp = tgt_dist.log_prob(tgt_idx)

        total_lp = act_lp + src_lp + tgt_lp

        # value (独立 Critic)
        v = self.critic.get_value(x, edge_index, batch, history_actions)
        return action_type, src_idx, tgt_idx, total_lp, v.squeeze(-1)

    # ─── RL: 评估 ────────────────────────────────────────
    def evaluate_actions(self, x, edge_index, batch,
                         history_actions, actions, src_indices, tgt_indices,
                         action_mask=None):
        """评估已有动作的 log_prob, entropy, value, q_value"""
        dense_nodes, pad_mask, state = self._get_decoder_state(
            x, edge_index, batch, history_actions
        )

        # action type
        act_logits = self.actor.action_predictor(state)
        if action_mask is not None:
            act_logits = act_logits.masked_fill(~action_mask, -1e9)
        act_dist = Categorical(logits=act_logits)
        act_lp = act_dist.log_prob(actions)
        entropy = act_dist.entropy()

        # pointer
        src_logits, tgt_logits = self.actor.pointer_network(
            state, dense_nodes, actions, src_idx=src_indices, node_mask=pad_mask
        )
        src_lp = Categorical(logits=src_logits).log_prob(src_indices)
        tgt_lp = Categorical(logits=tgt_logits).log_prob(tgt_indices)

        total_lp = act_lp + src_lp + tgt_lp

        # critic
        v, q = self.critic(
            x, edge_index, batch,
            action_type=actions, history_actions=history_actions,
        )
        return total_lp, entropy, v, q

    # ─── KL 参考策略 ─────────────────────────────────────
    @torch.no_grad()
    def get_ref_log_probs(self, x, edge_index, batch,
                          history_actions, actions, src_indices, tgt_indices):
        """冻结预训练 Actor 计算参考 log_prob (用于 KL 约束)"""
        dense_nodes, pad_mask, state = self._get_decoder_state(
            x, edge_index, batch, history_actions
        )
        act_lp = Categorical(logits=self.actor.action_predictor(state)).log_prob(actions)
        src_logits, tgt_logits = self.actor.pointer_network(
            state, dense_nodes, actions, src_idx=src_indices, node_mask=pad_mask
        )
        src_lp = Categorical(logits=src_logits).log_prob(src_indices)
        tgt_lp = Categorical(logits=tgt_logits).log_prob(tgt_indices)
        return act_lp + src_lp + tgt_lp