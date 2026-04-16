"""Module 3: Actor Network (策略网络)

子头:
  - ActionTypePredictor  → action_type (7种)
  - PointerNetwork       → src_idx, tgt_idx
  - LabelDecoder         → label token 序列

支持 LoRA 微调: 冻结预训练权重，仅训练低秩增量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from models.graph_encoder import GraphEncoder


# ─── LoRA 层 ───────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper"""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scale


# ─── 子模块 ────────────────────────────────────────────────
class ActionTypePredictor(nn.Module):
    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state):
        return self.mlp(state)


class PointerNetwork(nn.Module):
    """分层指针: state+action → src → state+action+src → tgt"""

    def __init__(self, hidden_dim: int, node_dim: int,
                 num_actions: int, max_atoms: int):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.src_emb    = nn.Embedding(max_atoms, hidden_dim)
        self.node_proj  = nn.Linear(node_dim, hidden_dim)
        self.q_src = nn.Linear(hidden_dim, hidden_dim)
        self.k_src = nn.Linear(hidden_dim, hidden_dim)
        self.q_tgt = nn.Linear(hidden_dim, hidden_dim)
        self.k_tgt = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, dense_nodes, action_type,
                src_idx=None, node_mask=None,
                src_valid=None, tgt_valid=None):
        """
        Args:
            state       : [B, hidden_dim]
            dense_nodes : [B, N, node_dim]       to_dense_batch 输出
            action_type : [B] or [B, 1]
            src_idx     : [B] or [B, 1] or None  None=推理时argmax采样
            node_mask   : [B, N]  True=padding位置 (to_dense_batch取反后传入)
            src_valid   : [B]  BoolTensor, False=Terminate步骤,跳过src嵌入查询
            tgt_valid   : [B]  BoolTensor, False=Terminate步骤,跳过tgt计算
        Returns:
            src_logits  : [B, N]
            tgt_logits  : [B, N]
        """
        # ── 维度统一 ───────────────────────────────────────────────
        if state.dim() == 2:
            state = state.unsqueeze(1)           # [B, 1, h]
        if action_type.dim() > 1:
            action_type = action_type.squeeze(-1)  # [B]

        B = state.size(0)

        # dense_nodes: 确保 [B, N, node_dim]
        if dense_nodes.dim() == 2:
            dense_nodes = dense_nodes.unsqueeze(0)
        if dense_nodes.size(0) == 1 and B > 1:
            dense_nodes = dense_nodes.expand(B, -1, -1)

        # node_mask: 确保 [B, N]
        if node_mask is not None and node_mask.size(0) == 1 and B > 1:
            node_mask = node_mask.expand(B, -1)

        # ── 公共投影 ───────────────────────────────────────────────
        nodes_proj = self.node_proj(dense_nodes)           # [B, N, h]
        act_emb    = self.action_emb(action_type)          # [B, h]
        act_emb    = act_emb.unsqueeze(1)                  # [B, 1, h]

        # ── src 预测 ───────────────────────────────────────────────
        q_src      = self.q_src(state + act_emb)           # [B, 1, h]
        k_src      = self.k_src(nodes_proj)                # [B, N, h]
        src_logits = torch.bmm(q_src, k_src.transpose(1, 2)).squeeze(1)  # [B, N]

        if node_mask is not None:
            src_logits = src_logits.masked_fill(node_mask, -1e4)

        # ── 确定 src_idx (推理 or Teacher Forcing) ─────────────────
        if src_idx is None:
            # 推理模式: argmax 采样
            src_idx = src_logits.argmax(dim=-1)            # [B]
        else:
            if src_idx.dim() > 1:
                src_idx = src_idx.squeeze(-1)              # [B]
            # ★ 关键修复: clamp 保证索引合法 (-1 → 0)
            #   但 Terminate 步骤的梯度由 src_valid mask 在 loss 处屏蔽
            src_idx = src_idx.clamp(0, self.src_emb.num_embeddings - 1)

        # ── src 嵌入 (Terminate 步骤用零向量替代, 避免污染梯度) ──────
        src_e = self.src_emb(src_idx)                      # [B, h]
        if src_valid is not None:
            # src_valid=False 的行置零，不让无效索引的嵌入影响 tgt 预测
            src_e = src_e * src_valid.float().unsqueeze(-1)  # [B, h]
        src_e = src_e.unsqueeze(1)                         # [B, 1, h]

        # ── tgt 预测 ───────────────────────────────────────────────
        q_tgt      = self.q_tgt(state + act_emb + src_e)  # [B, 1, h]
        k_tgt      = self.k_tgt(nodes_proj)               # [B, N, h]
        tgt_logits = torch.bmm(q_tgt, k_tgt.transpose(1, 2)).squeeze(1)  # [B, N]

        if node_mask is not None:
            tgt_logits = tgt_logits.masked_fill(node_mask, -1e4)

    return src_logits, tgt_logits

class LabelDecoder(nn.Module):
    """Transformer Decoder 生成 label token 序列"""

    def __init__(self, vocab_size: int, hidden_dim: int,
                 num_actions: int, max_pos_enc: int, num_layers: int = 4):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.token_emb  = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc    = nn.Embedding(max_pos_enc, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, state, action_type, tgt_seq):
        B, L = tgt_seq.shape
        if action_type.dim() > 1:
            action_type = action_type.squeeze(-1)
        act = self.action_emb(action_type).unsqueeze(1)
        memory = state.unsqueeze(1) + act
        pos = torch.arange(L, device=tgt_seq.device).unsqueeze(0).expand(B, L)
        tgt = self.token_emb(tgt_seq) + self.pos_enc(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_seq.device)
        out = self.transformer(tgt=tgt, memory=memory, tgt_mask=causal)
        return self.fc_out(out)


# ─── 完整 Actor ────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """完整策略网络 (预训练 + LoRA 微调)"""

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        nd, hd = cfg["node_dim"], cfg["hidden_dim"]
        na, ma = cfg["num_actions"], cfg["max_atoms"]
        mpe = cfg["max_pos_enc"]

        self.graph_encoder    = GraphEncoder(cfg["node_in_dim"], nd,
                                             cfg["gat_layers"], cfg["gat_heads"])
        self.state_proj       = nn.Linear(nd, hd)
        self.action_predictor = ActionTypePredictor(hd, na)
        self.pointer_network  = PointerNetwork(hd, nd, na, ma)
        self.label_decoder    = LabelDecoder(vocab_size, hd, na, mpe)

    def encode_graph(self, x, edge_index, batch):
        """编码图并返回 dense 格式"""
        node_emb, graph_emb = self.graph_encoder(x, edge_index, batch)
        dense_nodes, node_mask = to_dense_batch(node_emb, batch)
        return dense_nodes, ~node_mask, self.state_proj(graph_emb)

    def forward(self, x, edge_index, batch,
                target_action, target_src, decoder_input_seq,
                decoder_state=None):
        """预训练 forward (teacher forcing)"""
        dense_nodes, pad_mask, graph_state = self.encode_graph(x, edge_index, batch)
        state = decoder_state if decoder_state is not None else graph_state
        action_logits = self.action_predictor(state)
        src_logits, tgt_logits = self.pointer_network(
            state, dense_nodes, target_action, target_src, pad_mask
        )
        label_logits = self.label_decoder(state, target_action, decoder_input_seq)
        return action_logits, src_logits, tgt_logits, label_logits

    def inject_lora(self, rank: int = 8, alpha: float = 16.0):
        """将关键 Linear 层替换为 LoRA 版本"""
        targets = [self.action_predictor, self.pointer_network, self.label_decoder]
        count = 0
        for module in targets:
            for name, child in list(module.named_modules()):
                if isinstance(child, nn.Linear):
                    parts = name.split(".")
                    parent = module
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], LoRALinear(child, rank, alpha))
                    count += 1
        return count

    def freeze_pretrained(self):
        """冻结所有非 LoRA 参数"""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]