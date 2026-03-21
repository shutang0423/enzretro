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


class ActionTypePredictor(nn.Module):
    """Step 1: 预测动作类型"""
    def __init__(self, hidden_dim, num_actions):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, decoder_state):
        return self.mlp(decoder_state)   # [B, num_actions]


class PointerNetwork(nn.Module):
    def __init__(self, hidden_dim, node_dim, num_actions, max_atoms):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.src_emb    = nn.Embedding(max_atoms, hidden_dim)
        self.node_proj  = nn.Linear(node_dim, hidden_dim)
        
        # 替换 MultiheadAttention，使用简单的线性映射计算 Query 和 Key
        self.q_proj_src = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_src = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj_tgt = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_tgt = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_state, node_embeddings, action_type,
                target_src_idx=None, node_mask=None):
        if action_type.dim() > 1:
            action_type = action_type.squeeze(-1)

        nodes_proj = self.node_proj(node_embeddings) # [B, max_nodes, H]
        act_emb    = self.action_emb(action_type)    # [B, H]

        # --- 预测 SRC Logits ---
        query_src = self.q_proj_src(decoder_state + act_emb).unsqueeze(1) # [B, 1, H]
        key_src   = self.k_proj_src(nodes_proj)                           # [B, max_nodes, H]
        
        # 手动点积计算 Logits: [B, 1, H] @ [B, H, max_nodes] -> [B, 1, max_nodes]
        src_logits = torch.bmm(query_src, key_src.transpose(1, 2)).squeeze(1) # [B, max_nodes]
        
        # 处理 Mask：将 padding 节点的 logit 设为极小值
        if node_mask is not None:
            src_logits = src_logits.masked_fill(node_mask, -1e9)

        # 获取 src_idx (Teacher Forcing)
        src_idx = target_src_idx if target_src_idx is not None else src_logits.argmax(dim=-1)
        src_idx = src_idx.clamp(0, self.src_emb.num_embeddings - 1)
        src_e   = self.src_emb(src_idx)

        # --- 预测 TGT Logits ---
        query_tgt = self.q_proj_tgt(decoder_state + act_emb + src_e).unsqueeze(1)
        key_tgt   = self.k_proj_tgt(nodes_proj)
        
        tgt_logits = torch.bmm(query_tgt, key_tgt.transpose(1, 2)).squeeze(1)
        
        if node_mask is not None:
            tgt_logits = tgt_logits.masked_fill(node_mask, -1e9)

        return src_logits, tgt_logits


class LabelDecoder(nn.Module):
    """Step 3: 预测 Label 序列 (Transformer Decoder)"""
    def __init__(self, vocab_size, hidden_dim, num_actions, max_pos_enc):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.token_emb  = nn.Embedding(vocab_size,  hidden_dim)
        self.pos_enc    = nn.Embedding(max_pos_enc, hidden_dim)
        decoder_layer   = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc_out      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_state, action_type, tgt_seq):
        B, L = tgt_seq.shape
        if action_type.dim() > 1:
            action_type = action_type.squeeze(-1)

        act_emb  = self.action_emb(action_type).unsqueeze(1)
        memory   = decoder_state.unsqueeze(1) + act_emb

        positions = torch.arange(L, device=tgt_seq.device).unsqueeze(0).expand(B, L)
        tgt_emb   = self.token_emb(tgt_seq) + self.pos_enc(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_seq.device)
        out     = self.transformer(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
        return self.fc_out(out)   # [B, L, vocab_size]


class SimpleStateTracker(nn.Module):
    """历史动作状态追踪"""
    def __init__(self, hidden_dim, num_actions, pad_action_id):
        super().__init__()
        # Embedding 大小 = num_actions + 1（含 pad_action_id）
        self.act_emb = nn.Embedding(
            num_actions + 1, hidden_dim, padding_idx=pad_action_id
        )

    def forward(self, history_actions, graph_embedding):
        hist_context = self.act_emb(history_actions).sum(dim=1)
        return graph_embedding + hist_context


class ActorPretrainer(nn.Module):
    """
    完整的 Actor 预训练模型
    包含: Graph Encoder + (Action, Pointer, Label) 预测器
    """
    def __init__(self, vocab_size: int, cfg: dict = None):
        """
        Args:
            vocab_size : 由 tokenizer.get_vocab_size() 传入，运行时才确定
            cfg        : MODEL_CONFIG 字典；为 None 时自动从 config.py 导入
        """
        super().__init__()

        if cfg is None:
            from config.config import MODEL_CONFIG
            cfg = MODEL_CONFIG

        # ── 从 config 读取，集中在一处，一目了然 ──────────────────
        node_in_dim    = cfg["node_in_dim"]
        node_dim       = cfg["node_dim"]
        hidden_dim     = cfg["hidden_dim"]
        num_actions    = cfg["num_actions"]
        pad_action_id  = cfg["pad_action_id"]
        max_atoms      = cfg["max_atoms"]
        max_pos_enc    = cfg["max_pos_enc"]

        # ── 子模块实例化，只传普通数值，不传 cfg ──────────────────
        self.graph_encoder    = GraphEncoder(
            node_in_dim=node_in_dim,
            hidden_dim=node_dim,          # GraphEncoder 内部维度 = node_dim
        )
        self.state_proj       = nn.Linear(node_dim, hidden_dim)

        self.state_tracker    = SimpleStateTracker(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            pad_action_id=pad_action_id,
        )
        self.action_predictor = ActionTypePredictor(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
        self.pointer_network  = PointerNetwork(
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_actions=num_actions,
            max_atoms=max_atoms,
        )
        self.label_decoder    = LabelDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            max_pos_enc=max_pos_enc,
        )

    def forward(self, x, edge_index, batch, target_action, target_src, decoder_input_seq, history_state=None):
        """
        x, edge_index, batch: PyG 图数据格式
        history_state: 历史编辑状态 [B, 512]。如果是第0步，传入 None。
        """
        # ==========================================
        # 1. 提取图特征
        # ==========================================
        node_embeddings, graph_embedding = self.graph_encoder(x, edge_index, batch)
        
        # ==========================================
        # 【核心修复】：将 2D 节点特征转换为 3D Dense Batch 序列
        # ==========================================
        # dense_nodes: [B, max_nodes, hidden_dim]
        # node_mask: [B, max_nodes] (True 表示真实节点，False 表示 Padding)
        dense_nodes, node_mask = to_dense_batch(node_embeddings, batch)
        
        # PyTorch 的 MultiheadAttention 中，key_padding_mask=True 表示该位置被忽略 (Padding)
        # 而 to_dense_batch 返回的 node_mask 中 True 表示有效节点。
        # 因此我们需要取反 (~) 传给 Attention
        attn_padding_mask = ~node_mask 

        # ==========================================
        # 2. 确定当前状态 (Decoder State)
        # ==========================================
        if history_state is None:
            # 第 0 步：没有历史，直接用 graph_embedding 作为状态
            decoder_state = self.state_proj(graph_embedding)
        else:
            # 第 N 步：结合历史状态 (实际工程中这里会经过 State Tracker)
            decoder_state = history_state
            
        # ==========================================
        # 3. 三大预测器并行预测
        # ==========================================
        # 预测动作类型
        action_logits = self.action_predictor(decoder_state)
        
        # # 预测操作节点 (传入 target_src 进行 Teacher Forcing)
        # src_logits, tgt_logits = self.pointer_network(
        #     decoder_state, node_embeddings, target_action, target_src_idx=target_src
        # )
        src_logits, tgt_logits = self.pointer_network(
            decoder_state=decoder_state, 
            node_embeddings=dense_nodes,        # <--- 传入 3D 的 dense_nodes
            action_type=target_action, 
            target_src_idx=target_src,
            node_mask=attn_padding_mask         # <--- 传入取反后的 mask
        )
        
        # 预测标签序列 (传入 decoder_input_seq 进行 Teacher Forcing)
        label_logits = self.label_decoder(
            decoder_state, target_action, decoder_input_seq
        )
        
        return action_logits, src_logits, tgt_logits, label_logits



