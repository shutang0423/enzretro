import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class GraphEncoder(nn.Module):
    """
    产物图编码器 (基于 GAT)
    将分子图转换为节点嵌入(Node Embeddings)和图嵌入(Graph Embedding)
    """
    def __init__(self, node_in_dim=128, hidden_dim=256, num_layers=4):
        super().__init__()
        # 节点特征投影
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        
        # 多层 GAT 图注意力卷积
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        
        # 图级特征池化后的投影
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch):
        """
        x: [num_nodes, node_in_dim] 原子特征
        edge_index: [2, num_edges] 边连接关系
        batch: [num_nodes] 批次索引，用于区分同一个 batch 中的不同图
        """
        # 1. 节点特征投影
        h = self.node_proj(x)
        
        # 2. 图卷积与残差连接
        for conv in self.convs:
            h_res = h
            h = F.relu(conv(h, edge_index))
            h = h + h_res  # 残差连接，防止梯度消失
            
        node_embeddings = h  # [num_nodes, hidden_dim]
        
        # 3. 全局池化得到图嵌入
        graph_emb = global_mean_pool(node_embeddings, batch)  # [batch_size, hidden_dim]
        graph_emb = self.graph_pool(graph_emb)
        
        return node_embeddings, graph_emb



class ActionTypePredictor(nn.Module):
    """Step 1: 预测 7 种动作类型"""
    def __init__(self, hidden_dim=512, num_actions=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, decoder_state):
        # decoder_state: [batch_size, hidden_dim]
        return self.mlp(decoder_state)  # [batch_size, 7]


class PointerNetwork(nn.Module):
    """Step 2: 预测源节点(src)和目标节点(tgt)"""
    def __init__(self, hidden_dim=512, node_dim=256, num_actions=7):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.src_emb = nn.Embedding(1000, hidden_dim)  # 假设最大原子数为1000
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        self.src_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.tgt_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

    def forward(self, decoder_state, node_embeddings, action_type, target_src_idx=None, node_mask=None):
        # node_embeddings: [B, N, 256] -> [B, N, 512]
        nodes_proj = self.node_proj(node_embeddings)
        act_emb = self.action_emb(action_type)  # [B, 512]
        
        # 1. 预测 src_idx
        query_src = (decoder_state + act_emb).unsqueeze(1)  # [B, 1, 512]
        _, src_attn = self.src_attn(query_src, nodes_proj, nodes_proj, key_padding_mask=node_mask)
        src_logits = src_attn.squeeze(1)  # [B, N]
        
        # 2. 预测 tgt_idx (预训练时使用真实的 target_src_idx 进行 Teacher Forcing)
        src_idx = target_src_idx if target_src_idx is not None else src_logits.argmax(dim=-1)
        src_e = self.src_emb(src_idx)  # [B, 512]
        
        query_tgt = (decoder_state + act_emb + src_e).unsqueeze(1)  # [B, 1, 512]
        _, tgt_attn = self.tgt_attn(query_tgt, nodes_proj, nodes_proj, key_padding_mask=node_mask)
        tgt_logits = tgt_attn.squeeze(1)  # [B, N]
        
        return src_logits, tgt_logits


# class LabelDecoder(nn.Module):
#     """Step 3: 预测 Label 序列 (Transformer Decoder)"""
#     def __init__(self, vocab_size, hidden_dim=512, num_actions=7, max_len=20):
#         super().__init__()
#         self.action_emb = nn.Embedding(num_actions, hidden_dim)
#         self.token_emb = nn.Embedding(vocab_size, hidden_dim)
#         self.pos_enc = nn.Embedding(max_len, hidden_dim)
        
#         decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
#         self.fc_out = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, decoder_state, action_type, tgt_seq):
#         # tgt_seq: [B, L] (输入序列，通常是右移一位的 target)
#         B, L = tgt_seq.shape
        
#         # Memory 仅包含 decoder_state 和 action_type 的信息
#         act_emb = self.action_emb(action_type).unsqueeze(1)  # [B, 1, 512]
#         memory = decoder_state.unsqueeze(1) + act_emb        # [B, 1, 512]
        
#         # 目标序列 Embedding + 位置编码
#         positions = torch.arange(L, device=tgt_seq.device).unsqueeze(0).expand(B, L)
#         tgt_emb = self.token_emb(tgt_seq) + self.pos_enc(positions)
        
#         # 因果掩码 (防止看到未来的 token)
#         causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_seq.device)
        
#         out = self.transformer(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
#         logits = self.fc_out(out)  # [B, L, vocab_size]
#         return logits
    
class LabelDecoder(nn.Module):
    """Step 3: 预测 Label 序列 (Transformer Decoder)"""
    # 【修改1】将 max_len 调大到 256，防止长序列越界
    def __init__(self, vocab_size, hidden_dim=512, num_actions=7, max_len=256):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Embedding(max_len, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_state, action_type, tgt_seq):
        # tgt_seq: [B, L]
        B, L = tgt_seq.shape
        
        # 【修改2】确保 action_type 是 1D 张量 [B]，防止从 DataLoader 出来是 [B, 1]
        if action_type.dim() > 1:
            action_type = action_type.squeeze(-1)
            
        # Memory 仅包含 decoder_state 和 action_type 的信息
        act_emb = self.action_emb(action_type).unsqueeze(1)  # [B, 1, 512]
        memory = decoder_state.unsqueeze(1) + act_emb        # [B, 1, 512]
        
        # 目标序列 Embedding + 位置编码
        positions = torch.arange(L, device=tgt_seq.device).unsqueeze(0).expand(B, L)
        tgt_emb = self.token_emb(tgt_seq) + self.pos_enc(positions)
        
        # 因果掩码 (防止看到未来的 token)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_seq.device)
        
        out = self.transformer(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
        logits = self.fc_out(out)  # [B, L, vocab_size]
        return logits


class SimpleStateTracker(nn.Module):
    """极简版状态追踪：将历史动作序列 Embedding 后求平均"""
    def __init__(self, num_actions=8, hidden_dim=512):
        super().__init__()
        self.act_emb = nn.Embedding(num_actions, hidden_dim, padding_idx=7)
        
    def forward(self, history_actions, graph_embedding):
        # history_actions: [B, max_hist_len]
        # graph_embedding: [B, 512]
        hist_emb = self.act_emb(history_actions) # [B, L, 512]
        hist_context = hist_emb.sum(dim=1)       # [B, 512]
        
        # 融合图特征和历史特征
        return graph_embedding + hist_context

class ActorPretrainer(nn.Module):
    """
    完整的 Actor 预训练模型
    包含: Graph Encoder + (Action, Pointer, Label) 预测器
    """
    def __init__(self, vocab_size, node_in_dim=128, node_dim=256, hidden_dim=512):
        super().__init__()
        # 1. 编码器
        self.graph_encoder = GraphEncoder(node_in_dim=node_in_dim, hidden_dim=node_dim)
        
        # 维度转换: 将 Graph Embedding (256) 映射到 Decoder State (512)
        self.state_proj = nn.Linear(node_dim, hidden_dim)
        
        # 2. 三个预测器 (复用之前定义的类)
        self.action_predictor = ActionTypePredictor(hidden_dim)
        self.pointer_network = PointerNetwork(hidden_dim, node_dim)
        self.label_decoder = LabelDecoder(vocab_size, hidden_dim)

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



