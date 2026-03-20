import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from config.config import MODEL_CONFIG, TRAIN_CONFIG, PATH_CONFIG
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_pretrainer import ActorPretrainer   # ← 只导入顶层模型，不再导入 SimpleStateTracker

C  = MODEL_CONFIG
TC = TRAIN_CONFIG
PC = PATH_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. 数据 ────────────────────────────────────────────────────────────────
tokenizer = LabelTokenizer(vocab_file=PC["vocab_file"])

dataset = SSRGraphDataset(
    json_path=PC["test_data"],
    tokenizer=tokenizer,
    max_seq_len=C["max_seq_len"],
    max_hist_len=C["max_hist_len"],
)
dataloader = DataLoader(dataset, batch_size=TC["batch_size"], shuffle=True)

# ── 2. 模型 ────────────────────────────────────────────────────────────────
vocab_size = tokenizer.get_vocab_size()
model      = ActorPretrainer(vocab_size=vocab_size).to(device)
# ← 删除单独的 SimpleStateTracker 实例化，已整合进 model.state_tracker

optimizer  = optim.Adam(model.parameters(), lr=TC["lr"])
# ← 只需要 model.parameters()，无需合并两个模型的参数

# ── 3. Loss ────────────────────────────────────────────────────────────────
criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
criterion_seq = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ── 4. 训练循环 ────────────────────────────────────────────────────────────
model.train()

for epoch in range(TC["num_epochs"]):
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # A. 图特征
        node_embeddings, graph_embedding = model.graph_encoder(
            batch.x, batch.edge_index, batch.batch
        )
        graph_state = model.state_proj(graph_embedding)

        # B. 历史状态融合 ← 使用 model 内部的 state_tracker
        decoder_state = model.state_tracker(batch.history_actions, graph_state)

        # C. 统一 squeeze，保证所有 target 都是 1D [B]
        target_action = batch.target_action.squeeze(-1)
        target_src    = batch.target_src.squeeze(-1)
        target_tgt    = batch.target_tgt.squeeze(-1)

        # D. 动作类型预测
        action_logits = model.action_predictor(decoder_state)        # [B, num_actions]
        loss_action   = criterion_cls(action_logits, target_action)

        # E. 指针网络（Teacher Forcing）
        dense_nodes, node_mask_bool = to_dense_batch(node_embeddings, batch.batch)
        node_mask = ~node_mask_bool                                   # [B, N]

        src_logits, tgt_logits = model.pointer_network(
            decoder_state, dense_nodes, target_action,
            target_src_idx=target_src,
            node_mask=node_mask,
        )

        N = src_logits.size(-1)
        valid_src = target_src < N
        valid_tgt = target_tgt < N

        loss_src = criterion_cls(
            src_logits[valid_src], target_src[valid_src]
        ) if valid_src.any() else torch.tensor(0.0, device=device)

        loss_tgt = criterion_cls(
            tgt_logits[valid_tgt], target_tgt[valid_tgt]
        ) if valid_tgt.any() else torch.tensor(0.0, device=device)

        # F. Label 解码器（Teacher Forcing）
        decoder_input_seq = batch.target_label[:, :-1]
        label_target_seq  = batch.target_label[:, 1:]

        label_logits = model.label_decoder(
            decoder_state, target_action, decoder_input_seq
        )                                                             # [B, L-1, vocab_size]

        loss_label = criterion_seq(
            label_logits.reshape(-1, vocab_size),
            label_target_seq.reshape(-1),
        )

        # G. 总 Loss
        loss = loss_action + loss_src + loss_tgt + loss_label
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{TC['num_epochs']}]  Loss: {total_loss:.4f}")
