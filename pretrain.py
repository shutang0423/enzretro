import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from config.config import MODEL_CONFIG, TRAIN_CONFIG, PATH_CONFIG
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_pretrainer import ActorPretrainer

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
optimizer  = optim.Adam(model.parameters(), lr=TC["lr"])

criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
criterion_seq = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ── 3. NaN 检测工具 ────────────────────────────────────────────────────────
def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"  [NaN/Inf detected] {name}")
        return True
    return False

def safe_loss(logits, target, criterion, name):
    """计算 loss，若结果为 NaN 则打印诊断信息并返回 0"""
    loss = criterion(logits, target)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  [NaN Loss] {name} | logits range: "
              f"[{logits.min():.3f}, {logits.max():.3f}] "
              f"| target range: [{target.min()}, {target.max()}]")
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return loss

# ── 4. 训练循环 ────────────────────────────────────────────────────────────
model.train()

for epoch in range(TC["num_epochs"]):
    # 每个 epoch 的累计 loss
    sum_loss        = 0.0
    sum_loss_action = 0.0
    sum_loss_src    = 0.0
    sum_loss_tgt    = 0.0
    sum_loss_label  = 0.0
    num_batches     = 0

    # tqdm 进度条：显示当前 epoch 和 step 级别的实时 loss
    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch+1:>3}/{TC['num_epochs']}]",
        ncols=120,
        leave=True,
    )

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ── A. 图特征编码 ────────────────────────────────────────────────
        node_embeddings, graph_embedding = model.graph_encoder(
            batch.x, batch.edge_index, batch.batch
        )
        graph_state   = model.state_proj(graph_embedding)

        # ── B. 历史状态融合 ──────────────────────────────────────────────
        decoder_state = model.state_tracker(batch.history_actions, graph_state)

        # ── C. squeeze 保证 target 均为 1D [B] ──────────────────────────
        target_action = batch.target_action.squeeze(-1)   # [B]
        target_src    = batch.target_src.squeeze(-1)      # [B]
        target_tgt    = batch.target_tgt.squeeze(-1)      # [B]

        # ── D. 动作类型预测 ──────────────────────────────────────────────
        action_logits = model.action_predictor(decoder_state)   # [B, num_actions]
        loss_action   = safe_loss(action_logits, target_action,
                                  criterion_cls, "loss_action")

        # ── E. 指针网络 ──────────────────────────────────────────────────
        dense_nodes, node_mask_bool = to_dense_batch(node_embeddings, batch.batch)
        # key_padding_mask: True = 忽略该位置（padding）
        node_mask = ~node_mask_bool                        # [B, N]

        src_logits, tgt_logits = model.pointer_network(
            decoder_state, dense_nodes, target_action,
            target_src_idx=target_src,
            node_mask=node_mask,
        )
        # src/tgt logits: [B, N]，过滤掉 target 越界的样本
        N         = src_logits.size(-1)
        valid_src = (target_src >= 0) & (target_src < N)
        valid_tgt = (target_tgt >= 0) & (target_tgt < N)

        if valid_src.any():
            loss_src = safe_loss(src_logits[valid_src], target_src[valid_src],
                                 criterion_cls, "loss_src")
        else:
            loss_src = torch.tensor(0.0, device=device, requires_grad=True)

        if valid_tgt.any():
            loss_tgt = safe_loss(tgt_logits[valid_tgt], target_tgt[valid_tgt],
                                 criterion_cls, "loss_tgt")
        else:
            loss_tgt = torch.tensor(0.0, device=device, requires_grad=True)

        # ── F. Label 解码器 ──────────────────────────────────────────────
        decoder_input_seq = batch.target_label[:, :-1]    # [B, L-1]
        label_target_seq  = batch.target_label[:, 1:]     # [B, L-1]

        label_logits = model.label_decoder(
            decoder_state, target_action, decoder_input_seq
        )                                                  # [B, L-1, vocab_size]

        loss_label = safe_loss(
            label_logits.reshape(-1, vocab_size),
            label_target_seq.reshape(-1),
            criterion_seq, "loss_label"
        )

        # ── G. 总 Loss + 梯度裁剪 ────────────────────────────────────────
        loss = loss_action + loss_src + loss_tgt + loss_label
        loss.backward()
        # 梯度裁剪，防止梯度爆炸导致 NaN
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── H. 累计统计 ──────────────────────────────────────────────────
        la  = loss_action.item()
        ls  = loss_src.item()
        lt  = loss_tgt.item()
        ll  = loss_label.item()
        tot = loss.item()

        sum_loss        += tot
        sum_loss_action += la
        sum_loss_src    += ls
        sum_loss_tgt    += lt
        sum_loss_label  += ll
        num_batches     += 1

        # 进度条实时显示当前 step 的各项 loss
        pbar.set_postfix({
            "tot":    f"{tot:.3f}",
            "act":    f"{la:.3f}",
            "src":    f"{ls:.3f}",
            "tgt":    f"{lt:.3f}",
            "label":  f"{ll:.3f}",
        })

    # ── Epoch 结束：打印平均 loss ─────────────────────────────────────────
    n = max(num_batches, 1)
    print(
        f"  ▶ Epoch [{epoch+1:>3}/{TC['num_epochs']}] avg | "
        f"total: {sum_loss/n:.4f}  "
        f"act: {sum_loss_action/n:.4f}  "
        f"src: {sum_loss_src/n:.4f}  "
        f"tgt: {sum_loss_tgt/n:.4f}  "
        f"label: {sum_loss_label/n:.4f}"
    )