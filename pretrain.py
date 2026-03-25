import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter   # ← 新增
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

# ══════════════════════════════════════════════════════════════════════════
#  TensorBoard Writer
# ══════════════════════════════════════════════════════════════════════════
writer = SummaryWriter(log_dir=PC.get("log_dir", "ckpt/pretrain"))
# 启动命令（终端执行）: tensorboard --logdir ckpt/pretrain

# ══════════════════════════════════════════════════════════════════════════
#  Uncertainty Weighting
# ══════════════════════════════════════════════════════════════════════════
class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks: int,
                 s_min: float = -2.5,   
                 s_max: float =  2.5):  
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
        self.s_min = s_min
        self.s_max = s_max

    def forward(self, *losses):
        # 每次 forward 时 clamp，防止梯度把参数推到极端
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        weighted = [torch.exp(-s[i]) * l + s[i]
                    for i, l in enumerate(losses)]
        return sum(weighted), [l.item() for l in weighted]

    def weights(self):
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        return torch.exp(-s).detach().cpu().tolist()


# ══════════════════════════════════════════════════════════════════════════
#  分阶段课程训练配置
# ══════════════════════════════════════════════════════════════════════════
# STAGES = [
#     {"name": "Stage1-Action",  "epochs": TC.get("stage1_epochs", 10),
#      "tasks": ["action"],           "freeze": ["pointer_network", "label_decoder"]},
#     {"name": "Stage2-Pointer", "epochs": TC.get("stage2_epochs", 30),
#      "tasks": ["src", "tgt"],       "freeze": ["action_predictor", "label_decoder"]},
#     {"name": "Stage3-Label",   "epochs": TC.get("stage3_epochs", 30),
#      "tasks": ["label"],            "freeze": ["action_predictor", "pointer_network"]},
#     {"name": "Stage4-Joint",   "epochs": TC.get("stage4_epochs", 60),
#      "tasks": ["action", "src", "tgt", "label"], "freeze": []},
# ]

# ══════════════════════════════════════════════════════════════════════════
#  直接进行多任务联合训练配置
# ══════════════════════════════════════════════════════════════════════════
STAGES = [
    {
        "name": "Stage-Joint-All", 
        "epochs": TC.get("stage4_epochs", 200),  # 建议适当增加 Epoch 数量
        "tasks": ["action", "src", "tgt", "label"], 
        "freeze": []  # 不冻结任何模块，全参数更新
    }
]


def set_freeze(model, freeze_modules):
    for p in model.parameters():
        p.requires_grad = True
    for name in freeze_modules:
        module = getattr(model, name, None)
        if module:
            for p in module.parameters():
                p.requires_grad = False


def make_optimizer_scheduler(model, uw, total_steps, lr):
    params    = [p for p in model.parameters() if p.requires_grad] + list(uw.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-2)
    warmup    = int(total_steps * 0.1)
    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))
    return optimizer, LambdaLR(optimizer, lr_lambda)


def safe_loss(logits, target, criterion, name):
    loss = criterion(logits, target)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n  [NaN] {name}")
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return loss


# ══════════════════════════════════════════════════════════════════════════
#  数据 & 模型
# ══════════════════════════════════════════════════════════════════════════
tokenizer  = LabelTokenizer(vocab_file=PC["vocab_file"])
train_dataset    = SSRGraphDataset(
    json_path=PC["train_data"], tokenizer=tokenizer,
    max_seq_len=C["max_seq_len"], max_hist_len=C["max_hist_len"],
)
train_dataloader = DataLoader(train_dataset, batch_size=TC["batch_size"], shuffle=True)
vocab_size = tokenizer.get_vocab_size()
# ── 验证集 & 测试集 DataLoader ──────────────────────────────────────────
val_dataset  = SSRGraphDataset(
    json_path=PC["val_data"], tokenizer=tokenizer,
    max_seq_len=C["max_seq_len"], max_hist_len=C["max_hist_len"],
)
test_dataset = SSRGraphDataset(
    json_path=PC["test_data"], tokenizer=tokenizer,
    max_seq_len=C["max_seq_len"], max_hist_len=C["max_hist_len"],
)
val_loader  = DataLoader(val_dataset,  batch_size=TC["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TC["batch_size"], shuffle=False)

model = ActorPretrainer(vocab_size=vocab_size).to(device)

def init_weights(m):
    if isinstance(m, nn.Embedding):   nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_weights)

uw            = UncertaintyWeighting(num_tasks=4).to(device)
criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
criterion_seq = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ══════════════════════════════════════════════════════════════════════════
#  Evaluate（验证 / 测试通用）
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(loader, split="val"):
    model.eval()
    sum_loss = sum_act = sum_src = sum_tgt = sum_lbl = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)

        node_embeddings, graph_embedding = model.graph_encoder(
            batch.x, batch.edge_index, batch.batch)
        graph_state   = model.state_proj(graph_embedding)
        decoder_state = model.state_tracker(batch.history_actions, graph_state)

        target_action = batch.target_action.squeeze(-1)
        target_src    = batch.target_src.squeeze(-1)
        target_tgt    = batch.target_tgt.squeeze(-1)

        action_logits = model.action_predictor(decoder_state)
        loss_action   = criterion_cls(action_logits, target_action)

        dense_nodes, node_mask_bool = to_dense_batch(node_embeddings, batch.batch)
        src_logits, tgt_logits = model.pointer_network(
            decoder_state, dense_nodes, target_action,
            target_src_idx=target_src, node_mask=~node_mask_bool)

        N         = src_logits.size(-1)
        valid_src = (target_src >= 0) & (target_src < N)
        valid_tgt = (target_tgt >= 0) & (target_tgt < N)
        loss_src  = criterion_cls(src_logits[valid_src], target_src[valid_src]) if valid_src.any() else torch.tensor(0.0)
        loss_tgt  = criterion_cls(tgt_logits[valid_tgt], target_tgt[valid_tgt]) if valid_tgt.any() else torch.tensor(0.0)

        label_logits = model.label_decoder(
            decoder_state, target_action, batch.target_label[:, :-1])
        loss_label = criterion_seq(
            label_logits.reshape(-1, vocab_size),
            batch.target_label[:, 1:].reshape(-1))

        # 用 UW 加权得到总 loss（与训练一致）
        total, _ = uw(loss_action, loss_src, loss_tgt, loss_label)

        sum_loss += total.item()
        sum_act  += loss_action.item()
        sum_src  += loss_src.item()
        sum_tgt  += loss_tgt.item()
        sum_lbl  += loss_label.item()
        n        += 1

    avg = {
        "total":  sum_loss / max(n, 1),
        "action": sum_act  / max(n, 1),
        "src":    sum_src  / max(n, 1),
        "tgt":    sum_tgt  / max(n, 1),
        "label":  sum_lbl  / max(n, 1),
    }
    print(f"  📊 [{split}] total:{avg['total']:.4f}  act:{avg['action']:.4f}  "
          f"src:{avg['src']:.4f}  tgt:{avg['tgt']:.4f}  label:{avg['label']:.4f}")
    return avg


# ══════════════════════════════════════════════════════════════════════════
#  保存最优模型
# ══════════════════════════════════════════════════════════════════════════
best_loss = float("inf")
CKPT_PATH = PC.get("ckpt_path", "best_model.pt")

def save_best(avg_loss, epoch_tag):
    global best_loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "model": model.state_dict(), "uw": uw.state_dict(),
            "loss": best_loss, "epoch_tag": epoch_tag,
        }, CKPT_PATH)
        print(f"  ✅ Best model saved  loss={best_loss:.4f}  → {CKPT_PATH}")


# ══════════════════════════════════════════════════════════════════════════
#  分阶段训练主循环
# ══════════════════════════════════════════════════════════════════════════
global_epoch = 0
global_step  = 0   # ← step 级别记录用

for stage in STAGES:
    stage_name   = stage["name"]
    stage_epochs = stage["epochs"]
    active_tasks = set(stage["tasks"])
    is_joint     = (stage_name == "Stage4-Joint")

    print(f"\n{'='*60}\n  {stage_name}  tasks={active_tasks}  epochs={stage_epochs}\n{'='*60}")
    set_freeze(model, stage["freeze"])

    stage_lr = TC["lr"] / 5 if is_joint else TC["lr"]
    optimizer, scheduler = make_optimizer_scheduler(
        model, uw, stage_epochs * len(train_dataloader), lr=stage_lr
    )

    for epoch in range(stage_epochs):
        global_epoch += 1
        sum_loss = sum_act = sum_src = sum_tgt = sum_lbl = 0.0
        num_batches = 0

        pbar = tqdm(train_dataloader,
                    desc=f"{stage_name} Ep[{epoch+1}/{stage_epochs}]",
                    ncols=150, leave=True)
        model.train()

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            node_embeddings, graph_embedding = model.graph_encoder(
                batch.x, batch.edge_index, batch.batch)
            graph_state   = model.state_proj(graph_embedding)
            decoder_state = model.state_tracker(batch.history_actions, graph_state)

            target_action = batch.target_action.squeeze(-1)
            target_src    = batch.target_src.squeeze(-1)
            target_tgt    = batch.target_tgt.squeeze(-1)

            action_logits = model.action_predictor(decoder_state)
            loss_action   = safe_loss(action_logits, target_action, criterion_cls, "action")

            dense_nodes, node_mask_bool = to_dense_batch(node_embeddings, batch.batch)
            src_logits, tgt_logits = model.pointer_network(
                decoder_state, dense_nodes, target_action,
                target_src_idx=target_src, node_mask=~node_mask_bool)

            N         = src_logits.size(-1)
            valid_src = (target_src >= 0) & (target_src < N)
            valid_tgt = (target_tgt >= 0) & (target_tgt < N)
            loss_src  = (safe_loss(src_logits[valid_src], target_src[valid_src], criterion_cls, "src")
                         if valid_src.any() else torch.tensor(0.0, device=device, requires_grad=True))
            loss_tgt  = (safe_loss(tgt_logits[valid_tgt], target_tgt[valid_tgt], criterion_cls, "tgt")
                         if valid_tgt.any() else torch.tensor(0.0, device=device, requires_grad=True))

            label_logits = model.label_decoder(
                decoder_state, target_action, batch.target_label[:, :-1])
            loss_label   = safe_loss(
                label_logits.reshape(-1, vocab_size),
                batch.target_label[:, 1:].reshape(-1), criterion_seq, "label")

            if "action" not in active_tasks: loss_action = loss_action.detach()
            if "src"    not in active_tasks: loss_src    = loss_src.detach()
            if "tgt"    not in active_tasks: loss_tgt    = loss_tgt.detach()
            if "label"  not in active_tasks: loss_label  = loss_label.detach()

            loss, _ = uw(loss_action, loss_src, loss_tgt, loss_label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # ── Step 级别写入（每步记录，曲线最细粒度）────────────────
            cur_lr = scheduler.get_last_lr()[0]
            writer.add_scalars(f"{stage_name}/step_loss", {
                "total":  loss.item(),
                "action": loss_action.item(),
                "src":    loss_src.item(),
                "tgt":    loss_tgt.item(),
                "label":  loss_label.item(),
            }, global_step)
            writer.add_scalar("lr", cur_lr, global_step)
            writer.add_scalars("uw/log_sigma", {
                f"task_{i}": v.item()
                for i, v in enumerate(uw.log_sigma)
            }, global_step)

            la, ls, lt, ll = (loss_action.item(), loss_src.item(),
                              loss_tgt.item(),    loss_label.item())
            tot = loss.item()
            sum_loss += tot; sum_act += la; sum_src += ls
            sum_tgt  += lt;  sum_lbl += ll; num_batches += 1

            w = uw.weights()
            pbar.set_postfix({
                "tot":   f"{tot:.3f}",
                "act":   f"{la:.3f}({w[0]:.2f})",
                "src":   f"{ls:.3f}({w[1]:.2f})",
                "tgt":   f"{lt:.3f}({w[2]:.2f})",
                "label": f"{ll:.3f}({w[3]:.2f})",
                "lr":    f"{cur_lr:.1e}",
            })

        # ── Epoch 级别写入（每 epoch 记录平均 loss）───────────────────
        n   = max(num_batches, 1)
        avg = sum_loss / n
        w   = uw.weights()

        writer.add_scalars(f"{stage_name}/epoch_loss", {
            "total":  avg,
            "action": sum_act / n,
            "src":    sum_src / n,
            "tgt":    sum_tgt / n,
            "label":  sum_lbl / n,
        }, global_epoch)
        writer.add_scalars(f"{stage_name}/uw_weights", {
            "action": w[0], "src": w[1], "tgt": w[2], "label": w[3],
        }, global_epoch)

        print(f"  ▶ [Ep {global_epoch}] {stage_name} avg | "
              f"total:{avg:.4f} act:{sum_act/n:.4f}(w={w[0]:.2f}) "
              f"src:{sum_src/n:.4f}(w={w[1]:.2f}) "
              f"tgt:{sum_tgt/n:.4f}(w={w[2]:.2f}) "
              f"label:{sum_lbl/n:.4f}(w={w[3]:.2f})")

        # if is_joint:
        # save_best(avg, f"ep{global_epoch}")
        # ── 验证 & 保存最优（按验证集 loss）────────────────────────────
        val_avg = evaluate(val_loader, split="val")
        writer.add_scalars(f"{stage_name}/val_loss", val_avg, global_epoch)
        save_best(val_avg["total"], f"ep{global_epoch}")   # ← 改用验证集 loss



# ══════════════════════════════════════════════════════════════════════════
#  测试：加载最优模型后评估
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🧪 Loading best model for final test ...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
uw.load_state_dict(ckpt["uw"])
print(f"  Best ckpt: epoch={ckpt['epoch_tag']}  val_loss={ckpt['loss']:.4f}")

test_avg = evaluate(test_loader, split="test")
writer.add_scalars("final/test_loss", test_avg, 0)


writer.close()   # ← 训练结束关闭 writer