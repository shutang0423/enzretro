"""
pretrain.py —— Actor Network 监督预训练主循环

职责：
  - 构建数据集 / DataLoader
  - 初始化模型 + Loss 策略
  - 分阶段课程学习（由 TRAIN_CFG.stages 驱动）
  - TensorBoard 记录 + 最优模型保存
  - 训练结束后在测试集评估

数据流：
  SSRGraphDataset → DataLoader → batch
    batch.x / edge_index / batch_idx   → GraphEncoder
    batch.history_*                    → StateTracker (HistoryBatch)
    batch.target_*                     → TeacherForcingTargets + Loss

消融实验切换（只改 config）：
  MODEL_CFG.encoder_type  = "gat" | "fingerprint"
  TRAIN_CFG.loss_strategy = "uncertainty" | "equal" | "manual" | "single_task"
  TRAIN_CFG.stages        = [单阶段联合] 或 [多阶段课程]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from dataclasses import asdict

from config.config import PATH_CFG, MODEL_CFG, TRAIN_CFG, StageConfig
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_network import ActorNetwork, TeacherForcingTargets
from model.state_tracker    import HistoryBatch
from model.loss_strategy    import build_loss_strategy, LossStrategyBase


# ══════════════════════════════════════════════════════════════════════
#  设备
# ══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[pretrain] device = {device}")


# ══════════════════════════════════════════════════════════════════════
#  数据集 & DataLoader
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(tokenizer: LabelTokenizer):
    """构建训练 / 验证 / 测试 DataLoader"""
    def _make(path, shuffle):
        ds = SSRGraphDataset(
            json_path    = str(path),
            tokenizer    = tokenizer,
            max_seq_len  = MODEL_CFG.max_seq_len,
            max_hist_len = MODEL_CFG.max_hist_len,
        )
        return DataLoader(
            ds,
            batch_size = TRAIN_CFG.batch_size,
            shuffle    = shuffle,
            num_workers= 4,
            pin_memory = True,
        )

    return (
        _make(PATH_CFG.PRETRAIN_TRAIN_DATA_FILE, shuffle=True),
        _make(PATH_CFG.PRETRAIN_VAL_DATA_FILE,   shuffle=False),
        _make(PATH_CFG.PRETRAIN_TEST_DATA_FILE,  shuffle=False),
    )


# ══════════════════════════════════════════════════════════════════════
#  模型初始化
# ══════════════════════════════════════════════════════════════════════

def build_model(vocab_size: int) -> ActorNetwork:
    """构建模型并应用权重初始化"""
    # 将 ModelConfig dataclass 转为 dict 传给模型
    model    = ActorNetwork(vocab_size=vocab_size).to(device)

    def _init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_init_weights)
    return model


# ══════════════════════════════════════════════════════════════════════
#  冻结 / 解冻工具
# ══════════════════════════════════════════════════════════════════════

def set_requires_grad(model: ActorNetwork, freeze: list[str]) -> None:
    """解冻所有参数，再按名称冻结指定模块"""
    for p in model.parameters():
        p.requires_grad = True
    for module_name in freeze:
        module = getattr(model, module_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False


# ══════════════════════════════════════════════════════════════════════
#  优化器 & 调度器
# ══════════════════════════════════════════════════════════════════════

def build_optimizer_scheduler(
    model       : ActorNetwork,
    loss_strategy: LossStrategyBase,
    total_steps : int,
    lr          : float,
):
    """AdamW + Cosine with Warmup"""
    params    = (
        [p for p in model.parameters()      if p.requires_grad]
        + list(loss_strategy.parameters())
    )
    optimizer = optim.AdamW(params, lr=lr, weight_decay=TRAIN_CFG.weight_decay)
    warmup    = int(total_steps * TRAIN_CFG.warmup_ratio)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    return optimizer, LambdaLR(optimizer, _lr_lambda)


# ══════════════════════════════════════════════════════════════════════
#  Batch 解包：将 PyG batch 转为结构化容器
# ══════════════════════════════════════════════════════════════════════

def unpack_batch(batch) -> tuple[HistoryBatch, TeacherForcingTargets, dict]:
    """
    将 DataLoader 返回的 PyG batch 解包为：
      history  : HistoryBatch（完整历史动作序列）
      tf       : TeacherForcingTargets（训练目标）
      graph_kw : dict（传给 encoder 的关键字参数）
    """
    history = HistoryBatch(
        actions    = batch.history_actions,      # [B, T]
        src_idxs   = batch.history_src_idxs,    # [B, T]
        tgt_idxs   = batch.history_tgt_idxs,    # [B, T]
        label_seqs = batch.history_label_seqs,  # [B, T, L]
    )
    tf = TeacherForcingTargets(
        action    = batch.target_action.squeeze(-1),   # [B]
        src       = batch.target_src.squeeze(-1),      # [B]
        label_seq = batch.target_label,                # [B, L]
    )
    # 根据 encoder 类型决定传入图数据还是指纹
    if MODEL_CFG.encoder_type == "gat":
        graph_kw = dict(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
    else:
        graph_kw = dict(fingerprint=batch.fingerprint)

    return history, tf, graph_kw


# ══════════════════════════════════════════════════════════════════════
#  Loss 计算（含安全检查）
# ══════════════════════════════════════════════════════════════════════

def compute_losses(
    action_logits : torch.Tensor,   # [B, num_actions]
    src_logits    : torch.Tensor,   # [B, max_atoms]
    tgt_logits    : torch.Tensor,   # [B, max_atoms]
    label_logits  : torch.Tensor,   # [B, L, vocab_size]
    tf            : TeacherForcingTargets,
    target_tgt    : torch.Tensor,   # [B]
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
    active_tasks  : set[str],
) -> dict[str, torch.Tensor]:
    """
    计算四项 loss，对无效索引做安全过滤
    active_tasks 控制哪些 loss 参与梯度（课程学习用）
    """
    def _safe(logits, target, criterion, name):
        loss = criterion(logits, target)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [NaN/Inf] {name} loss，替换为 0")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss

    loss_action = _safe(action_logits, tf.action, criterion_cls, "action")

    N         = src_logits.size(-1)
    valid_src = (tf.src >= 0) & (tf.src < N)
    loss_src  = (
        _safe(src_logits[valid_src], tf.src[valid_src], criterion_cls, "src")
        if valid_src.any()
        else torch.tensor(0.0, device=src_logits.device, requires_grad=True)
    )
    valid_tgt = (target_tgt >= 0) & (target_tgt < N)
    loss_tgt  = (
        _safe(tgt_logits[valid_tgt], target_tgt[valid_tgt], criterion_cls, "tgt")
        if valid_tgt.any()
        else torch.tensor(0.0, device=tgt_logits.device, requires_grad=True)
    )
    loss_label = _safe(
        label_logits.reshape(-1, vocab_size),
        tf.label_seq[:, 1:].reshape(-1),   # 目标去掉 BOS
        criterion_seq, "label",
    )

    # 课程学习：非激活任务 detach，不传梯度
    def _maybe_detach(loss, name):
        return loss if name in active_tasks else loss.detach()

    return {
        "action": _maybe_detach(loss_action, "action"),
        "src"   : _maybe_detach(loss_src,    "src"),
        "tgt"   : _maybe_detach(loss_tgt,    "tgt"),
        "label" : _maybe_detach(loss_label,  "label"),
    }


# ══════════════════════════════════════════════════════════════════════
#  评估函数（验证 / 测试通用）
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model         : ActorNetwork,
    loader        : DataLoader,
    loss_strategy : LossStrategyBase,
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
    split         : str = "val",
) -> dict[str, float]:
    model.eval()
    sums = {k: 0.0 for k in ["total", "action", "src", "tgt", "label"]}
    n    = 0

    for batch in loader:
        batch = batch.to(device)
        history, tf, graph_kw = unpack_batch(batch)
        target_tgt = batch.target_tgt.squeeze(-1)

        action_logits, src_logits, tgt_logits, label_logits, _ = model(
            history=history, tf=tf, **graph_kw
        )
        losses = compute_losses(
            action_logits, src_logits, tgt_logits, label_logits,
            tf, target_tgt, vocab_size,
            criterion_cls, criterion_seq,
            active_tasks={"action", "src", "tgt", "label"},  # 评估时全部计算
        )
        total, _ = loss_strategy(
            losses["action"], losses["src"], losses["tgt"], losses["label"]
        )
        sums["total"]  += total.item()
        for k in ["action", "src", "tgt", "label"]:
            sums[k] += losses[k].item()
        n += 1

    avg = {k: v / max(n, 1) for k, v in sums.items()}
    print(
        f"  📊 [{split}] "
        f"total:{avg['total']:.4f}  act:{avg['action']:.4f}  "
        f"src:{avg['src']:.4f}  tgt:{avg['tgt']:.4f}  label:{avg['label']:.4f}"
    )
    return avg


# ══════════════════════════════════════════════════════════════════════
#  单阶段训练
# ══════════════════════════════════════════════════════════════════════

def train_stage(
    stage         : StageConfig,
    model         : ActorNetwork,
    loss_strategy : LossStrategyBase,
    train_loader  : DataLoader,
    val_loader    : DataLoader,
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
    writer        : SummaryWriter,
    global_state  : dict,           # {"epoch": int, "step": int, "best_loss": float}
) -> None:
    """执行单个训练阶段，更新 global_state"""

    print(f"\n{'='*60}")
    print(f"  {stage.name}  tasks={stage.tasks}  epochs={stage.epochs}")
    print(f"{'='*60}")

    set_requires_grad(model, stage.freeze)
    active_tasks = set(stage.tasks)
    stage_lr     = TRAIN_CFG.lr * stage.lr_scale
    total_steps  = stage.epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_scheduler(
        model, loss_strategy, total_steps, stage_lr
    )

    for epoch in range(stage.epochs):
        global_state["epoch"] += 1
        model.train()
        sums       = {k: 0.0 for k in ["total", "action", "src", "tgt", "label"]}
        n_batches  = 0

        pbar = tqdm(
            train_loader,
            desc   = f"{stage.name} Ep[{epoch+1}/{stage.epochs}]",
            ncols  = 160,
            leave  = True,
        )

        for batch in pbar:
            batch = batch.to(device)
            history, tf, graph_kw = unpack_batch(batch)
            target_tgt = batch.target_tgt.squeeze(-1)

            optimizer.zero_grad()

            action_logits, src_logits, tgt_logits, label_logits, _ = model(
                history=history, tf=tf, **graph_kw
            )
            losses = compute_losses(
                action_logits, src_logits, tgt_logits, label_logits,
                tf, target_tgt, vocab_size,
                criterion_cls, criterion_seq, active_tasks,
            )
            total, _ = loss_strategy(
                losses["action"], losses["src"], losses["tgt"], losses["label"]
            )

            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.max_norm)
            optimizer.step()
            scheduler.step()
            global_state["step"] += 1

            # ── 累计统计 ──────────────────────────────────────────────
            sums["total"]  += total.item()
            for k in ["action", "src", "tgt", "label"]:
                sums[k] += losses[k].item()
            n_batches += 1

            # ── Step 级 TensorBoard ───────────────────────────────────
            cur_lr = scheduler.get_last_lr()[0]
            writer.add_scalars(
                f"{stage.name}/step_loss",
                {k: losses[k].item() for k in ["action", "src", "tgt", "label"]}
                | {"total": total.item()},
                global_state["step"],
            )
            writer.add_scalar("lr", cur_lr, global_state["step"])

            # ── 记录 UW 权重（如果是 UncertaintyWeighting）────────────
            w = loss_strategy.weights()
            writer.add_scalars(
                "uw/weights", w, global_state["step"]
            )

            # ── tqdm 后缀 ─────────────────────────────────────────────
            pbar.set_postfix({
                "tot"  : f"{total.item():.3f}",
                "act"  : f"{losses['action'].item():.3f}(w={w.get('action',1):.2f})",
                "src"  : f"{losses['src'].item():.3f}(w={w.get('src',1):.2f})",
                "tgt"  : f"{losses['tgt'].item():.3f}(w={w.get('tgt',1):.2f})",
                "lbl"  : f"{losses['label'].item():.3f}(w={w.get('label',1):.2f})",
                "lr"   : f"{cur_lr:.1e}",
            })

        # ── Epoch 级 TensorBoard ──────────────────────────────────────
        n   = max(n_batches, 1)
        avg = {k: v / n for k, v in sums.items()}
        writer.add_scalars(
            f"{stage.name}/epoch_loss", avg, global_state["epoch"]
        )
        writer.add_scalars(
            f"{stage.name}/uw_weights",
            loss_strategy.weights(), global_state["epoch"]
        )
        print(
            f"  ▶ [Ep {global_state['epoch']}] {stage.name} | "
            + "  ".join(f"{k}:{avg[k]:.4f}" for k in ["total","action","src","tgt","label"])
        )

        # ── 验证 & 保存最优 ───────────────────────────────────────────
        if global_state["epoch"] % TRAIN_CFG.val_every_epoch == 0:
            val_avg = evaluate(
                model, val_loader, loss_strategy, vocab_size,
                criterion_cls, criterion_seq, split="val",
            )
            writer.add_scalars(
                f"{stage.name}/val_loss", val_avg, global_state["epoch"]
            )
            metric = val_avg[TRAIN_CFG.save_best_metric]
            if metric < global_state["best_loss"]:
                global_state["best_loss"] = metric
                torch.save(
                    {
                        "model"     : model.state_dict(),
                        "strategy"  : loss_strategy.state_dict(),
                        "loss"      : metric,
                        "epoch"     : global_state["epoch"],
                        "model_cfg" : MODEL_CFG,
                        "train_cfg" : TRAIN_CFG,
                    },
                    PATH_CFG.CKPT_BEST_MODEL_FILE,
                )
                print(
                    f"  ✅ Best model saved  "
                    f"val_{TRAIN_CFG.save_best_metric}={metric:.4f}"
                    f" → {PATH_CFG.CKPT_BEST_MODEL_FILE}"
                )


# ══════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── 分词器 & 数据集 ───────────────────────────────────────────────
    tokenizer = LabelTokenizer(vocab_file=str(PATH_CFG.VOCAB_FILE))
    train_loader, val_loader, test_loader = build_dataloaders(tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[pretrain] vocab_size={vocab_size}  "
          f"train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    # ── 模型 ─────────────────────────────────────────────────────────
    model = build_model(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] 模型参数量: {total_params:,}")

    # ── Loss 策略（消融实验切换点）────────────────────────────────────
    loss_strategy = build_loss_strategy({
        "loss_strategy": TRAIN_CFG.loss_strategy,
        "loss_weights" : TRAIN_CFG.loss_weights,
        "active_tasks" : TRAIN_CFG.active_tasks,
        "uw_s_min"     : TRAIN_CFG.uw_s_min,
        "uw_s_max"     : TRAIN_CFG.uw_s_max,
    }).to(device)

    # ── Loss 函数 ─────────────────────────────────────────────────────
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_seq = nn.CrossEntropyLoss(ignore_index=MODEL_CFG.pad_token_id)

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(PATH_CFG.TB_DIR))
    print(f"[pretrain] TensorBoard → tensorboard --logdir {PATH_CFG.TB_DIR}")

    # ── 全局状态（跨阶段共享）────────────────────────────────────────
    global_state = {"epoch": 0, "step": 0, "best_loss": float("inf")}

    # ── 分阶段课程训练 ────────────────────────────────────────────────
    for stage in TRAIN_CFG.stages:
        train_stage(
            stage         = stage,
            model         = model,
            loss_strategy = loss_strategy,
            train_loader  = train_loader,
            val_loader    = val_loader,
            vocab_size    = vocab_size,
            criterion_cls = criterion_cls,
            criterion_seq = criterion_seq,
            writer        = writer,
            global_state  = global_state,
        )

    # ── 保存最后一个 epoch 的模型 ─────────────────────────────────────
    torch.save(
        {"model": model.state_dict(), "strategy": loss_strategy.state_dict()},
        PATH_CFG.CKPT_LAST_MODEL_FILE,
    )
    print(f"\n[pretrain] Last model saved → {PATH_CFG.CKPT_LAST_MODEL_FILE}")

    # ── 测试集评估（加载最优模型）────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🧪 Loading best model for final test ...")
    ckpt = torch.load(
        PATH_CFG.CKPT_BEST_MODEL_FILE,
        map_location=device,
        weights_only=False,   # 需要加载 dataclass 对象
    )
    model.load_state_dict(ckpt["model"])
    loss_strategy.load_state_dict(ckpt["strategy"])
    print(f"  Best ckpt: epoch={ckpt['epoch']}  val_loss={ckpt['loss']:.4f}")

    test_avg = evaluate(
        model, test_loader, loss_strategy, vocab_size,
        criterion_cls, criterion_seq, split="test",
    )
    writer.add_scalars("final/test_loss", test_avg, 0)
    writer.close()
    print(f"\n[pretrain] 训练完成 ✅  project={PATH_CFG.project_name}")


if __name__ == "__main__":
    main()