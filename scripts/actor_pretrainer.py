"""
pretrain.py —— Actor Network 监督预训练主循环

职责：
  - 构建数据集 / DataLoader
  - 初始化模型 + Loss 策略
  - 分阶段课程学习（由 TRAIN_CFG.stages 驱动）
  - TensorBoard 记录 + 最优模型保存
  - 训练结束后在测试集评估

数据流：
  batch (PyG)
    → unpack_batch()
        → history    : HistoryBatch
        → tf         : TeacherForcingTargets
        → graph_kw   : dict (encoder 关键字参数)
    → ActorNetwork.forward(history, tf, **graph_kw)
        → action_logits / src_logits / tgt_logits / label_logits
    → compute_raw_losses()
    → LossStrategy.forward()
    → backward + clip + step
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config.config import PATH_CFG, MODEL_CFG, TRAIN_CFG, StageConfig, TASK_NAMES
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_network import ActorNetwork, TeacherForcingTargets
from model.state_tracker import HistoryBatch
from model.loss_strategy import build_loss_strategy, LossStrategyBase


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
    def _make(path, shuffle: bool) -> DataLoader:
        ds = SSRGraphDataset(
            json_path    = str(path),
            tokenizer    = tokenizer,
            max_seq_len  = MODEL_CFG.max_seq_len,
            max_hist_len = MODEL_CFG.max_hist_len,
        )
        return DataLoader(
            ds,
            batch_size  = TRAIN_CFG.batch_size,
            shuffle     = shuffle,
            num_workers = 4,
            pin_memory  = True,
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
    model = ActorNetwork(vocab_size=vocab_size).to(device)

    def _init_weights(m: nn.Module) -> None:
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
#  冻结 / 解冻
# ══════════════════════════════════════════════════════════════════════

def set_requires_grad(model: ActorNetwork, freeze: list[str]) -> None:
    """解冻所有参数，再按模块名冻结指定子模块"""
    for p in model.parameters():
        p.requires_grad = True
    for name in freeze:
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False
        else:
            print(f"[warn] freeze target not found: {name!r}")


# ══════════════════════════════════════════════════════════════════════
#  优化器 & 调度器
# ══════════════════════════════════════════════════════════════════════

def build_optimizer_scheduler(
    model         : ActorNetwork,
    loss_strategy : LossStrategyBase,
    total_steps   : int,
    lr            : float,
):
    """AdamW + Cosine Warmup（使用 math.cos 避免 torch 标量开销）"""
    params    = (
        [p for p in model.parameters() if p.requires_grad]
        + list(loss_strategy.parameters())
    )
    optimizer = optim.AdamW(params, lr=lr, weight_decay=TRAIN_CFG.weight_decay)
    warmup    = int(total_steps * TRAIN_CFG.warmup_ratio)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optimizer, LambdaLR(optimizer, _lr_lambda)


# ══════════════════════════════════════════════════════════════════════
#  Batch 解包
# ══════════════════════════════════════════════════════════════════════

def unpack_batch(
    batch,
) -> tuple[HistoryBatch, TeacherForcingTargets, dict, torch.Tensor]:
    """
    将 PyG DataLoader 返回的 batch 解包为 ActorNetwork.forward 所需的入参。

    形状修复（PyG collate 会沿 dim=0 拼接，导致多维张量被压平）：
      batch.target_label      : [B*L]      → reshape [B, L]
      batch.target_action     : [B] 或 [B,1]
      batch.target_src        : [B] 或 [B,1]
      batch.target_tgt        : [B] 或 [B,1]
      batch.history_actions   : [B*T]      → reshape [B, T]
      batch.history_src_idxs  : [B*T]      → reshape [B, T]
      batch.history_tgt_idxs  : [B*T]      → reshape [B, T]
      batch.history_label_seqs: [B*T*L]    → reshape [B, T, L]

    越界防御：
      action clamp [0, num_actions-1]
      atom   clamp [0, max_atoms-1]

    Returns:
      history    : HistoryBatch
      tf         : TeacherForcingTargets
      graph_kw   : dict  (encoder 关键字参数)
      target_tgt : Tensor [B]  (供 compute_raw_losses 使用)
    """
    B        = int(batch.num_graphs)
    max_seq  = MODEL_CFG.max_seq_len
    max_hist = MODEL_CFG.max_hist_len
    max_a    = MODEL_CFG.num_actions - 1
    max_n    = MODEL_CFG.max_atoms   - 1

    # ── 辅助：安全 reshape（形状已对则直接返回，否则 reshape）────────
    def _reshape(t: torch.Tensor, *shape) -> torch.Tensor:
        if t.shape == torch.Size(shape):
            return t
        return t.reshape(*shape)

    # ── target 字段 ───────────────────────────────────────────────────
    target_action = _reshape(batch.target_action, B).clamp(0, max_a)   # [B]
    target_src    = _reshape(batch.target_src,    B).clamp(0, max_n)   # [B]
    target_tgt    = _reshape(batch.target_tgt,    B).clamp(0, max_n)   # [B]
    target_label  = _reshape(batch.target_label,  B, max_seq)          # [B, L]

    tf = TeacherForcingTargets(
        action    = target_action,
        src       = target_src,
        label_seq = target_label,
    )

    # ── history 字段 ──────────────────────────────────────────────────
    h_actions    = _reshape(batch.history_actions,    B, max_hist).clamp(0, max_a)  # [B, T]
    h_src_idxs   = _reshape(batch.history_src_idxs,  B, max_hist).clamp(0, max_n)  # [B, T]
    h_tgt_idxs   = _reshape(batch.history_tgt_idxs,  B, max_hist).clamp(0, max_n)  # [B, T]
    h_label_seqs = _reshape(batch.history_label_seqs, B, max_hist, max_seq)         # [B, T, L]

    history = HistoryBatch(
        actions    = h_actions,
        src_idxs   = h_src_idxs,
        tgt_idxs   = h_tgt_idxs,
        label_seqs = h_label_seqs,
    )


    # ── encoder kwargs ────────────────────────────────────────────────────
    graph_kw = dict(
        x          = batch.x,
        edge_index = batch.edge_index,
        batch      = batch.batch,
    )

    return history, tf, graph_kw, target_tgt


# ══════════════════════════════════════════════════════════════════════
#  原始 Loss 计算（只做 CE，不做加权）
# ══════════════════════════════════════════════════════════════════════

def compute_raw_losses(
    action_logits : torch.Tensor,   # [B, num_actions]
    src_logits    : torch.Tensor,   # [B, max_atoms]
    tgt_logits    : torch.Tensor,   # [B, max_atoms]
    label_logits  : torch.Tensor,   # [B, L-1, vocab_size]  ← ActorNetwork 内部已截断
    tf            : TeacherForcingTargets,
    target_tgt    : torch.Tensor,   # [B]
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
) -> dict[str, torch.Tensor]:
    """
    计算四项原始 CrossEntropy loss，不做加权。

    Teacher Forcing 对齐说明：
      ActorNetwork 内部以 label_seq[:, :-1] 作为 decoder 输入
        → label_logits 形状为 [B, L-1, V]
      对应的预测目标为 label_seq[:, 1:]（去掉 BOS，保留到 EOS）
        → 形状 [B, L-1]
      两者在 L-1 维度上严格对齐。

    src / tgt / label 只在非 STOP 动作的样本上计算。

    Returns:
      {"action": Tensor, "src": Tensor, "tgt": Tensor, "label": Tensor}
    """
    # 用于无有效样本时的零梯度占位（保持在同一设备 & 计算图上）
    dummy = action_logits.sum() * 0.0

    # ── action loss（全样本）────────────────────────────────────────
    loss_action = criterion_cls(action_logits, tf.action)

    # ── 非 STOP 掩码 ────────────────────────────────────────────────
    non_stop = (tf.action != MODEL_CFG.stop_action_id)   # [B]  bool

    # ── src loss ────────────────────────────────────────────────────
    if non_stop.any():
        loss_src = criterion_cls(
            src_logits[non_stop],   # [M, max_atoms]
            tf.src[non_stop],       # [M]
        )
    else:
        loss_src = dummy

    # ── tgt loss ────────────────────────────────────────────────────
    if non_stop.any():
        loss_tgt = criterion_cls(
            tgt_logits[non_stop],   # [M, max_atoms]
            target_tgt[non_stop],   # [M]
        )
    else:
        loss_tgt = dummy

    # ── label loss ──────────────────────────────────────────────────
    # label_logits : [B, L-1, V]  (decoder 以 label_seq[:,:-1] 为输入)
    # 对齐目标     : label_seq[:, 1:]  → [B, L-1]  (去掉 BOS)
    if non_stop.any():
        ll = label_logits[non_stop]              # [M, L-1, V]
        lt = tf.label_seq[non_stop, 1:]          # [M, L-1]  ← 关键修复：取 1: 而非全部
        loss_label = criterion_seq(
            ll.reshape(-1, vocab_size),           # [M*(L-1), V]
            lt.reshape(-1),                       # [M*(L-1)]
        )
    else:
        loss_label = dummy

    return {
        "action": loss_action,
        "src"   : loss_src,
        "tgt"   : loss_tgt,
        "label" : loss_label,
    }


# ══════════════════════════════════════════════════════════════════════
#  单 Epoch 训练 / 验证
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    model         : ActorNetwork,
    loader        : DataLoader,
    loss_strategy : LossStrategyBase,
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
    optimizer     : optim.Optimizer | None,
    scheduler     : LambdaLR | None,
    writer        : SummaryWriter,
    global_step   : int,
    tag           : str = "train",
) -> tuple[float, int]:
    """
    单 epoch 训练或验证。

    Args:
      optimizer / scheduler : 训练时传入，验证时传 None
      tag                   : "train" | "val"，用于 TensorBoard key
    Returns:
      avg_loss    : epoch 平均 total_loss
      global_step : 更新后的全局步数（验证时原样返回）
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    n_batches      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"[{tag}]", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            # ── 解包 batch ──────────────────────────────────────────
            history, tf, graph_kw, target_tgt = unpack_batch(batch)

            # ── 前向传播 ────────────────────────────────────────────
            # ActorNetwork.forward 签名：
            #   forward(history, tf, gru_hidden=None, **encoder_kwargs)
            # 返回：action_logits, src_logits, tgt_logits, label_logits, gru_hidden
            action_logits, src_logits, tgt_logits, label_logits, _ = model(
                history, tf, **graph_kw
            )

            # ── 计算原始 loss ────────────────────────────────────────
            raw_losses = compute_raw_losses(
                action_logits = action_logits,
                src_logits    = src_logits,
                tgt_logits    = tgt_logits,
                label_logits  = label_logits,
                tf            = tf,
                target_tgt    = target_tgt,    # ← 直接使用，不再 batch.target_tgt.squeeze(-1)
                vocab_size    = vocab_size,
                criterion_cls = criterion_cls,
                criterion_seq = criterion_seq,
            )

            # ── 加权（active_tasks 由 strategy 内部管理）────────────
            total_loss, log_dict = loss_strategy(raw_losses)

            # ── 反向传播（仅训练）───────────────────────────────────
            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), TRAIN_CFG.grad_clip
                )
                optimizer.step()
                scheduler.step()

                # TensorBoard：逐步记录
                for name, val in log_dict.items():
                    writer.add_scalar(f"{tag}/loss_{name}", val, global_step)
                writer.add_scalar(f"{tag}/loss_total", total_loss.item(), global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                global_step += 1

            total_loss_sum += total_loss.item()
            n_batches      += 1
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    avg_loss = total_loss_sum / max(1, n_batches)

    # 验证时记录 epoch 级别指标
    if not is_train:
        writer.add_scalar(f"{tag}/loss_epoch", avg_loss, global_step)

    return avg_loss, global_step


# ══════════════════════════════════════════════════════════════════════
#  测试集评估
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model         : ActorNetwork,
    loader        : DataLoader,
    loss_strategy : LossStrategyBase,
    vocab_size    : int,
    criterion_cls : nn.CrossEntropyLoss,
    criterion_seq : nn.CrossEntropyLoss,
) -> dict[str, float]:
    """在测试集上计算各任务 loss 及 action 准确率"""
    model.eval()

    loss_sums      = {name: 0.0 for name in TASK_NAMES + ["total"]}
    action_correct = 0
    n_samples      = 0
    n_batches      = 0

    for batch in tqdm(loader, desc="[test]"):
        batch = batch.to(device)
        history, tf, graph_kw, target_tgt = unpack_batch(batch)

        action_logits, src_logits, tgt_logits, label_logits, _ = model(
            history, tf, **graph_kw
        )
        raw_losses = compute_raw_losses(
            action_logits = action_logits,
            src_logits    = src_logits,
            tgt_logits    = tgt_logits,
            label_logits  = label_logits,
            tf            = tf,
            target_tgt    = target_tgt,
            vocab_size    = vocab_size,
            criterion_cls = criterion_cls,
            criterion_seq = criterion_seq,
        )
        total_loss, log_dict = loss_strategy(raw_losses)

        for name, val in log_dict.items():
            loss_sums[name] += val
        loss_sums["total"] += total_loss.item()

        # action 准确率
        pred = action_logits.argmax(dim=-1)
        action_correct += (pred == tf.action).sum().item()
        n_samples      += tf.action.size(0)
        n_batches      += 1

    results = {
        f"test/loss_{k}": v / max(1, n_batches)
        for k, v in loss_sums.items()
    }
    results["test/action_acc"] = action_correct / max(1, n_samples)
    return results


# ══════════════════════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── 初始化 ────────────────────────────────────────────────────────
    tokenizer                             = LabelTokenizer(str(PATH_CFG.VOCAB_FILE))
    train_loader, val_loader, test_loader = build_dataloaders(tokenizer)

    model         = build_model(vocab_size=tokenizer.vocab_size)
    loss_strategy = build_loss_strategy(TRAIN_CFG).to(device)
    writer        = SummaryWriter(log_dir=str(PATH_CFG.TB_DIR))

    criterion_cls = nn.CrossEntropyLoss(ignore_index=MODEL_CFG.pad_action_id)
    criterion_seq = nn.CrossEntropyLoss(ignore_index=MODEL_CFG.pad_token_id)

    best_val_loss = float("inf")
    global_step   = 0

    # ── 课程学习：逐阶段执行 ──────────────────────────────────────────
    for stage in TRAIN_CFG.stages:
        stage_lr = TRAIN_CFG.lr * stage.lr_scale
        print(f"\n{'='*60}")
        print(f"  Stage : {stage.name}")
        print(f"  epochs={stage.epochs}  lr={stage_lr:.2e}")
        print(f"  freeze={stage.freeze or 'none'}")
        print(f"  active_tasks={stage.active_tasks or 'ALL'}")
        print(f"{'='*60}")

        # 1. 冻结 / 解冻参数
        set_requires_grad(model, freeze=stage.freeze)

        # 2. 更新 LossStrategy 的激活任务
        #    active_tasks=[] 表示全部激活，由 strategy 内部处理
        loss_strategy.set_active_tasks(stage.active_tasks)

        # 3. 构建优化器（每阶段重建，保证只优化可训练参数）
        total_steps = stage.epochs * len(train_loader)
        optimizer, scheduler = build_optimizer_scheduler(
            model, loss_strategy, total_steps, lr=stage_lr
        )

        # 4. 逐 epoch 训练
        for epoch in range(1, stage.epochs + 1):

            # 训练
            train_loss, global_step = run_epoch(
                model         = model,
                loader        = train_loader,
                loss_strategy = loss_strategy,
                vocab_size    = tokenizer.vocab_size,
                criterion_cls = criterion_cls,
                criterion_seq = criterion_seq,
                optimizer     = optimizer,
                scheduler     = scheduler,
                writer        = writer,
                global_step   = global_step,
                tag           = "train",
            )

            # 验证（按 val_every_epoch 频率）
            if epoch % TRAIN_CFG.val_every_epoch == 0:
                val_loss, _ = run_epoch(
                    model         = model,
                    loader        = val_loader,
                    loss_strategy = loss_strategy,
                    vocab_size    = tokenizer.vocab_size,
                    criterion_cls = criterion_cls,
                    criterion_seq = criterion_seq,
                    optimizer     = None,
                    scheduler     = None,
                    writer        = writer,
                    global_step   = global_step,
                    tag           = "val",
                )
                print(
                    f"  [{stage.name}] epoch {epoch:>3}/{stage.epochs}"
                    f"  train={train_loss:.4f}  val={val_loss:.4f}"
                )

                # 保存最优模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "epoch"        : epoch,
                            "stage"        : stage.name,
                            "global_step"  : global_step,
                            "model"        : model.state_dict(),
                            "loss_strategy": loss_strategy.state_dict(),
                            "val_loss"     : val_loss,
                        },
                        PATH_CFG.CKPT_BEST_MODEL_FILE,
                    )
                    print(f"  ✓ best model saved  val_loss={val_loss:.4f}")
            else:
                print(
                    f"  [{stage.name}] epoch {epoch:>3}/{stage.epochs}"
                    f"  train={train_loss:.4f}"
                )

        # 保存阶段末尾检查点
        torch.save(
            {
                "epoch"        : stage.epochs,
                "stage"        : stage.name,
                "global_step"  : global_step,
                "model"        : model.state_dict(),
                "loss_strategy": loss_strategy.state_dict(),
            },
            PATH_CFG.CKPT_LAST_MODEL_FILE,
        )

        # 记录阶段级权重（Uncertainty 时有意义）
        for name, w in loss_strategy.weights().items():
            writer.add_scalar(f"weights/{name}", w, global_step)

    # ── 测试集最终评估 ────────────────────────────────────────────────
    print("\n[pretrain] loading best checkpoint for test evaluation ...")
    ckpt = torch.load(PATH_CFG.CKPT_BEST_MODEL_FILE, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_results = evaluate(
        model         = model,
        loader        = test_loader,
        loss_strategy = loss_strategy,
        vocab_size    = tokenizer.vocab_size,
        criterion_cls = criterion_cls,
        criterion_seq = criterion_seq,
    )
    print("\n[Test Results]")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")
        writer.add_scalar(k, v, global_step)

    writer.close()
    print("\n[pretrain] done.")


if __name__ == "__main__":
    main()