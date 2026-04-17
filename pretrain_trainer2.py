"""预训练训练器 v2

融合改进:
  1. UncertaintyWeighting  → 替换固定权重，自适应平衡四任务
  2. Warmup + Cosine LR    → step 级调度，替换 epoch 级 CosineAnnealingLR
  3. safe_loss()           → NaN/Inf 防护
  4. 课程学习 (可选)        → 支持分阶段冻结，默认 Joint 训练
  5. 验证集 best model     → 按 val_loss 保存，而非 train_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from models.actor import ActorNetwork

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                        # stdout → pre.log
            logging.FileHandler("pretrain2.log", mode="a"), # 独立日志文件
        ],
        force=True,
    )
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  1. 自适应损失加权（移植自旧版，核心改进）
# ══════════════════════════════════════════════════════════════════
class UncertaintyWeighting(nn.Module):
    """多任务不确定性加权
    
    原理: loss_total = Σ [ exp(-σ_i) * L_i + σ_i ]
    σ_i 可学习，难任务自动获得更大 σ（更小权重），简单任务反之。
    ref: Kendall et al., CVPR 2018
    """
    def __init__(self, num_tasks: int = 4,
                 s_min: float = -2.5,
                 s_max: float =  2.5):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
        self.s_min = s_min
        self.s_max = s_max

    def forward(self, *losses):
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        weighted = [torch.exp(-s[i]) * l + s[i]
                    for i, l in enumerate(losses)]
        return sum(weighted)

    def weights(self):
        """返回当前各任务权重（用于日志）"""
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        return torch.exp(-s).detach().cpu().tolist()

    def log_sigmas(self):
        return self.log_sigma.detach().cpu().tolist()


# ══════════════════════════════════════════════════════════════════
#  2. NaN 安全 loss（移植自旧版）
# ══════════════════════════════════════════════════════════════════
def safe_loss(loss: torch.Tensor, name: str = "") -> torch.Tensor:
    """检测 NaN/Inf，出现时替换为 0 并打印警告"""
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(f"[NaN/Inf] loss '{name}' detected, set to 0.")
        return torch.tensor(0.0, device=loss.device, requires_grad=True)
    return loss


class PretrainTrainer:

    def __init__(self, actor: ActorNetwork, cfg, device: torch.device):
        self.actor  = actor.to(device)
        self.device = device
        self.cfg    = cfg

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(cfg.path.CKPT_DIR))

        # ── 改进1: UncertaintyWeighting 替换固定权重 ────────────────
        self.uw = UncertaintyWeighting(num_tasks=4).to(device)

        # ── 改进2: 优化器包含 uw 参数 ──────────────────────────────
        self.optimizer = torch.optim.AdamW(
            list(actor.parameters()) + list(self.uw.parameters()),
            lr=cfg.pretrain.lr,          # 建议 1e-4
            weight_decay=1e-2,
        )

        # scheduler 在 fit() 里按 total_steps 构建（需要知道数据量）
        self.scheduler = None

        # 训练状态
        self.start_epoch: int   = 0
        self.best_loss:   float = float("inf")
        self.no_improve:  int   = 0
        self.global_step: int   = 0


    # ── 改进3: Warmup + Cosine LR（step 级）─────────────────────
    def _build_scheduler(self, total_steps: int):
        warmup_steps = int(total_steps * 0.1)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(
                torch.tensor(3.14159 * progress)).item()))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Scheduler built: warmup={warmup_steps} steps, total={total_steps} steps")


    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        pyg            = batch["pyg_batch"].to(self.device)
        target_actions = batch["target_actions"].to(self.device)
        target_srcs    = batch["target_srcs"].to(self.device)
        target_tgts    = batch["target_tgts"].to(self.device)
        dec_inputs     = batch["decoder_inputs"].to(self.device)
        dec_targets    = batch["decoder_targets"].to(self.device)
        src_valid      = batch["src_valid_mask"].to(self.device)
        tgt_valid      = batch["tgt_valid_mask"].to(self.device)
        step_to_sample = batch["step_to_sample"].to(self.device)

        action_logits, src_logits, tgt_logits, label_logits = self.actor(
            x=pyg.x,
            edge_index=pyg.edge_index,
            batch=pyg.batch,
            target_action=target_actions,
            target_src=target_srcs,
            decoder_input_seq=dec_inputs,
            step_to_sample=step_to_sample,
            src_valid=src_valid,
            tgt_valid=tgt_valid,
        )

        # ── Action Loss ─────────────────────────────────────────
        action_loss = safe_loss(
            F.cross_entropy(action_logits, target_actions), "action"
        )

        # ── Pointer Loss（有效步骤过滤，移植自旧版）───────────────
        N = src_logits.size(-1)
        valid_src = src_valid & (target_srcs >= 0) & (target_srcs < N)
        valid_tgt = tgt_valid & (target_tgts >= 0) & (target_tgts < N)

        src_loss = (safe_loss(
            F.cross_entropy(src_logits[valid_src], target_srcs[valid_src]), "src"
        ) if valid_src.any() else torch.tensor(0.0, device=self.device, requires_grad=True))

        tgt_loss = (safe_loss(
            F.cross_entropy(tgt_logits[valid_tgt], target_tgts[valid_tgt]), "tgt"
        ) if valid_tgt.any() else torch.tensor(0.0, device=self.device, requires_grad=True))

        # ── Label Loss ──────────────────────────────────────────
        pad_id = self.cfg.model.PAD_ACTION_ID
        label_loss = safe_loss(
            F.cross_entropy(
                label_logits.reshape(-1, label_logits.size(-1)),
                dec_targets.reshape(-1),
                ignore_index=pad_id,
            ), "label"
        )

        # ── 改进1: UW 自适应加权总损失 ──────────────────────────
        total = self.uw(action_loss, src_loss, tgt_loss, label_loss)

        return {
            "loss":        total,
            "action_loss": action_loss,
            "src_loss":    src_loss,
            "tgt_loss":    tgt_loss,
            "label_loss":  label_loss,
        }


    # ── 断点续训 ──────────────────────────────────────────────
    def load_checkpoint(self, path: Optional[Path] = None) -> bool:
        ckpt_path = path or (self.cfg.path.CKPT_DIR / "actor_last.pt")
        if not ckpt_path.exists():
            logger.info("No checkpoint found, training from scratch.")
            return False

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["model"])
        self.uw.load_state_dict(ckpt["uw"])             # ← 新增：恢复 uw 参数
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss   = ckpt.get("best_loss", float("inf"))
        self.no_improve  = ckpt.get("no_improve", 0)
        self.global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}  best_loss={self.best_loss:.4f}")
        return True

    def save_checkpoint(self, epoch: int, path: Path):
        torch.save({
            "epoch":       epoch,
            "model":       self.actor.state_dict(),
            "uw":          self.uw.state_dict(),         # ← 新增：保存 uw 参数
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict() if self.scheduler else None,
            "best_loss":   self.best_loss,
            "no_improve":  self.no_improve,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Checkpoint saved → {path}")


    # ── 单 epoch 训练 ─────────────────────────────────────────
    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.actor.train()
        self.uw.train()
        stats = {k: 0.0 for k in ["loss","action_loss","src_loss","tgt_loss","label_loss"]}
        n = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            losses = self.compute_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.uw.parameters()),
                self.cfg.pretrain.max_grad_norm
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()   # ← step 级调度

            # TensorBoard step 级记录
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/step_loss",  losses["loss"].item(),   self.global_step)
            self.writer.add_scalar("train/lr",         cur_lr,                  self.global_step)

            # 记录 UW 权重变化（可观察各任务权重动态）
            w = self.uw.weights()
            for i, (task, wi) in enumerate(zip(["action","src","tgt","label"], w)):
                self.writer.add_scalar(f"uw/weight_{task}", wi, self.global_step)

            self.global_step += 1

            for k in stats:
                stats[k] += losses[k].item()
            n += 1

        return {k: v / max(n, 1) for k, v in stats.items()}


    # ── 单 epoch 验证 ─────────────────────────────────────────
    @torch.no_grad()
    def eval_epoch(self, dataloader) -> Dict[str, float]:
        self.actor.eval()
        self.uw.eval()
        stats = {k: 0.0 for k in ["loss","action_loss","src_loss","tgt_loss","label_loss"]}
        n = 0
        for batch in dataloader:
            losses = self.compute_loss(batch)
            for k in stats:
                stats[k] += losses[k].item()
            n += 1
        return {k: v / max(n, 1) for k, v in stats.items()}


    # ── 完整训练循环 ──────────────────────────────────────────
    def fit(self, train_loader, val_loader=None) -> None:
        pc = self.cfg.pretrain

        # ── 改进2: 先构建 step 级 scheduler ─────────────────────
        steps_per_epoch = len(train_loader)
        total_steps     = steps_per_epoch * pc.total_epochs
        self._build_scheduler(total_steps)

        if pc.resume:
            self.load_checkpoint()

        logger.info(f"Training from epoch {self.start_epoch} to {pc.total_epochs}")
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

        for epoch in range(self.start_epoch, pc.total_epochs):

            # ── 训练 ────────────────────────────────────────────
            train_stats = self.train_epoch(train_loader)
            lr_now = self.optimizer.param_groups[0]["lr"]
            w = self.uw.weights()

            # ── 验证 ────────────────────────────────────────────
            val_stats = self.eval_epoch(val_loader) if val_loader else train_stats
            val_loss  = val_stats["loss"]

            # ── TensorBoard epoch 级 ─────────────────────────────
            for k, v in train_stats.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_stats.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalars("uw/weights", {
                "action": w[0], "src": w[1], "tgt": w[2], "label": w[3]
            }, epoch)

            # ── 日志输出 ─────────────────────────────────────────
            logger.info(
                f"Epoch {epoch:03d}/{pc.total_epochs}  "
                f"train={train_stats['loss']:.4f}  val={val_loss:.4f}  "
                f"lr={lr_now:.2e}  "
                f"uw=[act:{w[0]:.2f} src:{w[1]:.2f} tgt:{w[2]:.2f} lbl:{w[3]:.2f}]"
            )

            # ── 按验证 loss 保存最优（移植自旧版）────────────────
            if val_loss < self.best_loss:
                self.best_loss  = val_loss
                self.no_improve = 0
                self.save_checkpoint(epoch, self.cfg.path.CKPT_BEST_MODEL_FILE)
                logger.info(f"  ★ Best model updated (val_loss={val_loss:.4f})")
            else:
                self.no_improve += 1

            if (epoch + 1) % pc.save_every == 0:
                self.save_checkpoint(epoch, self.cfg.path.CKPT_LAST_MODEL_FILE)

            # ── 早停 ─────────────────────────────────────────────
            if self.no_improve >= pc.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (no_improve={self.no_improve})")
                break

        self.save_checkpoint(epoch, self.cfg.path.CKPT_LAST_MODEL_FILE)
        self.writer.close()
        logger.info("Training complete.")


if __name__ == "__main__":
    from models.actor import ActorNetwork
    from config.config import config as cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = ActorNetwork(cfg.model.VOCAB_SIZE, 
                         cfg.model.NODE_IN_DIM, 
                         cfg.model.HIDDEN_DIM,
                         cfg.model.NUM_ACTIONS, 
                         cfg.model.MAX_ATOMS,
                         cfg.model.MAX_POS_ENC,
                         cfg.model.GAT_HEADS, 
                         cfg.model.GAT_LAYERS).to(device)

    from data.dataset import USPTO50KDataset
    from tokenizer.tokenizer import LabelTokenizer

    tokenizer  = LabelTokenizer(vocab_file=str(cfg.path.VOCAB_FILE))
    from data.dataset import build_dataloader
    train_dataloader = build_dataloader(json_path=str(cfg.path.RL_TRAIN_DATA_FILE), tokenizer=tokenizer,
                                        batch_size=cfg.model.BATCH_SIZE, shuffle=True, mode="pretrain")
    val_dataloader = build_dataloader(json_path=str(cfg.path.RL_VAL_DATA_FILE), tokenizer=tokenizer,
                                        batch_size=cfg.model.BATCH_SIZE, shuffle=False, mode="pretrain")

    pretrain_trainer = PretrainTrainer(actor, cfg, device)
    pretrain_trainer.fit(train_dataloader, val_dataloader)

