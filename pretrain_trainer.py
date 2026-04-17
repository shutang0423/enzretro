"""预训练训练器

处理 USPTO50K 数据集特有的:
  - src/tgt = -1 (Terminate步骤) 的 loss mask
  - label = "NONE" / "Terminate" 的特殊处理
  - 多步展开的 batch 训练
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional 
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter

from models.actor import ActorNetwork

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                        # stdout → pre.log
            logging.FileHandler("pretrain.log", mode="a"), # 独立日志文件
        ],
        force=True,
    )
logger = logging.getLogger(__name__)


class PretrainTrainer:

    def __init__(self, actor: ActorNetwork, cfg, device: torch.device):
        self.actor = actor.to(device)
        self.device = device
        self.cfg = cfg

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(cfg.path.CKPT_DIR))

        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

        # 训练状态
        self.start_epoch:  int   = 0
        self.best_loss:    float = float("inf")
        self.no_improve:   int   = 0          # 早停计数器
        self.global_step:  int   = 0



    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        pyg            = batch["pyg_batch"].to(self.device)
        target_actions = batch["target_actions"].to(self.device)   # [N_steps]
        target_srcs    = batch["target_srcs"].to(self.device)      # [N_steps]
        target_tgts    = batch["target_tgts"].to(self.device)      # [N_steps]
        dec_inputs     = batch["decoder_inputs"].to(self.device)   # [N_steps, L]
        dec_targets    = batch["decoder_targets"].to(self.device)  # [N_steps, L]
        src_valid      = batch["src_valid_mask"].to(self.device)   # [N_steps]
        tgt_valid      = batch["tgt_valid_mask"].to(self.device)   # [N_steps]
        step_to_sample = batch["step_to_sample"].to(self.device)   # [N_steps] ← 新增

        # ── 前向 ──────────────────────────────────────────────────
        action_logits, src_logits, tgt_logits, label_logits = self.actor(
            x=pyg.x,
            edge_index=pyg.edge_index,
            batch=pyg.batch,
            target_action=target_actions,
            target_src=target_srcs,
            decoder_input_seq=dec_inputs,
            step_to_sample=step_to_sample,    # ← 新增，解决维度不匹配
            src_valid=src_valid,
            tgt_valid=tgt_valid,
        )

        # ── Action Loss ───────────────────────────────────────────
        action_loss = F.cross_entropy(action_logits, target_actions)

        # ── Pointer Loss (仅 valid 步骤) ──────────────────────────
        src_loss_raw = F.cross_entropy(src_logits, target_srcs, reduction='none')
        tgt_loss_raw = F.cross_entropy(tgt_logits, target_tgts, reduction='none')
        n_src = src_valid.sum().clamp(min=1)
        n_tgt = tgt_valid.sum().clamp(min=1)
        src_loss = (src_loss_raw * src_valid.float()).sum() / n_src
        tgt_loss = (tgt_loss_raw * tgt_valid.float()).sum() / n_tgt

        # ── Label Loss ────────────────────────────────────────────
        pad_id = self.cfg.model.PAD_ACTION_ID
        label_loss = F.cross_entropy(
            label_logits.reshape(-1, label_logits.size(-1)),
            dec_targets.reshape(-1),
            ignore_index=pad_id,
        )

        # ── 加权总损失 ────────────────────────────────────────────
        w_act = self.cfg.model.W_ACTION
        w_ptr = self.cfg.model.W_POINTER
        w_lbl = self.cfg.model.W_LABEL
        total = w_act * action_loss + w_ptr * (src_loss + tgt_loss) + w_lbl * label_loss

        return {
            "loss":        total,
            "action_loss": action_loss,
            "src_loss":    src_loss,
            "tgt_loss":    tgt_loss,
            "label_loss":  label_loss,
        }

    # ── 断点续训 ──────────────────────────────────────────────
    def load_checkpoint(self, path: Optional[Path] = None) -> bool:
        """加载 checkpoint，返回是否成功"""
        ckpt_path = path or (self.cfg.path.CKPT_DIR / "actor_last.pt")
        if not ckpt_path.exists():
            logger.info("No checkpoint found, training from scratch.")
            return False

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
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
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "best_loss":   self.best_loss,
            "no_improve":  self.no_improve,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    # def train_epoch(self, dataloader) -> Dict[str, float]:
    #     self.actor.train()
    #     stats = {"loss": 0, "action_loss": 0, "src_loss": 0,
    #              "tgt_loss": 0, "label_loss": 0}
    #     n = 0

    #     for batch in dataloader:
    #         self.optimizer.zero_grad()
    #         losses = self.compute_loss(batch)
    #         losses["loss"].backward()
    #         torch.nn.utils.clip_grad_norm_(
    #             self.actor.parameters(),
    #             self.cfg.model.MAX_GRAD_NORM
    #         )
    #         self.optimizer.step()

    #         for k in stats:
    #             stats[k] += losses[k].item()
    #         n += 1

    #     self.scheduler.step()
    #     return {k: v / max(n, 1) for k, v in stats.items()}

    # def save(self, path: Path):
    #     torch.save(self.actor.state_dict(), path)
    #     logger.info(f"Actor saved to {path}")


    # ── 单 epoch 训练 ─────────────────────────────────────────
    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.actor.train()
        stats = {k: 0.0 for k in ["loss","action_loss","src_loss","tgt_loss","label_loss"]}
        n = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            losses = self.compute_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.pretrain.max_grad_norm
            )
            self.optimizer.step()

            # 逐步写 TensorBoard
            self.writer.add_scalar("train/step_loss", losses["loss"].item(), self.global_step)
            self.global_step += 1

            for k in stats:
                stats[k] += losses[k].item()
            n += 1

        return {k: v / max(n, 1) for k, v in stats.items()}

    # ── 单 epoch 验证 ─────────────────────────────────────────
    @torch.no_grad()
    def eval_epoch(self, dataloader) -> Dict[str, float]:
        self.actor.eval()
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
        """
        多 epoch 训练主循环，支持:
          - 断点续训 (resume=True 时自动加载)
          - TensorBoard 记录
          - 最优模型保存
          - 早停
        """
        pc = self.cfg.pretrain

        if pc.resume:
            self.load_checkpoint()

        logger.info(f"Training from epoch {self.start_epoch} to {pc.total_epochs}")

        for epoch in range(self.start_epoch, pc.total_epochs):
            # ── 训练 ──────────────────────────────────────────
            train_stats = self.train_epoch(train_loader)
            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]

            # ── 验证 ──────────────────────────────────────────
            val_stats = self.eval_epoch(val_loader) if val_loader else train_stats
            val_loss  = val_stats["loss"]

            # ── TensorBoard (epoch 级) ─────────────────────────
            for k, v in train_stats.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_stats.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("train/lr", lr_now, epoch)

            # ── 日志输出 ──────────────────────────────────────
            logger.info(
                f"Epoch {epoch:03d}/{pc.total_epochs}  "
                f"train_loss={train_stats['loss']:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={lr_now:.2e}"
            )

            # ── 最优模型保存 ──────────────────────────────────
            if val_loss < self.best_loss:
                self.best_loss  = val_loss
                self.no_improve = 0
                self.save_checkpoint(epoch, self.cfg.path.CKPT_BEST_MODEL_FILE)
                logger.info(f"  ★ Best model updated  (val_loss={val_loss:.4f})")
            else:
                self.no_improve += 1

            # ── 定期保存 last checkpoint ──────────────────────
            if (epoch + 1) % pc.save_every == 0:
                self.save_checkpoint(epoch, self.cfg.path.CKPT_LAST_MODEL_FILE)

            # ── 早停 ──────────────────────────────────────────
            if self.no_improve >= pc.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improve {self.no_improve})")
                break

        # 训练结束保存最后一个 checkpoint
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


