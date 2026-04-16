"""预训练训练器

处理 USPTO50K 数据集特有的:
  - src/tgt = -1 (Terminate步骤) 的 loss mask
  - label = "NONE" / "Terminate" 的特殊处理
  - 多步展开的 batch 训练
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict

from models.actor import ActorNetwork

logger = logging.getLogger(__name__)


class PretrainTrainer:

    def __init__(self, actor: ActorNetwork, cfg: Dict, device: torch.device):
        self.actor = actor.to(device)
        self.device = device
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=cfg.get("lr_pretrain", 1e-4),
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.get("total_epochs", 50)
        )

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """计算预训练损失

        关键: src/tgt 的 loss 只在 valid 步骤计算 (Terminate 步骤跳过)
        """
        pyg = batch["pyg_batch"].to(self.device)
        target_actions  = batch["target_actions"].to(self.device)
        target_srcs     = batch["target_srcs"].to(self.device)
        target_tgts     = batch["target_tgts"].to(self.device)
        dec_inputs      = batch["decoder_inputs"].to(self.device)
        dec_targets     = batch["decoder_targets"].to(self.device)
        src_valid       = batch["src_valid_mask"].to(self.device)   # [N_steps]
        tgt_valid       = batch["tgt_valid_mask"].to(self.device)

        # ── 前向 ──────────────────────────────────────────────
        action_logits, src_logits, tgt_logits, label_logits = self.actor(
            x=pyg.x, edge_index=pyg.edge_index, batch=pyg.batch,
            target_action=target_actions,
            target_src=target_srcs,
            decoder_input_seq=dec_inputs,
        )

        # ── Action Loss (所有步骤) ─────────────────────────────
        action_loss = F.cross_entropy(action_logits, target_actions)

        # ── Pointer Loss (仅 valid 步骤) ──────────────────────
        src_loss_raw = F.cross_entropy(src_logits, target_srcs, reduction='none')
        tgt_loss_raw = F.cross_entropy(tgt_logits, target_tgts, reduction='none')

        n_src = src_valid.sum().clamp(min=1)
        n_tgt = tgt_valid.sum().clamp(min=1)
        src_loss = (src_loss_raw * src_valid.float()).sum() / n_src
        tgt_loss = (tgt_loss_raw * tgt_valid.float()).sum() / n_tgt

        # ── Label Loss (非 Terminate 步骤) ────────────────────
        # dec_targets: [N_steps, L], label_logits: [N_steps, L, vocab]
        pad_id = self.cfg.get("label_pad_id", 0)
        label_loss = F.cross_entropy(
            label_logits.reshape(-1, label_logits.size(-1)),
            dec_targets.reshape(-1),
            ignore_index=pad_id,
        )

        # ── 加权总损失 ────────────────────────────────────────
        w_act = self.cfg.get("w_action", 1.0)
        w_ptr = self.cfg.get("w_pointer", 1.0)
        w_lbl = self.cfg.get("w_label", 0.5)

        total = w_act * action_loss + w_ptr * (src_loss + tgt_loss) + w_lbl * label_loss

        return {
            "loss":         total,
            "action_loss":  action_loss,
            "src_loss":     src_loss,
            "tgt_loss":     tgt_loss,
            "label_loss":   label_loss,
        }

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.actor.train()
        stats = {"loss": 0, "action_loss": 0, "src_loss": 0,
                 "tgt_loss": 0, "label_loss": 0}
        n = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            losses = self.compute_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.cfg.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()

            for k in stats:
                stats[k] += losses[k].item()
            n += 1

        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in stats.items()}

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)
        logger.info(f"Actor saved to {path}")