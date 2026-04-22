"""
loss_strategy.py —— 可插拔多任务 Loss 策略

消融实验 2：
  UncertaintyWeighting : 可学习 σ_i 自动平衡（主方案）
  EqualWeighting       : 所有任务权重 = 1.0（对照组）
  ManualWeighting      : config 中手动指定权重（对照组）
  SingleTaskWeighting  : 只激活指定任务（单任务基线）

统一接口：forward(*losses) → total_loss, loss_dict
工厂函数：build_loss_strategy(cfg) 根据配置自动选择
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


TASK_NAMES = ["action", "src", "tgt", "label"]


# ══════════════════════════════════════════════════════════════════════
#  抽象基类
# ══════════════════════════════════════════════════════════════════════

class LossStrategyBase(ABC, nn.Module):
    """
    所有 Loss 策略的基类
    forward 接收 4 个 loss 张量，返回加权总 loss 和各任务 loss 字典
    """

    @abstractmethod
    def forward(
        self,
        loss_action : torch.Tensor,
        loss_src    : torch.Tensor,
        loss_tgt    : torch.Tensor,
        loss_label  : torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        返回:
          total_loss : 标量张量，用于 backward
          loss_dict  : {"action": float, "src": float, ...}  用于日志
        """
        # ...
        pass

    def weights(self) -> Dict[str, float]:
        """返回当前各任务权重（用于 TensorBoard 记录）"""
        return {name: 1.0 for name in TASK_NAMES}


# ══════════════════════════════════════════════════════════════════════
#  实现 1：Uncertainty Weighting（主方案）
# ══════════════════════════════════════════════════════════════════════

class UncertaintyWeighting(LossStrategyBase):
    """
    基于任务不确定性的自动权重平衡
    L_total = Σ [ exp(-log_σ_i) * L_i + log_σ_i ]

    log_sigma 为可学习参数，训练时自动调整各任务权重。
    s_min / s_max 防止权重退化到极端值。
    """
    def __init__(
        self,
        num_tasks : int   = 4,
        s_min     : float = -2.5,
        s_max     : float =  2.5,
    ):
        super().__init__()
        self.s_min     = s_min
        self.s_max     = s_max
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_action, loss_src, loss_tgt, loss_label):
        losses = [loss_action, loss_src, loss_tgt, loss_label]
        s      = self.log_sigma.clamp(self.s_min, self.s_max)
        weighted_losses = [
            torch.exp(-s[i]) * l + s[i] for i, l in enumerate(losses)
        ]
        total    = sum(weighted_losses)
        loss_dict = {name: l.item() for name, l in zip(TASK_NAMES, losses)}
        return total, loss_dict

    def weights(self) -> Dict[str, float]:
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        w = torch.exp(-s).detach().cpu().tolist()
        return dict(zip(TASK_NAMES, w))


# ══════════════════════════════════════════════════════════════════════
#  实现 2：Equal Weighting（消融对照）
# ══════════════════════════════════════════════════════════════════════

class EqualWeighting(LossStrategyBase):
    """
    等权重多任务：L_total = L_action + L_src + L_tgt + L_label
    无可学习参数，作为最简单的多任务基线。
    """
    def __init__(self):
        super().__init__()

    def forward(self, loss_action, loss_src, loss_tgt, loss_label):
        losses    = [loss_action, loss_src, loss_tgt, loss_label]
        total     = sum(losses)
        loss_dict = {name: l.item() for name, l in zip(TASK_NAMES, losses)}
        return total, loss_dict


# ══════════════════════════════════════════════════════════════════════
#  实现 3：Manual Weighting（消融对照）
# ══════════════════════════════════════════════════════════════════════

class ManualWeighting(LossStrategyBase):
    """
    手动指定固定权重：L_total = Σ w_i * L_i
    权重在 config 中指定，不参与梯度更新。
    """
    def __init__(self, weights: List[float]):
        super().__init__()
        assert len(weights) == 4, "需要 4 个权重 [action, src, tgt, label]"
        # 注册为 buffer（不参与梯度，但随模型保存/加载）
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float))

    def forward(self, loss_action, loss_src, loss_tgt, loss_label):
        losses    = [loss_action, loss_src, loss_tgt, loss_label]
        total     = sum(self.w[i] * l for i, l in enumerate(losses))
        loss_dict = {name: l.item() for name, l in zip(TASK_NAMES, losses)}
        return total, loss_dict

    def weights(self) -> Dict[str, float]:
        return dict(zip(TASK_NAMES, self.w.tolist()))


# ══════════════════════════════════════════════════════════════════════
#  实现 4：Single Task Weighting（单任务基线）
# ══════════════════════════════════════════════════════════════════════

class SingleTaskWeighting(LossStrategyBase):
    """
    单任务训练：只激活指定任务，其余权重 = 0
    用于消融实验中验证各任务单独训练的效果。

    active_tasks: 激活的任务名列表，如 ["action"] 或 ["src", "tgt"]
    """
    def __init__(self, active_tasks: List[str]):
        super().__init__()
        assert all(t in TASK_NAMES for t in active_tasks), \
            f"active_tasks 必须是 {TASK_NAMES} 的子集"
        self.active_set = set(active_tasks)

    def forward(self, loss_action, loss_src, loss_tgt, loss_label):
        losses = dict(zip(TASK_NAMES, [loss_action, loss_src, loss_tgt, loss_label]))
        total  = sum(l for name, l in losses.items() if name in self.active_set)
        loss_dict = {name: l.item() for name, l in losses.items()}
        return total, loss_dict

    def weights(self) -> Dict[str, float]:
        return {name: (1.0 if name in self.active_set else 0.0)
                for name in TASK_NAMES}


# ══════════════════════════════════════════════════════════════════════
#  工厂函数
# ══════════════════════════════════════════════════════════════════════

def build_loss_strategy(cfg: dict) -> LossStrategyBase:
    """
    根据 cfg["loss_strategy"] 构建对应策略

    cfg["loss_strategy"]:
      "uncertainty"  → UncertaintyWeighting
      "equal"        → EqualWeighting
      "manual"       → ManualWeighting，需 cfg["loss_weights"] = [w0,w1,w2,w3]
      "single_task"  → SingleTaskWeighting，需 cfg["active_tasks"] = ["action",...]
    """
    strategy = cfg.get("loss_strategy", "uncertainty")
    if strategy == "uncertainty":
        return UncertaintyWeighting(
            s_min=cfg.get("uw_s_min", -2.5),
            s_max=cfg.get("uw_s_max",  2.5),
        )
    elif strategy == "equal":
        return EqualWeighting()
    elif strategy == "manual":
        return ManualWeighting(weights=cfg["loss_weights"])
    elif strategy == "single_task":
        return SingleTaskWeighting(active_tasks=cfg["active_tasks"])
    else:
        raise ValueError(
            f"Unknown loss_strategy: {strategy!r}. "
            "Choose from 'uncertainty', 'equal', 'manual', 'single_task'."
        )