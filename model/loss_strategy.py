"""
loss_strategy.py —— 可插拔多任务 Loss 策略

消融实验：
  UncertaintyWeighting : 可学习 σ_i 自动平衡（主方案）
  EqualWeighting       : 所有任务权重 = 1.0
  ManualWeighting      : config 中手动指定权重
  SingleTaskWeighting  : 只激活指定任务

改进点：
  - active_tasks 由 strategy 统一管理，消除 pretrain.py 中的双重控制
  - forward 接收 dict[str, Tensor]，接口更清晰
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
    统一接口：
      forward(loss_dict) → total_loss, weighted_dict
      active_tasks       → 当前激活的任务名集合
    """

    def __init__(self, active_tasks: Optional[List[str]] = None):
        super().__init__()
        # 空列表或 None 均表示全部激活
        self._active = set(active_tasks) if active_tasks else set(TASK_NAMES)

    @property
    def active_tasks(self) -> set[str]:
        return self._active

    def set_active_tasks(self, tasks: List[str]) -> None:
        """课程学习换阶段时动态更新激活任务"""
        self._active = set(tasks) if tasks else set(TASK_NAMES)

    @abstractmethod
    def forward(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
          loss_dict: {"action": Tensor, "src": Tensor, "tgt": Tensor, "label": Tensor}
        Returns:
          total_loss  : 标量张量，用于 backward
          log_dict    : {"action": float, ...} 用于 TensorBoard
        """
        ...

    def weights(self) -> Dict[str, float]:
        return {name: 1.0 for name in TASK_NAMES}


# ══════════════════════════════════════════════════════════════════════
#  实现 1：Uncertainty Weighting（主方案）
# ══════════════════════════════════════════════════════════════════════

class UncertaintyWeighting(LossStrategyBase):
    """
    基于任务不确定性的自动权重平衡
      L_total = Σ [ exp(-s_i) * L_i + s_i ]
      s_i = log_sigma_i，clamp 防止退化
    """

    def __init__(
        self,
        active_tasks: Optional[List[str]] = None,
        s_min: float = -2.5,
        s_max: float =  2.5,
    ):
        super().__init__(active_tasks)
        self.s_min     = s_min
        self.s_max     = s_max
        self.log_sigma = nn.Parameter(torch.zeros(len(TASK_NAMES)))

    def forward(self, loss_dict):
        s      = self.log_sigma.clamp(self.s_min, self.s_max)
        total  = torch.zeros(1, device=s.device)
        log    = {}
        for i, name in enumerate(TASK_NAMES):
            l = loss_dict[name]
            if name in self.active_tasks:
                weighted = torch.exp(-s[i]) * l + s[i]
                total    = total + weighted
            log[name] = l.item()
        return total.squeeze(), log

    def weights(self):
        s = self.log_sigma.clamp(self.s_min, self.s_max)
        return {name: float(torch.exp(-s[i]).item()) for i, name in enumerate(TASK_NAMES)}


# ══════════════════════════════════════════════════════════════════════
#  实现 2：Equal Weighting
# ══════════════════════════════════════════════════════════════════════

class EqualWeighting(LossStrategyBase):
    """所有激活任务权重均为 1.0"""

    def __init__(self, active_tasks: Optional[List[str]] = None):
        super().__init__(active_tasks)

    def forward(self, loss_dict):
        total = torch.zeros(1, device=next(iter(loss_dict.values())).device)
        log   = {}
        for name in TASK_NAMES:
            l = loss_dict[name]
            if name in self.active_tasks:
                total = total + l
            log[name] = l.item()
        return total.squeeze(), log


# ══════════════════════════════════════════════════════════════════════
#  实现 3：Manual Weighting
# ══════════════════════════════════════════════════════════════════════

class ManualWeighting(LossStrategyBase):
    """手动指定各任务权重"""

    def __init__(
        self,
        task_weights : Dict[str, float],
        active_tasks : Optional[List[str]] = None,
    ):
        super().__init__(active_tasks)
        self._weights = {name: task_weights.get(name, 1.0) for name in TASK_NAMES}

    def forward(self, loss_dict):
        total = torch.zeros(1, device=next(iter(loss_dict.values())).device)
        log   = {}
        for name in TASK_NAMES:
            l = loss_dict[name]
            if name in self.active_tasks:
                total = total + self._weights[name] * l
            log[name] = l.item()
        return total.squeeze(), log

    def weights(self):
        return dict(self._weights)


# ══════════════════════════════════════════════════════════════════════
#  实现 4：Single Task（单任务基线）
# ══════════════════════════════════════════════════════════════════════

class SingleTaskWeighting(LossStrategyBase):
    """只激活单个任务，其余 loss 不参与梯度"""

    def __init__(self, task_name: str):
        super().__init__([task_name])
        self.task_name = task_name

    def forward(self, loss_dict):
        l   = loss_dict[self.task_name]
        log = {name: loss_dict[name].item() for name in TASK_NAMES}
        return l, log

    def weights(self):
        return {name: (1.0 if name == self.task_name else 0.0) for name in TASK_NAMES}


# ══════════════════════════════════════════════════════════════════════
#  工厂函数
# ══════════════════════════════════════════════════════════════════════

def build_loss_strategy(cfg) -> LossStrategyBase:
    """根据 TrainConfig 构建 Loss 策略"""
    strategy = cfg.loss_strategy
    if strategy == "uncertainty":
        return UncertaintyWeighting()
    elif strategy == "equal":
        return EqualWeighting()
    elif strategy == "manual":
        return ManualWeighting(task_weights=dict(zip(TASK_NAMES, cfg.loss_weights)))
    elif strategy == "single_task":
        return SingleTaskWeighting(task_name=cfg.single_task)
    else:
        raise ValueError(f"Unknown loss_strategy: {strategy!r}")