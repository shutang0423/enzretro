"""
utils/early_stopping.py —— 早停工具类

使用方式：
  es = EarlyStopping(patience=10, mode="min", delta=1e-4)
  for epoch in ...:
      val_loss = ...
      if es.step(val_loss):
          print("早停触发")
          break
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停控制器

    Args:
      patience : int   连续多少个 epoch 无提升则停止，默认 10
      mode     : str   "min"（越小越好，如 loss）| "max"（越大越好，如 acc）
      delta    : float 最小改善幅度，小于此值视为"未提升"，默认 1e-4
      verbose  : bool  是否打印日志

    属性：
      best_value   : 当前最优指标值
      best_epoch   : 最优值出现的 epoch
      counter      : 当前连续未提升轮数
      should_stop  : 是否应该停止
    """

    def __init__(
        self,
        patience : int   = 10,
        mode     : str   = "min",
        delta    : float = 1e-4,
        verbose  : bool  = True,
    ):
        assert mode in ("min", "max"), f"mode 须为 'min' 或 'max'，got '{mode}'"
        self.patience    = patience
        self.mode        = mode
        self.delta       = delta
        self.verbose     = verbose

        self.best_value  : float | None = None
        self.best_epoch  : int          = 0
        self.counter     : int          = 0
        self.should_stop : bool         = False

    # ── 核心：判断是否提升 ─────────────────────────────────────────────
    def _is_improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.delta
        else:
            return value > self.best_value + self.delta

    # ── 每 epoch 调用一次 ──────────────────────────────────────────────
    def step(self, value: float, epoch: int) -> bool:
        """
        传入当前 epoch 的监控指标值。

        Returns:
          True  → 触发早停，应中断训练循环
          False → 继续训练
        """
        if self._is_improved(value):
            if self.verbose:
                prev = f"{self.best_value:.6f}" if self.best_value is not None else "N/A"
                logger.info(
                    f"[EarlyStopping] Epoch {epoch:>3d} | "
                    f"指标提升 {prev} → {value:.6f}  counter 重置"
                )
            self.best_value = value
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"[EarlyStopping] Epoch {epoch:>3d} | "
                    f"无提升 ({value:.6f} ≥ best {self.best_value:.6f})  "
                    f"counter = {self.counter}/{self.patience}"
                )

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                logger.info(
                    f"[EarlyStopping] *** 触发早停！***  "
                    f"连续 {self.patience} 个 epoch 无提升，"
                    f"最优 epoch = {self.best_epoch}，"
                    f"最优值 = {self.best_value:.6f}"
                )

        return self.should_stop

    # ── 状态重置（切换课程阶段时可选调用）────────────────────────────
    def reset(self) -> None:
        """重置计数器（课程学习切换阶段时使用）"""
        self.best_value  = None
        self.best_epoch  = 0
        self.counter     = 0
        self.should_stop = False
        if self.verbose:
            logger.info("[EarlyStopping] 状态已重置")

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode={self.mode!r}, "
            f"delta={self.delta}, counter={self.counter}/{self.patience}, "
            f"best={self.best_value}@epoch{self.best_epoch})"
        )