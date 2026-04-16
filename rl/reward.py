"""简化版奖励函数

设计原则:
  1. 正确性为主 (与 ground truth 对比)
  2. 稀疏终局 + 少量密集中间奖励
  3. 无状态 (不破坏 MDP 假设)
  4. 奖励尺度可控
"""

from typing import Dict, Optional


class RewardCalculator:

    def __init__(self, config: Dict):
        self.cfg = config

    def step_reward(self,
                    pred_action: int, pred_src: int, pred_tgt: int,
                    gt_action: Optional[int] = None,
                    gt_src: Optional[int] = None,
                    gt_tgt: Optional[int] = None,
                    is_valid: bool = True) -> float:
        """单步奖励: 合法性 + GT匹配"""
        if not is_valid:
            return self.cfg.get("invalid_action_penalty", -0.5)

        r = self.cfg.get("step_penalty", -0.02)

        if gt_action is not None:
            if pred_action == gt_action:
                r += self.cfg.get("correct_action_bonus", 0.5)
                if gt_src is not None and pred_src == gt_src:
                    r += 0.15
                if gt_tgt is not None and pred_tgt == gt_tgt:
                    r += 0.15
            else:
                r += self.cfg.get("wrong_action_penalty", -0.2)
        return r

    def terminal_reward(self,
                        n_steps: int, max_steps: int,
                        match_score: float = 0.0,
                        is_valid_path: bool = True) -> float:
        """终局奖励: 匹配度 + 效率 + 终止时机"""
        if not is_valid_path:
            return self.cfg.get("failure_penalty", -5.0)

        r = self.cfg.get("success_reward", 10.0) * match_score

        if n_steps <= max_steps:
            r += (max_steps - n_steps) / max_steps * 1.0

        if n_steps < 1:
            r += self.cfg.get("early_terminate_penalty", -3.0)
        elif match_score > 0.8:
            r += self.cfg.get("correct_terminate_bonus", 2.0)

        return r

    @staticmethod
    def compute_gae(rewards, values, gamma=0.99, lam=0.95):
        T = len(rewards)
        advantages, gae, nv = [0.0] * T, 0.0, 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * nv - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            nv = values[t]
        returns = [a + v for a, v in zip(advantages, values)]
        return returns, advantages