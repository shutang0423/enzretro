"""Rollout Buffer: PPO 轨迹数据存储"""

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Transition:
    graph_idx: int
    history_actions: torch.Tensor
    action_type: int
    src_idx: int
    tgt_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool = False


class RolloutBuffer:

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.trajectories: List[List[Transition]] = []
        self._current: List[Transition] = []

    def add(self, t: Transition):
        self._current.append(t)
        if t.done:
            self.trajectories.append(self._current)
            self._current = []

    def finish_episode(self):
        if self._current:
            self._current[-1].done = True
            self.trajectories.append(self._current)
            self._current = []

    def compute_gae(self) -> Dict[str, torch.Tensor]:
        """计算 GAE 优势和回报，返回扁平化的 batch 数据"""
        keys = ["graph_indices", "history_actions", "actions",
                "src_indices", "tgt_indices", "old_log_probs",
                "returns", "advantages"]
        out = {k: [] for k in keys}

        for episode in self.trajectories:
            T = len(episode)
            rewards = [t.reward for t in episode]
            values  = [t.value for t in episode] + [0.0]

            advs, gae = [0.0] * T, 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + self.gamma * values[t+1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advs[t] = gae
            rets = [a + v for a, v in zip(advs, values[:T])]

            for i, tr in enumerate(episode):
                out["graph_indices"].append(tr.graph_idx)
                out["history_actions"].append(tr.history_actions)
                out["actions"].append(tr.action_type)
                out["src_indices"].append(tr.src_idx)
                out["tgt_indices"].append(tr.tgt_idx)
                out["old_log_probs"].append(tr.log_prob)
                out["returns"].append(rets[i])
                out["advantages"].append(advs[i])

        advs_t = torch.tensor(out["advantages"], dtype=torch.float32)
        if len(advs_t) > 1:
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        out["actions"]       = torch.tensor(out["actions"], dtype=torch.long)
        out["src_indices"]   = torch.tensor(out["src_indices"], dtype=torch.long)
        out["tgt_indices"]   = torch.tensor(out["tgt_indices"], dtype=torch.long)
        out["old_log_probs"] = torch.tensor(out["old_log_probs"], dtype=torch.float32)
        out["returns"]       = torch.tensor(out["returns"], dtype=torch.float32)
        out["advantages"]    = advs_t
        return out

    def clear(self):
        self.trajectories.clear()
        self._current.clear()

    def __len__(self):
        return sum(len(ep) for ep in self.trajectories) + len(self._current)