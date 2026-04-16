"""逆合成 RL 环境

将 USPTO50K 数据集包装为 RL 环境接口。
GT 编辑序列用于计算奖励，不用于策略更新。
"""

import torch
from typing import Dict, Tuple, List, Optional
from config.config import TERMINATE_ACTION_ID, ACTION_TO_ID


class RetroSynthesisEnv:
    """单样本逆合成环境

    step() 返回:
        next_obs : Dict  (图数据不变，history 更新)
        gt_info  : Dict  (当前步的 GT 信息，用于奖励计算)
        done     : bool
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self._sample: Optional[Dict] = None
        self._step_idx = 0
        self._history: List[int] = []

    def reset(self, sample: Dict) -> Dict:
        """加载一个样本，返回初始观测"""
        self._sample = sample
        self._step_idx = 0
        self._history = []
        return self._make_obs()

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, bool]:
        """执行一步

        Args:
            action_dict: {"action_type": int, "src_idx": int, "tgt_idx": int}

        Returns:
            obs, gt_info, done
        """
        pred_action = action_dict["action_type"]
        pred_src    = action_dict["src_idx"]
        pred_tgt    = action_dict["tgt_idx"]

        # 获取当前步的 GT
        gt_edits = self._sample["edits"]
        gt_info  = {}

        if self._step_idx < len(gt_edits):
            gt_edit = gt_edits[self._step_idx]
            gt_info = {
                "action_type": gt_edit["action_id"],
                "src_idx":     gt_edit["src_idx"] if gt_edit["src_valid"] else None,
                "tgt_idx":     gt_edit["tgt_idx"] if gt_edit["tgt_valid"] else None,
                "is_valid":    True,
                "match_score": self._compute_match_score(
                    pred_action, pred_src, pred_tgt, gt_edit
                ),
            }
        else:
            # 超出 GT 步数：视为无效
            gt_info = {"is_valid": False, "match_score": 0.0}

        # 更新历史
        self._history.append(pred_action)
        self._step_idx += 1

        # 终止条件
        done = (
            pred_action == TERMINATE_ACTION_ID
            or self._step_idx >= self.max_steps
            or self._step_idx >= len(gt_edits) + 2  # 允许多走2步
        )

        return self._make_obs(), gt_info, done

    def _make_obs(self) -> Dict:
        return {
            "graph":         self._sample["graph"],
            "history":       list(self._history),
            "step_idx":      self._step_idx,
            "num_gt_edits":  self._sample["num_edits"],
            "graph_idx":     self._sample.get("rxn_id", 0),
        }

    def _compute_match_score(self, pred_action, pred_src, pred_tgt, gt_edit) -> float:
        """计算单步匹配分数 [0, 1]"""
        score = 0.0
        if pred_action == gt_edit["action_id"]:
            score += 0.5
            if gt_edit["src_valid"] and pred_src == gt_edit["src_idx"]:
                score += 0.25
            if gt_edit["tgt_valid"] and pred_tgt == gt_edit["tgt_idx"]:
                score += 0.25
        return score

    @property
    def history_tensor(self) -> Optional[torch.Tensor]:
        if not self._history:
            return None
        return torch.tensor([self._history], dtype=torch.long)


class BatchRetroEnv:
    """批量环境包装器 (并行处理多个样本)"""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.envs: List[RetroSynthesisEnv] = []

    def reset(self, samples: List[Dict]) -> List[Dict]:
        self.envs = [RetroSynthesisEnv(self.max_steps) for _ in samples]
        return [env.reset(s) for env, s in zip(self.envs, samples)]

    def step(self, action_dicts: List[Dict]) -> Tuple[List[Dict], List[Dict], List[bool]]:
        results = [env.step(a) for env, a in zip(self.envs, action_dicts)]
        obs_list  = [r[0] for r in results]
        gt_list   = [r[1] for r in results]
        done_list = [r[2] for r in results]
        return obs_list, gt_list, done_list