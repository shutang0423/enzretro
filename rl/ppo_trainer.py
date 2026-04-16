"""MCTS 推理模块 (修复版)

核心修复:
  1. 子节点 state 不再为 None
  2. 通过 state_update_fn 正确更新状态
  3. 支持与 RetroSynthesisPolicy 配合使用
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Callable


class MCTSNode:
    def __init__(self, state: torch.Tensor, parent=None,
                 action_taken: int = -1, prior_prob: float = 1.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_terminal = False

    @property
    def value(self):
        return self.total_value / max(self.visit_count, 1)

    def ucb_score(self, c: float = 1.414) -> float:
        if self.visit_count == 0:
            return float('inf')
        exploit = self.value
        explore = c * self.prior_prob * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploit + explore

    def best_child(self, c: float = 1.414):
        return max(self.children.items(), key=lambda x: x[1].ucb_score(c))


class MCTSPolicy:
    """MCTS 策略: 使用 policy 网络引导搜索"""

    def __init__(self, policy, state_update_fn: Callable,
                 num_simulations: int = 50, c_puct: float = 1.414,
                 temperature: float = 1.0, max_depth: int = 10,
                 terminate_action_id: int = 6, device=None):
        """
        Args:
            policy: RetroSynthesisPolicy 实例
            state_update_fn: callable(state, action) -> new_state
                根据动作更新状态的函数 (由环境提供)
        """
        self.policy = policy
        self.state_update_fn = state_update_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.max_depth = max_depth
        self.terminate_id = terminate_action_id
        self.device = device or torch.device('cpu')

    @torch.no_grad()
    def _evaluate(self, state: torch.Tensor):
        """获取动作概率和状态价值"""
        act_logits = self.policy.actor.action_predictor(state)
        probs = F.softmax(act_logits, dim=-1)
        value = self.policy.critic.value_head(
            self.policy.critic._encode_state_from_hidden(state)
        ) if hasattr(self.policy.critic, '_encode_state_from_hidden') else torch.zeros(1)
        return probs[0], value.item()

    def _expand(self, node: MCTSNode, action_probs: torch.Tensor,
                max_children: int = 5):
        """扩展节点: 创建子节点并正确设置 state"""
        k = min(len(action_probs), max_children)
        top_probs, top_actions = torch.topk(action_probs, k)

        for prob, action in zip(top_probs, top_actions):
            a = action.item()
            if a in node.children:
                continue
            # 关键修复: 通过 state_update_fn 计算子节点状态
            child_state = self.state_update_fn(node.state, a)
            child = MCTSNode(
                state=child_state, parent=node,
                action_taken=a, prior_prob=prob.item()
            )
            if a == self.terminate_id:
                child.is_terminal = True
            node.children[a] = child

    def simulate(self, root: MCTSNode) -> float:
        """一次 MCTS 模拟: Select → Expand → Evaluate → Backprop"""
        node = root
        path = [node]
        depth = 0

        # Select
        while node.children and not node.is_terminal and depth < self.max_depth:
            _, node = node.best_child(self.c_puct)
            path.append(node)
            depth += 1

        # Expand + Evaluate
        if not node.is_terminal and depth < self.max_depth and node.state is not None:
            probs, value = self._evaluate(node.state)
            self._expand(node, probs)
        else:
            value = node.value

        # Backprop
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value
            value *= 0.95

        return value

    def search(self, initial_state: torch.Tensor,
               temperature: float = None) -> Tuple[int, Dict]:
        """执行 MCTS 搜索"""
        temp = temperature or self.temperature
        probs, root_val = self._evaluate(initial_state)
        root = MCTSNode(initial_state)
        self._expand(root, probs)

        for _ in range(self.num_simulations):
            self.simulate(root)

        if not root.children:
            return 0, {"error": "no_children"}

        visits = np.array([c.visit_count for c in root.children.values()])
        actions = list(root.children.keys())

        if temp > 0 and len(actions) > 1:
            p = visits ** (1.0 / temp)
            p = p / (p.sum() + 1e-8)
            best = np.random.choice(actions, p=p)
        else:
            best = actions[np.argmax(visits)]

        return best, {
            "visits": {a: root.children[a].visit_count for a in actions},
            "values": {a: root.children[a].value for a in actions},
            "root_value": root_val,
        }