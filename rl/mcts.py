"""
mcts_generator.py —— Monte Carlo Tree Search for Molecular Editing

核心思想：
  通过模拟搜索在编辑动作空间中找到高回报的编辑序列
  适合需要全局优化和探索的分子生成任务

组件：
  - MCTSNode: 搜索树节点
  - MCTSState: 状态封装（history + gru_hidden）
  - RewardFunction: 可插拔的奖励函数
  - MCTSGenerator: 主搜索逻辑

使用示例：
  mcts = MCTSGenerator(actor_network, reward_fn, num_simulations=100)
  top_n_sequences = mcts.search(top_n=5, **encoder_kwargs)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F

from model.state_tracker import HistoryBatch
from model.actor_network import ActorNetwork, EditStep


# ══════════════════════════════════════════════════════════════════════
#  状态与节点定义
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSState:
    """
    MCTS 搜索状态封装
    
    Attributes:
        history     : 当前编辑历史
        gru_hidden  : GRU 隐状态
        is_terminal : 是否到达终止状态（STOP 或 max_steps）
        depth       : 当前深度（编辑步数）
    """
    history     : HistoryBatch
    gru_hidden  : Optional[torch.Tensor]
    is_terminal : bool = False
    depth       : int = 0


@dataclass
class MCTSNode:
    """
    MCTS 搜索树节点
    
    Attributes:
        state           : 当前状态
        parent          : 父节点
        action          : 从父节点到当前节点的动作
        children        : 子节点字典 {action_hash: MCTSNode}
        visits          : 访问次数
        total_reward    : 累计奖励
        untried_actions : 未尝试的动作列表
    """
    state           : MCTSState
    parent          : Optional[MCTSNode] = None
    action          : Optional[EditStep] = None
    children        : Dict[int, MCTSNode] = field(default_factory=dict)
    visits          : int = 0
    total_reward    : float = 0.0
    untried_actions : List[EditStep] = field(default_factory=list)
    
    @property
    def q_value(self) -> float:
        """平均奖励（Q值）"""
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def ucb1_score(self, exploration_weight: float = 1.414) -> float:
        """UCB1 上界置信度分数"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.q_value
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """是否所有动作都已扩展"""
        return len(self.untried_actions) == 0


# ══════════════════════════════════════════════════════════════════════
#  奖励函数接口
# ══════════════════════════════════════════════════════════════════════

class RewardFunction(ABC):
    """奖励函数抽象基类"""
    
    @abstractmethod
    def compute_terminal_reward(
        self, 
        edit_sequence: List[EditStep],
        **kwargs
    ) -> float:
        """计算终止状态的奖励"""
        pass
    
    def compute_step_reward(self, edit_step: EditStep, depth: int) -> float:
        """计算单步奖励（默认：鼓励短序列）"""
        return -0.01


class SimpleReward(RewardFunction):
    """
    简单奖励函数示例
    
    终止奖励 = 序列长度惩罚 + 有效性奖励
    可替换为实际的分子性质预测模型
    """
    
    def __init__(self, length_penalty: float = 0.1):
        self.length_penalty = length_penalty
    
    def compute_terminal_reward(
        self, 
        edit_sequence: List[EditStep],
        **kwargs
    ) -> float:
        """
        示例：基于序列长度的简单奖励
        实际应用中应替换为：
          - 分子性质预测得分
          - 与目标分子的相似度
          - 可合成性评分
        """
        length = len(edit_sequence)
        
        # 基础奖励：成功生成完整序列
        base_reward = 1.0
        
        # 长度惩罚：鼓励简洁的编辑序列
        length_penalty = -self.length_penalty * length
        
        # 有效性奖励：检查最后一步是否为 STOP
        validity_bonus = 0.5 if edit_sequence[-1].action_type == 0 else 0.0
        
        return base_reward + length_penalty + validity_bonus


# ══════════════════════════════════════════════════════════════════════
#  MCTS 主搜索器
# ══════════════════════════════════════════════════════════════════════

class MCTSGenerator:
    """
    Monte Carlo Tree Search 生成器
    
    Args:
        actor_network    : 训练好的 ActorNetwork
        reward_fn        : 奖励函数实例
        num_simulations  : 每次搜索的模拟次数
        max_depth        : 最大搜索深度
        exploration_weight: UCB1 探索权重
        top_k_actions    : 动作剪枝：保留前 K 个 action_type
        top_k_atoms      : 动作剪枝：保留前 K 个 src/tgt
        use_progressive_widening: 是否使用渐进扩展
    """
    
    def __init__(
        self,
        actor_network    : ActorNetwork,
        reward_fn        : RewardFunction,
        num_simulations  : int = 100,
        max_depth        : int = 10,
        exploration_weight: float = 1.414,
        top_k_actions    : int = 5,
        top_k_atoms      : int = 3,
        use_progressive_widening: bool = True,
    ):
        self.actor = actor_network
        self.actor.eval()  # 推理模式
        
        self.reward_fn = reward_fn
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.top_k_actions = top_k_actions
        self.top_k_atoms = top_k_atoms
        self.use_progressive_widening = use_progressive_widening
        
        self.stop_action_id = actor_network.stop_action_id
    
    # ──────────────────────────────────────────────────────────────────
    #  公共接口
    # ──────────────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def search(
        self,
        top_n: int = 5,
        label_len: Optional[int] = None,
        **encoder_kwargs
    ) -> List[Tuple[List[EditStep], float]]:
        """
        执行 MCTS 搜索，返回 Top-N 编辑序列
        
        Args:
            top_n          : 返回前 N 个最佳序列
            label_len      : HistoryBatch 的 label 维度
            encoder_kwargs : 传给 encoder 的图数据
        
        Returns:
            [(edit_sequence_1, score_1), ..., (edit_sequence_n, score_n)]
            按 score 降序排列
        """
        # 缓存 encoder 输出（避免重复编码）
        self.encoder_kwargs = encoder_kwargs
        
        # 初始化根节点
        device = next(iter(encoder_kwargs.values())).device
        batch_size = 1  # MCTS 通常单样本搜索
        
        if label_len is None:
            label_len = self.actor.cfg.max_seq_len
        
        root_state = MCTSState(
            history    = HistoryBatch.empty(batch_size, label_len, device),
            gru_hidden = None,
            is_terminal= False,
            depth      = 0,
        )
        root = MCTSNode(state=root_state)
        
        # 执行 N 次模拟
        for _ in range(self.num_simulations):
            node = self._select(root)
            
            if not node.state.is_terminal and node.state.depth < self.max_depth:
                node = self._expand(node)
            
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        
        # 收集所有完整路径并排序
        complete_paths = self._collect_complete_paths(root)
        
        # 返回 Top-N
        return sorted(complete_paths, key=lambda x: x[1], reverse=True)[:top_n]
    
    # ──────────────────────────────────────────────────────────────────
    #  MCTS 四阶段
    # ──────────────────────────────────────────────────────────────────
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection: 使用 UCB1 选择最有潜力的叶节点
        """
        while not node.state.is_terminal:
            if not node.is_fully_expanded():
                return node  # 返回未完全扩展的节点
            
            # 选择 UCB1 分数最高的子节点
            node = max(
                node.children.values(),
                key=lambda n: n.ucb1_score(self.exploration_weight)
            )
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion: 扩展一个新的子节点
        """
        # 首次访问：生成所有候选动作
        if node.visits == 0 and len(node.untried_actions) == 0:
            node.untried_actions = self._generate_candidate_actions(node.state)
        
        # 渐进扩展：根据访问次数动态调整扩展数量
        if self.use_progressive_widening:
            max_children = int(self.top_k_actions * math.log(node.visits + 2))
            if len(node.children) >= max_children:
                return node  # 已达到扩展上限
        
        # 选择一个未尝试的动作
        if len(node.untried_actions) == 0:
            return node
        
        action = node.untried_actions.pop(0)
        
        # 应用动作，生成新状态
        new_state = self._apply_action(node.state, action)
        
        # 创建子节点
        child = MCTSNode(
            state  = new_state,
            parent = node,
            action = action,
        )
        
        # 添加到父节点
        action_hash = self._hash_action(action)
        node.children[action_hash] = child
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation: 从当前节点 Rollout 到终止状态，返回奖励
        
        策略：使用 Actor Network 的贪心策略快速 Rollout
        """
        if node.state.is_terminal:
            # 已到达终止状态，直接计算奖励
            path = self._extract_path(node)
            return self.reward_fn.compute_terminal_reward(path, **self.encoder_kwargs)
        
        # Rollout：使用 Actor Network 贪心生成
        state = node.state
        rollout_steps = []
        
        for _ in range(self.max_depth - state.depth):
            edit, gru_hidden = self.actor.predict_step(
                history       = state.history,
                gru_hidden    = state.gru_hidden,
                **self.encoder_kwargs
            )
            
            rollout_steps.append(edit)
            
            # 检查终止
            if edit.action_type.item() == self.stop_action_id:
                break
            
            # 更新状态
            state = self._apply_action(state, edit)
        
        # 计算完整路径的奖励
        full_path = self._extract_path(node) + rollout_steps
        return self.reward_fn.compute_terminal_reward(full_path, **self.encoder_kwargs)
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation: 将奖励回传到路径上的所有节点
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    # ──────────────────────────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────────────────────────
    
    def _generate_candidate_actions(self, state: MCTSState) -> List[EditStep]:
        """
        生成候选动作列表（带剪枝）
        
        策略：
          1. 用 Actor Network 预测概率分布
          2. 对 action_type 取 Top-K
          3. 对每个 action，src/tgt 各取 Top-K
          4. 组合生成候选动作
        """
        # 获取 Actor Network 的预测分布
        enc_out, graph_emb = self.actor._encode(**self.encoder_kwargs)
        decoder_state, _ = self.actor.state_tracker(
            graph_emb, state.history, state.gru_hidden
        )
        
        # Action Type Top-K
        action_logits = self.actor.action_predictor(decoder_state)
        top_k_actions = torch.topk(action_logits, k=self.top_k_actions, dim=-1)
        
        candidates = []
        
        for action_idx, action_score in zip(top_k_actions.indices[0], top_k_actions.values[0]):
            action_type = action_idx.unsqueeze(0)  # [1]
            
            # Pointer Top-K
            src_logits, tgt_logits = self.actor.pointer_network(
                decoder_state  = decoder_state,
                dense_nodes    = enc_out.dense_nodes,
                action_type    = action_type,
                target_src_idx = None,
                node_mask      = enc_out.node_pad_mask,
                has_nodes      = enc_out.has_nodes,
            )
            
            top_k_src = torch.topk(src_logits, k=self.top_k_atoms, dim=-1)
            top_k_tgt = torch.topk(tgt_logits, k=self.top_k_atoms, dim=-1)
            
            # 组合生成候选
            for src_idx in top_k_src.indices[0]:
                for tgt_idx in top_k_tgt.indices[0]:
                    # Label 使用贪心解码（简化）
                    label_tokens = self.actor.label_decoder.greedy_decode(
                        decoder_state = decoder_state,
                        action_type   = action_type,
                        bos_token_id  = self.actor.bos_token_id,
                        max_len       = self.actor.label_max_len,
                    )
                    
                    candidates.append(EditStep(
                        action_type  = action_type,
                        src_idx      = src_idx.unsqueeze(0),
                        tgt_idx      = tgt_idx.unsqueeze(0),
                        label_tokens = label_tokens,
                    ))
        
        return candidates
    
    def _apply_action(self, state: MCTSState, action: EditStep) -> MCTSState:
        """应用动作，生成新状态"""
        # 更新 history
        new_history = self.actor._append_history(
            state.history, action, self.actor.cfg.max_seq_len
        )
        
        # 更新 gru_hidden（需要重新前向）
        enc_out, graph_emb = self.actor._encode(**self.encoder_kwargs)
        _, new_hidden = self.actor.state_tracker(
            graph_emb, new_history, state.gru_hidden
        )
        
        # 检查终止
        is_terminal = (
            action.action_type.item() == self.stop_action_id or
            state.depth + 1 >= self.max_depth
        )
        
        return MCTSState(
            history     = new_history,
            gru_hidden  = new_hidden,
            is_terminal = is_terminal,
            depth       = state.depth + 1,
        )
    
    def _extract_path(self, node: MCTSNode) -> List[EditStep]:
        """从根节点到当前节点的完整路径"""
        path = []
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))
    
    def _collect_complete_paths(self, root: MCTSNode) -> List[Tuple[List[EditStep], float]]:
        """收集所有到达终止状态的完整路径"""
        paths = []
        
        def dfs(node: MCTSNode):
            if node.state.is_terminal:
                path = self._extract_path(node)
                score = node.q_value  # 使用平均奖励作为分数
                paths.append((path, score))
            else:
                for child in node.children.values():
                    dfs(child)
        
        dfs(root)
        return paths
    
    @staticmethod
    def _hash_action(action: EditStep) -> int:
        """为动作生成唯一哈希值"""
        return hash((
            action.action_type.item(),
            action.src_idx.item(),
            action.tgt_idx.item(),
            tuple(action.label_tokens[0].tolist()),
        ))


# ══════════════════════════════════════════════════════════════════════
#  使用示例
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    使用示例（伪代码）
    """
    # 1. 加载训练好的 ActorNetwork
    # actor_network = ActorNetwork(vocab_size=100)
    # actor_network.load_state_dict(torch.load("actor.pth"))
    
    # 2. 定义奖励函数
    reward_fn = SimpleReward(length_penalty=0.1)
    
    # 3. 创建 MCTS 生成器
    mcts = MCTSGenerator(
        actor_network    = None,  # actor_network,
        reward_fn        = reward_fn,
        num_simulations  = 100,
        max_depth        = 10,
        exploration_weight = 1.414,
        top_k_actions    = 5,
        top_k_atoms      = 3,
    )
    
    # 4. 执行搜索
    # top_5_sequences = mcts.search(
    #     top_n=5,
    #     x=graph_x,
    #     edge_index=graph_edge_index,
    #     batch=graph_batch,
    # )
    
    # 5. 输出结果
    # for i, (edit_steps, score) in enumerate(top_5_sequences):
    #     print(f"Sequence {i+1} (Score: {score:.4f}):")
    #     for step in edit_steps:
    #         print(f"  Action: {step.action_type.item()}, "
    #               f"Src: {step.src_idx.item()}, "
    #               f"Tgt: {step.tgt_idx.item()}")
