"""
reward_calculator.py —— 奖励计算模块

提供多种奖励计算方法，用于强化学习推理。
可以根据真实的步骤进行比较来计算精确的奖励。
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config.config import MODEL_CFG
from model.actor_network import EditStep


@dataclass
class RewardConfig:
    """奖励计算配置"""
    method: str = "step_comparison"  # "step_comparison", "action_only", "comprehensive"
    action_weight: float = 1.0
    src_weight: float = 0.5
    tgt_weight: float = 0.5
    label_weight: float = 0.8
    step_weight: float = 0.3
    discount_factor: float = 0.9


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
    
    def calculate_reward(
        self, 
        predicted_steps: List[EditStep], 
        target_steps: List[Dict[str, Any]],
        step_idx: int = 0
    ) -> float:
        """
        计算奖励值
        
        Args:
            predicted_steps: 预测的编辑步骤列表
            target_steps: 目标编辑步骤列表（字典格式）
            step_idx: 当前步骤索引
            
        Returns:
            reward: 计算得到的奖励值
        """
        if self.config.method == "step_comparison":
            return self._step_comparison_reward(predicted_steps, target_steps, step_idx)
        elif self.config.method == "action_only":
            return self._action_only_reward(predicted_steps, target_steps, step_idx)
        elif self.config.method == "comprehensive":
            return self._comprehensive_reward(predicted_steps, target_steps, step_idx)
        else:
            raise ValueError(f"Unknown reward method: {self.config.method}")
    
    def _step_comparison_reward(
        self, 
        predicted_steps: List[EditStep], 
        target_steps: List[Dict[str, Any]],
        step_idx: int
    ) -> float:
        """
        步骤比较奖励：根据预测步骤与目标步骤的匹配程度计算奖励
        """
        if step_idx >= len(predicted_steps) or step_idx >= len(target_steps):
            return 0.0
        
        pred_step = predicted_steps[step_idx]
        target_step = target_steps[step_idx]
        
        # 提取预测值
        pred_action = pred_step.action_type.item()
        pred_src = pred_step.src_idx.item()
        pred_tgt = pred_step.tgt_idx.item()
        
        # 提取目标值
        target_action = target_step.get('target_action_type', -1)
        target_src = target_step.get('target_src_idx', -1)
        target_tgt = target_step.get('target_tgt_idx', -1)
        
        # 计算各项匹配度
        action_match = 1.0 if pred_action == target_action else 0.0
        src_match = 1.0 if pred_src == target_src else 0.0
        tgt_match = 1.0 if pred_tgt == target_tgt else 0.0
        
        # 计算标签匹配度（需要解码）
        label_match = self._calculate_label_match(pred_step, target_step)
        
        # 加权计算总奖励
        total_reward = (
            self.config.action_weight * action_match +
            self.config.src_weight * src_match +
            self.config.tgt_weight * tgt_match +
            self.config.label_weight * label_match
        )
        
        # 应用折扣因子
        discounted_reward = total_reward * (self.config.discount_factor ** step_idx)
        
        return discounted_reward
    
    def _action_only_reward(
        self, 
        predicted_steps: List[EditStep], 
        target_steps: List[Dict[str, Any]],
        step_idx: int
    ) -> float:
        """
        仅动作奖励：只关注动作类型的匹配
        """
        if step_idx >= len(predicted_steps) or step_idx >= len(target_steps):
            return 0.0
        
        pred_step = predicted_steps[step_idx]
        target_step = target_steps[step_idx]
        
        pred_action = pred_step.action_type.item()
        target_action = target_step.get('target_action_type', -1)
        
        action_match = 1.0 if pred_action == target_action else 0.0
        
        # 如果是终止动作且匹配，给予额外奖励
        if pred_action == MODEL_CFG.stop_action_id and action_match == 1.0:
            reward = 1.0
        else:
            reward = action_match
        
        # 应用折扣因子
        discounted_reward = reward * (self.config.discount_factor ** step_idx)
        
        return discounted_reward
    
    def _comprehensive_reward(
        self, 
        predicted_steps: List[EditStep], 
        target_steps: List[Dict[str, Any]],
        step_idx: int
    ) -> float:
        """
        综合奖励：考虑步骤序列的整体匹配度
        """
        # 计算当前步骤的匹配度
        step_reward = self._step_comparison_reward(predicted_steps, target_steps, step_idx)
        
        # 计算序列长度匹配度
        pred_len = len(predicted_steps)
        target_len = len(target_steps)
        length_match = 1.0 - abs(pred_len - target_len) / max(pred_len, target_len, 1)
        
        # 计算序列顺序匹配度
        sequence_match = self._calculate_sequence_match(predicted_steps, target_steps)
        
        # 综合计算奖励
        total_reward = (
            step_reward +
            self.config.step_weight * length_match +
            self.config.step_weight * sequence_match
        )
        
        # 归一化到[0, 1]范围
        normalized_reward = min(max(total_reward, 0.0), 1.0)
        
        return normalized_reward
    
    def _calculate_label_match(self, pred_step: EditStep, target_step: Dict[str, Any]) -> float:
        """计算标签匹配度"""
        # 这里需要根据具体的标签解码逻辑来实现
        # 暂时返回一个默认值
        return 0.5
    
    def _calculate_sequence_match(
        self, 
        predicted_steps: List[EditStep], 
        target_steps: List[Dict[str, Any]]
    ) -> float:
        """计算序列顺序匹配度"""
        if not predicted_steps or not target_steps:
            return 0.0
        
        min_len = min(len(predicted_steps), len(target_steps))
        if min_len == 0:
            return 0.0
        
        # 计算前min_len个步骤的动作类型匹配度
        match_count = 0
        for i in range(min_len):
            pred_action = predicted_steps[i].action_type.item()
            target_action = target_steps[i].get('target_action_type', -1)
            if pred_action == target_action:
                match_count += 1
        
        return match_count / min_len


def batch_calculate_rewards(
    calculator: RewardCalculator,
    all_predicted_steps: List[List[EditStep]],
    all_target_steps: List[List[Dict[str, Any]]]
) -> List[float]:
    """
    批量计算奖励
    
    Args:
        calculator: 奖励计算器
        all_predicted_steps: 所有样本的预测步骤列表
        all_target_steps: 所有样本的目标步骤列表
        
    Returns:
        rewards: 每个样本的奖励值列表
    """
    rewards = []
    
    for pred_steps, target_steps in zip(all_predicted_steps, all_target_steps):
        if not pred_steps or not target_steps:
            rewards.append(0.0)
            continue
        
        # 计算每个步骤的奖励并取平均
        step_rewards = []
        for step_idx in range(min(len(pred_steps), len(target_steps))):
            reward = calculator.calculate_reward(pred_steps, target_steps, step_idx)
            step_rewards.append(reward)
        
        if step_rewards:
            avg_reward = sum(step_rewards) / len(step_rewards)
        else:
            avg_reward = 0.0
        
        rewards.append(avg_reward)
    
    return rewards