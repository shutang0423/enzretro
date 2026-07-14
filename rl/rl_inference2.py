"""
rl_inference.py —— 强化学习推理模块

提供多种强化学习推理方法，作为自回归推理的对比。
包括简单的蒙特卡洛采样、策略梯度等方法。
支持beam search多结果输出。
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from collections import namedtuple

from config.config import MODEL_CFG
from model.actor_network import ActorNetwork, EditStep
from model.state_tracker import HistoryBatch
from rl.reward_calculator import RewardCalculator, RewardConfig
from config.config import RLInferenceConfig

# 候选序列数据结构
BeamCandidate = namedtuple('BeamCandidate', ['edit_steps', 'total_reward', 'history', 'gru_hidden', 'finished'])


class RLInference:
    """强化学习推理器"""
    
    def __init__(self, model: ActorNetwork, config: RLInferenceConfig = None):
        self.model = model
        self.config = config or RLInferenceConfig()
        
        # 初始化奖励计算器
        reward_config = RewardConfig()
        self.reward_calculator = RewardCalculator(reward_config)
        
    def infer(
        self, 
        encoder_kwargs,
        target_steps: Optional[List[Dict[str, Any]]] = None,
        beam_size: int = 1
    ) -> List[List[EditStep]]:
        """
        使用强化学习方法进行推理
        
        Args:
            encoder_kwargs: 编码器参数
            target_steps: 目标步骤列表，用于计算精确奖励
            beam_size: beam search的大小，1表示单结果，>1表示多结果
            
        Returns:
            edit_steps_list: 生成的编辑步骤列表（多个候选结果）
        """
        if beam_size > 1:
            return self._beam_search_inference(encoder_kwargs, target_steps, beam_size)
        elif self.config.inference_method == "monte_carlo":
            return [self._monte_carlo_inference(encoder_kwargs, target_steps)]
        elif self.config.inference_method == "policy_gradient":
            return [self._policy_gradient_inference(encoder_kwargs)]
        else:
            raise ValueError(f"Unknown RL inference method: {self.config.inference_method}")
    
    def _monte_carlo_inference(
        self, 
        encoder_kwargs,
        target_steps: Optional[List[Dict[str, Any]]] = None
    ) -> List[EditStep]:
        """
        蒙特卡洛推理方法
        对每个步骤进行多次采样，选择平均回报最高的动作
        
        Args:
            encoder_kwargs: 编码器参数
            target_steps: 目标步骤列表，用于计算精确奖励
        """
        # 推断batch_size和device
        batch_size = 1
        first_val = next(iter(encoder_kwargs.values()))
        device = first_val.device
        
        # 初始化空历史
        label_len = MODEL_CFG.max_seq_len
        history = HistoryBatch.empty(batch_size, label_len, device)
        gru_hidden = None
        edit_steps = []
        
        for step_idx in range(self.config.max_steps):
            # 对当前步骤进行多次采样
            all_edits = []
            all_rewards = []
            
            for rollout_idx in range(self.config.num_rollouts):
                # 保存当前状态
                current_history = history
                current_gru_hidden = gru_hidden
                
                # 执行一步预测
                edit, new_gru_hidden = self.model.predict_step(
                    history=current_history,
                    gru_hidden=current_gru_hidden,
                    **encoder_kwargs
                )
                
                # 计算奖励
                if target_steps:
                    # 如果有目标步骤，使用精确的奖励计算
                    temp_steps = edit_steps + [edit]
                    reward = self.reward_calculator.calculate_reward(
                        temp_steps, target_steps, step_idx
                    )
                else:
                    # 如果没有目标步骤，使用简单的奖励
                    action_type = edit.action_type.item()
                    if action_type == MODEL_CFG.stop_action_id:
                        reward = 1.0 * (self.config.discount_factor ** step_idx)
                    else:
                        reward = 0.0
                
                all_edits.append(edit)
                all_rewards.append(reward)
            
            # 选择平均回报最高的动作
            best_idx = np.argmax(all_rewards)
            best_edit = all_edits[best_idx]
            
            # 更新历史
            history = self.model._append_history(history, best_edit, label_len)
            edit_steps.append(best_edit)
            
            # 检查终止条件
            action_type = best_edit.action_type.item()
            if action_type == MODEL_CFG.stop_action_id:
                break
        
        return edit_steps
    
    def _policy_gradient_inference(self, encoder_kwargs) -> List[EditStep]:
        """
        策略梯度推理方法
        使用温度采样，根据策略概率分布进行采样
        """
        # 推断batch_size和device
        batch_size = 1
        first_val = next(iter(encoder_kwargs.values()))
        device = first_val.device
        
        # 初始化空历史
        label_len = MODEL_CFG.max_seq_len
        history = HistoryBatch.empty(batch_size, label_len, device)
        gru_hidden = None
        edit_steps = []
        
        for _ in range(self.config.max_steps):
            # 获取动作概率分布
            with torch.no_grad():
                enc_out, graph_emb = self.model._encode(**encoder_kwargs)
                decoder_state, gru_hidden = self.model.state_tracker(
                    graph_emb, history, gru_hidden
                )
                
                # 获取动作logits
                action_logits = self.model.action_predictor(decoder_state)
                
                # 应用温度采样
                action_probs = torch.softmax(action_logits / self.config.temperature, dim=-1)
                
                # 采样动作
                action_dist = torch.distributions.Categorical(action_probs)
                sampled_action = action_dist.sample()
                
                # 使用采样的动作进行预测
                edit, gru_hidden = self.model.predict_step(
                    history=history,
                    gru_hidden=gru_hidden,
                    **encoder_kwargs
                )
            
            # 更新历史
            history = self.model._append_history(history, edit, label_len)
            edit_steps.append(edit)
            
            # 检查终止条件
            action_type = edit.action_type.item()
            if action_type == MODEL_CFG.stop_action_id:
                break
        
        return edit_steps


    def _beam_search_inference(
        self, 
        encoder_kwargs,
        target_steps: Optional[List[Dict[str, Any]]] = None,
        beam_size: int = 5
    ) -> List[List[EditStep]]:
        """
        Beam search推理方法
        维护多个候选序列，在每个步骤扩展并选择top-k个最佳候选
        
        Args:
            encoder_kwargs: 编码器参数
            target_steps: 目标步骤列表，用于计算精确奖励
            beam_size: beam的大小
            
        Returns:
            top_candidates: top-n个候选序列
        """
        # 推断batch_size和device
        first_val = next(iter(encoder_kwargs.values()))
        device = first_val.device
        
        # 初始化候选序列
        label_len = MODEL_CFG.max_seq_len
        initial_history = HistoryBatch.empty(1, label_len, device)
        
        candidates = [
            BeamCandidate(
                edit_steps=[],
                total_reward=0.0,
                history=initial_history,
                gru_hidden=None,
                finished=False
            )
        ]
        
        for step_idx in range(self.config.max_steps):
            # 扩展所有未完成的候选
            new_candidates = []
            
            for candidate in candidates:
                if candidate.finished:
                    new_candidates.append(candidate)
                    continue
                
                # 对当前候选进行多次采样
                for _ in range(self.config.num_rollouts):
                    # 执行一步预测
                    edit, new_gru_hidden = self.model.predict_step(
                        history=candidate.history,
                        gru_hidden=candidate.gru_hidden,
                        **encoder_kwargs
                    )
                    
                    # 计算奖励
                    if target_steps:
                        temp_steps = candidate.edit_steps + [edit]
                        reward = self.reward_calculator.calculate_reward(
                            temp_steps, target_steps, step_idx
                        )
                    else:
                        action_type = edit.action_type.item()
                        if action_type == MODEL_CFG.stop_action_id:
                            reward = 1.0 * (self.config.discount_factor ** step_idx)
                        else:
                            reward = 0.0
                    
                    # 更新历史
                    new_history = self.model._append_history(
                        candidate.history, edit, label_len
                    )
                    
                    # 检查是否终止
                    finished = (edit.action_type.item() == MODEL_CFG.stop_action_id)
                    
                    # 创建新候选
                    new_candidate = BeamCandidate(
                        edit_steps=candidate.edit_steps + [edit],
                        total_reward=candidate.total_reward + reward,
                        history=new_history,
                        gru_hidden=new_gru_hidden,
                        finished=finished
                    )
                    
                    new_candidates.append(new_candidate)
            
            # 选择top-k个候选
            new_candidates.sort(key=lambda x: x.total_reward, reverse=True)
            candidates = new_candidates[:beam_size]
            
            # 如果所有候选都已完成，提前终止
            if all(c.finished for c in candidates):
                break
        
        # 返回top-n个候选序列
        return [c.edit_steps for c in candidates]

    def _policy_gradient_inference(self, encoder_kwargs) -> List[EditStep]:
        """
        策略梯度推理方法
        使用温度采样，根据策略概率分布进行采样
        """
        # 推断batch_size和device
        batch_size = 1
        first_val = next(iter(encoder_kwargs.values()))
        device = first_val.device
        
        # 初始化空历史
        label_len = MODEL_CFG.max_seq_len
        history = HistoryBatch.empty(batch_size, label_len, device)
        gru_hidden = None
        edit_steps = []
        
        for _ in range(self.config.max_steps):
            # 获取动作概率分布
            with torch.no_grad():
                enc_out, graph_emb = self.model._encode(**encoder_kwargs)
                decoder_state, gru_hidden = self.model.state_tracker(
                    graph_emb, history, gru_hidden
                )
                
                # 获取动作logits
                action_logits = self.model.action_predictor(decoder_state)
                
                # 应用温度采样
                action_probs = torch.softmax(action_logits / self.config.temperature, dim=-1)
                
                # 采样动作
                action_dist = torch.distributions.Categorical(action_probs)
                sampled_action = action_dist.sample()
                
                # 使用采样的动作进行预测
                edit, gru_hidden = self.model.predict_step(
                    history=history,
                    gru_hidden=gru_hidden,
                    **encoder_kwargs
                )
            
            # 更新历史
            history = self.model._append_history(history, edit, label_len)
            edit_steps.append(edit)
            
            # 检查终止条件
            action_type = edit.action_type.item()
            if action_type == MODEL_CFG.stop_action_id:
                break
        
        return edit_steps



def compare_inference_methods(
    model: ActorNetwork, 
    encoder_kwargs,
    methods: List[str] = ["autoregressive", "monte_carlo", "policy_gradient"],
    target_steps: Optional[List[Dict[str, Any]]] = None
) -> dict:
    """
    比较不同推理方法的结果
    
    Args:
        model: ActorNetwork模型
        encoder_kwargs: 编码器参数
        methods: 要比较的方法列表
        target_steps: 目标步骤列表，用于计算精确奖励
        
    Returns:
        results: 各方法的推理结果
    """
    results = {}
    
    # 自回归推理
    if "autoregressive" in methods:
        with torch.no_grad():
            edit_steps_ar = model.generate(**encoder_kwargs)
        results["autoregressive"] = edit_steps_ar
    
    # 强化学习推理
    if "monte_carlo" in methods:
        rl_config = RLInferenceConfig(inference_method="monte_carlo")
        rl_inference = RLInference(model, rl_config)
        edit_steps_mc = rl_inference.infer(encoder_kwargs, target_steps)
        results["monte_carlo"] = edit_steps_mc
    
    if "policy_gradient" in methods:
        rl_config = RLInferenceConfig(inference_method="policy_gradient")
        rl_inference = RLInference(model, rl_config)
        edit_steps_pg = rl_inference.infer(encoder_kwargs, target_steps)
        results["policy_gradient"] = edit_steps_pg
    
    return results