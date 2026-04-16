# config.py
# # 统一管理所有超参数，避免各文件中硬编码导致的越界
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 动作类型定义 (与数据集 action_type 字段对齐) ─────────────
ACTION_TYPES = [
    "DeleteBond",      # 0
    "ChangeBond",      # 1
    "ChangeAtom",      # 2
    "AddAtom",         # 3
    "AttachGroup",     # 4
    "AddRing",         # 5
    "Terminate",       # 6
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_TYPES)}
ID_TO_ACTION = {i: a for i, a in enumerate(ACTION_TYPES)}
TERMINATE_ACTION_ID = ACTION_TO_ID["Terminate"]   # 6
NUM_ACTIONS = len(ACTION_TYPES)                   # 7
PAD_ACTION_ID = NUM_ACTIONS                       # 7 (用于 padding)

PATH_CONFIG = {
    "vocab_file":   "tokenizer/vocab.txt",
    "train_data":   "dataset/uspto50k/pretrained/uspto50k_train_output.json",
    "val_data":     "dataset/uspto50k/pretrained/uspto50k_valid_output.json",
    "test_data":    "dataset/uspto50k/pretrained/uspto50k_test_output.json",
    "log_dir":      f"ckpt/{current_time}",
    "ckpt_path":    f"ckpt/{current_time}/best_model.pt",
}


MODEL_CONFIG = {
    # 图编码器
    "node_in_dim":  39,          # RDKit 原子特征维度 (可调)
    "node_dim":     256,         # GAT 内部维度
    "hidden_dim":   512,         # 解码器/策略网络维度
    "gat_layers":   4,
    "gat_heads":    4,

    # 动作空间
    "num_actions":  NUM_ACTIONS,
    "pad_action_id": PAD_ACTION_ID,

    # 指针网络
    "max_atoms":    100,         # USPTO50K 最大原子数约50，留余量

    # 标签解码器
    "max_pos_enc":  128,         # label序列最大长度

    # State Tracker
    "gru_layers":   2,
}


# ── RL 训练配置 ───────────────────────────────────────────────
RL_CONFIG = {
    # PPO 超参
    "clip_eps":           0.2,
    "ppo_epochs":         4,
    "mini_batch_size":    32,
    "gamma":              0.99,
    "gae_lambda":         0.95,
    "entropy_coef":       0.01,
    "value_loss_coef":    0.5,
    "max_grad_norm":      0.5,

    # 优化器
    "lr_actor":           1e-5,   # LoRA 学习率要小
    "lr_critic":          3e-4,

    # KL 约束 (防止偏离预训练策略)
    "kl_coef":            0.1,
    "kl_target":          0.01,

    # LoRA
    "use_lora":           True,
    "lora_rank":          8,
    "lora_alpha":         16.0,
    "freeze_encoder":     True,

    # 训练流程
    "total_updates":      500,
    "episodes_per_update": 64,
    "max_steps_per_episode": 10,  # USPTO50K 平均3-5步，上限10

    # 动作 padding
    "pad_action_id":      PAD_ACTION_ID,
}

# ── 奖励配置 ──────────────────────────────────────────────────
REWARD_CONFIG = {
    "step_penalty":            -0.02,
    "invalid_action_penalty":  -0.5,
    "correct_action_bonus":     0.5,
    "wrong_action_penalty":    -0.2,
    "failure_penalty":         -5.0,
    "success_reward":          10.0,
    "early_terminate_penalty": -3.0,
    "correct_terminate_bonus":  2.0,
}

# ── 标签特殊 token ────────────────────────────────────────────
LABEL_SPECIAL = {
    "NONE_TOKEN":      "NONE",
    "TERMINATE_TOKEN": "Terminate",
    "BOS_TOKEN":       "<bos>",
    "EOS_TOKEN":       "<eos>",
    "PAD_TOKEN":       "<pad>",
    "UNK_TOKEN":       "<unk>",
}