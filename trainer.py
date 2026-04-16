"""完整训练入口示例"""

import torch
import logging
from config.config import MODEL_CONFIG, RL_CONFIG, REWARD_CONFIG
from models.policy import RetroSynthesisPolicy
from rl.ppo_trainer import PPOTrainer

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1. 创建策略 ──────────────────────────────────────────
vocab_size = 137  # 由 tokenizer.get_vocab_size() 确定
policy = RetroSynthesisPolicy(vocab_size, MODEL_CONFIG)

# ─── 2. 创建训练器 ────────────────────────────────────────
trainer = PPOTrainer(policy, RL_CONFIG, REWARD_CONFIG, device)

# ─── 3. 加载预训练 Actor ──────────────────────────────────
trainer.load_pretrained_actor("pretrained_actor.pt")

# ─── 4. 配置 LoRA + 冻结 + 优化器 ────────────────────────
trainer.setup()
# 此时:
#   - Graph Encoder: 完全冻结
#   - Actor 预训练权重: 冻结
#   - Actor LoRA 参数: 可训练 (极少量参数)
#   - State Tracker: 可训练
#   - Critic: 可训练 (独立网络)

# ─── 5. 训练循环 ──────────────────────────────────────────
# 需要实现 env 和 graph_data_fn (根据你的数据格式)
#
# for update in range(RL_CONFIG["total_updates