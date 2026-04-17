# config.py
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
from utils.chem import get_atom_feat_dim

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


# @dataclass
# class PathConfig:
#     ROOT_DIR: Path = Path(".")
#     DATA_DIR: Path = Path("dataset/uspto50k/")

#     # 预训练数据集
#     PRETRAIN_DATA_DIR: Path = Path("dataset/uspto50k/pretrained/")
#     PRETRAIN_TRAIN_DATA_FILE: Path = PRETRAIN_DATA_DIR / "uspto50k_train_output.json"
#     PRETRAIN_VAL_DATA_FILE: Path = PRETRAIN_DATA_DIR / "uspto50k_valid_output.json"
#     PRETRAIN_TEST_DATA_FILE: Path = PRETRAIN_DATA_DIR / "uspto50k_test_output.json"

#     # RL 数据集
#     RL_DATA_DIR: Path = Path("dataset/uspto50k/processed/")
#     RL_TRAIN_DATA_FILE: Path = RL_DATA_DIR / "uspto50k_train_output.json"
#     RL_VAL_DATA_FILE: Path = RL_DATA_DIR / "uspto50k_valid_output.json"
#     RL_TEST_DATA_FILE: Path = RL_DATA_DIR / "uspto50k_test_output.json"

#     # 分词器
#     TOKENIZER_DIR: Path = Path("tokenizer/")
#     VOCAB_FILE: Path = TOKENIZER_DIR / "vocab.txt"

#     # 模型检查点
#     CKPT_DIR: Path = Path(f"ckpt2/")
#     LOG_DIR: Path = CKPT_DIR / "log"
#     TB_DIR: Path = CKPT_DIR / "tensorboard"
#     CKPT_BEST_MODEL_FILE: Path = CKPT_DIR / "best_model.pt"
#     CKPT_LAST_MODEL_FILE: Path = CKPT_DIR / "actor_last.pt"
    
#     def __post_init__(self):
#         """创建必要的目录"""
#         for attr_name in dir(self):
#             attr = getattr(self, attr_name)
#             if isinstance(attr, Path) and attr_name.endswith('_DIR'):
#                 attr.mkdir(parents=True, exist_ok=True)

@dataclass
class PathConfig:
    # ── 唯一需要手动传入的参数 ──────────────────────────────────
    project_name: str = "pretrain2"

    # ── 固定根目录 ───────────────────────────────────────────────
    ROOT_DIR: Path = Path(".")
    DATA_DIR: Path = Path("dataset/uspto50k/")

    # ── 预训练数据集（固定，与 project 无关）─────────────────────
    PRETRAIN_DATA_DIR: Path  = Path("dataset/uspto50k/pretrained/")
    PRETRAIN_TRAIN_DATA_FILE: Path = field(init=False)
    PRETRAIN_VAL_DATA_FILE:   Path = field(init=False)
    PRETRAIN_TEST_DATA_FILE:  Path = field(init=False)

    # ── RL 数据集（固定，与 project 无关）────────────────────────
    RL_DATA_DIR: Path        = Path("dataset/uspto50k/processed/")
    RL_TRAIN_DATA_FILE: Path = field(init=False)
    RL_VAL_DATA_FILE:   Path = field(init=False)
    RL_TEST_DATA_FILE:  Path = field(init=False)

    # ── 分词器（固定）────────────────────────────────────────────
    TOKENIZER_DIR: Path = Path("tokenizer/")
    VOCAB_FILE:    Path = field(init=False)

    # ── 检查点（按 project_name 区分）────────────────────────────
    CKPT_DIR:            Path = field(init=False)
    LOG_DIR:             Path = field(init=False)
    TB_DIR:              Path = field(init=False)
    CKPT_BEST_MODEL_FILE: Path = field(init=False)
    CKPT_LAST_MODEL_FILE: Path = field(init=False)

    def __post_init__(self):
        # ── 数据文件路径 ─────────────────────────────────────────
        self.PRETRAIN_TRAIN_DATA_FILE = self.PRETRAIN_DATA_DIR / "uspto50k_train_output.json"
        self.PRETRAIN_VAL_DATA_FILE   = self.PRETRAIN_DATA_DIR / "uspto50k_valid_output.json"
        self.PRETRAIN_TEST_DATA_FILE  = self.PRETRAIN_DATA_DIR / "uspto50k_test_output.json"

        self.RL_TRAIN_DATA_FILE = self.RL_DATA_DIR / "uspto50k_train_output.json"
        self.RL_VAL_DATA_FILE   = self.RL_DATA_DIR / "uspto50k_valid_output.json"
        self.RL_TEST_DATA_FILE  = self.RL_DATA_DIR / "uspto50k_test_output.json"

        self.VOCAB_FILE = self.TOKENIZER_DIR / "vocab.txt"

        # ── 按 project_name 构建 ckpt 路径 ───────────────────────
        self.CKPT_DIR             = Path("ckpt") / self.project_name
        self.LOG_DIR              = self.CKPT_DIR / "log"
        self.TB_DIR               = self.CKPT_DIR / "tensorboard"
        self.CKPT_BEST_MODEL_FILE = self.CKPT_DIR / "best_model.pt"
        self.CKPT_LAST_MODEL_FILE = self.CKPT_DIR / "actor_last.pt"

        # ── 自动创建所有 _DIR 目录 ───────────────────────────────
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith("_DIR"):
                attr.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    BATCH_SIZE: int = 32

    # 分词器
    VOCAB_SIZE: int = 137

    # 图编码器
    NODE_IN_DIM: int = get_atom_feat_dim()       # RDKit 原子特征维度 (可调)
    NODE_DIM: int = 256        # GAT 内部维度
    HIDDEN_DIM: int = 512        # 解码器/策略网络维度
    GAT_LAYERS: int = 4
    GAT_HEADS: int = 4

    # 动作空间
    NUM_ACTIONS: int = NUM_ACTIONS
    PAD_ACTION_ID: int = PAD_ACTION_ID

    # 指针网络
    MAX_ATOMS: int = 100         # USPTO50K 最大原子数约50，留余量

    # 标签解码器
    MAX_POS_ENC: int = 128       # label序列最大长度

    # State Tracker
    GRU_LAYERS: int = 2

    # Pretrain 配置
    W_ACTION:  float = 1.0
    W_POINTER: float = 1.0
    W_LABEL:   float = 0.5

    MAX_GRAD_NORM: float = 0.5

# ════════════════════════════════════════════════════════════
#  预训练配置
# ════════════════════════════════════════════════════════════
@dataclass
class PretrainConfig:
    # 优化器
    lr:              float = 1e-4
    weight_decay:    float = 1e-4
    max_grad_norm:   float = 1.0

    # 调度器
    total_epochs:    int   = 50
    warmup_epochs:   int   = 3

    # DataLoader
    batch_size:      int   = 32
    num_workers:     int   = 4
    pin_memory:      bool  = True

    # 损失权重
    w_action:        float = 1.0
    w_pointer:       float = 1.0
    w_label:         float = 0.5
    label_pad_id:    int   = 0

    # 断点续训
    resume:          bool  = True   # 是否自动加载最近 checkpoint
    save_every:      int   = 5      # 每N个epoch保存一次

    # 早停
    early_stop_patience: int = 10


# ═══════════════════════════════════════════════════════════
# 全局配置实例
# ═══════════════════════════════════════════════════════════
@dataclass
class Config:
    path = PathConfig()
    model = ModelConfig()
    pretrain = PretrainConfig()
    
# 创建全局配置实例
config = Config()

