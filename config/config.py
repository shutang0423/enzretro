"""
config/config.py —— 全局配置中心

所有超参统一在此定义，外部模块只 import 实例，不硬编码任何数值。

使用方式：
  from config.config import PATH_CFG, MODEL_CFG, TRAIN_CFG, LORA_CFG, RL_CFG

结构：
  动作类型常量          (模块级，非 dataclass)
  PathConfig            路径管理，__post_init__ 自动创建目录
  ModelConfig           网络结构超参
  StageConfig           单阶段课程学习配置
  TrainConfig           训练策略超参（含课程学习阶段配置）
  LoRAConfig            LoRA 注入配置（Phase 2 预留）
  RLConfig              强化学习超参（Phase 2 预留）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils.chem import get_atom_feat_dim


# ══════════════════════════════════════════════════════════════════════
#  动作类型常量（与数据集 action_type 字段严格对齐）
# ══════════════════════════════════════════════════════════════════════

ACTION_TYPES: List[str] = [
    "DeleteBond",    # 0
    "ChangeBond",    # 1
    "AttachGroup",   # 2
    "ChangeAtom",    # 3
    # "AddAtom",     # 暂不启用
    # "AddRing",     # 暂不启用
    "Terminate",     # 4
]

ACTION_TO_ID   : Dict[str, int] = {a: i for i, a in enumerate(ACTION_TYPES)}
ID_TO_ACTION   : Dict[int, str] = {i: a for i, a in enumerate(ACTION_TYPES)}
NUM_ACTIONS    : int = len(ACTION_TYPES)
PAD_ACTION_ID  : int = NUM_ACTIONS           # padding 用，不与任何真实动作冲突
STOP_ACTION_ID : int = ACTION_TO_ID["Terminate"]

# Loss 任务名（顺序与 UncertaintyWeighting.log_sigma 索引对齐）
TASK_NAMES: List[str] = ["action", "src", "tgt", "label"]


# ══════════════════════════════════════════════════════════════════════
#  PathConfig
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """
    路径配置
    只需传入 project_name，其余路径全部自动推导。
    __post_init__ 会自动创建所有 *_DIR 目录。
    """
    project_name: str = "pretrain_20260420_fp"

    # ── 根目录 ────────────────────────────────────────────────────────
    ROOT_DIR: Path = field(default_factory=lambda: Path("."))

    # ── 数据集目录（固定，与 project 无关）───────────────────────────
    PRETRAIN_DATA_DIR: Path = field(
        default_factory=lambda: Path("dataset/uspto50k/pretrained/")
    )
    RL_DATA_DIR: Path = field(
        default_factory=lambda: Path("dataset/uspto50k/processed/")
    )

    # ── 分词器目录（固定）────────────────────────────────────────────
    TOKENIZER_DIR: Path = field(
        default_factory=lambda: Path("tokenizer/")
    )

    # ── 以下字段由 __post_init__ 自动填充（init=False）───────────────
    # 预训练数据文件
    PRETRAIN_TRAIN_DATA_FILE: Path = field(init=False)
    PRETRAIN_VAL_DATA_FILE  : Path = field(init=False)
    PRETRAIN_TEST_DATA_FILE : Path = field(init=False)

    # RL 数据文件
    RL_TRAIN_DATA_FILE: Path = field(init=False)
    RL_VAL_DATA_FILE  : Path = field(init=False)
    RL_TEST_DATA_FILE : Path = field(init=False)

    # 分词器文件
    VOCAB_FILE: Path = field(init=False)

    # 检查点目录（按 project_name 区分）
    CKPT_DIR            : Path = field(init=False)
    LOG_DIR             : Path = field(init=False)
    TB_DIR              : Path = field(init=False)
    CKPT_BEST_MODEL_FILE: Path = field(init=False)
    CKPT_LAST_MODEL_FILE: Path = field(init=False)

    def __post_init__(self):
        # ── 数据文件 ─────────────────────────────────────────────────
        self.PRETRAIN_TRAIN_DATA_FILE = (
            self.PRETRAIN_DATA_DIR / "uspto50k_train_output.json"
        )
        self.PRETRAIN_VAL_DATA_FILE = (
            self.PRETRAIN_DATA_DIR / "uspto50k_valid_output.json"
        )
        self.PRETRAIN_TEST_DATA_FILE = (
            self.PRETRAIN_DATA_DIR / "uspto50k_test_output.json"
        )
        self.RL_TRAIN_DATA_FILE = self.RL_DATA_DIR / "uspto50k_train_output.json"
        self.RL_VAL_DATA_FILE   = self.RL_DATA_DIR / "uspto50k_valid_output.json"
        self.RL_TEST_DATA_FILE  = self.RL_DATA_DIR / "uspto50k_test_output.json"

        self.VOCAB_FILE = self.TOKENIZER_DIR / "vocab.txt"

        # ── 检查点路径（按 project_name 隔离）───────────────────────
        self.CKPT_DIR             = Path("ckpt") / self.project_name
        self.LOG_DIR              = self.CKPT_DIR / "log"
        self.TB_DIR               = self.CKPT_DIR / "tensorboard"
        self.CKPT_BEST_MODEL_FILE = self.CKPT_DIR / "best_model.pt"
        self.CKPT_LAST_MODEL_FILE = self.CKPT_DIR / "actor_last.pt"

        # ── 自动创建所有 *_DIR 目录 ──────────────────────────────────
        for attr_name, attr_val in vars(self).items():
            if isinstance(attr_val, Path) and attr_name.endswith("_DIR"):
                attr_val.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  ModelConfig
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    网络结构超参
    所有维度、层数、编码器类型集中在此，防止各模块硬编码。
    """
    # ── 编码器类型（消融实验切换点）──────────────────────────────────
    encoder_type : str = "fingerprint"    # "gat" | "fingerprint"

    # ── 节点 / 图特征维度 ─────────────────────────────────────────────
    node_in_dim  : int = field(default_factory=get_atom_feat_dim)
    node_dim     : int = 256             # GraphEncoder 内部 & 输出维度
    hidden_dim   : int = 256             # StateTracker / 预测头统一维度

    # ── GAT 超参 ──────────────────────────────────────────────────────
    gat_layers   : int = 4
    gat_heads    : int = 4

    # ── 指纹编码器超参（encoder_type="fingerprint" 时生效）──────────
    fp_dim       : int = 2048            # Morgan 指纹维度

    # ── 动作 / 原子 / 序列 ────────────────────────────────────────────
    num_actions  : int = NUM_ACTIONS     # 与模块级常量保持同步
    pad_action_id: int = PAD_ACTION_ID
    stop_action_id:int = STOP_ACTION_ID
    max_atoms    : int = 200             # 分子最大原子数（影响 Pointer Embedding）
    pad_atom_id  : int = 200             # = max_atoms，作为原子 idx padding
    max_seq_len  : int = 32              # label 序列最大长度
    max_hist_len : int = 20              # 历史动作最大步数
    max_pos_enc  : int = 64              # LabelDecoder 位置编码最大长度

    # ── 特殊 Token ────────────────────────────────────────────────────
    pad_token_id : int = 0
    bos_token_id : int = 1
    eos_token_id : int = 2


# ══════════════════════════════════════════════════════════════════════
#  StageConfig & TrainConfig
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StageConfig:
    """
    单个课程学习阶段配置

    active_tasks : 参与梯度计算的任务名列表，空列表 = 全部激活。
                   由 LossStrategy.set_active_tasks() 统一消费，
                   pretrain.py 不再单独维护任务激活状态。
    freeze       : 冻结的模块名列表（对应 ActorNetwork 的属性名）。
    lr_scale     : 相对于 TrainConfig.lr 的缩放系数。
    """
    name        : str
    epochs      : int
    active_tasks: List[str]   # 空列表 = 全部激活
    freeze      : List[str]   # 空列表 = 不冻结任何模块
    lr_scale    : float = 1.0


@dataclass
class TrainConfig:
    """
    训练策略超参
    包含：基础超参、Loss 策略、梯度控制、课程学习阶段。
    """
    # ── 基础超参 ──────────────────────────────────────────────────────
    batch_size   : int   = 64
    lr           : float = 3e-4
    weight_decay : float = 1e-2
    grad_clip    : float = 1.0           # 梯度裁剪阈值（clip_grad_norm_）
    warmup_ratio : float = 0.1           # warmup 占总步数比例

    # ── Loss 策略（消融实验切换点）───────────────────────────────────
    # "uncertainty" | "equal" | "manual" | "single_task"
    loss_strategy: str        = "uncertainty"

    # manual 策略：各任务权重（顺序与 TASK_NAMES 对齐）
    loss_weights : List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )

    # single_task 策略：指定唯一激活的任务名
    single_task  : str = "action"

    # UncertaintyWeighting clamp 范围
    uw_s_min     : float = -2.5
    uw_s_max     : float =  2.5

    # ── 课程学习阶段（顺序执行）──────────────────────────────────────
    # 默认：直接联合训练（不分阶段）
    stages: List[StageConfig] = field(default_factory=lambda: [
        StageConfig(
            name         = "Joint-All",
            epochs       = 500,
            active_tasks = [],            # 空 = 全部激活
            freeze       = [],
            lr_scale     = 1.0,
        )
    ])

    # ── 验证 & 保存策略 ───────────────────────────────────────────────
    val_every_epoch : int = 1            # 每隔几个 epoch 验证一次
    save_best_metric: str = "total"      # 以哪个 loss 判断最优模型


# ══════════════════════════════════════════════════════════════════════
#  LoRAConfig（Phase 2 预留）
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    """LoRA 注入配置（Phase 2 RL 微调时使用）"""
    target_modules: List[str] = field(default_factory=lambda: [
        "state_proj",
        "state_tracker",
        "action_predictor",
        "pointer_network",
        "label_decoder",
        # "encoder" 不注入 → 保持冻结
    ])
    r      : int   = 8
    alpha  : float = 16.0
    dropout: float = 0.05


# ══════════════════════════════════════════════════════════════════════
#  RLConfig（Phase 2 预留）
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RLConfig:
    """强化学习超参（Phase 2 PPO 训练时使用）"""
    lr            : float = 1e-4
    weight_decay  : float = 1e-2
    clip_eps      : float = 0.2
    ppo_epochs    : int   = 4
    batch_size    : int   = 64
    kl_beta       : float = 0.1          # KL 惩罚系数
    vf_coef       : float = 0.5
    entropy_coef  : float = 0.01
    gamma         : float = 0.99
    gae_lambda    : float = 0.95
    rollout_steps : int   = 2048
    max_epochs    : int   = 100


# ══════════════════════════════════════════════════════════════════════
#  全局单例（外部直接 import 使用）
# ══════════════════════════════════════════════════════════════════════

PATH_CFG  = PathConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
LORA_CFG  = LoRAConfig()
RL_CFG    = RLConfig()