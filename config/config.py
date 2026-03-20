# config.py
# 统一管理所有超参数，避免各文件中硬编码导致的越界

MODEL_CONFIG = {
    # ── 图编码器 ──────────────────────────────────────
    "node_in_dim":   128,   # 原子特征维度（smiles_to_graph 补齐目标）
    "node_dim":      256,   # GraphEncoder 隐层 / 输出维度
    "hidden_dim":    512,   # Decoder / MLP 隐层维度

    # ── 动作空间 ──────────────────────────────────────
    "num_actions":   7,     # 0-6 共 7 种动作（Terminate = 6）
    "pad_action_id": 7,     # History padding 用的特殊 id（不参与预测）
    #   → SimpleStateTracker Embedding 大小 = num_actions + 1 = 8

    # ── 指针网络 ──────────────────────────────────────
    "max_atoms":     300,   # src_emb Embedding 大小上限（分子最大原子数）

    # ── Label 解码器 ──────────────────────────────────
    "max_label_len": 64,    # Label 序列最大长度（含 BOS/EOS）
    "max_pos_enc":   256,   # LabelDecoder 位置编码上限

    # ── 数据集 ────────────────────────────────────────
    "max_hist_len":  10,    # 历史动作序列最大长度
    "max_seq_len":   64,    # 与 max_label_len 保持一致
}

TRAIN_CONFIG = {
    "batch_size":  32,
    "lr":          1e-4,
    "num_epochs":  10,
}

PATH_CONFIG = {
    "vocab_file":   "tokenizer/vocab.txt",
    "train_data":   "dataset/uspto50k/pretrained/uspto50k_train_output.json",
    "test_data":    "dataset/uspto50k/pretrained/uspto50k_test_output.json",
    "ckpt_path":    "ckpt/best_model.pt",
    "log_dir":      "ckpt/pretrain",
}