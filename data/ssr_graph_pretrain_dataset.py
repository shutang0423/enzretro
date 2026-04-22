"""
ssr_graph_pretrain_dataset.py —— 预训练图数据集

修改说明（相对原始版本）：
  1. 补全历史序列的完整 4 字段：
       history_actions    [T]
       history_src_idxs   [T]    ← 新增
       history_tgt_idxs   [T]    ← 新增
       history_label_seqs [T, L] ← 新增
  2. 修正 target_label / history_actions 多余的一维（去掉外层 []）
  3. smiles_to_graph 统一输出 NODE_FEATURE_DIM=79 维，不再手动 pad 到 128
  4. 覆写 StepData.__inc__ 防止原子索引被 PyG DataLoader 自动 offset
  5. 覆写 StepData.__cat_dim__ 确保 [T,L] 张量在正确维度拼接

数据流：
  JSON record
    product_smi  → smiles_to_graph() → x [N,79], edge_index [2,E]
    history[]    → _build_history()  → history_actions    [T]
                                        history_src_idxs   [T]
                                        history_tgt_idxs   [T]
                                        history_label_seqs [T, L]
    target_*     → _build_target()   → target_action [1]
                                        target_src    [1]
                                        target_tgt    [1]
                                        target_label  [L]
    → StepData（PyG Data 子类）
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

from config.config import MODEL_CFG, PAD_ACTION_ID
from utils.chem import get_atom_feat_dim, get_edge_feat_dim, smiles_to_graph, smiles_to_fingerprint

# ══════════════════════════════════════════════════════════════════════
#  历史序列构建（完整 4 字段）
# ══════════════════════════════════════════════════════════════════════

def _encode_label(label_str: str, tokenizer, max_len: int) -> torch.Tensor:
    """
    将 label 字符串编码为定长 token id 序列
    兼容原始代码的中括号补全逻辑
    返回: [max_len] long
    """
    # 兼容性补丁：补全中括号
    if label_str in {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "CW", "CCW"}:
        label_str = f"[{label_str}]"

    token_ids = tokenizer.encode_with_special(label_str, add_bos=True, add_eos=True)
    token_ids = token_ids[:max_len]

    result = torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long)
    result[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return result


def _build_history(
    history_records : list,
    tokenizer,
    max_hist_len    : int,
    max_label_len   : int,
) -> dict:
    """
    将 history 列表转为四个 padding 后的张量

    Args:
      history_records : JSON 中的 history 列表
                        每项: {action_type, src_idx, tgt_idx, label}
      tokenizer       : LabelTokenizer
      max_hist_len    : 历史最大步数 T
      max_label_len   : label 序列最大长度 L

    Returns（形状均已 padding 到固定长度）:
      history_actions    : [T]    long，padding = PAD_ACTION_ID
      history_src_idxs   : [T]    long，无效步 = pad_atom_id
      history_tgt_idxs   : [T]    long，无效步 = pad_atom_id
      history_label_seqs : [T, L] long，padding = pad_token_id
    """
    pad_atom = MODEL_CFG.pad_atom_id

    T = min(len(history_records), max_hist_len)

    # Add batch dimension to ensure proper batching
    actions    = torch.full((1, max_hist_len),               PAD_ACTION_ID,          dtype=torch.long)  # [1, T]
    src_idxs   = torch.full((1, max_hist_len),               pad_atom,               dtype=torch.long)  # [1, T]
    tgt_idxs   = torch.full((1, max_hist_len),               pad_atom,               dtype=torch.long)  # [1, T]
    label_seqs = torch.full((1, max_hist_len, max_label_len), tokenizer.pad_token_id, dtype=torch.long)  # [1, T, L]

    for i, rec in enumerate(history_records[:T]):
        # action_type（int）
        actions[0, i] = int(rec.get("action_type", PAD_ACTION_ID))

        # src_idx：None 或 -1 → pad_atom
        src = rec.get("src_idx", None)
        src_idxs[0, i] = src if (src is not None and src >= 0) else pad_atom

        # tgt_idx：None 或 -1 → pad_atom
        tgt = rec.get("tgt_idx", None)
        tgt_idxs[0, i] = tgt if (tgt is not None and tgt >= 0) else pad_atom

        # label 编码
        label_str = rec.get("label", None) or "NONE"
        label_seqs[0, i] = _encode_label(label_str, tokenizer, max_label_len)

    return {
        "history_actions"   : actions,      # [T]
        "history_src_idxs"  : src_idxs,     # [T]
        "history_tgt_idxs"  : tgt_idxs,     # [T]
        "history_label_seqs": label_seqs,   # [T, L]
    }


# ══════════════════════════════════════════════════════════════════════
#  目标值构建
# ══════════════════════════════════════════════════════════════════════

def _build_target(record: dict, tokenizer, max_label_len: int) -> dict:
    """
    构建单步预测目标

    Returns:
      target_action : [1]  long
      target_src    : [1]  long（无效 = -1，CrossEntropyLoss ignore_index=-1）
      target_tgt    : [1]  long（无效 = -1）
      target_label  : [L]  long（含 BOS/EOS，padding 到 max_label_len）
    """
    # action_type
    act = record.get("target_action_type", 0)
    target_action = torch.tensor([int(act) if act is not None else 0], dtype=torch.long)

    # src / tgt（None 或 -1 → -1）
    src = record.get("target_src_idx", None)
    tgt = record.get("target_tgt_idx", None)
    target_src = torch.tensor([src if (src is not None and src >= 0) else -1], dtype=torch.long)
    target_tgt = torch.tensor([tgt if (tgt is not None and tgt >= 0) else -1], dtype=torch.long)

    # label
    label_str    = record.get("target_label", None) or "NONE"
    target_label = _encode_label(label_str, tokenizer, max_label_len)  # [L]

    return {
        "target_action": target_action,
        "target_src"   : target_src,
        "target_tgt"   : target_tgt,
        "target_label" : target_label,
    }


# ══════════════════════════════════════════════════════════════════════
#  StepData：覆写 __inc__ / __cat_dim__ 防止 PyG 错误 offset
# ══════════════════════════════════════════════════════════════════════

class StepData(Data):
    """
    单步 MDP 样本的 PyG Data 子类

    覆写原因：
      PyG DataLoader 默认对名称含 "index" 的属性自动 += num_nodes。
      history_src_idxs / history_tgt_idxs / target_src / target_tgt
      是原子索引，不应被 offset，必须覆写 __inc__ 返回 0。

      history_label_seqs 是 [T, L] 的 2D 张量，
      需覆写 __cat_dim__ 确保在 dim=0（batch 维度）正确堆叠。
    """

    # 不做 offset 的属性集合
    _NO_OFFSET_KEYS = frozenset({
        "history_src_idxs",
        "history_tgt_idxs",
        "target_src",
        "target_tgt",
    })

    # 在 dim=0 拼接的自定义属性
    _CAT_DIM0_KEYS = frozenset({
        "history_actions",
        "history_src_idxs",
        "history_tgt_idxs",
        "history_label_seqs",
        "target_action",
        "target_src",
        "target_tgt",
        "target_label",
    })

    def __inc__(self, key: str, value, *args, **kwargs) -> int:
        if key in self._NO_OFFSET_KEYS:
            return 0   # 原子索引：不 offset
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs) -> int:
        if key in self._CAT_DIM0_KEYS:
            return 0   # 所有自定义属性在 batch 维度（dim=0）拼接
        return super().__cat_dim__(key, value, *args, **kwargs)


# ══════════════════════════════════════════════════════════════════════
#  SSRGraphDataset
# ══════════════════════════════════════════════════════════════════════

class SSRGraphDataset(torch.utils.data.Dataset):
    """
    单步 MDP 预训练数据集

    每条样本对应一个展开后的单步决策：
      输入：product graph + history（已执行的编辑序列，完整 4 字段）
      输出：下一步的 (action_type, src_idx, tgt_idx, label)

    Args:
      json_path    : 数据文件路径（JSON 列表 或 JSONL）
      tokenizer    : LabelTokenizer 实例
      max_seq_len  : label 序列最大长度（默认从 MODEL_CFG 读取）
      max_hist_len : 历史最大步数（默认从 MODEL_CFG 读取）
    """

    def __init__(
        self,
        json_path    : str,
        tokenizer,
        max_seq_len  : int = None,
        max_hist_len : int = None,
    ):
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len  or MODEL_CFG.max_seq_len
        self.max_hist_len = max_hist_len or MODEL_CFG.max_hist_len

        # 支持 JSON 列表 和 JSONL 两种格式
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            self.data = json.loads(content)
        else:
            self.data = [json.loads(l) for l in content.splitlines() if l.strip()]

        print(f"[SSRGraphDataset] loaded {len(self.data)} records ← {json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> StepData:
        """
        返回第 idx 条样本的 StepData

        StepData 字段一览：
          x                  : [N, 79]        float  节点特征
          edge_index         : [2, 2E]         long   边索引（无向双向）
          edge_attr          : [2E, 12]        float  边特征
          history_actions    : [T]             long   历史动作类型
          history_src_idxs   : [T]             long   历史 src 原子索引
          history_tgt_idxs   : [T]             long   历史 tgt 原子索引
          history_label_seqs : [T, L]          long   历史 label token 序列
          target_action      : [1]             long   目标动作类型
          target_src         : [1]             long   目标 src（-1=无效）
          target_tgt         : [1]             long   目标 tgt（-1=无效）
          target_label       : [L]             long   目标 label token 序列
        """
        item = self.data[idx]

        # ── 1. 分子图 ─────────────────────────────────────────────────
        x, edge_index, edge_attr = smiles_to_graph(item["product_smi"])
        if x is None:
            # SMILES 解析失败：单节点空图，不中断训练
            x          = torch.zeros(1, get_atom_feat_dim())
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, get_edge_feat_dim()))

        # # ── Morgan 指纹（始终计算，两种 encoder 模式均可用）──────────
        # fingerprint = smiles_to_fingerprint(item["product_smi"], fp_dim=MODEL_CFG.fp_dim)
        # fingerprint = fingerprint.unsqueeze(0)   # [1, fp_dim]

        # ── 2. 历史动作序列（完整 4 字段）────────────────────────────
        hist = _build_history(
            history_records = item.get("history", []),
            tokenizer       = self.tokenizer,
            max_hist_len    = self.max_hist_len,
            max_label_len   = self.max_seq_len,
        )

        # ── 3. 训练目标 ───────────────────────────────────────────────
        tgt = _build_target(
            record        = item,
            tokenizer     = self.tokenizer,
            max_label_len = self.max_seq_len,
        )

        # ── 4. 组装 StepData ──────────────────────────────────────────
        return StepData(
            # 图结构
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            # fingerprint = fingerprint,
            # 历史（完整 4 字段）
            history_actions    = hist["history_actions"],     # [T]
            history_src_idxs   = hist["history_src_idxs"],   # [T]
            history_tgt_idxs   = hist["history_tgt_idxs"],   # [T]
            history_label_seqs = hist["history_label_seqs"], # [T, L]
            # 目标
            target_action = tgt["target_action"],   # [1]
            target_src    = tgt["target_src"],       # [1]
            target_tgt    = tgt["target_tgt"],       # [1]
            target_label  = tgt["target_label"],     # [L]
        )