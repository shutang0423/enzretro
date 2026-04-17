"""USPTO50K 数据集加载器

职责:
  1. 加载 JSON 数据
  2. SMILES → PyG 图
  3. action_type 字符串 → 整数 ID
  4. 构建 Teacher Forcing 所需的 decoder_input_seq
  5. 处理 src_idx/tgt_idx = -1 的 Terminate 动作
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple, Optional

from utils.chem import smiles_to_pyg
from config.config import ACTION_TO_ID, TERMINATE_ACTION_ID

# ── 特殊索引常量 ───────────────────────────────────────────────
INVALID_IDX = 0   # Terminate 时 src/tgt=-1 → 映射为 0 (不参与 loss)


class USPTO50KDataset(Dataset):
    """USPTO50K 逆合成数据集

    每条样本返回:
        graph_data    : PyG Data (产物分子图)
        edit_steps    : List[Dict]  每步编辑的 (action_id, src, tgt, label_ids)
        num_edits     : int
    """

    def __init__(self, json_path: str, tokenizer,
                 max_edits: int = 10, max_label_len: int = 64):
        with open(json_path) as f:
            raw = json.load(f)

        self.tokenizer = tokenizer
        self.max_edits = max_edits
        self.max_label_len = max_label_len
        self.samples = []
        self._skipped = 0

        for item in raw:
            processed = self._process(item)
            if processed is not None:
                self.samples.append(processed)
            else:
                self._skipped += 1

        print(f"Loaded {len(self.samples)} samples, skipped {self._skipped}")

    def _process(self, item: Dict) -> Optional[Dict]:
        product_smi = item["input"]["product_smi"]
        graph = smiles_to_pyg(product_smi)
        if graph is None:
            return None

        edits = item["output"]["edits"]
        if len(edits) > self.max_edits:
            return None

        processed_edits = []
        for edit in edits:
            action_id = ACTION_TO_ID.get(edit["action_type"])
            if action_id is None:
                return None  # 未知动作类型

            # src/tgt: -1 (Terminate) → INVALID_IDX，并标记为 ignore
            src = edit["src_idx"]
            tgt = edit["tgt_idx"]
            src_valid = src >= 0
            tgt_valid = tgt >= 0
            src = max(src, 0)   # clamp -1 → 0
            tgt = max(tgt, 0)

            # label tokenize
            label_str = edit["label"]
            label_ids = self.tokenizer.encode(label_str)  # List[int]
            # 截断 + padding
            label_ids = label_ids[:self.max_label_len]

            processed_edits.append({
                "action_id":  action_id,
                "src_idx":    src,
                "tgt_idx":    tgt,
                "src_valid":  src_valid,   # False → 不计算 pointer loss
                "tgt_valid":  tgt_valid,
                "label_ids":  label_ids,
            })

        return {
            "rxn_id":      item["rxn_id"],
            "graph":       graph,
            "edits":       processed_edits,
            "num_edits":   len(processed_edits),
            "product_smi": product_smi,
            "reactant_smi": item["output"]["reactant_smi"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Collate ────────────────────────────────────────────────────
class PretrainCollateFn:
    def __init__(self, tokenizer, max_label_len: int = 64):
        self.tokenizer     = tokenizer
        self.max_label_len = max_label_len

    def __call__(self, batch):
        return collate_pretrain(batch, self.tokenizer, self.max_label_len)
    
def collate_pretrain(batch: List[Dict], tokenizer, max_label_len: int = 64):
    """预训练 collate_fn

    把 batch 内所有样本的所有步骤展平为独立训练数据。
    新增返回 step_to_sample_tensor，供 actor.forward 做 index select。
    """
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    graphs        = []
    actions       = []
    srcs          = []
    tgts          = []
    dec_inputs    = []
    dec_targets   = []
    src_valids    = []
    tgt_valids    = []
    step_to_sample = []          # ← 每步对应的图索引

    for sample_idx, sample in enumerate(batch):
        graphs.append(sample["graph"])

        for edit in sample["edits"]:
            actions.append(edit["action_id"])
            srcs.append(edit["src_idx"])
            tgts.append(edit["tgt_idx"])
            src_valids.append(edit["src_valid"])
            tgt_valids.append(edit["tgt_valid"])
            step_to_sample.append(sample_idx)   # ← 记录归属

            # Teacher Forcing 序列构建
            lids    = edit["label_ids"]
            L       = max_label_len + 1
            inp     = ([bos_id] + lids)[:L]
            tgt_seq = (lids + [eos_id])[:L]
            inp     = inp     + [pad_id] * max(0, L - len(inp))
            tgt_seq = tgt_seq + [pad_id] * max(0, L - len(tgt_seq))
            dec_inputs.append(inp)
            dec_targets.append(tgt_seq)

    pyg_batch = Batch.from_data_list(graphs)

    return {
        "pyg_batch":            pyg_batch,
        "target_actions":       torch.tensor(actions,     dtype=torch.long),
        "target_srcs":          torch.tensor(srcs,        dtype=torch.long),
        "target_tgts":          torch.tensor(tgts,        dtype=torch.long),
        "decoder_inputs":       torch.tensor(dec_inputs,  dtype=torch.long),
        "decoder_targets":      torch.tensor(dec_targets, dtype=torch.long),
        "src_valid_mask":       torch.tensor(src_valids,  dtype=torch.bool),
        "tgt_valid_mask":       torch.tensor(tgt_valids,  dtype=torch.bool),
        "step_to_sample":       torch.tensor(step_to_sample, dtype=torch.long),  # ← 新增
    }


def collate_rl(batch: List[Dict]):
    """RL rollout 的 collate_fn: 返回整条轨迹"""
    graphs = Batch.from_data_list([s["graph"] for s in batch])
    return {
        "pyg_batch": graphs,
        "samples":   batch,   # 保留完整样本供 env 使用
    }


def build_dataloader(json_path: str, tokenizer,
                     batch_size: int = 32, shuffle: bool = True,
                     mode: str = "pretrain",
                     max_label_len: int = 64, num_workers: int = 4) -> DataLoader:
    dataset = USPTO50KDataset(json_path, tokenizer)
    # fn = (lambda b: collate_pretrain(b, tokenizer)) if mode == "pretrain" else collate_rl
    if mode == "pretrain":
        collate_fn = PretrainCollateFn(tokenizer, max_label_len)  
    else:
        collate_fn = collate_rl
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=4, pin_memory=True)