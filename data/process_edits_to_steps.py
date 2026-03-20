"""
dataset_builder.py
将 SSREditsExtractor 输出的 JSON 文件展开为预训练样本 JSONL

输入 JSON 格式（SSREditsExtractor 输出）:
  单条反应: {...}  或  列表: [{...}, {...}, ...]

输出 JSONL 格式（每行一条样本）:
  {
    "rxn_id":       str,
    "step":         int,
    "product_smi":  str,
    "history":      [...],        # 已执行的编辑列表
    "target_type":  int,          # 预测器1 目标 (0-6)
    "cond_type":    int | null,   # 预测器2/3 的 GT 条件 (Teacher Forcing)
    "target_src":   int | null,   # 预测器2 目标
    "target_tgt":   int | null,   # 预测器2 目标
    "target_label": str | null,   # 预测器3 目标
  }
"""

import json
from pathlib import Path

# action_type 字符串 → 整数映射
ACTION_TYPE_MAP = {
    "DeleteBond":  0,
    "ChangeBond":  1,
    "AddBond":     2,
    "AttachGroup": 3,
    "LeaveGroup":  4,
    "ChangeAtom":  5,
    "Terminate":   6,
}

TERMINATE_INT = ACTION_TYPE_MAP["Terminate"]

# def expand_reaction(reaction: dict) -> list[dict]:
def expand_reaction(reaction):
    """
    将单条反应展开为多条独立训练样本。
    
    每步样本格式：
      - history     : 当前步之前的所有编辑（Teacher Forcing 的上文）
      - target_type : 当前步的 action_type 整数（预测器1 目标）
      - cond_type   : 同 target_type，供预测器2/3 作为 GT 条件；Terminate 时为 null
      - target_src  : 当前步的 src_idx；Terminate 时为 null
      - target_tgt  : 当前步的 tgt_idx；Terminate 时为 null
      - target_label: 当前步的 label；Terminate 时为 null
    """
    rxn_id      = reaction.get("rxn_id", "unknown")
    product_smi = reaction["input"]["product_smi"]
    edits       = reaction["output"]["edits"]

    samples = []
    history = []

    for step, edit in enumerate(edits):
        action_type_int = ACTION_TYPE_MAP[edit["action_type"]]
        is_terminate    = (action_type_int == TERMINATE_INT)

        sample = {
            "rxn_id":       rxn_id,
            "step":         step,
            "product_smi":  product_smi,
            "history":      list(history),          # 当前步之前的历史（浅拷贝）
            "target_action_type":  action_type_int,        # 预测器1
            "cond_type":    None if is_terminate else action_type_int,  # 预测器2/3 GT 条件
            "target_src_idx":   None if is_terminate else edit["src_idx"],  # 预测器2
            "target_tgt_idx":   None if is_terminate else edit["tgt_idx"],  # 预测器2
            "target_label": None if is_terminate else edit["label"],    # 预测器3
        }
        samples.append(sample)

        # 将当前编辑加入历史，供下一步使用
        history.append({
            "action_type": action_type_int,
            "src_idx":     edit["src_idx"],
            "tgt_idx":     edit["tgt_idx"],
            "label":       edit["label"],
        })

    return samples


# def load_json(input_path: str) -> list[dict]:
def load_json(input_path):
    """读取 JSON 文件，兼容单条 dict 和列表 list[dict] 两种格式。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


# def process_edits_to_steps(input_path: str, output_path: str) -> None:
def process_edits_to_steps(input_path, output_path):
    """
    读取 JSON 文件，展开为预训练样本，写入 JSONL 文件。

    Args:
        input_path : SSREditsExtractor 输出的 JSON 文件路径
        output_path: 输出的 JSONL 文件路径
    """
    reactions = load_json(input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    pretrain_samples = []
    for reaction in reactions:
        sample = expand_reaction(reaction)
        pretrain_samples.extend(sample)
        total_samples += len(sample)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pretrain_samples, f, indent=2, ensure_ascii=False)

    print(f"输入反应数 : {len(reactions)}")
    print(f"输出样本数 : {total_samples}")
    print(f"已保存至   : {output_path}")


# ==================== 验证 ====================

if __name__ == "__main__":

    datas = ['train', 'valid', 'test']
    for d in datas:
        input_json = f'dataset/uspto50k/processed/uspto50k_{d}_output.json'
        output_json = f'dataset/uspto50k/pretrained/uspto50k_{d}_output.json'
        process_edits_to_steps(input_json,output_json)

    
    all_output_json = 'dataset/uspto50k/pretrained/uspto50k_train_valid_test_output.json'
