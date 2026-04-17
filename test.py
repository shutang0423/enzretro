import torch
from collections import Counter
from data.dataset import USPTO50KDataset

from tokenizer.tokenizer import LabelTokenizer
from config.config import config as cfg
from config.config import ACTION_TO_ID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 tokenizer 和模型结构
tokenizer  = LabelTokenizer(vocab_file=str(cfg.path.VOCAB_FILE))

# 1. Action 分布
ds = USPTO50KDataset(str(cfg.path.RL_TEST_DATA_FILE), tokenizer)
action_counter = Counter()
for sample in ds.samples:
    for edit in sample["edits"]:
        action_counter[edit["action_id"]] += 1

id_to_action = {v: k for k, v in ACTION_TO_ID.items()}
total = sum(action_counter.values())
print("\n=== Action 分布 ===")
for aid, cnt in sorted(action_counter.items()):
    print(f"  [{aid}] {id_to_action.get(aid,'?'):<25} {cnt:>6}  {cnt/total*100:.2f}%")

# 2. 随机 baseline（如果模型只预测多数类，准确率是多少）
most_common_id, most_common_cnt = action_counter.most_common(1)[0]
print(f"\n多数类 baseline: {most_common_cnt/total*100:.2f}%")
print(f"随机 baseline:   {100/7:.2f}%")

# 3. 检查 ACTION_TO_ID
print(f"\n=== ACTION_TO_ID ===")
print(ACTION_TO_ID)

