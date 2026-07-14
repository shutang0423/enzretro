import torch
import json
from pathlib import Path
from torch_geometric.loader import DataLoader

from config.config import PATH_CFG, MODEL_CFG, ID_TO_ACTION, STOP_ACTION_ID
from model.actor_network import ActorNetwork
from utils.chem import smiles_to_graph
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inspect_batch(batch, tokenizer: LabelTokenizer, verbose: bool = True):
    """
    详细查看 batch 中的数据，特别是 target 信息
    
    Args:
        batch: PyG Data batch
        tokenizer: 用于解码 label tokens
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含所有关键信息的字典
    """
    B = int(batch.num_graphs)
    max_seq = MODEL_CFG.max_seq_len
    max_hist = MODEL_CFG.max_hist_len
    
    # ── 辅助函数：安全 reshape ─────────────────────────────────────
    def _reshape(t: torch.Tensor, *shape):
        if t.shape == torch.Size(shape):
            return t
        return t.reshape(*shape)
    
    # ── 解包 target 数据 ───────────────────────────────────────────
    target_action = _reshape(batch.target_action, B)
    target_src    = _reshape(batch.target_src, B)
    target_tgt    = _reshape(batch.target_tgt, B)
    target_label  = _reshape(batch.target_label, B, max_seq)
    
    # ── 解包 history 数据 ──────────────────────────────────────────
    h_actions    = _reshape(batch.history_actions, B, max_hist)
    h_src_idxs   = _reshape(batch.history_src_idxs, B, max_hist)
    h_tgt_idxs   = _reshape(batch.history_tgt_idxs, B, max_hist)
    h_label_seqs = _reshape(batch.history_label_seqs, B, max_hist, max_seq)
    
    # ── 构建信息字典 ───────────────────────────────────────────────
    info = {
        'batch_size': B,
        'num_nodes': batch.x.size(0),
        'num_edges': batch.edge_index.size(1),
        'target': {
            'action': target_action.cpu().numpy(),
            'src': target_src.cpu().numpy(),
            'tgt': target_tgt.cpu().numpy(),
            'label_ids': target_label.cpu().numpy(),
            'label_text': [
                tokenizer.decode(target_label[i].tolist()) 
                for i in range(B)
            ],
        },
        'history': {
            'actions': h_actions.cpu().numpy(),
            'src_idxs': h_src_idxs.cpu().numpy(),
            'tgt_idxs': h_tgt_idxs.cpu().numpy(),
            'label_seqs': h_label_seqs.cpu().numpy(),
        }
    }
    
    # ── 打印详细信息 ───────────────────────────────────────────────
    if verbose:
        print("\n" + "="*70)
        print("  BATCH INSPECTION")
        print("="*70)
        print(f"Batch Size: {B}")
        print(f"Total Nodes: {batch.x.size(0)}")
        print(f"Total Edges: {batch.edge_index.size(1)}")
        
        for i in range(B):
            print(f"\n--- Sample {i} ---")
            
            # Target 信息
            print(f"  Target Action: {target_action[i].item()} "
                  f"({ID_TO_ACTION[target_action[i].item()]})")
            print(f"  Target Src: {target_src[i].item()}")
            print(f"  Target Tgt: {target_tgt[i].item()}")
            print(f"  Target Label: '{info['target']['label_text'][i]}'")
            print(f"  Target Label IDs: {target_label[i].tolist()[:10]}...")  # 只显示前10个
            
            # History 信息（只显示非 padding 的步骤）
            print(f"\n  History (non-padding steps):")
            for t in range(max_hist):
                action = h_actions[i, t].item()
                if action == MODEL_CFG.pad_action_id:
                    break
                src = h_src_idxs[i, t].item()
                tgt = h_tgt_idxs[i, t].item()
                label_ids = h_label_seqs[i, t].tolist()
                label_text = tokenizer.decode(label_ids)
                
                print(f"    Step {t}: Action={action} ({ID_TO_ACTION[action]}),  "
                      f"Src={src}, Tgt={tgt}, Label='{label_text}'")
        
        print("\n" + "="*70)
    
    return info

def get_batch_by_index(loader: DataLoader, index: int):
    """
    从 DataLoader 中获取指定索引的 batch
    
    Args:
        loader: PyG DataLoader
        index: batch 索引 (0-based)
    
    Returns:
        batch: 指定索引的 batch，如果索引超出范围返回 None
    """
    for i, batch in enumerate(loader):
        if i == index:
            return batch
    return None


checkpoint_path = Path("ckpt/pretrain_20260426_gin_uncertainty/best_model.pt")

# 1. 加载分词器
tokenizer = LabelTokenizer(str(PATH_CFG.VOCAB_FILE))
vocab_size = tokenizer.vocab_size

# 2. 构建模型
model = ActorNetwork(vocab_size=vocab_size).to(device)

# 3. 加载训练好的模型
# checkpoint_path = PATH_CFG.CKPT_BEST_MODEL_FILE
if not checkpoint_path.exists():
    print(f"Error: Checkpoint file not found at {checkpoint_path}")

print(f"Loading checkpoint from {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model'])
model.eval()




# ----

# 4. 加载测试集
test_ds = SSRGraphDataset(str(PATH_CFG.PRETRAIN_TRAIN_DATA_FILE), 
                          tokenizer,
                          max_seq_len  = MODEL_CFG.max_seq_len,
                          max_hist_len = MODEL_CFG.max_hist_len,)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


## 取一个测试样本
# batch = next(iter(test_loader)).to(device)
batch = get_batch_by_index(test_loader, 110).to(device)

inspect_batch(batch, tokenizer=tokenizer)


# 准备 encoder 参数
encoder_kwargs = dict(
    x          = batch.x,
    edge_index = batch.edge_index,
    batch      = batch.batch,
)

# 生成 Top-5 结果
top_5_results = model.generate_top_n(
    n=5,
    temperature=0.8,
    top_k=10,
    **encoder_kwargs
)

# 打印结果
for rank, (edit_sequence, log_prob) in enumerate(top_5_results, 1):
    print(f"\n--- Rank {rank} (Score: {log_prob:.3f}) ---")
    for step_idx, edit in enumerate(edit_sequence):
        action = edit.action_type.item()

        if action == STOP_ACTION_ID:
            src, tgt, label_str = None, None, None  # Terminate
        else:
            src = edit.src_idx.item()
            tgt = edit.tgt_idx.item()
            label_ids = edit.label_tokens[0].tolist()
            label_str = tokenizer.decode(label_ids, skip_special_tokens=True)
        
        print(f"  Step {step_idx}: "
                f"Action={ID_TO_ACTION[action]}({action}), Src={src}, Tgt={tgt}, Label='{label_str}'")

print("\n" + "="*60)






