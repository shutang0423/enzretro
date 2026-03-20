import torch
import json 
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def smiles_to_graph(smiles: str) -> tuple:
    """将 SMILES 转换为 PyG 的 x 和 edge_index"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
        
    # 1. 提取节点特征 (这里仅用原子序数和度数作为示例，实际可扩充至 128 维)
    x = []
    for atom in mol.GetAtoms():
        feature = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetIsAromatic())
        ]
        # 补齐到 128 维 (对应 GraphEncoder 的 node_in_dim=128)
        feature += [0] * (128 - len(feature))
        x.append(feature)
        
    # 2. 提取边索引
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 无向图，双向添加
        edge_index.append([i, j])
        edge_index.append([j, i])
        
    x_tensor = torch.tensor(x, dtype=torch.float)
    if len(edge_index) > 0:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        
    return x_tensor, edge_index_tensor


class SSRGraphDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, tokenizer, max_seq_len: int = 20, max_hist_len: int = 10):
        """
        Args:
            json_path: 展开后的单步训练数据 (steps.json)
            tokenizer: 实例化的 LabelTokenizer 对象
            max_seq_len: Label 序列最大长度
            max_hist_len: 历史动作序列最大长度
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_hist_len = max_hist_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ==========================================
        # 1. 图结构特征 (复用之前的 smiles_to_graph 函数)
        # ==========================================
        x, edge_index = smiles_to_graph(item['product_smi'])
        
        # ==========================================
        # 2. 历史编辑状态 (History)
        # ==========================================
        history_actions = [h['action_type'] for h in item['history']]
        history_actions = history_actions[-self.max_hist_len:] 
        
        # 假设动作类型 7 为 PAD (对应没有历史动作的位置)
        history_actions += [7] * (self.max_hist_len - len(history_actions))
        
        # ==========================================
        # 3. Label 序列处理 (使用你的 LabelTokenizer)
        # ==========================================
        raw_label = item['target_label']
        
        # 【兼容性补丁】如果你没有修改 tokenizer.py 中的中括号，取消下面两行的注释：
        if raw_label in ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "CW", "CCW"]:
            raw_label = f"[{raw_label}]"
            
        # 调用 tokenizer 编码，自动添加 [BOS] 和 [EOS]
        token_ids = self.tokenizer.encode_with_special(
            raw_label, 
            add_bos=True, 
            add_eos=True
        )
        
        # 截断到 max_seq_len
        token_ids = token_ids[:self.max_seq_len]
        
        # 使用 tokenizer 的 pad_token_id 进行 Padding
        pad_id = self.tokenizer.pad_token_id
        token_ids += [pad_id] * (self.max_seq_len - len(token_ids))
        
        # ==========================================
        # 4. 打包为 PyG Data 对象
        # ==========================================
        data = Data(x=x, edge_index=edge_index)
        
        # 统一转为 Tensor 并增加一维，方便 PyG DataLoader 自动拼接 Batch

        # 处理可能为 None 的 target_src_idx 和 target_tgt_idx
        # 如果为 None，则赋值为 -1 (配合 CrossEntropyLoss 的 ignore_index=-1)
        src_idx = item['target_src_idx']
        tgt_idx = item['target_tgt_idx']
        
        src_idx = -1 if src_idx is None else src_idx
        tgt_idx = -1 if tgt_idx is None else tgt_idx
        
        # 统一转为 Tensor 并增加一维，方便 PyG DataLoader 自动拼接 Batch
        data.target_action = torch.tensor([item['target_action_type']], dtype=torch.long)
        data.target_src = torch.tensor([src_idx], dtype=torch.long)
        data.target_tgt = torch.tensor([tgt_idx], dtype=torch.long)
        data.target_label = torch.tensor([token_ids], dtype=torch.long)
        data.history_actions = torch.tensor([history_actions], dtype=torch.long)

        # data.target_action = torch.tensor([item['target_action_type']], dtype=torch.long)
        # data.target_src = torch.tensor([item['target_src_idx']], dtype=torch.long)
        # data.target_tgt = torch.tensor([item['target_tgt_idx']], dtype=torch.long)
        # data.target_label = torch.tensor([token_ids], dtype=torch.long)
        # data.history_actions = torch.tensor([history_actions], dtype=torch.long)
        
        return data




