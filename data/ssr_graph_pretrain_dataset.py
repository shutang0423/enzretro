import torch
import json 
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ══════════════════════════════════════════════════════════════════
#  特征候选列表
# ══════════════════════════════════════════════════════════════════

PERMITTED_SYMBOLS = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
    'As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co',
    'Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn',
    'Zr','Cr','Pt','Hg','Pb',
]  # 43种，+UNK = 44，one_hot_encoding 会再+1 = 45 ✅

PERMITTED_DEGREES       = [0, 1, 2, 3, 4, 5]          # +UNK = 7
PERMITTED_FORMAL_CHRGS  = [-3, -2, -1, 0, 1, 2, 3]    # +UNK = 8
PERMITTED_NUM_HS        = [0, 1, 2, 3, 4]              # +UNK = 6
PERMITTED_HYBRIDIZATION = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]  # +UNK = 6
PERMITTED_CHIRAL = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]  # +UNK = 5

PERMITTED_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]  # +UNK = 5
PERMITTED_BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]  # +UNK = 5

# 供外部 model/config 使用
NODE_FEATURE_DIM = 79
EDGE_FEATURE_DIM = 12

# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def one_hot_encoding(value, choices: list) -> list:
    """
    One-Hot 编码，不在 choices 中时最后一位(UNK)置1。
    输出长度 = len(choices) + 1
    """
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

# ══════════════════════════════════════════════════════════════════
#  节点特征（79维，无零填充）
# ══════════════════════════════════════════════════════════════════

def atom_features(atom) -> list:
    """
    提取单个原子的 79 维特征，每一维都有实际含义。

    维度分布:
      [0 :45]  原子类型      One-Hot 45维
      [45:52]  度数          One-Hot  7维
      [52:60]  形式电荷      One-Hot  8维
      [60:66]  隐式氢数      One-Hot  6维
      [66:72]  杂化方式      One-Hot  6维
      [72:77]  手性          One-Hot  5维
      [77]     是否芳香      连续     1维
      [78]     是否在环      连续     1维
    """
    feat = (
        one_hot_encoding(atom.GetSymbol(),        PERMITTED_SYMBOLS)      # 45
      + one_hot_encoding(atom.GetDegree(),         PERMITTED_DEGREES)      #  7
      + one_hot_encoding(atom.GetFormalCharge(),   PERMITTED_FORMAL_CHRGS) #  8
      + one_hot_encoding(atom.GetTotalNumHs(),     PERMITTED_NUM_HS)       #  6
      + one_hot_encoding(atom.GetHybridization(),  PERMITTED_HYBRIDIZATION)#  6
      + one_hot_encoding(atom.GetChiralTag(),      PERMITTED_CHIRAL)       #  5
      + [int(atom.GetIsAromatic())]                                        #  1
      + [int(atom.IsInRing())]                                             #  1
    )
    # 45+7+8+6+6+5+1+1 = 79
    # assert len(feat) == NODE_FEATURE_DIM, \
    #     f"节点特征维度错误: 期望 {NODE_FEATURE_DIM}, 实际 {len(feat)}"
    return feat

# ══════════════════════════════════════════════════════════════════
#  边特征（12维）—— 新增！
# ══════════════════════════════════════════════════════════════════

def bond_features(bond) -> list:
    """
    提取单条键的 12 维特征。

    维度分布:
      [0 :5]   键类型        One-Hot  5维  (单/双/三/芳香/UNK)
      [5]      是否共轭      连续     1维
      [6]      是否在环      连续     1维
      [7:12]   立体化学      One-Hot  5维
    """
    feat = (
        one_hot_encoding(bond.GetBondType(),   PERMITTED_BOND_TYPES)   # 5
      + [int(bond.GetIsConjugated())]                                   # 1
      + [int(bond.IsInRing())]                                          # 1
      + one_hot_encoding(bond.GetStereo(),     PERMITTED_BOND_STEREO)  # 5
    )
    # 5+1+1+5 = 12
    assert len(feat) == EDGE_FEATURE_DIM, \
        f"边特征维度错误: 期望 {EDGE_FEATURE_DIM}, 实际 {len(feat)}"
    return feat




def smiles_to_graph(smiles: str) -> tuple:
    """将 SMILES 转换为 PyG 的 x 和 edge_index"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
        
    # ── 1. 节点特征 ──────────────────────────────────────────────────
    x = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom) # [N, 79]
        # 补齐到 128 维 (对应 GraphEncoder 的 node_in_dim=128)
        # 当前 feature 长度大约在 78 维左右
        pad_length = 128 - len(feature)
        if pad_length > 0:
            feature += [0] * pad_length
        else:
            feature = feature[:128] # 防御性截断
            
        x.append(feature)
    x_tensor = torch.tensor(x, dtype=torch.float)

    # ── 2. 边索引 ───────────────────────────────────────────
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 无向图，双向添加
        edge_index.append([i, j])
        edge_index.append([j, i])
        

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




