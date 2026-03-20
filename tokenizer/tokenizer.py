"""
Label Tokenizer for SSR Graph2Sequence Model
专注于 Label 预测的分词器
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter
from SmilesPE.pretokenizer import atomwise_tokenizer

# ==================== 特殊Token定义 ====================

SPECIAL_TOKENS = {
    'pad_token': '[PAD]',
    'unk_token': '[UNK]',
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'sep_token': '[SEP]',
    'mask_token': '[MASK]'
}

# 固定的动作类型
ACTION_TYPES = [
    '[DeleteBond]', '[ChangeBond]', '[AddBond]',
    '[AttachGroup]', '[LeaveGroup]', 
    '[ChangeAtom]', 
    '[Terminate]'
]

# 固定的键类型
BOND_TYPES = ['[NONE]', '[SINGLE]', '[DOUBLE]', '[TRIPLE]', '[AROMATIC]']

# 固定的手性类型
ATOM_CHIRALITY = ['[NONE]', '[CW]', '[CCW]']

class VocabBuilder:
    """从 AtomTokenizer vocab 和 JSON 数据构建完整词表"""
    
    def __init__(
        self, 
        base_vocab_file: str = None,  # ← 新增：基础vocab文件
        min_freq: int = 1, 
        max_vocab_size: Optional[int] = None
    ):
        """
        Args:
            base_vocab_file: AtomTokenizer 的 vocab.txt 路径（可选）
            min_freq: Group SMILES 最小频率
            max_vocab_size: 词表最大大小（不包括基础vocab）
        """
        self.base_vocab_file = base_vocab_file
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.group_counter = Counter()
        
        # 加载基础vocab
        self.base_vocab_set = set()
        if base_vocab_file and Path(base_vocab_file).exists():
            with open(base_vocab_file, 'r') as f:
                self.base_vocab_set = {line.strip() for line in f.readlines()}
            print(f"Loaded base vocab: {len(self.base_vocab_set)} tokens from {base_vocab_file}")
    
    def add_from_json(self, json_file: str):
        """从单个 JSON 文件统计 Group SMILES"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for sample in data:
            edits = sample['output']['edits']
            for edit in edits:
                action_type = edit['action_type']
                label = edit['label']
                
                # 只统计 Group 操作的 label
                if action_type in ['Attach Group', 'Leave Group', 'Terminate']:
                    self.group_counter[label] += 1
    
    def build(self, output_file: str):
        """构建词表并保存（基于 AtomTokenizer vocab）"""
        vocab_set = set()

        # ========== 2. 添加特殊Token（如果不在基础vocab中） ==========
        special_tokens_added = 0
        for token in SPECIAL_TOKENS.values():
            if token not in vocab_set:
                vocab_set.add(token)
                special_tokens_added += 1
        print(f"Step 2: Added special tokens ({special_tokens_added} new tokens)")
        
        # ========== 3. 添加固定Token（Label专用） ==========
        label_tokens_added = 0
        for token in ACTION_TYPES + BOND_TYPES + ATOM_CHIRALITY:
            if token not in vocab_set:
                vocab_set.add(token)
                label_tokens_added += 1
        print(f"Step 3: Added label tokens ({label_tokens_added} new tokens)")
        
        # ========== 4. 添加 Group SMILES 的原子token ==========
        sorted_groups = sorted(
            self.group_counter.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        group_tokens_added = 0
        processed_groups = 0
        for group_smi, freq in sorted_groups:
            if freq < self.min_freq:
                break
            if self.max_vocab_size and processed_groups >= self.max_vocab_size:
                break
            
            tokens = atomwise_tokenizer(group_smi)
            for token in tokens:
                if token not in vocab_set:
                    vocab_set.add(token)
                    group_tokens_added += 1
            
            processed_groups += 1
        
        print(f"Step 4: Added group tokens ({group_tokens_added} new tokens from {processed_groups} groups)")
        
        # ========== 5. 添加基础vocab（AtomTokenizer） ==========
        vocab_set.update(self.base_vocab_set)
        print(f"Step 5: Added base vocab ({len(self.base_vocab_set)} tokens)")
        

        # ========== 5. 构建有序词表列表 ==========
        vocab_list = []
        
        # 5.1 特殊token（保持顺序）
        for token in SPECIAL_TOKENS.values():
            if token in vocab_set:
                vocab_list.append(token)
                vocab_set.discard(token)
        
        # 5.2 Label固定token（保持顺序）
        for token in ACTION_TYPES + BOND_TYPES + ATOM_CHIRALITY:
            if token in vocab_set:
                vocab_list.append(token)
                vocab_set.discard(token)
        
        # 5.3 剩余token（按字母排序）
        vocab_list.extend(sorted(vocab_set))
        
        # ========== 6. 保存词表 ==========
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for token in vocab_list:
                f.write(token + '\n')
        
        # ========== 7. 统计信息 ==========
        print("=" * 60)
        print(f"Vocabulary built: {len(vocab_list)} tokens (deduplicated)")
        print(f"  - Base vocab (AtomTokenizer): {len(self.base_vocab_set)}")
        print(f"  - Special tokens: {len(SPECIAL_TOKENS)}")
        print(f"  - Action types: {len(ACTION_TYPES)}")
        print(f"  - Bond types: {len(BOND_TYPES)}")
        print(f"  - Atom chirality: {len(ATOM_CHIRALITY)}")
        print(f"  - New tokens added: {special_tokens_added + label_tokens_added + group_tokens_added}")
        print(f"  - Processed groups: {processed_groups}/{len(self.group_counter)}")
        print(f"Saved to: {output_file}")
        print("=" * 60)
        
        return vocab_list


# ==================== Label 分词器 ====================

class LabelTokenizer:
    """
    Label 分词器
    
    功能：
    1. 编码 label（字符串 -> token_ids）
    2. 解码 label（token_ids -> 字符串）
    3. 支持 Bond/Atom/Group 三种类型
    """
    
    def __init__(self, vocab_file: str, special_token_dict: dict = SPECIAL_TOKENS):
        """
        Args:
            vocab_file: 词表文件路径
            special_token_dict: 特殊token字典
        """
        # 加载词表
        with open(vocab_file, 'r') as f:
            self.vocab_list = [line.strip() for line in f.readlines()]
        
        self.vocab_size = len(self.vocab_list)
        self.special_token_dict = special_token_dict
        
        # 构建 token -> id 映射
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab_list)}
        
        # 设置特殊token的id
        self.special_ids = []
        for key, value in self.special_token_dict.items():
            assert value in self.token_to_id, f"特殊token {value} 必须在词表中"
            setattr(self, key + '_id', self.token_to_id[value])
            self.special_ids.append(self.token_to_id[value])
        
        # 构建固定token集合（用于快速判断）
        self.bond_tokens = set(BOND_TYPES)
        self.atom_tokens = set(ATOM_CHIRALITY)
        self.action_tokens = set(ACTION_TYPES)
    
    # ========== 编码 ==========
    
    def encode(self, label: str, action_type: str = None) -> List[int]:
        """
        编码 label
        
        Args:
            label: 标签字符串
            action_type: 动作类型（用于判断是否需要原子级分词）
        
        Returns:
            token_ids 列表
        """
        if label is None or label == "":
            return []
        # 1. Bond/Atom 类型：直接映射
        if label in self.bond_tokens or label in self.atom_tokens:
            return [self.token_to_id.get(label, self.unk_token_id)]
        
        # 2. 特殊token：直接映射
        if label in self.action_tokens or label in self.special_token_dict.values():
            return [self.token_to_id.get(label, self.unk_token_id)]
        
        # 3. Group SMILES：原子级分词
        tokens = atomwise_tokenizer(label)
        token_ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # 未知token：逐字符处理
                for char in token:
                    token_ids.append(self.token_to_id.get(char, self.unk_token_id))
        
        return token_ids
    
    def encode_with_special(
        self, 
        label: str, 
        action_type: str = None,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """
        编码 label 并添加特殊token
        
        Args:
            label: 标签字符串
            action_type: 动作类型
            add_bos: 是否添加 BOS
            add_eos: 是否添加 EOS
        
        Returns:
            token_ids 列表
        """
        token_ids = self.encode(label, action_type)
        
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        
        return token_ids
    
    def batch_encode(
        self,
        labels: List[str],
        action_types: List[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True
    ) -> Dict[str, List[List[int]]]:
        """
        批量编码
        
        Args:
            labels: 标签列表
            action_types: 动作类型列表
            padding: 是否padding
            max_length: 最大长度
            truncation: 是否截断
        
        Returns:
            {'input_ids': [[...], ...], 'attention_mask': [[...], ...]}
        """
        if action_types is None:
            action_types = [None] * len(labels)
        
        # 编码所有label
        all_token_ids = []
        for label, action_type in zip(labels, action_types):
            token_ids = self.encode(label, action_type)
            all_token_ids.append(token_ids)
        
        # 确定最大长度
        if max_length is None:
            max_length = max(len(ids) for ids in all_token_ids)
        
        # Padding 和 Truncation
        input_ids = []
        attention_mask = []
        
        for token_ids in all_token_ids:
            # Truncation
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Padding
            mask = [1] * len(token_ids)
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_length
                mask = mask + [0] * pad_length
            
            input_ids.append(token_ids)
            attention_mask.append(mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    # ========== 解码 ==========
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        解码 token_ids
        
        Args:
            token_ids: token id 列表
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            解码后的字符串
        """
        tokens = []
        
        for token_id in token_ids:
            # 跳过特殊token
            if skip_special_tokens and token_id in self.special_ids:
                if token_id == self.eos_token_id:
                    break
                continue
            
            # 转换为token
            if 0 <= token_id < self.vocab_size:
                tokens.append(self.vocab_list[token_id])
        
        return ''.join(tokens)
    
    def batch_decode(
        self, 
        batch_token_ids: List[List[int]], 
        skip_special_tokens: bool = True
    ) -> List[str]:
        """批量解码"""
        return [
            self.decode(token_ids, skip_special_tokens) 
            for token_ids in batch_token_ids
        ]
    
    # ========== 工具方法 ==========
    
    def __call__(
        self, 
        label: str, 
        action_type: str = None,
        padding: str = None,
        max_length: Optional[int] = None,
        truncation: bool = False
    ) -> Dict[str, List[int]]:
        """
        兼容旧版接口
        
        Args:
            label: 标签字符串
            action_type: 动作类型
            padding: 'max_length' 或 None
            max_length: 最大长度
            truncation: 是否截断
        
        Returns:
            {'input_ids': [...], 'attention_mask': [...]}
        """
        token_ids = self.encode(label, action_type)
        attention_mask = [1] * len(token_ids)
        
        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        
        # Padding
        if padding == "max_length" and max_length:
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """获取特殊token的id"""
        return {
            key: getattr(self, key + '_id') 
            for key in self.special_token_dict.keys()
        }


# ==================== 主函数 ====================

def build_vocab_from_dataset(
    json_files: List[str],
    output_file: str = 'vocab/vocab.txt',
    base_vocab_file: str = None,  # ← 新增参数
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None
):
    """
    从数据集构建词表（基于 AtomTokenizer vocab）
    
    Args:
        json_files: JSON文件列表
        output_file: 输出词表文件路径
        base_vocab_file: AtomTokenizer 的 vocab.txt 路径（可选）
        min_freq: Group SMILES 最小频率
        max_vocab_size: 词表最大大小
    """
    # 构建词表
    builder = VocabBuilder(
        base_vocab_file=base_vocab_file,  # ← 传入基础vocab
        min_freq=min_freq, 
        max_vocab_size=max_vocab_size
    )
    
    # 统计所有文件
    for json_file in json_files:
        print(f"Processing {json_file}...")
        builder.add_from_json(json_file)
    
    # 构建词表
    vocab_list = builder.build(output_file)
    
    return vocab_list


# ==================== 使用示例 ====================

if __name__ == '__main__':
    
    # ========== 示例1：构建词表 ==========
    print("=" * 80)
    print("Example 1: Build Vocabulary")
    print("=" * 80)
    
    vocab_list = build_vocab_from_dataset(
        json_files=['dataset/uspto50k/processed/uspto50k_train_output.json'],  # 使用之前生成的示例文件
        output_file='dataset/uspto50k/processed/vocab.txt',
        base_vocab_file='dataset/uspto50k/processed/atom_vocab.txt',
        min_freq=0,
        max_vocab_size=5000
    )
    
    # ========== 示例2：初始化分词器 ==========
    print("\n" + "=" * 80)
    print("Example 2: Initialize Tokenizer")
    print("=" * 80)
    
    tokenizer = LabelTokenizer('dataset/uspto50k/processed/vocab.txt')
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")
    
    # ========== 示例3：编码 Bond Label ==========
    print("\n" + "=" * 80)
    print("Example 3: Encode Bond Label")
    print("=" * 80)
    
    bond_label = '[DOUBLE]'
    encoded = tokenizer.encode(bond_label)
    print(f"Label: {bond_label}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
    
    # ========== 示例4：编码 Atom Label ==========
    print("\n" + "=" * 80)
    print("Example 4: Encode Atom Label")
    print("=" * 80)
    
    atom_label = '[CW]'
    encoded = tokenizer.encode(atom_label)
    print(f"Label: {atom_label}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
    
    # ========== 示例5：编码 Group SMILES ==========
    print("\n" + "=" * 80)
    print("Example 5: Encode Group SMILES")
    print("=" * 80)
    
    group_label = '*C(=O)C'
    encoded = tokenizer.encode(group_label, action_type='Attach Group')
    print(f"Label: {group_label}")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[tokenizer.vocab_list[i] for i in encoded]}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
    
    # ========== 示例6：带特殊token的编码 ==========
    print("\n" + "=" * 80)
    print("Example 6: Encode with Special Tokens")
    print("=" * 80)
    
    encoded_with_special = tokenizer.encode_with_special(
        group_label, 
        add_bos=True, 
        add_eos=True
    )
    print(f"Label: {group_label}")
    print(f"Encoded (with BOS/EOS): {encoded_with_special}")
    print(f"Tokens: {[tokenizer.vocab_list[i] for i in encoded_with_special]}")
    print(f"Decoded: {tokenizer.decode(encoded_with_special)}")
    
    # ========== 示例7：批量编码 ==========
    print("\n" + "=" * 80)
    print("Example 7: Batch Encode")
    print("=" * 80)
    
    labels = ['[DOUBLE]', '[CW]', '*C(=O)C', '*c1ccccc1']
    action_types = ['ChangeBond', 'ChangeAtom', 'AttachGroup', 'LeaveGroup']
    
    batch_result = tokenizer.batch_encode(
        labels, 
        action_types,
        padding=True,
        max_length=20
    )
    
    print(f"Labels: {labels}")
    print(f"\nInput IDs:")
    for i, ids in enumerate(batch_result['input_ids']):
        print(f"  {i}: {ids}")
    
    print(f"\nAttention Mask:")
    for i, mask in enumerate(batch_result['attention_mask']):
        print(f"  {i}: {mask}")
    
    # ========== 示例8：批量解码 ==========
    print("\n" + "=" * 80)
    print("Example 8: Batch Decode")
    print("=" * 80)
    
    decoded_labels = tokenizer.batch_decode(batch_result['input_ids'])
    print(f"Decoded labels:")
    for i, label in enumerate(decoded_labels):
        print(f"  {i}: {label}")
    
    # ========== 示例9：兼容旧版接口 ==========
    print("\n" + "=" * 80)
    print("Example 9: Compatible with Old Interface")
    print("=" * 80)
    
    result = tokenizer(
        label='*C(=O)C',
        action_type='Attach Group',
        padding='max_length',
        max_length=15,
        truncation=True
    )
    
    print(f"Label: *C(=O)C")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Attention Mask: {result['attention_mask']}")
    print(f"Decoded: {tokenizer.decode(result['input_ids'])}")
    
    # ========== 示例10：完整流程 ==========
    print("\n" + "=" * 80)
    print("Example 10: Complete Pipeline for USPTO-50K")
    print("=" * 80)
    
    print("""
完整流程:

# 1. 构建词表
vocab_list = build_vocab_from_dataset(
    json_files=[
        'data/uspto50k/processed/train.json',
        'data/uspto50k/processed/val.json',
        'data/uspto50k/processed/test.json'
    ],
    output_file='data/uspto50k/vocab/vocab.txt',
    min_freq=5,
    max_vocab_size=10000
)

# 2. 初始化分词器
tokenizer = LabelTokenizer('data/uspto50k/vocab/vocab.txt')

# 3. 训练时编码
for sample in train_data:
    for edit in sample['output']['edits']:
        label = edit['label']
        action_type = edit['action_type']
        
        # 编码label
        token_ids = tokenizer.encode(label, action_type)
        
        # 或使用批量编码
        batch_result = tokenizer.batch_encode(
            [label], 
            [action_type],
            padding=True,
            max_length=50
        )

# 4. 推理时解码
predicted_ids = model.predict(...)  # [15, 23, 45, ...]
predicted_label = tokenizer.decode(predicted_ids)
print(f"Predicted label: {predicted_label}")
    """)
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
