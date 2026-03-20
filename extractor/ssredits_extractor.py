from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol, RWMol, rdchem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ==================== 数据结构定义 ====================

@dataclass
class EditAction:
    """
    统一的编辑动作格式
    
    action_type: 动作类型（7种）
    src_idx: 产物图中的源节点索引
    tgt_idx: 产物图中的目标节点索引
    label: 标签（统一为字符串Token）

    注意：
    - src_idx 和 tgt_idx 使用 Atom Index（即 atom.GetIdx()）
    - 这是 Product Graph 中的节点索引，从 0 开始连续
    - 直接对应 Pointer Network 的输出范围 [0, N-1]
    - Atom Mapping ID 仅保存在 metadata['atom_mapping'] 中用于调试
    """
    action_type: str
    src_idx: int      # Atom Index (for Pointer Network)
    tgt_idx: int      # Atom Index (for Pointer Network)
    label: str
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'action_type': self.action_type,
            'src_idx': self.src_idx,
            'tgt_idx': self.tgt_idx,
            'label': self.label
        }


@dataclass
class ProcessedReactionData:
    """处理后的反应数据"""
    rxn_id: str
    
    # 输入 (X)
    product_smi: str
    
    # 输出 (Y)
    reactant_smi: str
    edits: List[Dict]
    
    # 元数据
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """转换为字典格式（便于阅读）"""
        return {
            'rxn_id': self.rxn_id,
            'input': {
                'product_smi': self.product_smi
            },
            'output': {
                'reactant_smi': self.reactant_smi,
                'edits': self.edits
            },
            'metadata': self.metadata
        }
    
    def save_json(self, filepath: str):
        """保存为JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ==================== 核心提取器类 ====================

class SSREditsExtractor:
    """
    SSR编辑序列提取器
    
    Label设计:
    1. Bond Operations: 目标键类型Token
       - 'NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
    2. Atom Operations: 手性Token
       - 'NONE', 'CW', 'CCW'
    3. Group Operations: SMILES字符串
    """
    
    # 键类型映射（使用Token）
    BOND_TYPE_TO_TOKEN = {
        Chem.BondType.SINGLE: 'SINGLE',
        Chem.BondType.DOUBLE: 'DOUBLE',
        Chem.BondType.TRIPLE: 'TRIPLE',
        Chem.BondType.AROMATIC: 'AROMATIC',
    }
    
    TOKEN_TO_BOND_TYPE = {
        'SINGLE': Chem.BondType.SINGLE,
        'DOUBLE': Chem.BondType.DOUBLE,
        'TRIPLE': Chem.BondType.TRIPLE,
        'AROMATIC': Chem.BondType.AROMATIC,
    }
    
    # 手性映射（使用Token）
    CHIRAL_TO_TOKEN = {
        Chem.ChiralType.CHI_UNSPECIFIED: 'NONE',
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 'CW',
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 'CCW',
    }
    
    TOKEN_TO_CHIRAL = {
        'NONE': Chem.ChiralType.CHI_UNSPECIFIED,
        'CW': Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        'CCW': Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    
    def __init__(self, kekulize: bool = True, logger: Optional[logging.Logger] = None):
        self.kekulize = kekulize
        self.logger = logger or logging.getLogger(__name__)
        
    def extract(self, rxn_smi: str, rxn_id: str = None) -> Optional[ProcessedReactionData]:
        """
        从反应SMILES提取编辑序列（主入口）
        """
        try:
            # 1. 解析反应
            react_smi, prod_smi = rxn_smi.strip().split(">>")
            react_mol = Chem.MolFromSmiles(react_smi)
            prod_mol = Chem.MolFromSmiles(prod_smi)
            
            if not self._validate_molecules(react_mol, prod_mol):
                return None
            
            # 2. 标准化分子
            prod_mol, react_mol = self._standardize_molecules(prod_mol, react_mol)
            
            # 3. Kekulize处理
            if self.kekulize:
                react_mol, prod_mol = self._align_kekulize(react_mol, prod_mol)
            
            # 4. 提取编辑序列
            edits = self._extract_edits(prod_mol, react_mol)
            
            # 5. 生成无mapping的SMILES
            product_smi_clean = self._remove_atom_mapping(prod_mol)
            reactant_smi_clean = self._remove_atom_mapping(react_mol)
            
            # 6. 构建元数据
            metadata = {
                'rxn_smi_mapped': rxn_smi,
                'product_smi_mapped': Chem.MolToSmiles(prod_mol),
                'reactant_smi_mapped': Chem.MolToSmiles(react_mol),
                'num_product_atoms': prod_mol.GetNumAtoms(),
                'num_reactant_atoms': react_mol.GetNumAtoms(),
                'num_edits': len(edits),
                'prod_atom_mapping': {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()},
                'edit_summary': self._get_edit_summary(edits)
            }
            
            # 7. 构建返回数据
            return ProcessedReactionData(
                rxn_id=rxn_id or "unknown",
                product_smi=product_smi_clean,
                reactant_smi=reactant_smi_clean,
                edits=[edit.to_dict() for edit in edits],
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract edits for {rxn_id}: {e}")
            return None
    
    # ==================== 分子验证与标准化 ====================
    
    def _validate_molecules(self, react_mol: Mol, prod_mol: Mol) -> bool:
        """验证分子有效性"""
        if react_mol is None or prod_mol is None:
            return False
        if prod_mol.GetNumAtoms() <= 1 or react_mol.GetNumAtoms() <= 1:
            return False
        return True
    
    def _standardize_molecules(self, prod_mol: Mol, react_mol: Mol) -> Tuple[Mol, Mol]:
        """标准化分子：为所有原子添加atom mapping"""
        # 为产物添加atom mapping
        prod_mol = Chem.RWMol(prod_mol)
        max_amap = 0
        for atom in prod_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                max_amap += 1
                atom.SetAtomMapNum(max_amap)
            else:
                max_amap = max(max_amap, atom.GetAtomMapNum())
        
        # 为反应物中新增的原子添加mapping
        react_mol = Chem.RWMol(react_mol)
        for atom in react_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                max_amap += 1
                atom.SetAtomMapNum(max_amap)
        
        return prod_mol.GetMol(), react_mol.GetMol()
    
    def _align_kekulize(self, react_mol: Mol, prod_mol: Mol) -> Tuple[Mol, Mol]:
        """对齐Kekulize"""
        prod_bonds_old = self._get_bond_info(prod_mol)
        Chem.Kekulize(prod_mol)
        prod_bonds_new = self._get_bond_info(prod_mol)
        
        react_bonds_old = self._get_bond_info(react_mol)
        Chem.Kekulize(react_mol)
        react_bonds_new = self._get_bond_info(react_mol)
        
        react_mol = Chem.RWMol(react_mol)
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in react_mol.GetAtoms()}
        
        for bond_key in prod_bonds_new:
            if (bond_key in react_bonds_new and 
                react_bonds_old.get(bond_key) == prod_bonds_old.get(bond_key) and
                react_bonds_new[bond_key] != prod_bonds_new[bond_key]):
                
                a1, a2 = bond_key
                if a1 in amap_idx and a2 in amap_idx:
                    idx1, idx2 = amap_idx[a1], amap_idx[a2]
                    bt_token = prod_bonds_new[bond_key]
                    bond_type = self.TOKEN_TO_BOND_TYPE[bt_token]
                    react_mol.GetBondBetweenAtoms(idx1, idx2).SetBondType(bond_type)
        
        return react_mol.GetMol(), prod_mol
    
    def _remove_atom_mapping(self, mol: Mol) -> str:
        """移除atom mapping并返回SMILES"""
        mol_copy = Chem.RWMol(mol)
        for atom in mol_copy.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol_copy.GetMol())
    
    # ==================== 核心编辑提取逻辑 ====================
    
    def _extract_edits(self, prod_mol: Mol, react_mol: Mol) -> List[EditAction]:
        """
        提取编辑序列（核心方法）
        
        提取顺序:
        1. Bond Level: Delete/Change/Add Bond
        2. Group Level: Leave/Attach Group
        3. Atom Level: Change Atom (手性)
        4. Terminate
        """
        edits = []
        
        # 构建映射关系
        prod_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}
        react_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in react_mol.GetAtoms()}
        
        prod_bonds = self._get_bond_info(prod_mol)
        react_bonds = self._get_bond_info(react_mol)
        
        prod_atoms = self._get_atom_info(prod_mol)
        react_atoms = self._get_atom_info(react_mol)
        
        bond_edited_atoms = set()
        
        # ========== 1. Bond Level Edits ==========
        
        # 1.1 Delete Bond (产物中有，反应物中无)
        for bond_key in prod_bonds:
            if bond_key not in react_bonds:
                a1_map, a2_map = bond_key
                src_idx = prod_amap_idx[a1_map]
                tgt_idx = prod_amap_idx[a2_map]
                
                # Label: 'NONE' (删除后无键)
                edits.append(EditAction(
                    action_type='DeleteBond',
                    src_idx=src_idx,
                    tgt_idx=tgt_idx,
                    label='NONE'
                ))
                
                bond_edited_atoms.update([a1_map, a2_map])
        
        # 1.2 Change Bond (键类型改变)
        for bond_key in prod_bonds:
            if bond_key in react_bonds and prod_bonds[bond_key] != react_bonds[bond_key]:
                a1_map, a2_map = bond_key
                src_idx = prod_amap_idx[a1_map]
                tgt_idx = prod_amap_idx[a2_map]
                
                # Label: 目标键类型 (如 'DOUBLE')
                target_bond_type = react_bonds[bond_key]
                
                edits.append(EditAction(
                    action_type='ChangeBond',
                    src_idx=src_idx,
                    tgt_idx=tgt_idx,
                    label=target_bond_type
                ))
                
                bond_edited_atoms.update([a1_map, a2_map])
        
        # 1.3 Add Bond (反应物中有，产物中无，且形成环)
        for bond_key in react_bonds:
            if bond_key not in prod_bonds:
                a1_map, a2_map = bond_key
                
                if a1_map in prod_amap_idx and a2_map in prod_amap_idx:
                    bond = react_mol.GetBondBetweenAtoms(
                        react_amap_idx[a1_map], 
                        react_amap_idx[a2_map]
                    )
                    
                    if bond and bond.IsInRing():
                        src_idx = prod_amap_idx[a1_map]
                        tgt_idx = prod_amap_idx[a2_map]
                        
                        # Label: 添加的键类型 (如 'SINGLE')
                        added_bond_type = react_bonds[bond_key]
                        
                        edits.append(EditAction(
                            action_type='AddBond',
                            src_idx=src_idx,
                            tgt_idx=tgt_idx,
                            label=added_bond_type
                        ))
                        
                        bond_edited_atoms.update([a1_map, a2_map])
        
        # ========== 2. Group Level Edits ==========
        
        atoms_only_in_prod = set(prod_atoms.keys()) - set(react_atoms.keys())
        atoms_only_in_react = set(react_atoms.keys()) - set(prod_atoms.keys())
        
        # 2.1 Leave Group
        leave_group_edits = self._extract_leave_groups(
            prod_mol, prod_amap_idx, atoms_only_in_prod
        )
        edits.extend(leave_group_edits)
        
        # 2.2 Attach Group
        attach_group_edits = self._extract_attach_groups_bfs(
            react_mol, prod_mol, react_amap_idx, prod_amap_idx, atoms_only_in_react
        )
        edits.extend(attach_group_edits)
        
        # ========== 3. Atom Level Edits (只处理手性变化) ==========
        
        for atom_map in prod_atoms:
            if atom_map in react_atoms:
                prod_chiral = prod_atoms[atom_map]
                react_chiral = react_atoms[atom_map]
                
                # 只有手性发生变化时才添加编辑
                if prod_chiral != react_chiral:
                    src_idx = prod_amap_idx[atom_map]
                    
                    # Label: 目标手性 ('NONE', 'CW', 'CCW')
                    edits.append(EditAction(
                        action_type='ChangeAtom',
                        src_idx=src_idx,
                        tgt_idx=-1,
                        label=react_chiral
                    ))
        
        # ========== 4. Terminate ==========
        edits.append(EditAction(
            action_type='Terminate',
            src_idx=-1,
            tgt_idx=-1,
            label='Terminate'
        ))
        
        return edits
    
    # ==================== 基团提取（BFS方法） ====================
    
    def _extract_leave_groups(
        self, 
        prod_mol: Mol, 
        prod_amap_idx: Dict[int, int],
        atoms_only_in_prod: Set[int]
    ) -> List[EditAction]:
        """提取离去基团"""
        edits = []
        processed_atoms = set()
        
        for bond in prod_mol.GetBonds():
            a1_map = bond.GetBeginAtom().GetAtomMapNum()
            a2_map = bond.GetEndAtom().GetAtomMapNum()
            
            if (a1_map in atoms_only_in_prod or a2_map in atoms_only_in_prod):
                if a1_map not in atoms_only_in_prod:
                    anchor_map, leaving_map = a1_map, a2_map
                elif a2_map not in atoms_only_in_prod:
                    anchor_map, leaving_map = a2_map, a1_map
                else:
                    continue
                
                group_atoms = self._bfs_extract_group(
                    prod_mol, leaving_map, atoms_only_in_prod, exclude_atom=anchor_map
                )
                
                if group_atoms & processed_atoms:
                    continue
                processed_atoms.update(group_atoms)
                
                group_smi = self._build_group_smiles_with_dummy(
                    prod_mol, group_atoms, anchor_map, prod_amap_idx
                )
                
                if group_smi:
                    edits.append(EditAction(
                        action_type='LeaveGroup',
                        src_idx=prod_amap_idx[anchor_map],
                        tgt_idx=-1,
                        label=group_smi
                    ))
        
        return edits
    
    def _extract_attach_groups_bfs(
        self,
        react_mol: Mol,
        prod_mol: Mol,
        react_amap_idx: Dict[int, int],
        prod_amap_idx: Dict[int, int],
        atoms_only_in_react: Set[int]
    ) -> List[EditAction]:
        """使用BFS提取连接基团"""
        edits = []
        processed_atoms = set()
        
        for bond in react_mol.GetBonds():
            a1_map = bond.GetBeginAtom().GetAtomMapNum()
            a2_map = bond.GetEndAtom().GetAtomMapNum()
            
            if (a1_map in prod_amap_idx and a2_map in atoms_only_in_react):
                anchor_map, attach_start_map = a1_map, a2_map
            elif (a2_map in prod_amap_idx and a1_map in atoms_only_in_react):
                anchor_map, attach_start_map = a2_map, a1_map
            else:
                continue
            
            group_atoms = self._bfs_extract_group(
                react_mol, attach_start_map, atoms_only_in_react, exclude_atom=anchor_map
            )
            
            if group_atoms & processed_atoms:
                continue
            processed_atoms.update(group_atoms)
            
            group_smi = self._build_group_smiles_with_dummy(
                react_mol, group_atoms, anchor_map, react_amap_idx
            )
            
            if group_smi:
                src_idx = prod_amap_idx.get(anchor_map, -1) if '*' in group_smi else -1
                
                edits.append(EditAction(
                    action_type='AttachGroup',
                    src_idx=src_idx,
                    tgt_idx=-1,
                    label=group_smi
                ))
        
        # 处理独立分子
        remaining_atoms = atoms_only_in_react - processed_atoms
        if remaining_atoms:
            components = self._find_connected_components(react_mol, remaining_atoms)
            for component in components:
                group_smi = self._build_group_smiles_with_dummy(
                    react_mol, component, anchor_map=None, amap_idx=react_amap_idx
                )
                
                if group_smi:
                    edits.append(EditAction(
                        action_type='AttachGroup',
                        src_idx=-1,
                        tgt_idx=-1,
                        label=group_smi
                    ))
        
        return edits
    
    def _bfs_extract_group(
        self, mol: Mol, start_atom_map: int, 
        valid_atoms: Set[int], exclude_atom: Optional[int] = None
    ) -> Set[int]:
        """使用BFS提取连通基团"""
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        
        if start_atom_map not in amap_idx:
            return set()
        
        visited = set()
        queue = deque([start_atom_map])
        visited.add(start_atom_map)
        
        while queue:
            current_map = queue.popleft()
            current_idx = amap_idx[current_map]
            atom = mol.GetAtomWithIdx(current_idx)
            
            for neighbor in atom.GetNeighbors():
                neighbor_map = neighbor.GetAtomMapNum()
                
                if (neighbor_map in valid_atoms and 
                    neighbor_map not in visited and
                    neighbor_map != exclude_atom):
                    
                    visited.add(neighbor_map)
                    queue.append(neighbor_map)
        
        return visited
    
    def _build_group_smiles_with_dummy(
        self, mol: Mol, group_atom_maps: Set[int], 
        anchor_map: Optional[int], amap_idx: Dict[int, int]
    ) -> str:
        """构建带虚原子的基团SMILES"""
        if not group_atom_maps:
            return ""
        
        submol = Chem.RWMol()
        old_to_new_idx = {}
        
        # 添加基团中的原子
        for atom_map in sorted(group_atom_maps):
            if atom_map not in amap_idx:
                continue
            old_idx = amap_idx[atom_map]
            old_atom = mol.GetAtomWithIdx(old_idx)
            
            new_atom = Chem.Atom(old_atom.GetSymbol())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(old_atom.GetNumExplicitHs())
            new_atom.SetChiralTag(old_atom.GetChiralTag())
            
            new_idx = submol.AddAtom(new_atom)
            old_to_new_idx[old_idx] = new_idx
        
        # 添加虚原子
        dummy_idx = None
        if anchor_map is not None and anchor_map in amap_idx:
            dummy_atom = Chem.Atom(0)
            dummy_idx = submol.AddAtom(dummy_atom)
        
        # 添加键
        for bond in mol.GetBonds():
            begin_map = bond.GetBeginAtom().GetAtomMapNum()
            end_map = bond.GetEndAtom().GetAtomMapNum()
            
            begin_idx = amap_idx.get(begin_map)
            end_idx = amap_idx.get(end_map)
            
            # 基团内部的键
            if begin_map in group_atom_maps and end_map in group_atom_maps:
                if begin_idx in old_to_new_idx and end_idx in old_to_new_idx:
                    submol.AddBond(
                        old_to_new_idx[begin_idx],
                        old_to_new_idx[end_idx],
                        bond.GetBondType()
                    )
            
            # 连接到anchor的键
            elif dummy_idx is not None:
                if begin_map in group_atom_maps and end_map == anchor_map:
                    if begin_idx in old_to_new_idx:
                        submol.AddBond(old_to_new_idx[begin_idx], dummy_idx, bond.GetBondType())
                elif end_map in group_atom_maps and begin_map == anchor_map:
                    if end_idx in old_to_new_idx:
                        submol.AddBond(old_to_new_idx[end_idx], dummy_idx, bond.GetBondType())
        
        # 生成SMILES
        final_mol = submol.GetMol()
        for atom in final_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        try:
            smi = Chem.MolToSmiles(final_mol)
            return smi
        except:
            return ""
    
    def _find_connected_components(self, mol: Mol, atom_maps: Set[int]) -> List[Set[int]]:
        """找到原子集合中的所有连通分量"""
        unvisited = set(atom_maps)
        components = []
        
        while unvisited:
            start = next(iter(unvisited))
            component = self._bfs_extract_group(mol, start, atom_maps)
            components.append(component)
            unvisited -= component
        
        return components
    
    # ==================== 辅助方法 ====================
    
    def _get_bond_info(self, mol: Mol) -> Dict[Tuple[int, int], str]:
        """获取键信息（返回Token）"""
        bond_info = {}
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            key = tuple(sorted([a1, a2]))
            
            bt_token = self.BOND_TYPE_TO_TOKEN.get(bond.GetBondType(), 'SINGLE')
            bond_info[key] = bt_token
        
        return bond_info
    
    def _get_atom_info(self, mol: Mol) -> Dict[int, str]:
        """获取原子信息（只返回手性Token）"""
        atom_info = {}
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            chiral_token = self.CHIRAL_TO_TOKEN.get(atom.GetChiralTag(), 'NONE')
            atom_info[amap] = chiral_token
        
        return atom_info
    
    def _get_edit_summary(self, edits: List[EditAction]) -> Dict[str, int]:
        """统计编辑操作类型"""
        summary = defaultdict(int)
        for edit in edits:
            summary[edit.action_type] += 1
        return dict(summary)


# ==================== 批量处理工具 ====================

class DatasetProcessor:
    """数据集批量处理器"""
    
    def __init__(self, extractor: SSREditsExtractor):
        self.extractor = extractor
        self.logger = logging.getLogger(__name__)
    
    def process_dataset(
        self, 
        input_file: str, 
        output_file: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, int]:
        """批量处理数据集"""
        import pandas as pd
        
        # 读取数据
        df = pd.read_csv(input_file)
        if max_samples:
            df = df.head(max_samples)
        
        self.logger.info(f"Processing {len(df)} reactions from {input_file}")
        
        # 处理每个反应
        results = []
        stats = {
            'total': len(df),
            'success': 0,
            'failed': 0,
            'avg_edits': 0.0,
            'avg_product_atoms': 0.0
        }
        
        for idx, row in df.iterrows():
            rxn_id = row.get('id', str(idx))
            rxn_smi = row.get('reactants>reagents>production', '')
            
            if not rxn_smi or '>>' not in rxn_smi:
                stats['failed'] += 1
                continue
            
            result = self.extractor.extract(rxn_smi, rxn_id=rxn_id)
            
            if result:
                results.append(result.to_dict())
                stats['success'] += 1
                stats['avg_edits'] += result.metadata['num_edits']
                stats['avg_product_atoms'] += result.metadata['num_product_atoms']
            else:
                stats['failed'] += 1
            
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(df)} reactions")
        
        # 计算平均值
        if stats['success'] > 0:
            stats['avg_edits'] /= stats['success']
            stats['avg_product_atoms'] /= stats['success']
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
        self.logger.info(f"Statistics: {stats}")
        
        return stats
    
    def process_splits(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        output_dir: str
    ):
        """处理训练/验证/测试集"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理训练集
        self.logger.info("Processing training set...")
        train_stats = self.process_dataset(train_file, str(output_path / 'train.json'))
        
        # 处理验证集
        self.logger.info("Processing validation set...")
        val_stats = self.process_dataset(val_file, str(output_path / 'val.json'))
        
        # 处理测试集
        self.logger.info("Processing test set...")
        test_stats = self.process_dataset(test_file, str(output_path / 'test.json'))
        
        # 保存统计信息
        stats_summary = {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        }
        
        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        self.logger.info("All splits processed successfully!")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # # ========== 示例1：单个反应提取 ==========
    # print("=" * 80)
    # print("Example 1: Single Reaction Extraction")
    # print("=" * 80)
    
    extractor = SSREditsExtractor(kekulize=True)
    
    # # 示例：酰胺化反应
    # rxn_smi = "[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:6]"
    # result = extractor.extract(rxn_smi, rxn_id="amidation_001")
    
    # if result:
    #     print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    #     result.save_json('example_amidation.json')
    #     print("\nSaved to: example_amidation.json")
    
    # # ========== 示例2：键类型变化 ==========
    # print("\n" + "=" * 80)
    # print("Example 2: Bond Type Change")
    # print("=" * 80)
    
    # # 示例：单键变双键
    # rxn_smi2 = "[CH3:1][CH2:2][CH2:3][OH:4]>>[CH3:1][CH:2]=[CH:3][OH:4]"
    # result2 = extractor.extract(rxn_smi2, rxn_id="dehydration_001")
    
    # if result2:
    #     print(f"\nReaction: {result2.product_smi} -> {result2.reactant_smi}")
    #     print("\nEdits:")
    #     for i, edit in enumerate(result2.edits):
    #         print(f"  {i+1}. {edit['action_type']:15s} | "
    #               f"nodes ({edit['src_idx']:2d}, {edit['tgt_idx']:2d}) | "
    #               f"label: {edit['label']}")
    
    # # ========== 示例3：手性变化 ==========
    # print("\n" + "=" * 80)
    # print("Example 3: Chirality Change")
    # print("=" * 80)
    
    # # 示例：手性反转
    # rxn_smi3 = "[C@H:1]([CH3:2])([OH:3])[CH2:4][CH3:5]>>[C@@H:1]([CH3:2])([OH:3])[CH2:4][CH3:5]"
    # result3 = extractor.extract(rxn_smi3, rxn_id="inversion_001")
    
    # if result3:
    #     print(f"\nReaction: {result3.product_smi} -> {result3.reactant_smi}")
    #     print("\nEdits:")
    #     for i, edit in enumerate(result3.edits):
    #         if edit['action_type'] == 'Change Atom':
    #             print(f"  {i+1}. {edit['action_type']:15s} | "
    #                   f"node {edit['src_idx']:2d} | "
    #                   f"chirality: {edit['label']}")
    
    
    
    # ========== 示例6：批量处理 ==========
    print("\n" + "=" * 80)
    print("Example 6: Batch Processing")
    print("=" * 80)
    
    # 创建示例数据
    import pandas as pd
    
    # demo_data = pd.DataFrame({
    #     'id': ['rxn_001', 'rxn_002', 'rxn_003'],
    #     'reactants>reagents>production': [
    #         "[CH3:1][OH:2].[CH3:3][C:4](=[O:5])[Cl:6]>>[CH3:1][O:2][C:4](=[O:5])[CH3:3]",
    #         "[CH3:1][CH2:2][Br:3].[OH-:4]>>[CH3:1][CH2:2][OH:4]",
    #         "[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[c:1]1[cH:2][cH:3][c:4]([Br:7])[cH:5][cH:6]1"
    #     ]
    # })
    
    # demo_data.to_csv('demo_input.csv', index=False)
    data_name = 'valid'
    processor = DatasetProcessor(extractor)
    stats = processor.process_dataset(f'dataset/uspto50k/raw/canonicalized_{data_name}.csv', 
                                      f'dataset/uspto50k/processed/uspto50k_{data_name}_output.json')
    
    print(f"\nProcessing Statistics:")
    print(f"  Total:   {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Avg edits per reaction: {stats['avg_edits']:.2f}")
    
    # ========== 示例7：处理USPTO-50K ==========
    print("\n" + "=" * 80)
    print("Example 7: Process USPTO-50K Dataset")
    print("=" * 80)
    
    print("""
Usage for USPTO-50K:

processor = DatasetProcessor(extractor)

processor.process_splits(
    train_file='data/uspto50k/raw_train.csv',
    val_file='data/uspto50k/raw_val.csv',
    test_file='data/uspto50k/raw_test.csv',
    output_dir='data/uspto50k/processed'
)

Output files:
  - processed/train.json  (训练集)
  - processed/val.json    (验证集)
  - processed/test.json   (测试集)
  - processed/stats.json  (统计信息)
    """)
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
