from rdkit import Chem
from rdkit.Chem import Mol, rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
from collections import namedtuple

from extractor.reaction_actions import (GroupAction, AtomEditAction,
                                    BondEditAction, Termination)
from utils.chem import get_atom_info, get_bond_info, get_bond_substructure, align_kekulize_pairs, remap_rxn_smi_r, remap_rxn_smi_p, canonicalize_mol_smi


ReactionData = namedtuple(
    "ReactionData", ['rxn_name', 'rxn_smi', 'query_mol_smi', 'res_mol_smi',
                     'edits', 'edits_anno', 'edits_atom_mapid','edits_atom_id'])


def frag_is_a_mol(frag_smi, react_smi):
    reacts = react_smi.strip().split(".")
    reacts_can = [Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reacts]
    if Chem.MolToSmiles(Chem.MolFromSmiles(frag_smi)) in reacts_can:
        return True
    else:
        return False

def chiral2string(atom_type, formal_charge, chiral_tag):
    """返回包含手性和电荷信息的原子字符串"""
    # 处理电荷
    charge_str = ""
    if formal_charge != 0:
        if formal_charge == 1:
            charge_str = "+"
        elif formal_charge == -1:
            charge_str = "-"
        elif formal_charge > 0:
            charge_str = f"+{formal_charge}"
        else:
            charge_str = f"{formal_charge}"
    
    # 处理手性
    chiral_str = ""
    if rdchem.ChiralType.values[chiral_tag] == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_str = '@'
    elif rdchem.ChiralType.values[chiral_tag] == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_str = '@@'
    
    return f"{atom_type}{chiral_str}{charge_str}"

def template_extractor(rxn_smi: str, 
                       direction='backward', # "backward" or "forward"
                       kekulize: bool = True, 
                       rxn_class: int = None, 
                       rxn_id: str = None,
                       rxn_name: str = None,
                       remap=False) -> ReactionData:
    # generate bond and atom edits 
    if remap:
        if direction == 'backward': 
            rxn_smi, p_amap_idx = remap_rxn_smi_p(rxn_smi)
            r, p = rxn_smi.strip().split(">>")
            react_mol = Chem.MolFromSmiles(r)
            prod_mol = Chem.MolFromSmiles(p)
            r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                        for atom in react_mol.GetAtoms()}
            
            
        elif direction == 'forward': 
            rxn_smi, r_amap_idx = remap_rxn_smi_r(rxn_smi)
            r, p = rxn_smi.strip().split(">>")
            react_mol = Chem.MolFromSmiles(r)
            prod_mol = Chem.MolFromSmiles(p)
            p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in prod_mol.GetAtoms()}

    else:
        r, p = rxn_smi.strip().split(">>")
        react_mol = Chem.MolFromSmiles(r)
        prod_mol = Chem.MolFromSmiles(p)
        p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                    for atom in prod_mol.GetAtoms()}
        r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                    for atom in react_mol.GetAtoms()}
    

    if (react_mol is None) or (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
        return None

    r_new, p_new = Chem.MolToSmiles(react_mol), Chem.MolToSmiles(prod_mol)
    rxn_smi_new = r_new + ">>" + p_new

    if kekulize:
        react_mol, prod_mol = align_kekulize_pairs(react_mol, prod_mol)

    edits = []
    edits_strings = []
    edits_anno = []
    edits_atom_mapid = []
    edits_atom_id = []
    
    """
    Bond Level - 处理键的变化
    """
    prod_bonds = get_bond_info(prod_mol)
    react_bonds = get_bond_info(react_mol)

    # 1. 检测被删除的键（在product中存在但在reactant中不存在）
    for bond in prod_bonds:
        if bond not in react_bonds:
            a1, a2 = bond
            if direction == 'backward':
                action_vocab = 'Delete Bond'
                sub_smiles = get_bond_substructure(a1, a2, prod_mol) 
                atom_idx = [p_amap_idx[a1], p_amap_idx[a2]]
            elif direction == 'forward': 
                action_vocab = 'Add Bond'
                sub_smiles = get_bond_substructure(a1, a2, prod_mol)
                atom_idx = [r_amap_idx[a1], r_amap_idx[a2]] 
                    
            edit = BondEditAction(a1, a2, sub_smiles, action_vocab=action_vocab)
            edits.append(edit)
            edits_strings.append(str(edit))
            edits_anno.append(edit.get_tuple())
            edits_atom_mapid.append([a1, a2])
            edits_atom_id.append(atom_idx)
    
    # 2. 检测键类型变化的键
    for bond in prod_bonds:
        if bond in react_bonds and prod_bonds[bond][:2] != react_bonds[bond][:2]:
            a1, a2 = bond
            if direction == 'backward':
                sub_smiles = get_bond_substructure(a1, a2, react_mol)
                atom_idx = [p_amap_idx[a1], p_amap_idx[a2]]
            elif direction == 'forward':
                sub_smiles = get_bond_substructure(a1, a2, prod_mol) 
                atom_idx = [r_amap_idx[a1], r_amap_idx[a2]]
            
            edit = BondEditAction(a1, a2, sub_smiles, action_vocab='Change Bond')
            edits.append(edit)
            edits_strings.append(str(edit))
            edits_anno.append(edit.get_tuple())
            edits_atom_mapid.append([a1, a2])
            edits_atom_id.append(atom_idx)
    
    # 3. 检测新增的键（在反应物中存在但在产物中不存在）
    for bond in react_bonds:
        if bond not in prod_bonds:
            a1, a2 = bond
            action_vocab = 'Add Bond' if direction == 'backward' else 'Delete Bond'
            sub_smiles = get_bond_substructure(a1, a2, react_mol) 
            atom_idx = [r_amap_idx[a1], r_amap_idx[a2]]                    
            edit = BondEditAction(a1, a2, sub_smiles, action_vocab=action_vocab)
            edits.append(edit)
            edits_strings.append(str(edit))
            edits_anno.append(edit.get_tuple())
            edits_atom_mapid.append([a1, a2])
            edits_atom_id.append(atom_idx)

    """
    Atom Level - 处理原子属性变化
    """
    prod_atoms = get_atom_info(prod_mol)
    react_atoms = get_atom_info(react_mol)
    
    for atom_map_num in prod_atoms:
        # 只处理在反应物中也存在的原子（原子守恒）
        if atom_map_num in react_atoms:
            prod_atom_info = prod_atoms[atom_map_num]
            react_atom_info = react_atoms[atom_map_num]
            
            # 比较原子属性：符号、电荷、手性
            if prod_atom_info != react_atom_info:
                # 根据方向确定使用哪个状态
                if direction == 'backward':
                    # 逆向反应：目标状态是反应物状态
                    atom_string = chiral2string(
                        react_atom_info[0], 
                        react_atom_info[1], 
                        react_atom_info[2]
                    )
                    atom_idx = p_amap_idx[atom_map_num]
                else:
                    # 正向反应：目标状态是产物状态 
                    atom_string = chiral2string(
                        prod_atom_info[0],
                        prod_atom_info[1],
                        prod_atom_info[2]
                    )
                    atom_idx = r_amap_idx[atom_map_num]
                
                edit = AtomEditAction(atom_map_num, atom_string, action_vocab='Change Atom')
                edits.append(edit)
                edits_strings.append(str(edit))
                edits_anno.append(edit.get_tuple())
                edits_atom_mapid.append(atom_map_num)
                edits_atom_id.append(atom_idx)

    # 添加终止动作
    edit = Termination(action_vocab='Terminate')
    edits.append(edit)
    edits_anno.append(edit.get_tuple())
    
    # 清除原子映射号并生成查询分子和结果分子SMILES
    for atom in prod_mol.GetAtoms(): atom.SetAtomMapNum(0)
    for atom in react_mol.GetAtoms(): atom.SetAtomMapNum(0)
    if direction == 'backward':
        query_mol_smi = Chem.MolToSmiles(prod_mol)
        res_mol_smi = Chem.MolToSmiles(react_mol)
    elif direction == 'forward':
        query_mol_smi = Chem.MolToSmiles(react_mol)
        res_mol_smi = Chem.MolToSmiles(prod_mol)
    

    reaction_data = ReactionData(
        rxn_name=rxn_name, rxn_smi=rxn_smi_new, query_mol_smi=query_mol_smi, res_mol_smi=res_mol_smi,
        edits=edits, edits_anno=edits_anno, edits_atom_mapid=edits_atom_mapid, edits_atom_id=edits_atom_id)

    return reaction_data





if __name__ == "__main__":
    # Example usage
    rxn_smi = "[O:1]=[C:2]([O-:3])[C@H:4]([O:5][C@H:6]1[O:7][C@H:8]([CH2:9][OH:10])[C@@H:11]([OH:12])[C@H:13]([OH:14])[C@H:15]1[OH:16])[CH2:17][O:18][P:19](=[O:20])([O-:21])[O-:22].[*:23][C@@H:24]1[O:25][C@H:26]([CH2:27][O:28][P:29](=[O:30])([O-:31])[O:32][P:33](=[O:34])([O-:35])[OH:36])[C@@H:37]([OH:38])[C@H:39]1[OH:40]>>[O:34]=[P:33]([O-:35])([O:36][C@H:6]1[O:7][C@H:8]([CH2:9][OH:10])[C@@H:11]([OH:12])[C@H:13]([OH:14])[C@H:15]1[OH:16])[O:32][P:29]([O:28][CH2:27][C@H:26]2[O:25][C@@H:24]([*:23])[C@@H:39]([C@@H:37]2[OH:38])[OH:40])(=[O:30])[O-:31].[O:1]=[C:2]([O-:3])[C@H:4]([OH:5])[CH2:17][O:18][P:19](=[O:20])([O-:21])[O-:22]"
    reaction_data = template_extractor(rxn_smi, direction='backward', kekulize=True, remap=False)
    print(reaction_data)
    for edit in reaction_data.edits:
        print(edit)