"""
Definitions of basic 'edits' (Actions) to transform a product into synthons and reactants
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, List
from collections import namedtuple

from rdkit import Chem
from rdkit.Chem import Mol, rdchem
from utils.chem import canonicalize_mol_smi, get_atom_Chiral, fix_Hs_Charge
from extractor.features import string2chiral, BOND_STEREO_DICT, BOND_TYPE_DICT


ReactionData = namedtuple(
    "ReactionData", ['rxn_name', 'rxn_smi', 'query_mol_smi', 'res_mol_smi',
                     'edits', 'edits_anno', 'edits_atom_mapid','edits_atom_id'])


class ReactionAction(metaclass=ABCMeta):
    def __init__(self, atom_map1: int, atom_map2: int, action_vocab: str):
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.action_vocab = action_vocab

    @abstractmethod
    def get_tuple(self) -> Tuple[str, ...]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def apply(self, mol: Mol, map2idx: dict = None) -> Mol:
        raise NotImplementedError('Abstract method')


class AtomEditAction(ReactionAction):
    def __init__(self, atom_map1: int, 
                 atom_string: str,
                 action_vocab: str):
        super().__init__(atom_map1, -1, action_vocab)
        self.atom_string = atom_string
        self.atom_type, self.chiral_tag = string2chiral(self.atom_string)

    def get_tuple(self) -> Tuple[str, str]:
        return self.action_vocab, self.atom_string 

    def apply(self, mol: Mol, map2idx: dict) -> Mol:
        new_mol = Chem.RWMol(mol)
        atom_idx = map2idx[self.atom_map1]
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetChiralTag(rdchem.ChiralType.values[self.chiral_tag])
        pred_mol = fix_Hs_Charge(new_mol.GetMol())
        return pred_mol

    def __str__(self):
        return f'Edit Atom {self.atom_map1}: Atom_Type={self.atom_type}, Chiral_tag={self.chiral_tag}'


class BondEditAction(ReactionAction):
    def __init__(self, 
                 atom_map1: int, 
                 atom_map2: int,
                 bond_sub_smiles: str,
                 action_vocab: str):
        super().__init__(atom_map1, atom_map2, action_vocab)
        self.bond_sub_smiles = bond_sub_smiles
        
        bmol = Chem.MolFromSmiles(bond_sub_smiles)
        bond = bmol.GetBonds()[0]
        self.bond_type = int(bond.GetBondType()) if action_vocab != 'Delete Bond' else None
        self.bond_stereo = int(bond.GetStereo()) if action_vocab != 'Delete Bond' else None

    def get_tuple(self) -> Tuple[str, str]:
        return self.action_vocab, self.bond_sub_smiles

    def apply(self, mol: Mol, amap_idx: dict) -> Mol: 
        new_mol = Chem.RWMol(mol)
        atom1_idx = amap_idx[self.atom_map1]
        atom2_idx = amap_idx[self.atom_map2]
        
        if self.action_vocab == 'Add Bond': 
            new_mol.AddBond(atom1_idx, atom2_idx, rdchem.BondType.values[self.bond_type])
            return fix_Hs_Charge(new_mol.GetMol())
            
        elif self.action_vocab == 'Delete Bond':
            new_mol.RemoveBond(atom1_idx, atom2_idx)
            return new_mol.GetMol()
            
        else:  # Change Bond 
            bond = new_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            if bond:
                bond.SetBondType(rdchem.BondType.values[self.bond_type])
                bond.SetStereo(rdchem.BondStereo.values[self.bond_stereo])
            return fix_Hs_Charge(new_mol.GetMol())

    def __str__(self):
        if self.bond_type is None:
            return f'Delete bond {self.atom_map1, self.atom_map2}'
        return f'{self.action_vocab} {self.atom_map1, self.atom_map2}'


class Termination(ReactionAction):
    def __init__(self, action_vocab: str):
        super().__init__(-1, -1, action_vocab=action_vocab)

    def get_tuple(self) -> Tuple[str]:
        return self.action_vocab

    def apply(self, mol: Mol) -> Mol:
        atom_chiral = get_atom_Chiral(mol)
        for atom in mol.GetAtoms():
            amap_num = atom.GetAtomMapNum()
            atom.SetChiralTag(atom_chiral.get(amap_num, rdchem.ChiralType.CHI_UNSPECIFIED))
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    def __str__(self):
        return 'Terminate'


def apply_edit_to_mol(mol: Mol, edit: tuple, edit_atom: List[int], map2idx: dict) -> Mol:
    """ Apply edits to molecular graph """
    action_vocab = edit[0]
    
    if 'Bond' in action_vocab:
        a_map_1, a_map_2 = edit_atom[0], edit_atom[1]
        edit_exe = BondEditAction(
            atom_map1=a_map_1, 
            atom_map2=a_map_2, 
            bond_sub_smiles=edit[1], 
            action_vocab=action_vocab)
        return edit_exe.apply(mol, map2idx)
            
    elif action_vocab == 'Change Atom':
        edit_exe = AtomEditAction(
            edit_atom, 
            atom_string=edit[1], 
            action_vocab=action_vocab)
        return edit_exe.apply(mol, map2idx)

    return mol


def templates_to_reactants(query_smi: str, edit_annos: list, edit_atoms: list) -> Mol:
    """ Apply a sequence of edits to a query molecule """
    can_smi = canonicalize_mol_smi(query_smi)
    p_mol = Chem.MolFromSmiles(can_smi)
    Chem.Kekulize(p_mol)
    
    # Add atom mapping to unmapped atoms
    max_amap = max([atom.GetAtomMapNum() for atom in p_mol.GetAtoms()] or [0])
    for atom in p_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            max_amap += 1
            atom.SetAtomMapNum(max_amap)

    # Create mapping dictionaries
    map2idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in p_mol.GetAtoms()}
    idx2map = {atom.GetIdx(): atom.GetAtomMapNum() for atom in p_mol.GetAtoms()}
    
    # Convert atom indices to map numbers
    edit_atoms_map = []
    for a in edit_atoms:
        if isinstance(a, list): 
            edit_atoms_map.append([idx2map[x] for x in a])
        else:
            edit_atoms_map.append(idx2map[a] if a > -1 else a)
        
    # Apply edits sequentially
    int_mol = p_mol 
    for edit_anno, edit_atom in zip(edit_annos, edit_atoms_map):
        if edit_anno == 'Terminate':
            edit_exe = Termination(action_vocab='Terminate')
            return edit_exe.apply(int_mol)
        else:
            int_mol = apply_edit_to_mol(int_mol, edit_anno, edit_atom, map2idx)
            if int_mol is None:
                break
    
    return int_mol

if __name__=='__main__':

    # backward
    rxn = ReactionData(rxn_name='RXN-9874', 
                       rxn_smi='[CH:1]1=[CH:2][CH:5]([OH:9])[C:7]([C:6](=[O:10])[OH:11])([OH:12])[CH:3]=[C:4]1[Cl:8]>>[cH:1]1[cH:2][c:5]([OH:9])[c:7]([OH:12])[cH:3][c:4]1[Cl:8]', 
                       query_mol_smi='Oc1ccc(Cl)cc1O', 
                       res_mol_smi='O=C(O)C1(O)C=C(Cl)C=CC1O', 
                       edits=[], 
                       edits_anno=[('Attaching Group', '*C(=O)O'), ('Change Bond', 'CC'),'Terminate'], 
                       edits_atom_mapid=[7, [5, 7]], 
                       edits_atom_id=[7, [1, 7]])
    x = rxn.query_mol_smi
    y = rxn.res_mol_smi
    temp = rxn.edits_anno
    idx = rxn.edits_atom_id

    pred_mol = templates_to_reactants(x, temp, idx)
    print(f'idx={idx}\npred_smi={Chem.MolToSmiles(pred_mol)}\ntrue_smi={y}')
    
    print("=="*20)
    
    # 
    # forward
    # rxn = ReactionData(rxn_name=10, 
    #                    rxn_smi='[BH4-:49].[CH2:1]1[CH2:2][CH2:3][O:4][CH2:5]1.[CH2:6]=[O:7].[CH3:8][C:9]([CH3:10])([CH3:11])[O:12][C:13](=[O:14])[c:15]1[cH:16][cH:17][c:18]([NH:19][CH2:20][C:21]2([OH:22])[CH2:23][CH2:24][N:25]([CH2:26][CH2:27][c:28]3[cH:29][cH:30][c:31]([C:32]#[N:33])[cH:34][cH:35]3)[CH2:36][CH2:37]2)[cH:38][cH:39]1.[Na+:50].[Na+:51].[O:40]=[C:41]([OH:42])[OH:43].[O:44]=[S:45](=[O:46])([OH:47])[OH:48]>>[CH3:8][C:9]([CH3:10])([CH3:11])[O:12][C:13](=[O:14])[c:15]1[cH:16][cH:17][c:18]([N:19]([CH2:20][C:21]2([OH:22])[CH2:23][CH2:24][N:25]([CH2:26][CH2:27][c:28]3[cH:29][cH:30][c:31]([C:32]#[N:33])[cH:34][cH:35]3)[CH2:36][CH2:37]2)[CH3:41])[cH:38][cH:39]1', 
    #                    query_mol_smi='C1CCOC1.C=O.CC(C)(C)OC(=O)c1ccc(NCC2(O)CCN(CCc3ccc(C#N)cc3)CC2)cc1.O=C(O)O.O=S(=O)(O)O.[BH4-].[Na+].[Na+]', 
    #                    res_mol_smi='CN(CC1(O)CCN(CCc2ccc(C#N)cc2)CC1)c1ccc(C(=O)OC(C)(C)C)cc1', edits=[], 
    #                    edits_anno=[('Add Bond', 'CN'), ('Leaving Group', '[BH4-]'), ('Leaving Group', 'C1CCOC1'), ('Leaving Group', 'C=O'), ('Leaving Group', '[Na+]'), ('Leaving Group', '[Na+]'), ('Leaving Group', 'O=S(=O)(O)O'), ('Leaving Group', '*=O'), ('Leaving Group', '*O'), ('Leaving Group', '*O'), 'Terminate'], edits_atom_mapid=[[19, 41], 49, 1, 6, 50, 51, 44, 40, 43, 42], edits_atom_id=[[18, 40], 48, 0, 5, 49, 50, 43, 39, 42, 41])

    # x = rxn.query_mol_smi
    # y = rxn.res_mol_smi
    # temp = rxn.edits_anno
    # idx = rxn.edits_atom_id

    # pred_mol = templates_to_reactants(x, temp, idx)
    # print(f'pred_smi={Chem.MolToSmiles(pred_mol)}\ntrue_smi={Chem.MolToSmiles(Chem.MolFromSmiles(y))}')