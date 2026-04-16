import os
import re 
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO
from multiprocessing import Process, Queue
from typing import Dict, Tuple, List, Union
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Mol, RWMol, rdchem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs 

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1,
             'Cl': 1, 'F': 1, 'I': 1, 'Li': 1, 'Na': 1, 'K': 1}

# ── 原子特征定义 (共39维) ──────────────────────────────────────
ATOM_TYPES = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca',
              'Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn',
              'Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au',
              'Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','<unk>']
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(val, choices: list) -> list:
    return [1 if val == c else 0 for c in choices] + [0 if val in choices else 1]


def atom_features(atom) -> list:
    """提取单个原子特征 (39维)"""
    return (
        one_hot(atom.GetSymbol(), ATOM_TYPES[:-1])          # 44维 → 截断到43+1
        + one_hot(atom.GetDegree(), [0,1,2,3,4,5])          # 7维
        + one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])        # 6维
        + one_hot(atom.GetImplicitValence(), [0,1,2,3,4,5]) # 7维
        + one_hot(atom.GetHybridization(), HYBRIDIZATION)   # 6维
        + [atom.GetIsAromatic()]                             # 1维
        + [atom.GetFormalCharge()]                           # 1维
        + [atom.IsInRing()]                                  # 1维
    )
    # 注意: 实际维度由上面累加决定，需与 node_in_dim 对齐
    # 建议在 config 中设置 node_in_dim = get_atom_feat_dim()


def get_atom_feat_dim() -> int:
    """动态计算原子特征维度"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C")
    return len(atom_features(mol.GetAtomWithIdx(0)))


def bond_features(bond) -> list:
    """提取键特征 (6维)"""
    bt = bond.GetBondType()
    return (
        one_hot(bt, BOND_TYPES[:-1])   # 4维
        + [bond.GetIsConjugated()]      # 1维
        + [bond.IsInRing()]             # 1维
    )

def smiles_to_pyg(smiles: str, with_hydrogen: bool = False) -> Optional[Data]:
    """SMILES → PyG Data

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if with_hydrogen:
        mol = Chem.AddHs(mol)

    # 原子特征
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)  # [N, node_in_dim]

    # 键 (双向)
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 6), dtype=torch.float)
    else:
        src, dst, eattr = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            src += [i, j]; dst += [j, i]
            eattr += [bf, bf]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=mol.GetNumAtoms())

def make_sentence_to_string(rules):
    rules_data = list(map(lambda rule: ", ".join([" ".join(r) for r in eval(rule)]), rules))
    return rules_data


def frag_is_a_mol(frag_smi:str, react_smi:str)->bool:
    reacts = react_smi.strip().split(".")
    reacts_can = [Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reacts]
    if Chem.MolToSmiles(Chem.MolFromSmiles(frag_smi)) in reacts_can:
        return True
    else:
        return False

def map2idx(smi_with_mapped_id:str) -> Dict:
    orig_mol = Chem.MolFromSmiles(smi_with_mapped_id)
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in orig_mol.GetAtoms()}
    can_mol = Chem.MolFromSmiles(canonicalize_mol_smi(smi_with_mapped_id))
    matches = list(can_mol.GetSubstructMatches(orig_mol))

    map2idx = {}
    if matches:
        for idx, match_idx in enumerate(list(matches)[0]):
            # match_anum = can_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
            match_anum = match_idx
            old_anum = idx_amap[idx]
            map2idx[old_anum] = match_anum
    
    return map2idx 

def get_bond_substructure(atom_map1, atom_map2, ref_mol):
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in ref_mol.GetAtoms()
                if atom.GetAtomMapNum() != 0}
    atom1 = ref_mol.GetAtomWithIdx(amap_idx[atom_map1])
    atom2 = ref_mol.GetAtomWithIdx(amap_idx[atom_map2])
    bond = ref_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())

    bond_mol = Chem.RWMol()
    bond_mol.AddAtom(Chem.Atom(atom1.GetSymbol())) 
    bond_mol.AddAtom(Chem.Atom(atom2.GetSymbol())) 
    bond_mol.AddBond(0, 1, order=bond.GetBondType()) 

    smi = Chem.MolToSmiles(bond_mol)
    return smi  

def get_atom_info(mol: Mol) -> Dict:
    if mol is None:
        return {}

    atom_info = {}
    for atom in mol.GetAtoms():
        # feat = [atom.GetNumExplicitHs(), int(atom.GetChiralTag())]
        feat = [int(atom.GetChiralTag())] # 原子的立体信息
        amap_num = atom.GetAtomMapNum() 
        atom_info[amap_num] = tuple(feat)
    return atom_info  

def get_bond_info(mol: Mol) -> Dict:
    if mol is None:
        return {}
    
    Chem.Kekulize(mol)
    
    bond_info = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        a1_symbol, a2_symbol = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
        
        # is_r = int(bond.IsInRing())
        bt = int(bond.GetBondType())
        st = int(bond.GetStereo())
        
        bond_atoms = sorted([a1, a2])
        bond_info[tuple(bond_atoms)] = [bt, st, a1_symbol, a2_symbol]
    return bond_info


def get_atom_Chiral(mol: Mol) -> Dict:
    if mol is None:
        return {}

    atom_Chiral = {}
    for atom in mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        atom_Chiral[amap_num] = atom.GetChiralTag()
    return atom_Chiral

def get_bond_stereo(mol: Mol) -> Dict:
    if mol is None:
        return {}

    bond_stereo = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        bond_atoms = sorted([a1, a2])
        bond_stereo[tuple(bond_atoms)] = bond.GetStereo()
    return bond_stereo


def align_kekulize_pairs(r_mol: Mol, p_mol: Mol) -> Tuple[Mol, Mol]:
    prod_old = get_bond_info(p_mol)
    Chem.Kekulize(p_mol)
    prod_new = get_bond_info(p_mol)

    react_old = get_bond_info(r_mol)
    Chem.Kekulize(r_mol)
    react_new = get_bond_info(r_mol)

    r_mol = Chem.RWMol(r_mol)
    r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                  for atom in r_mol.GetAtoms()}
    for bond in prod_new:
        if bond in react_new and (react_old[bond][0] == prod_old[bond][0]) and (react_new[bond][0] != prod_new[bond][0]):
            idx1, idx2 = r_amap_idx[bond[0]], r_amap_idx[bond[1]]
            bt = prod_new[bond][0]
            b_type = rdchem.BondType.values[bt]
            r_mol.GetBondBetweenAtoms(idx1, idx2).SetBondType(b_type)

    return r_mol.GetMol(), p_mol


def get_atom_idx(mol: RWMol or Mol, atom_map: int) -> int: # type: ignore
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomMapNum() == atom_map:
            return i
    raise ValueError(f'No atom with map number: {atom_map}')


def fix_charge(mol):
    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    for atom in mol.GetAtoms():
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                        for b in atom.GetBonds()]))
        if atom.GetSymbol() == 'O' and bond_vals == 1 and charge == -1 and explicit_hs == 0:
            if atom.GetNeighbors()[0].GetSymbol() != 'N':
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'N' and bond_vals == 1 and charge == 1 and explicit_hs == 3:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(2)

        if atom.GetSymbol() == 'N' and bond_vals == 0 and charge == 1 and explicit_hs == 4:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(3)

        if atom.GetSymbol() == 'N' and bond_vals == 2 and charge == 1 and explicit_hs == 2:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'S' and charge == -1 and explicit_hs == 0 and bond_vals == 1:
            atom.SetNumExplicitHs(1)
            atom.SetFormalCharge(0)
    return mol


def fix_Hs_Charge(mol: Mol, atom_map_ids:List=[]) -> Mol:
    # fix explicit Hs and charge
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()

        if not atom.IsInRing():
            atom.SetIsAromatic(False)
            for b in atom.GetBonds(): 
                b.SetIsAromatic(False)
                if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    b.SetBondType(Chem.rdchem.BondType.SINGLE)
                    
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                             for b in atom.GetBonds()])) 

        if charge == 0:
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals > MAX_BONDS[atom_symbol]:
                num = int(explicit_hs + bond_vals - MAX_BONDS[atom_symbol])
                for i in range(num):
                    if explicit_hs > 0:
                        explicit_hs -= 1
                        atom.SetNumExplicitHs(explicit_hs)
                    else:
                        atom.SetFormalCharge(1)

            elif atom_symbol in MAX_BONDS and explicit_hs + bond_vals < MAX_BONDS[atom_symbol]:
                num = int(MAX_BONDS[atom_symbol] - explicit_hs - bond_vals)
                for i in range(num):
                    explicit_hs += 1
                    atom.SetNumExplicitHs(explicit_hs)

            # "-N=N+=N-"
            if atom_symbol == 'N' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 2 and atom.GetNeighbors()[0].GetSymbol() == 'N':
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(-1)
            
            # "-CNC-"
            # if atom_symbol == 'N' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 2 and bond_vals == 3 and atom.GetNeighbors()[0].GetSymbol() == 'C' and atom.IsInRingSize(5) and atom.GetAtomMapNum() in atom_map_ids:
            #     # rinfo = mol.GetRingInfo()
            #     # rings = rinfo.AtomRings()
            #     # atom_idx = atom.GetIdx()
            #     atom.SetNumExplicitHs(1)
                # atom.SetFormalCharge(0)

            # "NC-"
            if atom_symbol == 'C' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 3 and atom.GetNeighbors()[0].GetSymbol() == 'N':
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(-1)

            if atom_symbol == 'S' and explicit_hs == 0 and bond_vals == 1:
                atom.SetNumExplicitHs(1)

            if atom_symbol == 'S' and explicit_hs == 1 and bond_vals in [2, 4, 6]:
                atom.SetNumExplicitHs(0)

            if atom_symbol == 'P':  # 'P(OCC)3'
                bond_vals = [bond.GetBondTypeAsDouble()
                             for bond in atom.GetBonds()]
                if sum(bond_vals) == 3 and len(bond_vals) == 3:
                    atom.SetNumExplicitHs(0)
                if sum(bond_vals) == 4 and len(bond_vals) == 4:
                    atom.SetFormalCharge(1)

            if atom_symbol == 'Sn':
                if explicit_hs == 0 and bond_vals == 3:
                    atom.SetNumExplicitHs(1)
                if explicit_hs == 1 and bond_vals == 4:
                    atom.SetNumExplicitHs(0)

        else:
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals == MAX_BONDS[atom_symbol]:
                atom.SetFormalCharge(0)

            if atom_symbol == 'O':
                bond_vals = bond_vals + explicit_hs
                if bond_vals == 1 and charge == -1 and atom.GetNeighbors()[0].GetSymbol() != 'N':
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(1)

            if atom_symbol == 'N':
                if bond_vals == 4 and explicit_hs == 0 and charge == -1:
                    atom.SetFormalCharge(1)
                if bond_vals == 3 and explicit_hs == 1 and charge == -1:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)
                if bond_vals == 3 and explicit_hs == 2 and charge == 1:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)

    for atom in mol.GetAtoms():  # Dealing with the problem 'C+'
        if atom.GetSymbol() == 'C' and atom.GetFormalCharge() == 1:
            atom.SetFormalCharge(0)

    return mol


def get_substruct_match(q, mol, patt):
    q.put(mol.GetSubstructMatches(patt))

def get_matches(mol, patt):
    q = Queue()
    p = Process(target=get_substruct_match, args=(q, mol, patt))

    p.start()
    p.join(timeout=5.0)
    res = None
    if p.exitcode is None:
        p.terminate()
    else:
        res = q.get()
    p.terminate()
    return res
    
def infer_correspondence(p):
    orig_mol = Chem.MolFromSmiles(p)
    canon_smi, _ = canonicalize_prod(p)
    canon_mol = Chem.MolFromSmiles(canon_smi)
    matches = list(canon_mol.GetSubstructMatches(orig_mol))
    # matches = get_matches(canon_mol, orig_mol)
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
                for atom in orig_mol.GetAtoms()}

    correspondence = {}
    if matches:
        for idx, match_idx in enumerate(list(matches)[0]):
            match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
            old_anum = idx_amap[idx]
            correspondence[old_anum] = match_anum
    return correspondence


def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize_mol_smi(p)
    p_mol = Chem.MolFromSmiles(p)
    amap_idx = {}
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
        amap_idx[atom.GetIdx() + 1] = atom.GetIdx()
    p = Chem.MolToSmiles(p_mol)
    return p, amap_idx

def remap_rxn_smi_r(rxn_smi):
    r, p = rxn_smi.split(">>")
    r_canon_smi, r_amap_idx = canonicalize_prod(r)
    canon_mol = Chem.MolFromSmiles(r_canon_smi)
    correspondence = infer_correspondence(r)

    pmol = Chem.MolFromSmiles(p)
    if pmol is None or pmol.GetNumAtoms() <= 1:
        return rxn_smi, None

    flag = len(correspondence)
    for atom in pmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)
        else:
            flag += 1
            newatomnum = flag
            atom.SetAtomMapNum(newatomnum)

    max_amap = max([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
    for atom in pmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    pmol = fix_charge(pmol)
    canon_mol = fix_charge(canon_mol)

    pmol = Chem.MolFromSmiles(Chem.MolToSmiles(pmol))
    rxn_smi_new = Chem.MolToSmiles(canon_mol) + ">>" + Chem.MolToSmiles(pmol)
    return rxn_smi_new, r_amap_idx

def remap_rxn_smi_p(rxn_smi):
    """
    Canonicalize the product SMILES, and then use substructure matching to infer
    the correspondence to the original atom-mapped order. This correspondence is then
    used to renumber the reactant atoms.
    Product SMILES 标准化，然后使用子结构匹配推断与原始原子映射顺序的对应关系。然后利用这种对应关系对反应物原子重新编号。
    """
    r, p = rxn_smi.split(">>")
    p_canon_smi, p_amap_idx = canonicalize_prod(p)
    canon_mol = Chem.MolFromSmiles(p_canon_smi)
    correspondence = infer_correspondence(p)

    rmol = Chem.MolFromSmiles(r)
    if rmol is None or rmol.GetNumAtoms() <= 1:
        return rxn_smi, None

    flag = len(correspondence)
    for atom in rmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)
        else:
            flag += 1
            newatomnum = flag
            atom.SetAtomMapNum(newatomnum)

            
    max_amap = max([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])
    for atom in rmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    rmol = fix_charge(rmol)
    canon_mol = fix_charge(canon_mol)

    rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol))
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(canon_mol)
    return rxn_smi_new, p_amap_idx
    


class NotCanonicalizableSmilesException(ValueError):
    pass
def canonicalize_mol_smi(smi, 
                         remove_atom_mapping=True, 
                         isomericSmiles=False,
                         fix_Hs=True):
    r"""
    Canonicalize SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise NotCanonicalizableSmilesException("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    if fix_Hs:
        mol = fix_Hs_Charge(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)

def canonicalize_rxn_smi(smi, remove_atom_mapping=False, isomericSmiles=False, remove_H=True):
    smi = smi.replace("?", "")
    smi = smi.replace("/", "")
    smi = smi.replace("\\", "")

    reacts, prod = smi.split(">>")
    if remove_H:
        def rmH(smi):
            if re.match('^\[H\+\:\d+\]$', smi):
                return False
            else: return True
        reacts = filter(rmH, reacts.split("."))

    react_can_lst = [canonicalize_mol_smi(rsmi, remove_atom_mapping=remove_atom_mapping, isomericSmiles=isomericSmiles) for rsmi in reacts]
    react_can_smi = ".".join(react_can_lst)
    prod_can_lst = [canonicalize_mol_smi(psmi, remove_atom_mapping=remove_atom_mapping, isomericSmiles=isomericSmiles) for psmi in prod.split(".")]
    prod_can_smi = ".".join(prod_can_lst)

    return f"{react_can_smi}>>{prod_can_smi}"


def similarity_calculator(query_smile, target_smiles, fp_type='maccs')-> Union[List, float]: 
    
    """ 计算分子之间的相似性
    query_smile: one molecule
    target_smiles: list, a set of molecules
    """
    query_mol = Chem.MolFromSmiles(canonicalize_mol_smi(query_smile))
    if type(target_smiles) is list:
        target_mols = [Chem.MolFromSmiles(canonicalize_mol_smi(smi)) for smi in target_smiles]
    else: target_mols = [canonicalize_mol_smi(target_smiles)]

    if fp_type=='rdkit':# 
        query_fp = Chem.RDKFingerprint(query_mol)
        target_fps = [Chem.RDKFingerprint(mol) for mol in target_mols]
    elif fp_type=='maccs': # 
        query_fp = MACCSkeys.GenMACCSKeys(query_mol)
        target_fps = [MACCSkeys.GenMACCSKeys(mol) for mol in target_mols]
    elif fp_type=='morgen': # circular FP
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
        target_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in target_mols]

    similarities = DataStructs.BulkTanimotoSimilarity(query_fp,target_fps)

    # if len(similarities)==1: 
    #     return similarities[0]
    # else:
    #     return similarities
    
    return similarities



def show_mol(d2d,mol,legend='',highlightAtoms=[]):
    d2d.DrawMolecule(mol,legend=legend, highlightAtoms=highlightAtoms)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    return Image.open(bio)

def show_images(imgs,buffer=5):
    height = 0
    width = 0
    for img in imgs:
        height = max(height,img.height)
        width += img.width
    width += buffer*(len(imgs)-1)
    res = Image.new("RGBA",(width,height))
    x = 0
    for img in imgs:
        res.paste(img,(x,0))
        x += img.width + buffer
    return res

def show_mol_svg(smile, 
                 save_file_name='',
                 save_dir='.',
                 addAtomIndices = True, 
                 addBondIndices = False,
                 height=350 ,
                 width=300,
                 ):
    d = Draw.MolDraw2DSVG(height, width)
    do = d.drawOptions()
    do.addAtomIndices = addAtomIndices
    do.addBondIndices = addBondIndices
    do.addStereoAnnotation = True
    do.explicitMethyl = True
    do.annotationFontScale = 0.8
    # do.bondLineWidth = 2   
    # d.DrawMolecule(mol)
    d.DrawMolecule(Chem.MolFromSmiles(smile))
    d.FinishDrawing()
    img = d.GetDrawingText()

    if save_file_name:
        with open(os.path.join(save_dir, save_file_name), 'w') as f:
            f.write(img)
        return 
    return img


def show_rxn_svg(rxn_mapped_smi, 
                 save_file_name='',
                 save_dir='.',
                 addAtomIndices = False, 
                 addBondIndices = False,
                 height = 300,
                 width = 1000):
    d = Draw.MolDraw2DSVG(width, height)
    do = d.drawOptions()
    do.addAtomIndices = addAtomIndices
    do.addBondIndices = addBondIndices
    do.addStereoAnnotation = True
    do.explicitMethyl = True
    do.annotationFontScale = 0.8
    # do.bondLineWidth = 2  
    rxn = rdChemReactions.ReactionFromSmarts(rxn_mapped_smi, useSmiles=True)
    d.DrawReaction(rxn)
    # d.DrawMolecule(Chem.MolFromSmiles(react))
    d.FinishDrawing()
    img = d.GetDrawingText()

    if save_file_name:
        with open(os.path.join(save_dir, save_file_name), 'w') as f:
            f.write(img)
        return 
    return img
