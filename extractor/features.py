'''
Author: Caoyh
Date: 2025-06-29 13:18:14
LastEditors: BellaCaoyh caoyh_cyh@163.com
LastEditTime: 2025-06-29 13:18:17
'''
from rdkit import Chem
from rdkit.Chem import Mol, rdchem
from collections import namedtuple

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}
# CHIRAL_DICT = {str(chi): chi for i, chi in rdchem.ChiralType.values.items()}
BOND_TYPE_DICT = {str(t): t for i, t in rdchem.BondType.values.items()}
BOND_STEREO_DICT = {str(s): s for i, s in rdchem.BondStereo.values.items()}

def chiral2string(atom_type, chiral_tag):
    if rdchem.ChiralType.values[chiral_tag] == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return atom_type+'@@'
    elif rdchem.ChiralType.values[chiral_tag] == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return atom_type+'@'
    else:
        return atom_type 
    
def string2chiral(atom_string):
    """
    The tag "CHI_TETRAHEDRAL_CW" means clockwise and CCW means anti-clockwise
    The symbol "@" indicates that the following neighbors are listed anticlockwise (CCW)
    The symbol "@@" indicates that the neighbors are listed clockwise (CW)
    """
    if atom_string.endswith('@@'):
        atom_type = atom_string[:-2]
        chiral_tag = int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif atom_string.endswith('@'):
        atom_type = atom_string[:-1]
        chiral_tag = int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
    else:
        atom_type = atom_string
        chiral_tag = int(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    return atom_type, chiral_tag



