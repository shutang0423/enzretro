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



