import torch
from model.actor_pretrainer import ActorPretrainer
from tokenizer.tokenizer import LabelTokenizer
from config.config import MODEL_CONFIG, PATH_CONFIG

C  = MODEL_CONFIG
PC = PATH_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 tokenizer 和模型结构
tokenizer  = LabelTokenizer(vocab_file=PC["vocab_file"])

from data.dataset import USPTO50KDataset
json_path = "dataset/uspto50k/processed/uspto50k_test_output.json"
dataset = USPTO50KDataset(json_path, tokenizer)
print(dataset[0])



