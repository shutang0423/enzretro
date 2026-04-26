#!/usr/bin/env python3
"""
测试单个分子示例和遍历测试集

加载训练好的模型，对单个分子进行推理，生成编辑序列
同时支持遍历测试集，找到推理正确的结果并展示
"""

import torch
import json
from pathlib import Path

from config.config import PATH_CFG, MODEL_CFG, ACTION_TYPES, ID_TO_ACTION, ACTION_TO_ID, RLInferenceConfig, STOP_ACTION_ID
from model.actor_network import ActorNetwork
from utils.chem import smiles_to_graph
from tokenizer.tokenizer import LabelTokenizer
from rl.rl_inference import RLInferenceConfig, RLInference
from rl.reward_calculator import RewardCalculator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def test_single_example(test_smiles: str = 'CCOC(=O)C1CCC2(CC1)OCCO2',
                        checkpoint_path: Path = PATH_CFG.CKPT_BEST_MODEL_FILE,
                        inference_method: str = "monte_carlo",
                        reward_method: str = "step_comparison"):
    
    print(f"Testing molecule: {test_smiles}")
    
    # 1. 加载分词器
    tokenizer = LabelTokenizer(str(PATH_CFG.VOCAB_FILE))
    vocab_size = tokenizer.vocab_size
    
    # 2. 构建模型
    model = ActorNetwork(vocab_size=vocab_size).to(device)
    
    # 3. 加载训练好的模型
    # checkpoint_path = PATH_CFG.CKPT_BEST_MODEL_FILE
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 5. 转换为图表示
    x, edge_index, edge_attr = smiles_to_graph(test_smiles)
    if x is None:
        print("Error: Failed to parse SMILES")
        return
    
    # 6. 准备输入数据
    # 添加batch维度
    x = x.to(device)  # [N, 79]
    edge_index = edge_index.to(device)  # [2, E]
    edge_attr = edge_attr.to(device)  # [E, 12]
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)  # [N]
    
    # 7. 调用模型进行推理
    encoder_kwargs = {
        'x': x,
        'edge_index': edge_index.squeeze(0),  # GAT encoder expects [2, E]
        'batch': batch
    }

    # 初始化强化学习推理器
    rl_config = RLInferenceConfig(inference_method=inference_method)
    rl_inference = RLInference(model, config=rl_config)
    edit_steps = rl_inference.infer(encoder_kwargs, target_steps=None)
    
    
    # with torch.no_grad():
    #     edit_steps = model.generate(**encoder_kwargs)
    
    # 8. 处理输出结果
    print("\nGenerated edit steps:")
    for i, step in enumerate(edit_steps):
        action_type = step.action_type.item()
        # 转换动作类型
        action_name = ID_TO_ACTION[action_type]

        if action_type == STOP_ACTION_ID:
            src_idx, tgt_idx, label_str = None, None, None  # Terminate
        else:
            src_idx = step.src_idx.item()
            tgt_idx = step.tgt_idx.item()
        
            # 解码标签序列
            label_tokens = step.label_tokens.squeeze(0).tolist()
            label_str = tokenizer.decode(label_tokens, skip_special_tokens=True)
            
        
        print(f"Step {i+1}:")
        print(f"  Action: {action_name} (id={action_type})")
        print(f"  Source atom: {src_idx}")
        print(f"  Target atom: {tgt_idx}")
        print(f"  Label: {label_str}")
        print()
        
        if action_type == 4:  # Terminate
            print("Termination action detected, stopping generation")
            break

# def test_more():
#     """
#     遍历测试集，找到推理正确的结果并展示
#     """
#     # 1. 加载分词器
#     tokenizer = LabelTokenizer(str(PATH_CFG.VOCAB_FILE))
#     vocab_size = tokenizer.vocab_size
    
#     # 2. 构建模型
#     model = ActorNetwork(vocab_size=vocab_size).to(device)
    
#     # 3. 加载训练好的模型
#     checkpoint_path = PATH_CFG.CKPT_BEST_MODEL_FILE
#     if not checkpoint_path.exists():
#         print(f"Error: Checkpoint file not found at {checkpoint_path}")
#         return
    
#     print(f"Loading checkpoint from {checkpoint_path}")
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(ckpt['model'])
#     model.eval()
    
#     # 4. 加载测试集数据
#     test_data_path = PATH_CFG.PRETRAIN_TEST_DATA_FILE
#     if not test_data_path.exists():
#         print(f"Error: Test data file not found at {test_data_path}")
#         return
    
#     print(f"Loading test data from {test_data_path}")
#     with open(test_data_path, 'r', encoding='utf-8') as f:
#         test_data = json.load(f)
    
#     print(f"Loaded {len(test_data)} test samples")
    
#     # 5. 遍历测试集
#     correct_count = 0
#     total_count = 0
    
#     for i, sample in enumerate(test_data):
#         if i >= 100:  # 限制测试数量，避免运行时间过长
#             break
        
#         product_smi = sample.get('product_smi')
#         target_action = sample.get('target_action_type')
#         target_src = sample.get('target_src_idx')
#         target_tgt = sample.get('target_tgt_idx')
#         target_label = sample.get('target_label')
        
#         if not product_smi or target_action is None:
#             continue
        
#         total_count += 1
        
#         # 转换为图表示
#         x, edge_index, edge_attr = smiles_to_graph(product_smi)
#         if x is None:
#             continue
        
#         # 准备输入数据
#         x = x.to(device)  # [N, 79]
#         edge_index = edge_index.to(device)  # [2, E]
#         edge_attr = edge_attr.to(device)  # [E, 12]
#         batch = torch.zeros(x.size(0), dtype=torch.long, device=device)  # [N]
        
#         # 调用模型进行推理
#         encoder_kwargs = {
#             'x': x,
#             'edge_index': edge_index.squeeze(0),  # GAT encoder expects [2, E]
#             'batch': batch
#         }
        
#         with torch.no_grad():
#             edit_steps = model.generate(**encoder_kwargs)
        
#         # 检查第一步的预测是否正确
#         if edit_steps:
#             first_step = edit_steps[0]
#             pred_action = first_step.action_type.item()
#             pred_src = first_step.src_idx.item()
#             pred_tgt = first_step.tgt_idx.item()
            
#             # 解码标签序列
#             label_tokens = first_step.label_tokens.squeeze(0).tolist()
#             pred_label = tokenizer.decode(label_tokens, skip_special_tokens=True)
            
#             # 比较预测结果与目标值
#             action_correct = (pred_action == int(target_action))
#             src_correct = (pred_src == target_src) if target_src is not None else True
#             tgt_correct = (pred_tgt == target_tgt) if target_tgt is not None else True
#             label_correct = (pred_label == target_label) if target_label else True
            
#             if action_correct and src_correct and tgt_correct and label_correct:
#                 correct_count += 1
#                 print(f"\nCorrect prediction found at sample {i}:")
#                 print(f"Product SMILES: {product_smi}")
#                 print(f"Predicted action: {ID_TO_ACTION[pred_action]} (id={pred_action})")
#                 print(f"Target action: {ID_TO_ACTION[int(target_action)]} (id={target_action})")
#                 print(f"Predicted src: {pred_src}, Target src: {target_src}")
#                 print(f"Predicted tgt: {pred_tgt}, Target tgt: {target_tgt}")
#                 print(f"Predicted label: {pred_label}")
#                 print(f"Target label: {target_label}")
#                 print("=" * 80)
    
#     print(f"\nTest completed: {correct_count} correct out of {total_count} samples")
#     print(f"Accuracy: {correct_count / total_count * 100:.2f}%")


if __name__ == "__main__":
    print("1. Testing single example")
    smi = "CCOc1ccc(Oc2ncnc3c2cnn3C2CCN(C(=O)OC3CCCC3)CC2)c(F)c1"
    ckpt_path = Path("ckpt/pretrain_20260426_gat_uncertainty/best_model.pt")
    inference_method = "monte_carlo"
    test_single_example(smi, ckpt_path, inference_method=inference_method)
    
    print("\n" + "=" * 80 + "\n")
    
    # print("2. Testing test set")
    # test_more()

