import torch 
import torch.nn as nn
import torch.optim as optim
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_pretrainer import ActorPretrainer, SimpleStateTracker


# 2. 初始化模型、优化器和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vocab = 'tokenizer/vocab.txt'
tokenizer = LabelTokenizer(vocab_file=vocab)

dataset = SSRGraphDataset(
    json_path='dataset/uspto50k/pretrained/uspto50k_test_output.json',
    tokenizer=tokenizer,
)
# data = dataset[1]
# print(data)
# print(data.target_action)
# print(data.history_actions)


from torch_geometric.loader import DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# 3. 初始化模型 (传入 tokenizer 的真实词表大小)
vocab_size = tokenizer.get_vocab_size()
model = ActorPretrainer(vocab_size=vocab_size).to(device)
state_tracker = SimpleStateTracker(hidden_dim=512).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(state_tracker.parameters()), lr=1e-4)

# 4. Loss 函数使用真实的 pad_token_id
criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
pad_id = tokenizer.pad_token_id
criterion_seq = nn.CrossEntropyLoss(ignore_index=pad_id)

# 3. 开始训练
model.train()
state_tracker.train()

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # --- A. 提取图特征 ---
        node_embeddings, graph_embedding = model.graph_encoder(batch.x, batch.edge_index, batch.batch)
        graph_state = model.state_proj(graph_embedding)
        
        # --- B. 融合历史状态 ---
        # batch.history_actions 形状为 [B, max_hist_len]
        decoder_state = state_tracker(batch.history_actions, graph_state)
        
        # --- C. 准备 Target 序列 ---
        # batch.target_label 形状为 [B, max_seq_len]
        decoder_input_seq = batch.target_label[:, :-1] 
        label_target_seq = batch.target_label[:, 1:]   
        
        # --- D. 并行预测 (Teacher Forcing) ---
        action_logits, src_logits, tgt_logits, label_logits = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            batch=batch.batch, 
            target_action=batch.target_action, 
            target_src=batch.target_src, 
            decoder_input_seq=decoder_input_seq,
            history_state=decoder_state # 传入融合后的状态
        )
        
        # --- E. 计算损失 ---
        loss_action = criterion_cls(action_logits, batch.target_action)
        loss_src = criterion_cls(src_logits, batch.target_src)
        loss_tgt = criterion_cls(tgt_logits, batch.target_tgt)
        loss_label = criterion_seq(
            label_logits.reshape(-1, label_logits.size(-1)), 
            label_target_seq.reshape(-1)
        )
        
        loss = loss_action + loss_src + loss_tgt + loss_label
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}] Loss: {loss.item():.4f} "
                    f"(Act:{loss_action.item():.2f}, Src:{loss_src.item():.2f}, "
                    f"Tgt:{loss_tgt.item():.2f}, Lbl:{loss_label.item():.2f})")
            
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")



