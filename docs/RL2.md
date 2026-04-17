# 预训练 & 强化学习训练流程完整梳理

---

## 一、全局视角：两阶段训练

```
阶段一: Pretrain (监督学习)          阶段二: RL Fine-tuning (强化学习)
─────────────────────────────        ──────────────────────────────────
有标注数据 (USPTO50K)                 用预训练模型探索环境
Teacher Forcing                       PPO + KL约束
目标: 学会"模仿"正确编辑序列          目标: 学会"策略性"选择最优编辑
约需 50 epochs                        约需 500 updates
```

---

## 二、预训练流程 (Pretrain)

### 2.1 数据流

```
USPTO50K JSON
    │
    ▼
[dataset.py] USPTO50KDataset.__getitem__()
    │  SMILES → PyG图
    │  action_type字符串 → 整数ID
    │  src/tgt=-1 → clamp(0) + valid_mask标记
    │  label → tokenize → [BOS, t1, t2, ..., EOS]
    ▼
[dataset.py] collate_pretrain()
    │  把一个batch里所有样本的所有步骤"展平"
    │  例: batch_size=4, 每样本3步 → 12条训练数据
    ▼
DataLoader → batch字典
    ├── pyg_batch        [图数据]
    ├── target_actions   [12]
    ├── target_srcs      [12]
    ├── target_tgts      [12]
    ├── decoder_inputs   [12, L]  (BOS + label)
    ├── decoder_targets  [12, L]  (label + EOS)
    ├── src_valid_mask   [12]     (Terminate步=False)
    └── tgt_valid_mask   [12]
```

### 2.2 前向传播

```
pyg_batch (x, edge_index, batch)
    │
    ▼
[graph_encoder.py] GraphEncoder.forward()
    │  x: [N_total, 39] → node_proj → [N_total, 256]
    │  4层 GATConv + 残差
    │  global_mean_pool → graph_emb [B, 256]
    │  返回: node_emb [N_total, 256], graph_emb [B, 256]
    │
    ▼
[actor.py] ActorNetwork.encode_graph()
    │  to_dense_batch(node_emb, batch)
    │  → dense_nodes [B, N_max, 256]  (padding补齐)
    │  → node_mask   [B, N_max]       (True=有效节点)
    │  state_proj(graph_emb) → state  [B, 512]
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
[actor.py] ActionTypePredictor              [actor.py] LabelDecoder
    state [B,512] → MLP → [B,7]                state + action_emb → memory
    action_logits                               Transformer Decoder
                                                label_logits [B, L, vocab]
    ▼
[actor.py] PointerNetwork
    state + action_emb → src_logits [B, N_max]
    state + action_emb + src_emb → tgt_logits [B, N_max]
```

### 2.3 损失计算

```
action_logits [B,7]  vs  target_actions [B]
    → CrossEntropy → action_loss

src_logits [B,N]  vs  target_srcs [B]
    → CrossEntropy (reduction='none') → [B]
    → * src_valid_mask → 屏蔽Terminate步
    → .sum() / valid_count → src_loss

tgt_logits [B,N]  vs  target_tgts [B]
    → 同上 → tgt_loss

label_logits [B,L,vocab]  vs  decoder_targets [B,L]
    → CrossEntropy (ignore_index=PAD) → label_loss

total_loss = 1.0*action_loss + 1.0*(src_loss+tgt_loss) + 0.5*label_loss
```

### 2.4 完整预训练代码阅读顺序

```
① config/config.py          → 了解所有超参数和常量定义
② data/mol_utils.py         → 了解原子特征怎么提取 (39维)
③ data/dataset.py           → 了解数据怎么加载和预处理
④ models/graph_encoder.py   → Module1: 图怎么编码
⑤ models/actor.py           → Module3: 三个预测头
⑥ training/pretrain_trainer.py → 损失怎么计算、怎么更新
```

---

## 三、强化学习训练流程 (RL Fine-tuning)

### 3.1 整体循环

```
加载预训练Actor权重
    │
    ▼
注入LoRA + 冻结预训练权重
    │
    ▼
┌─────────────────────────────────────────┐
│              PPO 大循环                  │
│                                         │
│  ┌── Phase 1: Rollout (收集经验) ──┐    │
│  │  用当前策略与环境交互            │    │
│  │  收集 (s,a,r,logp,v) 轨迹      │    │
│  └─────────────────────────────────┘    │
│                 ↓                       │
│  ┌── Phase 2: Update (更新策略) ──┐    │
│  │  计算GAE优势                    │    │
│  │  PPO clip loss                  │    │
│  │  KL散度约束                     │    │
│  │  更新Actor(LoRA) + Critic       │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

### 3.2 Rollout 阶段详细数据流

```
从数据集取一批样本 samples[0..63]
    │
    ▼
[rl_env.py] BatchRetroEnv.reset(samples)
    │  每个样本初始化一个环境
    │  返回 obs_list: [{"graph":..., "history":[], ...}]
    │
    ▼ ← 循环 max_steps=10 次
    │
[policy.py] RetroSynthesisPolicy.sample_action()
    │
    │  ① encode_graph(x, ei, batch)
    │  │   → dense_nodes, pad_mask, graph_state [B,512]
    │  │
    │  ② GRUStateTracker(history_actions, graph_state)
    │  │   → state [B,512]   ← 包含历史信息！
    │  │
    │  ③ action_predictor(state) → act_logits [B,7]
    │  │   Categorical(logits).sample() → action_type [B]
    │  │   log_prob → act_lp [B]
    │  │
    │  ④ pointer_network(state, dense_nodes, action_type)
    │  │   → src_logits [B,N], tgt_logits [B,N]
    │  │   各自 Categorical.sample() → src_idx, tgt_idx
    │  │
    │  ⑤ critic.get_value(x, ei, batch, history)
    │      → value [B,1]
    │
    │  返回: action_type, src_idx, tgt_idx, log_prob, value
    │
    ▼
[rl_env.py] BatchRetroEnv.step(action_dicts)
    │  对比 pred vs GT → match_score
    │  更新 history
    │  返回: obs, gt_info, done
    │
    ▼
[reward.py] RewardCalculator.step_reward()
    │  合法性 + GT匹配 → step_reward
    │  若done: + terminal_reward
    │
    ▼
[rollout_buffer.py] buffer.add(Transition)
    │  存储: (graph_idx, history, action, src, tgt,
    │          log_prob, value, reward, done)
    │
    ▼ (episode结束后)
[rollout_buffer.py] buffer.compute_gae()
    对每条轨迹:
    T=3步: r=[0.3, 0.5, 2.0], v=[0.8, 1.2, 0.9]
    GAE反向计算:
      δ_2 = r_2 + 0 - v_2 = 1.1
      δ_1 = r_1 + γ*v_2 - v_1 = 0.191
      δ_0 = r_0 + γ*v_1 - v_0 = 0.488
      adv_2 = δ_2 = 1.1
      adv_1 = δ_1 + γλ*adv_2 = 1.228
      adv_0 = δ_0 + γλ*adv_1 = 1.703
    returns = adv + v = [2.503, 2.428, 2.0]
```

### 3.3 PPO Update 阶段详细数据流

```
buffer.compute_gae() → 扁平化数据
    actions [N], src [N], tgt [N]
    old_log_probs [N], returns [N], advantages [N] (已标准化)
    │
    ▼ ← 循环 ppo_epochs=4 次
    │
    随机打乱 → mini_batch (size=32)
    │
    ▼
[policy.py] evaluate_actions()
    │
    │  ① encode_graph → dense_nodes, pad_mask, graph_state
    │  ② GRUStateTracker → state
    │  ③ action_predictor → act_logits
    │     Categorical(act_logits).log_prob(mb_actions) → act_lp
    │     Categorical(act_logits).entropy() → entropy
    │  ④ pointer_network(src_idx=mb_src) → src_logits, tgt_logits
    │     各自 log_prob → src_lp, tgt_lp
    │  ⑤ total_lp = act_lp + src_lp + tgt_lp
    │  ⑥ critic(x,ei,batch,action,history) → v [B,1], q [B,1]
    │
    ▼
计算各项损失:

  ratio = exp(new_lp - old_lp)          # 重要性采样比
  surr1 = ratio * advantages
  surr2 = clip(ratio, 0.8, 1.2) * advantages
  policy_loss = -min(surr1, surr2).mean()

  ref_lp = get_ref_log_probs(...)        # 预训练策略的log_prob (no_grad)
  kl_div = (ref_lp - new_lp).mean()     # 与预训练策略的KL散度
  policy_loss += kl_coef * kl_div       # ← 防止策略崩溃的关键

  entropy_loss = -0.01 * entropy.mean() # 鼓励探索

  value_loss = MSE(v, returns)

  actor_loss  = policy_loss + entropy_loss  → 更新 LoRA参数 + StateTracker
  critic_loss = 0.5 * value_loss            → 更新 Critic网络
    │
    ▼
自适应调整 kl_coef:
  avg_kl > 2*kl_target → kl_coef *= 1.5  (收紧约束)
  avg_kl < 0.5*kl_target → kl_coef *= 0.5 (放松约束)
```

### 3.4 完整RL代码阅读顺序

```
① config/config.py          → RL_CONFIG 部分
② models/state_tracker.py   → GRU如何编码历史
③ models/critic.py          → 独立Critic网络结构
④ models/policy.py          → 整合接口: sample/evaluate
⑤ rl/reward.py              → 奖励信号设计
⑥ data/rl_env.py            → 环境交互逻辑
⑦ rl/rollout_buffer.py      → GAE计算
⑧ rl/ppo_trainer.py         → PPO核心更新
```

---

## 四、两阶段对比速查表

| 维度 | Pretrain | RL Fine-tuning |
|------|----------|----------------|
| **数据** | 全部有标注 | 有标注(算奖励用) |
| **目标** | 模仿GT序列 | 最大化累积奖励 |
| **损失** | CE Loss | PPO Clip + KL + Value |
| **Actor更新** | 全参数 | 仅LoRA参数 |
| **Critic** | 无 | 独立网络 |
| **状态** | 无历史(单步) | GRU追踪历史 |
| **batch含义** | 展平的编辑步骤 | 完整轨迹 |
| **src=-1处理** | valid_mask屏蔽loss | env不计算pointer奖励 |
| **关键风险** | 过拟合 | 策略崩溃 |
| **防护手段** | Dropout/WD | KL约束+LoRA |

---

## 五、一个具体样本的完整生命周期

```
样本: product_smi = "CC(=O)c1ccc2c(c1)ccn2C(=O)OC(C)(C)C"
GT:   [DeleteBond(10,11), AttachGroup(11,-1,"*OC..."), Terminate]

━━━━━━━━━━━━━━━━ Pretrain 阶段 ━━━━━━━━━━━━━━━━

Step0 输入:
  graph(product) + target_action=0(DeleteBond) + target_src=10 + target_tgt=11
  decoder_input=[BOS, NONE_ID]
  → 计算 action/src/tgt/label loss → 反向传播

Step1 输入:
  graph(product) + target_action=4(AttachGroup) + target_src=11 + target_tgt=0(clamp)
  src_valid=True, tgt_valid=False
  decoder_input=[BOS, *OC...tokens]
  → src_loss计算, tgt_loss被mask屏蔽

Step2 输入:
  graph(product) + target_action=6(Terminate) + target_src=0 + target_tgt=0
  src_valid=False, tgt_valid=False
  → 只计算action_loss, pointer_loss全部屏蔽

━━━━━━━━━━━━━━━━ RL 阶段 ━━━━━━━━━━━━━━━━

t=0: state=graph_emb, history=[]
     → sample: action=0(DeleteBond), src=10, tgt=11
     → env: match_score=1.0, reward=0.5+0.15+0.15-0.02=0.78
     → buffer: Transition(action=0, src=10, tgt=11, lp=-0.3, v=2.1, r=0.78)

t=1: state=GRU([0], graph_emb), history=[0]
     → sample: action=4(AttachGroup), src=11, tgt=0
     → env: match_score=0.75, reward=0.5+0.15-0.02=0.63
     → buffer: Transition(...)

t=2: state=GRU([0,4], graph_emb), history=[0,4]
     → sample: action=6(Terminate)
     → env: done=True, terminal_reward=10*0.9+2.0=11.0
     → buffer: Transition(..., done=True)

→ GAE计算 → PPO更新 → LoRA权重微调
```