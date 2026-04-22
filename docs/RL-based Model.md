# 强化学习 SSR 模型方案说明

## 一、整体架构概览

### 1.1 模型流程图

```
输入: Product Graph (产物分子图)
  ↓
【Module 1: Graph Encoder】
  - 编码分子图结构
  - 输出: 节点嵌入 + 图嵌入
  ↓
【Module 2: State Tracker】
  - 维护历史编辑序列
  - 输出: 当前状态表示
  ↓
【Module 3: Actor Network】(策略网络)
  ├─ 预测 action_type (7种动作)
  ├─ 预测 src_idx (源节点)
  ├─ 预测 tgt_idx (目标节点)
  └─ 预测 label (标签序列)
  ↓
【Module 4: Critic Network】(价值网络)
  - 评估状态价值 V(s)
  ↓
【Module 5: Q-Network】(动作价值网络)
  - 评估动作价值 Q(s,a)
  ↓
输出: Edit Sequence (编辑序列)
```

---

后续强化学习中需要微调，是否把Actor 冻结 + LoRA 微调：保护预训练知识实现更好？

✅ 推荐方案：LoRA + KL约束

具体配置：
  冻结：GraphEncoder（纯化学知识）
  LoRA：StateTracker + 三大预测头（策略适应）
  KL约束：β = 0.1~0.5（超参，需要调）
  LoRA rank：r = 8~16（化学任务复杂度中等）
  LoRA alpha：α = 16~32（通常 α = 2r）

训练流程：
  Phase 1: 监督预训练（当前已完成）
  Phase 2: LoRA + PPO（冻结主干，只更新 LoRA 参数）
  Phase 3: 可选 - 解冻部分层全参数微调（如效果不够）

阶段	训练内容	可训练参数	保存内容
Phase 1 监督预训练	全参数	全部	best_model.pt（全量）
Phase 2 LoRA+PPO	LoRA增量 + Critic	LoRA(~2%) + Critic	lora_best.pt + critic_best.pt
Phase 3（可选）全参微调	解冻部分层	LoRA merge + 指定层	finetuned_model.pt（全量）


---
## 二、Module 1: Graph Encoder（图编码器）

### 2.1 功能定位

将产物分子图编码为数值表示，为后续决策提供基础。

### 2.2 输入输出

**输入**：
- **节点特征**：每个原子的属性
  - 原子类型 (C, N, O, S...)
  - 形式电荷、杂化类型、芳香性
  - 氢原子数、手性
  - 维度：[num_nodes, 128]

- **边特征**：每个化学键的属性
  - 键类型 (单键、双键、三键、芳香键)
  - 立体化学信息
  - 维度：[num_edges, 16]

- **图结构**：邻接关系
  - Edge Index: [2, num_edges]

**输出**：
- **节点嵌入**：每个原子的向量表示
  - 维度：[num_nodes, 256]
  - 包含了原子的局部和全局信息

- **图嵌入**：整个分子的向量表示
  - 维度：[batch_size, 256]
  - 通过全局池化得到

### 2.3 核心技术

**图注意力网络 (GAT)**：
- **4层堆叠**：逐层聚合邻居信息
- **多头注意力**：4个注意力头，从不同角度关注邻居
- **残差连接**：防止梯度消失
- **Layer Normalization**：稳定训练

**工作原理**：
```
第1层: 每个原子关注其直接相连的邻居
第2层: 每个原子关注2步内的邻居
第3层: 每个原子关注3步内的邻居
第4层: 每个原子关注4步内的邻居

最终: 每个原子的表示包含了其周围的化学环境信息
```

**注意力机制的优势**：
- 自动学习哪些邻居更重要
- 例如：羰基碳会更关注与其相连的氧原子
- 比固定权重的图卷积更灵活

---

## 三、Module 2: State Tracker（状态追踪器）

### 3.1 功能定位

维护和编码历史编辑序列，生成当前状态的综合表示。

### 3.2 输入输出

**输入**：
- **历史编辑嵌入**：已执行的编辑序列
  - 维度：[batch, seq_len, 512]
  - 例如：已执行2步 → seq_len=2

- **节点嵌入**：来自 Graph Encoder
  - 维度：[batch, num_nodes, 256]

**输出**：
- **解码器状态**：当前状态的综合表示
  - 维度：[batch, 512]
  - 融合了历史信息和图结构信息

### 3.3 核心机制

**Transformer Decoder 架构**（6层）：

每层包含三个子模块：

#### **1. Self-Attention（历史编辑之间的关系）**

```
功能: 捕捉编辑之间的依赖关系

示例:
  Edit 1: Delete Bond(2,3)
  Edit 2: Attach Group(2, *OH)  ← 依赖于 Edit 1
  Edit 3: Change Atom(3, CW)    ← 可能依赖于 Edit 1 和 2

Self-Attention 让模型学习:
  - Edit 2 需要关注 Edit 1（因为在同一个原子上操作）
  - Edit 3 需要关注 Edit 1（因为涉及被断开的键）
```

#### **2. Cross-Attention（编辑序列 → 分子图）**

```
功能: 每个编辑关注分子图的所有原子

示例:
  当前已执行 2 步编辑
  下一步应该操作哪个原子？

Cross-Attention 计算:
  原子 1: 注意力权重 0.05  (不重要)
  原子 2: 注意力权重 0.45  (很重要)
  原子 3: 注意力权重 0.40  (很重要)
  原子 4: 注意力权重 0.10  (较重要)
  ...

解释: 原子 2 和 3 是反应中心，下一步很可能在这里操作
```

#### **3. Feed-Forward Network（非线性变换）**

```
功能: 对融合后的信息进行非线性变换
作用: 提取更高层次的特征
```

**因果掩码（Causal Mask）**：

```
作用: 防止模型"看到未来"

掩码矩阵:
  Step 1 可以看到: [Step 1]
  Step 2 可以看到: [Step 1, Step 2]
  Step 3 可以看到: [Step 1, Step 2, Step 3]

这样训练时模拟真实的推理场景
```

**位置编码（Positional Encoding）**：

```
作用: 告诉模型编辑的顺序

例如:
  Edit 1 的位置编码: [0.0, 1.0, 0.0, 1.0, ...]
  Edit 2 的位置编码: [0.8, 0.5, -0.8, 0.5, ...]
  Edit 3 的位置编码: [0.1, -0.4, -0.1, -0.4, ...]

模型可以区分"第1步"和"第2步"
```

### 3.4 输出的含义

**Decoder State 包含的信息**：
- 已执行了哪些编辑
- 这些编辑如何影响分子结构
- 当前分子处于什么状态
- 距离目标反应物还有多远
- 下一步应该关注哪些原子

---

## 四、Module 3: Actor Network（策略网络）- 核心

### 4.1 功能定位

**Actor Network 是强化学习的核心**，负责决策下一步的编辑操作。

### 4.2 整体结构

Actor Network 分为**三个子模块**，按顺序预测编辑的四个组成部分：

```
输入: decoder_state [batch, 512]
  ↓
【子模块 1: Action Type Predictor】
  预测: action_type (7种动作)
  ↓
【子模块 2: Pointer Network】
  预测: src_idx, tgt_idx (节点索引)
  条件: 基于 action_type
  ↓
【子模块 3: Label Decoder】
  预测: label (标签序列)
  条件: 基于 action_type
  ↓
输出: 完整的编辑操作
```

---

### 4.3 子模块 1: Action Type Predictor

#### **功能**

预测7种动作类型的概率分布。

#### **动作类型**

```
0: Delete Bond    - 删除化学键
1: Change Bond    - 改变键类型（单键↔双键↔三键）
2: Add Bond       - 添加新的化学键
3: Attach Group   - 连接官能团
4: Leave Group    - 离去官能团
5: Change Atom    - 改变原子手性（CW ↔ CCW）
6: Terminate      - 终止编辑序列
```

#### **输入输出**

**输入**：
- decoder_state: [batch, 512]
  - 当前状态的综合表示

**输出**：
- action_type_logits: [batch, 7]
  - 7种动作的未归一化分数

#### **工作原理**

```
使用简单的 MLP (多层感知机):

decoder_state [512]
  ↓
Linear + ReLU → [256]
  ↓
Dropout
  ↓
Linear → [7]
  ↓
action_type_logits

转换为概率:
action_probs = Softmax(action_type_logits)
```

#### **采样策略**

**训练时（探索）**：
```
按概率分布随机采样

例如:
  action_probs = [0.45, 0.13, 0.07, 0.03, 0.10, 0.02, 0.20]
  
  可能采样到:
    - Delete Bond (概率 45%)
    - Terminate (概率 20%)
    - Change Bond (概率 13%)
    - ...
```

**推理时（利用）**：
```
贪心选择概率最高的

例如:
  action_probs = [0.45, 0.13, 0.07, 0.03, 0.10, 0.02, 0.20]
  
  选择: Delete Bond (概率最高 45%)
```

#### **示例**

```
场景: 阿司匹林水解

当前状态: 完整的阿司匹林分子
decoder_state: [...]

预测:
  action_type_logits = [2.3, 1.1, 0.5, -0.2, 0.8, -1.0, -2.5]
  action_probs = [0.45, 0.13, 0.07, 0.03, 0.10, 0.02, 0.00]
  
  选择: Delete Bond (0.45)

化学解释:
  模型认为第一步应该删除酯键（水解反应的关键步骤）
```

---

### 4.4 子模块 2: Hierarchical Pointer Network（重点）

#### **功能**

预测源节点（src_idx）和目标节点（tgt_idx）。

#### **核心挑战**

```
挑战 1: 动作空间巨大
  - 对于 N=20 个原子的分子
  - src 有 20 种选择
  - tgt 有 20 种选择
  - 总共 400 种组合

挑战 2: 大部分组合无效
  - Delete Bond(2,3): 只有当 Bond(2,3) 存在时才有效
  - Add Bond(5,8): 只有当 Bond(5,8) 不存在时才有效

挑战 3: 不同动作的有效组合不同
  - Delete Bond: 只能选择已存在的边
  - Add Bond: 只能选择不存在的边
  - Attach Group: src = tgt (在单个原子上操作)
```

#### **解决方案：分层预测 + 动作掩码**

**核心思想**：
1. 先根据 action_type 确定有效的候选
2. 只对有效候选进行打分
3. 采样时应用掩码，保证选择有效

#### **详细流程**

**Step 1: 获取有效候选**

根据 action_type 动态生成有效的 (src, tgt) 对：

```
Action Type 0 (Delete Bond):
  有效候选 = 所有已存在的边
  
  示例: 分子有边 (2,3), (3,4), (4,5)
  valid_pairs = [(2,3), (3,4), (4,5)]

Action Type 1 (Change Bond):
  有效候选 = 所有已存在的边
  valid_pairs = [(2,3), (3,4), (4,5)]

Action Type 2 (Add Bond):
  有效候选 = 所有不存在的边
  
  示例: 10个原子，已有边 (2,3), (3,4), (4,5)
  valid_pairs = [(0,1), (0,2), (0,4), ..., (7,9)]
  (排除已存在的边)

Action Type 3 (Attach Group):
  有效候选 = 所有原子 (src = tgt)
  valid_pairs = [(0,0), (1,1), (2,2), ..., (9,9)]

Action Type 4 (Leave Group):
  有效候选 = 所有原子 (src = tgt)
  valid_pairs = [(0,0), (1,1), (2,2), ..., (9,9)]

Action Type 5 (Change Atom):
  有效候选 = 所有原子 (src = tgt)
  valid_pairs = [(0,0), (1,1), (2,2), ..., (9,9)]

Action Type 6 (Terminate):
  不需要 src/tgt
  valid_pairs = [(0,0)]
```

**Step 2: 预测 src_idx**

使用 **Multi-head Attention** 计算每个节点作为 src 的重要性：

```
Query: decoder_state + action_type_embedding
  - decoder_state: 当前状态
  - action_type_embedding: 动作类型的嵌入
  - 作用: 告诉模型"我们要执行什么类型的操作"

Key/Value: node_embeddings
  - 每个原子的表示

Attention 计算:
  attention_weights = Softmax(Query · Key^T / √d)
  
输出:
  src_logits = attention_weights
  维度: [batch, num_nodes]
  
示例:
  src_logits = [0.05, 0.45, 0.40, 0.10, ...]
  
  解释:
    原子 0: 0.05 (不太可能是 src)
    原子 1: 0.45 (很可能是 src)
    原子 2: 0.40 (很可能是 src)
    原子 3: 0.10 (可能是 src)
```

**为什么用 Attention？**

```
优势 1: 自动学习重要性
  - 模型会学习哪些原子是反应中心
  - 例如: 酯键的碳原子会有高注意力

优势 2: 考虑全局信息
  - 每个原子的分数考虑了整个分子的结构
  - 不是孤立地评估每个原子

优势 3: 可解释性
  - 注意力权重可以可视化
  - 可以看到模型在关注哪些原子
```

**Step 3: 预测 tgt_idx（条件化在 src 上）**

tgt 的预测依赖于 src：

```
Query: decoder_state + action_type_embedding + src_idx_embedding
  - 增加了 src_idx_embedding
  - 作用: 告诉模型"src 已经选好了，现在选 tgt"

Key/Value: node_embeddings

Attention 计算:
  tgt_logits = Attention(Query, Key, Value)
  
示例:
  假设 src_idx = 2
  
  tgt_logits = [0.02, 0.05, 0.01, 0.80, 0.12, ...]
  
  解释:
    原子 0: 0.02 (不太可能是 tgt)
    原子 1: 0.05 (不太可能是 tgt)
    原子 2: 0.01 (不可能是 tgt，因为 src=2)
    原子 3: 0.80 (很可能是 tgt)
    原子 4: 0.12 (可能是 tgt)
```

**为什么 tgt 要条件化在 src 上？**

```
化学直觉:
  - 如果要删除键，src 和 tgt 必须相连
  - 如果要添加键，src 和 tgt 不能已经相连
  - tgt 的选择依赖于 src 是谁

例如:
  src = 原子 2 (羰基的碳)
  
  对于 Delete Bond:
    tgt 应该是与原子 2 相连的原子
    → 原子 3 (氧) 的概率会很高
  
  对于 Add Bond:
    tgt 应该是与原子 2 不相连的原子
    → 原子 7 的概率会很高
```

**Step 4: 应用动作掩码**

将无效候选的 logit 设为 -∞：

```
示例: Delete Bond

有效候选: [(2,3), (3,4), (4,5)]

原始 src_logits:
  [0.05, 0.10, 0.45, 0.40, 0.30, 0.20, ...]
  
应用掩码后:
  [-∞, -∞, 0.45, 0.40, 0.30, -∞, ...]
  
  只有原子 2, 3, 4 是有效的 src
  (因为只有它们参与已存在的边)

Softmax 后:
  [0.0, 0.0, 0.38, 0.34, 0.25, 0.0, ...]
  
  无效原子的概率变为 0
```

**掩码的作用**：

```
作用 1: 保证采样有效
  - 采样时不会选到无效的 (src, tgt)
  - 避免生成无效的编辑操作

作用 2: 提高训练效率
  - 模型不需要学习"哪些是无效的"
  - 可以专注学习"有效候选中哪个最好"

作用 3: 符合化学约束
  - 强制模型遵守化学规则
  - 例如: 不能删除不存在的键
```

#### **采样策略**

**策略 1: 贪心采样**

```
选择概率最高的:

src_idx = argmax(src_logits)
tgt_idx = argmax(tgt_logits)

优点: 确定性，可重复
缺点: 缺乏探索
```

**策略 2: 随机采样**

```
按概率分布采样:

src_idx ~ Categorical(src_probs)
tgt_idx ~ Categorical(tgt_probs)

优点: 有探索性，可以发现新路径
缺点: 有随机性
```

**策略 3: 约束采样（推荐）**

```
只从有效候选中采样:

1. 获取有效候选 valid_pairs
2. 计算每个有效对的分数:
   score(src, tgt) = src_logits[src] + tgt_logits[tgt]
3. 从有效对中按分数采样

优点: 保证有效 + 有探索性
```

#### **完整示例**

```
场景: 阿司匹林水解的第一步

当前状态: 完整的阿司匹林分子 (13个原子)
action_type: Delete Bond (0)

Step 1: 获取有效候选
  已存在的边:
    (0,1), (1,2), (2,3), ..., (8,9) (酯键)
  valid_pairs = [(0,1), (1,2), ..., (8,9)]

Step 2: 预测 src
  src_logits = [0.02, 0.05, 0.08, ..., 0.45, ...]
                                        ↑
                                     原子 8 (酯键的碳)
  
  应用掩码后:
    只有参与边的原子有非零概率
  
  采样: src_idx = 8

Step 3: 预测 tgt (条件化在 src=8 上)
  tgt_logits = [0.01, 0.02, ..., 0.80, ...]
                                  ↑
                               原子 9 (酯键的氧)
  
  应用掩码后:
    只有与原子 8 相连的原子有非零概率
  
  采样: tgt_idx = 9

结果:
  预测的编辑: Delete Bond(8, 9)
  
化学解释:
  删除原子 8 和 9 之间的酯键
  这是阿司匹林水解的关键步骤
  会断开乙酰基和水杨酸之间的连接
```

---

### 4.5 子模块 3: Label Decoder

#### **功能**

预测 label 序列（键类型、手性、官能团 SMILES）。

#### **Label 的类型**

根据 action_type 不同，label 的含义不同：

```
Action Type 0,1,2 (Bond 操作):
  Label = 键类型
  - SINGLE (单键)
  - DOUBLE (双键)
  - TRIPLE (三键)
  - AROMATIC (芳香键)
  - NONE (无键，用于删除)
  
  长度: 1 个 token

Action Type 5 (Change Atom):
  Label = 手性
  - CW (顺时针)
  - CCW (逆时针)
  - NONE (无手性)
  
  长度: 1 个 token

Action Type 3,4 (Attach/Leave Group):
  Label = 官能团的 SMILES
  - 例如: *C(=O)C (乙酰基)
  - 例如: *OH (羟基)
  - 例如: *NH2 (氨基)
  
  长度: 可变 (1-50 个 token)
```

#### **架构**

使用 **Transformer Decoder** 进行序列生成：

```
输入: decoder_state + action_type_embedding
  - 条件化在 action_type 上
  - 不同类型的 label 共享同一个解码器

输出: label 序列
  - 自回归生成
  - 每次预测一个 token
```

#### **训练模式（Teacher Forcing）**

```
目标序列: [BOS, *, C, (, =, O, ), C, EOS]
          (乙酰基的 SMILES)

训练时:
  输入: [BOS, *, C, (, =, O, ), C]
  预测: [*, C, (, =, O, ), C, EOS]
  
  每个位置都用真实的前缀作为输入
  预测下一个 token
```

**因果掩码**：

```
Step 1: 只能看到 [BOS]          → 预测 *
Step 2: 只能看到 [BOS, *]       → 预测 C
Step 3: 只能看到 [BOS, *, C]    → 预测 (
...

防止模型"作弊"看到未来的 token
```

#### **推理模式（Autoregressive）**

```
自回归生成:

Step 1: 输入 [BOS]              → 预测 *
Step 2: 输入 [BOS, *]           → 预测 C
Step 3: 输入 [BOS, *, C]        → 预测 (
Step 4: 输入 [BOS, *, C, (]     → 预测 =
...
直到预测 [EOS] 或达到最大长度
```

#### **条件化机制**

Label Decoder 需要知道当前是什么类型的动作：

```
不同 action_type 的 label 不同:

Action Type 0 (Delete Bond):
  期望 label: [NONE] 或 [SINGLE] (表示删除前的键类型)
  
Action Type 3 (Attach Group):
  期望 label: [*, C, H, 3] (甲基)
  
模型需要根据 action_type 调整预测
```

**实现方式**：

```
将 action_type 的嵌入加到 decoder_state 上:

context = decoder_state + action_type_embedding

这样 Label Decoder 就知道:
  "我现在要预测的是键类型，不是官能团"
  或
  "我现在要预测的是官能团 SMILES"
```

#### **采样策略**

**贪心解码**：

```
每步选择概率最高的 token:

Step 1: token_probs = [0.01, 0.05, 0.80, ...]
        选择: token_2 (概率 0.80)

Step 2: token_probs = [0.02, 0.70, 0.10, ...]
        选择: token_1 (概率 0.70)

优点: 确定性
缺点: 可能陷入局部最优
```

**随机采样**：

```
按概率分布采样:

Step 1: token_probs = [0.01, 0.05, 0.80, 0.14]
        采样: 可能是 token_2 (80%) 或 token_3 (14%)

优点: 有探索性
缺点: 可能生成低质量序列
```

**Beam Search（推理时）**：

```
保留 top-K 个候选序列:

K=3:
  候选 1: [*, C, H, 3]      分数: 8.5
  候选 2: [*, O, H]         分数: 7.2
  候选 3: [*, N, H, 2]      分数: 6.8

选择分数最高的
```

#### **示例**

```
场景: Attach Group 操作

action_type: 3 (Attach Group)
src_idx: 2
tgt_idx: 2

Label Decoder 预测:

Step 1: 输入 [BOS]
  预测: * (连接点符号)
  概率: 0.95

Step 2: 输入 [BOS, *]
  预测: C (碳原子)
  概率: 0.85

Step 3: 输入 [BOS, *, C]
  预测: ( (左括号)
  概率: 0.70

Step 4: 输入 [BOS, *, C, (]
  预测: = (双键)
  概率: 0.90

Step 5: 输入 [BOS, *, C, (, =]
  预测: O (氧原子)
  概率: 0.95

Step 6: 输入 [BOS, *, C, (, =, O]
  预测: ) (右括号)
  概率: 0.92

Step 7: 输入 [BOS, *, C, (, =, O, )]
  预测: C (碳原子)
  概率: 0.80

Step 8: 输入 [BOS, *, C, (, =, O, ), C]
  预测: EOS (结束)
  概率: 0.88

最终 label: *C(=O)C (乙酰基)

化学解释:
  在原子 2 上连接乙酰基 (-COCH3)
  这是典型的酰化反应
```

---

## 五、Module 4: Critic Network（价值网络）

### 5.1 功能定位

评估当前状态的"好坏"，即状态价值 V(s)。

### 5.2 价值函数的含义

```
V(s) = 从状态 s 开始，按照当前策略，预期能获得的累积奖励

数学定义:
  V(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]

其中:
  - R_t: 第 t 步的即时奖励
  - γ: 折扣因子 (通常 0.99)
  - E[·]: 期望
```

### 5.3 输入输出

**输入**：
- decoder_state: [batch, 512]
  - 当前状态的表示

**输出**：
- value: [batch, 1]
  - 状态价值的估计

### 5.4 架构

```
简单的 MLP:

decoder_state [512]
  ↓
Linear + ReLU → [256]
  ↓
Dropout
  ↓
Linear + ReLU → [128]
  ↓
Dropout
  ↓
Linear → [1]
  ↓
value
```

### 5.5 示例

```
场景: 阿司匹林水解

状态 1 (初始):
  完整的阿司匹林分子
  V(s_1) = 0.0
  
  解释: 还没开始编辑，距离目标很远

状态 2 (执行 1 步后):
  Delete Bond(8,9) - 断开酯键
  V(s_2) = 5.5
  
  解释: 执行了关键步骤，状态价值提升

状态 3 (执行 2 步后):
  Attach Group(9, *H) - 在氧上加氢
  V(s_3) = 8.2
  
  解释: 继续接近目标，价值进一步提升

状态 4 (执行 3 步后):
  Attach Group(8, *OH) - 在碳上加羟基
  V(s_4) = 9.5
  
  解释: 几乎完成，价值接近最大值

状态 5 (终止):
  Terminate
  V(s_5) = 10.0
  
  解释: 成功生成正确的反应物
```

### 5.6 Critic 的作用

**作用 1: 提供 Baseline（减少方差）**

```
在 Policy Gradient 中:

∇J(θ) = E[∇log π(a|s) × (R - V(s))]
                              ↑
                           baseline

如果没有 baseline:
  ∇J(θ) = E[∇log π(a|s) × R]
  
  问题: R 的方差很大，训练不稳定

有了 baseline:
  优势函数 A(s,a) = R - V(s)
  
  如果 R > V(s): 这个动作比平均好，增加其概率
  如果 R < V(s): 这个动作比平均差，减少其概率
```

**作用 2: 评估状态质量**

```
可以用 V(s) 判断当前状态的好坏:

V(s) = 9.0: 状态很好，接近目标
V(s) = 5.0: 状态一般，还有距离
V(s) = 1.0: 状态不好，可能走错了

这对调试和可解释性很有帮助
```

**作用 3: 早停判断**

```
如果 V(s) 很低，说明当前路径不太对:

V(s) < 阈值 → 提前终止，重新采样

节省计算资源
```

---

## 六、Module 5: Q-Network（动作价值网络）

### 6.1 功能定位

评估在特定状态下执行特定动作的"好坏"，即动作价值 Q(s, a)。

### 6.2 Q 函数的含义

```
Q(s, a) = 在状态 s 执行动作 a，然后按照当前策略，预期能获得的累积奖励

数学定义:
  Q(s, a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, a_t = a]

与 V(s) 的关系:
  V(s) = E_{a~π}[Q(s, a)]
  
  V(s) 是所有动作的期望
  Q(s, a) 是特定动作的价值
```

### 6.3 输入输出

**输入**：
- decoder_state: [batch, 512]
- action: {
    'action_type': [batch],
    'src_idx': [batch],
    'tgt_idx': [batch]
  }

**输出**：
- q_value: [batch, 1]

### 6.4 架构

```
1. 嵌入动作的各个组成部分:
   action_type_emb = Embedding(action_type)  [128]
   src_emb = Embedding(src_idx)              [128]
   tgt_emb = Embedding(tgt_idx)              [128]

2. 拼接状态和动作:
   combined = [decoder_state, action_type_emb, src_emb, tgt_emb]
   维度: [512 + 128 + 128 + 128] = [896]

3. MLP:
   combined [896]
     ↓
   Linear + ReLU → [512]
     ↓
   Dropout
     ↓
   Linear + ReLU → [256]
     ↓
   Linear → [1]
     ↓
   q_value
```

### 6.5 示例

```
场景: 阿司匹林水解的第一步

状态 s: 完整的阿司匹林分子

候选动作及其 Q 值:

动作 1: Delete Bond(8, 9) - 断开酯键
  Q(s, a_1) = 9.2
  
  解释: 这是最优动作，能带来最高的长期回报

动作 2: Delete Bond(2, 3) - 断开苯环的键
  Q(s, a_2) = 2.1
  
  解释: 这个动作不好，会破坏苯环结构

动作 3: Change Bond(8, 9) - 改变酯键类型
  Q(s, a_3) = 5.3
  
  解释: 这个动作中等，但不如直接删除

动作 4: Attach Group(5, *OH) - 在苯环上加羟基
  Q(s, a_4) = 3.8
  
  解释: 这个动作与目标无关，Q 值较低

选择: 动作 1 (Q 值最高)
```

### 6.6 Q-Network 的作用

**作用 1: 可解释性（核心）**

```
Q 值提供了动作选择的依据:

为什么选择 Delete Bond(8,9)?
  → 因为 Q(s, Delete Bond(8,9)) = 9.2 最高

为什么不选择 Delete Bond(2,3)?
  → 因为 Q(s, Delete Bond(2,3)) = 2.1 很低

这是监督学习无法提供的
```

**作用 2: 反事实推理**

```
可以比较不同动作的效果:

如果选择动作 A: Q(s, A) = 9.2
如果选择动作 B: Q(s, B) = 5.3

差距: 9.2 - 5.3 = 3.9

解释: 动作 A 比动作 B 好 3.9 个单位的奖励
```

**作用 3: 辅助训练**

```
Q-Learning 更新:

Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
                              ↑
                        下一状态的最大 Q 值

帮助 Actor 学习更好的策略
```

**作用 4: 探索策略**

```
ε-greedy 策略:

以概率 ε: 随机选择动作 (探索)
以概率 1-ε: 选择 Q 值最高的动作 (利用)

或者 Boltzmann 探索:
  P(a|s) ∝ exp(Q(s,a) / τ)
  
  τ 大: 更随机 (探索)
  τ 小: 更确定 (利用)
```

---

## 七、Actor vs Critic vs Q-Network 对比

### 7.1 功能对比

| 网络 | 输入 | 输出 | 功能 |
|------|------|------|------|
| **Actor** | 状态 s | 动作分布 π(a\|s) | 决策：应该做什么 |
| **Critic** | 状态 s | 状态价值 V(s) | 评估：当前状态有多好 |
| **Q-Network** | 状态 s + 动作 a | 动作价值 Q(s,a) | 评估：这个动作有多好 |

### 7.2 关系

```
Actor 负责"做决策":
  π(a|s): 在状态 s 下选择动作 a 的概率

Critic 负责"评估状态":
  V(s): 状态 s 的好坏

Q-Network 负责"评估动作":
  Q(s,a): 在状态 s 下动作 a 的好坏

关系:
  V(s) = E_{a~π}[Q(s,a)]
  A(s,a) = Q(s,a) - V(s)  (优势函数)
```

### 7.3 训练中的作用

```
Actor-Critic 训练:

1. Actor 生成动作:
   a ~ π(·|s)

2. 执行动作，获得奖励:
   s' = env.step(s, a)
   r = reward(s, a, s')

3. Critic 评估状态:
   V(s), V(s')

4. 计算优势:
   A(s,a) = r + γV(s') - V(s)

5. 更新 Actor:
   ∇J = ∇log π(a|s) × A(s,a)

6. 更新 Critic:
   Loss = (V(s) - (r + γV(s')))²

7. 更新 Q-Network (可选):
   Loss = (Q(s,a) - (r + γ max Q(s',a')))²
```

---

## 八、完整的前向传播流程

### 8.1 单步预测

```
输入: Product Graph (产物分子图)

Step 1: Graph Encoder
  输入: node_feat, edge_index
  输出: node_embeddings, graph_embedding
  
Step 2: State Tracker
  输入: history_edits, node_embeddings
  输出: decoder_state
  
Step 3: Actor - Action Type
  输入: decoder_state
  输出: action_type_logits
  采样: action_type
  
Step 4: Actor - Pointer Network
  输入: decoder_state, node_embeddings, action_type
  输出: src_logits, tgt_logits
  采样: src_idx, tgt_idx
  
Step 5: Actor - Label Decoder
  输入: decoder_state, action_type
  输出: label_logits
  采样: label
  
Step 6: Critic
  输入: decoder_state
  输出: V(s)
  
Step 7: Q-Network
  输入: decoder_state, (action_type, src_idx, tgt_idx)
  输出: Q(s,a)

输出: 完整的编辑操作 + 价值估计
```

### 8.2 完整的 Episode

```
初始化: Product Graph

Episode:
  for step in range(max_steps):
    # 1. 编码当前状态
    state = encode_state(product, history_edits)
    
    # 2. Actor 预测动作
    action = actor.sample(state)
    
    # 3. Critic 评估状态
    value = critic(state)
    
    # 4. Q-Network 评估动作
    q_value = q_network(state, action)
    
    # 5. 执行动作
    product = apply_edit(product, action)
    
    # 6. 计算奖励
    reward = compute_reward(product, ground_truth)
    
    # 7. 记录轨迹
    trajectory.append({
      'state': state,
      'action': action,
      'value': value,
      'q_value': q_value,
      'reward': reward
    })
    
    # 8. 更新历史
    history_edits.append(action_embedding)
    
    # 9. 检查终止
    if action.type == TERMINATE or step == max_steps:
      break

返回: trajectory
```

---

## 九、与监督学习的对比

### 9.1 架构对比

| 组件 | 监督学习 | 强化学习 |
|------|---------|---------|
| **Graph Encoder** | ✅ 相同 | ✅ 相同 |
| **State Tracker** | ✅ 相同 | ✅ 相同 |
| **Action Predictor** | ✅ 有 | ✅ 有（Actor） |
| **Pointer Network** | ✅ 有 | ✅ 有（Actor） |
| **Label Decoder** | ✅ 有 | ✅ 有（Actor） |
| **Value Network** | ❌ 无 | ✅ 有（Critic） |
| **Q-Network** | ❌ 无 | ✅ 有 |

### 9.2 训练对比

| 维度 | 监督学习 | 强化学习 |
|------|---------|---------|
| **损失函数** | CrossEntropy | Policy Gradient + Value Loss |
| **训练信号** | Ground Truth Edits | Reward Signal |
| **采样策略** | Teacher Forcing | 自回归采样 |
| **探索** | 无 | ε-greedy / Boltzmann |
| **Baseline** | 无 | V(s) 作为 baseline |

### 9.3 预测对比

| 维度 | 监督学习 | 强化学习 |
|------|---------|---------|
| **决策依据** | 概率最大 | Q 值最大 |
| **可解释性** | 弱（只有概率） | 强（有 Q 值和 V 值） |
| **探索能力** | 无 | 有 |
| **等价路径** | 无法发现 | 可以发现 |

---

## 十、总结

### 10.1 核心模块总结

**Module 1: Graph Encoder**
- 作用：编码分子图
- 技术：GAT (4层)
- 输出：节点嵌入 + 图嵌入

**Module 2: State Tracker**
- 作用：维护历史编辑序列
- 技术：Transformer Decoder (6层)
- 输出：当前状态表示

**Module 3: Actor Network**（核心）
- 作用：预测下一步编辑
- 子模块：
  1. Action Type Predictor (MLP)
  2. Hierarchical Pointer Network (Attention + 动作掩码)
  3. Label Decoder (Transformer Decoder)
- 输出：完整的编辑操作

**Module 4: Critic Network**
- 作用：评估状态价值
- 技术：MLP
- 输出：V(s)

**Module 5: Q-Network**
- 作用：评估动作价值
- 技术：MLP
- 输出：Q(s,a)

### 10.2 关键技术点

1. **分层预测**：先 action_type，再 pointer，最后 label
2. **条件化**：每个模块都条件化在前面的预测上
3. **动作掩码**：保证预测的动作有效
4. **注意力机制**：自动学习重要性
5. **自回归生成**：序列化预测 label
6. **价值估计**：V(s) 和 Q(s,a) 提供可解释性

### 10.3 优势

✅ **可解释性强**：Q 值和 V 值提供决策依据  
✅ **探索能力**：可以发现新的编辑路径  
✅ **全局优化**：优化最终目标而非局部步骤  
✅ **化学约束**：动作掩码保证有效性  




---

# 强化学习模型各模块详细说明

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     RL-based SSR Model                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Product Graph                                        │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 1: Graph Encoder                             │   │
│  │  - 编码产物分子图                                     │   │
│  │  - 输出: Node Embeddings + Graph Embedding           │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 2: State Tracker (Transformer Decoder)       │   │
│  │  - 维护历史编辑序列                                   │   │
│  │  - 输出: Decoder State (当前状态表示)                │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 3: Actor Network (Policy)                    │   │
│  │  ├─ Step 1: Action Type Predictor                    │   │
│  │  ├─ Step 2: Pointer Network (src, tgt)               │   │
│  │  └─ Step 3: Label Decoder                            │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 4: Critic Network (Value Function)           │   │
│  │  - 评估状态价值 V(s)                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 5: Q-Network (Action-Value Function)         │   │
│  │  - 评估动作价值 Q(s,a)                                │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  Output: Edit Sequence                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、Module 1: Graph Encoder

### 2.1 功能

将产物分子图编码为节点嵌入和图嵌入。

### 2.2 详细架构

```python
class GraphEncoder(nn.Module):
    """
    产物图编码器
    
    输入: 
        - node_feat: [num_nodes, node_feat_dim] 原子特征
        - edge_index: [2, num_edges] 边索引
        - edge_attr: [num_edges, edge_feat_dim] 边特征
        - batch: [num_nodes] 批次索引
    
    输出:
        - node_embeddings: [num_nodes, hidden_dim] 每个原子的表示
        - graph_embedding: [batch_size, hidden_dim] 整个分子的表示
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 节点特征投影
        self.node_proj = nn.Linear(
            config.node_feat_dim,      # 输入: 128 (原子特征)
            config.graph_hidden_dim    # 输出: 256
        )
        
        # 边特征投影
        self.edge_proj = nn.Linear(
            config.edge_feat_dim,      # 输入: 16 (键特征)
            config.graph_hidden_dim    # 输出: 256
        )
        
        # 图注意力层 (GAT)
        self.convs = nn.ModuleList([
            GATConv(
                in_channels=config.graph_hidden_dim,   # 256
                out_channels=config.graph_hidden_dim,  # 256
                heads=4,                    # 4个注意力头
                concat=False,               # 输出维度不变
                dropout=config.graph_dropout,
                edge_dim=config.graph_hidden_dim  # 边特征维度
            )
            for _ in range(config.graph_num_layers)  # 4层
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.graph_hidden_dim)
            for _ in range(config.graph_num_layers)
        ])
        
        self.dropout = nn.Dropout(config.graph_dropout)
        
        # 图级池化
        self.graph_pool = nn.Sequential(
            nn.Linear(config.graph_hidden_dim, config.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.graph_dropout)
        )
    
    def forward(self, node_feat, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        
        示例:
            输入分子: 乙醇 (C2H6O)
            - 3个原子: C, C, O
            - node_feat: [3, 128]
            - edge_index: [[0,1,1,2], [1,0,2,1]] (C-C, C-O)
            - edge_attr: [4, 16]
        """
        # 1. 投影节点特征
        x = self.node_proj(node_feat)  # [num_nodes, 256]
        
        # 2. 投影边特征
        if edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr)  # [num_edges, 256]
        
        # 3. 图卷积 (多层GAT)
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            x_residual = x
            
            # GAT 层
            x = conv(x, edge_index, edge_attr=edge_attr)
            # x: [num_nodes, 256]
            
            # Layer Norm
            x = norm(x)
            
            # 激活函数
            x = F.relu(x)
            
            # Dropout
            x = self.dropout(x)
            
            # 残差连接
            x = x + x_residual
        
        node_embeddings = x  # [num_nodes, 256]
        
        # 4. 图级池化
        if batch is None:
            batch = torch.zeros(node_feat.size(0), dtype=torch.long, device=x.device)
        
        # Global Mean Pooling
        graph_embedding = global_mean_pool(x, batch)  # [batch_size, 256]
        graph_embedding = self.graph_pool(graph_embedding)
        
        return node_embeddings, graph_embedding
```

### 2.3 关键技术

**GAT (Graph Attention Network)**：
```
对于节点 i，计算其表示:

h_i^(l+1) = σ(Σ_{j∈N(i)} α_{ij} W h_j^(l))

其中:
- α_{ij}: 注意力权重（自动学习）
- W: 可学习的权重矩阵
- N(i): 节点 i 的邻居

优势:
- 自适应地学习原子之间的重要性
- 比 GCN 更灵活
```

---

## 三、Module 2: State Tracker

### 3.1 功能

维护历史编辑序列，生成当前状态的表示。

### 3.2 详细架构

```python
class StateTracker(nn.Module):
    """
    状态追踪器 (基于 Transformer Decoder)
    
    功能:
        - 编码历史编辑序列
        - 与图节点进行 Cross-Attention
        - 输出当前状态的表示
    
    输入:
        - history_edits: [batch, seq_len, hidden_dim] 历史编辑嵌入
        - node_embeddings: [batch, num_nodes, hidden_dim] 节点嵌入
    
    输出:
        - decoder_state: [batch, hidden_dim] 当前状态表示
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model=config.decoder_hidden_dim,
            max_len=config.max_edit_steps
        )
        
        # Transformer Decoder Layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config.decoder_hidden_dim,    # 512
                nhead=config.decoder_num_heads,       # 8
                dim_feedforward=config.decoder_hidden_dim * 4,  # 2048
                dropout=config.decoder_dropout,
                batch_first=True
            )
            for _ in range(config.decoder_num_layers)  # 6层
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(config.decoder_hidden_dim)
        
        # 图到解码器的投影
        self.graph_proj = nn.Linear(
            config.graph_hidden_dim,      # 256
            config.decoder_hidden_dim     # 512
        )
    
    def forward(self, history_edits, node_embeddings, node_mask=None):
        """
        前向传播
        
        示例:
            当前已执行 2 步编辑:
            - Edit 1: Delete Bond(2,3)
            - Edit 2: Attach Group(2, *OH)
            
            history_edits: [batch, 2, 512]
            node_embeddings: [batch, 10, 256] (10个原子)
        """
        batch_size, seq_len, _ = history_edits.size()
        
        # 1. 添加位置编码
        x = self.pos_encoding(history_edits)  # [batch, seq_len, 512]
        
        # 2. 投影节点嵌入
        memory = self.graph_proj(node_embeddings)  # [batch, num_nodes, 512]
        
        # 3. 生成因果掩码 (防止看到未来)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        # tgt_mask: [seq_len, seq_len]
        # [[0, -inf, -inf],
        #  [0,    0, -inf],
        #  [0,    0,    0]]
        
        # 4. Transformer Decoder
        for layer in self.layers:
            x = layer(
                tgt=x,                    # 目标序列 (历史编辑)
                memory=memory,            # 源序列 (图节点)
                tgt_mask=tgt_mask,        # 因果掩码
                memory_key_padding_mask=~node_mask.bool() if node_mask is not None else None
            )
        
        x = self.norm(x)
        
        # 5. 取最后一个时间步的输出作为当前状态
        decoder_state = x[:, -1, :]  # [batch, 512]
        
        return decoder_state
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
```

### 3.3 关键机制

**Self-Attention（历史编辑之间）**：
```
每个编辑可以关注之前的所有编辑

例如:
  Edit 1: Delete Bond(2,3)
  Edit 2: Attach Group(2, *OH)  ← 可以关注 Edit 1
  Edit 3: Change Atom(3, CW)    ← 可以关注 Edit 1, 2

这样模型可以学习编辑之间的依赖关系
```

**Cross-Attention（编辑 → 图节点）**：
```
每个编辑可以关注分子图的所有原子

例如:
  当前状态: 已执行 2 步编辑
  下一步预测: 需要关注哪些原子？
  
  Cross-Attention 会计算:
    - 原子 1 的重要性: 0.05
    - 原子 2 的重要性: 0.45  ← 高注意力
    - 原子 3 的重要性: 0.40  ← 高注意力
    - ...
  
  解释: 原子 2 和 3 是下一步操作的关键位点
```

---

## 四、Module 3: Actor Network（核心）

### 4.1 整体结构

```python
class ActorNetwork(nn.Module):
    """
    策略网络 (Actor)
    
    功能: 预测下一步的编辑操作
    
    输入:
        - decoder_state: [batch, 512] 当前状态表示
        - node_embeddings: [batch, num_nodes, 256] 节点嵌入
        - edge_index: [batch, 2, num_edges] 边索引
        - node_mask: [batch, num_nodes] 节点掩码
    
    输出:
        - action_dist: 动作的概率分布
            - action_type_logits: [batch, 7]
            - src_logits: [batch, num_nodes]
            - tgt_logits: [batch, num_nodes]
            - label_logits: [batch, max_len, vocab_size]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 子模块 1: Action Type Predictor
        self.action_predictor = ActionTypePredictor(config)
        
        # 子模块 2: Pointer Network
        self.pointer_network = HierarchicalPointer(config)
        
        # 子模块 3: Label Decoder
        self.label_decoder = UnifiedLabelDecoder(config)
    
    def forward(
        self, 
        decoder_state, 
        node_embeddings, 
        edge_index,
        node_mask=None,
        action_type=None  # 可选: 用于训练时的 teacher forcing
    ):
        """
        前向传播: 预测完整的编辑操作
        """
        # Step 1: 预测 action_type
        action_type_logits = self.action_predictor(decoder_state)
        # [batch, 7]
        
        # Step 2: 预测 src_idx 和 tgt_idx
        src_logits, tgt_logits, valid_pairs = self.pointer_network(
            decoder_state,
            node_embeddings,
            edge_index,
            action_type=action_type,  # 条件化
            node_mask=node_mask
        )
        # src_logits: [batch, num_nodes]
        # tgt_logits: [batch, num_nodes]
        # valid_pairs: List[Tensor] 每个样本的有效 (src,tgt) 对
        
        # Step 3: 预测 label
        label_logits = self.label_decoder(
            decoder_state,
            action_type=action_type
        )
        # [batch, max_len, vocab_size]
        
        return {
            'action_type_logits': action_type_logits,
            'src_logits': src_logits,
            'tgt_logits': tgt_logits,
            'label_logits': label_logits,
            'valid_pairs': valid_pairs
        }
```

---

### 4.2 子模块 1: Action Type Predictor

```python
class ActionTypePredictor(nn.Module):
    """
    动作类型预测器
    
    功能: 预测 7 种动作类型的概率分布
    
    输入:
        - decoder_state: [batch, 512]
    
    输出:
        - action_logits: [batch, 7]
    
    动作类型:
        0: Delete Bond
        1: Change Bond
        2: Add Bond
        3: Attach Group
        4: Leave Group
        5: Change Atom
        6: Terminate
    """
    
    def __init__(self, config):
        super().__init__()
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_hidden_dim // 2, config.num_actions)
        )
    
    def forward(self, decoder_state):
        """
        前向传播
        
        示例:
            decoder_state: [2, 512] (batch_size=2)
            
            输出:
            action_logits: [2, 7]
            [[2.3, 1.1, 0.5, -0.2, 0.8, -1.0, -2.5],  # 样本1
             [1.5, 0.9, 1.8,  0.3, 0.2, -0.5, -1.8]]   # 样本2
            
            转换为概率:
            action_probs = Softmax(action_logits)
            [[0.45, 0.13, 0.07, 0.03, 0.10, 0.02, 0.00],  # 样本1倾向于 Delete Bond
             [0.20, 0.11, 0.27, 0.06, 0.05, 0.03, 0.01]]   # 样本2倾向于 Add Bond
        """
        action_logits = self.mlp(decoder_state)  # [batch, 7]
        return action_logits
    
    def sample(self, decoder_state, deterministic=False):
        """
        采样动作类型
        
        Args:
            decoder_state: [batch, 512]
            deterministic: 是否使用确定性策略（贪心）
        
        Returns:
            action_type: [batch] 采样的动作类型
            log_prob: [batch] 对数概率
        """
        action_logits = self.forward(decoder_state)
        
        if deterministic:
            # 贪心: 选择概率最高的
            action_type = action_logits.argmax(dim=-1)
            log_prob = F.log_softmax(action_logits, dim=-1).gather(
                1, action_type.unsqueeze(1)
            ).squeeze(1)
        else:
            # 随机采样: 按概率分布采样
            action_dist = Categorical(logits=action_logits)
            action_type = action_dist.sample()
            log_prob = action_dist.log_prob(action_type)
        
        return action_type, log_prob
```

**关键点**：
- 输入是**当前状态的表示**（decoder_state）
- 输出是**7个动作类型的 logits**
- 采样时可以选择**贪心**或**随机**策略

---

### 4.3 子模块 2: Hierarchical Pointer Network（重点）

```python
class HierarchicalPointer(nn.Module):
    """
    分层指针网络
    
    功能: 预测 src_idx 和 tgt_idx
    
    核心思想:
        1. 根据 action_type 确定有效的 (src, tgt) 候选
        2. 对有效候选进行打分
        3. 采样时应用动作掩码
    
    输入:
        - decoder_state: [batch, 512]
        - node_embeddings: [batch, num_nodes, 256]
        - edge_index: [batch, 2, num_edges]
        - action_type: [batch] (可选)
        - node_mask: [batch, num_nodes]
    
    输出:
        - src_logits: [batch, num_nodes]
        - tgt_logits: [batch, num_nodes]
        - valid_pairs: List[Tensor] 有效的 (src, tgt) 对
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Action type embedding (用于条件化)
        self.action_embedding = nn.Embedding(
            config.num_actions,           # 7
            config.decoder_hidden_dim     # 512
        )
        
        # 投影节点嵌入到解码器维度
        self.node_proj = nn.Linear(
            config.graph_hidden_dim,      # 256
            config.decoder_hidden_dim     # 512
        )
        
        # Attention for src
        self.src_attn = nn.MultiheadAttention(
            embed_dim=config.decoder_hidden_dim,  # 512
            num_heads=config.decoder_num_heads,   # 8
            dropout=config.decoder_dropout,
            batch_first=True
        )
        
        # Attention for tgt (conditioned on src)
        self.tgt_attn = nn.MultiheadAttention(
            embed_dim=config.decoder_hidden_dim,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
            batch_first=True
        )
        
        # Src embedding (用于 tgt 的条件化)
        self.src_idx_embedding = nn.Embedding(1000, config.decoder_hidden_dim)
    
    def forward(
        self, 
        decoder_state, 
        node_embeddings, 
        edge_index,
        action_type=None,
        node_mask=None
    ):
        """
        前向传播
        
        详细流程:
        
        1. 根据 action_type 获取有效的 (src, tgt) 候选
        2. 预测 src_idx
        3. 根据 src_idx 预测 tgt_idx
        4. 应用动作掩码
        """
        batch_size, num_nodes, _ = node_embeddings.size()
        
        # 投影节点嵌入
        node_embeddings_proj = self.node_proj(node_embeddings)
        # [batch, num_nodes, 512]
        
        # ============ Step 1: 获取有效候选 ============
        if action_type is not None:
            valid_pairs_list = self._get_valid_pairs_batch(
                action_type, edge_index, node_mask
            )
        else:
            valid_pairs_list = None
        
        # ============ Step 2: 预测 src_idx ============
        
        # 构造 query (条件化在 action_type 上)
        if action_type is not None:
            action_emb = self.action_embedding(action_type)  # [batch, 512]
            query_src = (decoder_state + action_emb).unsqueeze(1)
        else:
            query_src = decoder_state.unsqueeze(1)
        # [batch, 1, 512]
        
        # Multi-head Attention
        _, src_attn_weights = self.src_attn(
            query=query_src,                    # [batch, 1, 512]
            key=node_embeddings_proj,           # [batch, num_nodes, 512]
            value=node_embeddings_proj,         # [batch, num_nodes, 512]
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        # src_attn_weights: [batch, 1, num_nodes]
        
        src_logits = src_attn_weights.squeeze(1)  # [batch, num_nodes]
        
        # ============ Step 3: 预测 tgt_idx (条件化在 src 上) ============
        
        # 方式 A: 使用预测的 src (推理时)
        if action_type is None:
            src_idx_pred = src_logits.argmax(dim=-1)  # [batch]
        else:
            # 方式 B: 使用真实的 src (训练时, teacher forcing)
            # 这里我们仍然用预测的，也可以传入真实值
            src_idx_pred = src_logits.argmax(dim=-1)
        
        # Embed src_idx
        src_idx_emb = self.src_idx_embedding(src_idx_pred)  # [batch, 512]
        
        # 构造 query (条件化在 action_type 和 src_idx 上)
        if action_type is not None:
            query_tgt = (decoder_state + action_emb + src_idx_emb).unsqueeze(1)
        else:
            query_tgt = (decoder_state + src_idx_emb).unsqueeze(1)
        # [batch, 1, 512]
        
        # Multi-head Attention
        _, tgt_attn_weights = self.tgt_attn(
            query=query_tgt,
            key=node_embeddings_proj,
            value=node_embeddings_proj,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        tgt_logits = tgt_attn_weights.squeeze(1)  # [batch, num_nodes]
        
        # ============ Step 4: 应用动作掩码 ============
        if valid_pairs_list is not None:
            src_logits, tgt_logits = self._apply_action_mask(
                src_logits, tgt_logits, valid_pairs_list
            )
        
        return src_logits, tgt_logits, valid_pairs_list
    
    def _get_valid_pairs_batch(self, action_type, edge_index, node_mask):
        """
        批量获取有效的 (src, tgt) 对
        
        Args:
            action_type: [batch]
            edge_index: [batch, 2, num_edges]
            node_mask: [batch, num_nodes]
        
        Returns:
            valid_pairs_list: List[Tensor]
                每个元素是 [num_valid, 2]
        """
        batch_size = action_type.size(0)
        valid_pairs_list = []
        
        for b in range(batch_size):
            valid_pairs = self._get_valid_pairs(
                action_type[b].item(),
                edge_index[b],
                node_mask[b] if node_mask is not None else None
            )
            valid_pairs_list.append(valid_pairs)
        
        return valid_pairs_list
    
    def _get_valid_pairs(self, action_type, edge_index, node_mask):
        """
        根据 action_type 获取有效的 (src, tgt) 对
        
        Args:
            action_type: int (0-6)
            edge_index: [2, num_edges]
            node_mask: [num_nodes]
        
        Returns:
            valid_pairs: [num_valid, 2]
        
        详细规则:
        
        0: Delete Bond
           - 只能删除存在的边
           - valid_pairs = edge_index.t()
        
        1: Change Bond
           - 只能改变存在的边
           - valid_pairs = edge_index.t()
        
        2: Add Bond
           - 可以在任意两个节点之间添加边（除了已存在的）
           - valid_pairs = all_pairs - existing_edges
        
        3: Attach Group
           - 可以在任意节点上添加基团
           - valid_pairs = [(i, i) for i in range(num_nodes)]
        
        4: Leave Group
           - 可以从任意节点移除基团
           - valid_pairs = [(i, i) for i in range(num_nodes)]
        
        5: Change Atom
           - 可以改变任意节点的手性
           - valid_pairs = [(i, i) for i in range(num_nodes)]
        
        6: Terminate
           - 不需要 src/tgt
           - valid_pairs = [(0, 0)]
        """
        device = edge_index.device
        
        if node_mask is not None:
            num_nodes = node_mask.sum().item()
        else:
            num_nodes = edge_index.max().item() + 1
        
        if action_type == 0:  # Delete Bond
            valid_pairs = edge_index.t()  # [num_edges, 2]
        
        elif action_type == 1:  # Change Bond
            valid_pairs = edge_index.t()
        
        elif action_type == 2:  # Add Bond
            # 所有可能的节点对
            all_pairs = torch.combinations(
                torch.arange(num_nodes, device=device), 
                r=2
            )  # [num_nodes*(num_nodes-1)/2, 2]
            
            # 已存在的边
            existing_edges = set(
                tuple(sorted([src.item(), tgt.item()]))
                for src, tgt in edge_index.t().tolist()
            )
            
            # 过滤掉已存在的边
            valid_pairs = torch.stack([
                pair for pair in all_pairs
                if tuple(sorted(pair.tolist())) not in existing_edges
            ]) if len(all_pairs) > 0 else torch.empty(0, 2, device=device)
        
        elif action_type in [3, 4, 5]:  # Attach/Leave Group, Change Atom
            # 可以在任意节点上操作 (src = tgt)
            valid_pairs = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, 2)
            # [[0, 0], [1, 1], [2, 2], ...]
        
        elif action_type == 6:  # Terminate
            # 不需要 src/tgt
            valid_pairs = torch.tensor([[0, 0]], device=device)
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return valid_pairs
    
    def _apply_action_mask(self, src_logits, tgt_logits, valid_pairs_list):
        """
        应用动作掩码
        
        Args:
            src_logits: [batch, num_nodes]
            tgt_logits: [batch, num_nodes]
            valid_pairs_list: List[Tensor]
        
        Returns:
            src_logits: [batch, num_nodes] (masked)
            tgt_logits: [batch, num_nodes] (masked)
        
        功能:
            将无效的 src/tgt 的 logit 设为 -inf
            这样在 softmax 后概率为 0
        """
        batch_size, num_nodes = src_logits.size()
        
        for b in range(batch_size):
            valid_pairs = valid_pairs_list[b]  # [num_valid, 2]
            
            if len(valid_pairs) == 0:
                # 没有有效对，全部mask
                src_logits[b] = float('-inf')
                tgt_logits[b] = float('-inf')
                continue
            
            # 提取有效的 src 和 tgt
            valid_src = valid_pairs[:, 0].unique()  # [num_valid_src]
            valid_tgt = valid_pairs[:, 1].unique()  # [num_valid_tgt]
            
            # 创建掩码
            src_mask = torch.ones(num_nodes, dtype=torch.bool, device=src_logits.device)
            src_mask[valid_src] = False  # 有效位置设为 False
            
            tgt_mask = torch.ones(num_nodes, dtype=torch.bool, device=tgt_logits.device)
            tgt_mask[valid_tgt] = False
            
            # 应用掩码
            src_logits[b].masked_fill_(src_mask, float('-inf'))
            tgt_logits[b].masked_fill_(tgt_mask, float('-inf'))
        
        return src_logits, tgt_logits
    
    def sample(
        self, 
        decoder_state, 
        node_embeddings, 
        edge_index,
        action_type,
        node_mask=None,
        deterministic=False
    ):
        """
        采样 (src_idx, tgt_idx)
        
        Args:
            decoder_state: [batch, 512]
            node_embeddings: [batch, num_nodes, 256]
            edge_index: [batch, 2, num_edges]
            action_type: [batch]
            deterministic: 是否使用确定性策略
        
        Returns:
            src_idx: [batch]
            tgt_idx: [batch]
            log_prob: [batch]
        """
        # 获取 logits
        src_logits, tgt_logits, valid_pairs_list = self.forward(
            decoder_state, node_embeddings, edge_index, action_type, node_mask
        )
        
        if deterministic:
            # 贪心
            src_idx = src_logits.argmax(dim=-1)
            tgt_idx = tgt_logits.argmax(dim=-1)
            
            src_prob = F.softmax(src_logits, dim=-1)
            tgt_prob = F.softmax(tgt_logits, dim=-1)
            
            log_prob = (
                torch.log(src_prob.gather(1, src_idx.unsqueeze(1)) + 1e-10).squeeze(1) +
                torch.log(tgt_prob.gather(1, tgt_idx.unsqueeze(1)) + 1e-10).squeeze(1)
            )
        else:
            # 随机采样
            src_dist = Categorical(logits=src_logits)
            tgt_dist = Categorical(logits=tgt_logits)
            
            src_idx = src_dist.sample()
            tgt_idx = tgt_dist.sample()
            
            log_prob = src_dist.log_prob(src_idx) + tgt_dist.log_prob(tgt_idx)
        
        # 验证有效性 (可选)
        # self._verify_validity(src_idx, tgt_idx, valid_pairs_list)
        
        return src_idx, tgt_idx, log_prob
```

**关键点总结**：

1. **条件化预测**：
   - src 的预测条件化在 action_type 上
   - tgt 的预测条件化在 action_type 和 src_idx 上

2. **动作掩码**：
   - 根据 action_type 和当前图结构，动态生成有效候选
   - 将无效候选的 logit 设为 -inf

3. **注意力机制**：
   - 使用 Multi-head Attention 计算每个节点的重要性
   - Query 是 decoder_state + action_emb + src_emb
   - Key/Value 是节点嵌入

---

### 4.4 子模块 3: Label Decoder

```python
class UnifiedLabelDecoder(nn.Module):
    """
    统一的 Label 解码器
    
    功能: 预测 label 序列
    
    根据 action_type 不同:
        - Bond (0,1,2): 预测键类型 [SINGLE, DOUBLE, ...]
        - Atom (5): 预测手性 [CW, CCW, NONE]
        - Group (3,4): 预测 SMILES 序列 [*, C, (, =, O, ), ...]
    
    输入:
        - decoder_state: [batch, 512]
        - action_type: [batch] (可选, 用于条件化)
        - target_seq: [batch, seq_len] (训练时)
    
    输出:
        - label_logits: [batch, max_len, vocab_size]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(
            config.label_vocab_size,      # 5000
            config.decoder_hidden_dim     # 512
        )
        
        # Action type embedding (用于条件化)
        self.action_embedding = nn.Embedding(
            config.num_actions,           # 7
            config.decoder_hidden_dim     # 512
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.decoder_hidden_dim, 
            config.max_label_len
        )
        
        # Transformer Decoder (3 layers)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.decoder_hidden_dim,
                nhead=config.decoder_num_heads,
                dim_feedforward=config.decoder_hidden_dim * 4,
                dropout=config.decoder_dropout,
                batch_first=True
            )
            for _ in range(3)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(
            config.decoder_hidden_dim,
            config.label_vocab_size
        )
        
        self.dropout = nn.Dropout(config.decoder_dropout)
    
    def forward(
        self, 
        decoder_state, 
        action_type=None,
        target_seq=None, 
        max_len=None
    ):
        """
        前向传播
        
        示例:
        
        Case 1: Bond label (action_type=0, Delete Bond)
            target_seq: [batch, 1] = [[SINGLE]]
            输出: [batch, 1, vocab_size]
        
        Case 2: Group label (action_type=3, Attach Group)
            target_seq: [batch, 7] = [[BOS, *, C, (, =, O, ), C]]
            输出: [batch, 7, vocab_size]
        """
        batch_size = decoder_state.size(0)
        
        # 条件化在 action_type 上
        if action_type is not None:
            action_emb = self.action_embedding(action_type)  # [batch, 512]
            context = decoder_state + action_emb
        else:
            context = decoder_state
        
        context = context.unsqueeze(1)  # [batch, 1, 512] (作为 memory)
        
        if target_seq is not None:
            # ========== 训练模式 (Teacher Forcing) ==========
            seq_len = target_seq.size(1)
            
            # Embed tokens
            tgt_emb = self.embedding(target_seq)  # [batch, seq_len, 512]
            tgt_emb = self.pos_encoding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)
            
            # Causal mask
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_emb.device)
            
            # Transformer decoder
            output = tgt_emb
            for layer in self.layers:
                output = layer(
                    tgt=output,
                    memory=context,
                    tgt_mask=tgt_mask
                )
            
            # Output projection
            logits = self.output_proj(output)  # [batch, seq_len, vocab_size]
            
            return logits
        
        else:
            # ========== 推理模式 (Autoregressive) ==========
            if max_len is None:
                max_len = self.config.max_label_len
            
            # Initialize with BOS token
            input_tokens = torch.full(
                (batch_size, 1),
                fill_value=2,  # BOS token id
                dtype=torch.long,
                device=decoder_state.device
            )
            
            all_logits = []
            
            for step in range(max_len):
                # Embed current sequence
                tgt_emb = self.embedding(input_tokens)
                tgt_emb = self.pos_encoding(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                
                # Causal mask
                seq_len = input_tokens.size(1)
                tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_emb.device)
                
                # Transformer decoder
                output = tgt_emb
                for layer in self.layers:
                    output = layer(tgt=output, memory=context, tgt_mask=tgt_mask)
                
                # Get last token logits
                last_logits = self.output_proj(output[:, -1, :])  # [batch, vocab_size]
                all_logits.append(last_logits)
                
                # Greedy decoding
                next_token = last_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
                
                # Append to sequence
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
                
                # Stop if all sequences predict EOS
                if (next_token == 3).all():  # EOS token id
                    break
            
            # Stack logits
            logits = torch.stack(all_logits, dim=1)  # [batch, seq_len, vocab_size]
            
            return logits
    
    def sample(self, decoder_state, action_type=None, deterministic=False, max_len=None):
        """
        采样 label 序列
        
        Args:
            decoder_state: [batch, 512]
            action_type: [batch]
            deterministic: 是否使用确定性策略（贪心）
        
        Returns:
            label_seq: [batch, seq_len]
            log_prob: [batch]
        """
        batch_size = decoder_state.size(0)
        
        if action_type is not None:
            action_emb = self.action_embedding(action_type)
            context = (decoder_state + action_emb).unsqueeze(1)
        else:
            context = decoder_state.unsqueeze(1)
        
        if max_len is None:
            max_len = self.config.max_label_len
        
        # Initialize
        input_tokens = torch.full(
            (batch_size, 1),
            fill_value=2,  # BOS
            dtype=torch.long,
            device=decoder_state.device
        )
        
        all_log_probs = []
        
        for step in range(max_len):
            # Embed
            tgt_emb = self.embedding(input_tokens)
            tgt_emb = self.pos_encoding(tgt_emb)
            
            # Decode
            seq_len = input_tokens.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_emb.device)
            
            output = tgt_emb
            for layer in self.layers:
                output = layer(tgt=output, memory=context, tgt_mask=tgt_mask)
            
            # Predict
            logits = self.output_proj(output[:, -1, :])  # [batch, vocab_size]
            
            if deterministic:
                next_token = logits.argmax(dim=-1, keepdim=True)
                log_prob = F.log_softmax(logits, dim=-1).gather(
                    1, next_token
                ).squeeze(1)
            else:
                dist = Categorical(logits=logits)
                next_token = dist.sample().unsqueeze(1)
                log_prob = dist.log_prob(next_token.squeeze(1))
            
            all_log_probs.append(log_prob)
            
            # Append
            input_tokens = torch.cat([input_tokens, next_token], dim=1)
            
            # Stop if EOS
            if (next_token == 3).all():
                break
        
        # Remove BOS
        label_seq = input_tokens[:, 1:]  # [batch, seq_len]
        
        # Total log prob
        total_log_prob = torch.stack(all_log_probs, dim=1).sum(dim=1)  # [batch]
        
        return label_seq, total_log_prob
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
```

**关键点**：
- 使用 Transformer Decoder 生成序列
- 训练时使用 Teacher Forcing
- 推理时自回归生成
- 条件化在 action_type 上（不同类型的 label 共享解码器）

---

## 五、Module 4: Critic Network

```python
class CriticNetwork(nn.Module):
    """
    价值网络 (Critic)
    
    功能: 评估状态的价值 V(s)
    
    输入:
        - decoder_state: [batch, 512]
    
    输出:
        - value: [batch, 1]
    
    价值的含义:
        V(s) = 从状态 s 开始，按照当前策略，预期能获得的累积奖励
    
    示例:
        状态 s: 已执行 2 步编辑
        V(s) = 8.5
        
        解释: 从这个状态开始，如果按照当前策略继续执行，
             预期最终能获得 8.5 的奖励
    """
    
    def __init__(self, config):
        super().__init__()
        
        # MLP
        self.value_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_hidden_dim // 2, config.decoder_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_hidden_dim // 4, 1)
        )
    
    def forward(self, decoder_state):
        """
        前向传播
        
        示例:
            decoder_state: [2, 512]
            
            输出:
            value: [2, 1]
            [[8.5],   # 样本1: 状态价值 8.5
             [3.2]]   # 样本2: 状态价值 3.2
        """
        value = self.value_head(decoder_state)  # [batch, 1]
        return value
```

---

## 六、Module 5: Q-Network

```python
class QNetwork(nn.Module):
    """
    动作价值网络 (Q-Network)
    
    功能: 评估动作的价值 Q(s, a)
    
    输入:
        - decoder_state: [batch, 512]
        - action: {
            'action_type': [batch],
            'src_idx': [batch],
            'tgt_idx': [batch]
          }
    
    输出:
        - q_value: [batch, 1]
    
    Q值的含义:
        Q(s, a) = 在状态 s 执行动作 a，然后按照当前策略，预期能获得的累积奖励
    
    示例:
        状态 s: 已执行 2 步编辑
        动作 a: Delete Bond(2, 3)
        Q(s, a) = 9.2
        
        解释: 在这个状态下执行 Delete Bond(2,3)，
             然后继续按当前策略执行，预期能获得 9.2 的奖励
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Action embedding
        self.action_type_emb = nn.Embedding(config.num_actions, 128)
        self.src_idx_emb = nn.Embedding(1000, 128)  # max 1000 atoms
        self.tgt_idx_emb = nn.Embedding(1000, 128)
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim + 128 * 3, config.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden_dim // 2, 1)
        )
    
    def forward(self, decoder_state, action):
        """
        前向传播
        
        示例:
            decoder_state: [2, 512]
            action: {
                'action_type': [2] = [0, 1],  # Delete Bond, Change Bond
                'src_idx': [2] = [2, 3],
                'tgt_idx': [2] = [3, 4]
            }
            
            输出:
            q_value: [2, 1]
            [[9.2],   # 样本1: Q(s, Delete Bond(2,3)) = 9.2
             [5.8]]   # 样本2: Q(s, Change Bond(3,4)) = 5.8
        """
        # Embed action components
        action_type_emb = self.action_type_emb(action['action_type'])  # [batch, 128]
        src_emb = self.src_idx_emb(action['src_idx'])                  # [batch, 128]
        tgt_emb = self.tgt_idx_emb(action['tgt_idx'])                  # [batch, 128]
        
        # Concatenate state and action
        combined = torch.cat([
            decoder_state,      # [batch, 512]
            action_type_emb,    # [batch, 128]
            src_emb,            # [batch, 128]
            tgt_emb             # [batch, 128]
        ], dim=-1)  # [batch, 512 + 384]
        
        # Predict Q-value
        q_value = self.q_head(combined)  # [batch, 1]
        
        return q_value
```

---

## 七、完整的 Actor 使用示例

```python
def example_actor_forward():
    """
    演示 Actor Network 的完整前向传播
    """
    # 配置
    config = ModelConfig(
        node_feat_dim=128,
        graph_hidden_dim=256,
        decoder_hidden_dim=512,
        num_actions=7,
        label_vocab_size=5000
    )
    
    # 初始化模型
    graph_encoder = GraphEncoder(config)
    state_tracker = StateTracker(config)
    actor = ActorNetwork(config)
    
    # ========== 模拟输入 ==========
    batch_size = 2
    num_nodes = 10
    num_edges = 15
    
    # 产物图
    node_feat = torch.randn(batch_size * num_nodes, 128)
    edge_index = torch.randint(0, num_nodes, (2, num_edges * batch_size))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # 历史编辑 (假设已执行 2 步)
    history_edits = torch.randn(batch_size, 2, 512)
    
    # ========== Step 1: 编码图 ==========
    print("Step 1: Encoding graph...")
    node_embeddings, graph_embedding = graph_encoder(
        node_feat, edge_index, batch=batch
    )
    print(f"  Node embeddings: {node_embeddings.shape}")  # [20, 256]
    print(f"  Graph embedding: {graph_embedding.shape}")  # [2, 256]
    
    # Reshape for batch
    node_embeddings_batched = node_embeddings.view(batch_size, num_nodes, -1)
    print(f"  Node embeddings (batched): {node_embeddings_batched.shape}")  # [2, 10, 256]
    
    # ========== Step 2: 追踪状态 ==========
    print("\nStep 2: Tracking state...")
    decoder_state = state_tracker(
        history_edits, 
        node_embeddings_batched
    )
    print(f"  Decoder state: {decoder_state.shape}")  # [2, 512]
    
    # ========== Step 3: Actor 预测 ==========
    print("\nStep 3: Actor prediction...")
    
    # 重构 edge_index 为 batch 格式
    edge_index_batched = edge_index.view(batch_size, 2, -1)
    
    outputs = actor(
        decoder_state,
        node_embeddings_batched,
        edge_index_batched
    )
    
    print(f"  Action type logits: {outputs['action_type_logits'].shape}")  # [2, 7]
    print(f"  Src logits: {outputs['src_logits'].shape}")                  # [2, 10]
    print(f"  Tgt logits: {outputs['tgt_logits'].shape}")                  # [2, 10]
    print(f"  Label logits: {outputs['label_logits'].shape}")              # [2, max_len, 5000]
    
    # ========== Step 4: 采样动作 ==========
    print("\nStep 4: Sampling actions...")
    
    # 采样 action_type
    action_type_dist = Categorical(logits=outputs['action_type_logits'])
    action_type = action_type_dist.sample()
    print(f"  Sampled action types: {action_type}")  # [2]
    print(f"  Action type 0: {['Delete Bond', 'Change Bond', 'Add Bond', 'Attach Group', 'Leave Group', 'Change Atom', 'Terminate'][action_type[0]]}")
    print(f"  Action type 1: {['Delete Bond', 'Change Bond', 'Add Bond', 'Attach Group', 'Leave Group', 'Change Atom', 'Terminate'][action_type[1]]}")
    
    # 采样 src, tgt
    src_dist = Categorical(logits=outputs['src_logits'])
    tgt_dist = Categorical(logits=outputs['tgt_logits'])
    
    src_idx = src_dist.sample()
    tgt_idx = tgt_dist.sample()
    
    print(f"  Sampled src indices: {src_idx}")  # [2]
    print(f"  Sampled tgt indices: {tgt_idx}")  # [2]
    
    # 采样 label
    label_dist = Categorical(logits=outputs['label_logits'][:, 0, :])  # 简化: 只看第一个token
    label = label_dist.sample()
    
    print(f"  Sampled labels: {label}")  # [2]
    
    # ========== Step 5: 计算 log prob ==========
    print("\nStep 5: Computing log probabilities...")
    
    action_log_prob = action_type_dist.log_prob(action_type)
    src_log_prob = src_dist.log_prob(src_idx)
    tgt_log_prob = tgt_dist.log_prob(tgt_idx)
    label_log_prob = label_dist.log_prob(label)
    
    total_log_prob = action_log_prob + src_log_prob + tgt_log_prob + label_log_prob
    
    print(f"  Action log prob: {action_log_prob}")
    print(f"  Src log prob: {src_log_prob}")
    print(f"  Tgt log prob: {tgt_log_prob}")
    print(f"  Label log prob: {label_log_prob}")
    print(f"  Total log prob: {total_log_prob}")
    
    # ========== Step 6: 完整的编辑操作 ==========
    print("\nStep 6: Complete edit operations...")
    
    for b in range(batch_size):
        print(f"\n  Sample {b}:")
        print(f"    Action: {['Delete Bond', 'Change Bond', 'Add Bond', 'Attach Group', 'Leave Group', 'Change Atom', 'Terminate'][action_type[b]]}")
        print(f"    Src: {src_idx[b].item()}")
        print(f"    Tgt: {tgt_idx[b].item()}")
        print(f"    Label: {label[b].item()}")
        print(f"    Log prob: {total_log_prob[b].item():.4f}")

if __name__ == '__main__':
    example_actor_forward()
```

---

## 八、总结

### 8.1 Actor Network 的核心流程

```
输入: decoder_state [batch, 512]
  ↓
┌─────────────────────────────────────────┐
│ Step 1: Action Type Predictor           │
│   - MLP(decoder_state)                  │
│   - 输出: action_type_logits [batch, 7] │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Step 2: Pointer Network                 │
│   - 条件化在 action_type 上              │
│   - Multi-head Attention                │
│   - 动作掩码                             │
│   - 输出: src_logits, tgt_logits        │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Step 3: Label Decoder                   │
│   - 条件化在 action_type 上              │
│   - Transformer Decoder                 │
│   - 自回归生成                           │
│   - 输出: label_logits                  │
└─────────────────────────────────────────┘
  ↓
输出: 完整的编辑操作
```

### 8.2 关键技术点

| 技术 | 作用 | 位置 |
|------|------|------|
| **Multi-head Attention** | 计算节点重要性 | Pointer Network |
| **条件化 (Conditioning)** | 根据 action_type 调整预测 | 所有模块 |
| **动作掩码 (Action Mask)** | 保证预测有效 | Pointer Network |
| **自回归生成 (Autoregressive)** | 序列生成 | Label Decoder |
| **Teacher Forcing** | 训练加速 | Label Decoder |
| **Causal Mask** | 防止看到未来 | State Tracker, Label Decoder |