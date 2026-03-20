
# Data Processing `USPTO50k`

## Extractor
`extractor/ssredits_extractor.py`:  extracting ssredits based on reaction smiles in USPTO50K dataset. 

```bash
python extractor/ssredits_extractor.py
```

For example:
- input
```python
rxn_smi_mapped = "[CH3:1][O:2][C:3](=[O:4])[C@H:5]([CH2:6][CH2:7][CH2:8][CH2:9][NH:10][C:29](=[O:28])[O:30][CH2:31][c:32]1[cH:33][cH:34][cH:35][cH:36][cH:37]1)[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]>>[CH3:1][O:2][C:3](=[O:4])[C@H:5]([CH2:6][CH2:7][CH2:8][CH2:9][NH2:10])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"
```

- output
```json
"edits": [
        {
          "action_type": "AttachGroup",
          "src_idx": 9,
          "tgt_idx": -1,
          "label": "*C(=O)OCC1=CC=CC=C1"
        },
        {
          "action_type": "Terminate",
          "src_idx": -1,
          "tgt_idx": -1,
          "label": "Terminate"
        }
      ]
```

There are three parts with four objects in edits:

```csharp
Edit = {
    action_type: int (0-6),      # 动作类型
    src_idx: int (0 to N-1),     # 源节点索引（Atom Index）
    tgt_idx: int (0 to N-1),     # 目标节点索引（Atom Index）
    label: List[int]             # 标签序列（token ids）
}
```
1. action_type:
    ```makefile
    0: Delete Bond    - 删除键
    1: Change Bond    - 改变键类型
    2: Add Bond       - 添加键
    3: Attach Group   - 连接基团
    4: Leave Group    - 离去基团
    5: Change Atom    - 改变原子手性
    6: Terminate      - 终止信号
    ```
2. src_idx & tgt_idx: Atom index 

3. label:
    - Bond Label (action 0-2)：[SINGLE], [DOUBLE], [TRIPLE], [AROMATIC], [NONE]
    - Atom Label (action 5)：[CW], [CCW], [NONE]
    - Group Label (action 3-4)：SMILES 序列，如 *, C, (, =, O, ), C
    

## Tokenizer
`tokenizer/tokenizer.py`: get common groups in ssredits. And add groups to `atom_vocab.txt`. The final `vocab.txt` includes:

```ini
索引 0-5:      特殊Token
               [PAD], [UNK], [BOS], [EOS], [SEP], [MASK]

索引 6-12:     动作类型
               [DeleteBond], [ChangeBond], [AddBond], 
               [AttachGroup], [LeaveGroup], [ChangeAtom], [Terminate]

索引 13-17:    键类型
               [NONE], [SINGLE], [DOUBLE], [TRIPLE], [AROMATIC]

索引 18-20:    手性类型
               [NONE], [CW], [CCW]

索引 21+:      原子级Token（从 Group SMILES 中提取）
               *, C, c, N, n, O, o, S, s, (, ), =, #, 1, 2, 3, ...
```


# Model Architecture (EnzRetro-RL)

## Overall of model

```vbnet
┌─────────────────────────────────────────────────────────────┐
│                     RL-based SSR Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Product Graph                                       │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 1: Graph Encoder                             │   │
│  │  - 编码产物分子图                                       │   │
│  │  - 输出: Node Embeddings + Graph Embedding            │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 2: State Tracker (Transformer Decoder)       │   │
│  │  - 维护历史编辑序列                                     │   │
│  │  - 输出: Decoder State (当前状态表示)                   │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 3: Actor Network (Policy)                    │   │
│  │  ├─ Step 1: Action Type Predictor                    │   │
│  │  ├─ Step 2: Pointer Network (src, tgt)               │   │
│  │  └─ Step 3: Label Decoder                            │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 4: Critic Network (Value Function)           │   │
│  │  - 评估状态价值 V(s)                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Module 5: Q-Network (Action-Value Function)         │   │
│  │  - 评估动作价值 Q(s,a)                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│  Output: Edit Sequence → Reactant SMILES                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


## 1. Graph Encoder (GAT 等)

Encoding product molecule into graph


## 2. State Tracker

维护历史编辑序列，生成当前状态的表示。


- Self-Attention（历史编辑之间）：

```css
每个编辑可以关注之前的所有编辑

例如:
  Edit 1: Delete Bond(2,3)
  Edit 2: Attach Group(2, *OH)  ← 可以关注 Edit 1
  Edit 3: Change Atom(3, CW)    ← 可以关注 Edit 1, 2

这样模型可以学习编辑之间的依赖关系
```

- Cross-Attention（编辑 → 图节点）：

```markdown
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


## 3. Actor Network 

### 整体结构
Actor Network 是强化学习的核心，负责决策下一步的编辑操作。Actor Network 分为三个子模块，按顺序预测编辑的四个组成部分：


```less
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

#### 3.1 基于Teacher Forcing预训练

- 三个预测器可以独立并行训练，互不干扰
- 避免预测器1的早期错误级联污染预测器2/3的梯度
- RL 微调阶段再切换为真实的自回归预测

```yaml
原始反应（3步编辑）:
  edits = [e₁(DeleteBond,2,3,NONE), e₂(AttachGroup,2,2,*OH), e₃(Terminate)]

展开为独立训练样本：

┌─────────────────────────────────────────────────────────┐
│ Sample 1                                                 │
│  graph_input : product_graph                            │
│  history     : []                                       │
│  target_type : DeleteBond (0)      ← 预测器1 目标       │
│  target_src  : 2                   ← 预测器2 目标       │
│  target_tgt  : 3                   ← 预测器2 目标       │
│  target_label: "NONE"              ← 预测器3 目标       │
│  cond_type   : DeleteBond (GT)     ← 预测器2/3 的条件   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Sample 2                                                 │
│  graph_input : product_graph                            │
│  history     : [e₁]                                     │
│  target_type : AttachGroup (3)                          │
│  target_src  : 2                                        │
│  target_tgt  : 2                                        │
│  target_label: "*OH"                                    │
│  cond_type   : AttachGroup (GT)                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Sample 3                                                 │
│  graph_input : product_graph                            │
│  history     : [e₁, e₂]                                 │
│  target_type : Terminate (6)                            │
│  target_src  : -                   ← Terminate 无需预测  │
│  target_tgt  : -                                        │
│  target_label: -                                        │
│  cond_type   : -                                        │
└─────────────────────────────────────────────────────────┘
```
功能：维护解码器状态，生成编辑序列

架构：

- 6 层 Transformer Decoder Layer，每层包含：
    - Self-Attention：捕捉历史编辑之间的依赖关系
    - Cross-Attention：每个编辑关注整个产物图的所有节点
    - Feed-Forward Network：非线性变换

- 关键机制：
    - 因果掩码 (Causal Mask)：防止看到未来的编辑
    - 位置编码 (Positional Encoding)：编码编辑的顺序信息
    - 残差连接 + Layer Norm：稳定训练

#### 3.2 三个预测器各自的损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{type} + \mathcal{L}_{pointer} + \mathcal{L}_{label}$$

$$\mathcal{L}_{type} = \text{CrossEntropy}(\hat{y}_{type},\ y_{type})$$

$$\mathcal{L}_{pointer} = \text{CrossEntropy}(\hat{y}_{src},\ y_{src}) + \text{CrossEntropy}(\hat{y}_{tgt},\ y_{tgt})$$

$$\mathcal{L}_{label} = \frac{1}{T}\sum_{t=1}^{T}\text{CrossEntropy}(\hat{y}_{label}^t,\ y_{label}^t)$$

其中 $\mathcal{L}_{pointer}$ 和 $\mathcal{L}_{label}$ **只在非 Terminate 步骤计算**，$\mathcal{L}_{label}$ 只在需要 label 的动作类型上计算（Bond 操作、Group 操作等）。



#### 3.3 训练策略

当前是一个经典的**多任务学习（Multi-Task Learning）**问题。

本项目采用的训练策略是：**分阶段课程训练（Curriculum Learning）+自动权重平衡（Uncertainty Weighting）**,先用课程学习让各任务独立收敛，再用 Uncertainty Weighting 做联合微调，兼顾稳定性和自动化。


1. 按依赖顺序分阶段预热，最后联合微调：
```makefile
阶段1: 只训 action（最简单，先收敛）
阶段2: 冻结 action，训 src + tgt
阶段3: 冻结前两阶段，训 label
阶段4: 解冻全部，联合微调（小 lr）
```
2. 自动权重平衡,全自动，无需手调权重
```python
# Uncertainty Weighting (Kendall et al. 2018)
loss = sum(exp(-log_sigma_i) * loss_i + log_sigma_i)
# log_sigma 作为可学习参数，自动平衡各项
```







## 4. Action Predictor

预测当前步的动作类型（7 分类）


## 5. Pointer Network

预测源节点和目标节点的索引
- Additive Attention 机制
- 分别为 src 和 tgt 设计两个独立的注意力头





