# Actor Network说明

为了对 Actor Network 的三个预测器（Action Type, Pointer, Label）进行预训练，我们需要将原本“一个反应对应一个完整的编辑序列”的数据，**展开为“单步决策”的马尔可夫决策过程（MDP）格式**。

也就是说，如果一个反应有 3 步编辑 $[e_1, e_2, e_3]$，我们需要将其拆分为 3 条独立的训练样本：
1. **输入**: 产物图 + 历史 `[]` $\rightarrow$ **输出目标**: $e_1$
2. **输入**: 产物图 + 历史 `[e_1]` $\rightarrow$ **输出目标**: $e_2$
3. **输入**: 产物图 + 历史 `[e_1, e_2]` $\rightarrow$ **输出目标**: $e_3$


```ini
step=0: history_edits = []               → 预测第1个edit
step=1: history_edits = [edit_0]         → 预测第2个edit
step=2: history_edits = [edit_0, edit_1] → 预测第3个edit
```


## 核心文件与功能速览

整个项目遵循非常标准的数据驱动型深度学习架构。

| **文件名称** | **核心职责** | **关键类 / 函数** |
| :--- | :--- | :--- |
| `config.py` | **超参数大管家**：统一管理维度、动作数量等，防止越界报错。 | `MODEL_CONFIG`, `TRAIN_CONFIG` |
| `process_edits_to_steps.py` | **数据展开**：将一整条反应拆解为多步独立的训练样本（Teacher Forcing）。 | `expand_reaction()` |
| `tokenizer.py` | **文本分词**：处理化学操作和基团 SMILES 的编码与解码。 | `LabelTokenizer`, `VocabBuilder` |
| `ssr_graph_pretrain_dataset.py` | **图数据加载**：将 SMILES 转为 PyG 图结构，并填充历史动作。 | `smiles_to_graph()`, `SSRGraphDataset` |
| `actor_pretrainer.py` | **模型大脑**：定义了从图编码到三大预测任务的完整网络结构。 | `ActorPretrainer`, `PointerNetwork` |
| `pretrain.py` | **训练主循环**：控制分阶段课程学习、多任务损失加权及模型保存。 | `UncertaintyWeighting`, `STAGES` |

*上面的表格是整个项目的骨架，接下来我们深入到具体的血肉中去。*



## 数据流转：从分子到张量

模型的输入格式

### 1. 反应拆解 
原始数据是一整条反应（包含多个编辑步骤）。`process_edits_to_steps.py` 会将其拆解。
假设一个反应有 3 步操作，它会被拆成 3 条独立样本：
- **Step 0**: 历史 `[]` 预测 `Action 1`
- **Step 1**: 历史 `[Action 1]` 预测 `Action 2`
- **Step 2**: 历史 `[Action 1, Action 2]` 预测 `Action 3`

### 2. 图结构构建 (Graph Construction)
在 `ssr_graph_pretrain_dataset.py` 中，`smiles_to_graph` 函数将目标分子的 SMILES 字符串转换为 PyTorch Geometric (PyG) 能识别的图：
- **节点特征 (`x`)**：原子的属性（如原子序数、度数等），补齐到 128 维。
- **边索引 (`edge_index`)**：记录哪些原子之间有化学键。

### 3. 标签分词 (Tokenization)
`tokenizer.py` 负责将字符串标签（如 `[ChangeBond]`, `[SINGLE]`, 或基团的 SMILES）转换为整数 ID 序列，方便模型进行交叉熵计算。

---

## 🧠 模型架构：Actor Pretrainer 拆解

`actor_pretrainer.py` 是整个项目的核心。模型接收图结构和历史操作，然后像流水线一样依次完成三个预测任务。

- **第一站：Graph Encoder (图编码器)**
  使用 GAT (图注意力网络) 提取分子特征。它会输出两个东西：每个原子的局部特征（`node_embeddings`），以及整个分子的全局特征（`graph_emb`）。
- **第二站：Simple State Tracker (状态追踪器)**
  将“历史动作序列”进行 Embedding 并求和，与分子的全局特征相加，形成当前的**上下文状态 (`decoder_state`)**。这告诉模型：“基于这个分子，且我已经做过这些操作了，现在状态如何？”
- **第三站：三大预测头 (Prediction Heads)**
  模型基于当前状态，依次进行三项预测：

  1. **Action Predictor**: 预测动作类型（如“删键”、“换原子”等 7 分类）。
  2. **Pointer Network**: 预测操作发生在哪两个原子上（源节点 `src` 和目标节点 `tgt`）。它通过计算状态特征与所有原子特征的注意力得分来“指向”特定原子。
  3. **Label Decoder**: 预测具体的操作内容（如键的类型、基团的 SMILES 等）。这是一个小型的 Transformer Decoder，采用自回归方式生成序列。

---

## 🎯 训练策略：多任务与课程学习

在 `pretrain.py` 中，我们并没有简单地把所有任务混在一起盲目训练，而是使用了两种高级技巧来保证模型平稳收敛。

### 1. 动态多任务损失加权 (Uncertainty Weighting)
模型同时有 4 个 Loss（Action, Src, Tgt, Label），它们的量级和学习难度各不相同。代码中使用了 `UncertaintyWeighting` 模块，通过可学习的参数 $$\sigma_i$$ 自动平衡各项损失：
$$L = \sum_{i=1}^{4} \left( \exp(-\log \sigma_i) L_i + \log \sigma_i \right)$$
*这避免了某个 Loss 过大导致其他任务的梯度被淹没。*

### 2. 课程学习 (Curriculum Learning)
像教小孩一样，模型训练被划分为 4 个阶段（`STAGES`）：
1. **Stage 1**: 只学最简单的动作分类（Action），冻结其他模块。
2. **Stage 2**: 学习找原子（Pointer），冻结动作和标签预测。
3. **Stage 3**: 学习生成具体标签（Label）。
4. **Stage 4**: 解冻所有模块，进行联合微调（Joint Training）。

---

## 💡 上手建议

建议你按照以下顺序阅读和运行代码：

1. 先跑通 `process_edits_to_steps.py`，打开生成的 JSONL 文件，直观感受一下 Teacher Forcing 的数据长什么样。
2. 在 `actor_pretrainer.py` 的 `forward` 函数中打几个 `print(x.shape)`，观察张量维度是如何从图特征一步步变成预测 Logits 的。
3. 重点关注 `PointerNetwork` 的注意力计算逻辑，这是图到序列任务中最容易出 Bug 也是最核心的地方。
