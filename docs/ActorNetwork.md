

为了对 Actor Network 的三个预测器（Action Type, Pointer, Label）进行预训练，我们需要将原本“一个反应对应一个完整的编辑序列”的数据，**展开（Unroll）为“单步决策”的马尔可夫决策过程（MDP）格式**。

也就是说，如果一个反应有 3 步编辑 $[e_1, e_2, e_3]$，我们需要将其拆分为 3 条独立的训练样本：
1. **输入**: 产物图 + 历史 `[]` $\rightarrow$ **输出目标**: $e_1$
2. **输入**: 产物图 + 历史 `[e_1]` $\rightarrow$ **输出目标**: $e_2$
3. **输入**: 产物图 + 历史 `[e_1, e_2]` $\rightarrow$ **输出目标**: $e_3$


```ini
step=0: history_edits = []               → 预测第1个edit
step=1: history_edits = [edit_0]         → 预测第2个edit
step=2: history_edits = [edit_0, edit_1] → 预测第3个edit
```



下面是实现该功能的代码。代码分为两部分：**数据展开脚本** 和 **PyTorch Dataset 骨架**。代码设计遵循“直接、扁平、无复杂嵌套”的原则。

### 1. 数据展开处理脚本 (`prepare_pretrain_data.py`)

这个脚本将 `ssredits_extractor.py` 提取出的 JSON 数据，转换为预训练所需的单步 Input-Output 格式。

### 2. PyTorch Dataset 接口 (`dataset.py`)

处理完数据后，我们需要一个简洁的 `Dataset` 类，将这些数据直接喂给你的三个预测器。


### 数据处理后的效果展示

经过上述脚本处理后，原本嵌套的 JSON 将变成如下扁平的结构，**非常适合直接作为监督学习的输入**：

### 接下来你的工作流建议：
1. 运行 `prepare_pretrain_data.py` 处理所有的 train/val/test 数据集。
2. 编写一个 `collate_fn`，将 `product_smi` 转换为 PyTorch Geometric 的 `Data` 格式（图结构）。
3. 将 `input_history` 转换为 Tensor 序列（通过 Embedding 映射）。
4. 将这三个 Target 直接用于计算 Loss：
   - Action Type: `CrossEntropyLoss(pred_action, target_action)`
   - Pointer: `CrossEntropyLoss(pred_src, target_src) + CrossEntropyLoss(pred_tgt, target_tgt)`
   - Label: `CrossEntropyLoss(pred_label_seq, target_label_seq)`





