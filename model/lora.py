"""
lora.py —— LoRA 实现与注入工具

核心组件：
  LoRALinear   : 替换 nn.Linear，添加低秩旁路 A·B
  inject_lora  : 递归扫描模型，将指定模块的 Linear 替换为 LoRALinear
  freeze_non_lora : 冻结所有非 LoRA 参数
  get_lora_params : 提取 LoRA 参数（用于优化器）
  save_lora / load_lora : 只保存/加载 LoRA 增量权重

数据流：
  原始权重 W [out, in]  （冻结）
  LoRA 旁路：x → A [r, in] → B [out, r] → ΔW·x = B(Ax) * (α/r)
  输出：W·x + ΔW·x
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import math

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
#  LoRA Linear 层
# ══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    带 LoRA 旁路的 Linear 层

    forward: y = x @ W.T + x @ A.T @ B.T * (alpha / r)
                = Linear(x) + LoRA(x)

    Args:
      in_features  : 输入维度
      out_features : 输出维度
      r            : LoRA 秩（建议 8~16）
      alpha        : 缩放因子（建议 = 2r）
      dropout      : LoRA 路径的 dropout（防止过拟合）
    """
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        bias         : bool  = True,
        r            : int   = 8,
        alpha        : float = 16.0,
        dropout      : float = 0.0,
    ):
        super().__init__()
        self.r     = r
        self.scale = alpha / r

        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None
        )

        # LoRA 旁路（可训练）
        self.lora_A   = nn.Parameter(torch.empty(r, in_features))
        self.lora_B   = nn.Parameter(torch.zeros(out_features, r))
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_lora()

    def _init_lora(self):
        # A: 高斯初始化；B: 零初始化 → 初始 ΔW = 0，不破坏预训练输出
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scale
        return base_out + lora_out

    @classmethod
    def from_linear(
        cls,
        linear : nn.Linear,
        r      : int   = 8,
        alpha  : float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """从已有 nn.Linear 创建 LoRALinear，复制原始权重"""
        has_bias = linear.bias is not None
        lora_linear = cls(
            in_features  = linear.in_features,
            out_features = linear.out_features,
            bias         = has_bias,
            r            = r,
            alpha        = alpha,
            dropout      = dropout,
        )
        # 复制原始权重（冻结）
        lora_linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora_linear.bias.data.copy_(linear.bias.data)
        return lora_linear

    def merge_weights(self) -> nn.Linear:
        """
        将 LoRA 增量合并回原始权重（推理加速用）
        返回合并后的普通 nn.Linear
        """
        merged_weight = self.weight + self.lora_B @ self.lora_A * self.scale
        linear = nn.Linear(
            self.weight.size(1), self.weight.size(0),
            bias=self.bias is not None
        )
        linear.weight.data.copy_(merged_weight)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear


# ══════════════════════════════════════════════════════════════════════
#  LoRA 注入工具
# ══════════════════════════════════════════════════════════════════════

# 默认注入目标模块名（与 actor_pretrainer.py 模块名对应）
DEFAULT_LORA_TARGETS: Set[str] = {
    "state_proj",
    "state_tracker",      # 内部 Linear 会被递归替换
    "action_predictor",
    "pointer_network",
    "label_decoder",
    # "graph_encoder" 不在列表中 → 保持冻结
}


def inject_lora(
    model       : nn.Module,
    target_modules : Set[str],
    r           : int   = 8,
    alpha       : float = 16.0,
    dropout     : float = 0.05,
) -> nn.Module:
    """
    递归扫描 model，将 target_modules 中模块下的所有 nn.Linear
    替换为 LoRALinear。

    注意：GRU 的 weight_ih/weight_hh 不是 nn.Linear，跳过（不注入）。
    """
    def _inject_recursive(module: nn.Module, module_name: str):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name

            # 检查当前子模块是否在注入目标中（前缀匹配）
            in_target = any(
                full_name == t or full_name.startswith(t + ".")
                for t in target_modules
            )

            if isinstance(child, nn.Linear) and in_target:
                # 替换为 LoRALinear
                setattr(module, child_name, LoRALinear.from_linear(
                    child, r=r, alpha=alpha, dropout=dropout
                ))
            else:
                # 递归处理子模块
                _inject_recursive(child, full_name)

    _inject_recursive(model, "")
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """
    冻结所有非 LoRA 参数（lora_A / lora_B 保持可训练）
    调用时机：inject_lora 之后
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """提取所有 LoRA 可训练参数（传给优化器）"""
    return [p for n, p in model.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


def lora_param_count(model: nn.Module) -> Dict[str, int]:
    """统计 LoRA 参数量 vs 总参数量"""
    total  = sum(p.numel() for p in model.parameters())
    lora   = sum(p.numel() for n, p in model.named_parameters()
                 if "lora_A" in n or "lora_B" in n)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "total"    : total,
        "lora"     : lora,
        "frozen"   : frozen,
        "trainable": total - frozen,
        "lora_ratio": f"{lora / total * 100:.2f}%",
    }


# ══════════════════════════════════════════════════════════════════════
#  LoRA 权重保存 / 加载（只存增量，不存全量）
# ══════════════════════════════════════════════════════════════════════

def save_lora(model: nn.Module, path: str) -> None:
    """只保存 LoRA 增量权重（文件极小）"""
    lora_state = {
        name: param.data
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} LoRA tensors → {path}")


def load_lora(model: nn.Module, path: str, strict: bool = True) -> None:
    """加载 LoRA 增量权重到已注入 LoRA 的模型"""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = [], []
    model_state = dict(model.named_parameters())

    for name, data in lora_state.items():
        if name in model_state:
            model_state[name].data.copy_(data)
        else:
            unexpected.append(name)

    if strict and unexpected:
        raise RuntimeError(f"[LoRA] Unexpected keys: {unexpected}")
    print(f"[LoRA] Loaded {len(lora_state)} LoRA tensors from {path}")
    if unexpected:
        print(f"[LoRA] Unexpected keys (ignored): {unexpected}")