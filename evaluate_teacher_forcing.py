#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_teacher_forcing.py

监督预训练单步评估脚本。

评估方式：
  Teacher Forcing 条件下评估单步编辑动作预测能力。
  即模型在真实历史编辑序列 H_t 条件下，预测当前第 t 步动作。

输出指标：
  test/loss_action
  test/loss_src
  test/loss_tgt
  test/loss_label
  test/loss_total
  test/action_acc
  test/src_acc
  test/tgt_acc
  test/label_token_acc
  test/label_seq_acc
  test/edit_exact_acc

注意：
  edit_exact_acc 是“单步完整动作准确率”，不是整条路径准确率。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config.config import PATH_CFG, MODEL_CFG, TASK_NAMES
from tokenizer.tokenizer import LabelTokenizer
from data.ssr_graph_pretrain_dataset import SSRGraphDataset
from model.actor_network import ActorNetwork, TeacherForcingTargets
from model.state_tracker import HistoryBatch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(
    json_path: str,
    tokenizer: LabelTokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """构建测试集 DataLoader。"""
    dataset = SSRGraphDataset(
        json_path=str(json_path),
        tokenizer=tokenizer,
        max_seq_len=MODEL_CFG.max_seq_len,
        max_hist_len=MODEL_CFG.max_hist_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def unpack_batch(
    batch,
) -> Tuple[HistoryBatch, TeacherForcingTargets, dict, torch.Tensor]:
    """
    将 PyG DataLoader 返回的 batch 解包为 ActorNetwork.forward 所需输入。

    Returns:
      history    : HistoryBatch
      tf         : TeacherForcingTargets
      graph_kw   : dict
      target_tgt : Tensor [B]
    """
    B = int(batch.num_graphs)
    max_seq = MODEL_CFG.max_seq_len
    max_hist = MODEL_CFG.max_hist_len
    max_a = MODEL_CFG.num_actions - 1
    max_n = MODEL_CFG.max_atoms - 1

    def _reshape(t: torch.Tensor, *shape) -> torch.Tensor:
        if t.shape == torch.Size(shape):
            return t
        return t.reshape(*shape)

    # target 字段
    target_action = _reshape(batch.target_action, B).clamp(0, max_a)
    target_src = _reshape(batch.target_src, B).clamp(0, max_n)
    target_tgt = _reshape(batch.target_tgt, B).clamp(0, max_n)
    target_label = _reshape(batch.target_label, B, max_seq)

    tf = TeacherForcingTargets(
        action=target_action,
        src=target_src,
        label_seq=target_label,
    )

    # history 字段
    h_actions = _reshape(batch.history_actions, B, max_hist).clamp(0, max_a)
    h_src_idxs = _reshape(batch.history_src_idxs, B, max_hist).clamp(0, max_n)
    h_tgt_idxs = _reshape(batch.history_tgt_idxs, B, max_hist).clamp(0, max_n)
    h_label_seqs = _reshape(batch.history_label_seqs, B, max_hist, max_seq)

    history = HistoryBatch(
        actions=h_actions,
        src_idxs=h_src_idxs,
        tgt_idxs=h_tgt_idxs,
        label_seqs=h_label_seqs,
    )

    graph_kw = dict(
        x=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch,
    )

    return history, tf, graph_kw, target_tgt


def compute_raw_losses(
    action_logits: torch.Tensor,
    src_logits: torch.Tensor,
    tgt_logits: torch.Tensor,
    label_logits: torch.Tensor,
    tf: TeacherForcingTargets,
    target_tgt: torch.Tensor,
    vocab_size: int,
    criterion_cls: nn.CrossEntropyLoss,
    criterion_seq: nn.CrossEntropyLoss,
) -> Dict[str, torch.Tensor]:
    """
    计算四项原始交叉熵损失，不做特殊权重策略加权。

    action loss:
      所有样本计算。

    src/tgt/label loss:
      仅非 Terminate 样本计算。
    """
    dummy = action_logits.sum() * 0.0

    loss_action = criterion_cls(action_logits, tf.action)

    non_stop = tf.action != MODEL_CFG.stop_action_id

    if non_stop.any():
        loss_src = criterion_cls(src_logits[non_stop], tf.src[non_stop])
        loss_tgt = criterion_cls(tgt_logits[non_stop], target_tgt[non_stop])

        # label_logits: [B, L-1, V]
        # target: tf.label_seq[:, 1:]，去掉 BOS，与训练对齐
        ll = label_logits[non_stop]
        lt = tf.label_seq[non_stop, 1:]

        loss_label = criterion_seq(
            ll.reshape(-1, vocab_size),
            lt.reshape(-1),
        )
    else:
        loss_src = dummy
        loss_tgt = dummy
        loss_label = dummy

    return {
        "action": loss_action,
        "src": loss_src,
        "tgt": loss_tgt,
        "label": loss_label,
    }


def load_model(ckpt_path: str, vocab_size: int) -> ActorNetwork:
    """加载训练好的 ActorNetwork。"""
    model = ActorNetwork(vocab_size=vocab_size).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


@torch.no_grad()
def evaluate_teacher_forcing(
    model: ActorNetwork,
    loader: DataLoader,
    vocab_size: int,
) -> Dict[str, float]:
    """Teacher Forcing 条件下评估单步预测指标。"""
    model.eval()

    criterion_cls = nn.CrossEntropyLoss(ignore_index=MODEL_CFG.pad_action_id)
    criterion_seq = nn.CrossEntropyLoss(ignore_index=MODEL_CFG.pad_token_id)

    loss_sums = {name: 0.0 for name in TASK_NAMES}
    n_batches = 0

    action_correct = 0
    action_total = 0

    src_correct = 0
    tgt_correct = 0
    non_stop_total = 0

    label_token_correct = 0
    label_token_total = 0

    label_seq_correct = 0
    edit_exact_correct = 0
    sample_total = 0

    for batch in tqdm(loader, desc="[teacher-forcing-eval]"):
        batch = batch.to(device)

        history, tf, graph_kw, target_tgt = unpack_batch(batch)

        action_logits, src_logits, tgt_logits, label_logits, _ = model(
            history, tf, **graph_kw
        )

        raw_losses = compute_raw_losses(
            action_logits=action_logits,
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            label_logits=label_logits,
            tf=tf,
            target_tgt=target_tgt,
            vocab_size=vocab_size,
            criterion_cls=criterion_cls,
            criterion_seq=criterion_seq,
        )

        for k, v in raw_losses.items():
            loss_sums[k] += float(v.item())
        n_batches += 1

        # =========================
        # 1. action accuracy
        # =========================
        pred_action = action_logits.argmax(dim=-1)  # [B]
        action_ok = pred_action == tf.action        # [B]

        action_correct += action_ok.sum().item()
        action_total += tf.action.size(0)

        # =========================
        # 2. non-stop mask
        # =========================
        non_stop = tf.action != MODEL_CFG.stop_action_id
        B = tf.action.size(0)

        pred_src = src_logits.argmax(dim=-1)
        pred_tgt = tgt_logits.argmax(dim=-1)

        # label_logits: [B, L-1, V]
        pred_label = label_logits.argmax(dim=-1)
        true_label = tf.label_seq[:, 1:]

        # =========================
        # 3. src/tgt/label accuracy
        # =========================
        if non_stop.any():
            src_ok = pred_src[non_stop] == tf.src[non_stop]
            tgt_ok = pred_tgt[non_stop] == target_tgt[non_stop]

            src_correct += src_ok.sum().item()
            tgt_correct += tgt_ok.sum().item()
            non_stop_total += non_stop.sum().item()

            pred_label_ns = pred_label[non_stop]
            true_label_ns = true_label[non_stop]

            label_mask = true_label_ns != MODEL_CFG.pad_token_id

            if label_mask.any():
                label_token_correct += (
                    (pred_label_ns == true_label_ns) & label_mask
                ).sum().item()
                label_token_total += label_mask.sum().item()

            # 标签序列完全匹配：忽略 PAD
            label_seq_ok = (
                ((pred_label_ns == true_label_ns) | (~label_mask)).all(dim=1)
            )
            label_seq_correct += label_seq_ok.sum().item()

            # 非 STOP 单步完整动作准确率
            edit_exact_non_stop = (
                action_ok[non_stop]
                & src_ok
                & tgt_ok
                & label_seq_ok
            )
            edit_exact_correct += edit_exact_non_stop.sum().item()

        # =========================
        # 4. STOP 样本的 edit_exact
        # =========================
        stop = ~non_stop
        if stop.any():
            # Terminate 样本只要求 action_type 正确
            edit_exact_correct += action_ok[stop].sum().item()

        sample_total += B

    avg_losses = {
        f"test/loss_{k}": v / max(1, n_batches)
        for k, v in loss_sums.items()
    }

    loss_total = (
        avg_losses["test/loss_action"]
        + avg_losses["test/loss_src"]
        + avg_losses["test/loss_tgt"]
        + avg_losses["test/loss_label"]
    )

    results = {
        **avg_losses,
        "test/loss_total": loss_total,
        "test/action_acc": action_correct / max(1, action_total),
        "test/src_acc": src_correct / max(1, non_stop_total),
        "test/tgt_acc": tgt_correct / max(1, non_stop_total),
        "test/label_token_acc": label_token_correct / max(1, label_token_total),
        "test/label_seq_acc": label_seq_correct / max(1, non_stop_total),
        "test/edit_exact_acc": edit_exact_correct / max(1, sample_total),
        "test/non_stop_samples": float(non_stop_total),
        "test/total_samples": float(sample_total),
    }

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PATH_CFG.CKPT_BEST_MODEL_FILE),
        help="best_model.pt 路径",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=str(PATH_CFG.PRETRAIN_TEST_DATA_FILE),
        help="预训练测试集 JSON 文件路径，例如 dataset/uspto50k/pretrained/uspto50k_test_output.json",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=str(PATH_CFG.VOCAB_FILE),
        help="tokenizer/vocab.txt 路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="测试 batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )

    args = parser.parse_args()

    print(f"[teacher-forcing-eval] device = {device}")
    print(f"[teacher-forcing-eval] ckpt = {args.ckpt}")
    print(f"[teacher-forcing-eval] test_json = {args.test_json}")
    print(f"[teacher-forcing-eval] vocab_file = {args.vocab_file}")

    tokenizer = LabelTokenizer(args.vocab_file)
    loader = build_dataloader(
        json_path=args.test_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model(args.ckpt, vocab_size=tokenizer.vocab_size)

    results = evaluate_teacher_forcing(
        model=model,
        loader=loader,
        vocab_size=tokenizer.vocab_size,
    )

    print("\n[Teacher Forcing Evaluation Results]")
    for k, v in results.items():
        if k.endswith("samples"):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()