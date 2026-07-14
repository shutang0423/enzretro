#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_path_generation_compare_v2.py

路径级自回归评估脚本，大改版。

新增：
  1. beam      : 基于 logits 的确定性 Beam Search
  2. beam_min  : Beam Search + 前 min_steps 步屏蔽 Terminate

保留：
  1. greedy
  2. greedy_min
  3. topn_sampling
  4. monte_carlo
  5. policy_gradient

注意：
  1. 整体路径评估必须使用 processed JSON：
     dataset/uspto50k/processed/uspto50k_test_output.json

  2. pretrained JSON 是单步展开样本，不能用于完整路径评估。

  3. 本脚本评估的是“编辑序列是否一致”，不是最终反应物 SMILES 是否一致。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.config import (
    PATH_CFG,
    MODEL_CFG,
    ACTION_TO_ID,
    ID_TO_ACTION,
    STOP_ACTION_ID,
    RLInferenceConfig,
)

from tokenizer.tokenizer import LabelTokenizer
from model.actor_network import ActorNetwork, EditStep
from model.state_tracker import HistoryBatch
from utils.chem import smiles_to_graph
from rl.rl_inference import RLInference


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 基础工具
# ============================================================

def normalize_label(label: Any) -> str:
    if label is None:
        return ""

    s = str(label).strip().replace(" ", "")

    if len(s) >= 2 and s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        if inner in {
            "NONE",
            "SINGLE",
            "DOUBLE",
            "TRIPLE",
            "AROMATIC",
            "CW",
            "CCW",
            "Terminate",
            "TERMINATE",
        }:
            s = inner

    return s


def load_reactions(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON format: {type(data)}")

    return data


def build_encoder_kwargs(product_smi: str) -> Dict[str, torch.Tensor]:
    graph = smiles_to_graph(product_smi)

    if isinstance(graph, tuple):
        x, edge_index, *rest = graph
    else:
        x = graph.x
        edge_index = graph.edge_index

    if x is None or edge_index is None:
        raise ValueError(f"Failed to parse product SMILES: {product_smi}")

    x = x.to(device)
    edge_index = edge_index.to(device)

    if edge_index.dim() == 3:
        edge_index = edge_index.squeeze(0)

    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch,
    }


def gold_path_from_reaction(reaction: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "output" not in reaction or "edits" not in reaction["output"]:
        raise ValueError(
            "当前 JSON 不包含 reaction['output']['edits']。"
            "路径级评估必须使用 dataset/uspto50k/processed/uspto50k_test_output.json。"
        )

    edits = reaction["output"]["edits"]

    gold_path: List[Dict[str, Any]] = []
    has_stop = False

    for edit in edits:
        action_name = edit["action_type"]
        if action_name not in ACTION_TO_ID:
            raise ValueError(f"Unknown action_type in gold edits: {action_name}")

        action_id = ACTION_TO_ID[action_name]

        gold_path.append({
            "action_type": action_id,
            "action_name": action_name,
            "src_idx": edit.get("src_idx", None),
            "tgt_idx": edit.get("tgt_idx", None),
            "label": normalize_label(edit.get("label", "")),
        })

        if action_id == STOP_ACTION_ID:
            has_stop = True
            break

    if not has_stop:
        gold_path.append({
            "action_type": STOP_ACTION_ID,
            "action_name": ID_TO_ACTION[STOP_ACTION_ID],
            "src_idx": None,
            "tgt_idx": None,
            "label": "",
        })

    return gold_path


def decode_label_tokens(tokenizer: LabelTokenizer, label_tokens: torch.Tensor) -> str:
    if label_tokens.dim() > 1:
        label_tokens = label_tokens.squeeze(0)

    ids = label_tokens.detach().cpu().tolist()

    try:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(ids)

    return normalize_label(text)


def pred_path_from_steps(
    pred_steps: List[EditStep],
    tokenizer: LabelTokenizer,
) -> List[Dict[str, Any]]:
    pred_path: List[Dict[str, Any]] = []

    for step in pred_steps:
        action_id = int(step.action_type.item())
        action_name = ID_TO_ACTION.get(action_id, str(action_id))

        if action_id == STOP_ACTION_ID:
            pred_path.append({
                "action_type": action_id,
                "action_name": action_name,
                "src_idx": None,
                "tgt_idx": None,
                "label": "",
            })
            break

        pred_path.append({
            "action_type": action_id,
            "action_name": action_name,
            "src_idx": int(step.src_idx.item()),
            "tgt_idx": int(step.tgt_idx.item()),
            "label": decode_label_tokens(tokenizer, step.label_tokens),
        })

    return pred_path


def path_to_key(path: List[Dict[str, Any]]) -> Tuple:
    return tuple(
        (
            step.get("action_type"),
            step.get("src_idx"),
            step.get("tgt_idx"),
            normalize_label(step.get("label")),
        )
        for step in path
    )


def deduplicate_candidates(
    candidates: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    seen = set()
    unique = []

    for cand in candidates:
        key = path_to_key(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)

    return unique


def load_model(ckpt_path: str, vocab_size: int) -> ActorNetwork:
    model = ActorNetwork(vocab_size=vocab_size).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ============================================================
# Greedy / Sampling / RL 生成
# ============================================================

@torch.no_grad()
def generate_greedy_candidates(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    encoder_kwargs: Dict[str, torch.Tensor],
    max_steps: int,
) -> List[List[Dict[str, Any]]]:
    pred_steps = model.generate(
        max_steps=max_steps,
        **encoder_kwargs,
    )

    return [pred_path_from_steps(pred_steps, tokenizer)]


@torch.no_grad()
def generate_greedy_min_candidates(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    encoder_kwargs: Dict[str, torch.Tensor],
    max_steps: int,
    min_steps: int,
) -> List[List[Dict[str, Any]]]:
    first_val = next(iter(encoder_kwargs.values()))
    device_ = first_val.device

    label_len = MODEL_CFG.max_seq_len
    label_max_len = MODEL_CFG.max_seq_len

    history = HistoryBatch.empty(1, label_len, device_)
    gru_hidden = None
    edit_steps: List[EditStep] = []

    for step_idx in range(max_steps):
        enc_out, graph_emb = model._encode(**encoder_kwargs)

        decoder_state, gru_hidden = model.state_tracker(
            graph_emb,
            history,
            gru_hidden,
        )

        action_logits = model.action_predictor(decoder_state)

        if step_idx < min_steps:
            action_logits = action_logits.clone()
            action_logits[:, STOP_ACTION_ID] = -1e9

        pred_action = action_logits.argmax(dim=-1)

        src_logits, tgt_logits = model.pointer_network(
            decoder_state=decoder_state,
            dense_nodes=enc_out.dense_nodes,
            action_type=pred_action,
            target_src_idx=None,
            node_mask=enc_out.node_pad_mask,
            has_nodes=enc_out.has_nodes,
        )

        pred_src = src_logits.argmax(dim=-1)
        pred_tgt = tgt_logits.argmax(dim=-1)

        pred_label = model.label_decoder.greedy_decode(
            decoder_state=decoder_state,
            action_type=pred_action,
            bos_token_id=model.bos_token_id,
            max_len=label_max_len,
        )

        edit = EditStep(
            action_type=pred_action,
            src_idx=pred_src,
            tgt_idx=pred_tgt,
            label_tokens=pred_label,
        )

        edit_steps.append(edit)

        if int(edit.action_type.item()) == STOP_ACTION_ID:
            break

        history = model._append_history(history, edit, label_len)

    return [pred_path_from_steps(edit_steps, tokenizer)]


@torch.no_grad()
def generate_topn_sampling_candidates(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    encoder_kwargs: Dict[str, torch.Tensor],
    max_steps: int,
    top_n: int,
    temperature: float,
    top_k: int,
    dedup: bool = True,
) -> List[List[Dict[str, Any]]]:
    results = model.generate_top_n(
        n=top_n,
        temperature=temperature,
        top_k=top_k,
        max_steps=max_steps,
        **encoder_kwargs,
    )

    candidates = [
        pred_path_from_steps(pred_steps, tokenizer)
        for pred_steps, score in results
    ]

    if dedup:
        candidates = deduplicate_candidates(candidates)

    return candidates


@torch.no_grad()
def generate_rl_candidates(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    encoder_kwargs: Dict[str, torch.Tensor],
    method: str,
    max_steps: int,
    num_rollouts: int,
    temperature: float,
    discount_factor: float,
    beam_size: int = 1,
) -> List[List[Dict[str, Any]]]:
    rl_config = RLInferenceConfig(
        inference_method=method,
        max_steps=max_steps,
        num_rollouts=num_rollouts,
        temperature=temperature,
        discount_factor=discount_factor,
    )

    rl_infer = RLInference(model, rl_config)

    result = rl_infer.infer(
        encoder_kwargs=encoder_kwargs,
        target_steps=None,
        beam_size=beam_size,
    )

    if result is None or len(result) == 0:
        return []

    # RLInference.infer 通常返回 List[List[EditStep]]
    if isinstance(result[0], EditStep):
        return [pred_path_from_steps(result, tokenizer)]

    return [
        pred_path_from_steps(steps, tokenizer)
        for steps in result
    ]


# ============================================================
# 新增：真正的 Beam Search
# ============================================================

@dataclass
class BeamState:
    edit_steps: List[EditStep]
    score: float
    history: HistoryBatch
    gru_hidden: Optional[torch.Tensor]
    finished: bool


def make_stop_edit(model: ActorNetwork, device_: torch.device) -> EditStep:
    action = torch.tensor([STOP_ACTION_ID], dtype=torch.long, device=device_)
    src = torch.zeros(1, dtype=torch.long, device=device_)
    tgt = torch.zeros(1, dtype=torch.long, device=device_)
    label = torch.full(
        (1, 1),
        model.cfg.pad_token_id,
        dtype=torch.long,
        device=device_,
    )
    return EditStep(action_type=action, src_idx=src, tgt_idx=tgt, label_tokens=label)


@torch.no_grad()
def expand_one_beam_state(
    model: ActorNetwork,
    cand: BeamState,
    encoder_kwargs: Dict[str, torch.Tensor],
    step_idx: int,
    min_steps: int,
    top_a: int,
    top_s: int,
    top_t: int,
    temperature: float,
) -> List[BeamState]:
    """
    对一个 beam candidate 扩展：
      top action × top src × top tgt

    label 采用 greedy_decode，避免组合空间过大。
    """
    first_val = next(iter(encoder_kwargs.values()))
    device_ = first_val.device
    label_len = MODEL_CFG.max_seq_len

    if cand.finished:
        return [cand]

    enc_out, graph_emb = model._encode(**encoder_kwargs)

    decoder_state, new_hidden = model.state_tracker(
        graph_emb,
        cand.history,
        cand.gru_hidden,
    )

    action_logits = model.action_predictor(decoder_state)

    if step_idx < min_steps:
        action_logits = action_logits.clone()
        action_logits[:, STOP_ACTION_ID] = -1e9

    action_log_probs = F.log_softmax(action_logits / temperature, dim=-1)

    top_a = min(top_a, action_log_probs.size(-1))
    action_values, action_indices = torch.topk(action_log_probs, k=top_a, dim=-1)

    expanded: List[BeamState] = []

    for a_rank in range(top_a):
        action_id = action_indices[0, a_rank].view(1)
        action_score = float(action_values[0, a_rank].item())

        # Terminate 只累加 action 分数，不展开 src/tgt/label
        if int(action_id.item()) == STOP_ACTION_ID:
            stop_edit = make_stop_edit(model, device_)
            expanded.append(
                BeamState(
                    edit_steps=cand.edit_steps + [stop_edit],
                    score=cand.score + action_score,
                    history=cand.history,
                    gru_hidden=new_hidden,
                    finished=True,
                )
            )
            continue

        src_logits, tgt_logits = model.pointer_network(
            decoder_state=decoder_state,
            dense_nodes=enc_out.dense_nodes,
            action_type=action_id,
            target_src_idx=None,
            node_mask=enc_out.node_pad_mask,
            has_nodes=enc_out.has_nodes,
        )

        src_log_probs = F.log_softmax(src_logits / temperature, dim=-1)
        tgt_log_probs = F.log_softmax(tgt_logits / temperature, dim=-1)

        top_s_ = min(top_s, src_log_probs.size(-1))
        top_t_ = min(top_t, tgt_log_probs.size(-1))

        src_values, src_indices = torch.topk(src_log_probs, k=top_s_, dim=-1)
        tgt_values, tgt_indices = torch.topk(tgt_log_probs, k=top_t_, dim=-1)

        # label 只按当前 action greedy decode 一次
        label_tokens = model.label_decoder.greedy_decode(
            decoder_state=decoder_state,
            action_type=action_id,
            bos_token_id=model.bos_token_id,
            max_len=MODEL_CFG.max_seq_len,
        )

        for s_rank in range(top_s_):
            src_id = src_indices[0, s_rank].view(1)
            src_score = float(src_values[0, s_rank].item())

            for t_rank in range(top_t_):
                tgt_id = tgt_indices[0, t_rank].view(1)
                tgt_score = float(tgt_values[0, t_rank].item())

                edit = EditStep(
                    action_type=action_id,
                    src_idx=src_id,
                    tgt_idx=tgt_id,
                    label_tokens=label_tokens,
                )

                new_history = model._append_history(
                    cand.history,
                    edit,
                    label_len,
                )

                expanded.append(
                    BeamState(
                        edit_steps=cand.edit_steps + [edit],
                        score=cand.score + action_score + src_score + tgt_score,
                        history=new_history,
                        gru_hidden=new_hidden,
                        finished=False,
                    )
                )

    return expanded


@torch.no_grad()
def generate_beam_candidates(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    encoder_kwargs: Dict[str, torch.Tensor],
    max_steps: int,
    beam_size: int,
    top_n: int,
    top_a: int,
    top_s: int,
    top_t: int,
    temperature: float,
    min_steps: int = 0,
    dedup: bool = True,
) -> List[List[Dict[str, Any]]]:
    first_val = next(iter(encoder_kwargs.values()))
    device_ = first_val.device
    label_len = MODEL_CFG.max_seq_len

    init_history = HistoryBatch.empty(1, label_len, device_)

    beams: List[BeamState] = [
        BeamState(
            edit_steps=[],
            score=0.0,
            history=init_history,
            gru_hidden=None,
            finished=False,
        )
    ]

    for step_idx in range(max_steps):
        all_expanded: List[BeamState] = []

        for cand in beams:
            all_expanded.extend(
                expand_one_beam_state(
                    model=model,
                    cand=cand,
                    encoder_kwargs=encoder_kwargs,
                    step_idx=step_idx,
                    min_steps=min_steps,
                    top_a=top_a,
                    top_s=top_s,
                    top_t=top_t,
                    temperature=temperature,
                )
            )

        # 同一路径去重，保留 score 最高的
        best_by_key: Dict[Tuple, BeamState] = {}

        for cand in all_expanded:
            path = pred_path_from_steps(cand.edit_steps, tokenizer)
            key = path_to_key(path)
            if key not in best_by_key or cand.score > best_by_key[key].score:
                best_by_key[key] = cand

        beams = sorted(
            best_by_key.values(),
            key=lambda c: c.score,
            reverse=True,
        )[:beam_size]

        if all(c.finished for c in beams):
            break

    paths = [
        pred_path_from_steps(cand.edit_steps, tokenizer)
        for cand in sorted(beams, key=lambda c: c.score, reverse=True)
    ]

    if dedup:
        paths = deduplicate_candidates(paths)

    return paths[:top_n]


# ============================================================
# 指标计算
# ============================================================

def compare_step(
    pred: Optional[Dict[str, Any]],
    gold: Dict[str, Any],
) -> Dict[str, bool]:
    if pred is None:
        return {
            "action_ok": False,
            "src_ok": False,
            "tgt_ok": False,
            "label_ok": False,
            "step_exact_ok": False,
        }

    action_ok = pred["action_type"] == gold["action_type"]

    if gold["action_type"] == STOP_ACTION_ID:
        return {
            "action_ok": action_ok,
            "src_ok": True,
            "tgt_ok": True,
            "label_ok": True,
            "step_exact_ok": action_ok,
        }

    src_ok = pred["src_idx"] == gold["src_idx"]
    tgt_ok = pred["tgt_idx"] == gold["tgt_idx"]
    label_ok = normalize_label(pred["label"]) == normalize_label(gold["label"])

    return {
        "action_ok": action_ok,
        "src_ok": src_ok,
        "tgt_ok": tgt_ok,
        "label_ok": label_ok,
        "step_exact_ok": action_ok and src_ok and tgt_ok and label_ok,
    }


def path_exact_match(
    pred_path: List[Dict[str, Any]],
    gold_path: List[Dict[str, Any]],
) -> bool:
    if len(pred_path) != len(gold_path):
        return False

    for pred_step, gold_step in zip(pred_path, gold_path):
        if not compare_step(pred_step, gold_step)["step_exact_ok"]:
            return False

    return True


def prefix_step_match_len(
    pred_path: List[Dict[str, Any]],
    gold_path: List[Dict[str, Any]],
) -> int:
    """
    从第 0 步开始，连续完全匹配的步数。
    用于分析自回归误差积累。
    """
    cnt = 0
    for pred_step, gold_step in zip(pred_path, gold_path):
        if compare_step(pred_step, gold_step)["step_exact_ok"]:
            cnt += 1
        else:
            break
    return cnt


def init_stats() -> Dict[str, float]:
    return {
        "num_paths": 0.0,
        "skipped": 0.0,

        "top1_correct": 0.0,
        "top3_correct": 0.0,
        "top5_correct": 0.0,
        "top10_correct": 0.0,

        "terminate_correct": 0.0,

        "step_total": 0.0,
        "step_action_correct": 0.0,
        "step_exact_correct": 0.0,

        "non_stop_total": 0.0,
        "src_correct": 0.0,
        "tgt_correct": 0.0,
        "label_correct": 0.0,

        "prefix_exact_sum": 0.0,
        "prefix_action_sum": 0.0,

        "total_gold_steps": 0.0,
        "total_pred_steps": 0.0,
        "total_candidates": 0.0,
    }


def update_top1_step_metrics(
    stats: Dict[str, float],
    pred_path: List[Dict[str, Any]],
    gold_path: List[Dict[str, Any]],
) -> None:
    stats["total_gold_steps"] += len(gold_path)
    stats["total_pred_steps"] += len(pred_path)

    gold_term_idx = next(
        (i for i, item in enumerate(gold_path)
         if item["action_type"] == STOP_ACTION_ID),
        len(gold_path) - 1,
    )

    pred_term_idx = next(
        (i for i, item in enumerate(pred_path)
         if item["action_type"] == STOP_ACTION_ID),
        None,
    )

    if pred_term_idx == gold_term_idx:
        stats["terminate_correct"] += 1

    stats["prefix_exact_sum"] += prefix_step_match_len(pred_path, gold_path)

    for i, gold_step in enumerate(gold_path):
        pred_step = pred_path[i] if i < len(pred_path) else None
        cmp_res = compare_step(pred_step, gold_step)

        stats["step_total"] += 1
        stats["step_action_correct"] += int(cmp_res["action_ok"])
        stats["step_exact_correct"] += int(cmp_res["step_exact_ok"])

        if gold_step["action_type"] != STOP_ACTION_ID:
            stats["non_stop_total"] += 1
            stats["src_correct"] += int(cmp_res["src_ok"])
            stats["tgt_correct"] += int(cmp_res["tgt_ok"])
            stats["label_correct"] += int(cmp_res["label_ok"])


def finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    n = max(1.0, stats["num_paths"])
    step_total = max(1.0, stats["step_total"])
    non_stop_total = max(1.0, stats["non_stop_total"])

    return {
        "num_paths": stats["num_paths"],
        "skipped": stats["skipped"],

        "step_action_acc": stats["step_action_correct"] / step_total,
        "step_src_acc": stats["src_correct"] / non_stop_total,
        "step_tgt_acc": stats["tgt_correct"] / non_stop_total,
        "step_label_acc": stats["label_correct"] / non_stop_total,
        "step_exact_acc": stats["step_exact_correct"] / step_total,

        "path_exact_acc": stats["top1_correct"] / n,
        "top1_path_acc": stats["top1_correct"] / n,
        "top3_path_acc": stats["top3_correct"] / n,
        "top5_path_acc": stats["top5_correct"] / n,
        "top10_path_acc": stats["top10_correct"] / n,

        "terminate_acc": stats["terminate_correct"] / n,
        "avg_gold_steps": stats["total_gold_steps"] / n,
        "avg_pred_steps": stats["total_pred_steps"] / n,
        "avg_num_candidates": stats["total_candidates"] / n,
        "avg_prefix_exact_steps": stats["prefix_exact_sum"] / n,
    }


@torch.no_grad()
def evaluate_methods(
    model: ActorNetwork,
    tokenizer: LabelTokenizer,
    reactions: List[Dict[str, Any]],
    methods: List[str],
    max_steps: int,
    min_steps: int,
    top_n: int,
    temperature: float,
    top_k: int,
    num_rollouts: int,
    discount_factor: float,
    dedup_candidates: bool,
    beam_size: int,
    top_a: int,
    top_s: int,
    top_t: int,
    debug_errors: bool = False,
) -> Dict[str, Dict[str, float]]:
    model.eval()

    all_stats = {method: init_stats() for method in methods}

    for reaction_idx, reaction in enumerate(tqdm(reactions, desc="[path-generation-compare-v2]")):
        try:
            product_smi = reaction["input"]["product_smi"]
            encoder_kwargs = build_encoder_kwargs(product_smi)
            gold_path = gold_path_from_reaction(reaction)
        except Exception as e:
            if debug_errors:
                print(f"[skip-build] idx={reaction_idx}, error={repr(e)}")
            for method in methods:
                all_stats[method]["skipped"] += 1
            continue

        for method in methods:
            stats = all_stats[method]

            try:
                if method == "greedy":
                    candidates = generate_greedy_candidates(
                        model, tokenizer, encoder_kwargs, max_steps
                    )

                elif method == "greedy_min":
                    candidates = generate_greedy_min_candidates(
                        model, tokenizer, encoder_kwargs, max_steps, min_steps
                    )

                elif method == "topn_sampling":
                    candidates = generate_topn_sampling_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        encoder_kwargs=encoder_kwargs,
                        max_steps=max_steps,
                        top_n=top_n,
                        temperature=temperature,
                        top_k=top_k,
                        dedup=dedup_candidates,
                    )

                elif method == "beam":
                    candidates = generate_beam_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        encoder_kwargs=encoder_kwargs,
                        max_steps=max_steps,
                        beam_size=beam_size,
                        top_n=top_n,
                        top_a=top_a,
                        top_s=top_s,
                        top_t=top_t,
                        temperature=temperature,
                        min_steps=0,
                        dedup=dedup_candidates,
                    )

                elif method == "beam_min":
                    candidates = generate_beam_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        encoder_kwargs=encoder_kwargs,
                        max_steps=max_steps,
                        beam_size=beam_size,
                        top_n=top_n,
                        top_a=top_a,
                        top_s=top_s,
                        top_t=top_t,
                        temperature=temperature,
                        min_steps=min_steps,
                        dedup=dedup_candidates,
                    )

                elif method in {"monte_carlo", "policy_gradient"}:
                    candidates = generate_rl_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        encoder_kwargs=encoder_kwargs,
                        method=method,
                        max_steps=max_steps,
                        num_rollouts=num_rollouts,
                        temperature=temperature,
                        discount_factor=discount_factor,
                        beam_size=1,
                    )

                    if dedup_candidates:
                        candidates = deduplicate_candidates(candidates)

                else:
                    raise ValueError(f"Unknown method: {method}")

            except Exception as e:
                stats["skipped"] += 1
                if debug_errors:
                    print(f"[skip-generate] idx={reaction_idx}, method={method}, error={repr(e)}")
                continue

            if not candidates:
                stats["skipped"] += 1
                continue

            stats["num_paths"] += 1
            stats["total_candidates"] += len(candidates)

            for k, key in [
                (1, "top1_correct"),
                (3, "top3_correct"),
                (5, "top5_correct"),
                (10, "top10_correct"),
            ]:
                topk_candidates = candidates[: min(k, len(candidates))]
                if any(path_exact_match(cand, gold_path) for cand in topk_candidates):
                    stats[key] += 1

            update_top1_step_metrics(
                stats=stats,
                pred_path=candidates[0],
                gold_path=gold_path,
            )

    return {
        method: finalize_stats(stats)
        for method, stats in all_stats.items()
    }


def save_results_json(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[save] JSON results saved to: {path}")


def save_results_csv(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = []
    for _, res in results.items():
        for k in res.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + all_keys)

        for method, res in results.items():
            writer.writerow([method] + [res.get(k, "") for k in all_keys])

    print(f"[save] CSV results saved to: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default=str(PATH_CFG.CKPT_BEST_MODEL_FILE))
    parser.add_argument("--test_json", type=str, default=str(PATH_CFG.RL_TEST_DATA_FILE))
    parser.add_argument("--vocab_file", type=str, default=str(PATH_CFG.VOCAB_FILE))

    parser.add_argument(
        "--methods",
        type=str,
        default="greedy,greedy_min,topn_sampling,beam,beam_min,monte_carlo,policy_gradient",
    )

    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--min_steps", type=int, default=2)

    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--discount_factor", type=float, default=0.9)

    parser.add_argument("--beam_size", type=int, default=30)
    parser.add_argument("--top_a", type=int, default=5)
    parser.add_argument("--top_s", type=int, default=5)
    parser.add_argument("--top_t", type=int, default=5)

    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--debug_errors", action="store_true")

    parser.add_argument(
        "--save_json",
        type=str,
        default="eval_path_results/path_generation_summary_v2.json",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="eval_path_results/path_generation_summary_v2.csv",
    )

    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[path-generation-compare-v2] device = {device}")
    print(f"[path-generation-compare-v2] ckpt = {args.ckpt}")
    print(f"[path-generation-compare-v2] test_json = {args.test_json}")
    print(f"[path-generation-compare-v2] vocab_file = {args.vocab_file}")
    print(f"[path-generation-compare-v2] methods = {methods}")
    print(f"[path-generation-compare-v2] max_steps = {args.max_steps}")
    print(f"[path-generation-compare-v2] min_steps = {args.min_steps}")
    print(f"[path-generation-compare-v2] top_n = {args.top_n}")
    print(f"[path-generation-compare-v2] temperature = {args.temperature}")
    print(f"[path-generation-compare-v2] top_k = {args.top_k}")
    print(f"[path-generation-compare-v2] beam_size = {args.beam_size}")
    print(f"[path-generation-compare-v2] top_a/top_s/top_t = {args.top_a}/{args.top_s}/{args.top_t}")
    print(f"[path-generation-compare-v2] dedup_candidates = {not args.no_dedup}")

    tokenizer = LabelTokenizer(args.vocab_file)
    model = load_model(args.ckpt, vocab_size=tokenizer.vocab_size)

    reactions = load_reactions(args.test_json)

    if args.limit and args.limit > 0:
        reactions = reactions[: args.limit]

    print(f"[path-generation-compare-v2] loaded reactions = {len(reactions)}")

    results = evaluate_methods(
        model=model,
        tokenizer=tokenizer,
        reactions=reactions,
        methods=methods,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        top_n=args.top_n,
        temperature=args.temperature,
        top_k=args.top_k,
        num_rollouts=args.num_rollouts,
        discount_factor=args.discount_factor,
        dedup_candidates=not args.no_dedup,
        beam_size=args.beam_size,
        top_a=args.top_a,
        top_s=args.top_s,
        top_t=args.top_t,
        debug_errors=args.debug_errors,
    )

    print("\n[Path Generation Comparison Results]")
    for method, res in results.items():
        print(f"\nMethod: {method}")
        for k, v in res.items():
            if k in {"num_paths", "skipped"}:
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.4f}")

    if args.save_json:
        save_results_json(results, args.save_json)

    if args.save_csv:
        save_results_csv(results, args.save_csv)


if __name__ == "__main__":
    main()