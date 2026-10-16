#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The DFlash host twin's fixture (spec 19b Task 2): a tiny random DFlash 2 drafter, a target embedding / head and
taps, and what tools/oracle/dflash_ref.py drafts from them - so tests/kernels/dflash_ref_test.cc can hold the C++
host twin (tests/kernels/dflash_ref.h) to the reference the engine's D1 gate is measured against.

    dflash_fixture.py [--out tests/loader/dflash_ref_fixture.safetensors]

Run in the oracle container (tools/oracle/run_in_container.sh). Deterministic (torch.manual_seed): re-running writes
the same bytes.

The shapes are tiny but keep every GEMV width a multiple of 64 (the engine's GEMV tiles, so the same file drives the
Mac's indicative kernel run, tools/mac/clrun/dflash_run.cc): hidden 64, 2 layers, 4 / 2 heads x 16, intermediate 96,
vocab 512 of which 508 are "defined" (rows 508..511 of the head are made large: the engine masks them as argmax.cl
masks Qwen3.8's padded rows, and the expected outputs here apply the same mask), mask id 500, block 8, taps {1, 3} of a
target of hidden 64, window 16 (so a 40-position context crosses it), conv group 4 (kernel projection 64 rows),
selector rank 64 / top 16, output_multiplier 1.5, softcap 30, input_embedding_scale 0.9.

Contents (one safetensors file; the drafter's tensors under their checkpoint names):
  target.embed bf16 [512][64], target.head bf16 [512][64]       the shared embedding / lm_head
  taps bf16 [40][128]                                          the two taps concatenated per position (fc's input)
  v2.ctx.fc f32 [40][64]; v2.ctx.k{l} / v2.ctx.v{l} f32 [40][2][16]   context_kv over positions 0..39
  v2.case{i}.{hidden f32 [1+K][64], cand i32 [K][16], unary f32 [K][16], scores f32 [K][16], chosen i32 [K],
             ids i32 [K]}                                      draft_block per case; v1.case{i}.* likewise for the
                                                               same weights as a DFlash (v1) drafter (no convs, no
                                                               selector; its scores = unary, logit_scale 1)
  metadata: config_v2, config_v1 (config.json text), vocab_used, cases (JSON: [{p, anchor, K, T, seed}, ...])
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py
import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("dflash_ref", os.path.join(_HERE, "dflash_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["dflash_ref"] = ref
_spec.loader.exec_module(ref)

H, L, NH, NKV, HD, I, V, USED, MASK, BLOCK, TH, N = 64, 2, 4, 2, 16, 96, 512, 508, 500, 8, 64, 40
TAPS = [1, 3]
WINDOW, GROUP, RANK, TOPK = 16, 4, 64, 16


def config(version: int) -> dict:
    dc = {"block_size": BLOCK, "mask_token_id": MASK, "target_layer_ids": TAPS}
    if version == 2:
        dc.update({"conv_group_size": GROUP, "conv_kernel_size": 2, "selector_rank": RANK, "selector_top_k": TOPK,
                   "output_multiplier": 1.5, "final_logit_softcapping": 30.0, "input_embedding_scale": 0.9})
    return {"architectures": ["DFlash2DraftModel" if version == 2 else "DFlashDraftModel"], "attention_bias": False,
            "is_causal": False, "dflash_config": dc, "head_dim": HD, "hidden_act": "silu", "hidden_size": H,
            "intermediate_size": I, "layer_types": ["sliding_attention"] * L, "model_type": "qwen3",
            "num_attention_heads": NH, "num_hidden_layers": L, "num_key_value_heads": NKV, "num_target_layers": 4,
            "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "sliding_window": WINDOW, "vocab_size": V}


def bf(*shape, std=0.1, mean=0.0):
    return (torch.randn(*shape) * std + mean).to(torch.bfloat16)


def weights() -> dict:
    w = {"fc.weight": bf(H, len(TAPS) * TH, std=0.08), "hidden_norm.weight": bf(H, std=0.1, mean=1.0),
         "norm.weight": bf(H, std=0.1, mean=1.0)}
    for i in range(L):
        p = f"layers.{i}."
        w.update({
            p + "input_layernorm.weight": bf(H, std=0.1, mean=1.0),
            p + "post_attention_layernorm.weight": bf(H, std=0.1, mean=1.0),
            p + "self_attn.q_proj.weight": bf(NH * HD, H, std=0.12),
            p + "self_attn.k_proj.weight": bf(NKV * HD, H, std=0.12),
            p + "self_attn.v_proj.weight": bf(NKV * HD, H, std=0.12),
            p + "self_attn.o_proj.weight": bf(H, NH * HD, std=0.12),
            p + "self_attn.q_norm.weight": bf(HD, std=0.1, mean=1.0),
            p + "self_attn.k_norm.weight": bf(HD, std=0.1, mean=1.0),
            p + "mlp.gate_proj.weight": bf(I, H, std=0.12),
            p + "mlp.up_proj.weight": bf(I, H, std=0.12),
            p + "mlp.down_proj.weight": bf(H, I, std=0.12),
        })
        for cv in ("attention_conv", "mlp_conv"):
            w[p + cv + ".base_kernel"] = bf(2, 2, H, std=0.2, mean=0.5)
            w[p + cv + ".kernel_projection.weight"] = bf(2 * 2 * (H // GROUP), H, std=0.05)
    w["candidate_selector.hidden_projection.weight"] = bf(RANK, H, std=0.12)
    w["candidate_selector.predecessor_codebook"] = bf(V, RANK, std=0.3)
    w["candidate_selector.successor_codebook"] = bf(V, RANK, std=0.3)
    return w


V2_ONLY = ("attention_conv", "mlp_conv", "candidate_selector")

# (p, anchor id, K, temperature, seed): window edges (p 15 / 16 / 17 with window 16), the wrap of a 16-slot ring
# (p 30, 39), K < block - 1, the mask id as the anchor, sampled walks.
CASES = [
    {"p": 1, "anchor": 7, "K": 7, "T": 0.0, "seed": 0},
    {"p": 15, "anchor": 123, "K": 7, "T": 0.0, "seed": 0},
    {"p": 16, "anchor": 4, "K": 7, "T": 0.0, "seed": 0},
    {"p": 17, "anchor": 300, "K": 5, "T": 0.0, "seed": 0},
    {"p": 30, "anchor": 500, "K": 7, "T": 0.0, "seed": 0},
    {"p": 39, "anchor": 77, "K": 3, "T": 0.0, "seed": 0},
    {"p": 39, "anchor": 77, "K": 1, "T": 0.0, "seed": 0},
    {"p": 23, "anchor": 9, "K": 7, "T": 0.8, "seed": 1234567},
    {"p": 33, "anchor": 410, "K": 6, "T": 1.3, "seed": (5 << 32) | 77},
]


def masked_draft(d, anchor, p, K, ctx, T, seed):
    """draft_block with the head's rows >= USED at -inf (argmax.cl's VOCAB_USED; the engine's top-16 masks them)."""
    orig = d.head_logits

    def head(h):
        lg = orig(h)
        lg[:, USED:] = float("-inf")
        return lg

    d.head_logits = head
    try:
        return d.draft_block(anchor, p, K, ctx, temperature=T, seed=seed if T > 0 else None)
    finally:
        d.head_logits = orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/loader/dflash_ref_fixture.safetensors")
    a = ap.parse_args()
    torch.manual_seed(20261016)
    w = weights()
    embed = bf(V, H, std=0.5)
    head = bf(V, H, std=0.5)
    head[USED:] = (torch.ones(V - USED, H) * 3.0).to(torch.bfloat16)   # would win unmasked
    taps = {i: bf(N, TH, std=1.0) for i in TAPS}
    pos = torch.arange(N)
    out = {"target.embed": embed, "target.head": head, "taps": torch.cat([taps[i] for i in TAPS], -1)}
    out.update(w)
    meta = {"vocab_used": str(USED), "cases": json.dumps(CASES)}
    for version in (2, 1):
        cfg = ref.config_from_dict(config(version))
        tens = {k: v for k, v in w.items() if version == 2 or not any(s in k for s in V2_ONLY)}
        d = ref.Drafter(cfg, tens, embed, head)
        ctx = d.context_kv(taps, pos)
        tag = f"v{version}."
        meta[f"config_v{version}"] = json.dumps(config(version))
        if version == 2:
            out[tag + "ctx.fc"] = ctx.fc.contiguous()
            for i in range(L):
                out[tag + f"ctx.k{i}"] = ctx.k[i].contiguous()
                out[tag + f"ctx.v{i}"] = ctx.v[i].contiguous()
        for ci, c in enumerate(CASES):
            if version == 1 and c["T"] > 0:
                continue                                   # v1 sampling is not built (full-vocabulary Gumbel)
            o = masked_draft(d, c["anchor"], c["p"], c["K"], ctx, c["T"], c["seed"])
            p = f"{tag}case{ci}."
            out[p + "hidden"] = o.hidden.contiguous()
            out[p + "cand"] = o.cand_ids.to(torch.int32).contiguous()
            out[p + "unary"] = o.unary.float().contiguous()
            out[p + "scores"] = o.scores.float().contiguous()
            out[p + "chosen"] = o.chosen.to(torch.int32).contiguous()
            out[p + "ids"] = o.ids.to(torch.int32).contiguous()
    from safetensors.torch import save_file
    save_file({k: v.contiguous().clone() for k, v in out.items()}, a.out, metadata=meta)
    print(f"wrote {a.out}: {len(out)} tensors, {os.path.getsize(a.out)} bytes")


if __name__ == "__main__":
    main()
