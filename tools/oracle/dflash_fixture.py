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

    dflash_fixture.py --muse [--out tests/loader/muse_dflash_fixture.safetensors]

Spec 23f: the same for the Muse-Glimmer assistant's form (dflash_ref.py's "muse", bitwise transformers 5.19.0) -
a tiny random MuseGlimmerAssistantModel at Muse's ratios: hidden 64, 2 layers, 8 / 2 heads x 16 (GQA 4), intermediate
128, 5 taps {1, 3, 5, 7, 9} of a target of hidden 64 (fc 320 -> 64), vocab 512 (every id a token), mask id 500, block
16, window 16 INCLUSIVE (|q - k| <= 16, so a 40-position context crosses it for row 0 and row 15), eps 1e-5, theta
5e5, every GEMV width a multiple of 64 (the int4 g64 arm). The reference runs in float32 (the D1 form):
  target.embed / target.head bf16 [512][64], taps bf16 [40][320], the drafter's tensors
  muse.ctx.fc f32 [40][64]; muse.ctx.k{l} / muse.ctx.v{l} f32 [40][2][16]
  muse.case{i}.{hidden f32 [16][64] (all 16 rows), cand i32 [K][16], unary f32 [K][16] (raw logits), ids i32 [K]}
  muse.norm.{src bf16 [4][64], w bf16 [64], out bf16 [4][64]}   _m_norm in bfloat16 (the two roundings)
  metadata: config_muse, vocab_used, cases

    dflash_fixture.py --lilicorr [--out tests/loader/lilicorr_dflash_fixture.safetensors]

Spec 26b: the same for the LiLiCorr form (dflash_ref.py's "lilicorr", 26a's port of vLLM's models/lilicorr.py) - 26a's
tiny drafter (tools/oracle/lilicorr_make_tiny.py: its names, NVFP4 packing and edges - a zero scale block, subnormal
scales) at narrower widths so the file stays small: hidden 128, 2 layers, 4 / 1 heads x 32, FF 256, 6 taps of a
hidden-128 target (fc 768 -> 128), conv groups of 16 (kernel projection 32 rows), window 16 (|q - k| <= 15), sinks,
block 8, mask id 990, the head 64 wide / 2 layers / 4 heads / MLP 128 / factor 64 / top-8, over a synthetic target of
vocab 1024 (embedding and head random bf16). The reference runs in float32 (its only mode):
  ckpt.<name>                     the tiny checkpoint's own tensors (U8 / F8_E4M3 / F32 NVFP4 triplets, BF16 the rest)
  target.embed / target.head bf16 [1024][128], taps bf16 [40][768]
  lil.ctx.fc f32 [40][128]; lil.ctx.k{l} / lil.ctx.v{l} f32 [40][1][32]   context_kv over positions 0..39
  lil.case{i}.{hidden f32 [8][128] (all 8 rows), cand i32 [7][8], logit f32 [7][8], lp f32 [7][8], lattice f32
              [7][8][8], anchor f32 [128] (zeros when invalid), chosen i32 [K], ids i32 [K], scores f32 [K][8]}
  lil.nvfp4.<base>.{wp bf16 [N][K], g f32 [1]}   nvfp4_decode of three of the linears (W' and g kept apart)
  lil.mut.{no_sink, anchor_p2}.lattice f32 [7][8][8]   case 3 under the reference's test-only mutations
  metadata: config_lilicorr, vocab_used, cases
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


MH, ML, MNH, MNKV, MHD, MI, MTAPS, MWIN, MBLOCK = 64, 2, 8, 2, 16, 128, [1, 3, 5, 7, 9], 16, 16
# window edges for row 0 and row 15 (p 15 / 16 / 17, p 30 / 39 past a 32-slot ring's wrap), K 15 / 7 / 1, the mask id
# as the anchor. Greedy only (a sampled Muse draft is drawn on the host from the readback).
MUSE_CASES = [
    {"p": 1, "anchor": 7, "K": 15, "T": 0.0, "seed": 0},
    {"p": 15, "anchor": 123, "K": 15, "T": 0.0, "seed": 0},
    {"p": 16, "anchor": 4, "K": 15, "T": 0.0, "seed": 0},
    {"p": 17, "anchor": 300, "K": 7, "T": 0.0, "seed": 0},
    {"p": 30, "anchor": 500, "K": 15, "T": 0.0, "seed": 0},
    {"p": 39, "anchor": 77, "K": 1, "T": 0.0, "seed": 0},
    {"p": 33, "anchor": 410, "K": 15, "T": 0.0, "seed": 0},
]


def muse_config() -> dict:
    return {"architectures": ["MuseGlimmerAssistantModel"], "model_type": "muse_glimmer_assistant", "block_size": MBLOCK,
            "head_dim": MHD, "hidden_act": "silu", "hidden_size": MH, "intermediate_size": MI,
            "layer_types": ["sliding_attention"] * ML, "mask_token_id": MASK, "num_attention_heads": MNH,
            "num_hidden_layers": ML, "num_key_value_heads": MNKV, "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "sliding_window": MWIN,
            "target_layer_ids": MTAPS}


def muse_weights() -> dict:
    w = {"fc.weight": bf(MH, len(MTAPS) * TH, std=0.06), "hidden_norm.weight": bf(MH, std=0.1, mean=1.0),
         "norm.weight": bf(MH, std=0.1, mean=1.0)}
    for i in range(ML):
        p = f"layers.{i}."
        w.update({
            p + "input_layernorm.weight": bf(MH, std=0.1, mean=1.0),
            p + "post_attention_layernorm.weight": bf(MH, std=0.1, mean=1.0),
            p + "self_attn.q_proj.weight": bf(MNH * MHD, MH, std=0.12),
            p + "self_attn.k_proj.weight": bf(MNKV * MHD, MH, std=0.12),
            p + "self_attn.v_proj.weight": bf(MNKV * MHD, MH, std=0.12),
            p + "self_attn.o_proj.weight": bf(MH, MNH * MHD, std=0.08),
            p + "self_attn.q_norm.weight": bf(MHD, std=0.1, mean=1.0),
            p + "self_attn.k_norm.weight": bf(MHD, std=0.1, mean=1.0),
            p + "mlp.gate_proj.weight": bf(MI, MH, std=0.12),
            p + "mlp.up_proj.weight": bf(MI, MH, std=0.12),
            p + "mlp.down_proj.weight": bf(MH, MI, std=0.08),
        })
    return w


def main_muse(out_path: str) -> None:
    torch.manual_seed(20261017)
    w = muse_weights()
    embed = bf(V, MH, std=0.5)
    head = bf(V, MH, std=0.5)
    taps = {i: bf(N, TH, std=1.0) for i in MTAPS}
    pos = torch.arange(N)
    out = {"target.embed": embed, "target.head": head, "taps": torch.cat([taps[i] for i in MTAPS], -1)}
    out.update(w)
    cfg_d = muse_config()
    meta = {"vocab_used": str(V), "cases": json.dumps(MUSE_CASES), "config_muse": json.dumps(cfg_d)}
    d = ref.Drafter(ref.config_from_dict(cfg_d), dict(w), embed, head, dtype=torch.float32)
    ctx = d.context_kv(taps, pos)
    out["muse.ctx.fc"] = ctx.fc.float().contiguous()
    for i in range(ML):
        out[f"muse.ctx.k{i}"] = ctx.k[i].float().contiguous()
        out[f"muse.ctx.v{i}"] = ctx.v[i].float().contiguous()
    for ci, c in enumerate(MUSE_CASES):
        o = d.draft_block(c["anchor"], c["p"], c["K"], ctx)
        p = f"muse.case{ci}."
        out[p + "hidden"] = o.hidden.float().contiguous()
        out[p + "cand"] = o.cand_ids.to(torch.int32).contiguous()
        out[p + "unary"] = o.unary.float().contiguous()
        out[p + "ids"] = o.ids.to(torch.int32).contiguous()
    # the two-rounding norm in bfloat16 (the checkpoint's own arithmetic): a bf16 drafter's _m_norm
    db = ref.Drafter(ref.config_from_dict(cfg_d), dict(w), embed, head, dtype=torch.bfloat16)
    src = bf(4, MH, std=1.5)
    nw = w["layers.0.input_layernorm.weight"]
    out["muse.norm.src"] = src
    out["muse.norm.w"] = nw
    out["muse.norm.out"] = db._m_norm(src, nw).contiguous()
    from safetensors.torch import save_file
    save_file({k: v.contiguous().clone() for k, v in out.items()}, out_path, metadata=meta)
    print(f"wrote {out_path}: {len(out)} tensors, {os.path.getsize(out_path)} bytes")


# Spec 26b: (p, anchor, K, T, seed) - no context (p 0: the anchor invalid), the window's edge for row 0 and row 7 (p 15 /
# 16 / 17 with window 16), the 16-slot ring's wrap (p 30 / 39 / 40), K 7 / 3 / 1 (prefixes of one walk), the mask id as
# the anchor, sampled walks (the seed's high word set: the slot keys seed ^ (s << 32)).
LIL_CASES = [
    {"p": 0, "anchor": 7, "K": 7, "T": 0.0, "seed": 0},
    {"p": 1, "anchor": 11, "K": 7, "T": 0.0, "seed": 0},
    {"p": 15, "anchor": 123, "K": 7, "T": 0.0, "seed": 0},
    {"p": 16, "anchor": 4, "K": 7, "T": 0.0, "seed": 0},
    {"p": 17, "anchor": 300, "K": 3, "T": 0.0, "seed": 0},
    {"p": 30, "anchor": 990, "K": 7, "T": 0.0, "seed": 0},
    {"p": 39, "anchor": 77, "K": 1, "T": 0.0, "seed": 0},
    {"p": 40, "anchor": 512, "K": 7, "T": 0.0, "seed": 0},
    {"p": 23, "anchor": 9, "K": 7, "T": 0.8, "seed": 1234567},
    {"p": 33, "anchor": 410, "K": 6, "T": 1.3, "seed": (5 << 32) | 77},
]
LV, LTH = 1024, 128


def lilicorr_cfg() -> tuple:
    spec = importlib.util.spec_from_file_location("lilicorr_make_tiny", os.path.join(_HERE, "lilicorr_make_tiny.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    cfg = mk.tiny_config(vocab=LV, window=16)
    cfg.update({"hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32,
                "intermediate_size": 256})
    cfg["dflash_config"].update({"lilicorr_hidden_size": 64, "lilicorr_num_heads": 4, "lilicorr_factor_dim": 64})
    return cfg, mk


def main_lilicorr(out_path: str) -> None:
    cfg_d, mk = lilicorr_cfg()
    ckpt = mk.make_tensors(cfg_d, seed=26)
    torch.manual_seed(20261023)
    embed = bf(LV, 128, std=0.5)
    head = bf(LV, 128, std=0.5)
    taps_ids = cfg_d["dflash_config"]["target_layer_ids"]
    taps = {i: bf(N, LTH, std=1.0) for i in taps_ids}
    pos = torch.arange(N)
    out = {f"ckpt.{k}": v for k, v in ckpt.items()}
    out.update({"target.embed": embed, "target.head": head, "taps": torch.cat([taps[i] for i in taps_ids], -1)})
    meta = {"vocab_used": str(LV), "cases": json.dumps(LIL_CASES), "config_lilicorr": json.dumps(cfg_d)}
    cfg = ref.config_from_dict(cfg_d)
    tens = ref._read_nvfp4_tensors(dict(ckpt), cfg_d["quantization_config"])
    d = ref.Drafter(cfg, dict(tens), embed, head)
    for base in ("fc", "layers.1.mlp.down_proj", "lilicorr.layers.0.attn.out_proj"):
        out[f"lil.nvfp4.{base}.wp"] = tens[base + ".weight"].to(torch.bfloat16)
        out[f"lil.nvfp4.{base}.g"] = torch.tensor([float(tens[base + ref.G_SUFFIX])], dtype=torch.float32)
    ctx = d.context_kv(taps, pos)
    out["lil.ctx.fc"] = ctx.fc.float().contiguous()
    for i in range(cfg.n_layers):
        out[f"lil.ctx.k{i}"] = ctx.k[i].float().contiguous()
        out[f"lil.ctx.v{i}"] = ctx.v[i].float().contiguous()
    for ci, c in enumerate(LIL_CASES):
        o = d.draft_block(c["anchor"], c["p"], c["K"], ctx, temperature=c["T"], seed=c["seed"] if c["T"] > 0 else None)
        p = f"lil.case{ci}."
        out[p + "hidden"] = o.hidden.float().contiguous()
        out[p + "cand"] = o.cand_ids.to(torch.int32).contiguous()
        lg = d.head_logits(o.hidden[1:])
        out[p + "logit"] = torch.gather(lg, 1, o.cand_ids).float().contiguous()
        out[p + "lp"] = o.log_probs.float().contiguous()
        out[p + "lattice"] = o.lattice.float().contiguous()
        out[p + "anchor"] = o.anchor.float().contiguous()
        out[p + "chosen"] = o.chosen.to(torch.int32).contiguous()
        out[p + "ids"] = o.ids.to(torch.int32).contiguous()
        out[p + "scores"] = o.scores.float().contiguous()
    c = LIL_CASES[3]
    for m in ("no_sink", "anchor_p2"):
        d.mut = {m}
        out[f"lil.mut.{m}.lattice"] = d.draft_block(c["anchor"], c["p"], c["K"], ctx).lattice.float().contiguous()
    d.mut = set()
    from safetensors.torch import save_file
    save_file({k: v.contiguous().clone() for k, v in out.items()}, out_path, metadata=meta)
    print(f"wrote {out_path}: {len(out)} tensors, {os.path.getsize(out_path)} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--muse", action="store_true", help="spec 23f: the Muse-Glimmer assistant's fixture")
    ap.add_argument("--lilicorr", action="store_true", help="spec 26b: the LiLiCorr drafter's fixture")
    a = ap.parse_args()
    if a.muse:
        main_muse(a.out or "tests/loader/muse_dflash_fixture.safetensors")
        return
    if a.lilicorr:
        main_lilicorr(a.out or "tests/loader/lilicorr_dflash_fixture.safetensors")
        return
    a.out = a.out or "tests/loader/dflash_ref_fixture.safetensors"
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
