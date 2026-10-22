#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 26a Task 1: a tiny random LiLiCorr drafter (no public tiny exists) in the checkpoint's own shape and names.

    lilicorr_make_tiny.py <out dir> [--seed 0] [--vocab 131072] [--taps 1,2,4,6,9,10] [--window 16]

Written like `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` @ 449d2414 (config.json, hf_quant_config.json,
one model.safetensors with the checkpoint's bare names: `fc.*`, `hidden_norm.weight`, `norm.weight`, `layers.N.*`,
`lilicorr.*`) at tiny widths, for 25a's tiny Bolt (`bolt_make_tiny.py`: hidden 256, 12 blocks `M E M E M E M * E M E
M`, vocab 131072):
  backbone      2 layers, hidden 256, 4 q / 1 kv heads x 64, FF 512, every layer sliding at window 16 (vLLM's
                |q - k| <= 15), non-causal, attention sinks (a learned per-head logit, ~N(0, 1.5)), DFlash 2's two
                grouped convolutions (2 taps, groups of 16), RMSNorm eps 1e-6, RoPE 1e4, block 8, mask id 990
  taps          target_layer_ids 1, 2, 4, 6, 9, 10 of the tiny Bolt (E M M M M E - the real 1 / 18 / 35 / 51 / 68 / 85's
                kinds: MoE first and last, Mamba between); fc 6 x 256 -> 256
  LiLiCorr head hidden 64, 2 layers, 4 heads (x 16), mlp_ratio 2.0, factor_dim 64, candidate_topk 8, vector_eps 1e-4,
                logit_scale 8.0; every linear with a bias
  NVFP4         the real checkpoint's 25 NVFP4 linears at these widths (fc, every layer's mlp.{gate,up,down}_proj, the
                head's token_proj / pass_hidden_proj / context_proj / layers.N.attn.out_proj / layers.N.mlp.{up,down}_proj)
                as ModelOpt W4A16 packs them: `weight` U8 [N][K/2] (k = 2i in the low nibble), `weight_scale` F8_E4M3
                [N][K/16], `weight_scale_2` F32 scalar g - g = amax / (6 x 448) x 1.37 (non-trivial: never a power of
                two), s_b = e4m3(block amax / (6 g)) round-to-nearest, e = the nearest e2m1 of w / (s_b g); a few blocks are
                forced to a zero scale and to subnormal scales (codes 0x01..0x07) so the decoder's edges are exercised
                BF16           everything else (attention q / k / v / o and norms, convolutions, sinks, norms, the head's other
                tensors)
`--vocab` smaller than 131072 is for unit tests on a synthetic target (the mask id 990 must stay a row).
"""
import argparse
import json
import math
import os
import sys

import torch

BLOCK = 8
MASK_ID = 990
TAPS = (1, 2, 4, 6, 9, 10)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
QUANT_EXCLUDE = ["*attention_conv*", "*mlp_conv*", "*self_attn*", "*norm", "*lilicorr.anchor_out_head",
                 "*lilicorr.factor_input_proj", "lilicorr.feature_mlp*", "*lilicorr.feature_mlp.up_proj",
                 "*lilicorr.feature_mlp.down_proj", "*lilicorr.in_head", "*lilicorr.out_head", "*qkv_proj", "*lm_head"]


def tiny_config(vocab: int = 131072, taps=TAPS, window: int = 16) -> dict:
    """The real config.json's keys (449d2414) at tiny widths."""
    q = {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
         "exclude_modules": QUANT_EXCLUDE, "quant_method": "modelopt_fp4", "ignore": QUANT_EXCLUDE}
    return {
        "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "block_size": BLOCK,
        "dflash_config": {"mask_token_id": MASK_ID, "target_layer_ids": list(taps), "use_swa": True,
                          "swa_window_size": window, "causal": False, "attention_sink_bias": True,
                          "projector_type": "lilicorr", "lilicorr_enabled": True, "lilicorr_candidate_topk": 8,
                          "lilicorr_hidden_size": 64, "lilicorr_num_layers": 2, "lilicorr_num_heads": 4,
                          "lilicorr_mlp_ratio": 2.0, "lilicorr_factor_dim": 64, "lilicorr_vector_eps": 0.0001,
                          "lilicorr_logit_scale": 8.0, "conv_kernel_size": 2, "conv_group_size": 16},
        "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
        "intermediate_size": 512, "hidden_act": "silu", "rms_norm_eps": 1e-06, "vocab_size": vocab,
        "max_position_embeddings": 262144, "initializer_range": 0.02, "attention_bias": False,
        "attention_dropout": 0.0, "rope_theta": 10000.0, "rope_scaling": None, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16", "layer_types": ["sliding_attention"] * 2, "sliding_window": window,
        "attention_sink_bias": True, "use_sliding_window": True, "max_window_layers": 0, "quantization_config": q}


def nvfp4_names(cfg: dict) -> list[str]:
    """The NVFP4 linears' base names (spec 26 §1: 25 on the real checkpoint)."""
    n = ["fc"]
    for i in range(cfg["num_hidden_layers"]):
        n += [f"layers.{i}.mlp.{p}_proj" for p in ("gate", "up", "down")]
    dc = cfg["dflash_config"]
    n += ["lilicorr.token_proj", "lilicorr.pass_hidden_proj", "lilicorr.context_proj"]
    for i in range(dc["lilicorr_num_layers"]):
        n += [f"lilicorr.layers.{i}.attn.out_proj", f"lilicorr.layers.{i}.mlp.up_proj", f"lilicorr.layers.{i}.mlp.down_proj"]
    return n


# ---------------------------------------------------------------------------------------------------- NVFP4 packing

def _e4m3_values() -> torch.Tensor:
    """The 126 finite non-negative e4m3fn values of codes 0x00..0x7E (ascending in the code)."""
    v = []
    for c in range(0x7F):
        e, m = c >> 3, c & 7
        v.append(m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))
    return torch.tensor(v, dtype=torch.float64)


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Non-negative fp64 -> the nearest e4m3fn code (ties to the even code), saturating at 448 (0x7E)."""
    vals = _e4m3_values()
    x = x.double().clamp(0, 448.0)
    hi = torch.searchsorted(vals, x).clamp(max=len(vals) - 1)
    lo = (hi - 1).clamp(min=0)
    dl, dh = (x - vals[lo]).abs(), (vals[hi] - x).abs()
    pick = torch.where(dh < dl, hi, torch.where(dl < dh, lo, torch.where(hi % 2 == 0, hi, lo)))
    return pick.to(torch.uint8)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """fp64 -> the nearest e2m1 code (sign in bit 3; ties to the even magnitude index), saturating at 6."""
    mags = torch.tensor(E2M1, dtype=torch.float64)
    a = x.abs().clamp(max=6.0)
    d = (a[..., None] - mags).abs()
    hit = d == d.min(-1, keepdim=True).values
    ar = torch.arange(8)
    even = torch.where(hit & (ar % 2 == 0), ar, 99).min(-1).values
    i = torch.where(even < 99, even, torch.where(hit, ar, 99).min(-1).values)
    return (i | (((x < 0) & (i > 0)).long() << 3)).to(torch.uint8)


def nvfp4_pack(w: torch.Tensor, gen: torch.Generator, edges: bool = True) -> dict[str, torch.Tensor]:
    """w [N][K] -> {weight U8 [N][K/2], weight_scale F8_E4M3 [N][K/16], weight_scale_2 F32 []} (ModelOpt's form)."""
    N, K = w.shape
    wd = w.double()
    g = float(torch.tensor(wd.abs().max().item() / (6.0 * 448.0) * 1.37, dtype=torch.float32))
    blk = wd.view(N, K // 16, 16)
    sc = e4m3_encode(blk.abs().amax(-1) / (6.0 * g))
    if edges and N >= 4 and K >= 64:      # a zero-scale block and subnormal scales: the decoder's edges
        sc[0, 0] = 0
        sc[1, : min(7, K // 16)] = torch.arange(1, min(7, K // 16) + 1, dtype=torch.uint8)
    sv = _e4m3_values()[sc.long()]
    q = e2m1_encode(torch.where(sv[..., None] > 0, blk / (sv[..., None] * g), torch.zeros_like(blk))).view(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    return {"weight": packed.contiguous(), "weight_scale": sc.view(torch.float8_e4m3fn).contiguous(),
            "weight_scale_2": torch.tensor(g, dtype=torch.float32)}


# ---------------------------------------------------------------------------------------------------- the weights

def make_tensors(cfg: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors (bare names; NVFP4 linears packed, everything else bf16)."""
    gen = torch.Generator().manual_seed(seed)
    dc = cfg["dflash_config"]
    H, nh, nkv, hd, ff = (cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"],
                          cfg["intermediate_size"])
    Hl, F_, k, S = dc["lilicorr_hidden_size"], dc["lilicorr_factor_dim"], dc["lilicorr_candidate_topk"], BLOCK - 1
    lh = dc["lilicorr_num_heads"]
    mlp = int(Hl * dc["lilicorr_mlp_ratio"])
    G = H // dc["conv_group_size"]
    T = len(dc["target_layer_ids"])
    f32: dict[str, torch.Tensor] = {}

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=gen) * s

    def lin(name, o, i, bias=False, s=1.0):
        f32[name + ".weight"] = rn(o, i, s=s / math.sqrt(i))
        if bias:
            f32[name + ".bias"] = rn(o, s=0.1)

    def norm(name, n):
        f32[name] = 1.0 + rn(n, s=0.1)
    lin("fc", H, T * H)
    norm("hidden_norm.weight", H)
    norm("norm.weight", H)
    for i in range(cfg["num_hidden_layers"]):
        L = f"layers.{i}."
        norm(L + "input_layernorm.weight", H)
        norm(L + "post_attention_layernorm.weight", H)
        lin(L + "self_attn.q_proj", nh * hd, H)
        lin(L + "self_attn.k_proj", nkv * hd, H)
        lin(L + "self_attn.v_proj", nkv * hd, H)
        lin(L + "self_attn.o_proj", H, nh * hd)
        norm(L + "self_attn.q_norm.weight", hd)
        norm(L + "self_attn.k_norm.weight", hd)
        f32[L + "self_attn.attention_sink_bias"] = rn(nh, s=1.5)
        lin(L + "mlp.gate_proj", ff, H)
        lin(L + "mlp.up_proj", ff, H)
        lin(L + "mlp.down_proj", H, ff)
        for cv in ("attention_conv", "mlp_conv"):
            base = rn(2, 2, H, s=0.3)
            base[:, 0] += 1.0
            f32[L + cv + ".base_kernel"] = base
            lin(L + cv + ".kernel_projection", 2 * 2 * G, H, s=0.3)
    P = "lilicorr."
    lin(P + "token_proj", Hl, H, bias=True)
    lin(P + "pass_hidden_proj", Hl, H, bias=True)
    f32[P + "feature_norm.weight"] = 1.0 + rn(5, s=0.1)
    f32[P + "feature_norm.bias"] = rn(5, s=0.1)
    lin(P + "feature_mlp.up_proj", Hl, 5, bias=True)
    lin(P + "feature_mlp.down_proj", Hl, Hl, bias=True)
    f32[P + "slot_embedding"] = rn(1, 1, S, 1, Hl, s=0.5)
    f32[P + "rank_embedding"] = rn(1, 1, 1, k, Hl, s=0.5)
    f32[P + "relative_slot_bias"] = rn(lh, 2 * BLOCK - 1, s=1.0)
    f32[P + "same_slot_bias"] = rn(lh, s=1.0)
    lin(P + "context_proj", Hl, H, bias=True)
    for i in range(dc["lilicorr_num_layers"]):
        L = f"{P}layers.{i}."
        norm(L + "attn_norm.weight", Hl)
        f32[L + "attn.in_proj_weight"] = rn(3 * Hl, Hl, s=1.0 / math.sqrt(Hl))
        f32[L + "attn.in_proj_bias"] = rn(3 * Hl, s=0.1)
        lin(L + "attn.out_proj", Hl, Hl, bias=True)
        norm(L + "mlp_norm.weight", Hl)
        lin(L + "mlp.up_proj", mlp, Hl, bias=True)
        lin(L + "mlp.down_proj", Hl, mlp, bias=True)
    norm(P + "output_norm.weight", Hl)
    norm(P + "anchor_norm.weight", Hl)
    lin(P + "factor_input_proj", Hl, 3 * Hl, bias=True)
    lin(P + "out_head", F_, Hl, bias=True)
    lin(P + "in_head", F_, Hl, bias=True)
    lin(P + "anchor_out_head", F_, Hl, bias=True)
    quant = set(nvfp4_names(cfg))
    out = {}
    for n, v in f32.items():
        base = n[: -len(".weight")] if n.endswith(".weight") else None
        if base in quant:
            for s, t in nvfp4_pack(v, gen).items():
                out[f"{base}.{s}"] = t
        else:
            out[n] = v.to(torch.bfloat16).contiguous()
    return out


def hf_quant_config(cfg: dict) -> dict:
    q = cfg["quantization_config"]
    return {"producer": {"name": "modelopt", "version": "0.0.0"},
            "quantization": {"quant_algo": q["quant_algo"], "kv_cache_quant_algo": None, "group_size": q["group_size"],
                             "exclude_modules": q["exclude_modules"]}}


def make(out_dir: str, seed: int = 0, vocab: int = 131072, taps=TAPS, window: int = 16) -> str:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    cfg = tiny_config(vocab, taps, window)
    t = make_tensors(cfg, seed)
    save_file(t, os.path.join(out_dir, "model.safetensors.tmp"), metadata={"format": "pt"})
    os.replace(os.path.join(out_dir, "model.safetensors.tmp"), os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "hf_quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_quant_config(cfg), f, indent=2)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=131072)
    ap.add_argument("--taps", default=",".join(str(i) for i in TAPS))
    ap.add_argument("--window", type=int, default=16)
    a = ap.parse_args()
    if a.vocab <= MASK_ID:
        sys.exit(f"--vocab {a.vocab}: the mask id {MASK_ID} must be a row")
    make(a.out, a.seed, a.vocab, [int(x) for x in a.taps.split(",")], a.window)
    print(f"tiny LiLiCorr -> {a.out}")


if __name__ == "__main__":
    main()
