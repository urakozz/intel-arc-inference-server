#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A synthetic LFM2.5-24B-A2B checkpoint at the REAL widths, in our int4 g64 export's form, both
expert spellings (spec 24b Task 1).

    make_synth.py <out-dir> --layers N [--expert-names w|proj] --tokenizer <dir> [--seed 0] [--plant-sign]

Writes config.json (the real LFM2.5-24B-A2B config - LiquidAI/LFM2.5-24B-A2B@1769516a's, embedded here: there
is no vendored copy under tools/oracle and make_synth.py's interface takes no --snapshot) truncated to
`num_hidden_layers = N` and `layer_types = the real pattern[:N]` (N in 3..40; the default 4 holds every layer
kind: layers 0-1 dense conv, layer 2 the first attention + MoE layer, layer 3 a conv + MoE layer), plus
`quantization_config` as AutoRound 0.17.0 writes it (spec 24 §5: no extra_config - the router, expert_bias,
conv.conv.weight, every norm and the embedding are never nn.Linear / nn.Conv1d, so AutoRound skips them without
an exclusion list).

Every `nn.Linear` (routed experts, dense w1/w2/w3, conv in_proj/out_proj, attention q/k/v/out_proj) is packed int4
g64 sym by `loader::rtn_int4_g64`'s rule (src/loader/rtn.h:20-27: s16 = f16(2 amax / 15), q = clamp(rint(w / s16)
+ 8, 0, 15)) - reusing tools/oracle/lfm2moe_make_tiny.py's `rtn_int4` (the same formula, already exercised by
test_lfm2moe_make_tiny.py's int4-g64 variant) rather than a second copy. Everything else ships bf16 as drawn
(`expert_bias` fp32: the module's buffer is fp32 whatever the model dtype, M:205). The expert spelling
(`feed_forward.experts.E.{w1,w3,w2}` or `.{gate_proj,up_proj,down_proj}`) is chosen by `--expert-names`
(spec 24 §5: 24q's export may carry either; the loader (24b) reads both, all-or-nothing per checkpoint).

`--plant-sign` (Review Focus 1 at the source): for expert 0 of the FIRST MoE layer, w3 (up) is written as the
exact negation of w1 (gate) before packing - the device's gate||up block must then hold the up tile as the
negation of the gate tile, whichever spelling the names arrived in (lfm2moe_repack_test.cc; test_gate_up_sign
here checks the packed bytes dequantise to exact negatives).

The draws (lfm2moe_make_tiny.py's, at the real widths): norms 1 + 0.1 N(0, 1) (RMSNorm's plain w is centred near
1), linears N(0, 1) / sqrt(K), the router 3 N(0, 1) / sqrt(hidden), expert_bias 0.05 N(0, 1) fp32, conv taps
0.5 N(0, 1), the embedding N(0, 1). One layer is generated, packed and written at a time (one shard a layer;
shard 0 holds the embedding and the final norm), so the peak is about one layer's draws, not the model (the
widest layer - a MoE layer - is ~160 MB of fp32 draws before packing). Size at N = 4 (derived, both dense conv
layers 47.3 MB each, the attention + MoE layer 326.7 MB, the conv + MoE layer 330.0 MB, the embedding 524.3 MB):
~1.28 GB.

Afterwards `check.py <out-dir>` must print ACCEPTED (names both ways, dtypes, shapes, qzeros, g64 scales,
quantization_config); the packer's arithmetic is test_lfm2moe_quant.py's test_synth_pack_roundtrip /
test_gate_up_sign. The tokenizer files (tokenizer.json, tokenizer_config.json, generation_config.json,
chat_template.jinja) are copied from --tokenizer when given; a missing one is reported, not fatal (the engine
reads none of them; tools/oracle reads tokenizer.json).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

AR_VERSION = "0.17.0"
H = 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 64
DENSE_INTER = 11776
MOE_INTER = 1536
EXPERTS, TOP_K = 64, 4
NUM_LAYERS = 40
NUM_DENSE_LAYERS = 2
VOCAB = 128000
EXPERT_SPELLINGS = {"w": ("w1", "w3", "w2"), "proj": ("gate_proj", "up_proj", "down_proj")}

REAL_LAYER_TYPES = ["full_attention" if l % 4 == 2 else "conv" for l in range(NUM_LAYERS)]


def real_config() -> dict:
    """The real LiquidAI/LFM2.5-24B-A2B config.json (snapshot 1769516a), embedded (spec 1.4: a configuration
    file the LFM Open License v1.0 covers like any other - its source is this docstring's citation, not a
    copied file)."""
    return {
        "architectures": ["Lfm2MoeForCausalLM"], "bos_token_id": 124894, "conv_L_cache": 3, "conv_bias": False,
        "dtype": "bfloat16", "eos_token_id": 124900, "hidden_size": H, "initializer_range": 0.02,
        "intermediate_size": DENSE_INTER, "layer_types": list(REAL_LAYER_TYPES),
        "max_position_embeddings": 131072, "model_type": "lfm2_moe", "moe_intermediate_size": MOE_INTER,
        "norm_eps": 1e-05, "norm_topk_prob": True, "num_attention_heads": Q_HEADS,
        "num_dense_layers": NUM_DENSE_LAYERS, "num_experts": EXPERTS, "num_experts_per_tok": TOP_K,
        "num_hidden_layers": NUM_LAYERS, "num_key_value_heads": KV_HEADS, "pad_token_id": 124893,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}, "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True, "use_cache": True, "use_expert_bias": True, "vocab_size": VOCAB,
    }


def synth_config(layers: int) -> dict:
    if not 1 <= layers <= NUM_LAYERS:
        raise ValueError(f"--layers {layers}: 1..{NUM_LAYERS}")
    if "full_attention" not in REAL_LAYER_TYPES[:layers]:
        raise ValueError(f"--layers {layers} keeps no attention layer (the first is layer 2)")
    d = real_config()
    d["num_hidden_layers"] = layers
    d["layer_types"] = REAL_LAYER_TYPES[:layers]
    d["quantization_config"] = {
        "quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "bits": 4,
        "group_size": C.GROUP, "sym": True, "data_type": "int", "iters": 0,
        "autoround_version": AR_VERSION,
    }
    return d


def _norm(n: int, g: torch.Generator) -> torch.Tensor:
    return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)


def _lin(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    return (torch.randn(n, k, generator=g) / math.sqrt(k)).to(torch.bfloat16)


def layer_tensors(i: int, layer_type: str, seed: int, plant_sign: bool) -> dict:
    """Layer i's checkpoint-name tensors (bf16 / fp32, unpacked) - plant_sign negates w3 for expert 0 when
    this is the first MoE layer (i >= NUM_DENSE_LAYERS and the smallest such i)."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + 17 + i)
    p = f"model.layers.{i}."
    out = {p + "operator_norm.weight": _norm(H, g), p + "ffn_norm.weight": _norm(H, g)}
    if layer_type == "full_attention":
        qn, kvn = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
        out[p + "self_attn.q_layernorm.weight"] = _norm(HEAD_DIM, g)
        out[p + "self_attn.k_layernorm.weight"] = _norm(HEAD_DIM, g)
        out[p + "self_attn.q_proj.weight"] = _lin(qn, H, g)
        out[p + "self_attn.k_proj.weight"] = _lin(kvn, H, g)
        out[p + "self_attn.v_proj.weight"] = _lin(kvn, H, g)
        out[p + "self_attn.out_proj.weight"] = _lin(H, qn, g)
    else:
        out[p + "conv.conv.weight"] = (0.5 * torch.randn(H, 1, 3, generator=g)).to(torch.bfloat16)
        out[p + "conv.in_proj.weight"] = _lin(3 * H, H, g)
        out[p + "conv.out_proj.weight"] = _lin(H, H, g)
    f = p + "feed_forward."
    if i < NUM_DENSE_LAYERS:
        out[f + "w1.weight"] = _lin(DENSE_INTER, H, g)
        out[f + "w3.weight"] = _lin(DENSE_INTER, H, g)
        out[f + "w2.weight"] = _lin(H, DENSE_INTER, g)
        return out
    out[f + "gate.weight"] = (3.0 * torch.randn(EXPERTS, H, generator=g) / math.sqrt(H)).to(torch.bfloat16)
    out[f + "expert_bias"] = (0.05 * torch.randn(EXPERTS, generator=g)).float()
    for e in range(EXPERTS):
        w1 = _lin(MOE_INTER, H, g)
        w3 = _lin(MOE_INTER, H, g)
        if plant_sign and e == 0 and i == next(l for l in range(NUM_LAYERS) if l >= NUM_DENSE_LAYERS):
            w3 = (-w1.float()).to(torch.bfloat16)
        out[f"{f}experts.{e}.w1.weight"] = w1
        out[f"{f}experts.{e}.w3.weight"] = w3
        out[f"{f}experts.{e}.w2.weight"] = _lin(H, MOE_INTER, g)
    return out


def top_tensors(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed * 1_000_003 + 5)
    return {"model.embed_tokens.weight": (torch.randn(VOCAB, H, generator=g)).to(torch.bfloat16),
            "model.embedding_norm.weight": _norm(H, g)}


LINEAR_SUFFIXES = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".out_proj.weight",
                   ".in_proj.weight", ".w1.weight", ".w2.weight", ".w3.weight")


def is_linear(name: str) -> bool:
    return name.endswith(LINEAR_SUFFIXES)


def expert_rename(name: str, spelling: str) -> str:
    if ".experts." not in name or spelling == "w":
        return name
    for src, dst in zip(EXPERT_SPELLINGS["w"], EXPERT_SPELLINGS[spelling]):
        suf = f".{src}.weight"
        if name.endswith(suf):
            return name[: -len(suf)] + f".{dst}.weight"
    return name


def pack(sd: dict, spelling: str) -> dict:
    """Every nn.Linear of `sd` packed int4 g64 (loader::rtn_int4_g64's rule); everything else kept as is.
    Expert tensors renamed to `spelling` first."""
    MT = C.lfm2moe_make_tiny()
    out = {}
    for name, w in sd.items():
        if is_linear(name):
            qw, qz, sc = MT.rtn_int4(w)
            base = expert_rename(name, spelling)[: -len(".weight")]
            out[base + ".qweight"], out[base + ".qzeros"], out[base + ".scales"] = qw, qz, sc
        else:
            out[expert_rename(name, spelling)] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--expert-names", choices=("w", "proj"), default="w")
    ap.add_argument("--tokenizer", help="an LFM2.5 snapshot directory (tokenizer.json etc.)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plant-sign", action="store_true",
                    help="Review Focus 1 at the source: w3 = -w1 for expert 0 of the first MoE layer")
    a = ap.parse_args()
    from safetensors.torch import save_file
    ref = C.lfm2moe_ref()
    cfg = synth_config(a.layers)
    layer_types = cfg["layer_types"]
    rc = ref.config(None, raw={k: v for k, v in cfg.items() if k != "quantization_config"})
    os.makedirs(a.out, exist_ok=True)
    n_shards = 1 + a.layers
    wm, total = {}, 0
    t0 = time.time()

    def write(k: int, sd: dict) -> None:
        nonlocal total
        fn = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(sd, os.path.join(a.out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in sd})
        total += sum(t.numel() * t.element_size() for t in sd.values())

    write(0, pack(top_tensors(a.seed), a.expert_names))
    for i in range(a.layers):
        write(1 + i, pack(layer_tensors(i, layer_types[i], a.seed, a.plant_sign), a.expert_names))
        print(f"  layer {i} ({layer_types[i]}) written, {time.time() - t0:.0f}s", flush=True)

    want = set(ref.expected_names(rc, "int4", a.expert_names))
    if want != set(wm):
        raise SystemExit(f"make_synth: names differ from lfm2moe_ref.expected_names: missing "
                         f"{sorted(want - set(wm))[:5]}, extra {sorted(set(wm) - want)[:5]}")
    C.write_json(os.path.join(a.out, "model.safetensors.index.json"),
                 {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))})
    C.write_json(os.path.join(a.out, "config.json"), cfg)
    C.write_json(os.path.join(a.out, "lfm2moe_recipe.json"),
                 {"synthetic": True, "layers": a.layers, "expert_names": a.expert_names, "seed": a.seed,
                  "plant_sign": a.plant_sign,
                  "maker": "tools/quantize/lfm2moe/make_synth.py (RTN of random weights, spec 24b)"})
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        src = os.path.join(a.tokenizer, fn) if a.tokenizer else ""
        if a.tokenizer and os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, fn))
        elif a.tokenizer:
            print(f"make_synth: no {fn} in {a.tokenizer} - not copied", file=sys.stderr)
    print(f"wrote {a.out}: {len(wm)} tensors in {n_shards} shards, {total / 1e9:.2f} GB, "
          f"{a.layers} layers ({layer_types.count('full_attention')} full attention), "
          f"expert names '{a.expert_names}', {time.time() - t0:.0f}s")
    print(json.dumps({"layers": a.layers, "expert_names": a.expert_names, "bytes": total}))


if __name__ == "__main__":
    main()
