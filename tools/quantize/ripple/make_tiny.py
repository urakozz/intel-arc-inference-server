#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""A tiny random Ripple-65B checkpoint for tools/quantize/quantize_ripple.sh's selftest.

    make_tiny.py <out-dir> --source <Ripple-65B-Preview snapshot> [--seed 0]

The real snapshot's config.json at toy size - hidden 256, 5 layers (4 sliding + the full NoPE layer 4, as layers
0-4 of the real model), 4 q / 2 kv heads x 128, MLP 1024, sliding window 32 (so a 128-token calibration row is
cut by the window, as the real 2048-token rows are by 1024), embed_scale sqrt(256) - with everything else as the
real config has it (value residuals, the gate in fp32, the FFN clamp at 20, the untied head, the 100352-row
vocabulary, so the real tokenizer and chat template work). Every in-feature is a multiple of 64 (the int4 group).
The remote code (configuration_ricursive.py, modeling_ricursive.py), the tokenizer files, the chat template and
generation_config.json are copied from the snapshot unchanged, so AutoRound loads the tiny model through the same
`auto_map` / trust_remote_code path as the real one. Weights are random bf16, written as two safetensors shards
plus an index like the real snapshot. The tensor names are checked against the real index (layers 0-4 and the
top-level tensors): the same set, or exit 1.
"""
import argparse
import json
import math
import os
import shutil
import sys

import torch

TINY = dict(hidden_size=256, intermediate_size=1024, num_hidden_layers=5, num_attention_heads=4,
            num_key_value_heads=2, head_dim=128, sliding_window=32, embed_scale=16.0,
            layer_types=["sliding_attention"] * 4 + ["full_attention"])
COPY = ("configuration_ricursive.py", "modeling_ricursive.py", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "chat_template.jinja", "generation_config.json")


def shapes(c: dict) -> dict:
    H, I, V, hd = c["hidden_size"], c["intermediate_size"], c["vocab_size"], c["head_dim"]
    q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
    out = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,), "lm_head.weight": (V, H)}
    for i in range(c["num_hidden_layers"]):
        L = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            out[L + n + ".weight"] = (H,)
        out.update({L + "self_attn.q_proj.weight": (q, H), L + "self_attn.k_proj.weight": (kv, H),
                    L + "self_attn.v_proj.weight": (kv, H), L + "self_attn.g_proj.weight": (q, H),
                    L + "self_attn.o_proj.weight": (H, q), L + "self_attn.q_norm.weight": (hd,),
                    L + "self_attn.k_norm.weight": (hd,), L + "self_attn.value_res_lambda": (1,),
                    L + "mlp.gate_proj.weight": (I, H), L + "mlp.up_proj.weight": (I, H),
                    L + "mlp.down_proj.weight": (H, I)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--source", required=True, help="the real Ripple-65B-Preview snapshot (config, code, tokenizer)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from safetensors.torch import save_file
    with open(os.path.join(a.source, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    with open(os.path.join(a.source, "model.safetensors.index.json"), encoding="utf-8") as f:
        real = set(json.load(f)["weight_map"])
    keep = {f"model.layers.{i}." for i in range(TINY["num_hidden_layers"])}
    real = {n for n in real if not n.startswith("model.layers.") or n.startswith(tuple(keep))}
    c.update(TINY)
    c.pop("hybrid_layer_pattern", None)   # derived from layer_types by RicursiveConfig
    shp = shapes(c)
    if set(shp) != real:
        sys.exit(f"make_tiny: tensor names differ from the real index: only ours {sorted(set(shp) - real)[:5]}, "
                 f"only the real {sorted(real - set(shp))[:5]}")
    g = torch.Generator().manual_seed(a.seed)
    H = c["hidden_size"]
    sd = {}
    for name, s in shp.items():
        if name.endswith("value_res_lambda"):
            t = torch.randn(s, generator=g)                       # blend sigmoid(lambda) away from 0.5
        elif name.endswith("norm.weight"):
            t = 1 + 0.2 * torch.randn(s, generator=g)
        elif name == "model.embed_tokens.weight":
            t = torch.randn(s, generator=g) / c["embed_scale"]    # rows of RMS ~1 after embed_scale
        elif name == "lm_head.weight":
            t = torch.randn(s, generator=g) * H ** -0.5
        else:
            t = torch.randn(s, generator=g) / math.sqrt(s[1])
        sd[name] = t.to(torch.bfloat16).contiguous()
    os.makedirs(a.out, exist_ok=True)
    names = sorted(sd)
    wm = {}
    for k, part in enumerate((names[: len(names) // 2], names[len(names) // 2:])):
        fn = f"model-{k + 1:05d}-of-00002.safetensors"
        save_file({n: sd[n] for n in part}, os.path.join(a.out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in part})
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())}, "weight_map": wm}, f, indent=1)
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
    for fn in COPY:
        shutil.copy(os.path.join(a.source, fn), os.path.join(a.out, fn))
    print(f"wrote {a.out}: {len(sd)} tensors ({len(real)} names as in the real index), "
          f"{sum(t.numel() for t in sd.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
