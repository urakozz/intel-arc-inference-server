#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A tiny random Kolibri-1 checkpoint for testing tools/quantize_kolibri1.sh end to end.

    make_tiny.py <out-dir> --tokenizer <dir with tokenizer.json + tokenizer_config.json> [--seed 0]

The real config.json's structure (third_party/kolibri1/config.json) at toy size - hidden 128, 5
layers (4 sliding + 1 full), 4 x 32 heads / 2 kv heads, 16 experts top 6 of width 128, a shared
expert of 128 - with the real vocabulary size (128000), so the real tokenizer and chat template
work. Every in-feature is a multiple of 64 (the int4 group). Weights are random bf16, written as two
safetensors shards plus an index, as the real snapshot is laid out; the tokenizer files are copied.
"""
import argparse
import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TINY = dict(hidden_size=128, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            num_experts=16, num_experts_per_tok=6, moe_intermediate_size=128, shared_expert_intermediate_size=128,
            sliding_window=33, max_position_embeddings=8192,
            layer_types=["sliding_attention"] * 4 + ["full_attention"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from safetensors.torch import save_file
    ref = C.kolibri_ref()
    d = C.read_json(os.path.join(C.PORT_DIR, "config.json"))
    d.update(TINY)
    c = ref.KConfig.from_dict(d)
    g = torch.Generator().manual_seed(a.seed)
    sd = {}
    for name in ref.expected_names(c):
        D = c.hidden_size
        if name.endswith("expert_bias"):
            t = 0.5 * torch.randn(c.num_experts, generator=g)
        elif name.endswith(("q_norm.weight", "k_norm.weight")):
            t = 1 + 0.2 * torch.randn(c.head_dim, generator=g)
        elif name.endswith("norm.weight"):
            t = 1 + 0.2 * torch.randn(D, generator=g)
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            t = torch.randn(c.vocab_size, D, generator=g) * (1.0 if "embed" in name else D ** -0.5)
        else:
            tail = name.split(".", 3)[-1]
            hd = c.num_attention_heads * c.head_dim
            kv = c.num_key_value_heads * c.head_dim
            shp = {"self_attn.q_proj.weight": (hd, D), "self_attn.k_proj.weight": (kv, D),
                   "self_attn.v_proj.weight": (kv, D), "self_attn.o_proj.weight": (D, hd),
                   "mlp.gate.weight": (c.num_experts, D)}.get(tail)
            if shp is None:
                w = c.shared_expert_intermediate_size if "shared_experts" in name else c.moe_intermediate_size
                shp = (D, w) if name.endswith("down_proj.weight") else (w, D)
            s = 3.0 if tail == "mlp.gate.weight" else 1.0
            t = torch.randn(shp, generator=g) * (s / math.sqrt(shp[1]))
        sd[name] = t.to(torch.bfloat16).contiguous()
    os.makedirs(a.out, exist_ok=True)
    names = sorted(sd)
    wm = {}
    for k, part in enumerate((names[: len(names) // 2], names[len(names) // 2:])):
        fn = f"model-{k + 1:05d}-of-00002.safetensors"
        save_file({n: sd[n] for n in part}, os.path.join(a.out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in part})
    C.write_json(os.path.join(a.out, "model.safetensors.index.json"),
                 {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())}, "weight_map": wm})
    C.write_json(os.path.join(a.out, "config.json"), d)
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(a.tokenizer, fn)
        if not os.path.exists(src):
            src = os.path.join(C.PORT_DIR, fn)
        shutil.copy(src, os.path.join(a.out, fn))
    print(f"wrote {a.out}: {len(sd)} tensors, {sum(t.numel() for t in sd.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
