#!/usr/bin/env python3
"""Spec 30 plan 30a Task 2: a random tiny Laguna S 2.1 in the real model's form (tests only).

    laguna_make_tiny.py <out dir> --vendor <the original's snapshot dir> [--hidden 128] [--layers 8] [--experts 16]
        [--topk 4] [--seed 0]

The vendor's own config.json, cut down: `--layers` layers of the real `layer_types` (full, sliding x 3, ...), layer 0
dense (`mlp_only_layers` [0]), the per-layer query heads 48 / 72 cut to 4 / 6 over 2 key-value heads (head size 128,
so the yarn table's 64 rotating dimensions and the sliding window of 512 stay the real ones), `--experts` routed
experts top-`--topk` of width 64, a shared expert of width 64, the dense MLP 256, the real vocabulary (the real
tokenizer then works on it). Everything else (rope parameters, gate kind, the 2.5 routed scale, sigmoid routing,
the eos / pad ids) is the original's.

The weights are written as the original's file names them: per-expert `mlp.experts.<e>.{gate,up,down}_proj.weight`,
the selection bias as `mlp.experts.e_score_correction_bias` (the vendor's code remaps it onto the router at load),
`mlp.shared_expert.*`, bf16. The selection bias is random (not zero) so the facts can tell ranking from weighting.
The vendor's `configuration_laguna.py` / `modeling_laguna.py` and the tokenizer files are copied from --vendor
(the snapshot's small files: on the box `~/b70/hdd/hf-hub/models--poolside--Laguna-S-2.1/snapshots/*/`).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

VENDOR_FILES = ("configuration_laguna.py", "modeling_laguna.py", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
HEADS = {48: 4, 72: 6}           # the real per-layer query heads -> the tiny's


def config(vendor: str, hidden=128, layers=8, experts=16, topk=4) -> dict:
    with open(os.path.join(vendor, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    c.update(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=HEADS[c["num_attention_heads"]],
             num_key_value_heads=2, intermediate_size=256, moe_intermediate_size=64,
             shared_expert_intermediate_size=64, num_experts=experts, num_experts_per_tok=topk)
    for k in ("layer_types", "mlp_layer_types", "gating_types"):
        c[k] = c[k][:layers]
    c["num_attention_heads_per_layer"] = [HEADS[h] for h in c["num_attention_heads_per_layer"][:layers]]
    c["torch_dtype"] = "bfloat16"
    return c


def checkpoint_shapes(c: dict) -> dict[str, tuple]:
    """Every tensor of the original's file form for config c, name -> shape."""
    H, D, KV = c["hidden_size"], c["head_dim"], c["num_key_value_heads"]
    E, I, S = c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    out = {"model.embed_tokens.weight": (c["vocab_size"], H), "model.norm.weight": (H,),
           "lm_head.weight": (c["vocab_size"], H)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        q = c["num_attention_heads_per_layer"][i]
        out.update({p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                    p + "self_attn.q_proj.weight": (q * D, H), p + "self_attn.k_proj.weight": (KV * D, H),
                    p + "self_attn.v_proj.weight": (KV * D, H), p + "self_attn.o_proj.weight": (H, q * D),
                    p + "self_attn.g_proj.weight": (q, H), p + "self_attn.q_norm.weight": (D,),
                    p + "self_attn.k_norm.weight": (D,)})
        if i in c["mlp_only_layers"]:
            F = c["intermediate_size"]
            out.update({p + "mlp.gate_proj.weight": (F, H), p + "mlp.up_proj.weight": (F, H),
                        p + "mlp.down_proj.weight": (H, F)})
            continue
        out.update({p + "mlp.gate.weight": (E, H), p + "mlp.experts.e_score_correction_bias": (E,),
                    p + "mlp.shared_expert.gate_proj.weight": (S, H), p + "mlp.shared_expert.up_proj.weight": (S, H),
                    p + "mlp.shared_expert.down_proj.weight": (H, S)})
        for e in range(E):
            out.update({p + f"mlp.experts.{e}.gate_proj.weight": (I, H), p + f"mlp.experts.{e}.up_proj.weight": (I, H),
                        p + f"mlp.experts.{e}.down_proj.weight": (H, I)})
    return out


def random_state(c: dict, gen: torch.Generator) -> dict[str, torch.Tensor]:
    sd = {}
    for k, s in sorted(checkpoint_shapes(c).items()):
        if k.endswith("e_score_correction_bias"):
            sd[k] = 0.05 * torch.randn(s, generator=gen)
        elif len(s) == 1:                                   # RMSNorm weights (plain w, not 1 + w)
            sd[k] = 1.0 + 0.1 * torch.randn(s, generator=gen)
        elif k.endswith("embed_tokens.weight"):
            sd[k] = torch.randn(s, generator=gen)
        elif k.endswith("mlp.gate.weight"):
            sd[k] = 3.0 * torch.randn(s, generator=gen) / math.sqrt(s[-1])
        else:
            sd[k] = torch.randn(s, generator=gen) / math.sqrt(s[-1])
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}


def make(out: str, vendor: str, hidden=128, layers=8, experts=16, topk=4, seed=0) -> dict:
    c = config(vendor, hidden, layers, experts, topk)
    sd = random_state(c, torch.Generator().manual_seed(seed))
    os.makedirs(out, exist_ok=True)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())},
                   "weight_map": {k: "model.safetensors" for k in sorted(sd)}}, f)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, indent=1)
    for fn in VENDOR_FILES:
        if os.path.exists(os.path.join(vendor, fn)):
            shutil.copyfile(os.path.join(vendor, fn), os.path.join(out, fn))
    for fn in ("configuration_laguna.py", "modeling_laguna.py"):
        if not os.path.exists(os.path.join(out, fn)):
            raise SystemExit(f"laguna_make_tiny: {vendor} has no {fn} (the vendor's model code)")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--vendor", required=True, help="the original's snapshot (its small files are read)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c = make(a.out, a.vendor, a.hidden, a.layers, a.experts, a.topk, a.seed)
    print(f"wrote {a.out}: {c['num_hidden_layers']} layers {c['layer_types']}, heads "
          f"{c['num_attention_heads_per_layer']} over {c['num_key_value_heads']}, {c['num_experts']} experts "
          f"top-{c['num_experts_per_tok']}")


if __name__ == "__main__":
    main()
