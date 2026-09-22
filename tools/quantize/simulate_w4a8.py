#!/usr/bin/env python3
"""Simulate W4A8 on the checkpoint we actually ship, without re-quantising it.

The engine's W4A8 design (docs/superpowers/specs/2026-09-22-w4a8-dpas-probe-design.md)
keeps the existing int4 g64 weights untouched and quantises ACTIVATIONS to int8
dynamically, per token, at runtime. So the accuracy question is not about a new
checkpoint: it is about what per-token int8 activations do to *these* weights.

This script answers that by loading the shipped checkpoint and inserting the
activation quantiser in front of each decoder linear, then comparing greedy
tokens against the same model with the quantiser off. Same weights, same
rounding, one variable.

What it is NOT: it does not simulate the int32 accumulation. The kernel computes
sum(q_w * q_x) exactly in int32 and rescales once per 64-element group, which is
mathematically what `(xq*xscale) @ W` does here up to fp32 accumulation order.
The activation rounding is the modelled error; accumulation order is not.

Usage (on the box, in a venv with torch+transformers and an XPU build):

  python3 tools/quantize/simulate_w4a8.py \
      --model ~/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/<sha> \
      --prompts tests/golden/prompts --n 32

  # localise damage: quantise only some layers
  ... --layer-range 0:16        # first 16 decoder layers only
  ... --stats-only              # no generation, just activation dynamic range
"""

import argparse
import glob
import os
import re
import sys

import torch

# The four int4 linears per layer, as the loader classifies them. lm_head stays
# bf16 in the engine (ruling A33) and must never be wrapped; norms, embeddings
# and every GDN internal op are outside the int4 path too.
TARGET = re.compile(r"(q_proj|k_proj|v_proj|qkv|qkvz|o_proj|out_proj|gate_proj|up_proj|gate_up|down_proj)$")
SKIP = re.compile(r"(lm_head|visual|vision)")

stats = {"n": 0, "ratio_sum": 0.0, "ratio_max": 0.0}


def quantise_per_token(x: torch.Tensor, collect: bool) -> torch.Tensor:
    """Symmetric per-token int8, the scale the kernel would compute at runtime."""
    f = x.float()
    amax = f.abs().amax(dim=-1, keepdim=True)
    if collect:
        med = f.abs().median(dim=-1, keepdim=True).values
        r = (amax / med.clamp_min(1e-12)).flatten()
        stats["n"] += r.numel()
        stats["ratio_sum"] += float(r.sum())
        stats["ratio_max"] = max(stats["ratio_max"], float(r.max()))
    scale = (amax / 127.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(f / scale), -127.0, 127.0)   # round() is RNE
    return (q * scale).to(x.dtype)


def wrap(model, layer_range, collect):
    """Install the activation quantiser in front of every targeted linear."""
    handles, names = [], []
    for name, mod in model.named_modules():
        if SKIP.search(name) or not TARGET.search(name):
            continue
        if layer_range is not None:
            m = re.search(r"layers\.(\d+)\.", name)
            if not m or not (layer_range[0] <= int(m.group(1)) < layer_range[1]):
                continue
        if not hasattr(mod, "forward"):
            continue
        handles.append(mod.register_forward_pre_hook(
            lambda _m, args, _c=collect: (quantise_per_token(args[0], _c),) + tuple(args[1:])))
        names.append(name)
    return handles, names


def read_ids(path):
    with open(path) as f:
        return [int(v) for v in f.read().split()]


@torch.no_grad()
def greedy(model, ids, n, vocab_used, device):
    """Greedy ids, argmax over [0, vocab_used) - the engine's masking."""
    out, cur = [], torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    for _ in range(n):
        r = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = r.past_key_values
        nxt = int(r.logits[0, -1, :vocab_used].argmax())
        out.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="tests/golden/prompts")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--vocab-used", type=int, default=248077)   # Qwen35::kVocabUsed
    ap.add_argument("--layer-range", default=None, help="e.g. 0:16, half-open")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    lr = None
    if args.layer_range:
        a, b = args.layer_range.split(":")
        lr = (int(a), int(b))

    device = args.device or ("xpu" if torch.xpu.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM
    print(f"loading {args.model} on {device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()

    files = sorted(glob.glob(os.path.join(args.prompts, "*.ids")))
    if not files:
        sys.exit(f"no .ids files under {args.prompts}")

    # Baseline first, with no hooks installed at all.
    base = {}
    if not args.stats_only:
        for f in files:
            ids = read_ids(f)
            base[f] = greedy(model, ids, args.n, args.vocab_used, device)
            print(f"  bf16-activation {os.path.basename(f):12s} {len(ids):5d} ids -> "
                  f"{base[f][:8]}{' ...' if args.n > 8 else ''}", flush=True)

    handles, names = wrap(model, lr, collect=True)
    print(f"\nwrapped {len(names)} linears"
          f"{'' if lr is None else f' in layers [{lr[0]},{lr[1]})'}; e.g. {names[:2]}", flush=True)
    if not names:
        sys.exit("wrapped nothing - check TARGET against this model's module names")

    worst = 0
    for f in files:
        ids = read_ids(f)
        got = greedy(model, ids, args.n, args.vocab_used, device)
        if args.stats_only:
            continue
        ref = base[f]
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
        same = sum(a == b for a, b in zip(ref, got))
        worst = max(worst, args.n - same)
        print(f"  int8-activation {os.path.basename(f):12s} {same}/{args.n} tokens match"
              f"{'' if first is None else f', first divergence at {first}'}", flush=True)
        if first is not None:
            print(f"      ref {ref[max(0,first-2):first+3]}\n      got {got[max(0,first-2):first+3]}")

    for h in handles:
        h.remove()

    if stats["n"]:
        print(f"\nactivation dynamic range over {stats['n']} tokens: "
              f"mean max/median {stats['ratio_sum']/stats['n']:.1f}, worst {stats['ratio_max']:.1f}")
        print("  (this, not the token count, is what predicts behaviour on layers not tested here)")
    if not args.stats_only:
        print(f"\nworst prompt lost {worst} of {args.n} tokens.")
        print("A token difference is NOT automatically a failure - the engine's own gate allows"
              " undetermined rows inside the golden tie set - but it is the number that decides"
              " whether W4A8 needs rotations/smoothing before it is worth a kernel.")


if __name__ == "__main__":
    main()
