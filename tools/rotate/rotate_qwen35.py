#!/usr/bin/env python3
"""Write a residual-rotated copy of a bf16 Qwen3.5 checkpoint (tools/rotate/rotation.py).

    rotate_qwen35.py <bf16 snapshot> <out dir>

Shard by shard, same shard names, same tensor names, same index: only the
tensors rotation.classify() names change, and each is computed in fp32 and
rounded to bf16 once. Every expected tensor must be transformed exactly once
or the run aborts. config.json gains a `b70_rotation` record, and the sign
vector is written to b70_rotation.safetensors (not in the index; engines and
the MTP note in rotation.py need it, loaders that follow the index ignore it).
The output is a plain Hugging Face checkpoint: quantise it with AutoRound as
the unrotated one was.

Needs torch and safetensors only; ~6 GB of RAM per shard in flight.
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotation as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    args = ap.parse_args()
    t0 = time.time()
    snap, out = args.snapshot.rstrip("/"), args.out.rstrip("/")
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"FATAL: {out} exists and is not empty")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    if tc.get("hidden_size") != R.HIDDEN or tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings")):
        sys.exit("FATAL: expected hidden_size 5120 and untied embeddings")
    if "quantization_config" in cfg:
        sys.exit("FATAL: this is a quantised checkpoint; rotate the bf16 one, then quantise")
    n_layers = tc["num_hidden_layers"]
    with open(os.path.join(snap, "model.safetensors.index.json"), encoding="utf-8") as f:
        index = json.load(f)
    wmap = index["weight_map"]

    # the ORIGINAL norm gains, needed to fold into the readers before they are zeroed
    norms = {}
    for k in R.norm_keys(n_layers):
        with safe_open(os.path.join(snap, wmap[k]), framework="pt", device="cpu") as h:
            norms[k] = h.get_tensor(k)
    d = R.signs()
    print(f"{n_layers} layers, {len(wmap)} tensors in {len(set(wmap.values()))} shards; "
          f"{len(norms)} norm gains preloaded; seed {R.SEED}", flush=True)

    done = {}
    for shard in sorted(set(wmap.values())):
        ts = time.time()
        outd = {}
        with safe_open(os.path.join(snap, shard), framework="pt", device="cpu") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                kind, nt = R.transform(k, t, norms, d)
                if kind != "copy":
                    if not torch.isfinite(nt).all():
                        sys.exit(f"FATAL: {k} not finite after {kind}")
                    nt = nt.to(t.dtype)
                    done[k] = kind
                outd[k] = nt.contiguous()
        save_file(outd, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"  {shard}: {len(outd)} tensors, "
              f"{sum(1 for k in outd if k in done)} rotated, {time.time() - ts:.0f}s", flush=True)
        del outd

    # every tensor the rotation must touch was touched, and nothing else
    want = {k for k in wmap if R.classify(k)[0] != "copy"}
    if set(done) != want:
        sys.exit(f"FATAL: transformed {len(done)} tensors, expected {len(want)}; "
                 f"missing {sorted(want - set(done))[:5]}")
    by_kind = {}
    for kind in done.values():
        by_kind[kind] = by_kind.get(kind, 0) + 1
    # 64 layers: norm 2*64 + final = 129, read 48*6 + 16*5 = 368, write 2*64 = 128
    print(f"transformed: {by_kind}")

    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        if name.endswith(".safetensors") or name in ("config.json", "crc32.txt") or os.path.isdir(src):
            continue
        shutil.copy(src, os.path.join(out, name))
    cfg["b70_rotation"] = {
        "scheme": "residual stream R = D * blockdiag(H_1024) / 32, folded; see tools/rotate/rotation.py",
        "block": R.BLOCK, "hidden": R.HIDDEN, "seed": R.SEED,
        "signs_file": "b70_rotation.safetensors",
        "rotated": "embed_tokens, lm_head (final norm folded), every decoder layer's norms "
                   "(folded, zeroed), stream readers and writers",
        "not_rotated": "down_proj input (runtime op), mtp.* (inconsistent with the folded "
                       "lm_head; see rotation.py), visual",
        "source_snapshot": os.path.basename(snap),
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_file({"r1_signs": d}, os.path.join(out, "b70_rotation.safetensors"))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
