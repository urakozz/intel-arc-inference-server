#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `bf16` of tools/quantize/quantize_qwen4exp.sh: every tensor the export did not quantise is the source's.

    bf16_same.py <export dir> <source snapshot>

Every tensor of the export that is not int4 (`.qweight` / `.scales` / `.qzeros`) and that the source has under the
same name (all but the MTP head's split experts, which are int4) must equal the source tensor bit for bit: the
ignore list's linears, the routers, the hyper-connections, norms, the embedding, the vision tower and the 128
shards of the PLE table (which ple_int8.py then converts). On the box's XPU AutoRound writes them bf16 as it found
them; on a CPU without bf16 (DEVICE=cpu) it writes fp32 widenings - those are checked to be exactly the widened
source and restored to the source's bf16 bytes (the shard rewritten, config.json's dtype set back to bfloat16).
Anything else - a changed value, another dtype than the source's (the PLE's I64 constants included) - is
refused (exit 1). Streams one tensor at a
time (the real export: ~172 GB read beside the source's same bytes).
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INT4 = (".qweight", ".scales", ".qzeros")


def weight_map(d: str) -> dict:
    with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("source")
    a = ap.parse_args()
    src_map = weight_map(a.source)
    exp_map = weight_map(a.export)
    handles: dict = {}

    def src(name):
        fn = src_map[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.source, fn), "pt")
        return handles[fn].get_tensor(name)

    same = restored = 0
    errs = []
    for fn in sorted(set(exp_map.values())):
        path = os.path.join(a.export, fn)
        widened = {}
        with safe_open(path, "pt") as f:
            names = list(f.keys())
            for n in names:
                if n.endswith(INT4) or n not in src_map:
                    continue
                t, s = f.get_tensor(n), src(n)
                if t.dtype == s.dtype and torch.equal(t, s):
                    same += 1
                elif t.dtype == torch.float32 and s.dtype == torch.bfloat16 and torch.equal(t, s.float()):
                    widened[n] = s
                else:
                    errs.append(f"{fn}: {n} {t.dtype} {list(t.shape)} is not the source's {s.dtype} {list(s.shape)}")
        if widened and not errs:
            out = {}
            with safe_open(path, "pt") as f:
                meta = f.metadata()
                for n in names:
                    out[n] = widened[n] if n in widened else f.get_tensor(n)
            save_file(out, path + ".tmp", metadata=meta or {"format": "pt"})
            os.replace(path + ".tmp", path)
            restored += len(widened)
    if errs:
        for e in errs[:20]:
            print("REFUSED:", e)
        sys.exit(f"bf16_same: {len(errs)} tensor(s) differ from the source")
    if restored:
        cfg_path = os.path.join(a.export, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for c in (cfg, cfg.get("text_config", {})):
            if "dtype" in c:
                c["dtype"] = "bfloat16"
            c.pop("torch_dtype", None)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    print(f"bf16_same: {same + restored} unquantised tensors bitwise the source's ({restored} restored from fp32)")


if __name__ == "__main__":
    main()
