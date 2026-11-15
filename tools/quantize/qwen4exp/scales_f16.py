#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `scales` of tools/quantize/quantize_qwen4exp.sh: every int4 group scale in f16, as the engine reads them.

    scales_f16.py <export dir>

At ITERS=0 AutoRound's RTN path writes the scales of the non-expert linears in bf16 whatever `scale_dtype` says
(the fused experts' come out f16). bf16 has fewer mantissa bits than f16, so a bf16 scale in f16's normal range
(6.1e-5 .. 65504) or zero converts exactly; a scale outside it is refused (exit 1) - nothing is rounded. Only the
shards holding a bf16 `.scales` tensor are rewritten (tmp file, then rename). A tuned export (all f16) is left as
it is.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

F16_MIN_NORMAL = 2.0 ** -14
F16_MAX = 65504.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    a = ap.parse_args()
    with open(os.path.join(a.export, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    converted = 0
    errs = []
    for fn in shards:
        path = os.path.join(a.export, fn)
        with safe_open(path, "pt") as f:
            bf = [n for n in f.keys() if n.endswith(".scales") and f.get_slice(n).get_dtype() == "BF16"]
        if not bf:
            continue
        out = {}
        with safe_open(path, "pt") as f:
            meta = f.metadata()
            for n in f.keys():
                t = f.get_tensor(n)
                if n in bf:
                    m = t.float().abs()
                    bad = ~torch.isfinite(m) | ((m != 0) & ((m < F16_MIN_NORMAL) | (m > F16_MAX)))
                    if bool(bad.any()):
                        errs.append(f"{fn}: {n} has {int(bad.sum())} scale(s) outside f16's normal range")
                        continue
                    t = t.to(torch.float16)
                    converted += 1
                out[n] = t
        if errs:
            break
        save_file(out, path + ".tmp", metadata=meta or {"format": "pt"})
        os.replace(path + ".tmp", path)
    if errs:
        for e in errs[:20]:
            print("REFUSED:", e)
        sys.exit("scales_f16: a bf16 scale has no exact f16 value")
    print(f"scales_f16: {converted} bf16 scale tensor(s) converted to f16 exactly")


if __name__ == "__main__":
    main()
