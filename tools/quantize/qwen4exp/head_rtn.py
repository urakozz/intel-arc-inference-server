#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `head` of tools/quantize/quantize_qwen4exp.sh: the output head packed by the engine's int4 g64 rule.

    head_rtn.py <export dir> <source snapshot>

AutoRound tunes the decoder blocks only; `lm_head` (outside every block) is quantised by plain RTN either way
("using rtn to quantize lm_head" in its log). At the pinned commit that path writes `lm_head.scales` as BF16,
which the loader refuses (scales are F16: src/loader/quant.cc, check_gptq_export.py) - proven on the tiny
model. So the head is re-packed here from the source's bf16 `lm_head.weight` by loader::rtn_int4_g64
(src/loader/rtn.h, spec 15e: symmetric, group 64, scale f16(2 amax / 15) rounded to nearest even, q =
clamp(rint(w / scale) + 8, 0, 15) with the stored scale, an all-zero group scale 0 and q 8; GPTQ v1 packing,
qzeros 0x77777777) and the export's `lm_head.qweight` / `scales` / `qzeros` replaced in their file. Nothing
is lost against AutoRound's own head: both are RTN, this one with the F16 scale the kernels read.
Idempotent (the same bytes on a rerun).
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QZ = 0x77777777


def rtn_int4_g64(w: torch.Tensor):
    """loader::rtn_int4_g64 on w [N][K] (bf16 values) -> qweight I32 [K/8][N], scales F16 [K/64][N],
    qzeros I32 [K/64][N/8]."""
    N, K = w.shape
    wf = w.float().reshape(N, K // 64, 64)
    s16 = (2.0 * wf.abs().amax(-1) / 15.0).to(torch.float16)            # [N][K/64], RNE, subnormals kept
    s = s16.float().unsqueeze(-1)
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, 1.0)) + 8.0, torch.full_like(wf, 8.0))
    q = q.clamp(0, 15).to(torch.int64).reshape(N, K).t().reshape(K // 8, 8, N)
    word = torch.zeros(K // 8, N, dtype=torch.int64)
    for j in range(8):
        word |= q[:, j, :] << (4 * j)
    qweight = (word - ((word >> 31) << 32)).to(torch.int32)
    qz = torch.full((K // 64, N // 8), QZ, dtype=torch.int64)
    qzeros = (qz - ((qz >> 31) << 32)).to(torch.int32)
    return qweight.contiguous(), s16.t().contiguous(), qzeros


def weight_map(d: str) -> dict:
    with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("source")
    a = ap.parse_args()
    with safe_open(os.path.join(a.source, weight_map(a.source)["lm_head.weight"]), "pt") as f:
        w = f.get_tensor("lm_head.weight")
    new = dict(zip(("lm_head.qweight", "lm_head.scales", "lm_head.qzeros"), rtn_int4_g64(w)))
    fn = weight_map(a.export)["lm_head.qweight"]
    path = os.path.join(a.export, fn)
    out = {}
    with safe_open(path, "pt") as f:
        meta = f.metadata()
        for n in f.keys():
            out[n] = new[n] if n in new else f.get_tensor(n)
    missing = set(new) - set(out)
    if missing:
        raise SystemExit(f"head_rtn: {fn} lacks {sorted(missing)} - not AutoRound's packed head")
    save_file(out, path + ".tmp", metadata=meta or {"format": "pt"})
    os.replace(path + ".tmp", path)
    print(f"head_rtn: lm_head [{w.shape[0]}][{w.shape[1]}] packed by loader::rtn_int4_g64 into {fn} (scales F16)")


if __name__ == "__main__":
    main()
