#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `head` of tools/quantize/quantize_qwen4exp.sh: the output head packed by the engine's int4 g64 rule.

    head_rtn.py <export dir>

AutoRound runs without `quant_lm_head` (autoround_run.py says why: a quantised layer outside the blocks turns its
low-CPU-memory mode and the per-block resume off), so the export ships the bf16 `lm_head.weight` (the `bf16`
stage checked it bitwise against the source). This packs it by loader::rtn_int4_g64 (src/loader/rtn.h, spec 15e:
symmetric, group 64, scale f16(2 amax / 15) rounded to nearest even, q = clamp(rint(w / scale) + 8, 0, 15) with
the stored scale, an all-zero group scale 0 and q 8; GPTQ v1 packing, qzeros 0x77777777) into `lm_head.qweight`
/ `scales` / `qzeros` in the same file, updates the index and declares the head 4-bit in `quantization_config`
(config.json and quantization_config.json: `extra_config["lm_head"]`, as AutoRound's `quant_lm_head` does).
AutoRound's own head would be RTN too ("using rtn to quantize lm_head", and at the pinned commit with BF16
scales the loader refuses); nothing tuned is lost. A rerun on a packed head does nothing.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QZ = 0x77777777
HEAD = {"bits": 4, "group_size": 64, "sym": True, "data_type": "int"}


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


def update_json(path: str, fn) -> None:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    fn(d)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(path + ".tmp", path)


def declare(qc: dict) -> None:
    qc.setdefault("extra_config", {})["lm_head"] = dict(HEAD)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    a = ap.parse_args()
    idx_path = os.path.join(a.export, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    if "lm_head.weight" not in wm:
        if "lm_head.qweight" in wm:
            print("head_rtn: lm_head is already packed - nothing to do")
            return
        raise SystemExit(f"head_rtn: {a.export} has no lm_head.weight")
    fn = wm["lm_head.weight"]
    path = os.path.join(a.export, fn)
    out = {}
    with safe_open(path, "pt") as f:
        meta = f.metadata()
        for n in f.keys():
            out[n] = f.get_tensor(n)
    w = out.pop("lm_head.weight")
    out.update(zip(("lm_head.qweight", "lm_head.scales", "lm_head.qzeros"), rtn_int4_g64(w)))
    save_file(out, path + ".tmp", metadata=meta or {"format": "pt"})
    os.replace(path + ".tmp", path)

    def index(d):
        m = d["weight_map"]
        del m["lm_head.weight"]
        for n in ("lm_head.qweight", "lm_head.scales", "lm_head.qzeros"):
            m[n] = fn
        d["weight_map"] = dict(sorted(m.items()))
    update_json(idx_path, index)
    update_json(os.path.join(a.export, "config.json"), lambda c: declare(c["quantization_config"]))
    qcj = os.path.join(a.export, "quantization_config.json")
    if os.path.exists(qcj):
        update_json(qcj, declare)
    print(f"head_rtn: lm_head [{w.shape[0]}][{w.shape[1]}] packed by loader::rtn_int4_g64 into {fn}, declared 4-bit")


if __name__ == "__main__":
    main()
