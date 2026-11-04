#!/usr/bin/env python3
"""Spec 28: the engine's plain row (probe_row_taps) against the CPU references (row_taps_ref.py) and the f32 judge.

    row_taps_compare.py <eng stem> <ref dir/label> <judge.golden.safetensors> <row> <id a> <id b> [--head bf16|int8]

Prints the two ids' logits and their gap in bf16 ulps (of the row's top logit) in the engine, transformers bf16,
transformers f32 and the judge; then, per layer, each tap's distance from the f32 tap (relative L2) for the engine
and for transformers bf16, and the projection of each tap onto the direction (W[a] - W[b]) after the final norm's
scaling - how much of the a-b logit gap each one carries. Numpy only (no torch): runs on the box host."""
import json
import math
import struct
import sys

import numpy as np

DT = {"F32": np.float32, "I32": np.int32, "BF16": np.uint16, "F16": np.float16, "I64": np.int64}


def st_load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            arr = np.frombuffer(f.read(b - a), dtype=DT[v["dtype"]]).reshape(v["shape"])
            if v["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[k] = arr.astype(np.float32) if arr.dtype != np.int32 else arr
        return out


def bf16_file(path, shape):
    a = np.fromfile(path, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def ulp(x):
    return 2.0 ** (math.frexp(abs(float(x)))[1] - 8)


def gap(row, a, b, VU=248077):
    top = float(row[:VU].max())
    return float(row[a]), float(row[b]), (float(row[a]) - float(row[b])) / ulp(top), int(row[:VU].argmax())


def main():
    eng, ref, judge, r, a, b = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    head = sys.argv[8] if len(sys.argv) > 8 and sys.argv[7] == "--head" else "bf16"
    rb, rf = st_load(ref + ".bf16.safetensors"), st_load(ref + ".f32.safetensors")
    L, H = rb["mid"].shape
    e_lg = np.fromfile(eng + ".logits.f32", dtype=np.float32)
    e_taps = bf16_file(eng + ".taps.bf16", (L, H))
    e_x = bf16_file(eng + ".x.bf16", (H,))
    e_pre = bf16_file(eng + ".resid.bf16", (H,))
    j = st_load(judge)["logits"][r]
    key = "logits_bf16head" if head == "bf16" else "logits_i8head"
    print(f"ids {a} vs {b} ({head} head): logit a, logit b, (a - b) in bf16 ulps of the top, argmax")
    for name, row in (("engine", e_lg), ("hf bf16", rb[key]), ("hf f32", rf[key]), ("f32 judge", j)):
        la, lb, g, t = gap(row, a, b)
        print(f"  {name:10s} {la:9.4f} {lb:9.4f} {g:+8.2f} ulp   top {t}")
    # The a-b direction in hidden space: logit_a - logit_b = x . (W[a] - W[b]); x = norm(pre).
    print("per layer: relL2 to f32 (engine, hf bf16) of the mid-layer tap and the a-b gap the tap carries"
          " if the rest of the net were the f32 one (not computed: the direction is only exact at the end)")
    for l in range(L):
        f = rf["mid"][l]
        nf = np.linalg.norm(f)
        de = np.linalg.norm(e_taps[l] - f) / nf
        db = np.linalg.norm(rb["mid"][l] - f) / nf
        deb = np.linalg.norm(e_taps[l] - rb["mid"][l]) / nf
        print(f"  L{l:02d} |f32| {nf:9.2f}  eng {de:.5f}  hf-bf16 {db:.5f}  eng-vs-bf16 {deb:.5f}  "
              f"max|.| eng {np.abs(e_taps[l]).max():8.2f} f32 {np.abs(f).max():8.2f}")
    for name, e, kb, kf in (("pre-norm", e_pre, rb["pre"], rf["pre"]), ("x (normed)", e_x, rb["x"], rf["x"])):
        nf = np.linalg.norm(kf)
        print(f"  {name:10s} eng {np.linalg.norm(e - kf) / nf:.5f}  hf-bf16 {np.linalg.norm(kb - kf) / nf:.5f}  "
              f"eng-vs-bf16 {np.linalg.norm(e - kb) / nf:.5f}")


if __name__ == "__main__":
    main()
