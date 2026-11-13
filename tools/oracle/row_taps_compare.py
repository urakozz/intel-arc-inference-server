#!/usr/bin/env python3
"""Spec 28: the engine's plain row (probe_row_taps) against the CPU references (row_taps_ref.py) and the f32 judge.

    row_taps_compare.py <eng stem> <ref dir/label> <judge.golden.safetensors> <row> <id a> <id b> [--head bf16|int8]
                        [--base TAG]

Prints the two ids' logits and their gap in bf16 ulps (of the row's top logit) in the engine, each reference present
(<ref>.bf16 / .bf16x / .f32 / .f32x / .f32p / .f32xp.safetensors: row_taps_ref.py --dtype, x = --exact-dequant, p =
--pfi8) and the judge; then, per layer, the relative L2 distance of the engine's tap and of each reference's from the
f32 reference with the engine's own weights (f32x; f32 when absent; --base names another, e.g. f32p for an l0-int8
engine row). Numpy only (no torch): runs on the box host."""
import json
import math
import os
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
    opt = dict(zip(sys.argv[7::2], sys.argv[8::2]))
    head = opt.get("--head", "bf16")
    refs = {t: st_load(f"{ref}.{t}.safetensors") for t in ("bf16", "bf16x", "f32", "f32x", "f32p", "f32xp")
            if os.path.exists(f"{ref}.{t}.safetensors")}
    L, H = next(iter(refs.values()))["mid"].shape
    e = {"logits": np.fromfile(eng + ".logits.f32", dtype=np.float32), "mid": bf16_file(eng + ".taps.bf16", (L, H)),
         "x": bf16_file(eng + ".x.bf16", (H,)), "pre": bf16_file(eng + ".resid.bf16", (H,))}
    j = st_load(judge)["logits"][r]
    key = "logits_bf16head" if head == "bf16" else "logits_i8head"
    print(f"ids {a} vs {b} ({head} head): logit a, logit b, (a - b) in bf16 ulps of the top, argmax")
    rows = [("engine", e["logits"])] + [(f"hf {t}", refs[t][key]) for t in refs] + [("f32 judge", j)]
    for name, row in rows:
        la, lb, g, t = gap(row, a, b)
        print(f"  {name:10s} {la:9.4f} {lb:9.4f} {g:+8.2f} ulp   top {t}")
    base = opt.get("--base", "f32x" if "f32x" in refs else "f32")
    others = [("eng", e)] + [(t, refs[t]) for t in refs if t != base]
    print(f"relative L2 distance to hf {base} (the mid-layer tap = the residual after layer L's mixer):")
    print("  layer   " + "  ".join(f"{n:>8s}" for n, _ in others))

    def line(name, f, get):
        nf = np.linalg.norm(f)
        print(f"  {name:7s} " + "  ".join(f"{np.linalg.norm(get(o) - f) / nf:8.5f}" for _, o in others))
    for l in range(L):
        line(f"L{l:02d}", refs[base]["mid"][l], lambda o, l=l: o["mid"][l])
    line("pre", refs[base]["pre"], lambda o: o["pre"])
    line("x", refs[base]["x"], lambda o: o["x"])


if __name__ == "__main__":
    main()
