#!/usr/bin/env python3
"""Spec 28: tools/probe/probe_verify_rows.cc's arms (row 0's logits and the drafter taps after layers 5 / 19 / 33 / 47 /
61, on the plain and the DFlash history) and probe_row_taps.cc's plain row against the f32 reference of the same
context row (row_taps_ref.py --dtype f32: `out` = the residual after layer L, `mid` = after its mixer). Numpy only.

    verify_rows_compare.py <probe_verify_rows --out stem> <probe_row_taps stem> <ref .f32.safetensors> <G> <a> <b>

Prints the f32 gap a - b (bf16 ulps of the row's top), then per (history, arm) the gap, the argmax and the relative
L2 distance of each tap to f32; then the plain row's mid-layer taps' distance to f32 every 4 layers."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from row_taps_compare import st_load, bf16_file  # noqa: E402

vr, plain, ref, G, A, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
taps = [5, 19, 33, 47, 61]
R = st_load(ref)
L, H = R["out"].shape
VU = 248077


def ulp(x):
    return 2.0 ** (math.frexp(abs(float(x)))[1] - 8)


def gap(row):
    top = float(row[:VU].max())
    return (float(row[A]) - float(row[B])) / ulp(top), int(row[:VU].argmax())


def rel(x, f):
    return float(np.linalg.norm(x - f) / np.linalg.norm(f))


g, t = gap(R["logits_i8head"])
# which f32 index the drafter taps are: out[L] or out[L - 1] - the nearer for the plain m1 arm
m1 = bf16_file(f"{vr}.{G}.plain.m1.taps.bf16", (5, H))
d_out = [rel(m1[i], R["out"][l]) for i, l in enumerate(taps)]
d_prev = [rel(m1[i], R["out"][l - 1]) for i, l in enumerate(taps)]
idx = [l if a <= b else l - 1 for l, a, b in zip(taps, d_out, d_prev)]
print(f"G {G}: f32 {A} - {B} {g:+.2f} ulp (top {t}); drafter taps = f32 out[{idx}]")
print(f"  {'state':6s} {'arm':10s} {'gap':>8s} {'top':>7s}   rel L2 to f32 at the taps " + " ".join(f"L{l:<7d}" for l in idx))
for st in ("plain", "dflash"):
    for arm in ("m1", "rule", "v2", "v2_gemv", "dpas_gemv"):
        s = f"{vr}.{G}.{st}.{arm}"
        if not os.path.exists(s + ".logits.f32"):
            continue
        lg = np.fromfile(s + ".logits.f32", dtype=np.float32)
        tp = bf16_file(s + ".taps.bf16", (5, H))
        gg, tt = gap(lg)
        print(f"  {st:6s} {arm:10s} {gg:+8.2f} {tt:7d}   " + " ".join(f"{rel(tp[i], R['out'][l]):.5f} " for i, l in enumerate(idx)))
if os.path.exists(plain + ".taps.bf16"):
    mid = bf16_file(plain + ".taps.bf16", (L, H))
    lg = np.fromfile(plain + ".logits.f32", dtype=np.float32)
    gg, tt = gap(lg)
    print(f"  probe_row_taps plain row (no drafter): {gg:+.2f} ulp top {tt}; rel L2 of its mid-layer taps to f32 mid:")
    print("   " + " ".join(f"L{l}:{rel(mid[l], R['mid'][l]):.4f}" for l in range(0, L, 4)) + f" L63:{rel(mid[63], R['mid'][63]):.4f}")
