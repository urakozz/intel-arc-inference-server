"""Write a SYNTHETIC capture file in probe_w4a8's format, for timing a shape no
capture path reaches (probe_w4a8 --capture only knows gate||up, --capture-down
only down).

    python mk_capture.py <K> <N> <out.bin> [--seed 0]

Activations: bf16 N(0, 1) [2048][K]. Weights: uniformly random int4 layout-0
words [K/8][N] and f16 scales U(0.005, 0.02) [K/64][N]. Values are random on
purpose: kernel TIMES do not depend on them, and probe_w8a8's bit-exact checks
hold for any data. Its accuracy rows (rel L2 against the control) mean nothing
on a synthetic file.
"""
import argparse
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("K", type=int)
ap.add_argument("N", type=int)
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
M, G = 2048, 64
rng = np.random.default_rng(a.seed)
act = rng.standard_normal((M, a.K), dtype=np.float32).view(np.uint32)
act = ((act + 0x7FFF + ((act >> 16) & 1)) >> 16).astype(np.uint16)
qw = rng.integers(0, 2**32, size=(a.K // 8, a.N), dtype=np.uint32)
sc = rng.uniform(0.005, 0.02, size=(a.K // G, a.N)).astype(np.float16)
with open(a.out, "wb") as f:
    f.write(b"B70W4A8\0" + struct.pack("<7I", 3, 0, M, a.K, a.N, G, 0))
    f.write(act.tobytes()); f.write(qw.tobytes()); f.write(sc.tobytes())
print(f"wrote {a.out}: synthetic K={a.K} N={a.N}")
