#!/usr/bin/env python3
"""The int8 KV simulation's fixture: rows of K / V values and what tools/oracle/int8_sim.py's kv8_encode /
kv8_dequant / kv8_fq make of them, so tests/common/kv8_sim_test.cc can hold the simulation (the f32 judges' --kv8:
tools/oracle/qwen_forced.py) bit for bit to src/common/kv8.h - which kv8_kernels_test holds bitwise to the writer
kernel (src/kernels/kv8.cl).

    kv8_sim_fixture.py [--out tests/loader/kv8_sim_fixture.safetensors]

Run in the oracle container (tools/oracle/run_in_container.sh). Deterministic (a seeded torch.Generator).

Rows (bf16 values, 256 a row): Gaussian rows over magnitudes 1e-9..3e3 (scales from 0 through fp16 subnormals to
normals), rows with a few outlier channels (the K rows' shape), a zero row, a one-hot row, a constant row.
Contents: x bf16 [R][256]; q int8 [R][256]; s16 int32 [R] (the fp16 scale's bits); deq f32 [R][256] (still rotated);
xhat f32 [R][256] (kv8_fq: back in the original basis).
"""
import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location("kv8_fixture_" + name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows():
    g = torch.Generator().manual_seed(12)
    out = []
    for mag in torch.logspace(-9, 3.5, 96).tolist():
        out.append(torch.randn(256, generator=g) * mag)
    for _ in range(24):
        r = torch.randn(256, generator=g)
        hot = torch.randint(0, 256, (3,), generator=g)
        r[hot] *= 40.0
        out.append(r * float(torch.empty(1).uniform_(0.05, 20.0, generator=g)))
    z = torch.zeros(256)
    one = torch.zeros(256)
    one[37] = 3.25
    out += [z, one, torch.full((256,), -0.75)]
    return torch.stack(out).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "..", "tests", "loader", "kv8_sim_fixture.safetensors"))
    a = ap.parse_args()
    from safetensors.torch import save_file
    I8 = _load("int8_sim")
    x = rows()
    q, s16 = I8.kv8_encode(x.float())
    deq = I8.kv8_dequant(q, s16)
    xhat = I8.kv8_fq(x.float())
    save_file({"x": x.contiguous(), "q": q.contiguous(), "s16": s16.contiguous(), "deq": deq.contiguous(),
               "xhat": xhat.contiguous()}, a.out, metadata={"rows": str(x.shape[0])})
    zero = int((s16 == 0).sum())
    sub = int(((s16 > 0) & (s16 < 0x400)).sum())
    print(f"{a.out}: {x.shape[0]} rows ({zero} zero scales, {sub} subnormal fp16 scales)")


if __name__ == "__main__":
    main()
