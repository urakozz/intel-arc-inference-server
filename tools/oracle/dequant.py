#!/usr/bin/env python3
"""The reference meaning of the int4 GPTQ bits, shared with the C++ loader.

dequant_gptq computes w = (q - 8) * scale with the product in fp32 and ONE
cast to bf16 - the same single-rounding path as common::f32_to_bf16 on the
C++ side, which is what makes the fixture bit-exact comparable.

Conventions (docs/02, docs/03): qweight int32 [K/8, N], nibble i of word
(r, n) is k = r*8 + i (low nibble first); scales f16 [K/64, N]; symmetric,
zero point 8 (the checkpoint's v1 qzeros store 7 == zero-1; dropped).

Usage: dequant.py --write-fixture tests/golden/dequant_fixture.safetensors
"""
import argparse

import torch
from safetensors.torch import save_file


def dequant_gptq(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    assert qweight.dtype == torch.int32 and scales.dtype == torch.float16
    rows, n = qweight.shape
    k = rows * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4                     # [8]
    nibbles = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF      # [K/8, 8, N]
    q = nibbles.reshape(k, n)                                           # k = r*8 + i
    w32 = (q.to(torch.float32) - 8.0) * scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    return w32.to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-fixture", required=True)
    args = ap.parse_args()
    torch.manual_seed(1234)
    K, N = 128, 32
    qweight = torch.randint(-(2**31), 2**31 - 1, (K // 8, N), dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(K // 64, N, dtype=torch.float32) * 0.06 + 0.02).to(torch.float16)
    fx = {"qweight": qweight, "scales": scales, "dequant": dequant_gptq(qweight, scales)}
    save_file(fx, args.write_fixture)
    print(f"wrote {args.write_fixture}: qweight {tuple(qweight.shape)}, "
          f"scales {tuple(scales.shape)}, dequant {tuple(fx['dequant'].shape)}")


if __name__ == "__main__":
    main()
