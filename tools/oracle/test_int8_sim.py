#!/usr/bin/env python3
"""tools/oracle/int8_sim.py against the engine's C++ rules, by fingerprint (FNV-1a 64 over the
codes and the scales' bytes). The fingerprints were printed by small programs compiled against
the repo (2026-10-23, clang++ -std=c++17 -O1 -ffp-contract=off -Isrc):
  * std::mt19937(K)'s sign bits (runtime/prefill/int8_signs.h): K, #negative, FNV of the bits;
  * loader/lm_head_int8.cc quantise_row_int8 over 64 LCG bf16 rows of 2560 (row 5 all zero);
  * pf_int8_test.cc's host reference of pf_quant_had (pinned bitwise to the card's kernel there)
    and the requant's per-column rule, over 8 LCG rows at K = 2048 and 4096.
Run in the oracle container: python3 tools/oracle/test_int8_sim.py"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tokenize.py shadows the stdlib

import importlib.util  # noqa: E402
import struct  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("int8_sim", os.path.join(_HERE, "int8_sim.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FNV0, FNVP, M64 = 1469598103934665603, 1099511628211, (1 << 64) - 1


def fnv(h, byte):
    return ((h ^ (byte & 0xFF)) * FNVP) & M64


def f32_bytes(v):
    return struct.pack("<f", float(v))


def lcg(seed, n):
    st, out = seed, []
    for _ in range(n):
        st = (st * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(st)
    return out


def bf16_bits_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


def f32_to_bf16_bits(f):
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16) & 0xFFFF


class Int8Sim(unittest.TestCase):
    def test_mt19937_reference(self):
        # the standard's own check value: the 10000th output of a default-seeded mt19937
        self.assertEqual(S.mt19937_u32(5489, 10000)[-1], 4123659995)

    def test_signs(self):
        want = {1024: (520, 3817297561231277475), 2048: (995, 4551716505372956806),
                2560: (1251, 7528334514084226188), 4096: (2053, 7093003887069929896)}
        for K, (neg, h_want) in want.items():
            d = S.int8_signs(K)
            h = FNV0
            for v in d.tolist():
                h = fnv(h, 1 if v < 0 else 0)
            self.assertEqual((int((d < 0).sum()), h), (neg, h_want), K)

    def test_lm_head_int8_rows(self):
        N, K = 64, 2560
        bits = []
        for st in lcg(12345, N * K):
            b = (st >> 16) & 0xFFFF
            if ((b >> 7) & 0xFF) == 0xFF:
                b &= 0xBFFF
            bits.append(b)
        for k in range(K):
            bits[5 * K + k] = 0
        w = torch.tensor([bf16_bits_to_f32(b) for b in bits], dtype=torch.float32).reshape(N, K)
        q, s = S.int8_rows(w.to(torch.bfloat16))
        h = FNV0
        for n in range(N):
            for c in f32_bytes(s[n]):
                h = fnv(h, c)
            for c in q[n].tolist():
                h = fnv(h, c)
        self.assertEqual(h, 8247283833515161150)

    def test_h8_codes(self):
        want = {2048: 7743004979853941744, 4096: 16114723854743546077}
        for K, h_want in want.items():
            rows = 8
            # the C++ program's expression, in fp32 step by step
            vals = []
            for st in lcg(777 + K, rows * K):
                f = struct.unpack("<f", struct.pack("<f", float(st >> 8)))[0]
                f = struct.unpack("<f", struct.pack("<f", f / 16777216.0))[0]
                f = struct.unpack("<f", struct.pack("<f", f - 0.5))[0]
                f = struct.unpack("<f", struct.pack("<f", f * 8.0))[0]
                vals.append(f32_to_bf16_bits(f))
            x = torch.tensor([bf16_bits_to_f32(b) for b in vals], dtype=torch.float32).reshape(rows, K)
            d = S.int8_signs(K)
            r = S._rot(x, d)
            s = r.abs().amax(-1, keepdim=True) / 127.0
            iv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
            q = S._sat_rte(r * iv).to(torch.int8)
            h = FNV0
            for m in range(rows):
                for c in f32_bytes(s[m, 0]):
                    h = fnv(h, c)
                for c in q[m].tolist():
                    h = fnv(h, c)
            self.assertEqual(h, h_want, K)

    def test_fake_quant_is_the_int8_product(self):
        torch.manual_seed(0)
        K, N, M = 2048, 96, 5
        x = torch.randn(M, K)
        w = torch.randn(N, K) * 0.02
        d = S.int8_signs(K)
        xr, wr = S._rot(x, d), S._rot(w, d)
        sx = xr.abs().amax(-1, keepdim=True) / 127.0
        sw = wr.abs().amax(-1, keepdim=True) / 127.0
        qx, qw = S._sat_rte(xr * (1.0 / sx)), S._sat_rte(wr * (1.0 / sw))
        want = (qx.double() @ qw.double().T) * sx.double() * sw.double().T
        got = S.fq_x(x).double() @ S.fq_w(w).double().T
        self.assertLess(float((got - want).abs().max() / want.abs().max()), 1e-5)
        # the rotation itself is orthonormal: unrot(rot(v)) == v
        self.assertLess(float((S._unrot(S._rot(x, d), d) - x).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
