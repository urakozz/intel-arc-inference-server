#!/usr/bin/env python3
"""Checks for kv_int8_probe.py (spec 12a, Task 1 Step 1). CPU, seconds; no weights.

    python3 tools/oracle/test_kv_int8_probe.py

1. per-token int8 round trip: every element within half a step (s / 2) of the input,
   s = amax / 127 per (head, position); zero rows survive; fp16 scales likewise;
2. the Hadamard rotation (256, random signs) is orthogonal and preserves q.k in fp64
   to 1e-6 (Review Focus 2);
3. KIVI-style K: per channel over groups of 64 positions, the last partial group left
   in bf16 untouched (Review Focus 3), and the decode tail start floor((p+1)/64)*64;
4. the replay's fp64 attention with the identity "scheme" equals a plain softmax
   attention, and its KIVI selection equals building the per-query K by hand;
5. the end-to-end attention: its `bf16` variant is BITWISE transformers'
   eager_attention_forward (so the batch's reference element is the oracle's math),
   and its `kivi` variant at query p equals the eager math over the K that a decoder
   holding p + 1 rows would have. Skipped, named, without transformers.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("kv_int8_probe", os.path.join(_HERE, "kv_int8_probe.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def kv_like(hkv: int, t: int, d: int, seed: int) -> torch.Tensor:
    """bf16 K-like rows: a few outlier channels with offsets, as real K has."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(hkv, t, d, generator=g)
    x[..., 3] = x[..., 3] * 20 + 15
    x[..., 70] = x[..., 70] * 8 - 30
    return x.to(torch.bfloat16)


def test_per_token() -> None:
    x = kv_like(4, 37, 256, 1).float()
    x[1, 5] = 0.0
    for sd in (torch.float32, torch.float16):
        y = P.quant_per_token(x, scale_dtype=sd)
        s = (x.abs().amax(-1, keepdim=True) / 127).to(sd).float()
        bound = torch.where(s > 0, s / 2, torch.zeros_like(s)) * (1 + 1e-6)
        ok = bool(((y - x).abs() <= bound + 1e-30).all()) and bool((y[1, 5] == 0).all())
        check(f"per-token round trip within s/2 ({sd})", ok,
              f"max err/s {float(((y - x).abs() / s.clamp_min(1e-30)).max()):.4f}")


def test_hadamard() -> None:
    r = P.hadamard(256, seed=0)
    eye = torch.eye(256, dtype=torch.float64)
    check("hadamard orthogonal", float((r @ r.T - eye).abs().max()) < 1e-12)
    check("hadamard has random signs (not plain Sylvester)",
          not torch.equal(r[0].sign(), torch.ones(256, dtype=torch.float64)))
    q = kv_like(24, 9, 256, 2).double()
    k = kv_like(24, 9, 256, 3).double()
    a = (q * k).sum(-1)
    b = ((q @ r) * (k @ r)).sum(-1)
    rel = float(((a - b).abs() / (q.norm(dim=-1) * k.norm(dim=-1))).max())
    check("rotation preserves q.k in fp64 (Review Focus 2)", rel < 1e-6, f"max rel {rel:.2e}")


def test_kivi() -> None:
    x = kv_like(4, 150, 256, 4).float()
    y = P.quant_kivi_k(x, group=64)
    check("kivi: partial last group untouched", torch.equal(y[:, 128:], x[:, 128:]))
    worst = -1.0
    for g0 in (0, 64):
        blk = x[:, g0:g0 + 64]
        s = blk.abs().amax(1, keepdim=True) / 127
        over = ((y[:, g0:g0 + 64] - blk).abs() - s / 2) / blk.abs().clamp_min(1e-30)
        worst = max(worst, float(over.max()))
    # fp32's own rounding of x / s and q * s: within a few ulp of |x| beyond s / 2
    check("kivi: full groups within s/2, scale per channel per group", worst <= 2 ** -21,
          f"excess over s/2 at most {worst:.1e} |x|")
    check("kivi: groups differ from per-token", not torch.equal(y[:, :128], P.quant_per_token(x)[:, :128]))
    check("kivi decode tail start", [P.kivi_tail_start(p) for p in (0, 62, 63, 64, 127, 150)]
          == [0, 0, 64, 64, 128, 128])


def test_replay_attention() -> None:
    h, hkv, t, d = 8, 2, 130, 256
    q = kv_like(h, t, d, 5).double()
    k = kv_like(hkv, t, d, 6).double()
    v = kv_like(hkv, t, d, 7).double()
    pos = torch.tensor([0, 63, 64, 100, 129])
    lens = pos + 1
    out = P.attend64(q[:, pos], k, v, lens, 1 / 16)
    want = torch.empty_like(out)
    for i, p in enumerate(pos.tolist()):
        for hh in range(h):
            s = (q[hh, p] @ k[hh // (h // hkv), :p + 1].T) / 16
            want[hh, i] = torch.softmax(s, -1) @ v[hh // (h // hkv), :p + 1]
    check("replay fp64 attention == plain softmax attention", float((out - want).abs().max()) < 1e-12)
    kq = P.quant_kivi_k(k.float()).double()
    got = P.attend64(q[:, pos], kq, v, lens, 1 / 16, k_tail=k)
    ok = True
    for i, p in enumerate(pos.tolist()):
        ts = P.kivi_tail_start(p)
        kp = torch.cat([kq[:, :ts], k[:, ts:p + 1]], 1)
        one = P.attend64(q[:, pos[i:i + 1]], kp, v[:, :p + 1], lens[i:i + 1], 1 / 16)
        ok &= float((one[:, 0] - got[:, i]).abs().max()) < 1e-12
    check("replay kivi tail selection == per-query hand-built K", ok)


def test_e2e_attention() -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  e2e attention vs eager (no transformers Qwen3_5: {type(e).__name__})")
        return
    h, hkv, t, d = 8, 2, 140, 256

    class Mod(torch.nn.Module):
        num_key_value_groups = h // hkv
        layer_idx = 3

    mod = Mod().eval()
    q = kv_like(h, t, d, 8).unsqueeze(0)
    k = kv_like(hkv, t, d, 9).unsqueeze(0)
    v = kv_like(hkv, t, d, 10).unsqueeze(0)
    mask = torch.full((t, t), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16).triu(1)[None, None]
    want, _ = m.eager_attention_forward(mod, q, k, v, mask, 1 / 16)
    got = P.attend_variant("bf16", q[0], k[0], v[0], 0, 1 / 16, chunk=48)
    check("e2e bf16 variant BITWISE eager_attention_forward", torch.equal(got, want[0]))
    got = P.attend_variant("kivi", q[0], k[0], v[0], 0, 1 / 16, chunk=48)
    ok = True
    for p in (0, 63, 64, 100, 139):
        ts = P.kivi_tail_start(p)
        kk = k[0, :, :p + 1].float()
        kd = torch.cat([P.quant_kivi_k(k[0].float())[:, :ts], kk[:, ts:]], 1).to(torch.bfloat16)
        vd = P.quant_per_token(v[0, :, :p + 1].float()).to(torch.bfloat16)
        one, _ = m.eager_attention_forward(mod, q[:, :, p:p + 1], kd[None], vd[None], None, 1 / 16)
        ok &= torch.equal(one[0, 0], got[p])
    check("e2e kivi at query p == eager over a (p+1)-row decoder's K", ok)
    batch = P.make_e2e_attention(("bf16", "pt", "kivi", "rot", "f32attn"))
    qb, kb, vb = (x.expand(5, -1, -1, -1).contiguous() for x in (q, k, v))
    out, _ = batch(mod, qb, kb, vb, mask, 1 / 16)
    check("e2e batch: element 0 is the bf16 variant", torch.equal(out[0], want[0]))
    check("e2e batch: int8 elements differ from bf16", all(not torch.equal(out[i], out[0]) for i in (1, 2, 3)))
    rk = P.attend_variant("rotkv", q[0], k[0], v[0], 0, 1 / 16).float()
    ro = P.attend_variant("rot", q[0], k[0], v[0], 0, 1 / 16).float()
    cos = float(torch.nn.functional.cosine_similarity(rk.flatten(1), want[0].float().flatten(1), dim=-1).min())
    # a missing un-rotation gives cos ~0; the synthetic outliers make even f32attn 0.9947 here
    check("e2e rotkv: V un-rotated (close to bf16, differs from rot)", cos > 0.98 and not torch.equal(rk, ro),
          f"row cos min {cos:.6f}")


if __name__ == "__main__":
    test_per_token()
    test_hadamard()
    test_kivi()
    test_replay_attention()
    test_e2e_attention()
    print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
    sys.exit(1 if FAILS else 0)
