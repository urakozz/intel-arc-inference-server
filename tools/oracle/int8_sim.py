"""Quantisation-aware pieces for the f32 judges (operator ruling 2026-10-23): f32 math that applies
the int8 roundings an engine variant applies, so a judged miss is a kernel error and not the
variant's quantisation. Both are the engine's rules, op for op where an op decides a code:

* `int8_rows(w)` - the int8 output head (`--lm-head int8`, spec 9): loader/lm_head_int8.cc's
  quantise_row_int8 per vocab row - s = max|w| / 127 in fp32 (0 -> the row is zeros),
  q = clamp(nearbyint(w / s), -127, 127) with an fp32 division, round half to even. The head is
  then (x . q) s (gemv_i8w.cl). Returns (q int8 [N][K], s fp32 [N]).

* `fq_x(x)` / `fq_w(w)` - spec 5's h8 prefill linears (`--prefill-backend l0-int8`,
  src/kernels/prefill/pf_int8.cl): R_K = D_K blockdiag(H_1024) / 32, D_K the mt19937(K) signs of
  runtime/prefill/int8_signs.h (bit 1 of each 32-bit output -> -1), H the natural-order Sylvester
  Hadamard. Activations: x R per row, s_x = max|x R| / 127, q = sat_rte((x R) * (1 / s_x)).
  Weights (per output column n over K): R^T w, s_w = max|R^T w| / 127 (1 for an all-zero column),
  q = sat_rte((R^T w) * (1 / s_w)). The int8 GEMM is (q_x . q_w) s_x s_w; here both sides are
  returned DE-quantised and rotated back (x_eff = (q_x s_x) R^T, w_eff = R (q_w s_w)), so
  x_eff . w_eff = (q_x s_x) . (q_w s_w) - the same product, in f32 - and a model's own linear
  computes it. Not bitwise the device: the FWHT's fp32 sum order and the weights' values (the
  reference's bf16-rounded dequant, the device's fp32 (q - 8) x scale) can move a code that sits
  on a rounding boundary.

* `kv8_encode(x)` / `kv8_dequant(q, s16)` / `kv8_fq(x)` - Qwen3.8's int8 KV cache (`--kv-cache int8`,
  spec 12's rotkv: src/common/kv8.h, which src/kernels/kv8.cl repeats bitwise): per (position, kv
  head) row of 256, y = s (.) FWHT(x) / 16 (fp32, stages h = 1..128 ascending), s16 = f16_rne(amax /
  127), q = clamp(rint(y / f16f(s16)), +-127) (0 for a zero scale), deq = q f16f(s16). Bitwise kv8.h:
  every op is one fp32 rounding in kv8.h's order (tests/common/kv8_sim_test.cc holds the codes, the
  scales and deq to kv8.h on tools/oracle/kv8_sim_fixture.py's rows). kv8_fq is what the attention
  reads, back in the original basis: the row rounded to bf16 first (the cache stores what the bf16
  cache would), encoded, de-quantised, un-rotated - in fp64, where the un-rotation of one row's deq
  (a common scale times integer sums) is exact - and rounded once to fp32. q . kv8_fq(k) is the
  engine's (q R) . deq up to fp32 rounding (R R^T = I).
"""
import os

import torch

_H = {}
_SIGNS = {}


def mt19937_u32(seed: int, n: int) -> list:
    """std::mt19937(seed)'s first n outputs (the reference algorithm, 32-bit)."""
    mt = [0] * 624
    mt[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
    out, idx = [], 624
    while len(out) < n:
        if idx >= 624:
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            idx = 0
        y = mt[idx]
        idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        out.append(y & 0xFFFFFFFF)
    return out


def int8_signs(K: int) -> torch.Tensor:
    """runtime/prefill/int8_signs.h: one mt19937 seeded with K, (rng() & 1) -> -1, else +1."""
    if K not in _SIGNS:
        _SIGNS[K] = torch.tensor([-1.0 if (v & 1) else 1.0 for v in mt19937_u32(K, K)], dtype=torch.float32)
    return _SIGNS[K]


def hadamard(n: int = 1024) -> torch.Tensor:
    """The natural-order Sylvester Hadamard (H[i][j] = (-1)^popcount(i & j)), unnormalised."""
    if n not in _H:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[n] = h
    return _H[n]


def fwht_blocks(v: torch.Tensor) -> torch.Tensor:
    """tests/prefill/pf_int8_test.cc's fwht_blocks, vectorised: the canonical Sylvester FWHT per
    1024-block, stages h = 1, 2, ..., 512 ascending, (a + c, a - c), unnormalised, fp32 - the
    same op per element in the same order, so bitwise that host reference (which pf_int8_test
    holds bitwise to pf_quant_had / pf_requant_rot on the card)."""
    shp = v.shape
    K = shp[-1]
    r = v.reshape(-1, K // 1024, 1024)
    h = 1
    while h < 1024:
        r = r.reshape(r.shape[0], r.shape[1], 1024 // (2 * h), 2, h)
        a, c = r[..., 0, :], r[..., 1, :]
        r = torch.stack([a + c, a - c], dim=-2).reshape(r.shape[0], r.shape[1], 1024)
        h *= 2
    return r.reshape(shp)


def _rot(v: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """v R along the last dim (R = D blockdiag(H) / 32): per 1024 block, H (D v) / 32, the
    kernels' order (sign, FWHT, x 1/32)."""
    return fwht_blocks(v * d) * (1.0 / 32.0)


def _unrot(v: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The inverse: D (H v) / 32 per block (H H = 1024 I, D D = I)."""
    K = v.shape[-1]
    h = hadamard(1024)
    return (v.reshape(*v.shape[:-1], K // 1024, 1024) @ h).reshape(v.shape) * (1.0 / 32.0) * d


def _sat_rte(v: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(v), -128.0, 127.0)   # convert_char_sat_rte


# Design knobs for the l0-int8 accuracy study (r13.prefill, operator 2026-10-31), the default the engine's scheme:
#   B70_SIM_I8_X  tok (h8: rotated, one scale per token) | blk (rotated, one scale per token and 1024-k block) |
#                 g64 (unrotated, one scale per token and 64-k group) | two (rotated, per token, a second int8 limb
#                 for the residual) | off (exact)
#   B70_SIM_I8_W  col (h8: rotated, one scale per column) | blk (rotated, one scale per column and 1024-k block) |
#                 off (the int4 weight exact, as an int4-on-int8 GEMM)
#   B70_SIM_I8_X_MOE / B70_SIM_I8_W_MOE  the same for the MoE block's gate||up (default: B70_SIM_I8_X / _W)
SIM_X = os.environ.get("B70_SIM_I8_X", "tok")
SIM_W = os.environ.get("B70_SIM_I8_W", "col")
SIM_X_MOE = os.environ.get("B70_SIM_I8_X_MOE", SIM_X)
SIM_W_MOE = os.environ.get("B70_SIM_I8_W_MOE", SIM_W)


def _q_rows(v: torch.Tensor) -> torch.Tensor:
    """Per-row (last dim) symmetric int8 of v, de-quantised."""
    s = v.abs().amax(-1, keepdim=True) / 127.0
    iv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
    return _sat_rte(v * iv) * s


def fq_x(x: torch.Tensor, moe: bool = False) -> torch.Tensor:
    """An activation row block [..., K] as the h8 int8 GEMM sees it, de-quantised, original basis."""
    K = x.shape[-1]
    assert K % 1024 == 0, f"K = {K} is not whole 1024-k rotation blocks"
    mode = SIM_X_MOE if moe else SIM_X
    if mode == "off":
        return x
    if mode == "g64":
        v = x.float().reshape(*x.shape[:-1], K // 64, 64)
        return _q_rows(v).reshape(x.shape).to(x.dtype)
    d = int8_signs(K)
    xr = _rot(x.float(), d)
    if mode == "blk":
        q = _q_rows(xr.reshape(*xr.shape[:-1], K // 1024, 1024)).reshape(xr.shape)
    elif mode == "two":
        q1 = _q_rows(xr)
        q = q1 + _q_rows(xr - q1)
    else:
        q = _q_rows(xr)
    return _unrot(q, d).to(x.dtype)


def fq_w(w: torch.Tensor, moe: bool = False) -> torch.Tensor:
    """A weight [N][K] (torch Linear layout: row n = output column n over K), as pf_requant_rot's
    per-column int8 of R^T w with pf_colmax_rot's scales, de-quantised, original basis."""
    K = w.shape[-1]
    assert K % 1024 == 0, f"K = {K} is not whole 1024-k rotation blocks"
    wmode = SIM_W_MOE if moe else SIM_W
    if wmode == "off":
        return w
    d = int8_signs(K)
    wr = _rot(w.float(), d)
    if wmode == "blk":   # one scale per column and 1024-k block
        return _unrot(_q_rows(wr.reshape(*wr.shape[:-1], K // 1024, 1024)).reshape(wr.shape), d).to(w.dtype)
    m = wr.abs().amax(-1, keepdim=True)
    ws = torch.where(m > 0, m / 127.0, torch.ones_like(m))
    q = _sat_rte(wr * (1.0 / ws))
    return _unrot(q * ws, d).to(w.dtype)


def int8_rows(w: torch.Tensor):
    """loader/lm_head_int8.cc quantise_row_int8 over each row of w [N][K] (bf16 values)."""
    w32 = w.float()
    s = w32.abs().amax(-1) / 127.0
    safe = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(w32 / safe[:, None]), -127.0, 127.0)
    q = torch.where(s[:, None] == 0, torch.zeros_like(q), q)
    return q.to(torch.int8), s


def int8_head_weight(w: torch.Tensor) -> torch.Tensor:
    """The int8 head as an f32 weight, q x s per row (gemv_i8w's (x . q) s up to one f32 rounding)."""
    q, s = int8_rows(w)
    return q.float() * s[:, None]


# kv8.h kSignWords: bit j of word j / 32 set when s[j] = -1 (the 12a probe's hadamard(256, 0)).
KV8_SIGN_WORDS = (0xA197D809, 0xA6850592, 0xB205A73E, 0x20AFA769, 0x45C2CF64, 0xBEAEAACB, 0x75F39ADD, 0x2385A70C)
KV8_HD = 256


def kv8_signs() -> torch.Tensor:
    """s[j] = +-1, fp32 [256]."""
    return torch.tensor([-1.0 if (KV8_SIGN_WORDS[j >> 5] >> (j & 31)) & 1 else 1.0 for j in range(KV8_HD)])


def kv8_fwht(v: torch.Tensor) -> torch.Tensor:
    """kv8.h fwht256 over the last dim in v's dtype: stages h = 1, 2, ..., 128 ascending, the pair (i, i + h) with
    i & h == 0 becoming (a + c, a - c) - one rounding per op, kv8.h's order."""
    shp = v.shape
    r = v.reshape(-1, KV8_HD)
    h = 1
    while h < KV8_HD:
        r = r.reshape(r.shape[0], KV8_HD // (2 * h), 2, h)
        a, c = r[..., 0, :], r[..., 1, :]
        r = torch.stack([a + c, a - c], dim=-2).reshape(-1, KV8_HD)
        h *= 2
    return r.reshape(shp)


def kv8_f16_rne(f: torch.Tensor) -> torch.Tensor:
    """kv8.h f32_to_f16_rne on a finite, non-negative fp32 tensor (a scale): the binary16 bits as int32,
    subnormals kept, >= 65504 saturating - op for op."""
    u = f.float().contiguous().view(torch.int32).to(torch.int64) & 0x7FFFFFFF
    sub = torch.round(f.float() * 16777216.0).to(torch.int64)          # x 2^24, exact; rint
    e = (u >> 23) - 112
    m = u & 0x7FFFFF
    h = (e << 10) | (m >> 13)
    rem = m & 0x1FFF
    h = h + ((rem > 0x1000) | ((rem == 0x1000) & ((h & 1) == 1))).to(torch.int64)
    h = torch.where(u < 0x38800000, sub, h)
    h = torch.where(u >= 0x477FE000, torch.full_like(h, 0x7BFF), h)
    return h.to(torch.int32)


def kv8_f16f(h: torch.Tensor) -> torch.Tensor:
    """kv8.h f16f: binary16 bits (non-negative) -> fp32, exact."""
    h = h.to(torch.int64)
    e, m = (h >> 10) & 0x1F, h & 0x3FF
    normal = (((e + 112) << 23) | (m << 13)).to(torch.int32).view(torch.float32)
    return torch.where(e == 0, m.to(torch.float32) * 5.9604644775390625e-08, normal)


def kv8_encode(x: torch.Tensor):
    """kv8.h encode over the last dim (256) of x, whose values are the cache's bf16 values in fp32: the int8 codes
    [..., 256] and the fp16 scales' bits (int32 [...])."""
    y = kv8_fwht(x.float()) * kv8_signs() * 0.0625
    s16 = kv8_f16_rne(y.abs().amax(-1) / 127.0)
    sf = kv8_f16f(s16)[..., None]
    q = torch.clamp(torch.round(y / torch.where(sf == 0, torch.ones_like(sf), sf)), -127.0, 127.0)
    q = torch.where(sf == 0, torch.zeros_like(q), q)
    return q.to(torch.int8), s16


def kv8_dequant(q: torch.Tensor, s16: torch.Tensor) -> torch.Tensor:
    """kv8.h dequant: q f16f(s16), fp32 (exact), still rotated."""
    return q.float() * kv8_f16f(s16)[..., None]


def kv8_fq(x: torch.Tensor) -> torch.Tensor:
    """A K or V tensor [..., 256] as the int8 cache gives it back to attention, in the original basis, fp32: rounded
    to bf16 (what the cache would store), encoded, de-quantised, un-rotated (s (.) deq, FWHT, / 16) in fp64 - exact for
    one row's deq - and rounded once to fp32."""
    q, s16 = kv8_encode(x.to(torch.bfloat16).float())
    return (kv8_fwht(kv8_dequant(q, s16).double() * kv8_signs().double()) * 0.0625).float()
