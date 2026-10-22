#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The Nemotron 3.5 Bolt decode-kernel fixture (spec 25c Task 2): tools/oracle/bolt_ref.py's OWN restated ops (rms_norm,
route, relu2, combine) and bolt_mamba_ops' training-kernel functions (conv_seq / conv_update, ssm_update with the
engine's A, rmsnorm_gated), plus R's NoPE eager attention restated (test_bolt_ref.test_nope's form), on deterministic
inputs at the real widths, written as a C++ header that tests/kernels/bolt_ref_test.cc holds tests/kernels/bolt_ref.h to.

    tools/oracle/bolt_fixture.py > tests/kernels/bolt_fixture.h

Run in the oracle container with 25a's 5.19.0 site first on PYTHONPATH (about a minute, < 8 GB, no checkpoint; on the
Mac under the CPU lock /tmp/b70-cpu-model.lock):

    docker run --rm --memory 16g --cpus 4 -v "$PWD:/ws" -w /ws -v <main checkout>/oracle-out-q4exp/site:/site:ro \\
      -e PYTHONPATH=/site --entrypoint python3 agnes-ref-img:latest tools/oracle/bolt_fixture.py > tests/kernels/bolt_fixture.h

Inputs are not stored where a hash can make them: both sides draw them from the same integer hash (lowbias32: `h()`
below, `boltref::fixture_hash` in C++). `coarse` values are small multiples of a power of two (every dot and sum of
squares of them is exact in fp32 in ANY order, so the comparison is a bitwise check of the rounding points); `fine` values
are generic bf16; `f32` values are generic fp32 (not rounded to bf16: the SSM state, the router bias). A = -exp(A_log) is
the ENGINE's (bolt_mamba_ops' A_EXP "engine": src/common/exp_torch.h in torch fp32 ops - plan 25c's dependency: the twin
takes A from the engine's arithmetic). The cases (plan 25c Task 2):

  (a) res_norm    8 rows x 4096, w random bf16: 0 coarse, 1 fine (Σ order-dependent), 2 huge (coarse x 2^10), 3 tiny
                  (coarse x 2^-20); 4-5 fine h + ONE source of split-K slices folded (S 2: h = rne(h + rne(s0 + s1)));
                  6-7 fine h + TWO sources (a MoE block's fc2 at S 1 and shared_down at S 2: h = rne(h + rne(rne(a) +
                  rne(b0 + b1)))) - slices coarse x 2^-3 so their sums are exact
  (b) conv        18 positions x 10240 raw xBC fine, the conv weight [10240][4] and bias fine: conv_seq from a zero history
                  (= conv_update step by step, asserted) - xc at positions 0 .. 17 (the ring wraps at 16) on the channels
                  c % 5 == 0
  (c) SSM         3 steps from a random fp32 state on all 128 heads (x, B, C, dt_raw fine; dt_bias with -10.5 / +8.5 /
                  -11.6875 planted at heads 0 / 1 / 6; A_log with -4.66 / +4.41 / +8.81 / -6.81 at heads 2..5 - the
                  checkpoint's extremes; D fine): ssm_update's y [3][8192] and the state of heads 0..6, 77, 127 at rows p
                  0, 17, 63 - fp32, and the bf16-state arm (the state rounded to bf16 after every step, y from the fp32
                  update); A's bits [128]
  (d) gnorm       4 rows: y fine (row 1 group 2 dominated by one element), z fine, w random: rmsnorm_gated over groups of
                  1024 (norm_before_gate false)
  (e) route       32 rows of an E = 496 and an E = 240 layer: x fine, the router fine (amp 0.05), the bias fp32 4.5 + f32 x
                  0.04: torch's fp32 logits, the ids ASCENDING, w at them, the top-10 sel, the gap, the sum of the nine s
                  in torch's order. Crafted rows (logits exact bf16 values through a one-hot x): a 9th / 10th tie at E 496
                  between ids 310 / 400 (the second lane slot), all-equal logits, a bias that reorders, one dominant expert
                  at 495, a 9th / 10th tie at E 240 between 238 / 239
  (f) MoE         4 rows of coarse normed x through one MoE block at E 240, I 1536 (route given: 9 ids ascending, fp32 w):
                  fc1_latent, the experts' up / relu2 / down in the latent, the ascending-id combine, fc2_latent, the
                  shared expert on the normed x, o + sh - every linear int4 GPTQ dequantised by the engine-format rule
                  bf16((q - 8) x scale) with SIGNED scales (+-(1 + j / 8) x 2^-k: every dequantised weight a bf16 value):
                  the latent [4][1024], row 0's nine expert outputs [9][1024], the combine r [4][1024], the shared
                  relu2(up) [4][5376], the block's output [4][4096]
  (g) attention   NoPE eager decode (R:1077-1101 restated) of 32 q heads over 2 kv heads (GQA 16, head_dim 128) at 6 / 300
                  / 3000 / 262143 keys, coarse q / k / v: head 0's scores and probabilities (bf16; the first three cases),
                  every head's output (bf16)
  (h) argmax      131072 fine logits with a planted maximum at 0 (50) and, in a second row, at 131071 (100)
"""
import importlib.util
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("bolt_ref")
ops = _load("bolt_mamba_ops")

M32 = 0xFFFFFFFF
BF, F32 = torch.bfloat16, torch.float32
H, HEADS, P, N, G, CONV, XBC, INNER = 4096, 128, 64, 128, 8, 4, 10240, 8192
NQ, NKV, HD, LAT, SH, TOPK, V = 32, 2, 128, 1024, 5376, 9, 131072
EPS = 1e-5
CONV_T, CONV_STRIDE = 18, 5
SSM_STEPS = 3
SSM_HEADS = [0, 1, 2, 3, 4, 5, 6, 77, 127]
SSM_ROWS = [0, 17, 63]
ROUTE_ROWS = 32
ATTN_L = [6, 300, 3000, 262143]
MOE_E, MOE_I = 240, 1536
MOE_IDS = [[3, 17, 42, 99, 120, 150, 200, 230, 239], [0, 1, 2, 3, 4, 5, 6, 7, 8],
           [17, 42, 64, 99, 128, 150, 175, 199, 238], [10, 20, 30, 40, 50, 60, 70, 80, 90]]


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2^32 on int64 tensors without overflowing int64."""
    lo = (x * (c & 0xFFFF)) & M32
    hi = ((x * (c >> 16)) & 0xFFFF) << 16
    return (lo + hi) & M32


def lowbias32(x: torch.Tensor) -> torch.Tensor:
    x = x & M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    x = x ^ (x >> 16)
    return x


def h(t: int, idx: torch.Tensor, seed: int) -> torch.Tensor:
    """boltref::fixture_hash(t, i, seed) for every i of `idx` (int64)."""
    a = lowbias32(torch.tensor([(seed * 3 + t) & M32], dtype=torch.int64))
    return lowbias32(a ^ idx)


def vals(coarse: bool, t: int, n: int, seed: int, step: float = 0.125, amp: float = 1.0, base: int = 0) -> torch.Tensor:
    """boltref::fixture_val for i in [base, base + n): fp32 tensor of bf16 values (coarse values are exact)."""
    hv = h(t, torch.arange(base, base + n, dtype=torch.int64), seed)
    if coarse:
        return ((hv % 17) - 8).float() * step
    f = ((hv >> 8).float() * 2.0 ** -23 - 1.0) * amp
    return f.to(BF).float()


def f32vals(t: int, n: int, seed: int, amp: float = 1.0, base: int = 0) -> torch.Tensor:
    """boltref::fixture_f32: ((h >> 8) x 2^-23 - 1) x amp in fp32, NOT rounded to bf16."""
    hv = h(t, torch.arange(base, base + n, dtype=torch.int64), seed)
    return ((hv >> 8).float() * 2.0 ** -23 - 1.0) * amp


def bits16(x: torch.Tensor) -> list:
    return (x.contiguous().to(BF).view(torch.int16).flatten().int() & 0xFFFF).tolist()


def bits32(x: torch.Tensor) -> list:
    return (x.contiguous().float().view(torch.int32).flatten().long() & M32).tolist()


def hexs(name: str, xs: list, width: int) -> str:
    s = "".join(f"{v:0{width}x}" for v in xs)
    lines = [f'    "{s[i:i + 128]}"' for i in range(0, len(s), 128)]
    return f"inline constexpr char {name}[] =\n" + "\n".join(lines) + ";\n"


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF)


def int4_weight(t: int, K: int, N: int, shift: int) -> torch.Tensor:
    """The fixture's int4 GPTQ linear `t` as the nn.Linear weight [N][K]: nibbles q(k, n) = h(t, k x N + n, 8) % 16 and
    SIGNED f16 scales s(g, n) = (-1)^(h(t, g x N + n, 9) & 1) x (1 + ((h(t, g x N + n, 9) >> 4) % 8) / 8) x 2^-shift
    dequantised by the engine-format rule bf16(float(q - 8) x float(s)) - exact (boltref::gptq_of is the C++ twin)."""
    q = (h(t, torch.arange(K * N, dtype=torch.int64), 8) % 16).reshape(K, N)
    hs = h(t, torch.arange((K // 64) * N, dtype=torch.int64), 9).reshape(K // 64, N)
    j = (hs >> 4) % 8
    sign = 1.0 - 2.0 * (hs & 1).double()
    s = sign * (1.0 + j.double() / 8.0) * 2.0 ** -shift
    w = ((q.double() - 8.0) * s.repeat_interleave(64, dim=0)).float()       # [K][N], exact
    assert torch.equal(w.to(BF).float(), w), "a dequantised weight is not a bf16 value"
    return bf(w.t().contiguous())


def eager_nope(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """R's eager attention for ONE query row over L keys (R:1077-1101, test_bolt_ref.test_nope's restatement; the
    decode row sees every key, the mask adds 0): q [1][32][1][128], k / v [1][2][L][128] bf16 -> (scores, probs [1][32][1][L]
    bf16, out [1][32][1][128] bf16)."""
    k = k.repeat_interleave(NQ // NKV, 1)
    v = v.repeat_interleave(NQ // NKV, 1)
    s = torch.matmul(q, k.transpose(2, 3)) * (HD ** -0.5)
    p = F.softmax(s, dim=-1, dtype=F32).to(BF)
    return s, p, torch.matmul(p, v)


def main() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = [
        "#pragma once",
        f"// GENERATED by tools/oracle/bolt_fixture.py in agnes-ref-img:latest (torch {torch.__version__}, CPU capability "
        f"{torch.backends.cpu.get_cpu_capability()}, transformers {ref_version()}) - do not edit.",
        "// Command: docker run --rm --memory 16g --cpus 4 -v \"$PWD:/ws\" -w /ws -v <25a's 5.19.0 site>:/site:ro -e "
        "PYTHONPATH=/site --entrypoint python3 agnes-ref-img:latest tools/oracle/bolt_fixture.py > tests/kernels/bolt_fixture.h",
        "// tools/oracle/bolt_ref.py's restated ops and bolt_mamba_ops' functions (A by the engine's exp) on boltref::",
        "// fixture_val's inputs; 4 hex digits per bf16 value, 8 per fp32 / u32. The script's docstring has the cases.",
        "namespace bolt_fixture {",
        f"inline constexpr unsigned kHidden = {H}, kHeads = {HEADS}, kXbc = {XBC}, kInner = {INNER}, kLatent = {LAT}, "
        f"kShared = {SH};",
    ]

    # (a) res_norm ----------------------------------------------------------------------------------------------
    w = bf(1.0 + vals(False, 1, H, 1, amp=0.5))
    rows = [vals(True, 0, H, 10, step=0.125), vals(False, 0, H, 11, amp=4.0), vals(True, 0, H, 12, step=128.0),
            vals(True, 0, H, 13, step=2.0 ** -20)] + [vals(False, 0, H, 14 + r, amp=2.0) for r in range(4)]
    xs, hs = [], []
    for r, row in enumerate(rows):
        hrow = bf(row)
        if 4 <= r < 6:     # one source, S 2 (an out_proj / o_proj): h = rne(h + rne(s0 + s1))
            s0 = vals(True, 2, H, 20 + r, step=2.0 ** -3)
            s1 = vals(True, 3, H, 20 + r, step=2.0 ** -3)
            hrow = hrow + bf(s0 + s1)
            hs.append(hrow)
        elif r >= 6:       # two sources: fc2 (S 1) and shared_down (S 2): mixer = rne(o) + rne(sh), h = h + mixer
            a = vals(True, 2, H, 20 + r, step=2.0 ** -3)
            b0 = vals(True, 3, H, 20 + r, step=2.0 ** -3)
            b1 = vals(True, 4, H, 20 + r, step=2.0 ** -3)
            hrow = hrow + (bf(a) + bf(b0 + b1))
            hs.append(hrow)
        xs.append(ref.rms_norm(hrow[None], w, EPS)[0])
    out.append("// (a) res_norm: w = rf(1 + fine(t 1, seed 1) x 0.5); rows: 0 coarse (t 0, seed 10, 1/8), 1 fine (seed 11, amp")
    out.append("//     4), 2 coarse x 128 (seed 12), 3 coarse x 2^-20 (seed 13), 4-7 fine (seed 14 + r, amp 2); 4-5 fold slices")
    out.append("//     coarse (t 2 / 3, seed 20 + r, 1/8); 6-7 fold a (t 2) and b0 + b1 (t 3 / 4). x [8][4096], h_out of rows 4-7")
    out.append(hexs("kNormX", bits16(torch.stack(xs)), 4))
    out.append(hexs("kNormH", bits16(torch.stack(hs)), 4))

    # (b) the conv -------------------------------------------------------------------------------------------------
    X = bf(vals(False, 20, CONV_T * XBC, 5, amp=2.0).reshape(CONV_T, XBC))
    cw = bf(vals(False, 21, XBC * CONV, 5, amp=0.5).reshape(XBC, CONV))
    cb = bf(vals(False, 22, XBC, 5, amp=0.5))
    y_seq = ops.conv_seq(X.t()[None].contiguous(), cw, cb, None)[0].t()           # [T][C] bf16
    ring = torch.zeros(1, XBC, CONV - 1, dtype=F32)
    for p in range(CONV_T):                                                       # conv_update step by step: the same
        yp = ops.conv_update(X[p][None], ring, cw, cb)
        assert torch.equal(yp[0], y_seq[p]), "conv_update and conv_seq disagree"
        ring = torch.cat([ring[..., 1:], X[p][None, :, None].float()], -1)
    sub = torch.arange(0, XBC, CONV_STRIDE)
    out.append(f"// (b) conv: raw xBC fine (t 20, seed 5, amp 2, index p x 10240 + c), w fine (t 21, index c x 4 + j - j 0 the")
    out.append(f"//     OLDEST tap, amp 0.5), bias fine (t 22, amp 0.5); xc [{CONV_T}][{XBC // CONV_STRIDE}] at positions 0 .. "
               f"{CONV_T - 1} from a zero history, channels c % {CONV_STRIDE} == 0")
    out.append(f"inline constexpr unsigned kConvT = {CONV_T}, kConvStride = {CONV_STRIDE};")
    out.append(hexs("kConvXc", bits16(y_seq[:, sub]), 4))

    # (c) the SSM update -------------------------------------------------------------------------------------------
    xc = bf(vals(False, 30, SSM_STEPS * XBC, 6).reshape(SSM_STEPS, XBC))
    dt_raw = bf(vals(False, 31, SSM_STEPS * HEADS, 6, amp=2.0).reshape(SSM_STEPS, HEADS))
    dt_bias = vals(False, 32, HEADS, 6, amp=4.0)
    dt_bias[0], dt_bias[1], dt_bias[6] = -10.5, 8.5, -11.6875
    A_log = vals(False, 33, HEADS, 6, amp=4.0)
    for hh, v in ((2, -4.66), (3, 4.41), (4, 8.81), (5, -6.81)):
        A_log[hh] = float(torch.tensor(v).to(BF))
    D = vals(False, 34, HEADS, 6, amp=2.0)
    S0 = f32vals(35, HEADS * P * N, 6).reshape(HEADS, P, N)
    A = ops.A_from_log(A_log.to(BF), "engine")
    A_t = ops.A_from_log(A_log.to(BF), "torch")
    ys, ys16 = [], []
    S, S16 = S0.clone(), S0.to(BF)
    for m in range(SSM_STEPS):
        x = xc[m, :INNER].reshape(1, HEADS, P)
        Bm = xc[m, INNER:INNER + G * N].reshape(1, G, N)
        Cm = xc[m, INNER + G * N:].reshape(1, G, N)
        y, S = ops.ssm_update(S[None], x, dt_raw[m][None], A, Bm, Cm, D, dt_bias)
        S = S[0]
        ys.append(y[0].reshape(INNER))
        y16, S16n = ops.ssm_update(S16.float()[None], x, dt_raw[m][None], A, Bm, Cm, D, dt_bias)
        S16 = S16n[0].to(BF)
        ys16.append(y16[0].reshape(INNER))
    pick = torch.tensor([[hh * P + p for p in SSM_ROWS] for hh in SSM_HEADS]).flatten()
    out.append("// (c) SSM: xc fine (t 30, seed 6, index m x 10240 + c), dt_raw fine (t 31, amp 2, index m x 128 + h), dt_bias")
    out.append("//     fine (t 32, amp 4) with heads 0 / 1 / 6 = -10.5 / 8.5 / -11.6875, A_log fine (t 33, amp 4) with heads 2..5 =")
    out.append("//     bf16(-4.66 / 4.41 / 8.81 / -6.81), D fine (t 34, amp 2), the state f32 (t 35, index (h x 64 + p) x 128 + n);")
    out.append("//     A = the ENGINE's -exp_torch(A_log) [128]; y [3][8192] and the state at kSsmHeads x kSsmRows [.][128] after")
    out.append("//     3 steps; the bf16-state arm likewise. kSsmATorchDiff: heads where torch.exp's A differs (1 ulp)")
    out.append(f"inline constexpr unsigned kSsmSteps = {SSM_STEPS}, kSsmNHeads = {len(SSM_HEADS)}, kSsmNRows = {len(SSM_ROWS)};")
    out.append("inline constexpr unsigned kSsmHeads[] = {" + ", ".join(map(str, SSM_HEADS)) + "};")
    out.append("inline constexpr unsigned kSsmRows[] = {" + ", ".join(map(str, SSM_ROWS)) + "};")
    out.append(f"inline constexpr unsigned kSsmATorchDiff = {int((A != A_t).sum())};")
    out.append(hexs("kSsmA", bits32(A), 8))
    out.append(hexs("kSsmY", bits16(torch.stack(ys)), 4))
    out.append(hexs("kSsmState", bits32(S.reshape(HEADS * P, N)[pick]), 8))
    out.append(hexs("kSsmYB16", bits16(torch.stack(ys16)), 4))
    out.append(hexs("kSsmStateB16", bits16(S16.reshape(HEADS * P, N)[pick]), 4))

    # (d) the gated norm -------------------------------------------------------------------------------------------
    yg = vals(False, 40, 4 * INNER, 7, amp=2.0).reshape(4, INNER)
    yg[1, 2 * 1024 + 5] = 60.0
    zg = vals(False, 41, 4 * INNER, 7, amp=4.0).reshape(4, INNER)
    wg = bf(1.0 + vals(False, 42, INNER, 7, amp=0.5))
    og = ops.rmsnorm_gated(bf(yg), bf(zg), wg, EPS, 1024)
    out.append("// (d) gnorm: y fine (t 40, seed 7, amp 2) [4][8192] with y[1][2053] = 60, z fine (t 41, amp 4), w = rf(1 + fine(t")
    out.append("//     42) x 0.5): rmsnorm_gated over groups of 1024, norm_before_gate false")
    out.append(hexs("kGnorm", bits16(og), 4))

    # (e) the route ------------------------------------------------------------------------------------------------
    for E, seed in ((496, 8), (240, 9)):
        x = bf(vals(False, 50, ROUTE_ROWS * H, seed).reshape(ROUTE_ROWS, H))
        gw = bf(vals(False, 51, E * H, seed, amp=0.05).reshape(E, H))
        bias = f32vals(52, E, seed) * 0.04 + 4.5
        logits = F.linear(x.float(), gw.float())
        ids, wr, ids_a, w_a, sel10, gap, tot = ref.route(x, gw, bias)
        out.append(f"// (e) route E {E}: x fine (t 50, seed {seed}, index row x 4096 + k), router fine (t 51, amp 0.05, index n x 4096")
        out.append(f"//     + k), bias f32 (t 52) x 0.04 + 4.5; torch's fp32 logits [{ROUTE_ROWS}][{E}], ids ascending [.][9], w at")
        out.append("//     them, the top-10 sel (descending), the gap, the sum of the nine s in torch's order")
        out.append(hexs(f"kRouteLogits{E}", bits32(logits), 8))
        out.append(hexs(f"kRouteIds{E}", ids_a.flatten().tolist(), 8))
        out.append(hexs(f"kRouteW{E}", bits32(w_a), 8))
        out.append(hexs(f"kRouteSel{E}", bits32(sel10), 8))
        out.append(hexs(f"kRouteGap{E}", bits32(gap), 8))
        out.append(hexs(f"kRouteSum{E}", bits32(tot), 8))
    # crafted rows: x = e_0, so the logits are the router's column 0 exactly (one product)
    CR = 5
    cE = [496, 496, 496, 496, 240]
    cl = torch.full((CR, 496), -4.0)
    cb_ = torch.zeros(CR, 496)
    cl[0, 300:308] = 2.0                                   # 0: 300..307 ahead, 310 / 400 tie for the 9th (2nd lane slot)
    cl[0, 310] = 1.0
    cl[0, 400] = 1.0
    cl[1] = 0.0                                            # 1: all-equal logits, zero bias - every sel ties
    cl[2] = -3.0                                           # 2: by s the top nine are 10..18; the bias lifts 20..28
    cl[2, 10:19] = 1.0
    cl[2, 20:29] = 0.5
    cb_[2, 20:29] = 0.25
    cl[3] = -8.0                                           # 3: one dominant expert, 495
    cl[3, 495] = 8.0
    cl[4] = -4.0                                           # 4: E 240: 230..237 ahead, 238 / 239 tie for the 9th
    cl[4, 230:238] = 2.0
    cl[4, 238] = 1.0
    cl[4, 239] = 1.0
    cr_ids, cr_w, cr_sel, cr_gap, cr_sum = [], [], [], [], []
    for r in range(CR):
        E = cE[r]
        e0 = torch.zeros(1, H, dtype=BF)
        e0[0, 0] = 1.0
        g_ = torch.zeros(E, H, dtype=BF)
        g_[:, 0] = bf(cl[r, :E])
        _, _, ia, wa, s10, gp, tt = ref.route(e0, g_, cb_[r, :E])
        cr_ids.append(ia[0])
        cr_w.append(wa[0].float())
        cr_sel.append(s10[0])
        cr_gap.append(gp)
        cr_sum.append(tt)
    out.append(f"// crafted rows (one-hot x): E [{CR}], logits [{CR}][496] fp32 (bf16 values; an E 240 row uses its first 240),")
    out.append("//     bias [.][496]; torch's ids ascending / w / top-10 sel / gap / sum")
    out.append(f"inline constexpr unsigned kCraftN = {CR};")
    out.append("inline constexpr unsigned kCraftE[] = {" + ", ".join(map(str, cE)) + "};")
    out.append(hexs("kCraftLogits", bits32(cl), 8))
    out.append(hexs("kCraftBias", bits32(cb_), 8))
    out.append(hexs("kCraftIds", torch.stack(cr_ids).flatten().tolist(), 8))
    out.append(hexs("kCraftW", bits32(torch.stack(cr_w)), 8))
    out.append(hexs("kCraftSel", bits32(torch.stack(cr_sel)), 8))
    out.append(hexs("kCraftGap", bits32(torch.cat(cr_gap)), 8))
    out.append(hexs("kCraftSum", bits32(torch.cat(cr_sum)), 8))

    # (f) the MoE block --------------------------------------------------------------------------------------------
    fc1 = int4_weight(300, H, LAT, 9)
    shu = int4_weight(301, H, SH, 9)
    fc2 = int4_weight(302, LAT, H, 7)
    shd = int4_weight(303, SH, H, 9)
    xm = bf(vals(True, 60, 4 * H, 10, step=0.125).reshape(4, H))
    wm = f32vals(61, 4 * TOPK, 10, amp=0.5).reshape(4, TOPK) + 0.6
    ids_m = torch.tensor(MOE_IDS, dtype=torch.int64)
    lat = F.linear(xm, fc1)
    d = torch.zeros(4, TOPK, LAT, dtype=BF)
    for e in sorted(set(ids_m.reshape(-1).tolist())):
        tok, slot = torch.where(ids_m == e)
        up = int4_weight(400 + 2 * e, LAT, MOE_I, 7)
        dn = int4_weight(401 + 2 * e, MOE_I, LAT, 7)
        d[tok, slot] = F.linear(ref.relu2(F.linear(lat[tok], up)), dn)
    r_ = ref.combine(d, wm, ids_m)
    o = F.linear(r_, fc2)
    sh_act = ref.relu2(F.linear(xm, shu))
    sh = F.linear(sh_act, shd)
    blk = o + sh
    out.append(f"// (f) MoE E {MOE_E} I {MOE_I}: x coarse (t 60, seed 10, 1/8) [4][4096]; ids [4][9] ascending; w f32 (t 61, amp 0.5)")
    out.append("//     + 0.6 [4][9]; int4 linears by boltref::gptq_of(t, K, N, shift): fc1 (300, 4096, 1024, 9), shared up (301,")
    out.append("//     4096, 5376, 9), fc2 (302, 1024, 4096, 7), shared down (303, 5376, 4096, 9), expert e up (400 + 2e, 1024,")
    out.append("//     1536, 7) / down (401 + 2e, 1536, 1024, 7); lat [4][1024], row 0's d [9][1024], r [4][1024], the shared")
    out.append("//     relu2(up) [4][5376], the block's output o + sh [4][4096]")
    out.append(f"inline constexpr unsigned kMoeE = {MOE_E}, kMoeI = {MOE_I};")
    out.append("inline constexpr unsigned kMoeIds[4][9] = {" + ", ".join("{" + ", ".join(map(str, r)) + "}" for r in MOE_IDS) + "};")
    out.append(hexs("kMoeW", bits32(wm), 8))
    out.append(hexs("kMoeLat", bits16(lat), 4))
    out.append(hexs("kMoeD0", bits16(d[0]), 4))
    out.append(hexs("kMoeR", bits16(r_), 4))
    out.append(hexs("kMoeShAct", bits16(sh_act), 4))
    out.append(hexs("kMoeOut", bits16(blk), 4))

    # (g) eager NoPE attention -------------------------------------------------------------------------------------
    out.append("// (g) eager attention: q [32][128] coarse (t 70, seed = case, step 1/2), key p's k / v [2][128] coarse (t 71 / 72,")
    out.append("//     seed 9, step 1/8, index p x 256 + j x 128 + d); head 0's scores / probabilities [L] (bf16; cases 0-2), every")
    out.append("//     head's output [32][128] (bf16)")
    out.append(f"inline constexpr unsigned kAttnCases = {len(ATTN_L)};")
    out.append("inline constexpr unsigned kAttnL[] = {" + ", ".join(map(str, ATTN_L)) + "};")
    for ci, L in enumerate(ATTN_L):
        q = bf(vals(True, 70, NQ * HD, ci, step=0.5).reshape(1, NQ, 1, HD))
        k = bf(vals(True, 71, L * NKV * HD, 9).reshape(L, NKV, HD).permute(1, 0, 2)[None])
        v = bf(vals(True, 72, L * NKV * HD, 9).reshape(L, NKV, HD).permute(1, 0, 2)[None])
        s, p, o = eager_nope(q, k, v)
        if ci < 3:
            out.append(hexs(f"kAttnS{ci}", bits16(s[0, 0, 0]), 4))
            out.append(hexs(f"kAttnP{ci}", bits16(p[0, 0, 0]), 4))
        out.append(hexs(f"kAttnO{ci}", bits16(o[0, :, 0]), 4))
        del k, v, s, p
    out.append("inline constexpr const char* kAttnS[] = {kAttnS0, kAttnS1, kAttnS2};")
    out.append("inline constexpr const char* kAttnP[] = {kAttnP0, kAttnP1, kAttnP2};")
    out.append("inline constexpr const char* kAttnO[] = {kAttnO0, kAttnO1, kAttnO2, kAttnO3};")

    # (h) argmax -----------------------------------------------------------------------------------------------------
    lg = vals(False, 90, V, 4)
    la, lb = lg.clone(), lg.clone()
    la[0] = 50.0
    lb[V - 1] = 100.0
    out.append("// (h) argmax: logits fine (t 90, seed 4) [131072]; row A logits[0] = 50, row B logits[131071] = 100; torch's argmax")
    out.append(f"inline constexpr unsigned kArgmaxA = {int(torch.argmax(la))}, kArgmaxB = {int(torch.argmax(lb))};")
    out.append("}  // namespace bolt_fixture")
    sys.stdout.write("\n".join(out) + "\n")


def ref_version() -> str:
    import transformers
    return transformers.__version__


if __name__ == "__main__":
    main()
