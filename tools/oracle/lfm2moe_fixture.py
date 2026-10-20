#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The LFM2.5-24B-A2B decode-kernel fixture (spec 24c Task 2): tools/oracle/lfm2moe_ref.py's OWN restated ops
(rms_norm, conv_taps, route, expert_outputs, combine) and transformers 5.19.0's own lfm2_moe pieces
(Lfm2MoeRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward) on deterministic inputs at the real
widths, written as a C++ header that tests/kernels/lfm2moe_ref_test.cc holds tests/kernels/lfm2moe_ref.h to.

    tools/oracle/lfm2moe_fixture.py > tests/kernels/lfm2moe_fixture.h

Run in the oracle container with 24a's 5.19.0 site first on PYTHONPATH (seconds, < 2 GB, no checkpoint):

    docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -w /ws -e PYTHONPATH=/ws/oracle-out-q4exp/site \\
      --entrypoint python3 agnes-ref-img:latest tools/oracle/lfm2moe_fixture.py > tests/kernels/lfm2moe_fixture.h

Inputs are not stored where a hash can make them: both sides draw them from the same integer hash (lowbias32:
`h()` below, `lfmref::fixture_hash` in C++). `coarse` values are small multiples of a power of two (every dot and
sum of squares of them is exact in fp32 in ANY order, so the comparison is a bitwise check of the rounding points);
`fine` values are generic bf16. The cases (spec 24c plan Task 2):

  (a) res_norm    8 rows x 2048 with w random bf16: 0 coarse, 1 fine (Σ order-dependent), 2 huge (coarse x 2^10),
                  3 tiny (coarse x 2^-20), 4-7 fine h with two fp32 split-K slices folded first (h = rne(h +
                  rne(s0 + s1)); slices coarse x 2^-3 so their sum is exact)
  (b) conv        18 positions x 2048 channels: B, C, x bf16 fine; the conv weight [2048][3] bf16 fine; Bx = B x x,
                  c = conv_taps(Bx, w, zero history), y = C x c - every position (0 .. 17: the ring wraps at 16)
  (c) route       64 rows of a real-width layer: x fine, the router [64][2048] fine (amp 0.05), expert_bias fp32
                  (amp 0.05): torch's bf16 logits, ids (rank order), w, the top-5 sel, the gap. Then four crafted
                  rows through a one-hot x (the logits are the router's column 0 exactly): a tie at the 4th / 5th,
                  all-equal logits, a bias that reorders, one dominant expert
  (d) MoE         4 rows of coarse x, four slots each (ids given, rank order), 7 int4 experts dequantised by 24b's
                  rule (bf16((q - 8) x scale)) with scales 2^-7 x (1 + j / 8) (so every dequantised weight is exactly
                  a bf16), weights w bf16: the experts' outputs `proj` [4][4][2048] (grouped_mm), the combine y
                  [4][2048], row 0's gate||up activations [4][1536]
  (e) attention   the q / k head norm (64, eps 1e-5) and RoPE at positions 0, 1, 4095, 131071; eager attention of
                  32 q heads over 8 kv heads (GQA 4) at 6 / 300 / 3000 keys, coarse q / k / v: head 0's scores and
                  probabilities (bf16), every head's output (bf16)
  (f) silu_mul    I 11776: g, u bf16 fine (amp 4): rne(rne(silu(g)) x u)
  (g) argmax      128000 fine logits with planted maxima at 125016 (50) and 127999 (100): torch's argmax over the
                  125017 real ids
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("lfm2moe_ref", os.path.join(_HERE, "lfm2moe_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["lfm2moe_ref"] = ref
_spec.loader.exec_module(ref)

M32 = 0xFFFFFFFF
H, NQ, NKV, HD, QKV, E, TOPK, I, DI, V, VU = 2048, 32, 8, 64, 3072, 64, 4, 1536, 11776, 128000, 125017
EPS = 1e-5
ROPE_POS = [0, 1, 4095, 131071]
CONV_T = 18
ATTN_L = [6, 300, 3000]
MOE_IDS = [[5, 17, 42, 63], [17, 5, 0, 33], [63, 0, 8, 5], [33, 8, 42, 17]]


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
    """lfmref::fixture_hash(t, i, seed) for every i of `idx` (int64)."""
    a = lowbias32(torch.tensor([(seed * 3 + t) & M32], dtype=torch.int64))
    return lowbias32(a ^ idx)


def vals(coarse: bool, t: int, n: int, seed: int, step: float = 0.125, amp: float = 1.0, base: int = 0) -> torch.Tensor:
    """lfmref::fixture_val for i in [base, base + n): fp32 tensor of bf16 values (coarse values are exact)."""
    hv = h(t, torch.arange(base, base + n, dtype=torch.int64), seed)
    if coarse:
        return ((hv % 17) - 8).float() * step
    f = ((hv >> 8).float() * 2.0 ** -23 - 1.0) * amp
    return f.to(torch.bfloat16).float()


def bias_vals(t: int, n: int, seed: int) -> torch.Tensor:
    """lfmref's trained-like expert_bias: fp32 ((h >> 8) x 2^-23 - 1) x 0.05, NOT rounded to bf16."""
    hv = h(t, torch.arange(n, dtype=torch.int64), seed)
    return ((hv >> 8).float() * 2.0 ** -23 - 1.0) * 0.05


def bits16(x: torch.Tensor) -> list:
    return (x.contiguous().to(torch.bfloat16).view(torch.int16).flatten().int() & 0xFFFF).tolist()


def bits32(x: torch.Tensor) -> list:
    return (x.contiguous().float().view(torch.int32).flatten().long() & M32).tolist()


def hexs(name: str, xs: list, width: int) -> str:
    s = "".join(f"{v:0{width}x}" for v in xs)
    lines = [f'    "{s[i:i + 128]}"' for i in range(0, len(s), 128)]
    return f"inline constexpr char {name}[] =\n" + "\n".join(lines) + ";\n"


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def expert_weight(e: int, mat: int, K: int, N: int) -> torch.Tensor:
    """Expert e's matrix `mat` (0 w1 / gate, 1 w3 / up, 2 w2 / down) as the nn.Linear weight [N][K]: dequantised
    from GPTQ nibbles q(k, n) = h(100 + 4e + mat, k x N + n, 8) % 16 and f16 scales s(g, n) = 2^-7 x (1 +
    ((h(100 + 4e + mat, g x N + n, 9) >> 4) % 8) / 8): bf16(float(q - 8) x float(s)) - exact (lfmref::expert_q /
    expert_scale are the C++ twins)."""
    t = 100 + 4 * e + mat
    q = (h(t, torch.arange(K * N, dtype=torch.int64), 8) % 16).reshape(K, N)
    j = ((h(t, torch.arange((K // 64) * N, dtype=torch.int64), 9) >> 4) % 8).reshape(K // 64, N)
    s = (1.0 + j.double() / 8.0) * 2.0 ** -7
    w = (q.double() - 8.0) * s.repeat_interleave(64, dim=0)       # [K][N], exact
    w = w.float()
    assert torch.equal(w.to(torch.bfloat16).float(), w), "a dequantised weight is not a bf16 value"
    return bf(w.t().contiguous())


def main() -> None:
    out = [
        "#pragma once",
        f"// GENERATED by tools/oracle/lfm2moe_fixture.py in agnes-ref-img:latest (torch {torch.__version__}, "
        f"CPU capability {torch.backends.cpu.get_cpu_capability()}, transformers {ref_version()}) - do not edit.",
        "// Command: docker run --rm --memory 8g --cpus 4 -v \"$PWD:/ws\" -w /ws -e PYTHONPATH=<24a's 5.19.0 site> "
        "--entrypoint python3 agnes-ref-img:latest tools/oracle/lfm2moe_fixture.py > tests/kernels/lfm2moe_fixture.h",
        "// tools/oracle/lfm2moe_ref.py's restated ops and transformers 5.19.0's lfm2_moe pieces on lfmref::fixture_val's",
        "// inputs; 4 hex digits per bf16 value, 8 per fp32 / u32. The script's docstring has the cases.",
        "namespace lfm2moe_fixture {",
        f"inline constexpr unsigned kHidden = {H}, kExperts = {E}, kTopK = {TOPK}, kInter = {I}, kDenseInter = {DI};",
    ]

    # (a) res_norm ----------------------------------------------------------------------------------------------
    w = (1.0 + vals(False, 1, H, 1, amp=0.5)).to(torch.bfloat16)
    rows = [
        vals(True, 0, H, 10, step=0.125),
        vals(False, 0, H, 11, amp=4.0),
        vals(True, 0, H, 12, step=128.0),
        vals(True, 0, H, 13, step=2.0 ** -20),
    ] + [vals(False, 0, H, 14 + r, amp=2.0) for r in range(4)]
    xs, hs = [], []
    for r, row in enumerate(rows):
        hrow = bf(row)
        if r >= 4:   # two coarse fp32 slices, their sum exact; the fold is the decoder layer's bf16 add
            s0 = vals(True, 2, H, 20 + r, step=2.0 ** -3)
            s1 = vals(True, 3, H, 20 + r, step=2.0 ** -3)
            hrow = hrow + bf(s0 + s1)
            hs.append(hrow)
        xs.append(ref.rms_norm(hrow[None], w, EPS)[0])
    out.append("// (a) res_norm: w = rf(1 + fine(t 1, seed 1) x 0.5); rows: 0 coarse (t 0, seed 10, 1/8), 1 fine (seed 11,")
    out.append("//     amp 4), 2 coarse x 128 (seed 12), 3 coarse x 2^-20 (seed 13), 4-7 fine (seed 14 + r, amp 2) folded with")
    out.append("//     slices coarse (t 2 / 3, seed 20 + r, step 1/8). x [8][2048], h_out of rows 4-7 [4][2048]")
    out.append(hexs("kNormX", bits16(torch.stack(xs)), 4))
    out.append(hexs("kNormH", bits16(torch.stack(hs)), 4))

    # (b) the short conv -----------------------------------------------------------------------------------------
    B = bf(vals(False, 20, CONV_T * H, 5, amp=2.0).reshape(CONV_T, H))
    C = bf(vals(False, 21, CONV_T * H, 5, amp=2.0).reshape(CONV_T, H))
    X = bf(vals(False, 22, CONV_T * H, 5, amp=2.0).reshape(CONV_T, H))
    cw = bf(vals(False, 23, H * 3, 5, amp=0.5).reshape(H, 1, 3))
    bx = B * X
    c = ref.conv_taps(bx, cw, torch.zeros(2, H, dtype=torch.bfloat16))
    y = C * c
    out.append(f"// (b) conv: B, C, x fine (t 20 / 21 / 22, seed 5, amp 2, index p x 2048 + c), w fine (t 23, index c x 3 + j,")
    out.append(f"//     amp 0.5); Bx [{CONV_T}][2048], y [{CONV_T}][2048] for positions 0 .. {CONV_T - 1} from a zero history")
    out.append(f"inline constexpr unsigned kConvT = {CONV_T};")
    out.append(hexs("kConvBx", bits16(bx), 4))
    out.append(hexs("kConvY", bits16(y), 4))

    # (c) the route ------------------------------------------------------------------------------------------------
    x = bf(vals(False, 30, 64 * H, 6).reshape(64, H))
    gw = bf(vals(False, 31, E * H, 6, amp=0.05).reshape(E, H))
    bias = bias_vals(32, E, 6)
    logits = F.linear(x, gw)
    ids, wr, sel, gap = ref.route(x, gw, bias, TOPK)
    top5 = sel.topk(TOPK + 1, dim=-1).values
    out.append("// (c) route: x fine (t 30, seed 6, index row x 2048 + k), router fine (t 31, seed 6, amp 0.05, index n x 2048")
    out.append("//     + k), bias fp32 (t 32, seed 6: lfmref bias_val); torch's bf16 logits [64][64], ids [64][4] (rank order),")
    out.append("//     w [64][4] (bf16 values as fp32), the top-5 sel [64][5], gap [64]")
    out.append(hexs("kRouteLogits", bits16(logits), 4))
    out.append(hexs("kRouteIds", ids.flatten().tolist(), 8))
    out.append(hexs("kRouteW", bits32(wr.float()), 8))
    out.append(hexs("kRouteSel5", bits32(top5), 8))
    out.append(hexs("kRouteGap", bits32(gap), 8))
    # crafted rows: x = e_0, so the logits are the router's column 0 exactly (one product)
    crafted_l = torch.zeros(4, E)
    crafted_b = torch.zeros(4, E)
    crafted_l[0] = -4.0                                    # 0: experts 3, 9, 20 ahead; 31 and 40 tie for the 4th
    for e_ in (3, 9, 20):
        crafted_l[0, e_] = 2.0
    crafted_l[0, 31] = 1.0
    crafted_l[0, 40] = 1.0
    crafted_l[1] = 0.0                                     # 1: all-equal logits, zero bias - every sel ties
    crafted_l[2] = -3.0                                    # 2: by s the top four are 10..13, the bias lifts 20..23
    crafted_l[2, 10:14] = 1.0
    crafted_l[2, 20:24] = 0.5
    crafted_b[2, 20:24] = 0.25
    crafted_l[3] = -8.0                                    # 3: one dominant expert
    crafted_l[3, 7] = 8.0
    e0 = torch.zeros(1, H, dtype=torch.bfloat16)
    e0[0, 0] = 1.0
    cr_ids, cr_w, cr_sel, cr_gap = [], [], [], []
    for r in range(4):
        g_ = torch.zeros(E, H, dtype=torch.bfloat16)
        g_[:, 0] = bf(crafted_l[r])
        i_, w_, s_, gp_ = ref.route(e0, g_, crafted_b[r], TOPK)
        cr_ids.append(i_[0])
        cr_w.append(w_[0].float())
        cr_sel.append(s_.topk(TOPK + 1, dim=-1).values[0])
        cr_gap.append(gp_)
    out.append("// crafted rows (one-hot x): logits [4][64] fp32 (bf16 values), bias [4][64]; torch's ids / w / top-5 sel / gap")
    out.append(hexs("kCraftLogits", bits32(crafted_l), 8))
    out.append(hexs("kCraftBias", bits32(crafted_b), 8))
    out.append(hexs("kCraftIds", torch.stack(cr_ids).flatten().tolist(), 8))
    out.append(hexs("kCraftW", bits32(torch.stack(cr_w)), 8))
    out.append(hexs("kCraftSel5", bits32(torch.stack(cr_sel)), 8))
    out.append(hexs("kCraftGap", bits32(torch.cat(cr_gap)), 8))

    # (d) the MoE block --------------------------------------------------------------------------------------------
    used = sorted({e_ for row in MOE_IDS for e_ in row})
    gate_up = torch.stack([torch.cat([expert_weight(e_, 0, H, I), expert_weight(e_, 1, H, I)], 0) for e_ in used])
    down = torch.stack([expert_weight(e_, 2, I, H) for e_ in used])
    remap = {e_: k for k, e_ in enumerate(used)}
    ids_m = torch.tensor([[remap[e_] for e_ in row] for row in MOE_IDS], dtype=torch.int64)
    xm = bf(vals(True, 40, 4 * H, 7, step=0.125).reshape(4, H))
    wm = bf(0.5 + vals(False, 41, 4 * TOPK, 7, amp=0.5).reshape(4, TOPK))
    proj = ref.expert_outputs(xm, gate_up, down, ids_m, "grouped_mm")         # [4][4][2048]
    ym = ref.combine(proj, wm, "grouped_mm")                                  # [4][2048]
    g0 = F.linear(xm[0:1], gate_up[:, :I].reshape(-1, H)).reshape(len(used), I)   # every used expert's gate on row 0
    u0 = F.linear(xm[0:1], gate_up[:, I:].reshape(-1, H)).reshape(len(used), I)
    act0 = torch.stack([F.silu(g0[remap[e_]]) * u0[remap[e_]] for e_ in MOE_IDS[0]])
    out.append("// (d) MoE: ids [4][4] (rank order), x coarse (t 40, seed 7, 1/8), w bf16 rf(0.5 + fine(t 41, seed 7) x 0.5)")
    out.append("//     [4][4]; experts by lfmref::expert_q / expert_scale (t 100 + 4e + mat); proj [4][4][2048], y [4][2048],")
    out.append("//     row 0's activations [4][1536]")
    out.append("inline constexpr unsigned kMoeIds[4][4] = {" + ", ".join("{" + ", ".join(map(str, r)) + "}" for r in MOE_IDS) + "};")
    out.append(hexs("kMoeW", bits32(wm.float()), 8))
    out.append(hexs("kMoeProj", bits16(proj), 4))
    out.append(hexs("kMoeY", bits16(ym), 4))
    out.append(hexs("kMoeAct0", bits16(act0), 4))

    # (e) attention: prep and eager --------------------------------------------------------------------------------
    m = ref.mm()
    cfg = ref.config(None, raw={
        "model_type": "lfm2_moe", "hidden_size": H, "num_attention_heads": NQ, "num_key_value_heads": NKV,
        "max_position_embeddings": 131072, "rope_parameters": {"rope_type": "default", "rope_theta": 1e7},
        "num_hidden_layers": 1, "layer_types": ["full_attention"], "num_dense_layers": 1, "intermediate_size": 64,
        "moe_intermediate_size": 64, "num_experts": 1, "num_experts_per_tok": 1, "vocab_size": 8, "conv_L_cache": 3,
        "conv_bias": False, "norm_eps": EPS, "use_expert_bias": True, "norm_topk_prob": True,
        "routed_scaling_factor": 1.0, "tie_word_embeddings": True, "bos_token_id": 0, "eos_token_id": 1,
        "pad_token_id": 2,
    })
    rope = m.Lfm2MoeRotaryEmbedding(cfg)
    qh = bf(vals(True, 60, HD, 3, step=0.25))
    kh = bf(vals(True, 61, HD, 3, step=0.25))
    qw = bf(1.0 + vals(False, 62, HD, 3, amp=0.5))
    kw = bf(1.0 + vals(False, 63, HD, 3, amp=0.5))
    qn = ref.rms_norm(qh[None], qw, EPS)[0]
    kn = ref.rms_norm(kh[None], kw, EPS)[0]
    cos, sin = rope(torch.zeros(1, 1, dtype=torch.bfloat16), torch.tensor([ROPE_POS]))   # [1, P, 64]
    qr, kr = m.apply_rotary_pos_emb(qn[None, None, None].expand(1, 1, len(ROPE_POS), HD),
                                    kn[None, None, None].expand(1, 1, len(ROPE_POS), HD), cos, sin)
    out.append("// (e) prep: q / k head coarse (t 60 / 61, seed 3, step 1/4), q_norm / k_norm rf(1 + fine(t 62 / 63) x 0.5);")
    out.append("//     the normed heads [64], then RoPE'd at kRopePos [P][64] (transformers' Lfm2MoeRotaryEmbedding)")
    out.append(f"inline constexpr unsigned kRopeN = {len(ROPE_POS)};")
    out.append("inline constexpr unsigned kRopePos[] = {" + ", ".join(map(str, ROPE_POS)) + "};")
    out.append(hexs("kQNormed", bits16(qn), 4))
    out.append(hexs("kKNormed", bits16(kn), 4))
    out.append(hexs("kQRoped", bits16(qr[0, 0]), 4))
    out.append(hexs("kKRoped", bits16(kr[0, 0]), 4))

    class _Mod:
        num_key_value_groups = NQ // NKV
        training = False

    out.append("// eager: q [32][64] coarse (t 70, seed = case, step 1/2), key p's k / v [8][64] coarse (t 71 / 72, seed 9, step")
    out.append("//     1/8, index p x 512 + j x 64 + d); head 0's scores / probabilities [L] (bf16), every head's output [32][64]")
    out.append(f"inline constexpr unsigned kAttnCases = {len(ATTN_L)};")
    out.append("inline constexpr unsigned kAttnL[] = {" + ", ".join(map(str, ATTN_L)) + "};")
    for ci, L in enumerate(ATTN_L):
        q = bf(vals(True, 70, NQ * HD, ci, step=0.5).reshape(1, NQ, 1, HD))
        k = bf(vals(True, 71, L * NKV * HD, 9).reshape(L, NKV, HD).permute(1, 0, 2)[None])
        v = bf(vals(True, 72, L * NKV * HD, 9).reshape(L, NKV, HD).permute(1, 0, 2)[None])
        o, p = m.eager_attention_forward(_Mod(), q, k, v, None, HD ** -0.5)
        s = torch.matmul(q, m.repeat_kv(k, NQ // NKV).transpose(2, 3)) * HD ** -0.5   # eager's own first line
        assert torch.equal(torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype), p)
        out.append(hexs(f"kAttnS{ci}", bits16(s[0, 0, 0]), 4))
        out.append(hexs(f"kAttnP{ci}", bits16(p[0, 0, 0]), 4))
        out.append(hexs(f"kAttnO{ci}", bits16(o[0, 0]), 4))
    out.append("inline constexpr const char* kAttnS[] = {kAttnS0, kAttnS1, kAttnS2};")
    out.append("inline constexpr const char* kAttnP[] = {kAttnP0, kAttnP1, kAttnP2};")
    out.append("inline constexpr const char* kAttnO[] = {kAttnO0, kAttnO1, kAttnO2};")

    # (f) silu_mul at 11776 ----------------------------------------------------------------------------------------
    g = bf(vals(False, 80, DI, 4, amp=4.0))
    u = bf(vals(False, 81, DI, 4, amp=4.0))
    out.append("// (f) silu_mul: g / u fine (t 80 / 81, seed 4, amp 4) [11776]: rne(rne(silu(g)) x u)")
    out.append(hexs("kSiluOut", bits16(F.silu(g) * u), 4))

    # (g) argmax -----------------------------------------------------------------------------------------------------
    lg = vals(False, 90, V, 4)
    lg[125016] = 50.0
    lg[127999] = 100.0
    out.append("// (g) argmax: logits fine (t 90, seed 4) [128000], logits[125016] = 50, logits[127999] = 100; torch's argmax over")
    out.append("//     the 125017 real ids")
    out.append(f"inline constexpr unsigned kArgmax = {int(torch.argmax(lg[:VU]))};")
    out.append("}  // namespace lfm2moe_fixture")
    sys.stdout.write("\n".join(out) + "\n")


def ref_version() -> str:
    import transformers
    return transformers.__version__


if __name__ == "__main__":
    main()
