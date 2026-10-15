#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The Muse-Glimmer decode kernel-chain fixture (spec 23c Task 1): transformers 5.19.0's OWN muse_glimmer modules
and functions (modeling_muse_glimmer.py - the norms, the normed embedding, the attention's qk_norm / qk scale, the
rotary module, apply_rotary_pos_emb, eager_attention_forward, the output gate's line, the MLP's act_fn line, the
four-rounding softcap's lines) run in bf16 on deterministic inputs at the REAL widths (hidden 6656, 32 q / 2 kv
heads x 128, window 2048, theta 5e5), their outputs written as a C++ header that tests/kernels/muse_ref_test.cc
holds tests/kernels/muse_ref.h to.

    tools/oracle/muse_fixture.py > tests/kernels/muse_fixture.h

Run it in the oracle container with 21a's 5.19.0 site first on PYTHONPATH (seconds, < 2 GB, no checkpoint):

    docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -w /ws -e PYTHONPATH=/ws/oracle-out-q4exp/site \\
      --entrypoint python3 agnes-ref-img:latest tools/oracle/muse_fixture.py > tests/kernels/muse_fixture.h

The modules whose weights the fixture does not need are built on `meta` (the attention, the MLP: only their
weight-less pieces are called - qk_norm, act_fn); the norms get the hash's weights. Inputs are not stored where a
hash can make them: both sides draw them from the same integer hash (lowbias32: `val()` below,
`museref::fixture_val` in C++). `coarse` values are small multiples of a power of two, so every sum below (a norm's
squares, a q.k dot) is exact in fp32 in ANY order and the comparison is a bitwise check of the rounding points
alone; `fine` values are generic bf16 (their long sums are torch's order: the test bounds and counts them). The cases:

  (a) embed     the normed embedding (MuseGlimmerTextNormedEmbedding: the bf16 row, then the weight-less
                MuseGlimmerRMSNorm with pow, eps 1e-5) of a coarse row and a fine row (rows 0 / 1 of a 2-row table)
  (b) centered  MuseGlimmerTextCenteredRMSNorm (x * rsqrt(ms + eps) * (1 + w), one rounding): coarse x at eps 1e-5
                with w = 0 and a random w; at eps 1e-8 with the random w; a fine x at 1e-5 with the random w
  (c) final     MuseGlimmerRMSNorm with weight (x * pow(ms + 1e-5, -0.5) * w, plain w): coarse and fine x
  (d) post_add  the decoder layer's `residual + post_attention_layernorm(sub)` (eps 1e-8): fine residual, coarse
                and fine sub
  (e) qk prep   one q row [32][128] and k row [2][128] (coarse, the GEMV's bf16 outputs): the attention's own
                qk_norm then `* qk_scale_factor` (q), qk_norm (k); NoPE (full layers): all 32 q heads and both k
                heads; RoPE (sliding: the rotary module's bf16 cos / sin + apply_rotary_pos_emb) at kRopePos:
                q heads 0 and 31, k heads 0 and 1; the cos / sin rows [P][64]
  (f) gate      `attn_output * torch.sigmoid(g)` over one 4096 row: fine attn, fine g (amp 8)
  (g) silu_mul  the MLP's `act_fn(g) * u` over 512: fine g (amp 8), fine u
  (h) attention eager_attention_forward at GQA 16 (32 q / 2 kv heads), coarse q (step 1/2) and key p's k / v
                (step 1/8) at index (p x 2 + j) x 128 + d: sliding at query 5 (the prompt pass's row: keys 0..7,
                6 visible), 2047 (keys 0..2047), 2048 (1..2048), 4097 (2050..4097) - the cached decode pass's rows
                - and full at 3000 (0..3000). Head 0's scores and probabilities, every head's output [32][128]
  (i) softcap   the head's four bf16 roundings on 64 raw logits (fine, amp 64): `* output_multiplier`, `/ 20`,
                tanh, `* 20`
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
M32 = 0xFFFFFFFF
D, HD, QH, KVH, I_SM = 6656, 128, 32, 2, 512
ROPE_POS = [0, 1, 2047, 2048, 4096, 9000, 100000, 131071]
ATTN_CASES = [  # name, sliding, query position, keys [lo, hi] in the row
    ("s5", True, 5, 0, 7),
    ("s2047", True, 2047, 0, 2047),
    ("s2048", True, 2048, 1, 2048),
    ("s4097", True, 4097, 2050, 4097),
    ("f3000", False, 3000, 0, 3000),
]


def lowbias32(x: int) -> int:
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    x ^= x >> 16
    return x


def val(coarse: bool, t: int, i: int, seed: int, step: float = 0.125, amp: float = 1.0) -> float:
    """museref::fixture_val's twin: element i of input tensor t. coarse: (h % 17 - 8) x step.
    fine: ((h >> 8) x 2^-23 - 1) x amp (exact in fp32), rounded to bf16."""
    h = lowbias32(lowbias32(seed * 3 + t) ^ i)
    if coarse:
        return float((h % 17) - 8) * step
    f = torch.tensor([(h >> 8) * 2.0 ** -23 - 1.0], dtype=torch.float32) * amp
    return float(f.to(torch.bfloat16).float()[0])


def vec(coarse, t, n, seed, step=0.125, amp=1.0, base=0):
    return torch.tensor([val(coarse, t, base + i, seed, step, amp) for i in range(n)], dtype=torch.float32)


def bits16(x: torch.Tensor) -> list:
    return (x.contiguous().to(torch.bfloat16).view(torch.int16).flatten().int() & 0xFFFF).tolist()


def hexs(name: str, xs: list, width: int = 4) -> str:
    h = "".join(f"{v:0{width}x}" for v in xs)
    lines = [f'    "{h[i:i + 128]}"' for i in range(0, len(h), 128)]
    return f"inline constexpr char {name}[] =\n" + "\n".join(lines) + ";\n"


def main() -> None:
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS:
        sys.exit(f"muse_fixture: transformers {transformers.__version__}; put the {PINNED_TRANSFORMERS} site first on "
                 "PYTHONPATH (tools/oracle/qwen4exp_env.sh)")
    from transformers.activations import ACT2FN  # noqa: F401  (the MLP's act_fn is ACT2FN[hidden_activation])
    from transformers.models.muse_glimmer import modeling_muse_glimmer as m
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig

    torch.manual_seed(0)
    tc = MuseGlimmerTextConfig(num_hidden_layers=4, rope_parameters={"rope_type": "default", "rope_theta": 5e5})
    tc._attn_implementation = "eager"
    assert (tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim) == (D, QH, KVH, HD)
    assert tc.layer_types[3] == "full_attention" and tc.layer_types[0] == "sliding_attention"
    assert tc.sliding_window == 2048 and tc.rms_norm_eps == 1e-5 and tc.post_norm_eps == 1e-8
    bf = torch.bfloat16
    with torch.device("meta"):
        attn = m.MuseGlimmerTextAttention(tc, 0)     # its qk_norm (weight-less), qk_scale_factor, scaling, groups
        mlp = m.MuseGlimmerTextMLP(tc)               # its act_fn
    attn.eval()

    out = [
        "#pragma once",
        "// GENERATED by tools/oracle/muse_fixture.py in agnes-ref-img:latest with transformers "
        f"{transformers.__version__} (torch {torch.__version__}, CPU capability {torch.backends.cpu.get_cpu_capability()})"
        " - do not edit.",
        "// transformers' own muse_glimmer modules (bf16) on museref::fixture_val's inputs; 4 hex digits per bf16 value.",
        "// The script's docstring has the cases.",
        "namespace muse_fixture {",
        f"inline constexpr unsigned kHidden = {D}, kHd = {HD}, kQHeads = {QH}, kKvHeads = {KVH}, kInterSmall = {I_SM};",
    ]

    # (a) the normed embedding: a 2-row table, row 0 coarse, row 1 fine
    emb = m.MuseGlimmerTextNormedEmbedding(2, D, None, tc.rms_norm_eps).to(bf)
    with torch.no_grad():
        emb.weight.copy_(torch.stack([vec(True, 0, D, 1), vec(False, 0, D, 2, amp=4.0)]).to(bf))
        e = emb(torch.tensor([[0, 1]]))[0]
    out.append("// (a) the normed embedding: row 0 coarse (t 0, seed 1, step 1/8), row 1 fine (t 0, seed 2, amp 4)")
    out.append(hexs("kEmbedCoarse", bits16(e[0])))
    out.append(hexs("kEmbedFine", bits16(e[1])))

    # (b) the centered norms
    xc = vec(True, 0, D, 1).to(bf)[None]
    xf = vec(False, 0, D, 2, amp=4.0).to(bf)[None]
    wr = vec(False, 1, D, 1, amp=0.5).to(bf)     # the checkpoint's w (the module adds 1 in fp32)

    def centered(x, w, eps):
        n = m.MuseGlimmerTextCenteredRMSNorm(D, eps=eps).to(bf)
        with torch.no_grad():
            n.weight.copy_(w)
            return n(x)[0]
    out.append("// (b) centered norm: x coarse (t 0, seed 1) / fine (t 0, seed 2, amp 4); w = 0 or random (t 1, seed 1, fine")
    out.append("//     x 0.5, bf16 - the module computes 1 + w in fp32)")
    out.append(hexs("kCenteredCoarseZero", bits16(centered(xc, torch.zeros(D, dtype=bf), 1e-5))))
    out.append(hexs("kCenteredCoarseRand", bits16(centered(xc, wr, 1e-5))))
    out.append(hexs("kCenteredCoarseRandPost", bits16(centered(xc, wr, 1e-8))))
    out.append(hexs("kCenteredFineRand", bits16(centered(xf, wr, 1e-5))))

    # (c) the final norm: plain w, pow
    wf = (1.0 + vec(False, 2, D, 1, amp=0.5)).to(bf)
    fn = m.MuseGlimmerRMSNorm(D, eps=tc.rms_norm_eps).to(bf)
    with torch.no_grad():
        fn.weight.copy_(wf)
        out.append("// (c) final norm (plain w = rf(1 + fine t 2 x 0.5)): coarse x, fine x")
        out.append(hexs("kFinalCoarse", bits16(fn(xc)[0])))
        out.append(hexs("kFinalFine", bits16(fn(xf)[0])))

    # (d) the sandwich's add: residual + post_attention_layernorm(sub)
    with torch.device("meta"):
        layer = m.MuseGlimmerTextDecoderLayer(tc, 0)
    pn = layer.post_attention_layernorm
    assert pn.eps == 1e-8
    pn.weight = torch.nn.Parameter(wr.clone(), requires_grad=False)
    resid = vec(False, 3, D, 4, amp=2.0).to(bf)[None]
    with torch.no_grad():
        pa_c = resid + pn(xc)
        pa_f = resid + pn(xf)
    out.append("// (d) post_add: residual fine (t 3, seed 4, amp 2) + post norm (eps 1e-8, w random as (b)) of coarse / fine x")
    out.append(hexs("kPostAddCoarse", bits16(pa_c[0])))
    out.append(hexs("kPostAddFine", bits16(pa_f[0])))

    # (e) q / k prep: the GEMV's bf16 q [32][128] and k [2][128] (coarse, step 1/4)
    q0 = vec(True, 4, QH * HD, 5, step=0.25).to(bf).view(1, 1, QH, HD).transpose(1, 2)   # [1, 32, 1, 128]
    k0 = vec(True, 5, KVH * HD, 5, step=0.25).to(bf).view(1, 1, KVH, HD).transpose(1, 2)
    with torch.no_grad():
        qn = attn.qk_norm(q0) * attn.qk_scale_factor          # the attention's lines, M:330-331
        kn = attn.qk_norm(k0)
        rot = m.MuseGlimmerTextRotaryEmbedding(tc)
        pos = torch.tensor([ROPE_POS])
        cos, sin = rot(q0, pos)                                # [1, P, 128] bf16
        qr, kr = [], []
        for i in range(len(ROPE_POS)):
            a, b = m.apply_rotary_pos_emb(qn, kn, cos[:, i:i + 1], sin[:, i:i + 1])
            qr.append(torch.cat([a[0, 0, 0], a[0, QH - 1, 0]]))
            kr.append(torch.cat([b[0, 0, 0], b[0, 1, 0]]))
    out.append("// (e) q (t 4, seed 5, coarse 1/4) [32][128] and k (t 5) [2][128]: NoPE q (normed x 3.87, all heads) and k;")
    out.append("//     RoPE'd at kRopePos: q heads 0 and 31, k heads 0 and 1 [P][4][128]; cos / sin [P][64]")
    out.append(f"inline constexpr unsigned kRopeN = {len(ROPE_POS)};")
    out.append("inline constexpr unsigned kRopePos[] = {" + ", ".join(map(str, ROPE_POS)) + "};")
    out.append(f"inline constexpr float kQkScale = {tc.qk_scale_factor!r}f, kRopeTheta = {tc.rope_parameters['rope_theta']!r}f;")
    out.append(hexs("kQNope", bits16(qn[0, :, 0])))
    out.append(hexs("kKNope", bits16(kn[0, :, 0])))
    out.append(hexs("kQRoped", bits16(torch.stack(qr))))
    out.append(hexs("kKRoped", bits16(torch.stack(kr))))
    out.append(hexs("kRopeCos", bits16(cos[0, :, :HD // 2])))
    out.append(hexs("kRopeSin", bits16(sin[0, :, :HD // 2])))

    # (f) the output gate: attn_output * torch.sigmoid(gate_proj(x)) - M:358's line
    ga = vec(False, 6, QH * HD, 7, amp=2.0).to(bf)[None]
    gg = vec(False, 7, QH * HD, 7, amp=8.0).to(bf)[None]
    with torch.no_grad():
        gated = ga * torch.sigmoid(gg)
    out.append("// (f) gate: attn fine (t 6, seed 7, amp 2), g fine (t 7, amp 8): attn * sigmoid(g), bf16 [4096]")
    out.append(hexs("kGate", bits16(gated[0])))

    # (g) SiLU x up: the MLP's act_fn(gate_proj(x)) * up_proj(x) - M:154's line
    sg = vec(False, 8, I_SM, 9, amp=8.0).to(bf)[None]
    su = vec(False, 9, I_SM, 9, amp=2.0).to(bf)[None]
    with torch.no_grad():
        sm = mlp.act_fn(sg) * su
    out.append(f"// (g) silu_mul: g fine (t 8, seed 9, amp 8), u fine (t 9, amp 2): act_fn(g) * u, bf16 [{I_SM}]")
    out.append(hexs("kSiluMul", bits16(sm[0])))

    # (h) eager attention at GQA 16
    out.append("// (h) eager attention: 32 q heads / 2 kv heads; q coarse (t 20, seed = case, step 1/2) [32][128], key p's k / v")
    out.append("//     coarse (t 21 / 22, seed 9, step 1/8) at index (p x 2 + j) x 128 + d. Head 0's scores s and probabilities p")
    out.append("//     over the row's visible keys (bf16), every head's output o [32][128] (bf16; before the gate)")
    out.append(f"inline constexpr unsigned kAttnCases = {len(ATTN_CASES)};")
    meta = []
    for ci, (name, sliding, qp, lo, hi) in enumerate(ATTN_CASES):
        L = hi - lo + 1
        q = vec(True, 20, QH * HD, ci, step=0.5).reshape(1, QH, 1, HD).to(bf)
        k = torch.stack([vec(True, 21, KVH * HD, 9, base=p * KVH * HD).view(KVH, HD) for p in range(lo, hi + 1)], dim=1)
        v = torch.stack([vec(True, 22, KVH * HD, 9, base=p * KVH * HD).view(KVH, HD) for p in range(lo, hi + 1)], dim=1)
        k, v = k[None].to(bf), v[None].to(bf)                   # [1, 2, L, 128]
        kidx = torch.arange(lo, hi + 1)
        ok = (kidx <= qp) & ((kidx > qp - tc.sliding_window) if sliding else True)
        mask = torch.where(ok, 0.0, torch.finfo(bf).min).to(bf).view(1, 1, 1, L)
        with torch.no_grad():
            o, w = m.eager_attention_forward(attn, q, k, v, mask, scaling=attn.scaling)   # o [1, 1, 32, 128]
            # the function's own lines, for the scores (held to its probabilities)
            kk = m.repeat_kv(k, attn.num_key_value_groups)
            s = torch.matmul(q, kk.transpose(2, 3)) * attn.scaling + mask
            p = torch.nn.functional.softmax(s, dim=-1, dtype=torch.float32).to(bf)
        assert torch.equal(p, w), name
        nvis = int(ok.sum())
        assert bool(ok[:nvis].all())
        meta.append((1 if sliding else 0, qp, lo, hi, nvis))
        out.append(f"// {name}: query {qp}, keys [{lo}, {hi}] ({L} in the row, {nvis} visible)")
        out.append(hexs(f"kAttnS_{name}", bits16(s[0, 0, 0, :nvis])))
        out.append(hexs(f"kAttnP_{name}", bits16(p[0, 0, 0, :nvis])))
        out.append(hexs(f"kAttnO_{name}", bits16(o[0, 0])))
    out.append("struct AttnCase { unsigned sliding, query, lo, hi, visible; const char* s; const char* p; const char* o; };")
    out.append("inline constexpr AttnCase kAttn[] = {")
    for (name, *_), mt in zip(ATTN_CASES, meta):
        out.append(f"    {{{', '.join(map(str, mt))}, kAttnS_{name}, kAttnP_{name}, kAttnO_{name}}},")
    out.append("};")

    # (i) the softcap's four bf16 roundings (M:1178-1181)
    z = vec(False, 10, 64, 11, amp=64.0).to(bf)[None]
    with torch.no_grad():
        c = z * tc.output_multiplier
        c = c / tc.final_logit_softcapping
        c = torch.tanh(c)
        c = c * tc.final_logit_softcapping
    out.append("// (i) softcap: raw logits fine (t 10, seed 11, amp 64) [64] -> the four-rounding capped row (bf16)")
    out.append(hexs("kSoftcap", bits16(c[0])))
    out.append("}  // namespace muse_fixture")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
