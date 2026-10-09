#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next kernel-chain fixture (spec 21c Task 1 Step 3): tools/oracle/qwen4exp_ref.py's OWN
restated ops (and transformers 5.19.0's qwen4_exp modules where the restated set has no twin) run in bf16 on
deterministic inputs at the real widths, their outputs written as a C++ header that
tests/kernels/qwen4exp_ref_test.cc holds tests/kernels/qwen4exp_ref.h to.

    tools/oracle/qwen4exp_fixture.py > tests/kernels/qwen4exp_fixture.h

Run it in the oracle container with 21a's 5.19.0 site first on PYTHONPATH (seconds, < 2 GB, no checkpoint):

    docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -w /ws -e PYTHONPATH=/ws/oracle-out-q4exp/site \\
      --entrypoint python3 agnes-ref-img:latest tools/oracle/qwen4exp_fixture.py > tests/kernels/qwen4exp_fixture.h

Inputs are not stored where a hash can make them: both sides draw them from the same integer hash (lowbias32:
`val()` below, `q4ref::fixture_val` in C++). `coarse` values are small multiples of a power of two, so a sum
over them (a norm's squares, a dot of coarse vectors) is exact in fp32 in ANY order and the comparison is of
the rounding points alone; `fine` values are generic bf16 (their long sums are torch's order: the test holds
those to one bf16 ulp and counts). The cases:

  (a) HC      one gated residual at the real widths: H coarse [10240], the (1 + w) norm, down||inject {324,
              10240} and up {10240, 320} fine weights -> xn, the linears' bf16 outputs, a, g, the block input x,
              inj; then the combine H + y (x) inj with a coarse y
  (b) PLE     ids: 64 positions with EOS at 5, 6 and 20 and the last id 248319 at 0 and 30, the real table's
              constants (base 20,000,000) -> [64][16] row ids (exact integers); the block: 10 positions of a
              sequence through the restated layer (held bitwise to transformers' Qwen4ExpTextPLELayer on a
              narrow config first) with kv and H from the hash -> H after the block, per position
  (c) indexer q heads normed + roped at 0, 2047, 2050, 2051, 9000 (coarse q, transformers' bf16 cos / sin);
              compressed keys of the blocks completing at 3, 2047, 9003 (qwen4exp_ref.block_keys)
  (d) select  (d1) a row's scores over 600 coarse block keys (exact); (d2) crafted score rows at 512 / 513 /
              600 complete blocks with exact ties planted at the 512 / 513 cut and a cut inside relu's zeros:
              torch's topk selection and the ruled one (score desc, block asc)
  (e) route   4 router rows at 512 / top-10: random, a tie at the 10th / 11th, every logit -1e30 but 10, a
              wide one; torch's ids (topk order), weights, gap and the ruled ids
  (f) combine 10 routed expert rows + the shared one through grouped_mm's combine (route row 0)
  (g) gated   the GDN gated norm with the sigmoid gate (qwen4exp_ref.gdn_gated_norm), 48 heads
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen4exp_ref", os.path.join(_HERE, "qwen4exp_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["qwen4exp_ref"] = ref
_spec.loader.exec_module(ref)

D, HC, HCN, LOW, INJ = 2560, 4, 10240, 320, 4
EPS = 1e-6
VOCAB, EOS = 248320, 248044
BF = torch.bfloat16
torch.manual_seed(0)


# --- the hash (q4ref::fixture_val's twin) ------------------------------------------------------------------------
def lowbias32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x = x * np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x = x * np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def val(coarse: bool, t: int, n: int, seed: int, step: float = 0.125, amp: float = 1.0, base: int = 0) -> torch.Tensor:
    """Elements base .. base + n of input tensor t, fp32 tensor of bf16-exact values. coarse: (h % 17 - 8) x step;
    fine: rf(((h >> 8) x 2^-23 - 1) x amp) (fp32 arithmetic, then round to bf16)."""
    with np.errstate(over="ignore"):
        s = lowbias32(np.array([seed * 3 + t], dtype=np.uint32))[0]
        h = lowbias32(np.arange(base, base + n, dtype=np.uint64).astype(np.uint32) ^ s)
    if coarse:
        return torch.from_numpy(((h % 17).astype(np.int32) - 8).astype(np.float32) * np.float32(step))
    f = ((h >> 8).astype(np.float32) * np.float32(2.0 ** -23) - np.float32(1.0)) * np.float32(amp)
    return torch.from_numpy(f).to(BF).float()


def hval(t: int, n: int, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        s = lowbias32(np.array([seed * 3 + t], dtype=np.uint32))[0]
        return lowbias32(np.arange(n, dtype=np.uint64).astype(np.uint32) ^ s)


# --- the header's arrays -------------------------------------------------------------------------------------------
def bits16(x: torch.Tensor) -> list:
    return (x.contiguous().to(BF).view(torch.int16).flatten().int() & 0xFFFF).tolist()


def bits32(x: torch.Tensor) -> list:
    return (x.contiguous().float().view(torch.int32).flatten().long() & 0xFFFFFFFF).tolist()


def hexs(name: str, xs: list, width: int) -> str:
    h = "".join(f"{int(v):0{width}x}" for v in xs)
    lines = [f'    "{h[i:i + 128]}"' for i in range(0, len(h), 128)]
    return f"inline constexpr char {name}[] =\n" + "\n".join(lines) + ";\n"


def u32s(name: str, xs: list) -> str:
    return f"inline constexpr unsigned {name}[] = {{" + ", ".join(str(int(v)) for v in xs) + "};"


# --- (a) hyper-connections -----------------------------------------------------------------------------------------
def case_hc(out: list) -> None:
    H = val(True, 0, HCN, 1).to(BF)
    w = val(False, 1, HCN, 1, amp=0.5).to(BF)                             # the checkpoint's hc_norm.weight
    down = val(False, 2, (LOW + INJ) * HCN, 1, amp=1 / 32).to(BF).view(LOW + INJ, HCN)
    up = val(False, 3, HCN * LOW, 1, amp=1 / 16).to(BF).view(HCN, LOW)
    y = val(True, 4, D, 1, step=0.25).to(BF)
    mixed, H0, inj = ref.gated_residual(H[None], w, down[:LOW], up, down[LOW:], HC, D, EPS)
    # the same chain line by line (M:1013-1023), for its intermediates - held to the restated op's outputs
    xn = ref.rms_norm(H[None], w, EPS, group=D)
    dlin = torch.cat([F.linear(xn, down[:LOW]), F.linear(xn, down[LOW:])], -1)   # down's 320 rows, inject's 4: bf16
    a = F.silu(dlin[..., :LOW] / HC)
    ulin = F.linear(a, up)
    g = torch.sigmoid(ulin)
    x = (g.unflatten(-1, (HC, D)) * xn.unflatten(-1, (HC, D))).mean(dim=-2)
    inj2 = 2 * torch.sigmoid(dlin[..., LOW:] / HC)
    assert torch.equal(x, mixed) and torch.equal(inj2, inj) and torch.equal(H0, H[None])
    H1 = ref.hc_combine(H[None], y[None], inj)
    out.append("// (a) HC: H coarse (t 0, seed 1), hc_norm w fine (t 1, amp 1/2), down||inject [324][10240] fine (t 2, amp")
    out.append("//     1/32, rows 320..323 = block_inject), up [10240][320] fine (t 3, amp 1/16), y coarse (t 4, step 1/4)")
    out.append(hexs("kHcXn", bits16(xn[0]), 4))
    out.append(hexs("kHcDown", bits16(dlin[0]), 4))                      # [324] bf16: the linears' outputs
    out.append(hexs("kHcA", bits16(a[0]), 4))
    out.append(hexs("kHcUp", bits16(ulin[0]), 4))
    out.append(hexs("kHcG", bits16(g[0]), 4))
    out.append(hexs("kHcX", bits16(x[0]), 4))
    out.append(hexs("kHcInj", bits16(inj[0]), 4))
    out.append(hexs("kHcH1", bits16(H1[0]), 4))


# --- (b) PLE -------------------------------------------------------------------------------------------------------
PLE_SEQ = 64
PLE_BLOCK_POS = 10
PLE_DIM = 160


def ple_constants(base: int):
    tc = {"ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": base, "make_ngram_vocab_size_divisible_by": 128}
    sizes, offsets, _total, _padded = ref.facts.head_table(tc, 0)
    mult = ref.facts.layer_multipliers(VOCAB, 3, 0, 1234)
    return mult, sizes, offsets


def ple_seq() -> list:
    h = hval(10, PLE_SEQ, 2)
    seq = [int(v % VOCAB) for v in h]
    for p in (5, 6, 20):
        seq[p] = EOS
    seq[0] = seq[30] = VOCAB - 1
    return seq


def ple_restated(m, H, e, key_fn=None, value_fn=None):
    """Qwen4ExpTextPLELayer.forward (M:1227-1247) without a cache, line by line over the module's own parts:
    H [1][T][10240] bf16, e [1][T][2560] the n-gram rows -> H + out (the decoder layer's add). key_fn /
    value_fn replace key_proj / value_proj (the fixture feeds the projections' rows directly)."""
    key = (key_fn or m.key_proj)(e)
    value = (value_fn or m.value_proj)(e)
    key_normed = m.norm_key(key).unflatten(-1, (HC, -1))
    query_normed = m.norm_query(H).unflatten(-1, (HC, -1))
    hidden = key_normed.shape[-1]
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / (hidden ** 0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = m.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    out = gated_value + m._short_conv(gated_value_normed, None)
    return H + out


def ple_check_module() -> None:
    """The restatement against transformers' own module, bitwise, on a narrow config (hidden 64, 4 streams)."""
    mq = ref.mq()
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    cfg = Qwen4ExpTextConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=32, linear_num_value_heads=2, linear_num_key_heads=1,
                             hc_count=HC, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32, ngram_vocab_size_base=1000,
                             eos_token_id=EOS, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
                             indexer_budget=16, indexer_compress_ratio=4, output_gate_type="sigmoid",
                             layer_types=["linear_attention"] * 3 + ["full_attention"])
    m = mq.Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0).to(BF)
    with torch.no_grad():
        for i, p in enumerate(m.parameters()):
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(100 + i)).to(BF) * 0.3)
        T = 12
        ids = torch.tensor([ple_seq()[:T]])
        H = (torch.randn(1, T, 4 * 64, generator=torch.Generator().manual_seed(7))).to(BF)
        want = H + m(H, ids, None)
        e = m.ple_embedding(ids, None)
        got = ple_restated(m, H, e)
    assert torch.equal(want, got), "the restated PLE block is not transformers' Qwen4ExpTextPLELayer"


class PleParts(torch.nn.Module):
    """The PLE layer's parts at the real widths (Qwen4ExpTextPLELayer's own classes)."""

    def __init__(self):
        super().__init__()
        mq = ref.mq()
        self.norm_key = mq.Qwen4ExpTextRMSNorm(HCN, group_size=D, eps=EPS)
        self.norm_query = mq.Qwen4ExpTextRMSNorm(HCN, group_size=D, eps=EPS)
        self.norm_conv = mq.Qwen4ExpTextRMSNorm(HCN, group_size=D, eps=EPS)
        self.conv1d = torch.nn.Conv1d(HCN, HCN, kernel_size=4, groups=HCN, dilation=3, bias=False)
        self.short_conv_state_len = 9

    _short_conv = None


def case_ple(out: list) -> None:
    mult, sizes, offsets = ple_constants(20_000_000)
    seq = ple_seq()
    ids = ref.ple_hash_ids(torch.tensor(seq), mult, sizes, offsets, EOS)
    out.append("// (b) PLE ids: the 64-id sequence (hash t 10, seed 2, mod vocab; EOS 248044 at 5, 6, 20; 248319 at 0, 30)")
    out.append("//     through qwen4exp_ref.ple_hash_ids with the real table's constants -> [64][16] global row ids")
    out.append(f"inline constexpr unsigned kPleSeqN = {PLE_SEQ};")
    out.append(u32s("kPleSeq", seq))
    out.append("inline constexpr unsigned long long kPleMult[] = {" + ", ".join(f"{v}ull" for v in mult) + "};")
    out.append("inline constexpr unsigned long long kPleSizes[] = {" + ", ".join(f"{v}ull" for v in sizes) + "};")
    out.append("inline constexpr unsigned long long kPleOffsets[] = {" + ", ".join(f"{v}ull" for v in offsets) + "};")
    out.append(hexs("kPleIds", [int(v) for v in ids.flatten().tolist()], 16))

    # the block: the restated layer, checked against transformers' module first
    ple_check_module()
    mq = ref.mq()
    parts = PleParts().to(BF)
    parts._short_conv = types.MethodType(mq.Qwen4ExpTextPLELayer._short_conv, parts)
    with torch.no_grad():
        parts.norm_key.weight.copy_(val(False, 20, HCN, 3, amp=0.5))
        parts.norm_query.weight.copy_(val(False, 21, HCN, 3, amp=0.5))
        parts.norm_conv.weight.copy_(val(False, 22, HCN, 3, amp=0.5))
        parts.conv1d.weight.copy_(val(False, 23, HCN * 4, 3, amp=0.5).view(HCN, 1, 4))
    T = PLE_BLOCK_POS
    kv = torch.stack([val(False, 24, HCN + D, 3, amp=2.0, base=p * (HCN + D)) for p in range(T)])[None].to(BF)
    H = torch.stack([val(True, 25, HCN, 3, base=p * HCN) for p in range(T)])[None].to(BF)
    with torch.no_grad():
        Hn = ple_restated(parts, H, kv, key_fn=lambda e: e[..., :HCN], value_fn=lambda e: e[..., HCN:])
    out.append(f"// (b) PLE block: {T} positions; kv fine (t 24, seed 3, amp 2, row p at base p x 12800: the key||value")
    out.append("//     GEMV's bf16 row), H coarse (t 25, seed 3, row p at base p x 10240), norm_key / norm_query / norm_conv")
    out.append("//     (1 + w) fine (t 20 / 21 / 22, amp 1/2), conv taps fine [10240][4] (t 23, amp 1/2) -> H + ple(H) per row")
    out.append(f"inline constexpr unsigned kPleBlockPos = {T};")
    out.append(hexs("kPleH", bits16(Hn[0]), 4))


# --- (c) the indexer, (d) the selection ---------------------------------------------------------------------------------
IDX_POS = [0, 2047, 2050, 2051, 9000]
KEY_BLOCKS = [0, 511, 2250]          # complete at p = 3, 2047, 9003; RoPE at 0, 2044, 9000


def rope_rows(positions: list) -> tuple:
    """transformers' cos / sin (bf16 [P][64]) at text positions (three equal ids: plain 1D RoPE)."""
    mq = ref.mq()
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    cfg = Qwen4ExpTextConfig(hidden_size=2560, head_dim=256, num_attention_heads=24, num_key_value_heads=2,
                             rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                              "mrope_section": [11, 11, 10]},
                             layer_types=["linear_attention"] * 3 + ["full_attention"], num_hidden_layers=4,
                             ple_layer_ids=[2], ple_embed_dim=2560, indexer_n_heads=4, indexer_kv_heads=1,
                             indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
                             output_gate_type="sigmoid", eos_token_id=EOS)
    rot = mq.Qwen4ExpTextRotaryEmbedding(cfg)
    pos = torch.tensor([positions])
    cos, sin = rot(torch.zeros(1, dtype=BF), pos)
    return cos[0], sin[0]


def case_indexer(out: list) -> None:
    mq = ref.mq()
    qn = mq.Qwen4ExpTextRMSNorm(128, eps=EPS).to(BF)
    kn = mq.Qwen4ExpTextRMSNorm(128, eps=EPS).to(BF)
    with torch.no_grad():
        qn.weight.copy_(val(False, 30, 128, 4, amp=0.5))
        kn.weight.copy_(val(False, 31, 128, 4, amp=0.5))
    cos, sin = rope_rows(IDX_POS)
    qs = []
    for i, p in enumerate(IDX_POS):
        q = val(True, 32, 4 * 128, 4, step=0.25, base=i * 512).to(BF).view(1, 1, 4, 128)
        qq = mq.apply_rotary_pos_emb(qn(q), cos=cos[i:i + 1][None], sin=sin[i:i + 1][None], unsqueeze_dim=2)
        qs.append(qq.flatten())
    out.append("// (c) indexer q: row i's q heads coarse (t 32, seed 4, step 1/4, base i x 512) at kIdxPos, q_layernorm w")
    out.append("//     fine (t 30, amp 1/2), transformers' bf16 cos / sin [P][64] -> the roped heads [P][4][128]")
    out.append(f"inline constexpr unsigned kIdxPosN = {len(IDX_POS)};")
    out.append(u32s("kIdxPos", IDX_POS))
    out.append(hexs("kIdxCos", bits16(cos[:, :64]), 4))
    out.append(hexs("kIdxSin", bits16(sin[:, :64]), 4))
    out.append(hexs("kIdxQ", bits16(torch.stack(qs)), 4))
    starts = [4 * b for b in KEY_BLOCKS]
    kc, ks = rope_rows(starts)
    keys = []
    ix = types.SimpleNamespace(compress_ratio=4, index_head_dim=128, k_layernorm=kn)
    for i, b in enumerate(KEY_BLOCKS):
        raw = val(True, 33, 4 * 128, 4, step=0.25, base=i * 512).to(BF).view(4, 128)
        keys.append(ref.block_keys(ix, raw, kc[i:i + 1], ks[i:i + 1])[0])
    out.append("// (c) compressed keys: block b's 4 raw keys coarse (t 33, seed 4, step 1/4, base i x 512), k_layernorm w")
    out.append("//     fine (t 31, amp 1/2), RoPE at 4b (cos / sin rows below) -> [3][128]")
    out.append(u32s("kKeyBlocks", KEY_BLOCKS))
    out.append(hexs("kKeyCos", bits16(kc[:, :64]), 4))
    out.append(hexs("kKeySin", bits16(ks[:, :64]), 4))
    out.append(hexs("kKeys", bits16(torch.stack(keys)), 4))


SEL_N = 600


def ruled_top(scores: torch.Tensor, k: int) -> list:
    order = sorted(range(scores.numel()), key=lambda b: (-float(scores[b]), b))
    return sorted(order[:k])


def case_select(out: list) -> None:
    # (d1) scores over 600 coarse keys (M:744-747's lines)
    q = val(True, 40, 4 * 128, 5, step=0.25).view(4, 128)
    keys = val(True, 41, SEL_N * 128, 5, step=0.25).view(SEL_N, 128)
    scores = torch.matmul(q.to(BF).float(), keys.to(BF).float().transpose(-1, -2)).transpose(-1, -2)
    scores = torch.relu(scores).sum(dim=-1) / (128 ** 0.5)
    out.append("// (d1) one row's scores: q [4][128] coarse (t 40, seed 5, step 1/4), 600 block keys coarse (t 41) -> fp32 [600]")
    out.append(f"inline constexpr unsigned kSelN = {SEL_N};")
    out.append(hexs("kSelScores", bits32(scores), 8))
    # (d2) crafted rows: n complete blocks, an exact tie planted across the 512 / 513 cut, a cut inside zeros
    g = torch.Generator().manual_seed(11)
    rows = []
    for n, kind in ((513, "tie"), (600, "tie"), (600, "zeros"), (512, "all")):
        s = torch.rand(n, generator=g) * 4 + 1
        if kind == "tie":   # ranks 508..513 made equal: the 512 / 513 cut falls inside the six
            order = torch.argsort(s, descending=True)
            tied = order[508:514]
            s[tied] = float(s[order[510]])
        elif kind == "zeros":   # 500 positive, the rest exactly 0: the 512th / 513th are 0
            s[torch.randperm(n, generator=g)[:n - 500]] = 0.0
        rows.append(s.float())
    out.append("// (d2) crafted score rows: 513 blocks with 6 equal scores across the cut, 600 likewise, 600 with only 500")
    out.append("//     positive (the cut inside zeros), 512 (no cut); torch's topk(min(512, n)) blocks and the ruled ones,")
    out.append("//     both ascending; -1 pads")
    out.append(u32s("kSelRowsN", [r.numel() for r in rows]))
    flat = torch.cat([torch.cat([r, torch.full((SEL_N - r.numel(),), -1.0)]) for r in rows])
    out.append(hexs("kSelRows", bits32(flat), 8))
    torch_sel, ruled = [], []
    for r in rows:
        k = min(512, r.numel())
        t = sorted(r.topk(k).indices.tolist())
        torch_sel += t + [0xFFFFFFFF] * (512 - k)
        ruled += ruled_top(r, k) + [0xFFFFFFFF] * (512 - k)
    out.append(hexs("kSelTorch", torch_sel, 8))
    out.append(hexs("kSelRuled", ruled, 8))


# --- (e) the route, (f) the combine, (g) the gated norm ------------------------------------------------------------------
E, K, RN = 512, 10, 528


def case_route(out: list) -> None:
    g = torch.Generator().manual_seed(20)
    L = torch.zeros(4, RN)
    L[0, :E] = torch.randn(E, generator=g) * 2.0
    L[1, :E] = torch.randn(E, generator=g) * 0.5 - 2.0                   # nine clear winners, a tie for the 10th
    for i, e in enumerate((3, 40, 77, 120, 200, 260, 301, 380, 450)):
        L[1, e] = 3.0 + 0.25 * i
    L[1, 99] = L[1, 431] = 2.5
    L[2, :E] = -1e30                                                     # every expert -1e30 but 10
    for i, e in enumerate(range(5, 505, 50)):
        L[2, e] = 0.125 * i
    L[3, :E] = torch.randn(E, generator=g) * 8.0
    L[:, E] = torch.tensor([0.5, -1.25, 3.0, -0.375])                   # the shared expert's gate logit
    L = L.to(BF).float()
    ids, ws, gaps, ruled, sgs = [], [], [], [], []
    for r in range(4):
        lg = L[r:r + 1, :E].to(BF)
        p = F.softmax(lg, dtype=torch.float, dim=-1)                     # M:964
        v, i = torch.topk(p, K, dim=-1)
        v = (v / v.sum(dim=-1, keepdim=True)).to(BF)
        ps = p[0].sort(descending=True).values
        ids += i[0].tolist()
        ws += bits32(v[0].float())
        gaps += bits32(ps[K - 1:K] - ps[K:K + 1])
        ruled += sorted(range(E), key=lambda e: (-float(p[0, e]), e))[:K]
        sgs += bits16(torch.sigmoid(L[r, E].to(BF)).view(1))
    out.append("// (e) route: logits [4][528] (bf16 values; column 512 the shared gate): random x2; nine winners and a")
    out.append("//     tie (99, 431) for the 10th; every expert -1e30 but ten; random x8 -> torch's ids (topk order),")
    out.append("//     weights (bf16 as fp32), gap p10 - p11, the ruled ids (p desc, id asc), the shared gate (bf16)")
    out.append(hexs("kRouteLogits", bits32(L), 8))
    out.append(u32s("kRouteTorchIds", ids))
    out.append(hexs("kRouteW", ws, 8))
    out.append(hexs("kRouteGap", gaps, 8))
    out.append(u32s("kRouteRuledIds", ruled))
    out.append(hexs("kRouteSg", sgs, 4))


def case_combine(out: list) -> None:
    # grouped_mm's combine (integrations/moe.py:470-488) + the shared expert (M:981-992), the route of row 0
    w = val(False, 50, K, 6, amp=0.5).abs().to(BF)                       # the routed weights (bf16)
    sg = torch.tensor([0.40625]).to(BF)
    down = torch.stack([val(False, 51 + j, D, 6, amp=4.0) for j in range(K + 1)])   # fp32 down sums (bf16 values)
    proj_out = down[:K].to(BF)                                           # each expert's down output, bf16
    weighted = proj_out * w.unsqueeze(-1)                                # proj_out * sample_weights: bf16
    routed = weighted.view(1, K, D).sum(dim=1).to(BF)                    # fp32 accumulation, slot order, one rounding
    shared = sg * down[K:K + 1].to(BF)                                  # sigmoid(gate) x the shared expert: bf16
    y = routed + shared
    out.append("// (f) combine: w [10] |fine| (t 50, seed 6, amp 1/2), sg 0.40625, down sums fine (t 51 + j, amp 4; the")
    out.append("//     shared expert's at j = 10) -> y [2560] = grouped_mm's sum + sg x shared")
    out.append(hexs("kCombineW", bits16(w), 4))
    out.append(hexs("kCombineY", bits16(y[0]), 4))


def case_gated(out: list) -> None:
    x = val(True, 60, 48 * 128, 7, step=0.25).view(48, 128).to(BF)
    z = val(False, 61, 48 * 128, 7, amp=6.0).view(48, 128).to(BF)
    w = val(False, 62, 128, 7, amp=1.5).to(BF)
    o = ref.gdn_gated_norm(x, z, w, EPS, act="sigmoid")
    out.append("// (g) gated norm (sigmoid): o coarse [48][128] (t 60, seed 7, step 1/4), z fine (t 61, amp 6), w fine")
    out.append("//     (t 62, amp 1.5) -> [48][128]")
    out.append(hexs("kGated", bits16(o), 4))


def main() -> None:
    import transformers
    out = [
        "#pragma once",
        "// GENERATED by tools/oracle/qwen4exp_fixture.py in agnes-ref-img:latest with transformers "
        f"{transformers.__version__} (torch {torch.__version__}, CPU capability {torch.backends.cpu.get_cpu_capability()})"
        " - do not edit.",
        "// tools/oracle/qwen4exp_ref.py's restated ops (and transformers' qwen4_exp modules) on q4ref::fixture_val's",
        "// inputs; 4 hex digits per bf16 value, 8 per fp32 / u32, 16 per u64. The script's docstring has the cases.",
        "namespace qwen4exp_fixture {",
    ]
    ref.require_version()
    with torch.no_grad():
        case_hc(out)
        case_ple(out)
        case_indexer(out)
        case_select(out)
        case_route(out)
        case_combine(out)
        case_gated(out)
    out.append("}  // namespace qwen4exp_fixture")
    print("\n".join(out))


if __name__ == "__main__":
    main()
