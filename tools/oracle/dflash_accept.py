#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Teacher-forced DFlash acceptance on tap dumps (spec 19a Task 3, Review Focus 2-5).

    dflash_accept.py run --dumps DIR --out-dir OUT [--arms bf16,int8,int8h,w4a16]
                     [--k 1-7] [--batch 32] [--drafter SNAP] [--w4a16 SNAP] [--target SNAP]
                     [--sources GLOB] [--dry-run]
    dflash_accept.py summary OUT/accept.*.json [--md OUT.md]
    dflash_accept.py drafts --dump F --target SNAP [--drafter SNAP] [--anchors 32] --out F.txt
                                                 Inkling: the first anchors' reference drafts (the engine's form:
                                                 bf16 drafter, int8 head) - inkling2_dflash_draft_test's expectation

**Protocol** (spec 19 §5's loop, teacher-forced; mtp_accept.py is the precedent). A dump
(dump_taps.py) holds ids x[0..N-1] = prompt + recorded greedy continuation, the taps after the
target layers, and g[t] = the bf16 target's argmax after position t. For every anchor p with
n_prompt <= p <= N - 7 (the same anchors for every K and arm):
  - the drafter's context is the taps of positions < p (dflash_ref.context_kv; the sliding
    window of 2048 is applied in the attention), the anchor is x[p] at position p, and one block
    of 1 + K rows drafts d_0..d_{K-1} for positions p + 1 .. p + K (K = 1..7, one block each:
    the block is non-causal, so K = 3's drafts are not K = 7's first three);
  - the verify of [x[p], d_0..d_{K-1}] keeps the longest prefix with d_j == g[p + j]. Depth j
    is only known when the text agrees with the drafts before it (x[p + 1 + i] == g[p + i],
    i < j: the teacher-forced target saw x, a live run would have seen the drafts); a row where
    that fails is CENSORED at j (counted up to j, not in the length histogram).
Per corpus (the source name's part before '/'), arm and K: alpha_j = accepted / counted at depth
j (conditional on depths < j accepted), E_K = 1 + sum_i prod_{j<=i} alpha_j tokens per verify
(the target's own token included; censoring-safe), the acceptance-length histogram over the
uncensored rows and their mean + 1, and how often the recorded text equals the bf16 greedy.

**Arms** (decision 2; the target's embedding is shared, bf16):
  bf16    the drafter as shipped (z-lab/Qwen3.8-27B-DFlash2), float32 compute; head bf16
  int8    every drafter linear (fc, q/k/v/o, gate/up/down, the conv kernel projections, the
          selector's hidden projection) int8 per output channel, symmetric RTN (spec 9's
          gemv_i8w form: s = amax / 127, q = rne(w / s) in [-127, 127]), simulated in float32;
          norms, conv base kernels and the selector codebooks stay bf16; head bf16
  int8h   int8 plus the target's lm_head int8 per row the same way (the engine's int8 head,
          spec 9 - what a deployed drafter would read)
  w4a16   syvai/Qwen3.8-27B-DFlash2-W4A16 (compressed-tensors int4 g128, dequantised exactly by
          dflash_ref.read_drafter_tensors); head bf16
**Batched.** Anchors go through the drafter B at a time (rows stacked as B blocks of 1 + K;
the conv's look-back is per block, `row % (1 + K)`; the attention mask gives each block the
context < its anchor plus its own rows); the head runs once per batch over all B x K mask rows
(the README's warning: per-call head upcasts are slow); test_dflash_accept.py checks the batch
against dflash_ref.Drafter.draft_block row by row. Greedy (T = 0) only.

**Inkling's form** (the target a looped Inkling 2 Mini: its config's text_config has a loop_config; drafter
incoai/Inkling-2-Mini-DFlash2; dumps from `b70-decode --dump-taps`, the engine's own tap rows): dflash_ref's Inkling
form (embed_norm on the block rows, the softcap after the correction, candidates over the 200,058 live rows), and one
whole 8-row block an anchor - SGLang and the engine always draft 7, so E_K for K < 7 reads the block's first K drafts.
Arms there: bf16 (head bf16, the release's), bf16h8 (head int8, ours), int8 / int8h, int4g64h / int4g128h (RTN int4,
s = amax / 7 a group, head int8: the round-to-nearest int4 form, which loses acceptance, and its g64 twin), and on int8h the 2-bit candidate head
(`int8h-c64` / `-c80`: the int8 head's 2-bit copy, loader::dflash_candidate_head2's host twin, picks C candidates whose
exact int8 logits the top 16 come from). Besides greedy E_K, per arm: T 1.0 along the greedy text under the point-mass
rule (a draft accepted with the target's p of it) and the exact rule (sum_c min(p, q), q the selector's softmax of the
realized 16 scores), and GTAU (verify up to the first draft where the walk's product of selector probabilities at T 1
falls under tau) priced on the one-card verify costs (--costs) against the best fixed K; and per anchor, in text
order, each draft's q at T 1, the greedy accepted length and the censoring (`_extra.anchors`: what a stop rule is priced
on offline).

**Qwen3.8-Flash-Next's form** (the target's config model_type qwen4_exp; drafter incoai/Qwen3.8-Flash-Next-DFlash2;
dumps from `b70-decode --dump-taps`, each tap the mean of the four hyper-connection streams after layers 5 / 15 / 25 /
35 / 45): vLLM's DFlash 2 form (the target's plain embedding, the softcap off, candidates over the 248,077 live rows)
with Inkling's whole block an anchor and its arms; T 1.0 and GTAU over the drafts the verify list holds (one fewer than
--costs' rows: Flash-Next verifies 4 rows, so K 3), priced on the pp verify costs.

Resumable: OUT/accept.<arm>.json is rewritten after every source and a re-run skips the sources
it already holds. --dry-run prints the planned anchors, the FLOP and a memory estimate.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

KMAX = 7
VOCAB_USED = 248077
INKLING_VOCAB_USED = 200058
TAUS = (0.1, 0.25, 0.5)
# T(1..8): one card's verify of 1..8 rows in plain steps at 4k (kOneCardCost, runtime/inkling2/inkling2_window.h)
INKLING_COSTS = "1,1.182,1.380,1.534,1.731,1.870,2.008,2.218"
# T(1..4): Qwen3.8-Flash-Next's verify of 1..4 rows under --parallel pp in plain steps (MtpCost::flash_next)
FLASH_NEXT_COSTS = "1,1.80,2.10,2.30"
ASSUMED_GFLOPS = 300.0
LINEAR_SUFFIXES = ("fc.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                   "kernel_projection.weight", "hidden_projection.weight")

torch = ref = dt = None


def _load_deps():
    global torch, ref, dt
    if torch is not None:
        return
    import torch as _t
    torch = _t

    def load(name):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod                    # dataclasses resolve annotations through it
        spec.loader.exec_module(mod)
        return mod
    ref = load("dflash_ref")
    dt = load("dump_taps")


# ------------------------------------------------------------------------------ arms


def rtn_int8_rows(w):
    """Symmetric int8 per output row: s = amax / 127, q = rne(w / s) clamped to [-127, 127].
    Returns (q int8, s float32 [rows]); a zero row gets s = 0 and q = 0."""
    wf = w.float()
    s = wf.abs().amax(dim=1) / 127.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round(wf / safe[:, None]), -127, 127).to(torch.int8)
    return q, s


def rtn_int4_groups(w, group: int):
    """Symmetric int4 per group of `group` input columns: s = amax / 7, q = rne(w / s) clamped to [-8, 7]; returns
    the dequantised float32 weight."""
    wf = w.float()
    o, i = wf.shape
    g = wf.view(o, i // group, group)
    s = g.abs().amax(-1, keepdim=True) / 7.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.clamp(torch.round(g / safe), -8, 7) * s).view(o, i)


def int4_simulate(drafter, group: int) -> int:
    n = 0
    for name in list(drafter.w):
        if name.endswith(LINEAR_SUFFIXES) and drafter.w[name].dim() == 2:
            drafter.w[name] = rtn_int4_groups(drafter.w[name], group)
            n += 1
    return n


def int8_simulate(drafter) -> int:
    """Replace every drafter linear weight by its int8-RTN dequantisation (float32). Returns the
    count of tensors quantised."""
    n = 0
    for name in list(drafter.w):
        if name.endswith(LINEAR_SUFFIXES) and drafter.w[name].dim() == 2:
            q, s = rtn_int8_rows(drafter.w[name])
            drafter.w[name] = q.float() * s[:, None]
            n += 1
    return n


class Head:
    """The target's lm_head for the draft: bf16 (upcast once to float32 when `upcast`) or int8
    rows + fp32 scales (dequantised per chunk)."""

    def __init__(self, w, int8: bool = False, upcast: bool = True, chunk: int = 32768):
        self.chunk = chunk
        self.int8 = int8
        if int8:
            self.q, self.s = rtn_int8_rows(w)
            self.w = None
        else:
            self.w = w.float() if upcast else w
        self.rows = w.shape[0]

    def logits(self, h):
        h = h.float()
        if not self.int8:
            return ref.linear(h, self.w, chunk=self.chunk)
        return torch.cat([h @ (self.q[i:i + self.chunk].float() * self.s[i:i + self.chunk, None]).t()
                          for i in range(0, self.rows, self.chunk)], -1)

    def two_bit_logits(self, h):
        """The int8 head's 2-bit copy (loader::dflash_candidate_head2): per row and 64 k, scale = fp16(amax / 1.5) of
        the int8 values times the row scale, code = clamp(rint(v / scale + 1.5), 0, 3), value (code - 1.5) scale."""
        if not hasattr(self, "q2"):
            parts = []
            for i in range(0, self.rows, self.chunk):
                v = (self.q[i:i + self.chunk].float() * self.s[i:i + self.chunk, None])
                g = v.view(v.shape[0], -1, 64)
                sc = (g.abs().amax(-1, keepdim=True) / 1.5).half().float()
                code = torch.where(sc > 0, torch.clamp(torch.round(g / torch.where(sc > 0, sc, 1) + 1.5), 0, 3),
                                   torch.ones_like(g))
                parts.append(((code - 1.5) * sc).view_as(v).to(torch.bfloat16))
            self.q2 = torch.cat(parts)
        h = h.float()
        return torch.cat([h @ self.q2[i:i + self.chunk].float().t() for i in range(0, self.rows, self.chunk)], -1)


# ------------------------------------------------------------------------------ batched drafting


def top_k_fast(logits, k: int):
    """dflash_ref.top_k_stable (descending, equal values in increasing id order) via topk: the
    k-th value is the threshold; a row with more than k ids at or above it (a tie across the
    boundary) falls back to the full stable sort."""
    v, i = torch.topk(logits, k, dim=-1)
    thr = v[:, -1:]
    over = (logits >= thr).sum(-1) > k
    o = torch.argsort(i, dim=-1, stable=True)
    v, i = v.gather(-1, o), i.gather(-1, o)
    o = torch.sort(v, dim=-1, descending=True, stable=True).indices
    v, i = v.gather(-1, o), i.gather(-1, o)
    if over.any():
        rows = over.nonzero().flatten()
        ii, vv = ref.top_k_stable(logits[rows], k)
        i[rows], v[rows] = ii, vv
    return i, v


def _attention(d, i, x, qpos, ps, R, ctx):
    """dflash_ref.Drafter._attention for B stacked blocks: block b (rows b*R .. b*R+R-1, anchor
    ps[b]) sees the context positions < ps[b] and its own rows, under the layer's window /
    causality (Drafter.attention_mask)."""
    c, L = d.cfg, f"layers.{i}.self_attn."
    BR = x.shape[0]
    B = BR // R
    qp = qpos.reshape(-1)
    q = ref.linear(x, d.w[L + "q_proj.weight"], d.w.get(L + "q_proj.bias")).view(BR, c.n_heads, c.head_dim)
    q = ref.rope(ref.rms_norm(q, d.w[L + "q_norm.weight"], c.eps), qp, c.rope_theta, c.neox)
    kb, vb = d._kv(i, x, qp)
    win = c.windows[i]
    sel = ctx.pos < int(ps.max())
    if win is not None:
        sel &= ctx.pos >= int(ps.min()) - win
    k = torch.cat([ctx.k[i][sel], kb], 0)
    v = torch.cat([ctx.v[i][sel], vb], 0)
    kpos = torch.cat([ctx.pos[sel], qp])
    nc = int(sel.sum())
    owner_k = torch.cat([torch.full((nc,), -1, dtype=torch.long), torch.arange(B).repeat_interleave(R)])
    owner_q = torch.arange(B).repeat_interleave(R)
    anchor_q = ps.repeat_interleave(R)
    vis = ((owner_k[None] == -1) & (kpos[None] < anchor_q[:, None])) | (owner_k[None] == owner_q[:, None])
    dd = qp[:, None] - kpos[None]
    if win is not None:
        vis &= dd.abs() <= win - 1
    if c.causal[i]:
        vis &= dd >= 0
    g = c.n_heads // c.n_kv_heads
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) * c.head_dim ** -0.5
    s = s.masked_fill(~vis[None], float("-inf"))
    if c.attention_sink_bias:
        sink = d.w[L + "attention_sink_bias"].float()[:, None, None].expand(-1, BR, 1)
        pr = torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]
    else:
        pr = torch.softmax(s, -1)
    o = torch.einsum("hqk,khd->qhd", pr, v).reshape(BR, c.n_heads * c.head_dim)
    return ref.linear(o, d.w[L + "o_proj.weight"], d.w.get(L + "o_proj.bias"))


def forward_blocks(d, anchors, ps, K: int, ctx):
    """B blocks [anchor_b, mask x K] at positions ps[b] .. ps[b] + K. Returns the final-norm
    hidden [B, 1 + K, H] (Drafter.forward_block per block)."""
    c = d.cfg
    B, R = len(anchors), K + 1
    ids = torch.cat([torch.as_tensor(anchors, dtype=torch.long)[:, None],
                     torch.full((B, K), c.mask_token_id, dtype=torch.long)], 1)
    x = d.embed_rows(ids.reshape(-1))
    ps = torch.as_tensor(ps, dtype=torch.long)
    qpos = ps[:, None] + torch.arange(R)[None]
    resid = x
    for i in range(c.n_layers):
        L = f"layers.{i}."
        h = ref.rms_norm(resid, d.w[L + "input_layernorm.weight"], c.eps)
        if c.version == 2:
            h, co = d._conv_prepare(i, "attention_conv", h, R)
            h = _attention(d, i, h, qpos, ps, R, ctx)
            h = d._conv_finish(i, "attention_conv", h, co, R)
        else:
            h = _attention(d, i, h, qpos, ps, R, ctx)
        resid = resid + h
        h = ref.rms_norm(resid, d.w[L + "post_attention_layernorm.weight"], c.eps)
        if c.version == 2:
            h, co = d._conv_prepare(i, "mlp_conv", h, R)
            h = d._mlp(i, h)
            h = d._conv_finish(i, "mlp_conv", h, co, R)
        else:
            h = d._mlp(i, h)
        resid = resid + h
    return ref.rms_norm(resid, d.w["norm.weight"], c.eps).view(B, R, -1)


def select_ids(d, hm, logits, anchors, K: int, walk: bool = False):
    """Greedy draft ids [B, K] from the mask rows' hidden hm [B*K, H] and their head logits
    [B*K, V]. DFlash 2: top-k, output_multiplier /
    softcap (before or after the correction, dflash_ref.Drafter.unary_and_cap), the bilinear edge scores, the argmax
    walk (ties to the lower candidate index); DFlash (v1): argmax of the logits (ties to the lower id).
    walk (DFlash 2): also the realized score rows [B, K, k], the candidates [B, K, k] and the chosen index [B, K]."""
    c = d.cfg
    B = len(anchors)
    if c.version != 2:
        if d.d2t is not None:
            raise NotImplementedError("v1 with d2t")
        lg = logits * c.logit_scale
        best = lg.max(-1, keepdim=True).values
        idx = torch.where(lg == best, torch.arange(lg.shape[1])[None], lg.shape[1]).min(-1).values
        return idx.view(B, K)
    kk = c.selector_top_k
    if d.valid_vocab is not None:
        logits = logits.clone()
        logits[:, d.valid_vocab:] = float("-inf")
    cand, unary = top_k_fast(logits, kk)
    unary, cap = d.unary_and_cap(unary)
    hid = ref.linear(hm, d.w["candidate_selector.hidden_projection.weight"])
    cand, unary, hid = cand.view(B, K, kk), unary.view(B, K, kk), hid.view(B, K, -1)
    anc = torch.as_tensor(anchors, dtype=torch.long)
    prev = torch.cat([anc[:, None, None].expand(B, 1, kk), cand[:, :-1]], 1)            # [B, K, k]
    P = d.w["candidate_selector.predecessor_codebook"][prev].float() * hid.float()[:, :, None, :]
    S_ = d.w["candidate_selector.successor_codebook"][cand].float()
    S = unary.float()[:, :, None, :] + torch.einsum("bjar,bjcr->bjac", P, S_)            # [B, K, k, k]
    if cap is not None:
        S = torch.tanh(S / cap) * cap
    chosen = torch.zeros(B, dtype=torch.long)
    ar = torch.arange(B)
    out, rows, picks = [], [], []
    idx = torch.arange(kk)
    for j in range(K):
        row = S[ar, j, chosen].double()                                                  # [B, k]
        best = row.max(-1, keepdim=True).values
        chosen = torch.where(row == best, idx[None], kk).min(-1).values
        out.append(cand[ar, j, chosen])
        rows.append(row)
        picks.append(chosen)
    if walk:
        return torch.stack(out, 1), torch.stack(rows, 1), cand, torch.stack(picks, 1)
    return torch.stack(out, 1)


def draft_batch(d, head, anchors, ps, K: int, ctx, candidate_counts: dict, walk: bool = False):
    """{variant: ids [B, K]} for one batch (walk: select_ids' walk tuple); candidate_counts maps a variant name to the
    number of 2-bit candidates to rescore, or None to use the full head."""
    h = forward_blocks(d, anchors, ps, K, ctx)
    B = len(anchors)
    hm = h[:, 1:].reshape(B * K, -1)
    logits = head.logits(hm)
    two_bit = None
    out = {}
    for name, count in candidate_counts.items():
        if count is not None:
            if two_bit is None:
                two_bit = head.two_bit_logits(hm)
                if d.valid_vocab is not None:
                    two_bit[:, d.valid_vocab:] = float("-inf")
            top = torch.topk(two_bit, count, dim=-1).indices
            lg = torch.full_like(logits, float("-inf")).scatter_(1, top, logits.gather(1, top))
        else:
            lg = logits
        out[name] = select_ids(d, hm, lg, anchors, K, walk)
    return out


# ------------------------------------------------------------------------------ acceptance


def accept_rows(drafts, ps, x, g, g_from: int):
    """drafts [A, K]; ps [A] anchors; x [N] ids; g[t - g_from] = the target's next id after t.
    Returns (length [A], censored [A] bool, counted [K], accepted [K]) under the protocol above."""
    A, K = drafts.shape
    N = len(x)
    length = torch.zeros(A, dtype=torch.long)
    censored = torch.zeros(A, dtype=torch.bool)
    counted = [0] * K
    accepted = [0] * K
    xs, gs, dl = x.tolist(), g.tolist(), drafts.tolist()
    for a, p in enumerate(ps.tolist()):
        n = 0
        cens = False
        for j in range(K):
            if p + j > N - 1 or (j > 0 and xs[p + j] != gs[p + j - 1 - g_from]):
                cens = True
                break
            counted[j] += 1
            if dl[a][j] != gs[p + j - g_from]:
                break
            accepted[j] += 1
            n += 1
        length[a], censored[a] = n, cens
    return length, censored, counted, accepted


def expected_tokens(alphas):
    e, prod = 1.0, 1.0
    for a in alphas:
        prod *= (a or 0.0)
        e += prod
    return e


def anchors_of(n_prompt: int, n: int):
    return list(range(n_prompt, n - KMAX + 1))


def target_probs(dump, rows, ids):
    """The target's p of ids [A, K] at dump rows [A, K] (the id after that position), from the top 64 and the
    log-sum-exp; 0 outside the top 64."""
    ti, tl, lse = dump["top_ids"][rows], dump["top_logits"][rows], dump["lse"][rows]
    hit = ti == ids[..., None]
    lp = torch.where(hit, tl, torch.full_like(tl, float("-inf"))).amax(-1)
    return torch.exp(lp.double() - lse.double())


def sampled_and_gtau(dump, walked, ps_all, length, censored, costs, draft_cost):
    """T 1.0 along the greedy text and GTAU for one variant's K = 7 walk, cut to the drafts the costs' verify rows hold
    (module docstring)."""
    K = min(walked[1].shape[1], len(costs) - 2)
    ids, real, cand, chosen = (t[:, :K] for t in walked)
    A, K, kk = real.shape
    f = dump["tap_from"]
    rows = ps_all[:, None] + torch.arange(K)[None] - f                           # the dump row of draft j's context
    q = torch.softmax(real, -1)                                                  # the selector's q at T 1
    pm = target_probs(dump, rows, ids)                                           # point mass: p(draft)
    pc = target_probs(dump, rows[..., None].expand(A, K, kk), cand)              # p of every candidate
    ex = torch.minimum(pc, q).sum(-1)                                            # exact: sum_c min(p, q)
    e_pm = 1 + torch.cumprod(pm, 1).cumsum(1)                                    # [A, K]: E at K = 1..7
    e_ex = 1 + torch.cumprod(ex, 1).cumsum(1)
    qc = q.gather(-1, chosen[..., None])[..., 0]                                 # each draft's own q at T 1
    conf = torch.cumprod(qc, 1)                                                  # the walk's confidence
    out = {"rows": A, "pm": e_pm.sum(0).tolist(), "exact": e_ex.sum(0).tolist(), "gtau": {}}
    # per anchor (in text order, for pricing a stop rule offline): the drafts' q, the greedy accepted length, censored
    out["anchors"] = {"q": [[round(v, 5) for v in r] for r in qc.tolist()], "length": length.tolist(),
                      "censored": censored.tolist()}
    ok = ~censored
    L = length[ok]
    out["fixed"] = {"n": int(ok.sum()), "ids": [float((1 + torch.clamp(L, max=k)).sum()) for k in range(1, K + 1)]}
    for tau in TAUS:
        below = conf[ok] < tau
        k_tau = torch.where(below.any(1), below.float().argmax(1) + 1, torch.full((int(ok.sum()),), K))
        got = 1 + torch.minimum(L, k_tau)
        out["gtau"][str(tau)] = {"ids": float(got.sum()), "rows": float((k_tau + 1).sum()),
                                 "cost": float(sum(costs[int(k) + 1] + draft_cost for k in k_tau.tolist())),
                                 "given_up": float((torch.clamp(L, max=K) - torch.minimum(L, k_tau)).sum())}
    return out


def run_source(d, head, dump, ks, candidate_counts, batch: int, log=print, whole_block: bool = False, costs=None,
               draft_cost: float = 0.0):
    """Per-variant, per-K stats for one dump (whole_block: one K = 7 block an anchor, every K its prefix; with it
    the T 1.0 and GTAU stats under the variant's '_extra')."""
    taps = dump["taps"]
    f = dump["tap_from"]
    pos = torch.arange(f, dump["n"])
    ctx = d.context_kv({i: taps[i] for i in d.cfg.target_layer_ids}, pos)
    x, g = dump["ids"], dump["greedy"]
    anchors = anchors_of(dump["n_prompt"], dump["n"])
    res = {name: {} for name in candidate_counts}
    if not anchors:
        return res, 0
    ps_all = torch.tensor(anchors)
    if whole_block:
        walked = {name: [] for name in candidate_counts}
        for b0 in range(0, len(anchors), batch):
            ps = ps_all[b0:b0 + batch]
            out = draft_batch(d, head, x[ps].tolist(), ps, KMAX, ctx, candidate_counts, walk=True)
            for name in candidate_counts:
                walked[name].append(out[name])
        for name in candidate_counts:
            w = tuple(torch.cat([o[i] for o in walked[name]]) for i in range(4))
            for K in ks:
                length, cens, counted, accepted = accept_rows(w[0][:, :K], ps_all, x, g, f)
                hist = torch.bincount(length[~cens], minlength=K + 1).tolist()
                res[name][str(K)] = {"rows": len(anchors), "censored": int(cens.sum()), "hist": hist,
                                     "counted": counted, "accepted": accepted}
            length, cens, _, _ = accept_rows(w[0][:, :len(costs) - 2], ps_all, x, g, f)
            res[name]["_extra"] = sampled_and_gtau(dump, w, ps_all, length, cens, costs, draft_cost)
        n_p = dump["n_prompt"]
        agree = int((x[n_p:] == g[n_p - 1 - f: dump["n"] - 1 - f]).sum())
        return res, agree
    for K in ks:
        drafts = {name: [] for name in candidate_counts}
        for b0 in range(0, len(anchors), batch):
            ps = ps_all[b0:b0 + batch]
            out = draft_batch(d, head, x[ps].tolist(), ps, K, ctx, candidate_counts)
            for name in candidate_counts:
                drafts[name].append(out[name])
        for name in candidate_counts:
            dr = torch.cat(drafts[name])
            length, cens, counted, accepted = accept_rows(dr, ps_all, x, g, f)
            hist = torch.bincount(length[~cens], minlength=K + 1).tolist()
            res[name][str(K)] = {"rows": len(anchors), "censored": int(cens.sum()), "hist": hist,
                                 "counted": counted, "accepted": accepted}
    # how often the recorded text is the bf16 greedy, over the continuation
    n_p = dump["n_prompt"]
    agree = int((x[n_p:] == g[n_p - 1 - f: dump["n"] - 1 - f]).sum())
    return res, agree


def pool(entries, ks):
    """Pool per-source stats (one variant) -> per-K summary."""
    out = {}
    for K in ks:
        sk = str(K)
        rows = [e[sk] for e in entries if sk in e]
        if not rows:
            continue
        counted = [sum(r["counted"][j] for r in rows) for j in range(K)]
        accepted = [sum(r["accepted"][j] for r in rows) for j in range(K)]
        hist = [sum(r["hist"][j] for r in rows) for j in range(K + 1)]
        alphas = [accepted[j] / counted[j] if counted[j] else None for j in range(K)]
        n_unc = sum(hist)
        out[sk] = {"rows": sum(r["rows"] for r in rows), "censored": sum(r["censored"] for r in rows),
                   "alpha": alphas, "E": expected_tokens(alphas), "hist": hist,
                   "mean_len_plus1": (1 + sum(i * h for i, h in enumerate(hist)) / n_unc) if n_unc else None}
    return out


def pool_extra(entries):
    """Pool sampled_and_gtau's per-source stats (one variant)."""
    es = [e["_extra"] for e in entries if "_extra" in e]
    if not es:
        return None
    rows = sum(e["rows"] for e in es)
    K = len(es[0]["pm"])
    out = {"rows": rows, "pm": [sum(e["pm"][k] for e in es) / rows for k in range(K)],
           "exact": [sum(e["exact"][k] for e in es) / rows for k in range(K)], "gtau": {}}
    n = sum(e["fixed"]["n"] for e in es)
    out["fixed_ids"] = [sum(e["fixed"]["ids"][k] for e in es) / max(1, n) for k in range(K)]
    out["n"] = n
    for tau in es[0]["gtau"]:
        g = [e["gtau"][tau] for e in es]
        out["gtau"][tau] = {k: sum(x[k] for x in g) / max(1, n) for k in ("ids", "rows", "cost", "given_up")}
    return out


def summarize(files, md_out=None, ks=tuple(range(1, KMAX + 1)), costs=None, draft_cost=0.0) -> dict:
    per_arm = {}
    agree = {}
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            r = json.load(f)
        for variant, srcs in r["variants"].items():
            per_arm.setdefault(variant, {}).update(srcs)
        agree.update(r.get("agree", {}))
    summary = {}
    for variant, srcs in per_arm.items():
        groups = {}
        for name, e in srcs.items():
            groups.setdefault(name.split("/")[0], []).append(e)
        groups["all"] = list(srcs.values())
        summary[variant] = {corp: pool(es, ks) for corp, es in groups.items()}
        for corp in groups:
            summary[variant][corp]["sources"] = len(groups[corp]) if corp != "all" else len(srcs)
    lines = []
    corpora = sorted({c for v in summary.values() for c in v}, key=lambda c: (c == "all", c))
    for corp in corpora:
        ag = [agree[n] for n in agree if corp == "all" or n.split("/")[0] == corp]
        txt = (f", recorded text = bf16 greedy at {sum(a for a, _ in ag) / max(1, sum(t for _, t in ag)):.3f} "
               f"of {sum(t for _, t in ag)} ids") if ag else ""
        lines.append(f"\n### {corp}{txt}\n")
        lines.append("| arm | rows | " + " | ".join(f"E_{K}" for K in ks) + " | best K | alpha_1..7 at K = 7 |")
        lines.append("|---|---:|" + "---:|" * len(ks) + "---:|---|")
        for variant in sorted(summary, key=_arm_order):
            s = summary[variant].get(corp)
            if not s:
                continue
            es = [s[str(K)]["E"] if str(K) in s else None for K in ks]
            best = max((e, K) for e, K in zip(es, ks) if e is not None)
            a7 = s.get(str(KMAX), {}).get("alpha") or []
            rows = s.get(str(ks[0]), {}).get("rows", 0)
            lines.append(f"| {variant} | {rows} | " + " | ".join(f"{e:.2f}" if e is not None else "-" for e in es)
                         + f" | {best[1]} | " + " ".join(f"{a:.2f}" if a is not None else "-" for a in a7) + " |")
    for corp in corpora:
        rows_x = [(v, pool_extra([e for n_, e in per_arm[v].items() if corp == "all" or n_.split("/")[0] == corp]))
                  for v in sorted(per_arm, key=_arm_order)]
        rows_x = [(v, x) for v, x in rows_x if x]
        if not rows_x:
            continue
        lines.append(f"\n### {corp}: T 1.0 along the greedy text (E at K = 1..7) and GTAU\n")
        lines.append("| arm | point mass E_1..7 | exact E_1..7 | best fixed K: ids / cost | "
                     + " | ".join(f"GTAU {t}: rows, ids, given up, ids / cost" for t in TAUS) + " |")
        lines.append("|---|---|---|---|" + "---|" * len(TAUS))
        for v, x in rows_x:
            fixed = ""
            if costs:
                per = [(x["fixed_ids"][k] / (costs[k + 2] + draft_cost), k + 1) for k in range(len(x["fixed_ids"]))]
                b = max(per)
                fixed = f"K {b[1]}: {x['fixed_ids'][b[1] - 1]:.2f} / {b[0]:.3f}"
            gt = []
            for t in TAUS:
                g = x["gtau"].get(str(t))
                gt.append(f"{g['rows']:.2f}, {g['ids']:.2f}, {g['given_up']:.2f}, {g['ids'] / g['cost']:.3f}"
                          if g and g["cost"] else "-")
            lines.append(f"| {v} | " + " ".join(f"{e:.2f}" for e in x["pm"]) + " | "
                         + " ".join(f"{e:.2f}" for e in x["exact"]) + f" | {fixed} | " + " | ".join(gt) + " |")
    text = "\n".join(lines)
    print(text)
    if md_out:
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return summary


def _arm_order(v):
    base = ["bf16", "bf16h8", "int8", "int8h", "int4g64h", "int4g128h", "w4a16"]
    b = v.split("-")[0]
    return (base.index(b) if b in base else 9, v)


# ------------------------------------------------------------------------------ main


def dump_files(dumps_dir, pattern):
    fs = sorted(glob.glob(os.path.join(dumps_dir, "*.taps.safetensors")))
    if pattern:
        import fnmatch
        fs = [f for f in fs if fnmatch.fnmatch(os.path.basename(f), pattern)]
    return fs


def cmd_run(a) -> None:
    arms = [s for s in a.arms.split(",") if s]
    bad = [s for s in arms if s not in ("bf16", "bf16h8", "int8", "int8h", "int4g64h", "int4g128h", "w4a16")]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    lo, _, hi = a.k.partition("-")
    ks = list(range(int(lo), int(hi or lo) + 1))
    if not (1 <= ks[0] and ks[-1] <= KMAX):
        raise SystemExit(f"--k within 1-{KMAX}")
    plan = {arm: [arm] for arm in arms}
    cands = [int(c) for c in a.candidates.split(",") if c]
    if "int8h" in plan:
        plan["int8h"] += [f"int8h-c{c}" for c in cands]
    files = dump_files(a.dumps, a.sources)
    os.makedirs(a.out_dir, exist_ok=True)
    # the planned work, from the dump headers (or sources.json before any dump exists)
    if files:
        _load_deps()
        heads = [dt.dump_header(f) for f in files]
        srcs = [(h["name"], int(h["n_prompt"]), int(h["n"])) for h in heads]
    else:
        sj = os.path.join(a.dumps, "sources.json")
        if not os.path.isfile(sj):
            raise SystemExit(f"no dumps and no sources.json in {a.dumps}")
        with open(sj, encoding="utf-8") as f:
            srcs = [(s["name"], s["n_prompt"], s["n"]) for s in json.load(f)]
        if a.sources:
            import fnmatch
            srcs = [s for s in srcs if fnmatch.fnmatch(s[0].replace("/", "__") + ".taps.safetensors", a.sources)]
    n_anchor = sum(len(anchors_of(p, n)) for _, p, n in srcs)
    rows = n_anchor * sum(K + 1 for K in ks)
    head_rows = n_anchor * sum(ks)
    gflop = (rows * 2 * 1.65e9 + head_rows * 2 * 248320 * 5120) / 1e9
    print(f"{len(srcs)} sources, {n_anchor} anchors, K {ks[0]}..{ks[-1]}; arms {plan}")
    print(f"per base arm: {rows} drafter rows + {head_rows} head rows = {gflop / 1e3:.0f} TFLOP, "
          f"~{gflop / ASSUMED_GFLOPS / 3600:.2f} h at {ASSUMED_GFLOPS:.0f} GFLOP/s (assumed); "
          f"{len(plan)} base arms ~{len(plan) * gflop / ASSUMED_GFLOPS / 3600:.1f} h")
    print("memory (estimated): drafter float32 6.9 GB + bf16 load 3.5 GB transient, embedding bf16 2.5 GB, "
          "head float32 5.1 GB (int8h: 1.3 GB), a source's context 40 KB/position, a batch's logits "
          f"{a.batch * KMAX * 248320 * 4 / 2**30:.2f} GiB x 2 -> ~16-20 GiB peak")
    if a.dry_run:
        return
    _load_deps()
    torch.set_grad_enabled(False)
    target = a.target or ref.find_snapshot("Qwen/Qwen3.8-27B", ("config.json", "tokenizer.json"))
    inkling = flash_next = False
    if target and os.path.isfile(os.path.join(target, "config.json")):
        with open(os.path.join(target, "config.json"), encoding="utf-8") as f:
            tc = json.load(f)
        inkling = "loop_config" in (tc.get("text_config") or {})
        flash_next = tc.get("model_type") == "qwen4_exp"
    drafter_dir = a.drafter or ref.find_snapshot(
        "incoai/Inkling-2-Mini-DFlash2" if inkling else "incoai/Qwen3.8-Flash-Next-DFlash2" if flash_next
        else "z-lab/Qwen3.8-27B-DFlash2", ("config.json", "model.safetensors"))
    w4_dir = a.w4a16 or ref.find_snapshot("syvai/Qwen3.8-27B-DFlash2-W4A16", ("config.json", "model.safetensors"))
    if target is None or drafter_dir is None:
        raise SystemExit(f"target {target} / drafter {drafter_dir} not found in {ref.hf_hub_dir()}")
    t0 = time.time()
    tgt = ref.read_inkling_target(target) if inkling else None
    embed, head_w = (tgt["embed"], tgt["lm_head"]) if inkling else ref.read_target_embed_head(target)
    vocab_used = INKLING_VOCAB_USED if inkling and a.vocab_used == VOCAB_USED else a.vocab_used
    costs = [0.0] + [float(c) for c in (a.costs or (INKLING_COSTS if inkling else FLASH_NEXT_COSTS if flash_next
                                                    else "")).split(",") if c]
    print(f"target embed / head {time.time() - t0:.0f} s", flush=True)
    for arm, variants in plan.items():
        out_path = os.path.join(a.out_dir, f"accept.{arm}.json")
        state = {"arm": arm, "variants": {v: {} for v in variants}, "agree": {}, "meta": {}}
        if os.path.isfile(out_path):
            with open(out_path, encoding="utf-8") as f:
                old = json.load(f)
            for v in variants:
                state["variants"][v].update(old["variants"].get(v, {}))
            state["agree"].update(old.get("agree", {}))
        todo = [f for f in files if any(dt.dump_header(f)["name"] not in state["variants"][v] for v in variants)]
        if not todo:
            print(f"[{arm}] done", flush=True)
            continue
        ta = time.time()
        src_dir = w4_dir if arm == "w4a16" else drafter_dir
        if src_dir is None:
            print(f"[{arm}] SKIP: drafter not in the cache", flush=True)
            continue
        d = ref.load_drafter(src_dir, embed=embed, lm_head=head_w,
                             embed_norm=tgt["embed_norm"] if inkling else None,
                             valid_vocab=tgt["valid_vocab"] if inkling else None)
        nq = (int8_simulate(d) if arm in ("int8", "int8h") else
              int4_simulate(d, 64) if arm == "int4g64h" else int4_simulate(d, 128) if arm == "int4g128h" else 0)
        head = Head(head_w, int8=arm in ("int8h", "bf16h8", "int4g64h", "int4g128h"))
        candidate_counts = {arm: None, **{v: int(v.rsplit("-c", 1)[1]) for v in variants[1:]}}
        state["meta"] = {"drafter": src_dir, "target": target, "int8_tensors": nq, "head": "int8" if head.int8 else "bf16",
                         "ks": ks, "batch": a.batch, "vocab_used": vocab_used,
                         "threads": torch.get_num_threads()}
        print(f"[{arm}] drafter loaded {time.time() - ta:.0f} s ({nq} linears int8), head {state['meta']['head']}", flush=True)
        for fn in todo:
            dump = dt.load_dump(fn)
            ts = time.time()
            res, agree = run_source(d, head, dump, ks, candidate_counts, a.batch, whole_block=inkling or flash_next,
                                    costs=costs, draft_cost=a.draft_cost)
            for v in variants:
                state["variants"][v][dump["name"]] = res[v]
            state["agree"][dump["name"]] = [agree, dump["n"] - dump["n_prompt"]]
            with open(out_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(state, f, indent=1)
            os.replace(out_path + ".tmp", out_path)
            r7 = res[arm].get(str(ks[-1]))
            e = pool([res[arm]], ks)
            es = " ".join("%.2f" % e[str(K)]["E"] for K in ks if str(K) in e)
            n_a = len(anchors_of(dump["n_prompt"], dump["n"]))
            print(f"[{arm}] {dump['name']}: {n_a} anchors, E_K {es}"
                  + (f", censored {r7['censored']}" if r7 else "") + f"; {time.time() - ts:.0f} s", flush=True)
        del d, head
    files_out = sorted(glob.glob(os.path.join(a.out_dir, "accept.*.json")))
    summarize(files_out, os.path.join(a.out_dir, "summary.md"), ks, costs, a.draft_cost)
    print(f"wall {time.time() - t0:.0f} s", flush=True)


def cmd_drafts(a) -> None:
    """The reference drafts at a dump's first anchors (positions n_prompt ..), as text: line 1 the dump's ids (prompt +
    text), line 2 n_prompt, then per anchor its position and its 7 drafts (the engine's form: bf16 drafter, int8 head)."""
    _load_deps()
    torch.set_grad_enabled(False)
    tgt = ref.read_inkling_target(a.target)
    drafter_dir = a.drafter or ref.find_snapshot("incoai/Inkling-2-Mini-DFlash2", ("config.json", "model.safetensors"))
    d = ref.load_drafter(drafter_dir, embed=tgt["embed"], lm_head=tgt["lm_head"], embed_norm=tgt["embed_norm"],
                         valid_vocab=tgt["valid_vocab"])
    head = Head(tgt["lm_head"], int8=True)
    dump = dt.load_dump(a.dump)
    f = dump["tap_from"]
    ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, torch.arange(f, dump["n"]))
    ps = torch.tensor(anchors_of(dump["n_prompt"], dump["n"])[:a.anchors])
    x = dump["ids"]
    ids = draft_batch(d, head, x[ps].tolist(), ps, KMAX, ctx, {"a": None})["a"]
    with open(a.out, "w", encoding="utf-8") as out:
        out.write(" ".join(map(str, x.tolist())) + "\n" + str(dump["n_prompt"]) + "\n")
        for p, row in zip(ps.tolist(), ids.tolist()):
            out.write(" ".join(map(str, [p] + row)) + "\n")
    print(f"{len(ps)} anchors from {ps[0].item()}: drafts written to {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dumps", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--arms", default="bf16,int8,int8h,w4a16")
    r.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    r.add_argument("--k", default=f"1-{KMAX}")
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--drafter")
    r.add_argument("--w4a16")
    r.add_argument("--target")
    r.add_argument("--sources", help="fnmatch over the dump file names")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--candidates", default="64,80", help="int8h's 2-bit candidate variants (Inkling)")
    r.add_argument("--costs", help="T(1..8) in plain steps for GTAU's pricing (Inkling: one card's at 4k)")
    r.add_argument("--draft-cost", type=float, default=0.2, help="the draft pass in plain steps (GTAU's pricing)")
    q = sub.add_parser("drafts")
    q.add_argument("--dump", required=True)
    q.add_argument("--target", required=True)
    q.add_argument("--drafter")
    q.add_argument("--anchors", type=int, default=32)
    q.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--md")
    s.add_argument("--costs", default=INKLING_COSTS)
    s.add_argument("--draft-cost", type=float, default=0.2)
    a = ap.parse_args()
    if a.cmd == "drafts":
        return cmd_drafts(a)
    if a.cmd == "summary":
        summarize(a.files, a.md, costs=[0.0] + [float(c) for c in a.costs.split(",")], draft_cost=a.draft_cost)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
