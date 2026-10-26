#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced DFlash acceptance on tap dumps (spec 19a Task 3, Review Focus 2-5).

    dflash_accept.py run --dumps DIR --out-dir OUT [--arms bf16,int8,int8h,w4a16]
                     [--vocab 32k,64k,128k] [--vocab-on bf16] [--draft-vocab-ids FILE]
                     [--k 1-7] [--batch 32] [--drafter SNAP] [--w4a16 SNAP] [--target SNAP]
                     [--sources GLOB] [--dry-run]
    dflash_accept.py summary OUT/accept.*.json [--md OUT.md]

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
**Draft vocabulary** (decision 6, Review Focus 5): V' = loader::select_draft_vocab exactly
(tokenizer.json's added tokens, then generation_config's EOS ids, then --draft-vocab-ids, then
the lowest ids; ids >= vocab_used never). The head rows outside V' are masked to -inf before the
top-16, so every candidate is in V'; a target id outside V' is then always a miss. Arms
`<base>-v<size>` (e.g. bf16-v32k) share the base arm's forward and use no ranked list (added +
EOS + the lowest ids, b70-serve's V' without --draft-vocab-ids); with --draft-vocab-ids
(tools/draft_vocab/rank.py's output) `<base>-v<size>-r` arms use it too.

**Batched.** Anchors go through the drafter B at a time (rows stacked as B blocks of 1 + K;
the conv's look-back is per block, `row % (1 + K)`; the attention mask gives each block the
context < its anchor plus its own rows); the head runs once per batch over all B x K mask rows
(the README's warning: per-call head upcasts are slow); test_dflash_accept.py checks the batch
against dflash_ref.Drafter.draft_block row by row. Greedy (T = 0) only.

Resumable: OUT/accept.<arm>.json is rewritten after every source and a re-run skips the sources
it already holds. --dry-run prints the planned anchors, the FLOP and a memory estimate.

**The Muse-Glimmer form (spec 23f Task 3, `--model muse`).** meta-models/Muse-Glimmer-30B-assistant
(dflash_ref.py's form "muse") over muse_taps.py's dumps, beside the int4 target's embedding and
head (Vishva007/Muse-Glimmer-30B-W4A16-AutoRound: the served checkpoint):
  - the block is ALWAYS 16 rows (anchor + 15 mask rows, bidirectional): ONE forward per anchor
    gives the drafts of every K = 1..15 (rows 1..K - forward_blocks_muse; the same anchors for
    every K: n_prompt <= p <= N - --kmax, default 15);
  - greedy drafts = argmax of the RAW head rows (G:1658, ties to the lower id) - the engine's;
  - arms: bf16 (as shipped, float32 compute), int8 (per-row RTN, simulated as above), int4g64
    (loader/rtn.h's rtn_int4_g64 restated: per output row and group of 64 k, s16 = f16(2 amax /
    15), q = clamp(rint(w / s16) + 8, 0, 15), w' = (q - 8) s16 - spec 23 decision 9's format), and
    `h` variants (int8h, int4g64h) with the target's head int8 per row (spec 9: the head the
    engine serves Muse with by default); dflash2 = incoai/Muse-Glimmer-30B-DFlash2 (decision 12)
    through the vLLM DFlash 2 path when it is in the cache and its taps are the dumps' - else
    SKIPPED (never downloaded here);
  - **the sampled estimate** (spec 8 §11 / mtp_accept.py's expected-acceptance formula): per
    draft row j of anchor p, alpha_j = mean of sum_x min(p(x), q(x)) where p is the target's
    capped row after p + j (the dump's top-64 capped logits) filtered at --temperature / --top-k /
    --top-p (default generation_config.json's: 1.0 / 64 / 0.95) and q the draft row's top 16
    (the engine's readback) at the same temperature - `--q raw` (softmax(raw / T): transformers'
    form) and `--q capped` (the target's multiplier and softcap first: 20 tanh(z 0.196... / 20),
    decision 11) both by default; E_K = 1 + sum prod alpha_j. Rows are counted wherever the
    target row exists (the teacher-forced text stands for the accepted drafts).
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
VOCAB_SIZES = {"32k": 32768, "64k": 65536, "128k": 131072}
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


# ------------------------------------------------------------------------------ draft vocab


def select_draft_vocab(added, eos, ranked, vocab_used: int, size: int) -> list[int]:
    """src/loader/draft_vocab.cc `select_draft_vocab`, line for line: added, then EOS (always),
    then ranked, then the lowest remaining ids; duplicates and ids >= vocab_used skipped;
    returned sorted ascending."""
    if size == 0:
        raise ValueError("select_draft_vocab: |V'| = 0 (off is not a vocabulary)")
    if size > vocab_used:
        raise ValueError(f"select_draft_vocab: |V'| {size} exceeds the {vocab_used} usable ids")
    taken = bytearray(vocab_used)
    out = []

    def take(i):
        if i >= vocab_used or taken[i]:
            return
        taken[i] = 1
        out.append(i)
    for i in added:
        take(i)
    for i in eos:
        take(i)
    if len(out) > size:
        raise ValueError(f"select_draft_vocab: the {len(out)} added tokens and EOS ids alone exceed |V'| {size}")
    for i in ranked:
        if len(out) >= size:
            break
        take(i)
    i = 0
    while len(out) < size:
        take(i)
        i += 1
    return sorted(out)


def vocab_inputs(target_dir: str):
    """(added token ids in file order, generation_config EOS ids) - what b70-serve passes."""
    with open(os.path.join(target_dir, "tokenizer.json"), encoding="utf-8") as f:
        added = [int(t["id"]) for t in json.load(f)["added_tokens"]]
    eos = []
    gp = os.path.join(target_dir, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            e = json.load(f).get("eos_token_id")
        eos = [int(x) for x in (e if isinstance(e, list) else [e] if e is not None else [])]
    return added, eos


def read_ranked(path):
    """loader::read_ranked_ids: whitespace-separated ids, '#' comment lines skipped."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids += [int(w) for w in s.split()]
    return ids


# ------------------------------------------------------------------------------ arms


def rtn_int8_rows(w):
    """Symmetric int8 per output row: s = amax / 127, q = rne(w / s) clamped to [-127, 127].
    Returns (q int8, s float32 [rows]); a zero row gets s = 0 and q = 0."""
    wf = w.float()
    s = wf.abs().amax(dim=1) / 127.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round(wf / safe[:, None]), -127, 127).to(torch.int8)
    return q, s


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
    x = d.embed[ids.reshape(-1)].float()
    if d.mask_embedding is not None:
        x[ids.reshape(-1) == c.mask_token_id] = d.mask_embedding
    if c.version == 2:
        x = x * c.input_embedding_scale
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


def select_ids(d, hm, logits, anchors, K: int):
    """Greedy draft ids [B, K] from the mask rows' hidden hm [B*K, H] and their head logits
    [B*K, V] (already masked for a draft vocabulary). DFlash 2: top-k, output_multiplier /
    softcap, the bilinear edge scores, the argmax walk (ties to the lower candidate index);
    DFlash (v1): argmax of the logits (ties to the lower id)."""
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
    cand, unary = top_k_fast(logits, kk)
    unary = unary * c.output_multiplier
    if c.final_logit_softcapping > 0:
        cap = c.final_logit_softcapping
        unary = torch.tanh(unary / cap) * cap
    hid = ref.linear(hm, d.w["candidate_selector.hidden_projection.weight"])
    cand, unary, hid = cand.view(B, K, kk), unary.view(B, K, kk), hid.view(B, K, -1)
    anc = torch.as_tensor(anchors, dtype=torch.long)
    prev = torch.cat([anc[:, None, None].expand(B, 1, kk), cand[:, :-1]], 1)            # [B, K, k]
    P = d.w["candidate_selector.predecessor_codebook"][prev].float() * hid.float()[:, :, None, :]
    S_ = d.w["candidate_selector.successor_codebook"][cand].float()
    S = unary.float()[:, :, None, :] + torch.einsum("bjar,bjcr->bjac", P, S_)            # [B, K, k, k]
    chosen = torch.zeros(B, dtype=torch.long)
    ar = torch.arange(B)
    out = []
    idx = torch.arange(kk)
    for j in range(K):
        row = S[ar, j, chosen].double()                                                  # [B, k]
        best = row.max(-1, keepdim=True).values
        chosen = torch.where(row == best, idx[None], kk).min(-1).values
        out.append(cand[ar, j, chosen])
    return torch.stack(out, 1)


def draft_batch(d, head, anchors, ps, K: int, ctx, masks: dict):
    """{variant: ids [B, K]} for one batch; masks maps a variant name to a bool [V] keep-mask
    (None = the full head)."""
    h = forward_blocks(d, anchors, ps, K, ctx)
    B = len(anchors)
    hm = h[:, 1:].reshape(B * K, -1)
    logits = head.logits(hm)
    out = {}
    for name, keep in masks.items():
        lg = logits if keep is None else logits.masked_fill(~keep[None], float("-inf"))
        out[name] = select_ids(d, hm, lg, anchors, K)
    return out


# ------------------------------------------------------------------------------ the Muse-Glimmer form (spec 23f)


def rtn_int4_g64(w):
    """loader/rtn.h's rtn_int4_g64, restated: w [N][K] (bf16 values) -> q uint8 [N][K] in 0..15 and the f16 scales
    [N][K/64]: per output row n and group of 64 k, amax = max |w|, s16 = f16(2 amax / 15) (round to nearest even,
    subnormals kept), q = clamp(rint(w / f32(s16)) + 8, 0, 15) (q = 8 where s16 = 0). Dequant (q - 8) x s16."""
    N, K = w.shape
    if K % 64:
        raise ValueError(f"int4 g64: K {K} is not a multiple of 64")
    wf = w.float().reshape(N, K // 64, 64)
    s16 = ((2.0 * wf.abs().amax(-1)) / 15.0).to(torch.float16)
    s = s16.float()[..., None]
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))) + 8.0, torch.full_like(wf, 8.0))
    q = q.clamp(0.0, 15.0)
    return q.to(torch.uint8).reshape(N, K), s16


def int4g64_dequant(w):
    q, s16 = rtn_int4_g64(w)
    N, K = w.shape
    return ((q.float().reshape(N, K // 64, 64) - 8.0) * s16.float()[..., None]).reshape(N, K)


def int4g64_simulate(drafter) -> int:
    """Every drafter linear replaced by its int4 g64 RTN dequantisation (float32); the norms stay bf16-widened."""
    n = 0
    for name in list(drafter.w):
        if name.endswith(LINEAR_SUFFIXES) and drafter.w[name].dim() == 2:
            drafter.w[name] = int4g64_dequant(drafter.w[name]).to(drafter.w[name].dtype)
            n += 1
    return n


def _muse_attend(d, i, q, k, v, vis):
    """Drafter._m_attend with a per-block mask: q [B, nh, R, hd], k / v [B, nkv, Kv, hd], vis [B, R, Kv] bool."""
    c = d.cfg
    g = c.n_heads // c.n_kv_heads
    B, nkv, kv_len, hd = k.shape
    if g > 1:
        k = k[:, :, None].expand(B, nkv, g, kv_len, hd).reshape(B, nkv * g, kv_len, hd)
        v = v[:, :, None].expand(B, nkv, g, kv_len, hd).reshape(B, nkv * g, kv_len, hd)
    s = torch.matmul(q, k.transpose(2, 3)) * (c.head_dim ** -0.5)
    s = s.masked_fill(~vis[:, None], float("-inf"))
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    o = torch.matmul(p, v).transpose(1, 2).contiguous()
    return d._m_lin(o.reshape(B, o.shape[1], -1), f"layers.{i}.self_attn.o_proj.weight")


def forward_blocks_muse(d, anchors, ps, ctx):
    """B Muse blocks of 16 rows (row 0 the anchor at ps[b], rows 1..15 the mask at ps[b] + 1 ..), each over the
    context positions in [ps[b] - window, ps[b] - 1] and its own rows (bidirectional) - Drafter._m_draft_block per
    block, stacked. Returns the final-norm hidden [B, 16, H] in the drafter's dtype."""
    c = d.cfg
    B, R, W = len(anchors), c.block_size, c.windows[0]
    lim = W if c.window_inclusive else W - 1
    ids = torch.cat([torch.as_tensor(anchors, dtype=torch.long)[:, None],
                     torch.full((B, R - 1), c.mask_token_id, dtype=torch.long)], 1)
    x = d.embed[ids].to(d.dtype)                                              # [B, R, H]
    ps = torch.as_tensor(ps, dtype=torch.long)
    qpos = ps[:, None] + torch.arange(R)[None]                                # [B, R]
    sel = (ctx.pos >= int(ps.min()) - lim) & (ctx.pos < int(ps.max()))
    cpos = ctx.pos[sel]
    vis_c = (cpos[None, None, :] < ps[:, None, None]) & ((qpos[:, :, None] - cpos[None, None, :]).abs() <= lim)
    vis = torch.cat([vis_c, torch.ones(B, R, R, dtype=torch.bool)], -1)       # [B, R, Nc + R]
    cos, sin = d._m_cos_sin(qpos.reshape(-1))
    cos, sin = cos.view(B, 1, R, -1), sin.view(B, 1, R, -1)
    resid = x
    for i in range(c.n_layers):
        L = f"layers.{i}."
        h = d._m_norm(resid, d.w[L + "input_layernorm.weight"])
        q = d._m_rope(d._m_heads(i, h, "q"), cos, sin)
        kb = d._m_rope(d._m_heads(i, h, "k"), cos, sin)
        vb = d._m_heads(i, h, "v")
        kc = ctx.k[i][sel].permute(1, 0, 2)[None].expand(B, -1, -1, -1)
        vc = ctx.v[i][sel].permute(1, 0, 2)[None].expand(B, -1, -1, -1)
        resid = resid + _muse_attend(d, i, q, torch.cat([kc, kb], 2), torch.cat([vc, vb], 2), vis)
        resid = resid + d._m_mlp(i, d._m_norm(resid, d.w[L + "post_attention_layernorm.weight"]))
    return d._m_norm(resid, d.w["norm.weight"])


def muse_draft_logits(d, head, anchors, ps, ctx):
    """The raw head rows of mask rows 1..15 for B anchors: float32 [B, 15, V] (Head: bf16 / int8 rows)."""
    h = forward_blocks_muse(d, anchors, ps, ctx)
    B, R, H = h.shape
    return head.logits(h[:, 1:].reshape(B * (R - 1), H)).view(B, R - 1, -1)


def greedy_ids(logits):
    """argmax over the last dim, the lowest id among equal maxima (torch.argmax's rule, made explicit)."""
    best = logits.max(-1, keepdim=True).values
    return torch.where(logits == best, torch.arange(logits.shape[-1]), logits.shape[-1]).min(-1).values


def q_rows(logits, form: str, mult: float, cap: float, T: float, k: int = 16):
    """The draft distribution the engine samples from (spec 23 decision 11): the row's top-k candidates (by raw logit,
    ties lower id) with softmax(x / T) over x = the raw logit (`raw`) or 20 tanh(z mult / 20) (`capped`). Returns
    (ids [R, k], probs [R, k])."""
    ids, v = top_k_fast(logits, k)
    if form == "capped":
        v = torch.tanh(v * mult / cap) * cap
    elif form != "raw":
        raise ValueError(f"--q {form}: raw or capped")
    return ids, torch.softmax(v.double() / T, -1)


def p_rows(top_ids, top_logits, temp: float, top_k: int, top_p: float):
    """The target's filtered distribution from the dump's capped top-64 (vLLM's order: temperature, top-k, top-p -
    mtp_accept.filt). Returns (ids [R, k], probs [R, k])."""
    k = min(top_k, top_logits.shape[1])
    v = top_logits[:, :k].double() / temp
    p = torch.softmax(v, -1)
    cum = p.cumsum(-1)
    p = p * ((cum - p) < top_p)
    return top_ids[:, :k], p / p.sum(-1, keepdim=True)


def overlap(pi, pp, qi, qp):
    """sum_x min(p(x), q(x)) per row (the Leviathan rule's exact acceptance probability)."""
    m = pi[:, :, None] == qi[:, None, :]
    return (torch.minimum(pp[:, :, None], qp[:, None, :]) * m).sum((1, 2))


def run_source_muse(d, head, dump, ks, batch: int, kmax: int, sp: dict, qforms):
    """Per-K greedy stats (the same layout run_source writes) + per q form the sampled sums for one dump; one
    forward per anchor batch gives every K."""
    taps = dump["taps"]
    f = dump["tap_from"]
    ctx = d.context_kv({i: taps[i] for i in d.cfg.target_layer_ids}, torch.arange(f, dump["n"]))
    x, g = dump["ids"], dump["greedy"]
    anchors = anchors_of(dump["n_prompt"], dump["n"], kmax)
    res = {}
    if not anchors:
        return res, 0
    ps_all = torch.tensor(anchors)
    R = d.cfg.block_size - 1
    drafts = []
    ssum = {qf: [0.0] * R for qf in qforms}
    scnt = [0] * R
    for b0 in range(0, len(anchors), batch):
        ps = ps_all[b0:b0 + batch]
        lg = muse_draft_logits(d, head, x[ps].tolist(), ps, ctx)              # [B, 15, V]
        drafts.append(greedy_ids(lg))
        for j in range(R):
            rows = ps + j - f                                                 # the target's row after p + j
            ok = (ps + j) <= dump["n"] - 1
            if not bool(ok.any()):
                continue
            rr = rows[ok]
            pi, pp = p_rows(dump["top_ids"][rr], dump["top_logits"][rr], sp["temp"], sp["top_k"], sp["top_p"])
            scnt[j] += int(ok.sum())
            for qf in qforms:
                qi, qp = q_rows(lg[ok, j], qf, sp["mult"], sp["cap"], sp["temp"])
                ssum[qf][j] += float(overlap(pi, pp, qi, qp).sum())
    dr = torch.cat(drafts)                                                    # [A, 15]
    for K in ks:
        length, cens, counted, accepted = accept_rows(dr[:, :K], ps_all, x, g, f)
        hist = torch.bincount(length[~cens], minlength=K + 1).tolist()
        res[str(K)] = {"rows": len(anchors), "censored": int(cens.sum()), "hist": hist,
                       "counted": counted, "accepted": accepted}
    res["sampled"] = {qf: {"counted": scnt, "sum": ssum[qf]} for qf in qforms}
    n_p = dump["n_prompt"]
    agree = int((x[n_p:] == g[n_p - 1 - f: dump["n"] - 1 - f]).sum())
    return res, agree


def pool_sampled(entries, kmax: int) -> dict:
    """{q form: {"alpha": [kmax], "E": [E_1..E_kmax]}} pooled over sources."""
    out = {}
    forms = sorted({qf for e in entries for qf in (e.get("sampled") or {})})
    for qf in forms:
        cnt = [sum(e["sampled"][qf]["counted"][j] for e in entries if "sampled" in e) for j in range(kmax)]
        tot = [sum(e["sampled"][qf]["sum"][j] for e in entries if "sampled" in e) for j in range(kmax)]
        alpha = [t / c if c else None for t, c in zip(tot, cnt)]
        out[qf] = {"alpha": alpha, "E": [expected_tokens(alpha[:K]) for K in range(1, kmax + 1)], "counted": cnt}
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


def anchors_of(n_prompt: int, n: int, kmax: int | None = None):
    """The anchors every K is measured at: n_prompt <= p <= n - kmax (7 for the vLLM drafters, 15 for Muse's)."""
    return list(range(n_prompt, n - (KMAX if kmax is None else kmax) + 1))


def run_source(d, head, dump, ks, masks, batch: int, log=print):
    """Per-variant, per-K stats for one dump."""
    taps = dump["taps"]
    f = dump["tap_from"]
    pos = torch.arange(f, dump["n"])
    ctx = d.context_kv({i: taps[i] for i in d.cfg.target_layer_ids}, pos)
    x, g = dump["ids"], dump["greedy"]
    anchors = anchors_of(dump["n_prompt"], dump["n"])
    res = {name: {} for name in masks}
    if not anchors:
        return res, 0
    ps_all = torch.tensor(anchors)
    for K in ks:
        drafts = {name: [] for name in masks}
        for b0 in range(0, len(anchors), batch):
            ps = ps_all[b0:b0 + batch]
            out = draft_batch(d, head, x[ps].tolist(), ps, K, ctx, masks)
            for name in masks:
                drafts[name].append(out[name])
        for name in masks:
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


def summarize(files, md_out=None, ks=tuple(range(1, KMAX + 1))) -> dict:
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
            sm = pool_sampled(groups[corp], ks[-1])
            if sm:
                summary[variant][corp]["sampled"] = sm
    lines = []
    corpora = sorted({c for v in summary.values() for c in v}, key=lambda c: (c == "all", c))
    for corp in corpora:
        ag = [agree[n] for n in agree if corp == "all" or n.split("/")[0] == corp]
        txt = (f", recorded text = bf16 greedy at {sum(a for a, _ in ag) / max(1, sum(t for _, t in ag)):.3f} "
               f"of {sum(t for _, t in ag)} ids") if ag else ""
        lines.append(f"\n### {corp}{txt}\n")
        kx = ks[-1]
        lines.append("| arm | rows | " + " | ".join(f"E_{K}" for K in ks) + f" | best K | alpha_1..{kx} at K = {kx} |")
        lines.append("|---|---:|" + "---:|" * len(ks) + "---:|---|")
        for variant in sorted(summary, key=_arm_order):
            s = summary[variant].get(corp)
            if not s:
                continue
            es = [s[str(K)]["E"] if str(K) in s else None for K in ks]
            best = max((e, K) for e, K in zip(es, ks) if e is not None)
            a7 = s.get(str(kx), {}).get("alpha") or []
            rows = s.get(str(ks[0]), {}).get("rows", 0)
            lines.append(f"| {variant} | {rows} | " + " | ".join(f"{e:.2f}" if e is not None else "-" for e in es)
                         + f" | {best[1]} | " + " ".join(f"{a:.2f}" if a is not None else "-" for a in a7) + " |")
        sampled = [(v, summary[v][corp]["sampled"]) for v in sorted(summary, key=_arm_order)
                   if corp in summary[v] and "sampled" in summary[v][corp]]
        if sampled:
            lines.append(f"\nSampled (the expected-acceptance formula, q = the draft row's top 16 at T): E_K\n")
            lines.append("| arm | q | " + " | ".join(f"E_{K}" for K in ks) + " | best K |")
            lines.append("|---|---|" + "---:|" * len(ks) + "---:|")
            for v, sm in sampled:
                for qf, r in sm.items():
                    es = [r["E"][K - 1] for K in ks]
                    best = max((e, K) for e, K in zip(es, ks))
                    lines.append(f"| {v} | {qf} | " + " | ".join(f"{e:.2f}" for e in es) + f" | {best[1]} |")
    text = "\n".join(lines)
    print(text)
    if md_out:
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return summary


def _arm_order(v):
    base = ["bf16", "int8", "int8h", "w4a16", "int4g64", "int4g64h", "dflash2"]
    b = v.split("-")[0]
    return (base.index(b) if b in base else 9, v)


# ------------------------------------------------------------------------------ main


def plan_arms(arms, vocab, vocab_on, ranked: bool):
    """{arm: [variant names]}: each base arm, plus its draft-vocabulary variants (unranked, and
    ranked when a ranked list is given)."""
    out = {}
    for a in arms:
        out[a] = [a]
        if a in vocab_on:
            out[a] += [f"{a}-v{s}{r}" for r in ([""] + (["-r"] if ranked else [])) for s in vocab]
    return out


def dump_files(dumps_dir, pattern):
    fs = sorted(glob.glob(os.path.join(dumps_dir, "*.taps.safetensors")))
    if pattern:
        import fnmatch
        fs = [f for f in fs if fnmatch.fnmatch(os.path.basename(f), pattern)]
    return fs


MUSE_ARMS = ("bf16", "int8", "int8h", "int4g64", "int4g64h", "dflash2")
MUSE_DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"
MUSE_TARGETS = ("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound", "Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
MUSE_DFLASH2 = "incoai/Muse-Glimmer-30B-DFlash2"


def muse_sampling(target_dir: str, a) -> dict:
    """The sampled estimate's parameters: --temperature / --top-k / --top-p over generation_config.json's (Muse: 1.0 /
    64 / 0.95, spec 23 decision 4), and the target's multiplier / softcap (text_config: 0.19611613513818404 / 20)."""
    gc = {}
    gp = os.path.join(target_dir, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            gc = json.load(f)
    with open(os.path.join(target_dir, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    tc = c.get("text_config", c)
    return {"temp": a.temperature if a.temperature is not None else float(gc.get("temperature", 1.0)),
            "top_k": a.top_k if a.top_k is not None else int(gc.get("top_k", 64)),
            "top_p": a.top_p if a.top_p is not None else float(gc.get("top_p", 0.95)),
            "mult": float(tc["output_multiplier"]), "cap": float(tc["final_logit_softcapping"])}


def cmd_run_muse(a) -> None:
    """--model muse: the Muse-Glimmer assistant's arms over muse_taps.py's dumps (module docstring)."""
    arms = [s for s in a.arms.split(",") if s]
    bad = [s for s in arms if s not in MUSE_ARMS]
    if bad:
        raise SystemExit(f"--model muse: unknown arms {bad} (of {MUSE_ARMS})")
    kmax = a.kmax or 15
    lo, _, hi = a.k.partition("-")
    ks = list(range(int(lo), int(hi or lo) + 1))
    if not (1 <= ks[0] and ks[-1] <= kmax <= 15):
        raise SystemExit(f"--k within 1-{kmax} (the block's 15 drafts)")
    qforms = [q for q in a.q.split(",") if q]
    if any(q not in ("raw", "capped") for q in qforms):
        raise SystemExit("--q takes raw,capped")
    files = dump_files(a.dumps, a.sources)
    os.makedirs(a.out_dir, exist_ok=True)
    if not files:
        raise SystemExit(f"no *.taps.safetensors in {a.dumps} (muse_oracle.sh <data> taps)")
    _load_deps()
    heads = [dt.dump_header(f) for f in files]
    n_anchor = sum(len(anchors_of(int(h["n_prompt"]), int(h["n"]), kmax)) for h in heads)
    gflop = n_anchor * (16 * 2 * 2.556e9 + 15 * 2 * 6656 * 202048) / 1e9
    print(f"--model muse: {len(files)} dumps, {n_anchor} anchors (one 16-row block each: K 1..{kmax} from one "
          f"forward); arms {arms}; q {qforms}; {gflop / 1e3:.1f} TFLOP a base arm, ~{gflop / ASSUMED_GFLOPS / 60:.0f} "
          f"min at {ASSUMED_GFLOPS:.0f} GFLOP/s (assumed)")
    print("memory (estimated): the drafter float32 10.2 GB (+ the bf16 load 5.1 GB transient), the target embedding "
          "bf16 2.7 GB, the head float32 5.4 GB (h arms: int8 1.3 GB) -> ~20-24 GiB peak")
    if a.dry_run:
        return
    torch.set_grad_enabled(False)
    target = a.target
    if target is None:
        for rid in MUSE_TARGETS:
            target = target or ref.find_snapshot(rid, ("config.json", "model.safetensors.index.json"))
    drafter_dir = a.drafter or ref.find_snapshot(MUSE_DRAFTER, ("config.json", "model.safetensors"))
    if target is None or drafter_dir is None:
        raise SystemExit(f"target {target} / drafter {drafter_dir} not found in {ref.hf_hub_dir()}")
    sp = muse_sampling(target, a)
    t0 = time.time()
    embed, head_w = ref.read_target_embed_head(target)
    print(f"target embed / head {time.time() - t0:.0f} s ({target}); sampled at T {sp['temp']}, top-k {sp['top_k']}, "
          f"top-p {sp['top_p']}; the target's cap {sp['mult']} / {sp['cap']}", flush=True)
    for arm in arms:
        out_path = os.path.join(a.out_dir, f"accept.{arm}.json")
        state = {"arm": arm, "variants": {arm: {}}, "agree": {}, "meta": {}}
        if os.path.isfile(out_path):
            with open(out_path, encoding="utf-8") as f:
                old = json.load(f)
            state["variants"][arm].update(old["variants"].get(arm, {}))
            state["agree"].update(old.get("agree", {}))
        todo = [f for f in files if dt.dump_header(f)["name"] not in state["variants"][arm]]
        if not todo:
            print(f"[{arm}] done", flush=True)
            continue
        dd = None
        if arm == "dflash2":
            dd = a.dflash2 or ref.find_snapshot(MUSE_DFLASH2, ("config.json", "model.safetensors"))
            if dd is None:
                print(f"[{arm}] SKIP: {MUSE_DFLASH2} is not in the cache (decision 12's optional arm; never "
                      "downloaded here)", flush=True)
                continue
            cfg2 = ref.read_config(dd)
            layers = [int(x) for x in dt.dump_header(todo[0])["layers"].split(",")]
            if not set(cfg2.target_layer_ids) <= set(layers):
                print(f"[{arm}] SKIP: its taps {cfg2.target_layer_ids} are not the dumps' {layers}", flush=True)
                continue
        ta = time.time()
        if dd is not None:
            d = ref.load_drafter(dd, embed=embed, lm_head=head_w)
        else:
            d = ref.load_drafter(drafter_dir, embed=embed, lm_head=head_w, dtype=torch.float32)
        base = arm[:-1] if arm.endswith("h") else arm
        nq = int8_simulate(d) if base == "int8" else int4g64_simulate(d) if base == "int4g64" else 0
        head = Head(head_w, int8=arm.endswith("h"))
        state["meta"] = {"drafter": dd or drafter_dir, "target": target, "quantised": nq,
                         "head": "int8" if head.int8 else "bf16", "ks": ks, "kmax": kmax, "batch": a.batch,
                         "sampling": sp, "q": qforms, "threads": torch.get_num_threads()}
        print(f"[{arm}] drafter loaded {time.time() - ta:.0f} s ({nq} linears quantised), head {state['meta']['head']}",
              flush=True)
        for fn in todo:
            dump = dt.load_dump(fn)
            ts = time.time()
            if dd is not None:
                r, agree = run_source(d, head, dump, [k for k in ks if k <= d.cfg.max_k], {arm: None}, a.batch)
                r = r[arm]
            else:
                r, agree = run_source_muse(d, head, dump, ks, a.batch, kmax, sp, qforms)
            state["variants"][arm][dump["name"]] = r
            state["agree"][dump["name"]] = [agree, dump["n"] - dump["n_prompt"]]
            with open(out_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(state, f, indent=1)
            os.replace(out_path + ".tmp", out_path)
            e = pool([r], ks)
            es = " ".join("%.2f" % e[str(K)]["E"] for K in ks if str(K) in e)
            print(f"[{arm}] {dump['name']}: {len(anchors_of(dump['n_prompt'], dump['n'], kmax))} anchors, E_K {es}; "
                  f"{time.time() - ts:.0f} s", flush=True)
        del d, head
    files_out = sorted(glob.glob(os.path.join(a.out_dir, "accept.*.json")))
    summarize(files_out, os.path.join(a.out_dir, "summary.md"), ks)
    print(f"wall {time.time() - t0:.0f} s", flush=True)


def cmd_run(a) -> None:
    if getattr(a, "model", "qwen38") == "muse":
        return cmd_run_muse(a)
    arms = [s for s in a.arms.split(",") if s]
    bad = [s for s in arms if s not in ("bf16", "int8", "int8h", "w4a16")]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    vocab = [s for s in a.vocab.split(",") if s] if a.vocab else []
    if any(s not in VOCAB_SIZES for s in vocab):
        raise SystemExit(f"--vocab takes {sorted(VOCAB_SIZES)}")
    vocab_on = [s for s in a.vocab_on.split(",") if s]
    lo, _, hi = a.k.partition("-")
    ks = list(range(int(lo), int(hi or lo) + 1))
    if not (1 <= ks[0] and ks[-1] <= KMAX):
        raise SystemExit(f"--k within 1-{KMAX}")
    plan = plan_arms(arms, vocab, vocab_on, bool(a.draft_vocab_ids))
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
    drafter_dir = a.drafter or ref.find_snapshot("z-lab/Qwen3.8-27B-DFlash2", ("config.json", "model.safetensors"))
    w4_dir = a.w4a16 or ref.find_snapshot("syvai/Qwen3.8-27B-DFlash2-W4A16", ("config.json", "model.safetensors"))
    if target is None or drafter_dir is None:
        raise SystemExit(f"target {target} / drafter {drafter_dir} not found in {ref.hf_hub_dir()}")
    t0 = time.time()
    embed, head_w = ref.read_target_embed_head(target)
    vocab_used = a.vocab_used
    added, eos = vocab_inputs(target)
    ranked = read_ranked(a.draft_vocab_ids) if a.draft_vocab_ids else []
    vmasks = {}
    for s in vocab:
        for suffix, rk in (("", []), ("-r", ranked)) if ranked else (("", []),):
            ids = select_draft_vocab(added, eos, rk, vocab_used, VOCAB_SIZES[s])
            keep = torch.zeros(head_w.shape[0], dtype=torch.bool)
            keep[torch.tensor(ids)] = True
            vmasks[f"v{s}{suffix}"] = keep
    print(f"target embed / head {time.time() - t0:.0f} s; V' {[(k, int(m.sum())) for k, m in vmasks.items()]}", flush=True)
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
        d = ref.load_drafter(src_dir, embed=embed, lm_head=head_w)
        nq = int8_simulate(d) if arm in ("int8", "int8h") else 0
        head = Head(head_w, int8=(arm == "int8h"))
        masks = {arm: None}
        for v in variants[1:]:
            masks[v] = vmasks[v[len(arm) + 1:]]
        state["meta"] = {"drafter": src_dir, "target": target, "int8_tensors": nq, "head": "int8" if head.int8 else "bf16",
                         "ks": ks, "batch": a.batch, "vocab_used": vocab_used,
                         "draft_vocab_ids": a.draft_vocab_ids or "", "threads": torch.get_num_threads()}
        print(f"[{arm}] drafter loaded {time.time() - ta:.0f} s ({nq} linears int8), head {state['meta']['head']}", flush=True)
        for fn in todo:
            dump = dt.load_dump(fn)
            ts = time.time()
            res, agree = run_source(d, head, dump, ks, masks, a.batch)
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
    summarize(files_out, os.path.join(a.out_dir, "summary.md"), ks)
    print(f"wall {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dumps", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--arms", default="bf16,int8,int8h,w4a16")
    r.add_argument("--vocab", default="32k,64k,128k")
    r.add_argument("--vocab-on", default="bf16")
    r.add_argument("--draft-vocab-ids")
    r.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    r.add_argument("--k", default=f"1-{KMAX}")
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--drafter")
    r.add_argument("--w4a16")
    r.add_argument("--target")
    r.add_argument("--sources", help="fnmatch over the dump file names")
    r.add_argument("--dry-run", action="store_true")
    # spec 23f: the Muse-Glimmer form
    r.add_argument("--model", default="qwen38", choices=("qwen38", "muse"))
    r.add_argument("--kmax", type=int, help="--model muse: the anchors leave this many drafts (default 15)")
    r.add_argument("--q", default="raw,capped", help="--model muse: the sampled estimate's q forms")
    r.add_argument("--temperature", type=float)
    r.add_argument("--top-k", type=int)
    r.add_argument("--top-p", type=float)
    r.add_argument("--dflash2", help="--model muse: incoai/Muse-Glimmer-30B-DFlash2's snapshot (decision 12)")
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--md")
    s.add_argument("--kmax", type=int, default=KMAX)
    a = ap.parse_args()
    if a.cmd == "summary":
        summarize(a.files, a.md, tuple(range(1, a.kmax + 1)))
    else:
        if a.model == "muse":
            if a.arms == "bf16,int8,int8h,w4a16":
                a.arms = ",".join(MUSE_ARMS)
            if a.k == f"1-{KMAX}":
                a.k = f"1-{a.kmax or 15}"
        cmd_run(a)


if __name__ == "__main__":
    main()
