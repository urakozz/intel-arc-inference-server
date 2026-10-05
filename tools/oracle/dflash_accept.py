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
    text = "\n".join(lines)
    print(text)
    if md_out:
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return summary


def _arm_order(v):
    base = ["bf16", "int8", "int8h", "w4a16"]
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


def cmd_run(a) -> None:
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
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "summary":
        summarize(a.files, a.md)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
