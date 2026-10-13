#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 22 P0.6 (plan 22a Task 3): the expert-offload tier's hit-rate curves from spec 21a's routing traces, and
the routing profile plan 22b's loader reads (decision 8).

    offload_curve.py counts  --traces DIR [--json FILE]
    offload_curve.py curve   --traces DIR [--split hash|none] [--f 0.30:1.00:0.01] [--budget FILE] [--json FILE]
    offload_curve.py swap    --traces DIR --f F --S 1,4,16,64 --B <MB list> --copy-ms X --step-ms Y [--W 64,256,1024]
    offload_curve.py mtp     --traces DIR --K 1,2,3,4 --f F [--json FILE]
    offload_curve.py prefill --traces DIR --chunk 2048,4096 --f 0.6:0.9:0.05 [--json FILE]
    offload_curve.py profile --traces DIR --out FILE [--sources CLASS|NAME,...]

The traces are `<DIR>/<name>.routes.safetensors` (21a's TRACE_FORMAT, qwen4exp_ref.py trace): ids i32 [T][L][k]
ascending, p f32, onorm f32, mtp_ids / mtp_p / mtp_onorm [T][k] (row T-1 of the head -1 / 0); metadata layers,
experts, top_k, source. A source's class is A4 (`t<N>_<task>-<file>`, the 36 tool-call scenarios), agentic
(q4exp_agentic), code, prose, or opencode (anything else: the recorded sessions).

**Hold-out (Review Focus 4).** Sources split into two halves by sha256(name) - the A4 scenarios by the parity of
their rank in sha256 order, so 18 / 18 - and the other half scores: held out = the mean of (built on half 0,
scored on half 1) and (1 on 0); in sample = the mean of each half scored on its own profile. Both are printed;
`--split none` builds on everything and scores everything (in sample only).

**Fills.** Every expert of a layer has the same bytes (2,611,200 at int4 g64), so a byte budget is a slot count.
Uniform: round(f x E) residents a layer, the top by the profile's counts (ties to the lower id). Per layer: the
global greedy over all (layer, expert) pairs by the profile's counts - the marginal hits per byte, layer by layer's
own curve - round(f x L x E) slots in total (ties: lower layer, lower id). h = routed slots that hit / all routed
slots, over every layer and row.

Every table is printed and, with --json FILE, written as JSON for the record (docs/probe-offload-<date>.md).

P0.8 (plan 22a Task 4), the pruned mode's masks and their curve:

    offload_curve.py reap --traces DIR --sources A4,agentic,opencode[,code] --kept 0.5,0.625,0.75,0.875 --out DIR
                          [--criterion reap|freq|reap,freq] [--holdout-a4 | --kfold 4 --fold i|all]
    offload_curve.py prune-report --dir <maskeval JSONs> [--a4 <A4 outputs>] [--budget FILE] [--a4-bar B] [--json F]

`reap`: score(l, e) = the sum over the calibration tokens routed to e of p x onorm (REAP; `freq` counts tokens - the
weaker proxy, written for comparison); per layer the top round(E x kept) by score (ties to the lower id) ->
<out>/mask-<criterion>-<kept>[-noa4|-fold<i>].safetensors (spec22-mask-1: keep u8 [L][E], mtp_keep u8 [E] all ones,
metadata model, checkpoint_revision, criterion, kept_fraction, per_layer uniform, calibration = names + sha256 of the
traces used, held_out, fold) and reap.json (the masks, the REAP / freq overlap per layer). Every layer keeps >= top_k
(refused otherwise). --holdout-a4 calibrates without the A4 scenarios (Review Focus 4's rule on P0.8). **Decision 9's
rule (operator, 2026-10-10):** --kfold 4 --fold i calibrates without fold i's A4 scenarios (the 36 in 4 fixed folds by
their sha256 order, a4_folds; every other source kept), so maskeval / the A4 run grade each fold on a mask that never
saw it; the 4-fold held-out curve picks the kept fraction, the all-traces mask at that fraction ships, and
prune-report prints the in-sample minus held-out gap per fraction.
`prune-report`: kept fraction x {KL mean / p99 and argmax on the golden set and on A4 (in sample / held out), A4
tool-call accuracy by score.py's rule against the full run (.bf16) in sample and 4-fold held out and their gap, the
experts' bytes at int4 g64, whether they fit the two-card cache at 32k (Task 1's budget, the head and prefill scratch)}.
`project` (P0.7) is below.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# tools/oracle holds a tokenize.py: keep the script's own directory off sys.path (it would shadow the stdlib's)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

EXPERT_BYTES = 1740800 + 870400        # loader/qwen4exp_layout.h: q4_gate_up_block_bytes + q4_down_block_bytes
A4_RE = re.compile(r"^t\d+_[A-Za-z]+-[A-Za-z0-9_]+$")
CLASSES = ("A4", "agentic", "code", "prose", "opencode")


def source_class(name: str) -> str:
    if A4_RE.match(name):
        return "A4"
    if name == "q4exp_agentic":
        return "agentic"
    if name in ("code", "prose"):
        return name
    return "opencode"


def _h(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def halves(names) -> dict:
    """name -> 0 | 1: sha256(name)'s parity; the A4 scenarios alternate in sha256 order (balanced)."""
    out = {}
    a4 = sorted((n for n in names if source_class(n) == "A4"), key=_h)
    for i, n in enumerate(a4):
        out[n] = i % 2
    for n in names:
        if n not in out:
            out[n] = int(_h(n)[-1], 16) % 2
    return out


def a4_folds(names, k: int = 4) -> dict:
    """The A4 scenarios' fixed folds: rank in sha256 order mod k (decision 9's 4-fold rule; 9 a fold at 36)."""
    a4 = sorted((n for n in names if source_class(n) == "A4"), key=_h)
    return {n: i % k for i, n in enumerate(a4)}


class Trace:
    def __init__(self, path: str):
        from safetensors import safe_open
        with safe_open(path, framework="np") as f:
            self.meta = dict(f.metadata() or {})
            keys = set(f.keys())
            self.ids = f.get_tensor("ids").astype(np.int64)
            self.p = f.get_tensor("p").astype(np.float32) if "p" in keys else None
            self.onorm = f.get_tensor("onorm").astype(np.float32) if "onorm" in keys else None
            self.mtp_ids = f.get_tensor("mtp_ids").astype(np.int64) if "mtp_ids" in keys else None
            self.mtp_p = f.get_tensor("mtp_p").astype(np.float32) if "mtp_p" in keys else None
            self.mtp_onorm = f.get_tensor("mtp_onorm").astype(np.float32) if "mtp_onorm" in keys else None
        self.path = path
        self.name = os.path.basename(path)[: -len(".routes.safetensors")]
        self.cls = source_class(self.name)
        self.T, self.layers, self.top_k = (int(x) for x in self.ids.shape)
        self.experts = int(self.meta.get("experts", 512))
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()


def load_traces(d: str, names=None) -> list:
    files = sorted(f for f in os.listdir(d) if f.endswith(".routes.safetensors"))
    out = [Trace(os.path.join(d, f)) for f in files]
    if names is not None:
        out = [t for t in out if t.name in names or t.cls in names]
    if not out:
        raise SystemExit(f"offload_curve: no traces (*.routes.safetensors) in {d}")
    L, E, k = out[0].layers, out[0].experts, out[0].top_k
    for t in out:
        if (t.layers, t.experts, t.top_k) != (L, E, k):
            raise SystemExit(f"offload_curve: {t.name} is L {t.layers} E {t.experts} k {t.top_k}, {out[0].name} "
                             f"L {L} E {E} k {k}")
    return out


def frange(spec: str) -> list:
    """'a:b:s' inclusive of b (rounded to 1e-9), or a comma list."""
    if ":" in spec:
        a, b, s = (float(x) for x in spec.split(":"))
        n = int(math.floor((b - a) / s + 1e-9))
        return [round(a + i * s, 9) for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


def ilist(spec: str) -> list:
    return [int(x) for x in spec.split(",")]


# --- counts and fills -----------------------------------------------------------------------------------------------
def counts_of(traces, L: int, E: int) -> np.ndarray:
    """Routed counts per (layer, expert), int64 [L][E]."""
    c = np.zeros((L, E), np.int64)
    for t in traces:
        flat = (np.arange(L)[None, :, None] * E + t.ids).reshape(-1)
        c += np.bincount(flat, minlength=L * E).reshape(L, E)
    return c


def mtp_counts_of(traces, E: int) -> np.ndarray:
    c = np.zeros(E, np.int64)
    for t in traces:
        if t.mtp_ids is not None:
            m = t.mtp_ids[t.mtp_ids >= 0]
            c += np.bincount(m, minlength=E)
    return c


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """Indices of v by descending value, ties to the lower index (stable)."""
    return np.argsort(-v, kind="stable")


def resident_uniform(counts: np.ndarray, f: float) -> np.ndarray:
    L, E = counts.shape
    n = min(E, max(0, int(round(f * E))))
    res = np.zeros((L, E), bool)
    for l in range(L):
        res[l, _rank_desc(counts[l])[:n]] = True
    return res


def resident_perlayer(counts: np.ndarray, f: float) -> np.ndarray:
    L, E = counts.shape
    n = min(L * E, max(0, int(round(f * L * E))))
    res = np.zeros(L * E, bool)
    res[_rank_desc(counts.reshape(-1))[:n]] = True
    return res.reshape(L, E)


def resident_1d(counts: np.ndarray, f: float) -> np.ndarray:
    E = counts.shape[0]
    res = np.zeros(E, bool)
    res[_rank_desc(counts)[: min(E, max(0, int(round(f * E))))]] = True
    return res


def hit_rate(traces, res: np.ndarray):
    """(hits, total routed slots) of `traces` against resident mask res [L][E]."""
    hits = total = 0
    for t in traces:
        L = t.layers
        h = res[np.arange(L)[None, :, None], t.ids]
        hits += int(h.sum())
        total += int(h.size)
    return hits, total


# --- curve ------------------------------------------------------------------------------------------------------------
def _ratio(h, t):
    return h / t if t else float("nan")


def curve(traces, L: int, E: int, fs, split: str = "hash") -> dict:
    """h(f) for the uniform and per-layer fills, held out and in sample, per source class."""
    names = [t.name for t in traces]
    half = halves(names) if split == "hash" else {n: 0 for n in names}
    groups = [[t for t in traces if half[t.name] == g] for g in (0, 1)]
    if split == "hash" and (not groups[0] or not groups[1]):
        raise SystemExit("offload_curve curve: one half is empty (too few sources for --split hash; use --split none)")
    cnt = [counts_of(g, L, E) if g else None for g in groups]
    allc = counts_of(traces, L, E)
    rows = []
    for f in fs:
        row = {"f": f}
        for pol, fill in (("uniform", resident_uniform), ("perlayer", resident_perlayer)):
            if split == "hash":
                r0, r1 = fill(cnt[0], f), fill(cnt[1], f)
                held = [hit_rate(groups[1], r0), hit_rate(groups[0], r1)]
                ins = [hit_rate(groups[0], r0), hit_rate(groups[1], r1)]
                row[pol + "_held"] = (_ratio(*held[0]) + _ratio(*held[1])) / 2
                row[pol + "_in"] = (_ratio(*ins[0]) + _ratio(*ins[1])) / 2
                for c in CLASSES:   # held out per class: each source scored on the other half's profile
                    hh = tt = 0
                    for g, r_other in ((0, r1), (1, r0)):
                        sel = [t for t in groups[g] if t.cls == c]
                        a, b = hit_rate(sel, r_other)
                        hh, tt = hh + a, tt + b
                    if tt:
                        row[f"{pol}_held_{c}"] = hh / tt
            else:
                r = fill(allc, f)
                row[pol + "_in"] = _ratio(*hit_rate(traces, r))
                row[pol + "_held"] = float("nan")
                for c in CLASSES:
                    sel = [t for t in traces if t.cls == c]
                    a, b = hit_rate(sel, r)
                    if b:
                        row[f"{pol}_in_{c}"] = a / b
        rows.append(row)
    reach = {}
    key = "held" if split == "hash" else "in"
    for pol in ("uniform", "perlayer"):
        for target in (0.90, 0.95, 0.99):
            fr = next((r["f"] for r in rows if r[f"{pol}_{key}"] >= target), None)
            reach[f"{pol}_{key}_{target:.2f}"] = fr
    return {"split": split, "halves": half, "rows": rows, "reach": reach,
            "sources": {t.name: {"class": t.cls, "T": t.T} for t in traces}}


def read_budget(path: str) -> list:
    """Task 1's `budget-json` lines (qwen4exp_plan_test --budget-json FILE)."""
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln.startswith("budget-json "):
                ln = ln[len("budget-json "):]
            if ln.startswith("{"):
                out.append(json.loads(ln))
    if not out:
        raise SystemExit(f"offload_curve: no budget lines in {path}")
    return out


def interp(rows, f: float, key: str) -> float:
    """h at f by linear interpolation of rows' (f, key)."""
    pts = sorted((r["f"], r[key]) for r in rows if not math.isnan(r.get(key, float("nan"))))
    if not pts:
        return float("nan")
    if f <= pts[0][0]:
        return pts[0][1]
    for (f0, h0), (f1, h1) in zip(pts, pts[1:]):
        if f0 <= f <= f1:
            return h0 + (h1 - h0) * (f - f0) / (f1 - f0) if f1 > f0 else h1
    return pts[-1][1]


# --- swap -------------------------------------------------------------------------------------------------------------
def swap_sim(trace, res0: np.ndarray, S: int, B: float, W: int, copy_ms: float, step_ms: float,
             expert_bytes: int = EXPERT_BYTES):
    """Adaptive swaps over one trace's rows as decode steps (spec 22 §4's protocol). Every S steps (when no round is
    in flight) up to floor(B / expert_bytes) experts move: over all layers, the most-routed non-resident experts of
    the last W steps replace the least-routed residents of the same layer (only where the incoming count is higher).
    A victim points at the host at once (a miss from that step); the incoming expert serves from the step after
    ceil(n x copy_ms / step_ms) steps; its slot is the victim's (decisions fall between steps, so the step that last
    used it has retired). A round whose copy cannot land (copy_ms = inf, or nothing to move) is not started: with an
    infinite copy time the result is the static fill exactly. Returns (hits, total)."""
    ids = trace.ids
    T, L, k = ids.shape
    E = res0.shape[1]
    res = res0.copy()
    nmax = int(B // expert_bytes)
    win = np.zeros((L, E), np.int64)
    larange = np.arange(L)[:, None]
    pending = None            # (ready step, [(layer, incoming)])
    hits = 0
    for t in range(T):
        if pending is not None and t >= pending[0]:
            for l, e in pending[1]:
                res[l, e] = True
            pending = None
        if pending is None and t > 0 and t % S == 0 and nmax > 0 and math.isfinite(copy_ms):
            cand = []
            for l in range(L):
                out_c = np.where(res[l], -1, win[l])
                in_c = np.where(res[l], win[l], np.iinfo(np.int64).max)
                order_in = _rank_desc(out_c)
                order_vic = np.argsort(in_c, kind="stable")
                for e_in, e_v in zip(order_in, order_vic):
                    if out_c[e_in] <= 0 or not res[l, e_v] or out_c[e_in] <= in_c[e_v]:
                        break
                    cand.append((int(out_c[e_in]) - int(in_c[e_v]), l, int(e_in), int(e_v)))
            cand.sort(key=lambda c: (-c[0], c[1], c[2]))
            moves = cand[:nmax]
            if moves:
                delay = math.ceil(len(moves) * copy_ms / step_ms)
                for _, l, e_in, e_v in moves:
                    res[l, e_v] = False
                pending = (t + delay + 1, [(l, e_in) for _, l, e_in, _ in moves])
        row = ids[t]
        hits += int(res[larange, row].sum())
        np.add.at(win, (np.repeat(np.arange(L), k), row.reshape(-1)), 1)
        if t >= W:
            old = ids[t - W]
            np.add.at(win, (np.repeat(np.arange(L), k), old.reshape(-1)), -1)
    return hits, T * L * k


# --- MTP verify unions --------------------------------------------------------------------------------------------------
def mtp_unions(trace, K: int, res: np.ndarray) -> dict:
    """Over every window of M = K + 1 consecutive rows (teacher-forced rows stand for accepted drafts): the distinct
    experts a layer (mean over layers and windows), the misses a verify step against res (summed over layers), and
    the iid prior: sum_e 1 - (1 - q_e)^M with q_e the layer's per-row frequency of e (mean over layers)."""
    ids = trace.ids
    T, L, k = ids.shape
    E = res.shape[1]
    M = K + 1
    nwin = T - M + 1
    if nwin <= 0:
        return {"K": K, "M": M, "windows": 0}
    dist = np.zeros(L)
    miss = 0.0
    for t in range(nwin):
        w = ids[t:t + M]                                    # [M][L][k]
        for l in range(L):
            u = np.unique(w[:, l, :])
            dist[l] += u.size
            miss += float((~res[l, u]).sum())
    q = np.zeros((L, E))
    for l in range(L):
        q[l] = np.bincount(ids[:, l, :].reshape(-1), minlength=E) / T
    iid = float(np.mean([(1 - (1 - q[l]) ** M).sum() for l in range(L)]))
    single_miss = float((~res[np.arange(L)[None, :, None], ids]).sum()) / T
    return {"K": K, "M": M, "windows": nwin, "distinct_per_layer": float(dist.sum() / (nwin * L)),
            "iid_per_layer": iid, "misses_per_step": miss / nwin, "misses_single_row": single_miss,
            "misses_M_rows_unshared": single_miss * M}


def mtp_head(traces, train_counts: np.ndarray, f: float) -> dict:
    """The MTP head's own routes: distinct experts, and the hit rate if they were cached with the rest (a profile of
    the head's counts at the same f)."""
    E = train_counts.shape[0]
    res = resident_1d(train_counts, f)
    hits = total = 0
    distinct = set()
    for t in traces:
        if t.mtp_ids is None:
            continue
        m = t.mtp_ids[t.mtp_ids >= 0]
        distinct.update(np.unique(m).tolist())
        hits += int(res[m].sum())
        total += int(m.size)
    return {"distinct": len(distinct), "of": E, "routes": total, "h_cached": _ratio(hits, total)}


# --- prefill ------------------------------------------------------------------------------------------------------------
def prefill_bytes(trace, C: int, res: np.ndarray, expert_bytes: int = EXPERT_BYTES) -> dict:
    """Chunks of C consecutive rows (a source shorter than C is one chunk): the unique experts a layer a chunk and
    the streamed bytes a chunk under res - the experts the chunk touches that are not resident x expert_bytes."""
    ids = trace.ids
    T, L, k = ids.shape
    chunks = max(1, T // C)
    uniq = streamed = 0.0
    for c in range(chunks):
        w = ids[c * C:(c + 1) * C] if T >= C else ids
        for l in range(L):
            u = np.unique(w[:, l, :])
            uniq += u.size
            streamed += float((~res[l, u]).sum()) * expert_bytes
    return {"C": C, "chunks": chunks, "short": T < C, "unique_per_layer": uniq / (chunks * L),
            "streamed_bytes_per_chunk": streamed / chunks}


# --- profile --------------------------------------------------------------------------------------------------------------
def write_profile(path: str, traces, L: int, E: int) -> dict:
    from safetensors.numpy import save_file
    c = counts_of(traces, L, E)
    mc = mtp_counts_of(traces, E)
    if c.max(initial=0) >= 2 ** 32 or mc.max(initial=0) >= 2 ** 32:
        raise SystemExit("offload_curve profile: a count does not fit u32")
    t0 = traces[0]
    meta = {"model": t0.meta.get("model", ""), "checkpoint": t0.meta.get("checkpoint", ""),
            "checkpoint_revision": t0.meta.get("checkpoint_revision", ""),
            "traces": json.dumps([{"name": t.name, "sha256": t.sha256} for t in traces]),
            "sources": json.dumps(sorted({t.cls for t in traces})), "tokens": str(sum(t.T for t in traces)),
            "layers": str(L), "experts": str(E), "top_k": str(t0.top_k), "format": "spec22-profile-1"}
    save_file({"counts": c.astype(np.uint32), "mtp_counts": mc.astype(np.uint32)}, path, metadata=meta)
    return meta


def read_profile(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        meta = dict(f.metadata() or {})
        if meta.get("format") != "spec22-profile-1":
            raise SystemExit(f"offload_curve: {path} is not a spec22-profile-1 file ({meta.get('format')})")
        return {"counts": f.get_tensor("counts"), "mtp_counts": f.get_tensor("mtp_counts"), "meta": meta}


# --- P0.8: REAP masks (plan 22a Task 4) -----------------------------------------------------------------------------------
MASK_FORMAT = "spec22-mask-1"


def reap_scores(traces, L: int, E: int, criterion: str = "reap") -> np.ndarray:
    """score(l, e) over the calibration traces' routed tokens: REAP = sum of p x onorm (the router weight times the
    norm of the expert's output); freq = the routed count (the weaker proxy, for comparison). float64 [L][E]."""
    s = np.zeros((L, E), np.float64)
    for t in traces:
        flat = (np.arange(L)[None, :, None] * E + t.ids).reshape(-1)
        if criterion == "reap":
            if t.p is None or t.onorm is None:
                raise SystemExit(f"offload_curve reap: {t.name} has no p / onorm (21a's trace writes both)")
            w = (t.p.astype(np.float64) * t.onorm.astype(np.float64)).reshape(-1)
        elif criterion == "freq":
            w = np.ones(flat.shape[0])
        else:
            raise SystemExit(f"offload_curve reap: criterion {criterion!r} (reap | freq)")
        s += np.bincount(flat, weights=w, minlength=L * E).reshape(L, E)
    return s


def mask_keep(scores: np.ndarray, kept: float, top_k: int | None = None) -> np.ndarray:
    """Per layer the top round(E x kept) experts by score, ties to the lower id. Refuses a mask below top_k."""
    L, E = scores.shape
    n = int(round(E * kept))
    if top_k is not None and n < top_k:
        raise SystemExit(f"offload_curve reap: kept {kept:g} - layer 0 keeps {n} experts, fewer than top_k {top_k}")
    keep = np.zeros((L, E), bool)
    for l in range(L):
        keep[l, _rank_desc(scores[l])[:n]] = True
    return keep


def write_mask(path: str, keep: np.ndarray, mtp_keep: np.ndarray, meta: dict) -> None:
    from safetensors.numpy import save_file
    meta = dict(meta, format=MASK_FORMAT)
    save_file({"keep": keep.astype(np.uint8), "mtp_keep": mtp_keep.astype(np.uint8)}, path,
              metadata={k: str(v) for k, v in meta.items()})


def read_mask(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        meta = dict(f.metadata() or {})
        if meta.get("format") != MASK_FORMAT:
            raise SystemExit(f"offload_curve: {path} is not a {MASK_FORMAT} mask ({meta.get('format')})")
        return {"keep": f.get_tensor("keep"), "mtp_keep": f.get_tensor("mtp_keep"), "meta": meta}


def calibration_sets(traces, holdout_a4: bool, kfold: int, fold: str) -> list:
    """[(variant, traces, held-out A4 names)]: `all` (no option), `noa4` (--holdout-a4: without the A4 scenarios),
    `fold<i>` (--kfold K --fold i|all: without fold i's A4 scenarios, every other source kept - decision 9's rule)."""
    a4 = [t.name for t in traces if t.cls == "A4"]
    if fold is not None:
        folds = a4_folds(a4, kfold)
        which = range(kfold) if fold == "all" else [int(fold)]
        out = []
        for i in which:
            if not 0 <= i < kfold:
                raise SystemExit(f"offload_curve reap: --fold {i} outside 0..{kfold - 1}")
            held = sorted(n for n in a4 if folds[n] == i)
            out.append((f"fold{i}", [t for t in traces if t.name not in held], held))
        return out
    if holdout_a4:
        return [("noa4", [t for t in traces if t.cls != "A4"], sorted(a4))]
    return [("all", list(traces), [])]


def mask_name(criterion: str, kept: float, variant: str) -> str:
    return f"mask-{criterion}-{kept:g}" + ("" if variant == "all" else f"-{variant}") + ".safetensors"


def cmd_reap(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E, k = tr[0].layers, tr[0].experts, tr[0].top_k
    crits = a.criterion.split(",")
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "reap.json")
    report = json.load(open(jpath)) if os.path.exists(jpath) else {"masks": {}, "overlap": {}}
    if a.fold is not None:
        folds = a4_folds([t.name for t in tr if t.cls == "A4"], a.kfold)
        with open(os.path.join(a.out, "folds.json"), "w") as fh:
            json.dump({"kfold": a.kfold, "folds": folds}, fh, indent=1, sort_keys=True)
        print(f"reap: {a.kfold} folds of {len(folds)} A4 scenarios: " +
              ", ".join(f"fold {i}: {sum(1 for v in folds.values() if v == i)}" for i in range(a.kfold)))
    t0 = tr[0]
    for variant, cal, held in calibration_sets(tr, a.holdout_a4, a.kfold, a.fold):
        if not cal:
            raise SystemExit(f"offload_curve reap: the {variant} calibration set is empty")
        scores = {c: reap_scores(cal, L, E, c) for c in crits}
        for kept in frange(a.kept):
            keeps = {}
            for c in crits:
                keep = mask_keep(scores[c], kept, k)
                keeps[c] = keep
                meta = {"model": t0.meta.get("model", ""), "checkpoint": t0.meta.get("checkpoint", ""),
                        "checkpoint_revision": t0.meta.get("checkpoint_revision", ""), "criterion": c,
                        "kept_fraction": f"{kept:g}", "per_layer": "uniform", "variant": variant,
                        "calibration": json.dumps([{"name": t.name, "sha256": t.sha256} for t in cal]),
                        "held_out": json.dumps(held), "sources": ",".join(a.sources or ["all"]),
                        "tokens": str(sum(t.T for t in cal))}
                if variant.startswith("fold"):
                    meta["fold"] = f"{variant[4:]}/{a.kfold}"
                path = os.path.join(a.out, mask_name(c, kept, variant))
                write_mask(path, keep, np.ones(E, bool), meta)
                report["masks"][os.path.basename(path)] = {"criterion": c, "kept": kept, "variant": variant,
                                                           "calibration": [t.name for t in cal], "held_out": held}
                print(f"reap: {os.path.basename(path)}: {int(keep.sum(1)[0])} of {E} experts a layer kept, calibrated on "
                      f"{len(cal)} traces ({sum(t.T for t in cal)} rows){'; held out ' + str(len(held)) if held else ''}")
            if "reap" in keeps and "freq" in keeps:
                ov = (keeps["reap"] & keeps["freq"]).sum(1).tolist()
                report["overlap"].setdefault(f"{kept:g}", {})[variant] = ov
                print(f"reap: kept {kept:g} ({variant}): REAP and freq masks share {min(ov)}..{max(ov)} of "
                      f"{int(keeps['reap'].sum(1)[0])} experts a layer (mean {np.mean(ov):.1f})")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)


# --- prune-report -----------------------------------------------------------------------------------------------------------
def _score_mod():
    import importlib.util
    path = os.path.join(_HERE, "..", "toolcall", "score.py")
    spec = importlib.util.spec_from_file_location("oc_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def a4_accuracy(a4_dir: str, run_of, ref: str = "bf16"):
    """score.py's rule (the first call's name and arguments equal the reference's) over the scenarios with a
    <name>.<ref>.txt: run_of(name) -> the candidate run's tag. (matches, scored) or None when no candidate exists."""
    sc = _score_mod()
    names = sorted(n[: -len(f".{ref}.txt")] for n in os.listdir(a4_dir) if n.endswith(f".{ref}.txt"))
    hits = scored = 0
    for n in names:
        p = os.path.join(a4_dir, f"{n}.{run_of(n)}.txt")
        if not os.path.exists(p):
            continue
        r = sc.parse_first_call(open(os.path.join(a4_dir, f"{n}.{ref}.txt"), encoding="utf-8").read())
        g = sc.parse_first_call(open(p, encoding="utf-8").read())
        scored += 1
        hits += int(r[0] == "call" and g[0] == "call" and g[1:3] == r[1:3])
    return (hits, scored) if scored else None


def pick_budget(lines, devices=2, max_len=32768, dense="int4"):
    c = [b for b in lines if b["devices"] == devices and b["max_len"] == max_len and b.get("dense") == dense and b["mtp"]]
    c.sort(key=lambda b: not b.get("prefill", False))   # the serving configuration: the head and the prefill scratch
    return c[0] if c else None


def cmd_prune_report(a) -> None:
    ev = []
    for f in sorted(os.listdir(a.dir)):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(a.dir, f)))
            if "kl_mean" in j and j.get("criterion", a.criterion) == a.criterion:
                ev.append(j)
    budget = pick_budget(read_budget(a.budget)) if a.budget else None
    layers = int(budget["layers"]) if budget else 48
    E = int(budget["experts"]) if budget else 512
    eb = int(budget["expert_bytes"]) if budget else EXPERT_BYTES
    a4_names = None
    if a.a4:
        a4_names = sorted(n[: -len(".bf16.txt")] for n in os.listdir(a.a4) if n.endswith(".bf16.txt"))
        folds = a4_folds(a4_names, a.kfold)
    kepts = sorted({float(j["kept"]) for j in ev} | set(frange(a.kept) if a.kept else []))
    rows = []
    for kept in kepts:
        base = f"mask-{a.criterion}-{kept:g}"
        gi = [j for j in ev if j["set"] == "golden" and j["tag"] == base]
        ai = [j for j in ev if j["set"] == "a4" and j["tag"] == base]
        ah = [j for j in ev if j["set"] == "a4" and j["tag"].startswith(base + "-fold")]
        mean = (lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else None)
        row = {"kept": kept, "kl_mean_golden": mean(gi, "kl_mean"), "kl_p99_golden": mean(gi, "kl_p99"),
               "argmax_golden": mean(gi, "argmax"), "kl_mean_a4_in": mean(ai, "kl_mean"),
               "argmax_a4_in": mean(ai, "argmax"), "kl_mean_a4_held": mean(ah, "kl_mean"),
               "argmax_a4_held": mean(ah, "argmax"), "golden_prompts": len(gi)}
        if a4_names is not None:
            ins = a4_accuracy(a.a4, lambda n: base)
            held = a4_accuracy(a.a4, lambda n: f"{base}-fold{folds[n]}")
            row["a4_in_sample"] = ins[0] / ins[1] if ins else None
            row["a4_held_out"] = held[0] / held[1] if held else None
            row["a4_scored"] = [ins[1] if ins else 0, held[1] if held else 0]
            row["a4_gap"] = (row["a4_in_sample"] - row["a4_held_out"]) if ins and held else None
        n_kept = int(round(E * kept))
        row["experts_gb"] = round(layers * n_kept * eb / 1e9, 3)
        row["fits_two_cards"] = (layers * n_kept * eb <= int(budget["cache_total"])) if budget else None
        rows.append(row)
    fmt = (lambda v, p=4: "-" if v is None else (("yes" if v else "no") if isinstance(v, bool) else f"{v:.{p}f}"))
    print(f"prune-report ({a.criterion}): {len(ev)} maskeval files" + (f", A4 outputs in {a.a4}" if a.a4 else "") +
          (f"; two-card cache at 32k (head + prefill): {int(budget['cache_total']) / 1e9:.2f} GB" if budget else ""))
    print("| kept | KL mean golden | KL p99 golden | argmax golden | argmax A4 (in / held) | A4 in sample | A4 held out "
          "| gap | experts GB | fits two cards |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['kept']:g} | {fmt(r['kl_mean_golden'], 5)} | {fmt(r['kl_p99_golden'], 5)} | {fmt(r['argmax_golden'])} "
              f"| {fmt(r['argmax_a4_in'])} / {fmt(r['argmax_a4_held'])} | {fmt(r.get('a4_in_sample'))} | "
              f"{fmt(r.get('a4_held_out'))} | {fmt(r.get('a4_gap'))} | {r['experts_gb']:.3f} | {fmt(r['fits_two_cards'])} |")
    pick = None
    if a.a4_bar is not None and a4_names is not None:
        ok = [r for r in rows if r.get("a4_held_out") is not None and r["a4_held_out"] >= a.a4_bar]
        pick = min((r["kept"] for r in ok), default=None)
        print(f"decision 9 (held-out A4 >= {a.a4_bar:g}): kept {pick if pick is not None else 'none qualifies'} - the "
              f"shipped mask is calibrated on every coding trace at that fraction")
    _dump(a, {"criterion": a.criterion, "rows": rows, "pick": pick, "budget": budget})


# --- P0.7: the projection and the stopping rule ------------------------------------------------------------------------------
STOP_TPS = 40.0                   # spec 22 §3: two cards, 32k, our int4 g64 dense layers, the best fill
ROOFLINE_GBPS = 590.0             # the B70's roofline (spec 22 §2)
ROOFLINE_SHARE = 0.776            # the share Qwen3.8 reaches (spec 21 §3, docs/05)
KV_ROW_BYTES = 24576              # spec 21 §3: a selected position's KV over the 12 QSA layers
INDEXER_BYTES = 768               # spec 21 §3: the indexer scan a position
QSA_SELECTED = 2051               # spec 21 §2.3: at most 2051 positions attended


def decode_ms(D: float, R: float, BW: float, P: float, h: float, law: str = "serial", alpha: float = 1.0) -> float:
    """Spec 22 §2 (bytes, GB/s -> ms a token): serial (D + hR)/BW + (1-h)R/P; overlapped D/BW + max(hR/BW, (1-h)R/P);
    `mix` = overlapped + alpha (serial - overlapped) - P0.3's measured law as an interpolation."""
    ser = ((D + h * R) / BW + (1 - h) * R / P) / 1e6
    ovl = (D / BW + max(h * R / BW, (1 - h) * R / P)) / 1e6
    if law == "serial":
        return ser
    if law == "overlapped":
        return ovl
    return ovl + alpha * (ser - ovl)


def attn_bytes(ctx: int) -> float:
    return min(ctx, QSA_SELECTED) * KV_ROW_BYTES + INDEXER_BYTES * ctx


def read_p0(path: str) -> dict:
    """probe_offload's `po` lines: per card the mixed-launch law and P (the largest P's law_summary of the single-card
    runs, dev=0 / dev=1), the copy rates (an expert's copy on the copy queue; the two cards' 256-expert aggregate)."""
    out = {"law": {}, "P_gbps": {}, "alpha": {}, "copy_expert_us": {}, "copy_both_gbps": None, "lines": 0}
    best = {}
    for ln in open(path):
        if not ln.startswith("po "):
            continue
        out["lines"] += 1
        f = dict(t.split("=", 1) for t in ln.split() if "=" in t)
        mode = ln.split()[1]
        if mode == "zc" and "law_summary" in ln and f.get("dev") in ("0", "1"):
            card, P = int(f["card"]), int(f["P"])
            if P >= best.get(card, -1):
                best[card] = P
                out["law"][card], out["alpha"][card], out["P_gbps"][card] = f["law"], float(f["alpha"]), float(f["P_gbps"])
        if mode == "copy" and f.get("shape") == "expert" and f.get("list") == "regular" and f.get("dev") in ("0", "1"):
            if f.get("ordinal") == "copy" or int(f["card"]) not in out["copy_expert_us"]:
                out["copy_expert_us"][int(f["card"])] = float(f["us"])
        if mode == "copy" and f.get("card") == "all" and f.get("shape") == "256_experts_one" and f.get("list") == "regular":
            v = float(f["aggregate_gbps"])
            out["copy_both_gbps"] = max(v, out["copy_both_gbps"] or 0.0)
    return out


def best_h(curve_rows, f: float, key: str, swap_gain: float = 0.0):
    """The best held-out fill at f: max(uniform, per layer) (+ the swap simulation's best gain when it is positive)."""
    hu, hp = interp(curve_rows, f, f"uniform_{key}"), interp(curve_rows, f, f"perlayer_{key}")
    pol, h = ("per-layer", hp) if hp >= hu else ("uniform", hu)
    if swap_gain > 0:
        return min(1.0, h + swap_gain), pol + " + swaps"
    return h, pol


def cmd_project(a) -> None:
    budget = read_budget(a.budget)
    p0 = read_p0(a.p0) if a.p0 else {"law": {}, "P_gbps": {}, "alpha": {}, "copy_expert_us": {}, "copy_both_gbps": None,
                                      "lines": 0}
    cj = json.load(open(a.curve))
    key = "held" if cj.get("split") == "hash" else "in"
    rows = cj["rows"]
    swap_gain = 0.0
    if a.swap:
        sw = json.load(open(a.swap))
        swap_gain = max((r["gain"] for r in sw.get("rows", [])), default=0.0)
    mtpj = json.load(open(a.mtp)) if a.mtp else None
    prej = json.load(open(a.prefill)) if a.prefill else None
    acc = None
    if a.accept:
        aj = json.load(open(a.accept))
        acc = aj["pooled"].get("reuse") or next(iter(aj["pooled"].values()))
    BW = a.bw if a.bw else ROOFLINE_GBPS * ROOFLINE_SHARE
    bw_src = "row 32's measured rate" if a.bw else f"the {ROOFLINE_GBPS:g} GB/s roofline at {ROOFLINE_SHARE:.1%} (spec 21 §3)"
    if p0["law"]:
        alphas = [p0["alpha"][c] for c in sorted(p0["alpha"])]
        alpha = float(np.mean(alphas))
        law = "serial" if alpha > 0.75 else "overlapped" if alpha < 0.25 else "mix"
        law_src = f"P0.3 measured: alpha {alpha:.3f} ({', '.join(p0['law'][c] for c in sorted(p0['law']))})"
    else:
        alpha, law, law_src = 1.0, "serial", "NOT MEASURED - serial assumed (the conservative law)"
    P_of = dict(p0["P_gbps"])
    if a.P:
        for c in (0, 1):
            P_of.setdefault(c, a.P)
    p_src = "P0.3 arm C (per card)" if p0["P_gbps"] else (f"--P {a.P} GB/s (ASSUMED)" if a.P else None)
    if not P_of:
        raise SystemExit("offload_curve project: no P (probe_offload zc's law_summary lines, or --P)")
    out = {"BW": BW, "bw_source": bw_src, "law": law, "alpha": alpha, "law_source": law_src, "P": P_of, "p_source": p_src,
           "swap_gain": swap_gain, "rows": [], "mtp": [], "prefill": []}
    print(f"project: BW {BW:.1f} GB/s ({bw_src}); law {law} ({law_src}); P {', '.join(f'card {c} {v:.2f}' for c, v in sorted(P_of.items()))} "
          f"GB/s ({p_src}); h = the best held-out fill ({key}){f' + swaps {swap_gain:+.4f}' if swap_gain > 0 else ''}")
    print("| form | cards | max_len | f per card | h per card | policy | ms a token | t/s |")
    print("|---|---:|---:|---|---|---|---:|---:|")

    def project_line(b, ctx):
        devs = int(b["devices"])
        D = float(b["read_non_routed"]) + attn_bytes(ctx)
        t = 0.0
        fs, hs, pol = [], [], ""
        for i in range(devs):
            Ri = (int(b["end"][i]) - int(b["first"][i])) * int(b["top_k"]) * int(b["expert_bytes"])
            routed_i = (int(b["end"][i]) - int(b["first"][i])) * int(b["experts"]) * int(b["expert_bytes"])
            fi = min(1.0, int(b["cache"][i]) / routed_i) if routed_i else 1.0
            hi, pol = best_h(rows, fi, key, swap_gain)
            Pi = P_of.get(i, P_of.get(0))
            Di = D * (Ri / float(b["read_routed"]))   # the non-routed bytes split as the layers are
            t += decode_ms(Di, Ri, BW, Pi, hi, law, alpha)
            fs.append(fi)
            hs.append(hi)
        return t, fs, hs, pol
    stop_row = None
    for b in budget:
        if not (b["mtp"] and b["prefill"]) and b["devices"] == 2:
            continue                                  # the serving configuration: the head and the prefill scratch
        if b["devices"] == 1 and not (b["mtp"] and b["prefill"]):
            continue
        if b["max_len"] not in (4096, 32768, 131072):
            continue
        t, fs, hs, pol = project_line(b, b["max_len"])
        r = {"dense": b["dense"], "devices": b["devices"], "max_len": b["max_len"], "f": fs, "h": hs, "policy": pol,
             "ms": t, "tps": 1e3 / t}
        out["rows"].append(r)
        form = "ours (int4 dense)" if b["dense"] == "int4" else "Intel's (bf16 dense, informational)"
        print(f"| {form} | {b['devices']} | {b['max_len']} | {' / '.join(f'{x:.3f}' for x in fs)} | "
              f"{' / '.join(f'{x:.3f}' for x in hs)} | {pol} | {t:.2f} | {1e3 / t:.1f} |")
        if b["devices"] == 2 and b["max_len"] == 32768 and b["dense"] == "int4":
            stop_row = r
        # MTP: a verify at M = K + 1 reads the non-routed bytes once, M rows' attention and the union of the M rows'
        # experts; K draft steps read the resident head (decision 6's proposal).
        if b["devices"] == 2 and b["dense"] == "int4" and mtpj:
            for K in (1, 2, 3):
                u = next((x for x in mtpj["K"] if x["K"] == K), None)
                M = K + 1
                distinct = u["distinct_per_layer"] if u else M * int(b["top_k"])
                Rm = distinct * int(b["layers"]) * int(b["expert_bytes"])
                h = float(np.mean(hs))
                Pm = float(np.mean(list(P_of.values())))
                tv = decode_ms(float(b["read_non_routed"]) + M * attn_bytes(b["max_len"]), Rm, BW, Pm, h, law, alpha)
                td = K * (float(b["read_mtp_step"]) / BW) / 1e6
                if acc:
                    Es = [(acc["expected_tokens_at_K"][K - 1], "r34.accept")]
                else:
                    Es = [(sum(x ** j for j in range(K + 1)), f"a {x:g} (PROPOSED range)") for x in (0.5, 0.65, 0.8)]
                for E_K, src in Es:
                    m = {"max_len": b["max_len"], "K": K, "M": M, "distinct_per_layer": distinct, "E_K": E_K,
                         "accept_source": src, "tps": E_K / (tv + td) * 1e3}
                    out["mtp"].append(m)
    if out["mtp"]:
        print("| MTP max_len | K | distinct a layer (M rows) | E_K | t/s |")
        print("|---:|---:|---:|---:|---:|")
        for m in out["mtp"]:
            print(f"| {m['max_len']} | {m['K']} | {m['distinct_per_layer']:.2f} | {m['E_K']:.3f} ({m['accept_source']}) | "
                  f"{m['tps']:.1f} |")
    if prej:
        comp = dict((int(x.split(":")[0]), float(x.split(":")[1])) for x in a.prefill_compute_ms.split(",")) \
            if a.prefill_compute_ms else {}
        bus = p0["copy_both_gbps"] or a.copy_gbps
        b32 = pick_budget(budget)
        if b32 and bus:
            f2 = b32["f"]
            print(f"| prefill C | f | streamed GB a chunk | bus ms (both cards, {bus:.2f} GB/s) | compute ms | t/s |")
            print("|---:|---:|---:|---:|---:|---:|")
            for C in (2048, 4096):
                pts = sorted((r["f"], r["streamed_gb_per_chunk"]) for r in prej["rows"] if r["C"] == C)
                if not pts:
                    continue
                gb = interp([{"f": x, "v": y} for x, y in pts], f2, "v")
                bus_ms = gb / bus * 1e3
                cm = comp.get(C)
                tps = C / (max(bus_ms, cm) / 1e3) if cm else C / (bus_ms / 1e3)
                out["prefill"].append({"C": C, "f": f2, "streamed_gb": gb, "bus_ms": bus_ms, "compute_ms": cm, "tps": tps,
                                       "derived": C == 4096})
                print(f"| {C}{' (derived)' if C == 4096 else ''} | {f2:.3f} | {gb:.2f} | {bus_ms:.0f} | "
                      f"{'%.0f' % cm if cm else 'not measured (bus cap only)'} | {tps:.0f} |")
    if stop_row is None:
        raise SystemExit("offload_curve project: no two-card 32768 int4 budget line (with the head and the prefill)")
    verdict = "STOP" if stop_row["tps"] < STOP_TPS else "GO-CANDIDATE"
    out["verdict"] = verdict
    out["stop_inputs"] = {"devices": 2, "max_len": 32768, "dense": "int4", "tps": stop_row["tps"], "law": law,
                          "policy": stop_row["policy"], "bar_tps": STOP_TPS}
    _dump(a, out)
    print(f"{verdict}: two-card decode at 32k projects {stop_row['tps']:.1f} t/s (ours, int4 g64 dense, {stop_row['policy']} "
          f"fill, {law} law, P measured per card) against the {STOP_TPS:g} t/s rule (spec 22 §3); the operator's go is "
          f"separate")


def project_args(sub) -> None:
    p = sub.add_parser("project")
    p.add_argument("--budget", required=True, help="Task 1's budget-json lines (qwen4exp_plan_test --budget-json)")
    p.add_argument("--p0", help="probe_offload's po lines (copy, zc): the law and P per card")
    p.add_argument("--curve", required=True, help="offload_curve.py curve --json")
    p.add_argument("--swap", help="offload_curve.py swap --json (its best gain, when positive)")
    p.add_argument("--mtp", help="offload_curve.py mtp --json (the verify unions)")
    p.add_argument("--prefill", help="offload_curve.py prefill --json (the streamed bytes a chunk)")
    p.add_argument("--accept", help="r34.accept's accept_<norm>.json (else the 0.5-0.8 range, PROPOSED)")
    p.add_argument("--bw", type=float, help="row 32's measured per-layer rate (GB/s; else the roofline at 77.6%%)")
    p.add_argument("--P", type=float, help="a P to assume when the p0 file has none (GB/s, marked ASSUMED)")
    p.add_argument("--copy-gbps", type=float, help="the prefill bus rate when the p0 file has none")
    p.add_argument("--prefill-compute-ms", help="row 33's measured compute a chunk: 2048:X[,4096:Y]")
    p.add_argument("--json")


# --- commands -------------------------------------------------------------------------------------------------------------
def _dump(a, obj) -> None:
    if getattr(a, "json", None):
        with open(a.json, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, default=lambda o: None if o != o else o)


def _gini(v: np.ndarray) -> float:
    v = np.sort(v.astype(np.float64))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def cmd_counts(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E, k = tr[0].layers, tr[0].experts, tr[0].top_k
    pooled = counts_of(tr, L, E)
    tops = [n for n in (E // 8, E // 4, E // 2) if n > 0]
    out = {"layers": L, "experts": E, "top_k": k, "tokens": sum(t.T for t in tr), "sources": {}, "per_layer": []}
    print(f"counts: {len(tr)} sources, {out['tokens']} rows, L {L}, E {E}, k {k}; routed slots {int(pooled.sum())}")
    print("| source | class | rows | distinct experts a layer (min / median / max) |")
    print("|---|---|---:|---|")
    for t in tr:
        c = counts_of([t], L, E)
        dist = (c > 0).sum(1)
        out["sources"][t.name] = {"class": t.cls, "T": t.T, "distinct": dist.tolist()}
        print(f"| {t.name} | {t.cls} | {t.T} | {int(dist.min())} / {int(np.median(dist))} / {int(dist.max())} |")
    print("| layer | distinct (pooled) | Gini | " + " | ".join(f"top-{n} share" for n in tops) + " |")
    print("|---:|---:|---:|" + "---:|" * len(tops))
    for l in range(L):
        c = pooled[l]
        srt = np.sort(c)[::-1]
        shares = [float(srt[:n].sum() / max(1, c.sum())) for n in tops]
        out["per_layer"].append({"layer": l, "distinct": int((c > 0).sum()), "gini": _gini(c),
                                 "top_share": dict(zip(map(str, tops), shares))})
        print(f"| {l} | {int((c > 0).sum())} | {_gini(c):.3f} | " + " | ".join(f"{s:.3f}" for s in shares) + " |")
    _dump(a, out)


def cmd_curve(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E = tr[0].layers, tr[0].experts
    r = curve(tr, L, E, frange(a.f), a.split)
    key = "held" if a.split == "hash" else "in"
    print(f"curve: split {a.split}; {len(tr)} sources (half 1: "
          f"{sorted(n for n, h in r['halves'].items() if h == 1)})")
    print(f"| f | uniform in | uniform held | per-layer in | per-layer held | " +
          " | ".join(f"per-layer {key} {c}" for c in CLASSES) + " |")
    print("|---:|" + "---:|" * (4 + len(CLASSES)))
    for row in r["rows"]:
        cls = [row.get(f"perlayer_{key}_{c}") for c in CLASSES]
        print(f"| {row['f']:.3f} | {row['uniform_in']:.4f} | {row['uniform_held']:.4f} | {row['perlayer_in']:.4f} | "
              f"{row['perlayer_held']:.4f} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in cls) + " |")
    for k_, v in r["reach"].items():
        print(f"reach {k_}: f = {'not reached' if v is None else f'{v:.3f}'}")
    if a.budget:
        r["budget"] = []
        for b in read_budget(a.budget):
            hu, hp = interp(r["rows"], b["f"], f"uniform_{key}"), interp(r["rows"], b["f"], f"perlayer_{key}")
            r["budget"].append({**{k_: b[k_] for k_ in ("max_len", "devices", "mtp", "prefill", "dense")}, "f": b["f"],
                                "h_uniform": hu, "h_perlayer": hp})
            print(f"budget max_len {b['max_len']} devices {b['devices']} dense {b['dense']} mtp {b['mtp']} prefill "
                  f"{b['prefill']}: f {b['f']:.4f} -> h uniform {hu:.4f}, per layer {hp:.4f} ({key})")
    _dump(a, r)


def cmd_swap(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E = tr[0].layers, tr[0].experts
    half = halves([t.name for t in tr])
    cnt = [counts_of([t for t in tr if half[t.name] == g], L, E) for g in (0, 1)]
    one_half = not any(half[t.name] == 1 for t in tr) or not any(half[t.name] == 0 for t in tr)
    copy_ms = math.inf if a.copy_ms in ("inf", "none") else float(a.copy_ms)
    rows = []
    static_h = static_t = 0
    res_of = {}
    for t in tr:   # the static fill each source sees: the other half's profile (held out), or all when one half
        prof = counts_of(tr, L, E) if one_half else cnt[1 - half[t.name]]
        res_of[t.name] = resident_uniform(prof, a.f) if a.fill == "uniform" else resident_perlayer(prof, a.f)
        h, n = hit_rate([t], res_of[t.name])
        static_h, static_t = static_h + h, static_t + n
    static = _ratio(static_h, static_t)
    print(f"swap: f {a.f} ({a.fill}, {'in sample' if one_half else 'held out'}), copy {copy_ms} ms an expert, step "
          f"{a.step_ms} ms; static h {static:.4f}")
    print("| S | B MB | W | h swap | gain |")
    print("|---:|---:|---:|---:|---:|")
    for S in ilist(a.S):
        for Bmb in frange(a.B):
            for W in ilist(a.W):
                hh = tt = 0
                for t in tr:
                    h, n = swap_sim(t, res_of[t.name], S, Bmb * 1e6, W, copy_ms, a.step_ms)
                    hh, tt = hh + h, tt + n
                hs = _ratio(hh, tt)
                rows.append({"S": S, "B_mb": Bmb, "W": W, "h": hs, "gain": hs - static})
                print(f"| {S} | {Bmb:g} | {W} | {hs:.4f} | {hs - static:+.4f} |")
    _dump(a, {"f": a.f, "fill": a.fill, "copy_ms": None if math.isinf(copy_ms) else copy_ms, "step_ms": a.step_ms,
              "static": static, "rows": rows})


def cmd_mtp(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E = tr[0].layers, tr[0].experts
    res = resident_uniform(counts_of(tr, L, E), a.f) if a.fill == "uniform" else resident_perlayer(counts_of(tr, L, E), a.f)
    out = {"f": a.f, "fill": a.fill, "K": []}
    print(f"mtp: f {a.f} ({a.fill}, in sample); per layer the union over M = K + 1 consecutive rows")
    print("| K | M | distinct a layer | iid prior | misses a verify step | M single rows' misses |")
    print("|---:|---:|---:|---:|---:|---:|")
    for K in ilist(a.K):
        acc = {"distinct_per_layer": 0.0, "iid_per_layer": 0.0, "misses_per_step": 0.0, "misses_M_rows_unshared": 0.0}
        w = 0
        for t in tr:
            u = mtp_unions(t, K, res)
            if not u.get("windows"):
                continue
            for k_ in acc:
                acc[k_] += u[k_] * u["windows"]
            w += u["windows"]
        row = {"K": K, "M": K + 1, "windows": w, **{k_: (v / w if w else float("nan")) for k_, v in acc.items()}}
        out["K"].append(row)
        print(f"| {K} | {K + 1} | {row['distinct_per_layer']:.2f} | {row['iid_per_layer']:.2f} | "
              f"{row['misses_per_step']:.2f} | {row['misses_M_rows_unshared']:.2f} |")
    head = mtp_head(tr, mtp_counts_of(tr, E), a.f)
    out["head"] = head
    print(f"mtp head: {head['distinct']} of {head['of']} experts routed over {head['routes']} slots; h if cached at f "
          f"{a.f}: {head['h_cached']:.4f} (decision 6: resident costs one layer's 1.34 GB at int4)")
    _dump(a, out)


def cmd_prefill(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E = tr[0].layers, tr[0].experts
    cnt = counts_of(tr, L, E)
    rows = []
    print("prefill: chunks of C rows; streamed = the touched experts not resident (uniform fill, in sample)")
    print("| C | f | chunks | unique a layer | streamed GB a chunk |")
    print("|---:|---:|---:|---:|---:|")
    for C in ilist(a.chunk):
        for f in frange(a.f):
            res = resident_uniform(cnt, f)
            n = uq = st = 0.0
            short = 0
            for t in tr:
                r = prefill_bytes(t, C, res)
                n += r["chunks"]
                uq += r["unique_per_layer"] * r["chunks"]
                st += r["streamed_bytes_per_chunk"] * r["chunks"]
                short += int(r["short"])
            row = {"C": C, "f": f, "chunks": int(n), "short_sources": short, "unique_per_layer": uq / n,
                   "streamed_gb_per_chunk": st / n / 1e9}
            rows.append(row)
            print(f"| {C} | {f:.2f} | {int(n)}{f' ({short} short)' if short else ''} | {row['unique_per_layer']:.1f} | "
                  f"{row['streamed_gb_per_chunk']:.3f} |")
    _dump(a, {"rows": rows})


def cmd_profile(a) -> None:
    tr = load_traces(a.traces, a.sources)
    L, E = tr[0].layers, tr[0].experts
    meta = write_profile(a.out, tr, L, E)
    print(f"profile: {a.out} ({meta['format']}): counts u32 [{L}][{E}], mtp_counts u32 [{E}], {meta['tokens']} rows "
          f"from {len(tr)} traces ({', '.join(json.loads(meta['sources']))})")
    _dump(a, meta)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, json_out=True):
        p.add_argument("--traces", required=True)
        p.add_argument("--sources", type=lambda s: s.split(","), help="classes or names to use (default: all)")
        if json_out:
            p.add_argument("--json", help="write the table as JSON here")
    common(sub.add_parser("counts"))
    p = sub.add_parser("curve")
    common(p)
    p.add_argument("--split", default="hash", choices=["hash", "none"])
    p.add_argument("--f", default="0.30:1.00:0.01")
    p.add_argument("--budget", help="Task 1's budget-json lines: h at each budget's f")
    p = sub.add_parser("swap")
    common(p)
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--fill", default="uniform", choices=["uniform", "perlayer"])
    p.add_argument("--S", default="1,4,16,64")
    p.add_argument("--B", required=True, help="MB a round, a comma list or a:b:s")
    p.add_argument("--W", default="64,256,1024")
    p.add_argument("--copy-ms", required=True, help="P0.2's measured copy time of one expert (ms), or inf")
    p.add_argument("--step-ms", type=float, required=True, help="decode's step time (P0.7's model)")
    p = sub.add_parser("mtp")
    common(p)
    p.add_argument("--K", default="1,2,3,4")
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--fill", default="uniform", choices=["uniform", "perlayer"])
    p = sub.add_parser("prefill")
    common(p)
    p.add_argument("--chunk", default="2048,4096")
    p.add_argument("--f", default="0.6:0.9:0.05")
    p = sub.add_parser("profile")
    common(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("reap")
    common(p, json_out=False)
    p.add_argument("--kept", default="0.5,0.625,0.75,0.875")
    p.add_argument("--criterion", default="reap,freq", help="reap, freq or both (the overlap is written)")
    p.add_argument("--out", required=True, help="the masks' directory (mask-<criterion>-<kept>[-noa4|-fold<i>])")
    p.add_argument("--holdout-a4", action="store_true", help="calibrate without the A4 scenarios")
    p.add_argument("--kfold", type=int, default=4, help="decision 9: the A4 scenarios in K fixed folds")
    p.add_argument("--fold", help="i | all: calibrate without fold i's A4 scenarios (every other source kept)")
    p = sub.add_parser("prune-report")
    p.add_argument("--dir", required=True, help="maskeval JSONs (qwen4exp_ref.py maskeval --json)")
    p.add_argument("--a4", help="the A4 outputs: <name>.bf16.txt and <name>.mask-<tag>.txt (score.py's rule)")
    p.add_argument("--budget", help="Task 1's budget-json lines: the two-card cache at 32k")
    p.add_argument("--criterion", default="reap")
    p.add_argument("--kept", help="kept fractions to list even without maskeval files")
    p.add_argument("--kfold", type=int, default=4)
    p.add_argument("--a4-bar", type=float, help="decision 9's bar on the held-out A4 accuracy (prints the pick)")
    p.add_argument("--json")
    a = ap.parse_args(argv)
    {"counts": cmd_counts, "curve": cmd_curve, "swap": cmd_swap, "mtp": cmd_mtp, "prefill": cmd_prefill,
     "profile": cmd_profile, "reap": cmd_reap, "prune-report": cmd_prune_report}[a.cmd](a)


if __name__ == "__main__":
    main()
