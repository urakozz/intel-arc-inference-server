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
Task 4 adds `reap`, `prune-report` and `project` (P0.8, P0.7) below.
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
    a = ap.parse_args(argv)
    {"counts": cmd_counts, "curve": cmd_curve, "swap": cmd_swap, "mtp": cmd_mtp, "prefill": cmd_prefill,
     "profile": cmd_profile}[a.cmd](a)


if __name__ == "__main__":
    main()
