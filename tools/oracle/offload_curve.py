#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Spec 22 P0.6 (plan 22a Task 3): the expert-offload tier's hit-rate curves from spec 21a's routing traces, and
the routing profile plan 22b's loader reads (decision 8).

    offload_curve.py counts  --traces DIR [--json FILE]
    offload_curve.py curve   --traces DIR [--split hash|none] [--f 0.30:1.00:0.01] [--budget FILE] [--json FILE]
    offload_curve.py swap    --traces DIR --f F --S 1,4,16,64 --B <MB list> --copy-ms X --step-ms Y [--W 64,256,1024]
    offload_curve.py mtp     --traces DIR --K 1,2,3,4 --f F [--json FILE]
    offload_curve.py prefill --traces DIR --chunk 2048,4096 --f 0.6:0.9:0.05 [--json FILE]
    offload_curve.py profile --traces DIR --out FILE [--sources CLASS|NAME,...]
    offload_curve.py project --shape laguna --curve FILE [--mtp FILE] [--json FILE]

Every command takes --shape flash-next|laguna (default flash-next): the model the traces are of (SHAPES: experts,
top-k, the decoder layers the traces' L axis holds, an expert's record bytes; Laguna's bytes a token too). A shape
that names its layers is checked against the traces.

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

**Prompt and answer.** A trace whose metadata carries `answer_from` (laguna_ref.py trace) has prompt rows [0, a) and
answer rows [a, T); `curve` then also scores the two apart (decode speed depends on the answer rows). Profiles are
built from all rows.

Every table is printed and, with --json FILE, written as JSON for the record.

`project` (P0.7) is below; with --shape laguna it is spec 30 §7.1's two-constant decode model, §8's drafter formula
and §13's stop rule instead.
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
# A model's shape. layers: the decoder layer of each index of a trace's L axis (None: 0 .. L-1, not checked). Laguna
# (spec 30 §1-2, derived from the config and the tensor shapes): 256 experts top-10 in layers 1..47 (layer 0 dense), an
# expert 3,342,336 + 1,671,168 B at int4 g64; D / R the bytes a decode token reads outside / inside the routed experts,
# tp the same per card in tensor parallel; 48 decoder layers; 48 KiB of bf16 KV a position (12 full layers).
SHAPES = {
    "flash-next": {"experts": 512, "top_k": 10, "layers": None, "expert_bytes": EXPERT_BYTES},
    "laguna": {"experts": 256, "top_k": 10, "layers": list(range(1, 48)), "expert_bytes": 5013504,
               "D": 2.181e9, "R": 2.356e9, "tp": (1.137e9, 1.178e9), "decoder_layers": 48, "kv_row_bytes": 49152},
}
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
        self.answer_from = int(self.meta.get("answer_from", self.T))
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()


def load_traces(d: str, names=None, shape: str = "flash-next") -> list:
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
    sh = SHAPES[shape]
    if sh["layers"] is not None and (L, E, k) != (len(sh["layers"]), sh["experts"], sh["top_k"]):
        raise SystemExit(f"offload_curve: the traces are L {L} E {E} k {k}, the {shape} shape L {len(sh['layers'])} "
                         f"E {sh['experts']} k {sh['top_k']}")
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


def rows_of(t, part: str) -> np.ndarray:
    """A trace's ids [T][L][k] restricted to its prompt rows [0, answer_from), its answer rows, or all."""
    return t.ids[: t.answer_from] if part == "prompt" else t.ids[t.answer_from:] if part == "answer" else t.ids


def hit_rate(traces, res: np.ndarray, part: str = "all"):
    """(hits, total routed slots) of `traces` (their `part` rows) against resident mask res [L][E]."""
    hits = total = 0
    for t in traces:
        L = t.layers
        h = res[np.arange(L)[None, :, None], rows_of(t, part)]
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
    parts = ("prompt", "answer") if any(t.answer_from < t.T for t in traces) else ()
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
                for part in parts:   # held out per part: each source scored on the other half's profile
                    a0, b0 = hit_rate(groups[1], r0, part)
                    a1, b1 = hit_rate(groups[0], r1, part)
                    row[f"{pol}_held_{part}"] = _ratio(a0 + a1, b0 + b1)
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
                for part in parts:
                    row[f"{pol}_in_{part}"] = _ratio(*hit_rate(traces, r, part))
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
    return {"split": split, "halves": half, "rows": rows, "reach": reach, "parts": list(parts),
            "sources": {t.name: {"class": t.cls, "T": t.T} for t in traces}}


class LayerSlice:
    """A trace restricted to layers [lo, hi) (a device's), for the per-device curve."""

    def __init__(self, t, lo: int, hi: int):
        self.name, self.cls, self.T, self.top_k, self.answer_from = t.name, t.cls, t.T, t.top_k, t.answer_from
        self.ids = t.ids[:, lo:hi, :]
        self.layers = hi - lo


def device_h(traces, E: int, lo: int, hi: int, f: float, split: str = "hash") -> dict:
    """h at f over layers [lo, hi) alone - the fill and the score on that device's layers (held out when split hash)."""
    sl = [LayerSlice(t, lo, hi) for t in traces]
    r = curve(sl, hi - lo, E, [f], split)["rows"][0]
    key = "held" if split == "hash" else "in"
    return {"f": f, "first": lo, "end": hi, "h_uniform": r[f"uniform_{key}"], "h_perlayer": r[f"perlayer_{key}"]}


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


def pick_budget(lines, devices=2, max_len=32768, dense="int4"):
    c = [b for b in lines if b["devices"] == devices and b["max_len"] == max_len and b.get("dense") == dense and b["mtp"]]
    c.sort(key=lambda b: not b.get("prefill", False))   # the serving configuration: the head and the prefill scratch
    return c[0] if c else None


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
             expert_bytes: int = EXPERT_BYTES, stats: dict | None = None):
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
            if moves and stats is not None:
                stats["moved"] = stats.get("moved", 0) + len(moves)
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


# --- swap policies (spec 21 v2 §6) --------------------------------------------------------------------------------------
def _land(pending: list, t: int, res: np.ndarray) -> list:
    """Admit the copies whose event has fired by step t (pending: [(ready step, layer, expert)])."""
    keep = []
    for r, l, e in pending:
        if t >= r:
            res[l, e] = True
        else:
            keep.append((r, l, e))
    return keep


def policy_basalt(trace, res0: np.ndarray, window: int, copy_ms: float, step_ms: float, decay: float = 0.85,
                  margin: float = 1.5, cap: int = 96) -> tuple:
    """basalt's adaptive placement: per (layer, expert) routing counts, decayed by `decay` at the end of every round of
    `window` steps; then over all layers the non-resident experts with the highest counts replace the residents with the
    lowest, where the incoming count is at least the victim's + `margin`, at most `cap` a round (largest gain first). A
    victim reads from the host at once; the copies land ceil(n copy_ms / step_ms) steps after the round (between steps,
    after their event); no round starts while one is in flight. Returns (hits, total, experts copied)."""
    ids = trace.ids
    T, L, k = ids.shape
    res = res0.copy()
    c = np.zeros(res.shape, np.float64)
    lar = np.arange(L)[:, None]
    pending, hits, copied = [], 0, 0
    for t in range(T):
        pending = _land(pending, t, res)
        row = ids[t]
        hits += int(res[lar, row].sum())
        np.add.at(c, (np.repeat(np.arange(L), k), row.reshape(-1)), 1.0)
        if (t + 1) % window or pending or not math.isfinite(copy_ms):
            continue
        cand = []
        for l in range(L):
            out_c = np.where(res[l], -np.inf, c[l])
            in_c = np.where(res[l], c[l], np.inf)
            for e_in, e_v in zip(_rank_desc(out_c), np.argsort(in_c, kind="stable")):
                if not np.isfinite(out_c[e_in]) or not res[l, e_v] or out_c[e_in] < in_c[e_v] + margin:
                    break
                cand.append((out_c[e_in] - in_c[e_v], l, int(e_in), int(e_v)))
        cand.sort(key=lambda x: (-x[0], x[1], x[2]))
        moves = cand[:cap]
        if moves:
            ready = t + 1 + math.ceil(len(moves) * copy_ms / step_ms)
            for _, l, e_in, e_v in moves:
                res[l, e_v] = False
                pending.append((ready, l, e_in))
            copied += len(moves)
        c *= decay
    return hits, T * L * k, copied


def policy_saber(trace, res0: np.ndarray, prof: np.ndarray, copy_ms: float, step_ms: float, inserts: int = 2,
                 chance: int = 8, through: float = 0.5) -> tuple:
    """saber's online placement: an expert is promoted on its second host miss within `chance` steps (second chance),
    at most `inserts` a layer a step; a layer whose misses are more than `through` of its routed slots that step reads
    through (no promotion); the victim is the layer's least recently used resident (the profile's order before first
    use: the least routed goes first). Copies land as in policy_basalt. Returns (hits, total, experts copied)."""
    ids = trace.ids
    T, L, k = ids.shape
    E = res0.shape[1]
    res = res0.copy()
    last_use = np.zeros((L, E), np.int64)
    for l in range(L):                        # before any use: the most routed is the most recent
        last_use[l, _rank_desc(prof[l])] = -np.arange(1, E + 1)
    last_miss = np.full((L, E), -(1 << 40), np.int64)
    incoming = np.zeros((L, E), bool)
    lar = np.arange(L)[:, None]
    pending, hits, copied = [], 0, 0
    for t in range(T):
        for r, l, e in pending:
            if t >= r:
                incoming[l, e] = False
        pending = _land(pending, t, res)
        row = ids[t]
        hit = res[lar, row]
        hits += int(hit.sum())
        last_use[lar, row] = np.where(hit, t, last_use[lar, row])
        miss = ~hit
        second = miss & (last_miss[lar, row] >= t - chance) & ~incoming[lar, row]
        second &= (miss.sum(1) <= through * k)[:, None]
        last_miss[lar, row] = np.where(miss, t, last_miss[lar, row])
        if not second.any() or not math.isfinite(copy_ms):
            continue
        moves = []
        for l in np.nonzero(second.any(1))[0]:
            for e in row[l][second[l]][:inserts]:
                v = int(np.argmin(np.where(res[l], last_use[l], np.iinfo(np.int64).max)))
                res[l, v] = False
                incoming[l, e] = True
                moves.append((int(l), int(e)))
        ready = t + 1 + math.ceil(len(moves) * copy_ms / step_ms)
        pending += [(ready, l, e) for l, e in moves]
        copied += len(moves)
    return hits, T * L * k, copied


# --- MTP verify unions --------------------------------------------------------------------------------------------------
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def expert_bits(ids: np.ndarray, E: int) -> np.ndarray:
    """ids [T][L][k] -> each row's expert set per layer as a bitset, uint8 [T][L][ceil(E / 8)]."""
    T, L, _ = ids.shape
    hot = np.zeros((T, L, E), bool)
    hot[np.arange(T)[:, None, None], np.arange(L)[None, :, None], ids] = True
    return np.packbits(hot, axis=-1, bitorder="little")


def mtp_unions(trace, K: int, res: np.ndarray) -> dict:
    """Over every window of M = K + 1 consecutive rows (teacher-forced rows stand for accepted drafts): the distinct
    experts a layer (mean over layers and windows) and u(M) = that / top_k (the verify's routed bytes in single rows'),
    the misses a verify step against res (summed over layers), and the iid prior: sum_e 1 - (1 - q_e)^M with q_e the
    layer's per-row frequency of e (mean over layers). The unions are bitsets OR-ed over the window."""
    ids = trace.ids
    T, L, k = ids.shape
    E = res.shape[1]
    M = K + 1
    nwin = T - M + 1
    if nwin <= 0:
        return {"K": K, "M": M, "windows": 0}
    bits = expert_bits(ids, E)
    u = bits[:nwin].copy()
    for j in range(1, M):
        u |= bits[j:j + nwin]
    cold = ~np.packbits(res, axis=-1, bitorder="little")
    dist = POPCOUNT[u].sum(dtype=np.int64)
    miss = float(POPCOUNT[u & cold[None]].sum(dtype=np.int64))
    q = np.zeros((L, E))
    for l in range(L):
        q[l] = np.bincount(ids[:, l, :].reshape(-1), minlength=E) / T
    iid = float(np.mean([(1 - (1 - q[l]) ** M).sum() for l in range(L)]))
    single_miss = float((~res[np.arange(L)[None, :, None], ids]).sum()) / T
    return {"K": K, "M": M, "windows": nwin, "distinct_per_layer": float(dist / (nwin * L)),
            "u": float(dist / (nwin * L)) / k,
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


# --- P0.7: the projection and the stopping rule ------------------------------------------------------------------------------
STOP_PP_TPS = 30.0                # spec 21 v2 §11: STOP when (a) pp + tier < 30 t/s and (b) tp + tier < 50 t/s
STOP_TP_TPS = 50.0
HC_BYTES = 1.29e9                 # spec 21 v2 §3: the hyper-connections a token, bf16, replicated on both cards under tp
TP_ALLREDUCE_MS = 0.34            # spec 21 v2 §5 (b): two all-reduces a layer over P2P, ~98 a token x 3.5 us
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


def tp_card_bytes(D: float, R: float) -> tuple:
    """Spec 21 v2 §5 (b): a card's bytes a token under tp with the expert split - the hyper-connections replicated, the
    rest of D and every routed expert halved."""
    return HC_BYTES + (D - HC_BYTES) / 2, R / 2


def tp_ms(Dc: float, Rc: float, BW: float, P: float, h: float, law: str = "serial", alpha: float = 1.0) -> float:
    """(b): both cards read their halves at once, each over its own link; plus the all-reduces."""
    return decode_ms(Dc, Rc, BW, P, h, law, alpha) + TP_ALLREDUCE_MS


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


SWAP_F_TOL = 0.02                 # a card's f within this of the swap simulation's f: its gain applies there


def card_options(dev: dict | None, curve_rows, f: float, key: str, swap):
    """A card's fills as (h, policy, swap moves a step): the static best (its own layers when curve --budget scored them,
    else the pooled curve, marked), and - only where the swap simulation ran at this f (uniform fill) - uniform + its best
    gain with the experts it moves a step."""
    if dev is not None:
        hu, hp = dev["h_uniform"], dev["h_perlayer"]
        src = ""
    else:
        hu, hp = interp(curve_rows, f, f"uniform_{key}"), interp(curve_rows, f, f"perlayer_{key}")
        src = " (pooled curve)"
    opts = [(hp, "per-layer" + src, 0.0) if hp >= hu else (hu, "uniform" + src, 0.0)]
    if swap and abs(f - swap["f"]) <= SWAP_F_TOL and swap["fill"] == "uniform":
        r = swap["best"]
        opts.append((min(1.0, hu + r["gain"]), f"uniform{src} + swaps (S {r['S']}, B {r['B_mb']:g} MB, W {r['W']})",
                     float(r.get("moved_per_step", 0.0))))
    return opts


# --- spec 30 (Laguna): the two-constant decode model (§7.1), the drafter (§8), the stop rule (§13) ------------------------
B_FIT_GBPS = 518.0        # §7.1: B and c fitted to Qwen3.8 (15.54 GB, 64 layers, 29.45 t/s) and Ornith (2.345 GB, 40, 142.8)
C_LAYER_MS = 0.062
B_21V2_GBPS = 457.8       # spec 21 v2's single ratio (Qwen3.8's), no per-layer term: the upper bound
P_LINK_GBPS = 11.2        # host RAM to a card, pinned, in place (measured on Flash-Next's 2.6 MB records)
ALLREDUCE_MS = 0.0035     # one P2P all-reduce (Qwen3.8 tensor parallel); two a layer
KV_MS_PER_GB = 2.19       # Qwen3.8's measured decode slope with context (31.41 t/s at 4k -> 20.13 at 128k)
DRAFT_MS = 3.0            # §8: a draft pass of the 1.1 GB int8 drafter (derived)
CONTEXT_F = {32768: 0.88, 65536: 0.85, 131072: 0.80}   # §4: the resident fraction the planner reaches, bf16 KV
STOP_H, GO_H, DRAFT_TAU, DRAFT_BAR = 0.90, 0.95, 5.0, 1.25   # §13's stop rule and the drafter's bar


def decode2_ms(D: float, R: float, h: float, L: int, B: float = B_FIT_GBPS, c: float = C_LAYER_MS,
               P: float = P_LINK_GBPS, extra_ms: float = 0.0) -> float:
    """Spec 30 §7.1, the serial law (bytes, GB/s -> ms a token): (D + h R) / B + L c + (1 - h) R / P + extra."""
    return (D + h * R) / B / 1e6 + L * c + (1 - h) * R / P / 1e6 + extra_ms


def two_constant_tps(sh: dict, h: float, tp: bool = False, upper: bool = False, ctx: int = 0) -> float:
    """t/s of one request: the layer split (D, R over one link at a time) or tensor parallel (each card half of D and R
    on its own link, two all-reduces a layer); `upper` = spec 21 v2's constants; ctx adds the full layers' KV reads at
    Qwen3.8's measured slope (half a card in tensor parallel)."""
    L = sh["decoder_layers"]
    D, R = sh["tp"] if tp else (sh["D"], sh["R"])
    extra = 2 * L * ALLREDUCE_MS if tp else 0.0
    extra += ctx * sh["kv_row_bytes"] / 1e9 * KV_MS_PER_GB * (0.5 if tp else 1.0)
    B, c = (B_21V2_GBPS, 0.0) if upper else (B_FIT_GBPS, C_LAYER_MS)
    return 1e3 / decode2_ms(D, R, h, L, B, c, extra_ms=extra)


def drafter_speedup(sh: dict, h: float, u: float, tau: float, draft_ms: float = DRAFT_MS) -> float:
    """Spec 30 §8 (layer split): t_plain / ((t_verify(u) + t_draft) / tau), the verify reading D once and u R of
    experts (u = distinct experts a layer over the M rows / top_k)."""
    L = sh["decoder_layers"]
    return decode2_ms(sh["D"], sh["R"], h, L) / ((decode2_ms(sh["D"], u * sh["R"], h, L) + draft_ms) / tau)


def project_laguna(a) -> None:
    sh = SHAPES[a.shape]
    cj = json.load(open(a.curve))
    key = "held" if cj.get("split") == "hash" else "in"
    if "answer" not in cj.get("parts", []):
        raise SystemExit("offload_curve project: the curve has no answer rows (traces without answer_from)")

    def h_at(f):
        """The best static fill's answer-position h at f, and which fill."""
        hu, hp = interp(cj["rows"], f, f"uniform_{key}_answer"), interp(cj["rows"], f, f"perlayer_{key}_answer")
        return (hp, "per-layer") if hp >= hu else (hu, "uniform")
    out = {"key": key, "contexts": [], "spec_table": [], "drafter": []}
    print(f"project ({a.shape}): spec 30 §7.1's serial law, B {B_FIT_GBPS:g} GB/s, c {C_LAYER_MS * 1e3:g} us a layer "
          f"(fitted), P {P_LINK_GBPS:g} GB/s; the upper bound B {B_21V2_GBPS:g}, c 0; h = the answer positions' "
          f"best static fill ({key}); all speeds DERIVED")
    print("| h | layer split | tensor parallel | layer split, upper | tensor parallel, upper |")
    print("|---:|---:|---:|---:|---:|")
    for h in (1.0, 0.99, 0.976, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80):
        r = [two_constant_tps(sh, h, tp, up) for up in (False, True) for tp in (False, True)]
        out["spec_table"].append({"h": h, "pp": r[0], "tp": r[1], "pp_upper": r[2], "tp_upper": r[3]})
        print(f"| {h:.3f} | {r[0]:.0f} | {r[1]:.0f} | {r[2]:.0f} | {r[3]:.0f} |")
    print("| context | f | h (measured) | fill | layer split t/s | tensor parallel t/s |")
    print("|---:|---:|---:|---|---:|---:|")
    for ctx, f in CONTEXT_F.items():
        h, fill = h_at(f)
        pp, tp = two_constant_tps(sh, h, False, ctx=ctx), two_constant_tps(sh, h, True, ctx=ctx)
        out["contexts"].append({"ctx": ctx, "f": f, "h": h, "fill": fill, "pp": pp, "tp": tp})
        print(f"| {ctx} | {f:.2f} | {h:.4f} | {fill} | {pp:.1f} | {tp:.1f} |")
    h32 = out["contexts"][0]["h"]
    u8 = None
    if a.mtp:
        u8 = next((x["u"] for x in json.load(open(a.mtp))["K"] if x["M"] == 8), None)
    if u8 is None:
        print("drafter: no 8-row verify union (offload_curve.py mtp --K 7 --json) - not projected")
        drafter = "NOT PROJECTED"
    else:
        print(f"| tau | u(8) | h | drafter speedup (layer split) |")
        print("|---:|---:|---:|---:|")
        for tau in (3.5, 4.5, DRAFT_TAU, 6.0):
            sp = drafter_speedup(sh, h32, u8, tau)
            out["drafter"].append({"tau": tau, "u": u8, "h": h32, "speedup": sp})
            print(f"| {tau:g} | {u8:.3f} | {h32:.4f} | {sp:.2f} |")
        drafter = "KEEP" if drafter_speedup(sh, h32, u8, DRAFT_TAU) >= DRAFT_BAR else "STRUCK"
    verdict = "STOP" if h32 < STOP_H else "NEEDS A DECISION" if h32 < GO_H else "GO"
    out.update(verdict=verdict, drafter_verdict=drafter, h32=h32, u8=u8)
    _dump(a, out)
    print(f"{verdict}: answer-position h {h32:.4f} at f {CONTEXT_F[32768]:.2f} (32,768 tokens, static, {key}); "
          f"thresholds {STOP_H:.2f} / {GO_H:.2f}. DRAFTER: {drafter}" +
          ("" if u8 is None else f" (tau {DRAFT_TAU:g}, u(8) {u8:.3f}: {drafter_speedup(sh, h32, u8, DRAFT_TAU):.2f}x "
                                 f"against {DRAFT_BAR:g}x)"))


def cmd_project(a) -> None:
    if a.shape == "laguna":
        return project_laguna(a)
    if not a.budget:
        raise SystemExit("offload_curve project: --budget is required for this shape")
    budget = read_budget(a.budget)
    p0 = read_p0(a.p0) if a.p0 else {"law": {}, "P_gbps": {}, "alpha": {}, "copy_expert_us": {}, "copy_both_gbps": None,
                                      "lines": 0}
    cj = json.load(open(a.curve))
    key = "held" if cj.get("split") == "hash" else "in"
    rows = cj["rows"]
    swap_gain, swap = 0.0, None
    if a.swap:
        sw = json.load(open(a.swap))
        best = max(sw.get("rows", []), key=lambda r: r["gain"], default=None)
        if best is not None and best["gain"] > 0:
            swap_gain, swap = best["gain"], {"f": sw["f"], "fill": sw.get("fill", "uniform"), "best": best}
    per_dev = {}
    for e in cj.get("budget", []):
        if e.get("per_device"):
            per_dev[(e["max_len"], e["devices"], e["mtp"], e["prefill"], e["dense"])] = e["per_device"]
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
          f"GB/s ({p_src}); h = each card's best held-out fill on its own layers ({key})" +
          (f"; swaps (+{swap_gain:.4f} at f {swap['f']:g}, uniform) only where a card's f is within {SWAP_F_TOL} of it, "
           f"their copies on the link" if swap else ""))
    print("| form | cards | max_len | f per card | h per card | policy | ms a token | pp t/s | tp t/s (h, expert split) |")
    print("|---|---:|---:|---|---|---|---:|---:|---|")

    def tp_line(b, ctx):
        """(b) tp + tier on the same budget: each card holds half of every expert at the two-card f, h from the pooled
        curve (the best static fill), each card on its own link at the slower card's P."""
        Dc, Rc = tp_card_bytes(float(b["read_non_routed"]) + attn_bytes(ctx), float(b["read_routed"]))
        h = max(interp(rows, float(b["f"]), f"uniform_{key}"), interp(rows, float(b["f"]), f"perlayer_{key}"))
        return tp_ms(Dc, Rc, BW, min(P_of.values()), h, law, alpha), h

    def project_line(b, ctx):
        devs = int(b["devices"])
        D = float(b["read_non_routed"]) + attn_bytes(ctx)
        pdl = per_dev.get((b["max_len"], b["devices"], b["mtp"], b["prefill"], b["dense"]))
        t = 0.0
        fs, hs, pols = [], [], []
        for i in range(devs):
            Ri = (int(b["end"][i]) - int(b["first"][i])) * int(b["top_k"]) * int(b["expert_bytes"])
            routed_i = (int(b["end"][i]) - int(b["first"][i])) * int(b["experts"]) * int(b["expert_bytes"])
            fi = min(1.0, float(b["cache"][i]) / routed_i) if routed_i else 1.0
            Pi = P_of.get(i, P_of.get(0))
            share = Ri / float(b["read_routed"])     # the card's share of the layers (and so of D and of the swaps)
            Di = D * share
            best = None
            for hi, pol, moved in card_options(pdl[i] if pdl and i < len(pdl) else None, rows, fi, key, swap):
                ti = decode_ms(Di, Ri, BW, Pi, hi, law, alpha) + moved * share * float(b["expert_bytes"]) / Pi / 1e6
                if best is None or ti < best[0]:
                    best = (ti, hi, pol)
            t += best[0]
            fs.append(fi)
            hs.append(best[1])
            pols.append(best[2])
        return t, fs, hs, " / ".join(dict.fromkeys(pols))
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
        tp_txt = "-"
        if b["devices"] == 2:
            tt, th = tp_line(b, b["max_len"])
            r.update(tp_ms=tt, tp_tps=1e3 / tt, tp_h=th)
            tp_txt = f"{1e3 / tt:.1f} ({th:.3f})"
        out["rows"].append(r)
        form = "ours (int4 dense)" if b["dense"] == "int4" else "Intel's (bf16 dense, informational)"
        print(f"| {form} | {b['devices']} | {b['max_len']} | {' / '.join(f'{x:.3f}' for x in fs)} | "
              f"{' / '.join(f'{x:.3f}' for x in hs)} | {pol} | {t:.2f} | {1e3 / t:.1f} | {tp_txt} |")
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
    verdict = "STOP" if stop_row["tps"] < STOP_PP_TPS and stop_row["tp_tps"] < STOP_TP_TPS else "GO-CANDIDATE"
    out["verdict"] = verdict
    out["stop_inputs"] = {"devices": 2, "max_len": 32768, "dense": "int4", "tps": stop_row["tps"],
                          "tp_tps": stop_row["tp_tps"], "law": law, "policy": stop_row["policy"],
                          "bar_pp_tps": STOP_PP_TPS, "bar_tp_tps": STOP_TP_TPS}
    _dump(a, out)
    print(f"{verdict}: two-card decode at 32k projects {stop_row['tps']:.1f} t/s with the layer split and "
          f"{stop_row['tp_tps']:.1f} with tensor parallel (ours, int4 g64 dense, {stop_row['policy']} fill, {law} law); "
          f"STOP only when both miss ({STOP_PP_TPS:g} / {STOP_TP_TPS:g} t/s); a go is a separate decision")


def project_args(sub) -> None:
    p = sub.add_parser("project")
    p.add_argument("--shape", default="flash-next", choices=sorted(SHAPES))
    p.add_argument("--budget", help="Task 1's budget-json lines (qwen4exp_plan_test --budget-json); not for laguna")
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
    tr = load_traces(a.traces, a.sources, a.shape)
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
    names = SHAPES[a.shape]["layers"] or list(range(L))
    print("| layer | distinct (pooled) | Gini | " + " | ".join(f"top-{n} share" for n in tops) + " |")
    print("|---:|---:|---:|" + "---:|" * len(tops))
    for l in range(L):
        c = pooled[l]
        srt = np.sort(c)[::-1]
        shares = [float(srt[:n].sum() / max(1, c.sum())) for n in tops]
        out["per_layer"].append({"layer": l, "distinct": int((c > 0).sum()), "gini": _gini(c),
                                 "top_share": dict(zip(map(str, tops), shares))})
        print(f"| {names[l]} | {int((c > 0).sum())} | {_gini(c):.3f} | " + " | ".join(f"{s:.3f}" for s in shares) + " |")
    _dump(a, out)


def cmd_curve(a) -> None:
    tr = load_traces(a.traces, a.sources, a.shape)
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
    if r["parts"]:
        print("| f | " + " | ".join(f"{pol} {key} {part}" for part in r["parts"] for pol in ("uniform", "per-layer"))
              + " |")
        print("|---:|" + "---:|" * (2 * len(r["parts"])))
        for row in r["rows"]:
            print(f"| {row['f']:.3f} | " + " | ".join(f"{row[f'{pol}_{key}_{part}']:.4f}" for part in r["parts"]
                                                      for pol in ("uniform", "perlayer")) + " |")
    for k_, v in r["reach"].items():
        print(f"reach {k_}: f = {'not reached' if v is None else f'{v:.3f}'}")
    if a.budget:
        r["budget"] = []
        for b in read_budget(a.budget):
            hu, hp = interp(r["rows"], b["f"], f"uniform_{key}"), interp(r["rows"], b["f"], f"perlayer_{key}")
            pd = []
            if int(b.get("layers", L)) == L:     # each device's h on its own layers, at its own f
                for i in range(int(b["devices"])):
                    lo, hi = int(b["first"][i]), int(b["end"][i])
                    routed_i = (hi - lo) * int(b["experts"]) * int(b["expert_bytes"])
                    fi = min(1.0, float(b["cache"][i]) / routed_i) if routed_i else 1.0
                    pd.append(device_h(tr, E, lo, hi, fi, a.split))
            r["budget"].append({**{k_: b[k_] for k_ in ("max_len", "devices", "mtp", "prefill", "dense")}, "f": b["f"],
                                "h_uniform": hu, "h_perlayer": hp, "per_device": pd})
            print(f"budget max_len {b['max_len']} devices {b['devices']} dense {b['dense']} mtp {b['mtp']} prefill "
                  f"{b['prefill']}: f {b['f']:.4f} -> h uniform {hu:.4f}, per layer {hp:.4f} ({key})" +
                  "".join(f"; device {i} layers [{x['first']}, {x['end']}) f {x['f']:.4f}: uniform {x['h_uniform']:.4f}, "
                          f"per layer {x['h_perlayer']:.4f}" for i, x in enumerate(pd)))
    _dump(a, r)


def cmd_swap(a) -> None:
    tr = load_traces(a.traces, a.sources, a.shape)
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
                hh = tt = steps = 0
                st = {}
                for t in tr:
                    h, n = swap_sim(t, res_of[t.name], S, Bmb * 1e6, W, copy_ms, a.step_ms, stats=st)
                    hh, tt, steps = hh + h, tt + n, steps + t.T
                hs = _ratio(hh, tt)
                rows.append({"S": S, "B_mb": Bmb, "W": W, "h": hs, "gain": hs - static,
                             "moved_per_step": st.get("moved", 0) / max(1, steps)})
                print(f"| {S} | {Bmb:g} | {W} | {hs:.4f} | {hs - static:+.4f} |")
    # The policies of spec 21 v2 §6 from the same held-out start, by h and the link bytes a token (misses + copies).
    eb = EXPERT_BYTES
    steps = sum(t.T for t in tr)
    pols = [{"policy": "static", "h": static, "copied": 0, "link_mb_per_token": (static_t - static_h) * eb / steps / 1e6}]
    runs = [(f"basalt (window {w})", lambda t, w=w: policy_basalt(t, res_of[t.name], w, copy_ms, a.step_ms))
            for w in ilist(a.policy_window)]
    allc = counts_of(tr, L, E)
    runs.append(("saber", lambda t: policy_saber(t, res_of[t.name], allc if one_half else cnt[1 - half[t.name]],
                                                 copy_ms, a.step_ms)))
    for name, run in runs:
        hh = tt = cp = 0
        for t in tr:
            h, n, c = run(t)
            hh, tt, cp = hh + h, tt + n, cp + c
        pols.append({"policy": name, "h": _ratio(hh, tt), "copied": cp,
                     "link_mb_per_token": ((tt - hh) + cp) * eb / steps / 1e6})
    print("| policy | h | gain | experts copied | link MB a token (misses + copies) | against static |")
    print("|---|---:|---:|---:|---:|---:|")
    for p in pols:
        print(f"| {p['policy']} | {p['h']:.4f} | {p['h'] - static:+.4f} | {p['copied']} | {p['link_mb_per_token']:.2f} | "
              f"{p['link_mb_per_token'] / pols[0]['link_mb_per_token']:.3f}x |")
    _dump(a, {"f": a.f, "fill": a.fill, "copy_ms": None if math.isinf(copy_ms) else copy_ms, "step_ms": a.step_ms,
              "static": static, "rows": rows, "policies": pols})


def cmd_mtp(a) -> None:
    tr = load_traces(a.traces, a.sources, a.shape)
    L, E = tr[0].layers, tr[0].experts
    res = resident_uniform(counts_of(tr, L, E), a.f) if a.fill == "uniform" else resident_perlayer(counts_of(tr, L, E), a.f)
    out = {"f": a.f, "fill": a.fill, "K": []}
    print(f"mtp: f {a.f} ({a.fill}, in sample); per layer the union over M = K + 1 consecutive rows")
    print("| K | M | distinct a layer | u(M) | iid prior | misses a verify step | M single rows' misses |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for K in ilist(a.K):
        acc = {"distinct_per_layer": 0.0, "u": 0.0, "iid_per_layer": 0.0, "misses_per_step": 0.0,
               "misses_M_rows_unshared": 0.0}
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
        print(f"| {K} | {K + 1} | {row['distinct_per_layer']:.2f} | {row['u']:.3f} | {row['iid_per_layer']:.2f} | "
              f"{row['misses_per_step']:.2f} | {row['misses_M_rows_unshared']:.2f} |")
    head = mtp_head(tr, mtp_counts_of(tr, E), a.f)
    out["head"] = head
    print(f"mtp head: {head['distinct']} of {head['of']} experts routed over {head['routes']} slots; h if cached at f "
          f"{a.f}: {head['h_cached']:.4f} (decision 6: resident costs one layer's 1.34 GB at int4)")
    _dump(a, out)


def cmd_prefill(a) -> None:
    tr = load_traces(a.traces, a.sources, a.shape)
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
                r = prefill_bytes(t, C, res, SHAPES[a.shape]["expert_bytes"])
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
    tr = load_traces(a.traces, a.sources, a.shape)
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
        p.add_argument("--shape", default="flash-next", choices=sorted(SHAPES))
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
    p.add_argument("--policy-window", default="1,8", help="basalt's rounds: steps a round, a comma list")
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
    project_args(sub)
    a = ap.parse_args(argv)
    {"counts": cmd_counts, "curve": cmd_curve, "swap": cmd_swap, "mtp": cmd_mtp, "prefill": cmd_prefill,
     "profile": cmd_profile, "project": cmd_project}[a.cmd](a)


if __name__ == "__main__":
    main()
