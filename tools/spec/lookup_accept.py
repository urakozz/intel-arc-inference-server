#!/usr/bin/env python3
"""Prompt-lookup speculative acceptance, offline (spec 19e Task 1, plan 19e Review Focus 3-5).

    lookup_accept.py [--log DIR]... [--a4 DIR --a4-expected JSON --tokenizer tokenizer.json]
                     [--ids-json FILE]... [--seed-session N] [--max-match 64]
                     [--mtp-accept 0.92,0.87,0.82] [--verify-cost ...] [--json OUT]

Replays recorded id sequences and measures, for a GREEDY target whose outputs are the
recorded ids, what a prompt-lookup proposer would have had accepted. Inputs:

- `--log DIR`: what `b70-serve --log-requests DIR` writes (NNNNNN.json per request with
  `prompt_ids` and `out_ids`). With `--seed-session N` each request's matcher also holds the
  generated ids of the N requests before it in the log, each behind a boundary no match
  crosses, before the prompt: what `b70-serve --spec lookup --spec-history N` sees.
- `--a4 DIR --a4-expected JSON --tokenizer FILE`: the A4 tool-call set. DIR/<name>.ids is
  the rendered prompt; the reference output is the `text` of the scenario's first entry in
  tests/server/toolcall_expected.json (bf16 backend first, then l0, then l0-int8),
  re-encoded with the tokenizer (needs the `tokenizers` package) plus <|im_end|> when the
  call is complete. Re-encoding is how the reference is held; the model's own ids can
  differ where a text has two tokenisations.
- `--ids-json FILE`: [{"name", "prompt_ids", "out_ids"}, ...] (any other source).

**The protocol is the engine's** (runtime::Engine, spec 8 §3.3): at an iteration the pending
id x_n is known (the previous verify's sample), the matcher holds prompt + output up to and
including x_n, and the drafts are its continuation d_1..d_K. A verify of M = K + 1 rows keeps
the longest prefix of drafts equal to the recorded ids, j, and emits x_n, d_1..d_j (j + 1
ids); the next pending id is the target's own. No match: one plain step (1 id). Tokens per
verify counts the iterations that verified; the speed-up counts every iteration, at cost
verify(K + 1) for a verify (the lookup's draft cost is host-side and ~0) and 1 for a plain
step.

**The matcher** is src/server/prompt_lookup.h's semantics with no candidate cap: the
longest earlier match of the context's suffix, capped at --max-match (64), the most recent
occurrence on ties, a draft only when the match is >= n; the continuation copies forward
from the match (overlapping a repeat: a run of one id drafts that id again).

**The cost model** (spec 8 §10): plain step = 1. Verify M = 1..4 is the int8-head table
1.00 / 1.18 / 1.56 / 1.79 (derived, spec 8 §10); M = 5..8 is ESTIMATED as its linear
extension (+0.305 per row: 2.10 / 2.40 / 2.71 / 3.01, spec 19 §3 A's ~2.6-3.1 at M = 8).
MTP drafts cost 0.13 / 0.26 / 0.38 at K = 1 / 2 / 3 (int8 head). `--mtp auto` is projected
analytically from `--mtp-accept` read as per-depth acceptance a_1, a_2, a_3 (P(draft d kept
| d - 1 kept); spec 8 A11's 0.92 / 0.87 / 0.82 by default): E(K) = 1 + a_1 + a_1 a_2 + ...,
the rate max_K E(K) / (verify(K + 1) + draft(K)). The combination (Review Focus 5: one
proposer per iteration) verifies the lookup's drafts when its match is >= n and otherwise an
MTP iteration at --mtp auto's best K, whose kept count is drawn from the same per-depth
Bernoulli model (seeded; mean over --mc-seeds) - so it assumes MTP accepts at its average
rate where lookup has no match, which over-rates it (copy-like text is where both accept).

Dependency-free Python 3 (the `tokenizers` package only for --a4). Tests:
`python3 tools/spec/test_lookup_accept.py`.
"""
import argparse
import json
import os
import random
import sys

LEVELS = (2, 4, 8, 16, 32, 64)   # the C++ matcher's key lengths (prompt_lookup.h)
VERIFY_INT8 = [1.00, 1.18, 1.56, 1.79]   # spec 8 §10, int8 head, derived
VERIFY_SLOPE = (VERIFY_INT8[3] - VERIFY_INT8[1]) / 2   # 0.305 per row, the M = 2..4 slope
VERIFY_EST = VERIFY_INT8 + [round(VERIFY_INT8[3] + VERIFY_SLOPE * r, 3) for r in range(1, 5)]
MTP_DRAFT_INT8 = [0.13, 0.26, 0.38]
IM_END = 248046
BOUNDARY = 0xFFFFFFFF   # boundary ids count down from here; never a real id


class Lookup:
    """The longest earlier match of the context's suffix (capped), most recent on ties.

    `append` is O(len(LEVELS)); `propose` walks, from the longest key length down, every
    earlier end position whose last k ids equal the context's (no candidate cap here: this is
    the reference the C++ matcher's capped walk is held to)."""

    def __init__(self, max_match=64):
        if max_match < LEVELS[0]:
            raise ValueError("max_match must be >= 2")
        self.max_match = max_match
        self.levels = tuple(k for k in LEVELS if k <= max_match)
        self.s = []
        self.chains = {k: {} for k in self.levels}
        self.next_boundary = BOUNDARY

    def append(self, tok):
        s = self.s
        s.append(tok)
        t = len(s)
        for k in self.levels:
            if t >= k:
                self.chains[k].setdefault(tuple(s[t - k:]), []).append(t - 1)

    def extend(self, ids):
        for tok in ids:
            self.append(tok)

    def boundary(self):
        """A segment end (an earlier request of the session): a unique id, so no match and
        no continuation crosses it."""
        self.append(self.next_boundary)
        self.next_boundary -= 1

    def match(self, min_n):
        """(end, length) of the chosen earlier occurrence, or (None, best length < min_n)."""
        s = self.s
        t = len(s)
        for li in range(len(self.levels) - 1, -1, -1):
            k = self.levels[li]
            if k > t - 1:
                continue
            # A level below the largest level <= min_n cannot reach min_n.
            if li + 1 < len(self.levels) and self.levels[li + 1] <= min_n:
                break
            ends = self.chains[k].get(tuple(s[t - k:]))
            if not ends or len(ends) < 2:
                continue   # only the current end itself
            best_e, best_m = None, -1
            for e in reversed(ends[:-1]):   # most recent first; ends[-1] == t - 1
                m = k
                while m < self.max_match and e - m >= 0 and s[e - m] == s[t - 1 - m]:
                    m += 1
                if m > best_m:
                    best_e, best_m = e, m
                    if m == self.max_match:
                        break
            if best_m >= min_n:
                return best_e, best_m
            return None, best_m
        return None, 0

    def continuation(self, e, max_k):
        s = self.s
        t = len(s)
        out = []
        for j in range(1, max_k + 1):
            idx = e + j
            tok = s[idx] if idx < t else out[idx - t]
            if tok > BOUNDARY - (1 << 20):   # a boundary: stop before it
                break
            out.append(tok)
        return out

    def propose(self, max_k, min_n):
        e, m = self.match(min_n)
        if e is None:
            return [], m
        return self.continuation(e, max_k), m


def brute_match(s, min_n, max_match):
    """Reference: scan every earlier end; longest common suffix (capped), most recent on ties."""
    t = len(s)
    best_e, best_m = None, 0
    for e in range(t - 2, -1, -1):
        m = 0
        while m < max_match and e - m >= 0 and s[e - m] == s[t - 1 - m]:
            m += 1
        if m > best_m:
            best_e, best_m = e, m
    if best_m >= min_n:
        return best_e, best_m
    return None, best_m


class AdaptiveK:
    """server::AdaptiveK (src/server/adaptive_k.h), line for line."""

    def __init__(self, verify, draft, max_k, prior=0.8, prior_weight=4, decay=0.9, warmup=2,
                 hysteresis=0.03, probe_every=4):
        self.verify, self.draft = verify, draft
        self.max_k = min(max_k, min(len(verify) - 1, len(draft)))
        self.s, self.t = prior * prior_weight, prior_weight
        self.decay, self.warmup, self.hyst, self.probe_every = decay, warmup, hysteresis, probe_every
        self.iters, self.cur, self.plain_run = 0, self.max_k, 0

    def cost(self, k):
        return 1.0 if k == 0 else self.verify[k] + self.draft[k - 1]

    @staticmethod
    def expected_ids(a, k):
        return k + 1.0 if a >= 1.0 else (1.0 - a ** (k + 1)) / (1.0 - a)

    def rate(self, a, k):
        return self.expected_ids(a, k) / self.cost(k)

    def best(self, a):
        b, r = 0, self.rate(a, 0)
        for k in range(1, self.max_k + 1):
            rk = self.rate(a, k)
            if rk > r:
                b, r = k, rk
        return b

    def next(self):
        if self.iters < self.warmup:
            return self.max_k
        if self.cur == 0 and self.max_k > 0 and self.probe_every > 0 and \
                self.plain_run >= self.probe_every:
            return 1
        return self.cur

    def observe(self, k, accepted):
        self.iters += 1
        if k == 0:
            self.plain_run += 1
        else:
            self.plain_run = 0
            tried = accepted + 1.0 if accepted < k else float(k)
            self.s = self.decay * self.s + accepted
            self.t = self.decay * self.t + tried
        if self.iters < self.warmup:
            return
        a = self.s / self.t
        b = self.best(a)
        if b != self.cur and self.rate(a, b) > self.rate(a, self.cur) * (1.0 + self.hyst):
            self.cur = b


def mtp_expected(accept, k):
    e, p = 1.0, 1.0
    for d in range(k):
        p *= accept[d]
        e += p
    return e


def mtp_auto_rate(accept, verify, draft):
    """--mtp auto's projected ids per plain-step cost: the best fixed K (0..3)."""
    best_k, best_r = 0, 1.0
    for k in range(1, min(len(accept), len(draft), len(verify) - 1) + 1):
        r = mtp_expected(accept, k) / (verify[k] + draft[k - 1])
        if r > best_r:
            best_k, best_r = k, r
    return best_k, best_r


class Totals:
    def __init__(self):
        self.ids = 0           # recorded output ids
        self.cost = 0.0        # plain-step units
        self.verifies = 0      # lookup verify iterations
        self.verified_ids = 0  # ids those iterations emitted (j + 1 each)
        self.drafted = 0
        self.accepted = 0
        self.plain = 0
        self.mtp_iters = 0

    def add(self, o):
        for k, v in vars(o).items():
            setattr(self, k, getattr(self, k) + v)

    def row(self):
        return {"ids": self.ids, "cost": round(self.cost, 3),
                "tokens_per_verify": self.verified_ids / self.verifies if self.verifies else 0.0,
                "accept_rate": self.accepted / self.drafted if self.drafted else 0.0,
                "verify_share": self.verifies / max(1, self.verifies + self.plain + self.mtp_iters),
                "speedup": self.ids / self.cost if self.cost else 0.0}


def make_matcher(seq, max_match):
    lk = Lookup(max_match)
    for seg in seq.get("seed", []):
        lk.extend(seg)
        lk.boundary()
    lk.extend(seq["prompt_ids"])
    return lk


def precompute(seq, max_match, max_k):
    """Per pending index i: (m_i, a_i). The target is greedy and recorded, so the context at
    pending index i is always prompt + out[:i + 1], whatever earlier iterations kept: the
    match is a function of i alone. m_i is the match length (0: none), a_i how many of the
    continuation's first max_k drafts equal the recorded ids after i. Cached on the
    sequence (keyed by max_match, max_k)."""
    key = ("pre", max_match, max_k)
    if key in seq:
        return seq[key]
    out = seq["out_ids"]
    lk = make_matcher(seq, max_match)
    rows = []
    for i, tok in enumerate(out):
        lk.append(tok)
        drafts, m = lk.propose(max_k, LEVELS[0])
        a = 0
        while a < len(drafts) and i + 1 + a < len(out) and drafts[a] == out[i + 1 + a]:
            a += 1
        rows.append((m if drafts else 0, a, len(drafts)))
    seq[key] = rows
    return rows


def simulate(seq, k, n, verify, max_match=64, mode="lookup", mtp=None, rng=None, max_k=7):
    """One sequence. mode: "lookup" (fixed K), "auto" (AdaptiveK with draft cost 0, K <= k),
    "combo" (lookup when its match >= n, else MTP at mtp["k"] drafts with mtp["accept"])."""
    out = seq["out_ids"]
    tot = Totals()
    tot.ids = len(out)
    if not out:
        return tot
    pre = precompute(seq, max_match, max(max_k, k))
    i = 0
    pol = AdaptiveK(verify, [0.0] * (len(verify) - 1), k) if mode == "auto" else None
    while i < len(out):
        kk = pol.next() if pol else k
        m, a, avail = pre[i]
        if kk > 0 and m >= n and avail > 0:
            kd = min(kk, avail)
            j = min(a, kd)
            tot.cost += verify[kd]
            tot.verifies += 1
            tot.verified_ids += j + 1
            tot.drafted += kd
            tot.accepted += j
            emitted = j + 1
            if pol:
                pol.observe(kd, j)
        elif mode == "combo" and mtp["k"] > 0:
            km = mtp["k"]
            j = 0
            while j < km and rng.random() < mtp["accept"][j]:
                j += 1
            j = min(j, len(out) - i - 1)
            tot.cost += verify[km] + mtp["draft"][km - 1]
            tot.mtp_iters += 1
            emitted = j + 1
        else:
            tot.cost += 1.0
            tot.plain += 1
            emitted = 1
            if pol:
                pol.observe(0, 0)
        i += emitted
    return tot


def run_all(seqs, k, n, verify, **kw):
    tot = Totals()
    for s in seqs:
        tot.add(simulate(s, k, n, verify, **kw))
    return tot


def match_profile(seqs, n, k, verify, max_match=64):
    """Iterations by match length bucket at fixed K: count, tokens per verify."""
    buckets = [(2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 64)]
    stats = {b: [0, 0] for b in buckets}
    for seq in seqs:
        out = seq["out_ids"]
        pre = precompute(seq, max_match, max(7, k))
        i = 0
        while i < len(out):
            m, a, avail = pre[i]
            emitted = 1
            if m >= n and avail > 0:
                emitted = min(a, k, avail) + 1
                for b in buckets:
                    if b[0] <= m <= b[1]:
                        stats[b][0] += 1
                        stats[b][1] += emitted
            i += emitted
    return [(f"{a}-{b}" if a != b else f"{a}", c, (e / c if c else 0.0))
            for (a, b), (c, e) in stats.items()]


# ---- inputs ------------------------------------------------------------------------------

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(w) for w in f.read().split()]


def load_logs(log_dir, seed_session=0):
    names = sorted(n for n in os.listdir(log_dir) if n.endswith(".json") and n[:-5].isdigit())
    seqs, history = [], []
    for n in names:
        with open(os.path.join(log_dir, n), encoding="utf-8") as f:
            r = json.load(f)
        s = {"name": f"{os.path.basename(os.path.normpath(log_dir))}/{n}",
             "prompt_ids": r["prompt_ids"], "out_ids": r.get("out_ids", [])}
        if seed_session > 0:
            s["seed"] = history[-seed_session:]
        history.append(list(r.get("out_ids", [])))
        seqs.append(s)
    return seqs


def load_a4(a4_dir, expected_path, tokenizer_path):
    from tokenizers import Tokenizer   # only this source needs it
    tok = Tokenizer.from_file(tokenizer_path)
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)
    order = {"bf16": 0, "l0": 1, "l0-int8": 2}
    chosen = {}
    for e in sorted(expected, key=lambda e: order.get(e["backend"], 9)):
        chosen.setdefault(e["scenario"], e)
    seqs = []
    for name in sorted(chosen):
        e = chosen[name]
        path = os.path.join(a4_dir, name + ".ids")
        if not os.path.exists(path):
            continue
        out = tok.encode(e["text"], add_special_tokens=False).ids
        if e["kind"] == "call":
            out.append(IM_END)
        seqs.append({"name": name, "prompt_ids": read_ids(path), "out_ids": out,
                     "group": name.split("-")[0]})
    return seqs


# ---- report ------------------------------------------------------------------------------

def report(seqs, label, args, out):
    verify = args.verify
    accept = args.mtp_accept
    mk, mrate = mtp_auto_rate(accept, verify, MTP_DRAFT_INT8)
    n_ids = sum(len(s["out_ids"]) for s in seqs)
    p_ids = sum(len(s["prompt_ids"]) for s in seqs)
    print(f"\n## {label}: {len(seqs)} sequences, {p_ids} prompt ids, {n_ids} output ids\n")
    print(f"--mtp auto (projected, a = {'/'.join(f'{a:g}' for a in accept)}): K = {mk}, "
          f"{mrate:.3f}x plain\n")
    res = {"label": label, "sequences": len(seqs), "prompt_ids": p_ids, "output_ids": n_ids,
           "mtp_auto": {"k": mk, "rate": mrate}, "grid": {}, "auto": {}, "combo": {}}
    ks = range(1, args.max_k + 1)
    ns = range(2, 7)
    print("Tokens per verify (iterations that verified) | speed-up vs plain, by K and n:\n")
    print("| n \\ K | " + " | ".join(f"K={k}{'*' if k > 3 else ''}" for k in ks) + " |")
    print("|---|" + "---|" * len(ks))
    for n in ns:
        cells = []
        for k in ks:
            r = run_all(seqs, k, n, verify, max_match=args.max_match).row()
            res["grid"][f"{n},{k}"] = r
            cells.append(f"{r['tokens_per_verify']:.2f} / {r['speedup']:.3f}x")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n(* K > 3: verify M = 5..8 costs estimated; the engine's verify lists stop at M = 4.)")
    print("\nVerify share (iterations with a match >= n) and per-draft acceptance, K = 3:\n")
    print("| n | verify share | accepted / drafted |")
    print("|---|---|---|")
    for n in ns:
        r = res["grid"][f"{n},3"]
        print(f"| {n} | {r['verify_share']:.3f} | {r['accept_rate']:.3f} |")
    print("\n`--spec lookup` with spec 8 §10's policy (AdaptiveK, draft cost 0), K <= max:\n")
    print("| n | K <= 3 | K <= 7* |")
    print("|---|---|---|")
    for n in ns:
        cells = []
        for kmax in (3, 7):
            if kmax > args.max_k:
                cells.append("-")
                continue
            r = run_all(seqs, kmax, n, verify, max_match=args.max_match, mode="auto").row()
            res["auto"][f"{n},{kmax}"] = r
            cells.append(f"{r['speedup']:.3f}x ({r['speedup'] / mrate:.3f}x mtp auto)")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print(f"\nCombined (lookup when match >= n at K, else MTP K = {mk}; MTP kept ids drawn per "
          f"depth, mean of {args.mc_seeds} seeds): speed-up vs plain / vs --mtp auto:\n")
    print("| n \\ K | " + " | ".join(f"K={k}{'*' if k > 3 else ''}" for k in ks) + " |")
    print("|---|" + "---|" * len(ks))
    mtp = {"k": mk, "accept": accept, "draft": MTP_DRAFT_INT8}
    for n in ns:
        cells = []
        for k in ks:
            sp = 0.0
            for seed in range(args.mc_seeds):
                rng = random.Random(1000 + seed)
                sp += run_all(seqs, k, n, verify, max_match=args.max_match, mode="combo",
                              mtp=mtp, rng=rng).row()["speedup"]
            sp /= args.mc_seeds
            res["combo"][f"{n},{k}"] = sp
            cells.append(f"{sp:.3f}x / {sp / mrate:.3f}x")
        print(f"| {n} | " + " | ".join(cells) + " |")
    prof = match_profile(seqs, 2, 3, verify, args.max_match)
    print("\nBy match length (n = 2, K = 3): iterations, tokens per verify:\n")
    print("| match | iterations | tokens/verify |")
    print("|---|---|---|")
    for b, c, t in prof:
        print(f"| {b} | {c} | {t:.2f} |")
    res["profile"] = prof
    groups = sorted({s.get("group") for s in seqs if s.get("group")})
    if groups:
        print("\nBy scenario group (n = 3, K = 3, fixed): tokens per verify / speed-up:\n")
        print("| group | out ids | tokens/verify | speed-up | auto K<=3 |")
        print("|---|---|---|---|---|")
        res["groups"] = {}
        for g in groups:
            sub = [s for s in seqs if s.get("group") == g]
            r = run_all(sub, 3, 3, verify, max_match=args.max_match).row()
            a = run_all(sub, 3, 3, verify, max_match=args.max_match, mode="auto").row()
            res["groups"][g] = {"fixed": r, "auto": a}
            print(f"| {g} | {r['ids']} | {r['tokens_per_verify']:.2f} | {r['speedup']:.3f}x | "
                  f"{a['speedup']:.3f}x |")
    out.append(res)


def parse_floats(text):
    return [float(x) for x in text.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", action="append", default=[], help="a --log-requests directory")
    ap.add_argument("--seed-session", type=int, default=0, metavar="N",
                    help="also match the generated ids of the N requests before each one")
    ap.add_argument("--a4", help="tests/golden/toolcall")
    ap.add_argument("--a4-expected", help="tests/server/toolcall_expected.json")
    ap.add_argument("--tokenizer", help="the checkpoint's tokenizer.json (for --a4)")
    ap.add_argument("--ids-json", action="append", default=[])
    ap.add_argument("--max-match", type=int, default=64)
    ap.add_argument("--max-k", type=int, default=7)
    ap.add_argument("--mtp-accept", type=parse_floats, default=[0.92, 0.87, 0.82])
    ap.add_argument("--verify-cost", type=parse_floats, default=VERIFY_EST,
                    help="verify M = 1..8 in plain steps (default: int8 table + estimate)")
    ap.add_argument("--mc-seeds", type=int, default=20)
    ap.add_argument("--json", help="write every number here")
    args = ap.parse_args(argv)
    args.verify = args.verify_cost
    if len(args.verify) < args.max_k + 1:
        ap.error(f"--verify-cost needs M = 1..{args.max_k + 1}")
    corpora = []
    if args.a4:
        if not (args.a4_expected and args.tokenizer):
            ap.error("--a4 needs --a4-expected and --tokenizer")
        corpora.append(("A4 tool-call set (reference outputs re-encoded)",
                        load_a4(args.a4, args.a4_expected, args.tokenizer)))
    for d in args.log:
        corpora.append((f"log {d}" + (f" (seeded: {args.seed_session} earlier outputs)"
                                      if args.seed_session else ""),
                        load_logs(d, args.seed_session)))
    for p in args.ids_json:
        with open(p, encoding="utf-8") as f:
            corpora.append((p, json.load(f)))
    if not corpora:
        ap.error("no input: --a4, --log or --ids-json")
    print(f"verify cost M = 1..{len(args.verify)}: " + " / ".join(f"{v:g}" for v in args.verify)
          + f" (plain steps); max match {args.max_match}")
    out = []
    for label, seqs in corpora:
        report(seqs, label, args, out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
