#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Price the DFlash 2 verify stop on the per-anchor acceptance records (CPU only).

    price_stop.py [accept.json] [--variant int8h-c64]

Reads the CPU job's per-anchor records (variants[<variant>][<set>]["_extra"]["anchors"]: per anchor
`q`, the greedy accepted `length`, `censored`) and prices, per set (corpus) and mode (one card, pp,
tp at depth 4096 - runtime/inkling2/inkling2_window.h's tables, MtpCost::inkling_dflash2*), the stop
variants in ids per plain step:

- `today`: the cost stop on the stated-probability prior (server::DraftCalib as it stands: prior
  10 * p, decay 0.98), at each decay in --decays;
- `seed<W>`: the same stop with the measured per-(position, bin) rates as the prior at weight W
  (cells with too few samples fall back to the stated probability), at each decay;
- `K7`: verify all seven drafts.

The seed comes from the seed sets only (--seed, default below); the held-out sets price it. The
seed table is the one gen_dflash2_seed.py writes into the engine - same computation, same split.
"""
import os
import sys

# this directory holds tokenize.py, which would shadow the stdlib's when run as a script
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != os.path.dirname(os.path.abspath(__file__))]

import argparse  # noqa: E402
import json  # noqa: E402

# runtime/inkling2/inkling2_window.h at 4k: verify T(1..8), the draft pass (flat in K).
TABLES = {
    "one card": ([1.000, 1.167, 1.355, 1.503, 1.679, 1.826, 1.963, 2.175], 0.232),
    "pp": ([1.000, 1.177, 1.368, 1.518, 1.702, 1.844, 1.987, 2.194], 0.231),
    "tp": ([1.000, 1.134, 1.275, 1.405, 1.552, 1.663, 1.796, 2.027], 0.290),
}
# server::DraftCalib: the bins' upper edges, the stated-probability prior's weight, the min samples
# a seed cell needs to be used (kSeedMinN), the positions and bins.
EDGE = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995]
PRIOR_N = 10.0
SEED_MIN_N = 20
POSITIONS, BINS = 7, 9

# The seed split (named here and in gen_dflash2_seed.py): the A4 tool-call sets plus code and chat
# seed the table; the agentic and prose sets (and code under the int8 cache) are held out for the
# check only.
SEED_SETS = ("a4", "code", "chat")
HELD_OUT_SETS = ("agentic", "code-kv8", "prose")


def bin_of(p):
    b = 0
    while b < BINS - 1 and p >= EDGE[b]:
        b += 1
    return b


class Calib:
    """Twin of server::DraftCalib: `seed` (POSITIONS x BINS of (rate, n)) replaces the stated
    probability as the prior at `weight` where the cell has SEED_MIN_N samples; decay per test."""

    def __init__(self, seed=None, weight=PRIOR_N, decay=0.98):
        self.ok = [[0.0] * BINS for _ in range(POSITIONS)]
        self.n = [[0.0] * BINS for _ in range(POSITIONS)]
        self.seed = seed
        self.weight = weight
        self.decay = decay

    def rate(self, i, p):
        b = bin_of(p)
        if self.seed is not None and self.seed[i][b][1] >= SEED_MIN_N:
            return (self.ok[i][b] + self.weight * self.seed[i][b][0]) / (self.n[i][b] + self.weight)
        c = min(max(p, 0.0), 1.0)
        return (self.ok[i][b] + PRIOR_N * c) / (self.n[i][b] + PRIOR_N)

    def observe(self, i, p, accepted):
        b = bin_of(p)
        self.ok[i][b] = self.decay * self.ok[i][b] + (1.0 if accepted else 0.0)
        self.n[i][b] = self.decay * self.n[i][b] + 1.0


def load_anchors(path, variant):
    """[(set, corpus, q, length, censored)] in the file's order."""
    with open(path) as f:
        d = json.load(f)["variants"][variant]
    out = []
    for name, v in d.items():
        an = v["_extra"]["anchors"]
        for q, length, censored in zip(an["q"], an["length"], an["censored"]):
            out.append((name, name.split("/")[0], q, length, censored))
    return out


def seed_table(anchors):
    """The measured conditional acceptance rate per (position, bin): a draft at position i counts as
    tested only when every draft before it was accepted, censored anchors skipped. Returns
    POSITIONS x BINS of (rate, n)."""
    ok = [[0] * BINS for _ in range(POSITIONS)]
    n = [[0] * BINS for _ in range(POSITIONS)]
    for name, corp, q, length, censored in anchors:
        if corp not in SEED_SETS or censored:
            continue
        for i in range(min(length + 1, POSITIONS)):
            b = bin_of(q[i])
            n[i][b] += 1
            if i < length:
                ok[i][b] += 1
    return [[(ok[i][b] / n[i][b] if n[i][b] else 0.0, n[i][b]) for b in range(BINS)] for i in range(POSITIONS)]


def price_stop(anchors, table, calib):
    """The cost stop over every anchor, one calib carried across the sets in file order (a session).
    Returns {corpus: [ids, cost]} for the stop, plus the same for fixed K = 7."""
    verify, draft = table
    stop = {}
    k7 = {}
    for name, corp, q, length, censored in anchors:
        if censored:
            continue
        s = stop.setdefault(corp, [0.0, 0.0])
        k = k7.setdefault(corp, [0.0, 0.0])
        k[0] += 1 + min(length, 7)
        k[1] += verify[7] + draft
        kept, ids, best, at = 1.0, 1.0, 0.0, 7
        for j in range(POSITIONS):
            kept *= calib.rate(j, q[j])
            ids += kept
            r = ids / (verify[j + 1] + draft)
            if r > best:
                best, at = r, j + 1
        s[0] += 1 + min(length, at)
        s[1] += verify[at] + draft
        for j in range(min(at, length + 1)):
            calib.observe(j, q[j], j < length)
    return stop, k7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?",
                    default=os.path.join(os.environ.get("B70_DATA", "/nonexistent"),
                                         "oracle-out/inkling/df2-stop/accept.int8h.json"))
    ap.add_argument("--variant", default="int8h-c64")
    ap.add_argument("--weights", default="10,30,100", help="seed prior weights to price")
    ap.add_argument("--decays", default="0.98,0.995", help="decays to price")
    args = ap.parse_args()

    anchors = load_anchors(args.json, args.variant)
    seed = seed_table(anchors)
    weights = [float(w) for w in args.weights.split(",")]
    decays = [float(d) for d in args.decays.split(",")]
    variants = ([("today", None, PRIOR_N)] + [(f"seed{int(w)}", seed, w) for w in weights])

    def tags():
        return [label if len(decays) == 1 else f"{label} d{d:g}"
                for label, _, _ in variants for d in decays] + ["K7"]

    held = [(n, "held", q, l, c) for n, cr, q, l, c in anchors if cr in HELD_OUT_SETS]
    every = [(n, "all", q, l, c) for n, _, q, l, c in anchors]
    for mode, table in TABLES.items():
        print(f"## {mode}")
        rows = {}
        for label, sd, w in variants:
            for d in decays:
                tag = label if len(decays) == 1 else f"{label} d{d:g}"
                for recs in (anchors, every, held):
                    stop, k7 = price_stop(recs, table, Calib(sd, w, d))
                    for corp, (i, c) in stop.items():
                        rows.setdefault(corp, {})[tag] = i / c
                    for corp, (i, c) in k7.items():
                        rows.setdefault(corp, {})["K7"] = i / c
        print("  " + " ".join(f"{t:>14s}" for t in tags()))
        corps = sorted(c for c in rows if c not in ("all", "held"))
        for corp in corps + ["all", "held-out"]:
            key = "held" if corp == "held-out" else corp
            r = rows.get(key)
            if r is None:
                continue
            print(f"  {corp:9s}" + " ".join(f"{r.get(t, 0.0):14.3f}" for t in tags()))


if __name__ == "__main__":
    sys.exit(main())
