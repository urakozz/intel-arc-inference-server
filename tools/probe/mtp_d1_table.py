#!/usr/bin/env python3
"""Spec 8 D1/D2 table from tools/probe/mtp_d1.sh's log: median tokens/s per (K, prompt)
over the rounds, the speedup against K = 0, and the pooled acceptance."""
import re
import statistics
import sys
from collections import defaultdict

rows = defaultdict(list)
acc = defaultdict(list)
per_iter = defaultdict(list)
for line in open(sys.argv[1]):
    if not line.startswith("BENCH"):
        continue
    f = dict(kv.split("=", 1) for kv in line.split()[1:])
    key = (int(f["k"]), f["prompt"])
    rows[key].append(float(f["tps"]))
    acc[key].append(float(f["acc"]))
    per_iter[key].append(float(f["per_iter"]))

prompts = sorted({p for _, p in rows}, key=lambda p: (not p.startswith("golden"), p))
ks = sorted({k for k, _ in rows})
print("| prompt | " + " | ".join(f"K={k} t/s (x)" for k in ks) + " |")
print("|---|" + "---:|" * len(ks))
geo = defaultdict(list)
for p in prompts:
    base = statistics.median(rows[(0, p)]) if (0, p) in rows else None
    cells = []
    for k in ks:
        if (k, p) not in rows:
            cells.append("-")
            continue
        m = statistics.median(rows[(k, p)])
        x = m / base if base else float("nan")
        group = "golden" if p.startswith("golden") else "toolcall"
        geo[(k, group)].append(x)
        extra = "" if k == 0 else f" ({x:.3f}x, acc {statistics.median(acc[(k, p)]):.3f}, {statistics.median(per_iter[(k, p)]):.2f}/it)"
        cells.append(f"{m:.2f}{extra}")
    print(f"| {p} | " + " | ".join(cells) + " |")
for (k, group), xs in sorted(geo.items()):
    if k == 0:
        continue
    g = 1.0
    for x in xs:
        g *= x
    print(f"geomean {group} K={k}: {g ** (1 / len(xs)):.3f}x over {len(xs)} prompts "
          f"(runs per cell: {len(rows[(k, prompts[0])])})")
