#!/usr/bin/env python3
"""3l: base / new probe_longctx_decode logs paired by round - medians and paired wall reductions per (KV, depth, M).

  compare_longctx_ab.py <dir>   reads <dir>/{base,new}-{int8,bf16}-r<N>.log, prints a markdown table and the
                                3l acceptance verdict (gain >= 5 % at M = 2..4, depths 81920 / 150000, both forms;
                                every other row no worse than -2 %)
"""
import glob
import json
import os
import statistics
import sys


def load(path):
    rows = {}
    with open(path) as f:
        text = f.read()
    if "ALLDONE rc=0" not in text:
        raise SystemExit(f"{path}: incomplete")
    for line in text.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            key = (r["depth"], r["M"])
            if key in rows:
                raise SystemExit(f"{path}: duplicate {key}")
            rows[key] = r
    return rows


def main():
    d = sys.argv[1]
    runs = {}
    for path in sorted(glob.glob(os.path.join(d, "*-r*.log"))):
        tag, kv, rnd = os.path.basename(path)[:-4].split("-")
        runs[(tag, kv, int(rnd[1:]))] = load(path)
    rounds = sorted({k[2] for k in runs})
    print("| KV | depth | M | base wall ms | new wall ms | gain % [range] | base attn ms | new attn ms |")
    print("|:---|---:|---:|---:|---:|:---|---:|---:|")
    fails = []
    for kv in ("bf16", "int8"):
        keys = sorted(runs[("base", kv, rounds[0])])
        for key in keys:
            bw, nw, ba, na, gains = [], [], [], [], []
            for r in rounds:
                b, n = runs[("base", kv, r)][key], runs[("new", kv, r)][key]
                bw.append(b["wall_ms"]); nw.append(n["wall_ms"])
                ba.append(b["prep_ms"] + b["core_ms"] + b["reduce_ms"])
                na.append(n["prep_ms"] + n["core_ms"] + n["reduce_ms"])
                gains.append(100.0 * (b["wall_ms"] - n["wall_ms"]) / b["wall_ms"])
            depth, m = key
            g = statistics.median(gains)
            print(f"| {kv} | {depth} | {m} | {statistics.median(bw):.3f} | {statistics.median(nw):.3f} | "
                  f"{g:.2f} [{min(gains):.2f}, {max(gains):.2f}] | {statistics.median(ba):.3f} | "
                  f"{statistics.median(na):.3f} |")
            if m >= 2 and depth >= 81920:
                if g < 5.0:
                    fails.append(f"{kv} depth {depth} M {m}: gain {g:.2f} % < 5 %")
            elif g < -2.0:
                fails.append(f"{kv} depth {depth} M {m}: {g:.2f} % (regression beyond 2 %)")
    print()
    print("acceptance: " + ("PASS" if not fails else "FAIL - " + "; ".join(fails)))


if __name__ == "__main__":
    main()
