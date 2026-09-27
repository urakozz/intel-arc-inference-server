#!/usr/bin/env python3
"""The prefix reuse a request log allows (spec 7 P0).

    analyze_log.py <log dir>

<log dir> is what `b70-serve --log-requests DIR` writes: NNNNNN.json per request
with prompt_ids and out_ids. For each request: its prompt length, the longest
common prefix with the previous request's ids (prompt + generated) and with any
earlier request's, which earlier request gives that best match, whether the match
ends exactly at an earlier prompt's end or inside an earlier request's generated
ids, and how many prompt tokens a perfect prefix cache could reuse (the match,
capped at prompt length - 1: the last prompt token is always prefilled).
"""
import json
import os
import sys


def load(log_dir):
    names = sorted(n for n in os.listdir(log_dir) if n.endswith(".json") and n[:-5].isdigit())
    out = []
    for n in names:
        with open(os.path.join(log_dir, n), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def analyze(records):
    rows = []
    seqs = []
    total_prompt = 0
    total_reusable = 0
    for i, r in enumerate(records):
        prompt = r["prompt_ids"]
        prev = lcp(prompt, seqs[-1]) if seqs else 0
        best, best_n = 0, None
        for j, s in enumerate(seqs):
            m = lcp(prompt, s)
            if m >= best and m > 0:
                best, best_n = m, j + 1
        at_end = any(len(records[j]["prompt_ids"]) == best and best > 0 and
                     prompt[:best] == records[j]["prompt_ids"] for j in range(i))
        in_gen = best_n is not None and best > len(records[best_n - 1]["prompt_ids"])
        reusable = min(best, max(len(prompt) - 1, 0))
        rows.append({"n": i + 1, "endpoint": r.get("endpoint", ""), "prompt": len(prompt),
                     "out": len(r.get("out_ids", [])), "lcp_prev": prev, "lcp_any": best,
                     "best": best_n, "at_prompt_end": at_end, "in_generated": in_gen,
                     "reusable": reusable})
        total_prompt += len(prompt)
        total_reusable += reusable
        seqs.append(prompt + r.get("out_ids", []))
    totals = {"requests": len(records), "prompt_tokens": total_prompt, "reusable": total_reusable,
              "fraction": total_reusable / total_prompt if total_prompt else 0.0}
    return rows, totals


def main():
    rows, totals = analyze(load(sys.argv[1]))
    print("| req | endpoint | prompt | out | lcp prev | lcp any | best | ends at | reusable |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        where = ("prompt end" if r["at_prompt_end"] else
                 "generated" if r["in_generated"] else "-" if r["best"] is None else "inside")
        print(f"| {r['n']} | {r['endpoint']} | {r['prompt']} | {r['out']} | {r['lcp_prev']} | "
              f"{r['lcp_any']} | {r['best'] or '-'} | {where} | {r['reusable']} |")
    print()
    print(f"{totals['requests']} requests, {totals['prompt_tokens']} prompt tokens, "
          f"{totals['reusable']} reusable by a perfect prefix cache ({100 * totals['fraction']:.1f}%)")


if __name__ == "__main__":
    main()
