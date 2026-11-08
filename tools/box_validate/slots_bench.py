#!/usr/bin/env python3
"""Several requests at once on one card (plan 13c Task 5, spec 13 §11): b70-serve's throughput and latency, stdlib only.

  slots_bench.py --url URL --server-log DIR [--counts 1,2,4,8] [--max-tokens 512] [--cold-tokens 32] [--label L]
                 [--prompt tests/golden/prompts/long.txt] [--out FILE]

For each count N, two rounds of N /v1/completions requests sent together, greedy, ignore_eos, each the prompt
(long.txt, 2820 ids) behind its own first line (no two share a prefix):

  cold   --cold-tokens ids each, the prompts new to the server: TTFT under load - t_first_token - t_start on the
         server's clock (its --log-requests records, matched by the request's `user`): queue wait + its own prefill +
         the other requests' prefill chunks and steps meanwhile;
  warm   the same N prompts again, --max-tokens ids each: the prefix cache (on) restores every prompt, so the N
         requests start decoding nearly together and the card decodes all of them at once. Reported:
           throughput    every request's ids after its first / (the last t_end - the first t_first_token): the card's
                         decode rate over the requests (with N above the server's slots, the waves after one another)
           per request   the median of (ids - 1) / (t_end - t_first_token)
           overlap       the share of the decode spans all N requests decode together (1.0: fully together)

Prints `SLOTS_BENCH <label> N=<n> ...` per count and a table. Needs the server's prefix cache ON (the warm round).
"""
import argparse
import glob
import json
import os
import statistics
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def records(d, users):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        u = j.get("request", {}).get("user")
        if u in users:
            out[u] = j
    return out


def together(base, bodies, timeout):
    errs = []

    def go(b):
        try:
            post(base + "/v1/completions", b, timeout)
        except Exception as e:  # noqa: BLE001 - recorded, the row fails
            errs.append(f"{b['user']}: {e}")

    ts = [threading.Thread(target=go, args=(b,)) for b in bodies]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8013")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--cold-tokens", type=int, default=32)
    ap.add_argument("--label", default="slots")
    ap.add_argument("--prompt", default=os.path.join(ROOT, "tests", "golden", "prompts", "long.txt"))
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    with open(a.prompt, encoding="utf-8") as f:
        text = f.read()
    base = a.url.rstrip("/")
    post(base + "/v1/completions", {"prompt": "Warm up.", "max_tokens": 8, "temperature": 0}, a.timeout)
    rows = []
    for n in [int(x) for x in a.counts.split(",")]:
        prompts = [f"Request {n}.{i}. Read the text below.\n\n" + text for i in range(n)]   # the same in every arm
        res = {}
        for phase, toks in (("cold", a.cold_tokens), ("warm", a.max_tokens)):
            users = [f"{a.label}.n{n}.{phase}.{i}" for i in range(n)]
            bodies = [{"prompt": p, "max_tokens": toks, "temperature": 0, "ignore_eos": True, "user": u}
                      for p, u in zip(prompts, users)]
            errs = together(base, bodies, a.timeout)
            if errs:
                print(f"SLOTS_BENCH {a.label} N={n} {phase}: {len(errs)} request(s) FAILED: {errs[0]}", flush=True)
                break
            time.sleep(0.3)   # the records are written after the reply
            rec = records(a.server_log, set(users))
            if len(rec) != n:
                print(f"SLOTS_BENCH {a.label} N={n} {phase}: {len(rec)}/{n} server records found", flush=True)
                break
            js = [rec[u] for u in users]
            ttft = [j["t_first_token"] - j["t_start"] for j in js]
            spans = [(j["t_first_token"], j["t_end"], len(j.get("out_ids", []))) for j in js]
            rate = [(k - 1) / (e - s) if e > s and k > 1 else 0.0 for s, e, k in spans]
            first, last = min(s for s, _, _ in spans), max(e for _, e, _ in spans)
            ids = sum(k - 1 for _, _, k in spans)
            both = min(e for _, e, _ in spans) - max(s for s, _, _ in spans)
            mean_span = statistics.mean(e - s for s, e, _ in spans)
            res[phase] = {"ttft": statistics.median(ttft), "ttft_max": max(ttft), "tps": statistics.median(rate),
                          "thr": ids / (last - first) if last > first else 0.0,
                          "overlap": max(0.0, both) / mean_span if mean_span > 0 else 0.0}
        if "warm" not in res:
            continue
        c, w = res["cold"], res["warm"]
        rows.append((n, c, w))
        print(f"SLOTS_BENCH {a.label} N={n} throughput {w['thr']:.1f} t/s, per request {w['tps']:.1f} t/s (overlap "
              f"{w['overlap']:.2f}); TTFT cold median {c['ttft']:.2f} s / max {c['ttft_max']:.2f} s, warm median "
              f"{w['ttft']:.2f} s", flush=True)
    print(f"| {a.label} | requests | decode throughput t/s | per request t/s | overlap | TTFT cold median s | "
          f"TTFT cold max s | TTFT warm median s |")
    for n, c, w in rows:
        print(f"| {a.label} | {n} | {w['thr']:.1f} | {w['tps']:.1f} | {w['overlap']:.2f} | {c['ttft']:.2f} | "
              f"{c['ttft_max']:.2f} | {w['ttft']:.2f} |")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for n, c, w in rows:
                f.write(json.dumps({"label": a.label, "n": n, "cold": c, "warm": w}) + "\n")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
