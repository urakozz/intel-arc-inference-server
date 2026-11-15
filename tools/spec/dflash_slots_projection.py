#!/usr/bin/env python3
"""Plan 19v3b Task 1: is DFlash with several slots a card worth building? Host arithmetic on saved records, no GPU.

    dflash_slots_projection.py <acceptance.json> <task8 dir>
        (on the box: ~/b70/logs/dflash-v2/baseline/acceptance.json  ~/b70/logs/dflash-v2/task8/main-0b5ef628)

Tokens a step, per corpus:
  DFlash  acceptance.json's engine_E at K (ids a round: the accepted drafts and the target's own id; spec 19 v2
          Task 1, the engine's taps, one card) - K = 3 for two requests active (8 rows / 2), K = 1 for four.
  MTP     the served MTP arm of spec 19 v2 Task 8 (one card, `--spec mtp`, K adaptive per step <= 3): its server prints
          per request "mtp: ... A/D drafts accepted ... iterations at K 0/1/2/3: n0/n1/n2/n3", which does not split the
          accepted drafts by K. So per corpus one per-position acceptance p is fitted (a geometric chain: a K-draft
          iteration accepts p + .. + p^K on average) to sum(A) = sum over requests of n1 f1(p) + n2 f2(p) + n3 f3(p),
          and MTP's ids a step at K is 1 + p + .. + p^K - derived, not measured per K (the head's first draft is
          accepted more often than its later ones, so K = 1 is understated and K = 3 overstated a little).
          Requests are matched to their prompts (requests.jsonl) by prompt and completion lengths, rounds r1..r3;
          the opencode corpus is every request of the opencode-session* arms.
Costs a step in plain steps (spec 19 v3 §5.4, plan 19v3b Task 1 Step 2):
  ROW[n]   the row step of 8 rows from n slots: 1.53 at four (probe-batching, measured); at two 1.40 - measured on
           the paged-KV branch's probe_slots_speed "2 K=3,3" arm at 4k (int8 1.38, bf16 1.42; plan: "1.4 estimated")
  DRAFT    the batched DFlash draft pass: 0.217 (plan 19v3d's re-measured one-card pass; the plan's 0.29 before it)
  MTP      0.15 a draft index (the row draft; 0.136 measured at 4k, same arm) + 0.05 the head's KV fill
DFlash / MTP = (DFlash ids a step / DFlash cost) / (MTP ids a step / MTP cost). Gate (plan): geometric mean over the
corpora >= 0.95 at two requests and >= 0.90 at four; below - stop and ask the operator.
"""
import glob
import json
import math
import os
import re
import sys

ROW = {2: 1.40, 4: 1.53}
DRAFT = 0.217
MTP_DRAFT, MTP_HEAD_KV = 0.15, 0.05
KS = {2: 3, 4: 1}
GATE = {2: 0.95, 4: 0.90}
CORPORA = ["a4", "cjk", "code", "opencode", "prose"]

LINE_REQ = re.compile(r"^request: prompt (\d+) ids .*?; (\d+) ids generated")
LINE_MTP = re.compile(r"^mtp: K \d+, \d+ iterations, (\d+)/(\d+) drafts accepted .*K 0/1/2/3: (\d+)/(\d+)/(\d+)/(\d+)")


def corpus_of(rec):
    if rec["set"] == "toolcall":
        return "a4"
    if rec["set"] == "golden":
        return rec["name"]
    return None


def mtp_requests(log):
    """(prompt ids, generated ids, accepted, drafted, n1, n2, n3) per request of one server log."""
    out, pending = [], None
    for line in open(log, errors="replace"):
        m = LINE_REQ.match(line)
        if m:
            pending = (int(m.group(1)), int(m.group(2)))
            continue
        m = LINE_MTP.match(line)
        if m and pending:
            a, d, _, n1, n2, n3 = map(int, m.groups())
            out.append(pending + (a, d, n1, n2, n3))
            pending = None
    return out


def collect(task8):
    """corpus -> [(accepted, n1, n2, n3)]"""
    by = {c: [] for c in CORPORA}
    for arm in sorted(glob.glob(os.path.join(task8, "corpora", "mtp.r*"))):
        recs = [json.loads(l) for l in open(os.path.join(arm, "requests.jsonl"))]
        reqs = mtp_requests(os.path.join(arm, "server.log"))
        i = 0
        for rec in recs:   # in order; the log's extra requests (warm-up) are skipped
            while i < len(reqs) and (reqs[i][0] != rec["prompt_tokens"] or abs(reqs[i][1] - rec["completion_tokens"]) > 1):
                i += 1
            if i == len(reqs):
                sys.exit(f"{arm}: no server line for {rec['set']}/{rec['name']}")
            c = corpus_of(rec)
            if c:
                by[c].append((reqs[i][2],) + reqs[i][4:])
            i += 1
    for log in sorted(glob.glob(os.path.join(task8, "opencode-session*", "mtp.r*", "server.log"))):
        by["opencode"] += [(r[2],) + r[4:] for r in mtp_requests(log)]
    return by


def fit_p(rows):
    f = lambda p, k: sum(p ** i for i in range(1, k + 1))
    acc = sum(r[0] for r in rows)
    lo, hi = 0.0, 1.0
    for _ in range(60):   # sum_k n_k f_k(p) rises with p: bisect
        p = (lo + hi) / 2
        if sum(r[1] * f(p, 1) + r[2] * f(p, 2) + r[3] * f(p, 3) for r in rows) < acc:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def main():
    acc_path, task8 = sys.argv[1], sys.argv[2]
    dflash = json.load(open(acc_path))["corpora"]
    mtp = collect(task8)
    print("Plan 19v3b Task 1 - DFlash / MTP with several requests a card (derived; host arithmetic)")
    print(f"costs in plain steps: row step of 8 rows {ROW[2]:.2f} (2 slots) / {ROW[4]:.2f} (4 slots), DFlash batched "
          f"draft {DRAFT}, MTP {MTP_DRAFT} a draft index + {MTP_HEAD_KV} head KV")
    print("| corpus | MTP p (requests, drafts) | DFlash ids/round K=3 / K=1 | MTP ids/step K=3 / K=1 | "
          "DFlash / MTP, 2 active | DFlash / MTP, 4 active |")
    print("|---|---|---|---|---:|---:|")
    ratios = {2: [], 4: []}
    for c in CORPORA:
        rows = mtp[c]
        p = fit_p(rows)
        drafted = sum(r[1] + 2 * r[2] + 3 * r[3] for r in rows)
        e_m = {k: 1 + sum(p ** i for i in range(1, k + 1)) for k in (1, 3)}
        e_d = {k: dflash[c][str(k)]["engine_E"] for k in (1, 3)}
        r = {}
        for n, k in KS.items():
            d_rate = n * e_d[k] / (ROW[n] + DRAFT)
            m_rate = n * e_m[k] / (ROW[n] + k * MTP_DRAFT + MTP_HEAD_KV)
            r[n] = d_rate / m_rate
            ratios[n].append(r[n])
        print(f"| {c} | {p:.3f} ({len(rows)}, {drafted}) | {e_d[3]:.3f} / {e_d[1]:.3f} "
              f"(anchors {dflash[c]['1']['anchors']}) | {e_m[3]:.3f} / {e_m[1]:.3f} | {r[2]:.3f} | {r[4]:.3f} |")
    gm = {n: math.exp(sum(map(math.log, v)) / len(v)) for n, v in ratios.items()}
    print(f"| **geometric mean** | | | | **{gm[2]:.3f}** | **{gm[4]:.3f}** |")
    go = all(gm[n] >= GATE[n] for n in GATE)
    print(f"gate: >= {GATE[2]} at two, >= {GATE[4]} at four -> {'GO' if go else 'STOP'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
