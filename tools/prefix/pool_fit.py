#!/usr/bin/env python3
"""Recorded sessions against fixed slots and a KV page pool (spec 31 Task 1, measurement (a)).

    pool_fit.py <log dir> [<log dir> ...] [--seqs 4] [--slot 32768]
                [--slot0-bf16 46080] [--slot0-int8 167936] [--pages-bf16 70] [--pages-int8 124]

Each <log dir> is what `b70-serve --log-requests DIR` writes (see analyze_log.py): NNNNNN.json per request with
prompt_ids, out_ids, t_start, t_first_token and t_end. The requests are replayed by start time against two models
of one card, in bf16 and in int8 KV:

- fixed: slot 0 of --slot0-* positions and --seqs - 1 slots of --slot. A request goes to the free slot long
  enough for its prompt + 1 that holds its longest matching prefix, then the one holding least; none free and
  long enough: it waits; longer than every slot: refused; output past the slot's end: cut.
- pool: --seqs sequences sharing --pages-* pages of 2048 positions; a request may grow to the whole pool. Admission
  reserves the prompt, one page of output and the verify rows; short of pages it evicts idle sessions least
  recently used first, else waits; a running request with no page left is cut ("length").

A request keeps its recorded duration (shifted by any wait); its output ids arrive evenly between its first token
and its end, and it takes a page when its last position plus the verify rows crosses a page boundary.

"Shared" is the longest prefix a request's prompt has in common with a session resident on another sequence at
its start (running or idle, in that pool model), rounded down to whole pages: an upper bound of what read-only
shared pages could hold.
"""
import argparse
import heapq
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_log import lcp, load  # noqa: E402

PAGE = 2048
VERIFY_ROWS = 8          # the batched verify's rows past the committed length
TRAINED_CONTEXT = 262144


class Req:
    def __init__(self, n, r):
        self.n = n
        self.prompt = r["prompt_ids"]
        self.out = r.get("out_ids", [])
        self.t0 = r["t_start"]
        self.t_first = r.get("t_first_token", self.t0)
        self.t1 = r["t_end"]


class Seq:
    def __init__(self, length):
        self.length = length     # fixed: the slot's positions; pool: unused
        self.job = None          # the running Job
        self.held = []           # the ids whose KV the sequence holds
        self.pages = 0           # pool: pages it holds
        self.used = -1.0         # when it was last given back


class Job:
    def __init__(self, req, start):
        self.req = req
        self.shift = start - req.t0
        self.produced = len(req.out)
        self.cut = False
        self.shared_pages = 0
        self.version = 0


def outcome_label(o):
    return ("cut" if o["cut"] else "refused" if o["refused"] else
            f"waited {o['wait']:.1f} s" if o["wait"] > 0 else "ok")


def simulate(reqs, model, kv, a):
    """Replays reqs against one card; returns per-request outcomes and the run's peaks."""
    pool = model == "pool"
    pages_total = a.pages_bf16 if kv == "bf16" else a.pages_int8
    cap = min(pages_total * PAGE, TRAINED_CONTEXT) if pool else None
    slot0 = a.slot0_bf16 if kv == "bf16" else a.slot0_int8
    seqs = [Seq(slot0 if i == 0 else a.slot) for i in range(a.seqs)]
    free = pages_total
    out = {r.n: {"cut": False, "refused": False, "wait": 0.0, "shared": 0, "evictions": 0} for r in reqs}
    stats = {"evictions": 0, "peak_running": 0, "peak_all": 0, "peak_shared": 0}
    events = []
    order = [0]

    def push(t, kind, data):
        order[0] += 1
        heapq.heappush(events, (t, order[0], kind, data))

    for r in reqs:
        push(r.t0, "arrive", r)
    waiting = []

    def evict_for(need, keep, now):
        """Frees idle sequences' pages, least recently used first, until `need` are free; False if impossible."""
        nonlocal free
        idle = sorted((s for s in seqs if s.job is None and s is not keep and s.pages > 0), key=lambda s: s.used)
        if free + sum(s.pages for s in idle) < need:
            return False
        for s in idle:
            if free >= need:
                break
            free += s.pages
            s.pages, s.held = 0, []
            stats["evictions"] += 1
        return True

    def gen_time(job, o):
        """When the job's o-th output id (1-based) is produced."""
        r = job.req
        span = r.t1 - r.t_first
        return r.t_first + job.shift + (span * o / len(r.out) if r.out else 0.0)

    def schedule_growth(seq, now):
        job = seq.job
        r = job.req
        o = seq.pages * PAGE - len(r.prompt) - VERIFY_ROWS + 1   # first output id whose rows pass the pages
        if seq.pages * PAGE >= cap or o > job.produced:
            return
        push(max(gen_time(job, max(o, 0)), now), "grow", (seq, job, job.version))

    def admit(r, now):
        nonlocal free
        need = len(r.prompt) + 1
        cands = [s for s in seqs if s.job is None and (pool or s.length >= need)]
        if not cands:
            return False
        best = max(cands, key=lambda s: (min(lcp(r.prompt, s.held), len(r.prompt) - 1),
                                         -(s.pages if pool else len(s.held))))
        job = Job(r, now)
        if pool:
            reserve = min(math.ceil((len(r.prompt) + PAGE + VERIFY_ROWS) / PAGE), cap // PAGE)
            extra = reserve - best.pages   # the sequence's own pages count; past the match they are rewritten
            if free < extra and not evict_for(extra, best, now):
                return False
            # Shared: the longest prefix resident on another sequence, whole pages only.
            shared = 0
            for s in seqs:
                if s is best:
                    continue
                ids = s.job.req.prompt if s.job else s.held if s.pages > 0 else []
                shared = max(shared, lcp(r.prompt, ids))
            job.shared_pages = min(shared, len(r.prompt) - 1) // PAGE
            out[r.n]["shared"] = job.shared_pages * PAGE
            free -= extra
            best.pages = reserve
            room = cap - len(r.prompt)
        else:
            room = best.length - len(r.prompt)
        if job.produced > room:
            job.produced, job.cut = room, True
        best.job = job
        best.held = []
        push(r.t1 + job.shift, "end", (best, job, job.version))
        if pool:
            schedule_growth(best, now)
        out[r.n]["wait"] = now - r.t0
        return True

    def finish(seq, job, now):
        r = job.req
        seq.held = r.prompt + r.out[:job.produced]
        seq.job = None
        seq.used = now
        out[r.n]["cut"] = job.cut
        if pool:
            nonlocal free
            keep = math.ceil(len(seq.held) / PAGE)
            free += seq.pages - keep
            seq.pages = keep

    while events:
        now, _, kind, data = heapq.heappop(events)
        if kind == "arrive":
            r = data
            longest = cap if pool else max(s.length for s in seqs)
            if len(r.prompt) + 1 > longest:
                out[r.n]["refused"] = True
            elif not admit(r, now):
                waiting.append(r)
        elif kind == "grow":
            seq, job, version = data
            if seq.job is not job or version != job.version:
                continue
            if free < 1 and not evict_for(1, seq, now):
                # The pool is out of pages: the request ends here with "length".
                o = seq.pages * PAGE - len(job.req.prompt) - VERIFY_ROWS
                job.produced, job.cut = max(min(o, job.produced), 0), True
                job.version += 1
                finish(seq, job, now)
            else:
                free -= 1
                seq.pages += 1
                schedule_growth(seq, now)
        else:
            seq, job, version = data
            if seq.job is not job or version != job.version:
                continue
            finish(seq, job, now)
        if kind != "arrive" or waiting:
            still = []
            for r in waiting:
                if not admit(r, now):
                    still.append(r)
            waiting = still
        if pool:
            running = sum(s.pages for s in seqs if s.job is not None)
            stats["peak_running"] = max(stats["peak_running"], running)
            stats["peak_all"] = max(stats["peak_all"], pages_total - free)
            stats["peak_shared"] = max(stats["peak_shared"],
                                       sum(s.job.shared_pages for s in seqs if s.job is not None))
    for r in waiting:   # never admitted: the log ended with it waiting
        out[r.n]["refused"] = True
    return out, stats


def overlaps(reqs):
    """The most requests running at once in the recording."""
    edges = sorted([(r.t0, 1) for r in reqs] + [(r.t1, -1) for r in reqs], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in edges:
        cur += d
        peak = max(peak, cur)
    return peak


MODELS = [("fixed", "bf16"), ("pool", "bf16"), ("fixed", "int8"), ("pool", "int8")]


def analyze(records, a):
    reqs = [Req(i + 1, r) for i, r in enumerate(records) if "t_start" in r and r.get("prompt_ids")]
    runs = {m: simulate(reqs, m[0], m[1], a) for m in MODELS}
    return reqs, runs


def verdict(results):
    """Rule (a) over every log: results is a list of (reqs, runs)."""
    def failed(runs, model, kv, r):
        o = runs[(model, kv)][0][r.n]
        return o["cut"] or o["refused"] or o["wait"] > 0
    int8_fail = bf16_fail = int8_pool_gain = False
    shared = 0
    for reqs, runs in results:
        for r in reqs:
            if failed(runs, "fixed", "int8", r):
                int8_fail = True
                if not failed(runs, "pool", "int8", r):
                    int8_pool_gain = True
            if failed(runs, "fixed", "bf16", r):
                bf16_fail = True
        shared = max(shared, runs[("pool", "bf16")][1]["peak_shared"], runs[("pool", "int8")][1]["peak_shared"])
    if int8_pool_gain:
        v = "go: the int8 fixed slots cut, refuse or delay a request the pool serves"
    elif not int8_fail and not bf16_fail and shared < 8:
        v = "stop: the fixed slots serve every request unchanged in bf16 and int8"
    elif not int8_fail and bf16_fail:
        v = "decide: only the bf16 fixed slots fail a request (int8 KV serves them all)"
    else:
        v = "no pool gain: the int8 fixed slots fail requests the pool fails too"
    if shared >= 8:
        v += f"; shared pages: go ({shared} whole pages at the peak)"
    else:
        v += f"; shared pages: not worth it ({shared} whole pages at the peak)"
    return v


def report(log_dir, reqs, runs, a):
    print(f"## {log_dir}\n")
    if not reqs:
        print("no requests with ids and times\n")
        return
    big = max(reqs, key=lambda r: len(r.prompt) + len(r.out))
    conc = overlaps(reqs)
    print(f"{len(reqs)} requests; largest: request {big.n}, {len(big.prompt)} prompt + {len(big.out)} out = "
          f"{len(big.prompt) + len(big.out)} positions; "
          + ("requests overlap in time (up to " + str(conc) + " at once)" if conc > 1
             else "no two requests overlap in time (recorded one at a time)") + "\n")
    print("| model | kv | positions | ok | cut | refused | waited | evictions | peak pages running | "
          "peak pages with idle | peak shared pages |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        o, st = runs[m]
        model, kv = m
        if model == "pool":
            pages = a.pages_bf16 if kv == "bf16" else a.pages_int8
            size = f"{pages} x {PAGE} = {pages * PAGE}"
        else:
            size = f"{a.slot0_bf16 if kv == 'bf16' else a.slot0_int8} + {a.seqs - 1} x {a.slot}"
        labels = [outcome_label(x) for x in o.values()]
        pool = model == "pool"
        print(f"| {model} | {kv} | {size} | {labels.count('ok')} | {labels.count('cut')} | "
              f"{labels.count('refused')} | {sum(lb.startswith('waited') for lb in labels)} | "
              f"{st['evictions'] if pool else '-'} | {st['peak_running'] if pool else '-'} | "
              f"{st['peak_all'] if pool else '-'} | {st['peak_shared'] if pool else '-'} |")
    print()
    print("| req | prompt | out | fixed bf16 | pool bf16 | fixed int8 | pool int8 | shared bf16 | shared int8 |")
    print("|---|---:|---:|---|---|---|---|---:|---:|")
    for r in reqs:
        cells = [outcome_label(runs[m][0][r.n]) for m in MODELS]
        print(f"| {r.n} | {len(r.prompt)} | {len(r.out)} | " + " | ".join(cells) +
              f" | {runs[('pool', 'bf16')][0][r.n]['shared']} | {runs[('pool', 'int8')][0][r.n]['shared']} |")
    print()


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--slot", type=int, default=32768)
    ap.add_argument("--slot0-bf16", type=int, default=46080)
    ap.add_argument("--slot0-int8", type=int, default=167936)
    ap.add_argument("--pages-bf16", type=int, default=70)
    ap.add_argument("--pages-int8", type=int, default=124)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    results = []
    for d in a.logs:
        reqs, runs = analyze(load(d), a)
        report(d, reqs, runs, a)
        results.append((reqs, runs))
    print("verdict: " + verdict(results))


if __name__ == "__main__":
    main()
