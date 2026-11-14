#!/usr/bin/env python3
"""Open-work 28 step 1: how the output ids of recorded requests split into exact-copy runs against their context.

For each request (prompt_ids, out_ids) the context at output position i is prompt + out[:i]. Two views:
  policy: the server's lookup policy (PromptLookup: longest suffix match, most recent on ties, levels 2..64,
          64 candidates a level; LookupInDFlash with no window: ENTER on a match >= 8, STAY on >= 3 while each
          verify keeps >= 3). A run = the ids of out[i:] equal to the chosen source's continuation (unbounded).
          After a run of r the verify's own id is out[i+r] (i += r + 1); with no proposal i += 1.
  oracle: the longest prefix of out[i:] that occurs anywhere in the context (key >= 8 ids), greedy.
Prints the run-length histogram, the share of output ids in runs >= 8 / >= 16, and how many verifies a
lookup-covered run takes at K = 3 (M = 4), 7 (M = 8) and 15 (M = 16).
  lookup_runs.py DIR [DIR...]   (b70-serve --log-requests directories or recordings: NNNNNN.json with prompt_ids, out_ids)
"""
import json
import sys
from collections import Counter
from pathlib import Path

LEVELS = (64, 32, 16, 8, 4, 2)
MAXC = 64


class Index:
    def __init__(self):
        self.s = []
        self.tab = {k: {} for k in LEVELS}

    def append(self, x):
        self.s.append(x)
        p = len(self.s) - 1
        for k in LEVELS:
            if p + 1 >= k:
                self.tab[k].setdefault(hash(tuple(self.s[p - k + 1:p + 1])), []).append(p)

    def match(self, min_n):
        """(end, m) of the longest suffix match (most recent on ties) with m >= min_n, or None."""
        s, t = self.s, len(self.s)
        for k in LEVELS:
            if k > 64 or t < k + 1:
                continue
            if k < min_n and k != max(l for l in LEVELS if l <= min_n):
                break
            key = s[t - k:t]
            chain = self.tab[k].get(hash(tuple(key)))
            if not chain:
                continue
            best = None
            seen = 0
            for e in reversed(chain):
                if e >= t - 1 or s[e - k + 1:e + 1] != key:
                    continue
                m = k
                while m < 64 and e - m >= 0 and s[e - m] == s[t - 1 - m]:
                    m += 1
                if best is None or m > best[1]:
                    best = (e, m)
                seen += 1
                if m >= 64 or seen >= MAXC:
                    break
            if best is not None:
                return best if best[1] >= min_n else None
        return None


def policy_runs(prompt, out):
    ix = Index()
    for x in prompt:
        ix.append(x)
    full = prompt + out
    runs, i, inl, n = [], 0, False, len(prompt)
    covered = 0
    while i < len(out):
        mt = ix.match(3 if inl else 8)
        if mt is None:
            inl = False
            ix.append(out[i])
            i += 1
            continue
        e = mt[0]
        r = 0
        while i + r < len(out) and full[e + 1 + r] == out[i + r]:
            r += 1
        runs.append(r)
        covered += r
        inl = r >= 3
        step = min(r + 1, len(out) - i)
        for x in out[i:i + step]:
            ix.append(x)
        i += step
    return runs


def oracle_runs(prompt, out):
    ix = {}
    full = prompt + out
    K = 8

    def add(p):
        if p + 1 >= K:
            ix.setdefault(tuple(full[p - K + 1:p + 1]), []).append(p - K + 1)

    for p in range(len(prompt)):
        add(p)
    runs, i, n = [], 0, len(prompt)
    while i < len(out):
        best = 0
        if i + K <= len(out):
            for st in reversed(ix.get(tuple(out[i:i + K]), [])[-256:]):
                r = 0
                while i + r < len(out) and st + r < n + i and full[st + r] == out[i + r]:
                    r += 1
                best = max(best, r)
        step = best if best >= K else 1
        if best >= K:
            runs.append(best)
        for p in range(n + i, n + i + step):
            add(p)
        i += step
    return runs


def load(d):
    for f in sorted(Path(d).glob('*.json')):
        j = json.loads(f.read_text())
        if j.get('out_ids') and j.get('prompt_ids'):
            yield f.name, j['prompt_ids'], j['out_ids']


def ceil(a, b):
    return -(-a // b)


def report(name, reqs, kind):
    total = 0
    runs = []
    per_req = []
    for fn, p, o in reqs:
        r = policy_runs(p, o) if kind == 'policy' else oracle_runs(p, o)
        total += len(o)
        runs += r
        per_req.append((fn, len(o), sum(x for x in r if x >= 8)))
    bins = [(0, 0), (1, 2), (3, 7), (8, 15), (16, 31), (32, 63), (64, 10**9)]
    h = Counter()
    hid = Counter()
    for x in runs:
        for lo, hi in bins:
            if lo <= x <= hi:
                h[(lo, hi)] += 1
                hid[(lo, hi)] += x
    cov = sum(runs)
    ge8 = sum(x for x in runs if x >= 8)
    ge16 = sum(x for x in runs if x >= 16)
    print(f'== {name} [{kind}]: {len(reqs)} requests, {total} output ids; ids in lookup runs {cov} '
          f'({100 * cov / max(total, 1):.1f} %); in runs >= 8: {ge8} ({100 * ge8 / max(total, 1):.1f} %); '
          f'>= 16: {ge16} ({100 * ge16 / max(total, 1):.1f} %)')
    for b in bins:
        lab = f'{b[0]}-{b[1]}' if b[1] < 10**9 else f'{b[0]}+'
        print(f'   run {lab:>7}: {h[b]:6d} runs, {hid[b]:7d} ids ({100 * hid[b] / max(total, 1):5.1f} %)')
    if kind == 'policy':
        for k in (3, 7, 15):
            v = sum(ceil(x, k) if x else 1 for x in runs)
            print(f'   verifies over the lookup runs at K = {k:2d} (M = {k + 1:2d}): {v}')
    top = sorted(per_req, key=lambda t: -t[2])[:5]
    print('   most copy (file, out, ids in runs >= 8):', top)


def main():
    for d in sys.argv[1:]:
        reqs = list(load(d))
        for kind in ('policy', 'oracle'):
            report(d, reqs, kind)


if __name__ == '__main__':
    main()
