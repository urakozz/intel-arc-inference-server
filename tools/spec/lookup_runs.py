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

Spec 32e Task 0: --price walks each request with the server's policy (server::LookupInDFlash inside MTP, window 0,
max_k 3; EngineAdapter::step_drafts_or_confirmed) and prices every iteration from a cost table, for the ladders
  16 only;  earn (a wide verify kept whole asks for the next width, any other goes back to 15 drafts - the server's
  observe());  jump (a wide verify kept whole asks for the narrowest list that holds the source's remaining
  continuation, the ids after the source's end in the context; the same fall-back);  served (spec 32e, the shipped
  policy: earn up a width on a whole keep, down a width on a keep of at least half, back to 15 drafts below half, and
  a request whose last two wide verifies both missed stays at 15 drafts).
Earn and jump are walked with lists up to 32 and up to 64 rows. Assumed (no record of it): the MTP head drafts the
true ids (a confirmation is cand[:2] == the next two ids; an MTP iteration is K = 3, kept whole).
Costs: the table's ms a step by rows (1 = plain, 16 / 32 / 64) at its key counts, linear between and beyond them;
a K-draft lookup verify on the head's lists is plain x VERIFY[K]; an MTP iteration plain x (DRAFT[3] + VERIFY[3]);
a confirmed entry adds plain x DRAFT[2]; a wide verify adds the replay's ms for each committed row.
  lookup_runs.py --price [TABLE.json] DIR   (TABLE: {"keys": [...], "1": [...], "16": [...], "32": [...],
                                             "64": [...], "replay": ms}; default PRICE_TABLE)
"""
import json
import math
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

    def match(self, min_n, before=None):
        """(end, m) of the longest suffix match (most recent on ties) with m >= min_n, or None; `before`: only ends
        whose continuation starts before it (PromptLookup's source_before)."""
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
                if e >= t - 1 or (before is not None and e + 1 >= before) or s[e - k + 1:e + 1] != key:
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


# Plan 32e Task 0's table (derived): ms a step at the edit / copy set's 8k / 32k / 64k / 99k keys. 16 / 32 / 64 rows
# from the plan; 1 row (plain) interpolated from the measured 30.83 ms at 4,096, 34.92 at 32,768 and 49.78 at 150,000.
PRICE_TABLE = {'keys': [8192, 32768, 65536, 99000], '1': [31.4, 34.9, 39.1, 43.3],
               '16': [46, 59, 74, 92], '32': [70, 93, 123, 156], '64': [97, 144, 203, 270], 'replay': 0.086}
VERIFY = (1.000, 1.040, 1.070, 1.100)   # b70-serve's mtp auto table: a verify of M = 1..4 rows, plain steps
DRAFT = (0.0, 0.130, 0.260, 0.380)      # the head drafting k = 0..3, plain steps
LADDERS = (('16 only', 16, 'earn'), ('earn 32', 32, 'earn'), ('jump 32', 32, 'jump'),
           ('earn 64', 64, 'earn'), ('jump 64', 64, 'jump'), ('served 64', 64, 'served'))


def cost_at(table, rows, keys):
    xs, ys = table['keys'], table[str(rows)]
    j = max(1, min(len(xs) - 1, sum(1 for x in xs if x <= keys)))
    return ys[j - 1] + (ys[j] - ys[j - 1]) * (keys - xs[j - 1]) / (xs[j] - xs[j - 1])


def drafts(s, e, k):
    """k drafts continuing the source ending at e (an overlapping copy past the end of s, as PromptLookup)."""
    d = []
    for j in range(k):
        p = e + 1 + j
        d.append(s[p] if p < len(s) else d[p - len(s)])
    return d


def price(prompt, out, table, maxw, ladder):
    """(ms, emitted ids after the first, {rows: verifies}) of one request under the ladder, lists up to maxw rows."""
    lists = [w for w in (15, 31, 63) if w < maxw]
    ix = Index()
    for x in prompt + out[:1]:
        ix.append(x)
    i, ms, lookup, hot, width = 1, 0.0, False, False, 15
    misses, narrow = 0, False   # 'served': wide verifies in a row that missed; two of them keep the request at 15
    n = Counter()

    def emit(c):
        nonlocal i
        for x in out[i:i + c]:
            ix.append(x)
        i += c

    while i < len(out):
        t = len(ix.s)
        plain = cost_at(table, 1, t)
        before = t - 1   # window 0: a source before the pending id
        d, src = [], None
        if lookup:
            mt = ix.match(3, before)
            if mt is not None:
                src = mt[0]
                if hot:
                    w = width
                    if ladder == 'jump' and w > 15:   # the narrowest list holding the source's remaining ids
                        w = next((l for l in lists if l >= t - 1 - src), lists[-1])
                    d = drafts(ix.s, src, w)
                else:
                    d = drafts(ix.s, src, 3)
            else:
                lookup = hot = False
                width = 15
        else:
            mt = ix.match(8, before)
            if mt is not None:
                d, lookup = drafts(ix.s, mt[0], 3), True
        if not d:
            mt = ix.match(2, before)
            cand = drafts(ix.s, mt[0], 15) if mt is not None else []
            if cand and cand[:2] == out[i:i + 2]:   # the head's first two drafts agree: entered hot
                ms += plain * DRAFT[2]
                d, lookup, hot, width = cand, True, True, 15
                n['confirmed'] += 1
            else:   # an MTP iteration, K = 3 kept whole
                ms += plain * (DRAFT[3] + VERIFY[3])
                n['mtp'] += 1
                emit(4)
                continue
        r = 0
        while r < len(d) and i + r < len(out) and d[r] == out[i + r]:
            r += 1
        rows = len(d) + 1
        if len(d) > 3:
            ms += cost_at(table, rows, t) + table['replay'] * (r + 1)
        else:
            ms += plain * VERIFY[len(d)]
        n[rows] += 1
        emit(r + 1)
        wide = len(d) > 3   # observe(): proposed = len(d), kept = r
        hot = r >= 8 if wide else r == len(d)
        if ladder == 'served':   # server::LookupInDFlash::observe (spec 32e)
            misses = misses + 1 if wide and r < len(d) else 0 if wide else misses
            narrow = narrow or misses >= 2
            at = lists.index(len(d)) if wide else 0
            width = 15 if narrow or not wide else lists[min(at + 1, len(lists) - 1)] if r == len(d) \
                else lists[max(at - 1, 0)] if 2 * r >= len(d) else 15
        else:
            width = (lists[min(lists.index(len(d)) + 1, len(lists) - 1)] if ladder == 'earn' else lists[-1]) \
                if wide and r == len(d) else 15
        if r < min(3, len(d)):
            lookup = hot = False
            width = 15
    return ms, len(out) - 1, n


def price_report(table, d):
    reqs = list(load(d))
    rows = []
    for fn, p, o in reqs:
        tps = []
        for _, maxw, ladder in LADDERS:
            ms, ids, n = price(p, o, table, maxw, ladder)
            tps.append((1000 * ids / ms, n))
        rows.append((fn, len(p), tps))
    print('| request | keys | ' + ' | '.join(name for name, _, _ in LADDERS) + ' |')
    print('|---|---:|' + '---:|' * len(LADDERS))
    for fn, k, tps in rows:
        print(f'| {fn} | {k} | ' + ' | '.join(f'{v:.1f}' for v, _ in tps) + ' |')
    for c, (name, _, _) in enumerate(LADDERS):
        mean = sum(t[c][0] for _, _, t in rows) / len(rows)
        geo = math.exp(sum(math.log(t[c][0] / t[0][0]) for _, _, t in rows) / len(rows))
        print(f'{name}: mean {mean:.1f} t/s, geomean {geo:.3f}x 16 only')
    for fn, _, tps in rows:
        print(f'   verifies {fn}: ' + '; '.join(
            f'{name} ' + ' '.join(f'{k}:{v}' for k, v in sorted(n.items(), key=str))
            for (name, _, _), (_, n) in zip(LADDERS, tps)))


def main():
    if sys.argv[1:2] == ['--price']:
        args = sys.argv[2:]
        table = PRICE_TABLE
        if args[0].endswith('.json'):
            table = json.loads(Path(args.pop(0)).read_text())
        for d in args:
            price_report(table, d)
        return
    for d in sys.argv[1:]:
        reqs = list(load(d))
        for kind in ('policy', 'oracle'):
            report(d, reqs, kind)


if __name__ == '__main__':
    main()
