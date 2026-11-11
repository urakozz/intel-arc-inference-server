#!/usr/bin/env python3
"""Spec 19 v2 G-accept: GPU and CPU drafts on the engine's taps and the same anchors.

  dflash_engine_accept.py --engine DIR --drafter SNAPSHOT --target SNAPSHOT [--out FILE] [--threads 16]

The captured lists allocate eight rows, but noncausal attention masks the block at K + 1. Draft each K separately
on both sides. A recorded continuation which disagrees with the target's greedy id censors later rows, as in
dflash_accept.accept_rows. Actual in-loop acceptance is reported separately.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse
import glob
import importlib.util
import json
from collections import defaultdict
from types import SimpleNamespace


def prefix_counts(ids, drafts, greedy, p, tap_from):
    counted, accepted = [0] * len(drafts), [0] * len(drafts)
    for j, token in enumerate(drafts):
        if p + j >= len(ids) or (j and ids[p + j] != greedy[p + j - 1 - tap_from]):
            break
        counted[j] = 1
        if token != greedy[p + j - tap_from]:
            break
        accepted[j] = 1
    return counted, accepted


def expected(counted, accepted):
    if not counted or not counted[0]:
        raise ValueError('no observed anchors')
    total, product = 1.0, 1.0
    for n, a in zip(counted, accepted):
        product *= a / n if n else 0.0
        total += product
    return total


def evaluate(engine_dir, drafter_dir, target_dir, out_path, threads):
    spec = importlib.util.spec_from_file_location('dflash_d1', os.path.join(_HERE, 'dflash_d1.py'))
    d1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d1)
    import torch
    from safetensors import safe_open
    if threads:
        torch.set_num_threads(threads)
    with open(os.path.join(engine_dir, 'engine.jsonl')) as f:
        lines = [json.loads(s) for s in f if s.strip()]
    run = lines[0]['run']
    drafter = d1.reference_drafter(run, drafter_dir, target_dir)
    acc = d1._load('dflash_accept')
    head = SimpleNamespace(logits=drafter.head_logits)
    records = defaultdict(list)
    for rec in lines[1:]:
        records[rec['dump']].append(rec)
    if not records:
        raise ValueError('no engine drafts')
    paths = {}
    for path in glob.glob(os.path.join(engine_dir, '*.taps.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as f:
            paths[f.metadata()['name']] = path
    totals = {}
    detail = []
    with torch.inference_mode():
        for name, recs in records.items():
            with safe_open(paths[name], framework='pt', device='cpu') as f:
                meta = f.metadata()
                dump = {'tap_from': int(meta['tap_from']), 'n': int(meta['n']),
                        'ids': f.get_tensor('ids').long(), 'greedy': f.get_tensor('greedy').long(),
                        'taps': {int(i): f.get_tensor('tap.L' + i) for i in meta['layers'].split(',')}}
            ctx = drafter.context_kv(dump['taps'], torch.arange(dump['tap_from'], dump['n']))
            ids, greedy = dump['ids'].tolist(), dump['greedy'].tolist()
            by_anchor = defaultdict(dict)
            for rec in recs:
                if rec['k'] in by_anchor[rec['p']]:
                    raise ValueError(f'{name}: duplicate anchor/K')
                by_anchor[rec['p']][rec['k']] = rec
            for p, rows in by_anchor.items():
                if set(rows) != set(range(1, 8)):
                    raise ValueError(f'{name} p={p}: need K=1..7 on every anchor')
            ps = list(by_anchor)
            references = {}
            for k in range(1, 8):
                references[k] = acc.draft_batch(drafter, head, [ids[p] for p in ps], ps, k, ctx,
                                                {'full': None})['full'].tolist()
                if not detail and references[k][0] != drafter.draft_block(ids[ps[0]], ps[0], k, ctx).ids.tolist():
                    raise ValueError(f'{name} K={k}: batched reference differs from the single-anchor calibration')
            if not detail:
                print('REFERENCE_BATCH_CALIBRATION K=1..7 ids match the single-anchor reference', flush=True)
            for index, (p, rows) in enumerate(by_anchor.items()):
                for k, rec in sorted(rows.items()):
                    ref = references[k][index]
                    corpus = name.split('/')[0]
                    key = (corpus, k)
                    if key not in totals:
                        totals[key] = {'engine_counted': [0]*k, 'engine_accepted': [0]*k,
                                       'reference_counted': [0]*k, 'reference_accepted': [0]*k,
                                       'anchors': 0, 'verified_ids': 0}
                    total = totals[key]
                    for arm, drafts in [('engine', rec['ids']), ('reference', ref)]:
                        c, a = prefix_counts(ids, drafts, greedy, p, dump['tap_from'])
                        for j in range(k):
                            total[arm + '_counted'][j] += c[j]
                            total[arm + '_accepted'][j] += a[j]
                    total['anchors'] += 1
                    total['verified_ids'] += 1 + rec['accepted']
                    detail.append({'name': name, 'p': p, 'k': k, 'reference': ref, 'engine': rec['ids']})
            print(f"ENGINE_ACCEPT {name}: {len(by_anchor)} anchors batched, K=1..7 (matching active widths)", flush=True)
    report = {'run': run, 'corpora': {}, 'drafts': detail}
    passed = True
    for (corpus, k), row in sorted(totals.items()):
        row['engine_E'] = expected(row['engine_counted'], row['engine_accepted'])
        row['reference_E'] = expected(row['reference_counted'], row['reference_accepted'])
        row['gap'] = abs(row['engine_E'] - row['reference_E'])
        row['verified_ids_per_anchor'] = row['verified_ids'] / row['anchors']
        report['corpora'].setdefault(corpus, {})[str(k)] = row
        # G-accept is the A4/code gate. Other corpora remain recorded for G-drafter.
        gated = corpus in ('a4', 'code') and k == 7
        ok = row['gap'] <= 0.15
        passed &= not gated or ok
        print(f"G_ACCEPT corpus={corpus} K={k} engine_E={row['engine_E']:.5f} reference_E={row['reference_E']:.5f} "
              f"gap={row['gap']:.5f} verified_ids_per_anchor={row['verified_ids_per_anchor']:.5f} "
              f"{'PASS' if ok else 'FAIL'}{' (gate)' if gated else ' (recorded)'}", flush=True)
    if not {'a4', 'code'} <= set(report['corpora']):
        passed = False
        print('G_ACCEPT FAIL: A4 and code corpora are both required', flush=True)
    report['pass'] = passed
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--drafter', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--out')
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    ok = evaluate(a.engine, a.drafter, a.target, a.out or os.path.join(a.engine, 'acceptance.json'), a.threads)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
