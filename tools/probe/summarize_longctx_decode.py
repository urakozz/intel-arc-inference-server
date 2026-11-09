#!/usr/bin/env python3
"""Validate complete 3c measurements before printing their anatomy tables."""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import median

DEPTHS = (4096, 32768, 81920, 150000)
KV = ('bf16', 'int8')
TIMES = ('wall_ms', 'gpu_ms', 'prep_ms', 'core_ms', 'reduce_ms', 'head_attn_ms')


def validate(rows, complete=True):
    seen = set()
    for x in rows:
        key = (x['kv'], x['round'], x['depth'], x['M'])
        if key in seen:
            raise ValueError(f'duplicate measurement {key}')
        seen.add(key)
        m = x['M']
        if m not in (1, 4) or x['kv'] not in KV:
            raise ValueError(f'invalid arm {key}')
        if x['launches'] != (774 if m == 1 else 784) or x['attn_launches'] != (48 if m == 1 else 49):
            raise ValueError(f'launch accounting {key}')
        if x['capacity'] <= x['depth'] + 128:
            raise ValueError(f'insufficient capacity {key}')
        if any(not math.isfinite(x[f]) or x[f] < 0 for f in TIMES):
            raise ValueError(f'invalid timing {key}')
        if any(x[f] <= 0 for f in ('wall_ms', 'gpu_ms', 'prep_ms', 'core_ms', 'reduce_ms')):
            raise ValueError(f'empty timing category {key}')
        attn = x['prep_ms'] + x['core_ms'] + x['reduce_ms']
        if not 0 < attn < x['gpu_ms']:
            raise ValueError(f'attention exceeds total {key}')
        if not 0 <= x['head_attn_ms'] <= x['prep_ms'] or (m == 1 and x['head_attn_ms'] != 0):
            raise ValueError(f'unexpected head attention {key}')
        if not math.isclose(x['rows_per_s'] * x['wall_ms'], m * 1000, rel_tol=1e-6):
            raise ValueError(f'throughput arithmetic {key}')
    expected = set(itertools.product(KV, range(3), DEPTHS, (1, 4)))
    if complete and seen != expected:
        raise ValueError(f'incomplete matrix: missing {sorted(expected - seen)}, extra {sorted(seen - expected)}')
    if complete and len({x['capacity'] for x in rows}) != 1:
        raise ValueError('matrix capacities differ')


def summarize(rows):
    table = []
    for kv, depth, m in itertools.product(KV, DEPTHS, (1, 4)):
        arm = [x for x in rows if (x['kv'], x['depth'], x['M']) == (kv, depth, m)]
        if not arm:
            continue
        attn = lambda x: x['prep_ms'] + x['core_ms'] + x['reduce_ms']
        item = dict(kv=kv, depth=depth, M=m, samples=len(arm))
        item.update({f: median(x[f] for x in arm) for f in TIMES})
        item.update(attention_ms=median(attn(x) for x in arm),
                    attention_share_pct=median(100 * attn(x) / x['gpu_ms'] for x in arm),
                    other_ms=median(x['gpu_ms'] - attn(x) for x in arm),
                    rows_per_s=median(x['rows_per_s'] for x in arm),
                    wall_range_ms=[min(x['wall_ms'] for x in arm), max(x['wall_ms'] for x in arm)])
        table.append(item)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--partial', action='store_true', help='validate a smoke subset without a verdict')
    args = parser.parse_args()
    rows = []
    for path in args.logs:
        text = path.read_text()
        if 'ALLDONE rc=0' not in text:
            raise ValueError(f'unfinished log {path}')
        rows.extend(json.loads(line) for line in text.splitlines() if line.startswith('{'))
    validate(rows, complete=not args.partial)
    if not rows:
        raise ValueError('no measurements')
    print(json.dumps(dict(complete=not args.partial, measurements=len(rows), table=summarize(rows)), indent=2))


if __name__ == '__main__':
    main()
