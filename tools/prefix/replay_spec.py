#!/usr/bin/env python3
"""Task 7b: three rotated speculation arms with the prefix cache on, using one idle GPU.

  python3 tools/prefix/replay_spec.py RECORDING OUT [--model SNAPSHOT] [--draft-model SNAPSHOT]

Takes the whole-box GPU lock, or only --card N, and forces --parallel 1. The lock helper sets device affinity.
For a card run beside another GPU job, first confirm more than 16 CPU threads are available. All arms use bf16 KV,
int8 target head, context 131072, prefix budget 32 GiB and greedy, recorded-length requests. MTP must print
'adaptive per step'. Writes serve_client-compatible requests.jsonl per arm/round, table.txt and the f32 judge's
judge.log/forced contexts. Missing f32 references return 77, never a pass. Final G-speed also needs Task 8's
other corpora and prefill checks. Existing output is refused so stale request records cannot enter the result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import replay_log

ROOT = Path(__file__).resolve().parents[2]
ARMS = ['off', 'mtp', 'dflash']


def records(report, server_log, arm, round_, corpus):
    replies = json.loads(report.read_text())
    logged = [r for _, r in replay_log.load_log(server_log)]
    by_user = {r['request'].get('user'): r for r in logged}
    if len(by_user) != len(logged) or set(by_user) != {r['user'] for r in replies}:
        raise RuntimeError('server records do not match the replay request tags')
    out = []
    for reply in replies:
        if reply['error'] or not reply['usage']:
            raise RuntimeError(f"{reply['name']}: failed request or missing usage")
        row = by_user[reply['user']]
        n = len(row['out_ids'])
        gen = row['t_end'] - row['t_first_token']
        if not n or (n > 1 and gen <= 0):
            raise RuntimeError(f"{reply['name']}: no valid decode span")
        out.append({'label': arm, 'round': round_, 'set': corpus, 'name': reply['name'],
                    'prompt_ids': row['prompt_ids'], 'out_ids': row['out_ids'],
                    'tokens_in': len(row['prompt_ids']), 'tokens_out': n, 'usage': reply['usage'],
                    'cached_tokens': reply['cached_tokens'], 'text': row.get('response_text', ''),
                    'gen_s': gen, 'gen_tps': (n - 1) / gen if n > 1 else None,
                    'prefill_s': row['t_first_token'] - row['t_start'], 'wall_s': reply['total_s'],
                    'wall_tps': n / reply['total_s']})
    if not out or not any(r['gen_tps'] is not None for r in out):
        raise RuntimeError('no requests with a measurable decode span')
    return out


def run(a):
    if not replay_log.load_log(a.log):
        raise RuntimeError('empty recording')
    if a.rounds < 1:
        raise ValueError('rounds must be positive')
    if a.out.exists() and any(a.out.iterdir()):
        raise RuntimeError('output directory must be empty')
    a.out.mkdir(parents=True, exist_ok=True)
    orders = [ARMS[i % 3:] + ARMS[:i % 3] for i in range(a.rounds)]
    env = dict(os.environ)
    env.pop('B70_MTP_STEP_K', None)   # measure main's default per-step policy, even after an earlier A/B shell
    url = f'http://127.0.0.1:{a.port}'
    binary = Path(env.get('B70_SERVE', str(ROOT/'build/src/cli/b70-serve')))
    if not binary.is_absolute():
        binary = ROOT / binary
    with binary.open('rb') as f:
        binary_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    for round_, order in enumerate(orders, 1):
        for arm in order:
            d = a.out / f'{arm}.r{round_}'
            logs = d / 'server-log'
            logs.mkdir(parents=True)
            spec = ['--spec', arm]
            if arm == 'mtp':
                spec += ['--mtp', 'auto']
            elif arm == 'dflash':
                spec += ['--draft-model', a.drafter]
            command = [str(binary), a.model,
                       '--parallel', '1', '--max-num-seqs', '1', '--max-len', '131072', '--kv-cache', 'bf16',
                       '--lm-head', 'int8', '--prefix-cache-gb', '32', '--port', str(a.port),
                       '--log-requests', str(logs), *spec]
            print(f'REPLAY_SPEC {arm} round {round_}, cache on', flush=True)
            with (d/'server.log').open('w') as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + a.wait
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f'{arm}: server exited before readiness')
                        try:
                            with urllib.request.urlopen(url+'/v1/models', timeout=1):
                                break
                        except (OSError, urllib.error.URLError):
                            if time.monotonic() >= deadline:
                                raise RuntimeError(f'{arm}: server did not become ready')
                            time.sleep(.1)
                    replay_log.cmd_run(SimpleNamespace(log=a.log, url=url, out=d/'replay.json', seed=0, cap=0,
                                                       user_prefix=f'{arm}.r{round_}'))
                finally:
                    if server.poll() is None:
                        subprocess.run([str(ROOT/'tools/box_stop.sh'), 'child', str(server.pid)], check=True)
                    server.wait()
            if arm == 'mtp' and 'adaptive per step' not in (d/'server.log').read_text():
                raise RuntimeError('MTP arm did not enable the default per-step policy')
            rows = records(d/'replay.json', logs, arm, round_, Path(a.log).name)
            with (d/'requests.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row)+'\n')
    with (a.out/'table.txt').open('w') as f:
        subprocess.run([sys.executable, str(ROOT/'tools/box_validate/serve_client.py'), 'table', '--dir', str(a.out),
                        '--ref', 'off', '--ratio', 'dflash/mtp'], stdout=f, stderr=subprocess.STDOUT, check=True)
    with (a.out/'judge.log').open('w') as f:
        verdict = subprocess.run([sys.executable, str(ROOT/'tools/box_validate/serve_judge.py'), '--dir', str(a.out),
                                  '--ref', 'off', '--dump', str(a.out/'forced'), '--judge', str(a.judge),
                                  '--bf16', str(a.bf16)], stdout=f, stderr=subprocess.STDOUT)
    (a.out/'run.json').write_text(json.dumps({'orders': orders, 'model': a.model, 'drafter': a.drafter,
                                             'recording': str(a.log.resolve()), 'binary_sha256': binary_sha,
                                             'prefix_cache_gb': 32, 'parallel': 1, 'max_len': 131072,
                                             'card': a.card, 'grade': 'busy box' if a.card is not None else 'quiet box',
                                             'mtp_policy': 'per-step default', 'judge_rc': verdict.returncode}, indent=2)+'\n')
    print(f'REPLAY_SPEC complete: timings in table.txt; f32 judge rc={verdict.returncode}', flush=True)
    return verdict.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--model', default='urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ')
    ap.add_argument('--draft-model', dest='drafter', default='z-lab/Qwen3.8-27B-DFlash2')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--card', type=int, choices=(0, 1), help='reserve only this idle card; otherwise reserve the box')
    ap.add_argument('--port', type=int, default=8218)
    ap.add_argument('--wait', type=float, default=900)
    ap.add_argument('--judge', type=Path, default=ROOT/'oracle-out-qwen-f32-i8head-forced')
    ap.add_argument('--bf16', type=Path, default=ROOT/'oracle-out-qwen-bf16top-forced')
    ap.add_argument('--under-lock', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not a.under_lock:
        lock = ['box'] if a.card is None else ['card', str(a.card)]
        command = [str(ROOT/'tools/gpu_lock.sh'), *lock, '--', sys.executable, str(Path(__file__).resolve()),
                   '--under-lock', *(sys.argv[1:] if argv is None else argv)]
        return subprocess.run(command, cwd=ROOT).returncode
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, frame: sys.exit(128+s))
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
