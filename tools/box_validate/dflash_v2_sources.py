#!/usr/bin/env python3
"""Spec 19 v2's recorded continuations, without loading a CPU model or the old tap tensors.

  dflash_v2_sources.py DUMPS HELD_OUT_RECORDING GENERATION_CONFIG OUT.json

A4/golden come from 19a's dumps. The held-out source is the first request in the separate recording whose prompt
fits 16k and whose EOS-stopped continuation has at least eight tokens. Task 6 must exclude that recording entirely.
"""
import argparse
import glob
import json
import os
import struct


def stopped(ids, eos):
    for i, token in enumerate(ids):
        if token in eos:
            return ids[:i+1]
    return ids


def dump_source(path, eos):
    with open(path, 'rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
        meta, tensor = header['__metadata__'], header['ids']
        if tensor['dtype'] != 'I32':
            raise ValueError(f'{path}: ids must be I32')
        begin, end = tensor['data_offsets']
        f.seek(8 + length + begin)
        ids = list(struct.unpack('<' + 'i' * ((end - begin) // 4), f.read(end - begin)))
    n = int(meta['n_prompt'])
    name = meta['name']
    leaf = name.rsplit('/', 1)[-1]
    if leaf in ('code', 'prose', 'cjk'):
        name = leaf + '/' + leaf
    return {'name': name, 'prompt_ids': ids[:n], 'cont_ids': stopped(ids[n:], eos)}


def prepare(dumps, held_out, config):
    with open(config) as f:
        eos = json.load(f)['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    sources = [dump_source(p, eos) for p in sorted(glob.glob(os.path.join(dumps, '*.taps.safetensors')))]
    sources = [s for s in sources if len(s['cont_ids']) >= 8]
    for path in sorted(glob.glob(os.path.join(held_out, '*.json'))):
        with open(path) as f:
            rec = json.load(f)
        prompt, cont = rec.get('prompt_ids') or [], stopped((rec.get('out_ids') or [])[:256], eos)
        if prompt and len(prompt) <= 16000 and len(cont) >= 8:
            sources.append({'name': 'opencode/' + os.path.basename(path)[:-5], 'prompt_ids': prompt,
                            'cont_ids': cont, 'held_out_recording': os.path.basename(os.path.normpath(held_out))})
            break
    if not sources or not any(s['name'].startswith('opencode/') for s in sources):
        raise ValueError('need the A4/golden dumps and a usable held-out request')
    if len({s['name'] for s in sources}) != len(sources):
        raise ValueError('duplicate source names')
    for source in sources:
        source['eos_ids'] = sorted(eos)
    return sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dumps')
    ap.add_argument('held_out')
    ap.add_argument('generation_config')
    ap.add_argument('out')
    a = ap.parse_args()
    sources = prepare(a.dumps, a.held_out, a.generation_config)
    with open(a.out, 'w') as f:
        json.dump(sources, f)
    for s in sources:
        print(f"DFLASH_SOURCE {s['name']} prompt={len(s['prompt_ids'])} continuation={len(s['cont_ids'])}")


if __name__ == '__main__':
    main()
