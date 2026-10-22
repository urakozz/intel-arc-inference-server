#!/usr/bin/env python3
"""Spec 25e: b70-serve's A4 outputs as run files for score.py.

    serve_a4_texts.py <server --log-requests dir> <set dir> <out dir> <run>

<set dir> is an A4 set (tests/golden/toolcall-bolt: manifest.json, <name>.ids - the prompt ids the reference rendered with the
checkpoint's template). tools/box_validate/serve_run.sh runs tools/box_validate/serve_client.py over it against b70-serve, which
writes one record per request into its --log-requests directory: the prompt ids as the SERVER rendered and encoded them and
`response_text`, the raw generated text (special tokens kept, the EOS id's text never pushed). This writes
<out dir>/<name>.<run>.txt for every scenario - the text of the request whose prompt ids equal the scenario's <name>.ids, the
last such request when there are several (a warm-up) - so `score.py --format qwen-think <out dir> int4 <run>` reads the server's
outputs beside the reference's <name>.int4.txt, and prints how many scenarios matched. A scenario whose prompt the server never
saw (its rendered ids differ from the reference's) is reported by name and makes the exit status 1: that is the template or
tokenizer path itself differing, the first thing the row must say.
"""
import json
import os
import sys


def read_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def main(argv):
    if len(argv) != 5:
        sys.exit(__doc__)
    log_dir, set_dir, out_dir, run = argv[1:]
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        names = [e["name"] for e in json.load(f)]
    by_ids = {}
    for n in sorted(os.listdir(log_dir)):
        if n.endswith(".json") and n[:-5].isdigit():
            with open(os.path.join(log_dir, n), encoding="utf-8") as f:
                rec = json.load(f)
            by_ids[tuple(rec["prompt_ids"])] = rec
    os.makedirs(out_dir, exist_ok=True)
    missing = []
    for name in names:
        rec = by_ids.get(tuple(read_ids(os.path.join(set_dir, name + ".ids"))))
        if rec is None:
            missing.append(name)
            continue
        with open(os.path.join(out_dir, f"{name}.{run}.txt"), "w", encoding="utf-8", newline="") as f:
            f.write(rec["response_text"])
    print(f"serve_a4_texts: {len(names) - len(missing)} of {len(names)} scenarios written as <name>.{run}.txt")
    for name in missing:
        print(f"serve_a4_texts: no request with the prompt ids of {name} - the server rendered or encoded it differently")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
