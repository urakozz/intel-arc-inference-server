#!/usr/bin/env python3
"""A long prompt from committed ids, no tokenizer: python3 long_ids.py <n> <out.ids> [tail.ids]

tests/golden/prompts/long32k.ids repeated (and cut) to n - len(tail) ids, then the tail
(default tests/golden/prompts/prose.ids), one id per line - what b70-decode --ids reads.
The box runbook's long-context MTP greedy check (r6.greedy_long) runs b70-decode on it with
and without --mtp; the ids only need to be fixed and legal, not natural text.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    n, out = int(argv[1]), argv[2]
    tail = read(argv[3] if len(argv) == 4 else os.path.join(ROOT, "tests/golden/prompts/prose.ids"))
    filler = read(os.path.join(ROOT, "tests/golden/prompts/long32k.ids"))
    if n <= len(tail):
        sys.exit(f"long_ids: n {n} <= the tail's {len(tail)} ids")
    body = (filler * (1 + (n - len(tail)) // len(filler)))[: n - len(tail)]
    with open(out, "w") as f:
        f.write("\n".join(map(str, body + tail)) + "\n")
    print(f"long_ids: {n} ids -> {out} ({n - len(tail)} of long32k.ids repeated, then {len(tail)} tail ids)")


if __name__ == "__main__":
    main(sys.argv)
