#!/usr/bin/env python3
"""tests/tokenizer/corpus.txt -> corpus.ids with Python `tokenizers` (the reference).
Run ON THE BOX: ~/auto-round/.venv/bin/python tools/tokenizer/dump_parity.py <tokenizer.json>
Also asserts decode(encode(x)) == x for every case and prints the version used."""
import sys

import tokenizers
from tokenizers import Tokenizer


tok = Tokenizer.from_file(sys.argv[1])
src = open("tests/tokenizer/corpus.txt", encoding="utf-8", newline="").read().split("\n")
if src and src[-1] == "":
    src.pop()
out, bad = [], 0
for line in src:
    ids = tok.encode(line, add_special_tokens=False).ids
    if tok.decode(ids, skip_special_tokens=False) != line:
        bad += 1
    out.append(" ".join(map(str, ids)))
open("tests/tokenizer/corpus.ids", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"tokenizers {tokenizers.__version__}: {len(src)} cases, {sum(len(o.split()) for o in out)} ids, {bad} roundtrip failures")
sys.exit(1 if bad else 0)
