"""Tokenise text files into a 2048-id prompt for probe_w4a8 --capture[-down].

usage: python mk_ids.py '<snapshot glob>' <out.ids> <text file>...
The snapshot's own tokenizer.json is used; the first 2048 ids are written.
docs/probe-w4a8-2026-09-23.md section 14 used docs/05-perf-model.md and
src/runtime/prefill/context.cc as the held-out evaluation text.
"""
import sys, glob
from tokenizers import Tokenizer
snap = glob.glob(sys.argv[1])[0]
tok = Tokenizer.from_file(snap + "/tokenizer.json")
text = "".join(open(p).read() for p in sys.argv[3:])
ids = tok.encode(text, add_special_tokens=False).ids
assert len(ids) >= 2048, len(ids)
open(sys.argv[2], "w").write(" ".join(map(str, ids[:2048])) + "\n")
print(len(ids), "tokens; wrote first 2048")
