#!/usr/bin/env python3
"""mk_long_ids.py - a long, natural-text prompt for the spec 6 long-context gates.

    mk_long_ids.py <snapshot> <out.ids> <n>

The text is the concatenation of docs/*.md and src/**/*.cc (sorted paths, each
file once, the whole list repeated until it is long enough), tokenised with the
snapshot's own tokenizer.json (no special tokens, no chat template), and cut to
exactly n ids. Written one id per line, the format b70-decode --ids and
golden::read_ids read.

Run it in the oracle container (it has `tokenizers`), from the repo root:
    tools/oracle/run_in_container.sh 'python3 tools/probe/mk_long_ids.py "$SNAP" \
        tests/golden/prompts/long32k.ids 32768'
Plan: docs/superpowers/plans/2026-09-25-spec6b-flash-attn-integration-and-128k.md, Tasks 4-5.
"""
import glob
import os
import sys


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    snap, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        encode = lambda t: tok.encode(t, add_special_tokens=False).ids  # noqa: E731
    except ImportError:
        from transformers import AutoTokenizer
        atok = AutoTokenizer.from_pretrained(snap)
        encode = lambda t: atok.encode(t, add_special_tokens=False)  # noqa: E731
    paths = sorted(glob.glob("docs/*.md")) + sorted(glob.glob("src/**/*.cc", recursive=True))
    if not paths:
        sys.exit("mk_long_ids.py: run from the repo root (no docs/*.md found)")
    ids = []
    rounds = 0
    while len(ids) < n:
        rounds += 1
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                ids.extend(encode(f.read() + "\n\n"))
            if len(ids) >= n:
                break
    ids = ids[:n]
    with open(out, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print(f"mk_long_ids: {len(ids)} ids from {len(paths)} files ({rounds} round(s)) -> {out}")


if __name__ == "__main__":
    main()
