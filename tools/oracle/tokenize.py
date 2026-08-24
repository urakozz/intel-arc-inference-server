#!/usr/bin/env python3
"""Tokenize a golden prompt with the checkpoint's own tokenizer.

The golden prompts are RAW TEXT continuations - no chat template, no special
tokens (doc 11: the template is a separate chat_template.jinja and is not part
of the decode-core contract). Trailing newlines are stripped so the model
continues the sentence rather than a blank line; interior newlines are kept
(code.txt needs them).

Usage, inside the reference container (doc 10):
    tokenize.py <snapshot> encode <text-file>   -> ids, whitespace-separated
    tokenize.py <snapshot> decode <id> [<id>..] -> the text those ids spell
"""
import os
import sys

# `tokenize.py` next to this script shadows the stdlib `tokenize` that `inspect`
# imports, and torch/transformers import `inspect`. Drop this directory from
# sys.path BEFORE importing anything third-party (python -P does the same, but
# this works however the script is invoked).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="local snapshot directory of the checkpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("text_file")
    dec = sub.add_parser("decode")
    dec.add_argument("ids", nargs="+", type=int)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.snapshot)

    if args.cmd == "encode":
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().rstrip("\n")
        ids = tok.encode(text)
        # Qwen2Tokenizer adds no BOS/EOS; say so loudly if a future tokenizer does.
        bare = tok.encode(text, add_special_tokens=False)
        if ids != bare:
            print(f"tokenizer added special tokens: {len(ids)} vs {len(bare)} bare ids", file=sys.stderr)
            sys.exit(1)
        print(f"{args.text_file}: {len(ids)} ids", file=sys.stderr)
        print(" ".join(str(i) for i in ids))
    else:
        print(tok.decode(args.ids))


if __name__ == "__main__":
    main()
