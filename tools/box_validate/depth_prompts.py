#!/usr/bin/env python3
"""Text prompts of a given depth for the server speed rows (stdlib only):

  depth_prompts.py <out dir> <tokens>...      e.g. depth_prompts.py /tmp/d 4096 32768

Writes <out dir>/depth<N>.txt per N: the repo's own prose (docs/**/*.md, sorted by path, then the spec documents)
concatenated and cut at ~N tokens (3.14 characters a token: Qwen3.8's tokenizer gave this text 5199 / 40046 tokens
for 16334 / 130977 characters, 2026-11-10; the server reports the exact count as prompt_tokens), ending with a line asking for a summary - so the
continuation is natural text and both arms of a row decode the same prompt. serve_client.py run --set <out dir>
sends every *.txt as a /v1/completions prompt.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARS_PER_TOKEN = 3.14
TAIL = "\n\nSummarise the documents above in plain words, section by section:\n\n"


def corpus():
    paths = []
    for base, _, files in os.walk(os.path.join(ROOT, "docs")):
        paths += [os.path.join(base, f) for f in files if f.endswith(".md")]
    text = []
    for p in sorted(paths):
        with open(p, encoding="utf-8", errors="replace") as f:
            text.append(f.read())
    return "\n\n".join(text)


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    out = argv[1]
    os.makedirs(out, exist_ok=True)
    body = corpus()
    for n in map(int, argv[2:]):
        want = int(n * CHARS_PER_TOKEN) - len(TAIL)
        if want > len(body):
            sys.exit(f"depth_prompts: the corpus has {len(body)} characters, {want} asked")
        cut = body[:want]
        cut = cut[: cut.rfind("\n") + 1] if "\n" in cut else cut   # end on a line
        with open(os.path.join(out, f"depth{n}.txt"), "w", encoding="utf-8") as f:
            f.write(cut + TAIL)
        print(f"depth{n}.txt: {len(cut) + len(TAIL)} characters (~{n} tokens)")


if __name__ == "__main__":
    main(sys.argv)
