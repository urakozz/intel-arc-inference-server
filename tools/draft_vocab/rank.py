#!/usr/bin/env python3
"""A ranked token-id list for `b70-serve --draft-vocab-ids` (spec 8 §11).

    rank.py [-o FILE] [--gen-weight 4] [--prompt-weight 1] [--ids-weight 1]
            [--prompt-all] [--vocab-used N] [--top N] INPUT...

INPUT is any mix of
  - request logs: what `b70-serve --log-requests DIR` writes, NNNNNN.json per request with
    `prompt_ids` and `out_ids` (a directory is scanned for NNNNNN.json and *.ids);
  - `.ids` files: whitespace-separated token ids (tests/golden/prompts/*.ids, b70-decode's
    output, any tokenised corpus).

Writes the ids, most frequent first, one per line, after a `#` header the C++ reader skips
(loader::read_ranked_ids). Ties go to the lower id, so the list is a pure function of the
inputs. Only ids that occur are listed: the loader fills the rest of V' with the lowest
remaining ids.

**The weights are a choice, not a measurement.** The draft head predicts the model's OUTPUT,
so a generated id (`out_ids`) counts `--gen-weight` (default 4). Agentic output copies from
its context - file contents, tool results, the user's words - so prompt ids are evidence too,
at `--prompt-weight` (default 1). `.ids` files are a corpus of unknown role and count
`--ids-weight` (default 1).

**Prompt ids are counted once per session, not once per request.** An agentic client resends
the whole conversation every turn, so a log's prompts repeat the early context once per
request after it; counting them all would rank the system prompt's ids by the session's
length. A request's prompt is counted only from the end of its longest common prefix with
the previous request's ids (its prompt + generated, in file order) - the part that is new
(tools/prefix/analyze_log.py's measure). `--prompt-all` counts every prompt id.

Dependency-free Python 3. Tests: `python3 tools/draft_vocab/test_rank.py`.
"""
import argparse
import json
import os
import sys


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for w in s.split():
            if not w.isdigit():
                raise ValueError(f"{path}: '{w}' is not a token id")
            ids.append(int(w))
    return ids


def expand(inputs):
    """(kind, path) for every input file, in a stable order: 'log' or 'ids'."""
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                full = os.path.join(p, n)
                if n.endswith(".json") and n[:-5].isdigit():
                    out.append(("log", full))
                elif n.endswith(".ids"):
                    out.append(("ids", full))
        elif p.endswith(".json"):
            out.append(("log", p))
        else:
            out.append(("ids", p))
    return out


def count(inputs, gen_weight=4, prompt_weight=1, ids_weight=1, prompt_all=False):
    """id -> weighted count over the inputs, and a summary of what was read."""
    counts = {}
    seen = {"logs": 0, "ids_files": 0, "gen": 0, "prompt": 0, "prompt_skipped": 0, "corpus": 0}

    def add(ids, w):
        for i in ids:
            counts[i] = counts.get(i, 0) + w

    prev = None   # the previous request's prompt + generated ids (logs only, file order)
    for kind, path in expand(inputs):
        if kind == "ids":
            ids = read_ids(path)
            add(ids, ids_weight)
            seen["ids_files"] += 1
            seen["corpus"] += len(ids)
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        prompt = list(rec.get("prompt_ids") or [])
        out = list(rec.get("out_ids") or [])
        start = 0 if (prompt_all or prev is None) else lcp(prompt, prev)
        add(prompt[start:], prompt_weight)
        add(out, gen_weight)
        seen["logs"] += 1
        seen["prompt"] += len(prompt) - start
        seen["prompt_skipped"] += start
        seen["gen"] += len(out)
        prev = prompt + out
    return counts, seen


def rank(counts, vocab_used=None, top=None):
    """Ids by descending count, ties to the lower id; ids >= vocab_used dropped."""
    items = [(c, i) for i, c in counts.items() if c > 0 and (vocab_used is None or i < vocab_used)]
    items.sort(key=lambda t: (-t[0], t[1]))
    ids = [i for _, i in items]
    return ids if top is None else ids[:top]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="request-log dirs or files, .ids files")
    ap.add_argument("-o", "--output", help="write here instead of stdout")
    ap.add_argument("--gen-weight", type=int, default=4)
    ap.add_argument("--prompt-weight", type=int, default=1)
    ap.add_argument("--ids-weight", type=int, default=1)
    ap.add_argument("--prompt-all", action="store_true",
                    help="count every prompt id, not only the part new to the session")
    ap.add_argument("--vocab-used", type=int, default=None,
                    help="drop ids at or above this (the loader skips them anyway)")
    ap.add_argument("--top", type=int, default=None, help="list at most this many ids")
    a = ap.parse_args(argv)
    if min(a.gen_weight, a.prompt_weight, a.ids_weight) < 0:
        ap.error("weights must be >= 0")
    counts, seen = count(a.inputs, a.gen_weight, a.prompt_weight, a.ids_weight, a.prompt_all)
    ids = rank(counts, a.vocab_used, a.top)
    if not ids:
        print("rank.py: no ids in the inputs", file=sys.stderr)
        return 1
    header = (f"# tools/draft_vocab/rank.py: {seen['logs']} request logs ({seen['gen']} generated"
              f" x{a.gen_weight}, {seen['prompt']} new prompt x{a.prompt_weight},"
              f" {seen['prompt_skipped']} repeated prompt skipped), {seen['ids_files']} .ids files"
              f" ({seen['corpus']} x{a.ids_weight}); {len(ids)} distinct ids, most frequent first\n")
    text = header + "".join(f"{i}\n" for i in ids)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    print(header.strip()[2:], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
