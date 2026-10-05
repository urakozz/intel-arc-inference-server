#!/usr/bin/env python3
"""Two tokenizer.json files compared beyond their added tokens (stdlib only), for the box
queue's row 16: Ornith's against Qwen3.8's (spec 15 §12 checked only the added tokens).

  tok_diff.py <a/tokenizer.json> <b/tokenizer.json>

Prints one `TOKDIFF` line per part: the model's scalar fields (type, unk_token, ...), its
vocabulary (entries on one side only or at another id - the added tokens' contents left out),
its merges, the normalizer / pre_tokenizer / post_processor / decoder / truncation / padding,
and the added tokens (ids on one side only, an id with another content). Then
`TOKDIFF beyond the added tokens: none` or the parts that differ. A record, not a gate: exit 0
either way; 2 on a file that cannot be read.
"""
import json
import sys

PARTS = ("normalizer", "pre_tokenizer", "post_processor", "decoder", "truncation", "padding")


def merge_key(m):
    return tuple(m) if isinstance(m, list) else tuple(str(m).split(" ", 1))


def show(items, n=8):
    items = list(items)
    if not items:
        return "none"
    s = ", ".join(repr(x) for x in items[:n])
    return s + (f", ... (+{len(items) - n})" if len(items) > n else "")


def main(argv):
    if len(argv) != 3:
        print("usage: tok_diff.py <a/tokenizer.json> <b/tokenizer.json>", file=sys.stderr)
        return 2
    try:
        docs = []
        for p in argv[1:]:
            with open(p, encoding="utf-8") as f:
                docs.append(json.load(f))
    except (OSError, ValueError) as e:
        print(f"tok_diff: {e}")
        return 2
    a, b = docs
    print(f"TOKDIFF a = {argv[1]}")
    print(f"TOKDIFF b = {argv[2]}")
    differ = []

    added_a = {t["id"]: t["content"] for t in a.get("added_tokens") or []}
    added_b = {t["id"]: t["content"] for t in b.get("added_tokens") or []}
    added_text = set(added_a.values()) | set(added_b.values())

    ma, mb = a.get("model") or {}, b.get("model") or {}
    for k in sorted((set(ma) | set(mb)) - {"vocab", "merges"}):
        if ma.get(k) != mb.get(k):
            differ.append(f"model.{k}")
            print(f"TOKDIFF model.{k}: {ma.get(k)!r} vs {mb.get(k)!r}")
    va, vb = ma.get("vocab"), mb.get("vocab")
    if isinstance(va, dict) and isinstance(vb, dict):
        va = {t: i for t, i in va.items() if t not in added_text}
        vb = {t: i for t, i in vb.items() if t not in added_text}
        only_a = sorted(set(va) - set(vb), key=va.get)
        only_b = sorted(set(vb) - set(va), key=vb.get)
        moved = sorted((t for t in set(va) & set(vb) if va[t] != vb[t]), key=va.get)
        print(f"TOKDIFF model.vocab: {len(va)} vs {len(vb)} entries (added tokens left out); "
              f"only in a {len(only_a)}, only in b {len(only_b)}, at another id {len(moved)}")
        if only_a:
            print(f"TOKDIFF   only in a: {show((t, va[t]) for t in only_a)}")
        if only_b:
            print(f"TOKDIFF   only in b: {show((t, vb[t]) for t in only_b)}")
        if moved:
            print(f"TOKDIFF   at another id: {show((t, va[t], vb[t]) for t in moved)}")
        if only_a or only_b or moved:
            differ.append("model.vocab")
    elif va != vb:
        differ.append("model.vocab")
        print("TOKDIFF model.vocab: differs (not both BPE mappings)")
    else:
        print("TOKDIFF model.vocab: equal")
    la = [merge_key(m) for m in ma.get("merges") or []]
    lb = [merge_key(m) for m in mb.get("merges") or []]
    if la == lb:
        print(f"TOKDIFF model.merges: equal ({len(la)})")
    else:
        first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
        differ.append("model.merges")
        print(f"TOKDIFF model.merges: {len(la)} vs {len(lb)}, first difference at rank {first}")

    for part in PARTS:
        if a.get(part) == b.get(part):
            print(f"TOKDIFF {part}: equal")
        else:
            differ.append(part)
            print(f"TOKDIFF {part}: differs: {json.dumps(a.get(part), sort_keys=True)[:200]} vs "
                  f"{json.dumps(b.get(part), sort_keys=True)[:200]}")

    ids_a, ids_b = set(added_a), set(added_b)
    other = sorted(i for i in ids_a & ids_b if added_a[i] != added_b[i])
    print(f"TOKDIFF added_tokens: {len(ids_a)} vs {len(ids_b)}; only in a "
          f"{show((i, added_a[i]) for i in sorted(ids_a - ids_b))}; only in b "
          f"{show((i, added_b[i]) for i in sorted(ids_b - ids_a))}; another content at "
          f"{show((i, added_a[i], added_b[i]) for i in other)}")
    print(f"TOKDIFF beyond the added tokens: {', '.join(differ) if differ else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
