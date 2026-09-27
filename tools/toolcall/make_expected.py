#!/usr/bin/env python3
"""Build tests/server/toolcall_expected.json (spec 7 C6) from engine outputs.

    make_expected.py <toolcall-out dir> <set dir> <out json>

<toolcall-out dir> holds engine_generate.sh's <name>.<backend>.txt files (192
greedy ids, special tokens kept). Each text is cut at the first <|im_end|> (the
server stops there) and stored with score.py's reading of it: the first call's
name and raw parameter strings, and the number of complete calls. Scenarios come
from <set dir>/manifest.json; enable_thinking from <set dir>/<name>.json. Distinct
texts only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import parse_first_call  # noqa: E402

BACKENDS = ("bf16", "l0", "l0-int8")


def main() -> None:
    out_dir, set_dir, dest = sys.argv[1:4]
    names = [m["name"] for m in json.load(open(os.path.join(set_dir, "manifest.json")))]
    cases, seen = [], set()
    for name in sorted(names):
        scenario = json.load(open(os.path.join(set_dir, f"{name}.json"), encoding="utf-8"))
        for backend in BACKENDS:
            path = os.path.join(out_dir, f"{name}.{backend}.txt")
            if not os.path.exists(path):
                continue
            text = open(path, encoding="utf-8").read().split("<|im_end|>")[0]
            if text in seen:
                continue
            seen.add(text)
            kind, fn, params, n_calls = parse_first_call(text)
            cases.append({"scenario": name, "backend": backend,
                          "thinking": bool(scenario["enable_thinking"]), "text": text,
                          "kind": kind, "name": fn, "params": params, "n_calls": n_calls})
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(cases)} distinct outputs -> {dest}")


if __name__ == "__main__":
    main()
