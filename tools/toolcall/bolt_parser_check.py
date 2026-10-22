#!/usr/bin/env python3
"""Spec 25e: hold the server's Bolt output parser to the A4 scorer on real run files (row 62's r62.a4, plan 25e Task 2).

    bolt_parser_check.py <bolt_parse_dump> <out dir> <run>...
    bolt_parser_check.py --selftest <bolt_parse_dump> <tests/server/bolt_parse_cases.json>

Reads <out dir>/<scenario>.<run>.txt (the generated text after the prompt's `<think>\\n`, special tokens kept - the files
tools/toolcall/oracle_generate.py / engine_generate.sh write), reads the first call of each with score.py's `qwen-think`
reader (the A4 scorer, 25a) and with the server's parser (the `bolt_parse_dump` executable: server::parse_bolt_output), and
counts: files, files where the scorer finds a call, files where the server does, and FAILURES - a file where the scorer finds a
call and the server's first call differs by name or by an argument (a string equals the raw text, any other value is the raw
text's JSON). Exit status 1 on any failure. The server may read a call the scorer does not (it types values, and a cut call is
dropped where the scorer says `incomplete`): those are counted, not failed.

--selftest: the texts of the parse-case file's thinking-on cases that carry the scorer's first call, written to a temporary
directory and checked the same way (so the checker itself is held to the cases).
"""
import json
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import score  # noqa: E402


def server_reads(exe, files):
    out = subprocess.run([exe] + files, check=True, capture_output=True, text=True).stdout.splitlines()
    return [json.loads(line) for line in out]


def same_value(server_value, raw):
    if isinstance(server_value, str) and server_value == raw:
        return True
    try:
        return json.loads(raw) == server_value
    except ValueError:
        return False


def check(exe, paths):
    texts = [open(p, encoding="utf-8", newline="").read() for p in paths]
    got = server_reads(exe, paths)
    n = len(paths)
    scorer_calls = server_calls = failures = 0
    for p, text, s in zip(paths, texts, got):
        kind, name, params, _ = score.parse_first_call(text, "qwen-think")
        server_calls += 1 if s["tool_calls"] else 0
        if kind != "call":
            continue
        scorer_calls += 1
        calls = s["tool_calls"]
        why = None
        if not calls:
            why = "the server read no call"
        elif calls[0]["name"] != name:
            why = f"name {calls[0]['name']!r} != {name!r}"
        else:
            args = calls[0]["arguments"]
            if set(args) != set(params):
                why = f"argument keys {sorted(args)} != {sorted(params)}"
            else:
                for k, raw in params.items():
                    if not same_value(args[k], raw):
                        why = f"argument {k!r}: {args[k]!r} vs raw {raw!r}"
                        break
        if why:
            failures += 1
            print(f"FAIL {p}: {why}")
    print(f"bolt_parser_check: {n} files, the scorer finds a call in {scorer_calls}, the server in {server_calls}, "
          f"{failures} failures")
    return failures


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--selftest":
        exe, cases = sys.argv[2], sys.argv[3]
        facts = json.load(open(cases, encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for c in facts["cases"]:
                if c["thinking"] and "scorer" in c:
                    p = os.path.join(d, c["name"] + ".txt")
                    with open(p, "w", encoding="utf-8", newline="") as f:
                        f.write(c["text"])
                    paths.append(p)
            assert len(paths) >= 5, "the case file carries too few scorer cases"
            sys.exit(1 if check(exe, paths) else 0)
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    exe, out_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    paths = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                   if f.endswith(".txt") and any(f.endswith("." + r + ".txt") for r in runs))
    if not paths:
        sys.exit(f"bolt_parser_check: no <scenario>.<run>.txt in {out_dir} for runs {runs}")
    sys.exit(1 if check(exe, paths) else 0)


if __name__ == "__main__":
    main()
