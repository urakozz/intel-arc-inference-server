#!/usr/bin/env python3
"""ctest's test list and JUnit results, for tools/box_validate (stdlib only: runs on the box's
host python3 and on the Mac).

  junit.py names   TESTS_JSON [--re ERE] [--label L]... [--no-label L]... [--also-in JSON]
  junit.py todo    TESTS_JSON --re ERE [filters] --done GLOB
        the selected tests with no result yet in the JUnit files GLOB matches, as one anchored
        ctest -R alternation; empty output when every one has a result
  junit.py pick    TESTS_JSON --re ERE [filters] --junit GLOB
        the verdict over the selected tests, each at its latest result across GLOB (files in
        mtime order): exit 0 all passed, 1 a failure or a test with no result, 77 all skipped
  junit.py verdict FILE...           the same over every test in FILE...
  junit.py compare A B [--strip-a P] [--strip-b P] [--out DIFF]
        the output of every test present in both, normalised (timings dropped, tree paths and
        addresses replaced), compared line for line: exit 0 identical, 1 a difference
  junit.py grep    --junit GLOB --test ERE --pattern ERE [--label L]
        lines of the matching tests' output, as "METRIC <label>\t[<test>] <line>"

TESTS_JSON is `ctest --show-only=json-v1`. A test's status: `run` passed; `fail` failed (the
message says Failed / Timeout / ...); `notrun` skipped when ctest's message is its skip rule
(SKIP_RETURN_CODE, SKIP_REGULAR_EXPRESSION), failed otherwise (an executable not found is a
`notrun` too); `disabled` skipped.
"""
import argparse
import difflib
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# A line carrying a wall-clock quantity differs from run to run whatever the code does; the
# bitwise comparison drops it. Everything else (cosines, ids, counts, byte sizes) must match.
TIMING = re.compile(
    r"(\b\d+(\.\d+)?\s*(ns|us|µs|ms|s|sec|secs|seconds|min|minutes)\b"
    r"|[GMK]i?B/s|\bt/s\b|tok/s|tokens/s|\belapsed\b|\btook\b|Test time|\bwall\b|\buptime\b)",
    re.I)
HEX = re.compile(r"0x[0-9a-fA-F]+")


def tests_from_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for t in data.get("tests", []):
        labels = []
        for p in t.get("properties", []) or []:
            if p.get("name") == "LABELS":
                v = p.get("value")
                labels = list(v) if isinstance(v, list) else [v]
        out.append((t["name"], labels))
    return out


def select(args):
    tests = tests_from_json(args.tests)
    rx = re.compile(args.re or ".")
    also = None
    if getattr(args, "also_in", None):
        also = {n for n, _ in tests_from_json(args.also_in)}
    names = []
    for name, labels in tests:
        if not rx.search(name):
            continue
        if any(l not in labels for l in args.label or []):
            continue
        if any(l in labels for l in args.no_label or []):
            continue
        if also is not None and name not in also:
            continue
        names.append(name)
    return names


def parse_junit(path):
    """-> {name: (status, message, output)}"""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        print(f"junit: cannot read {path}: {e}", file=sys.stderr)
        return {}
    res = {}
    for tc in root.iter("testcase"):
        name = tc.get("name")
        st = tc.get("status", "")
        out = tc.findtext("system-out") or ""
        msg = ""
        f = tc.find("failure")
        s = tc.find("skipped")
        if f is not None:
            msg = f.get("message", "")
        elif s is not None:
            msg = s.get("message", "")
        if st == "run" and f is None:
            v = PASS
        elif st == "disabled":
            v, msg = SKIP, msg or "disabled"
        elif st == "notrun" and msg.startswith(("SKIP_RETURN_CODE", "SKIP_REGULAR_EXPRESSION")):
            v = SKIP
        elif st == "notrun":
            v, msg = FAIL, "not run: " + (msg or "?")
        else:
            v, msg = FAIL, msg or "Failed"
        res[name] = (v, msg, out)
    return res


def results(globs):
    files = []
    for g in globs:
        files.extend(glob.glob(g))
    files = sorted(set(files), key=lambda p: (os.path.getmtime(p), p))
    merged = {}
    for p in files:
        for name, r in parse_junit(p).items():
            merged[name] = r + (p,)
    return merged


def anchored(names):
    for n in names:
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", n):
            raise SystemExit(f"junit: test name {n!r} is not a plain identifier")
    esc = [n.replace(".", r"\.").replace("+", r"\+") for n in names]
    return "^(" + "|".join(esc) + ")$"


def report(rows):
    """rows: [(name, verdict, message)] -> exit code"""
    n = {PASS: 0, FAIL: 0, SKIP: 0}
    for name, v, msg in rows:
        n[v] += 1
        if v != PASS:
            print(f"{'FAILED' if v == FAIL else 'SKIPPED'} {name}: {msg}")
    print(f"JUNIT total={len(rows)} pass={n[PASS]} fail={n[FAIL]} skip={n[SKIP]}")
    if not rows:
        print("JUNIT no test selected")
        return 1
    if n[FAIL]:
        return 1
    if n[PASS] == 0:
        return 77
    return 0


def cmd_names(a):
    for n in select(a):
        print(n)
    return 0


def cmd_todo(a):
    done = results([a.done]) if a.done else {}
    todo = [n for n in select(a) if n not in done]
    if todo:
        print(anchored(todo))
    return 0


def cmd_pick(a):
    got = results([a.junit])
    rows = []
    for n in select(a):
        if n in got:
            v, msg, _, src = got[n]
            rows.append((n, v, msg))
        else:
            rows.append((n, FAIL, "no result in this run"))
    return report(rows)


def cmd_verdict(a):
    rows = []
    for p in a.files:
        if not os.path.exists(p):
            print(f"FAILED {p}: no JUnit file (ctest did not run)")
            return 1
        rows.extend((n, v, msg) for n, (v, msg, _) in parse_junit(p).items())
    return report(rows)


def normalise(text, strips):
    out = []
    for line in text.splitlines():
        for s in strips:
            if s:
                line = line.replace(s, "<TREE>")
        line = HEX.sub("0x?", line).rstrip()
        if TIMING.search(line):
            continue
        out.append(line)
    return out


def cmd_compare(a):
    ra, rb = parse_junit(a.a), parse_junit(a.b)
    common = sorted(set(ra) & set(rb))
    diff_lines, differ = [], []
    for name in sorted(set(ra) ^ set(rb)):
        print(f"NOTE only in {'A' if name in ra else 'B'}: {name}")
    for name in common:
        la = normalise(ra[name][2], [a.strip_a, a.strip_b])
        lb = normalise(rb[name][2], [a.strip_a, a.strip_b])
        if la != lb:
            differ.append(name)
            diff_lines.extend(difflib.unified_diff(la, lb, f"baseline/{name}", f"under-test/{name}",
                                                   lineterm="", n=2))
        print(f"{'DIFFER' if la != lb else 'SAME  '} {name} ({len(la)} lines compared)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(diff_lines) + ("\n" if diff_lines else ""))
    print(f"COMPARE common={len(common)} identical={len(common) - len(differ)} differ={len(differ)}")
    if not common:
        print("COMPARE nothing in common to compare")
        return 1
    return 1 if differ else 0


def cmd_grep(a):
    trx, prx = re.compile(a.test), re.compile(a.pattern)
    got = results([a.junit])
    n = 0
    for name in sorted(got):
        if not trx.search(name):
            continue
        for line in got[name][2].splitlines():
            if prx.search(line):
                print(f"METRIC {a.label}\t[{name}] {line.strip()}")
                n += 1
                if n >= a.max:
                    return 0
    if n == 0:
        print(f"METRIC {a.label}\t(not printed)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def filters(p):
        p.add_argument("tests")
        p.add_argument("--re", default=".")
        p.add_argument("--label", action="append")
        p.add_argument("--no-label", action="append")
        p.add_argument("--also-in")

    p = sub.add_parser("names"); filters(p); p.set_defaults(fn=cmd_names)
    p = sub.add_parser("todo"); filters(p); p.add_argument("--done"); p.set_defaults(fn=cmd_todo)
    p = sub.add_parser("pick"); filters(p); p.add_argument("--junit", required=True); p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("verdict"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_verdict)
    p = sub.add_parser("compare")
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--strip-a", default=""); p.add_argument("--strip-b", default="")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("grep")
    p.add_argument("--junit", required=True); p.add_argument("--test", required=True)
    p.add_argument("--pattern", required=True); p.add_argument("--label", default="output")
    p.add_argument("--max", type=int, default=40)
    p.set_defaults(fn=cmd_grep)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
