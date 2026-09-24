#!/usr/bin/env python3
"""Spec 5's gates A2 and A3 from two prefill_gate_test logs (plan 5b Task 3).

    gate_compare.py <l0 log> <l0-int8 log>

Each log is the stdout of one `prefill_gate_test` run (directly, or through
`ctest -V`, whose "NN: " line prefix is stripped). Both runs must cover the
same prompts at the same chunk width. Per prompt it reads

  * the section header   "================ <name>: <T> prompt ids PREFILLED at
                            chunk <C> (<n> chunk(s)), <G> generated, ..."
  * the table header     "greedy <G>:  step  engine  golden  det?  logit-cos ..."
  * one row per greedy step under it (step, engine id, golden id, yes|TIE,
    logit-cos against the CPU oracle, ...), exactly <G> of them, steps 0..G-1
  * the verdict line     "<name>: <a> determined-exact / <b> tie-agreements /
                            <c> tie-set-members   (<d> determined + <u>
                            undetermined = <G>)[   ** GATE FAILURE **]"

and fails loudly (exit 2) on anything it cannot find, rather than guessing.

The bars (plan 5b Task 3, amending spec 5 section 3):

  A3  per prompt: the int8 verdict is PASS whenever the l0 verdict is PASS, with
      the same determined-exact count. PASS means every determined row exact and
      every undetermined row inside the golden argmax set (the gate's own rule).
  A2  per prompt: mean logit-cos(int8) >= mean logit-cos(l0) - 0.002 and
      min logit-cos(int8) >= min logit-cos(l0) - 0.01.

The logit-cos rows are the decode steps after the prefill, against the oracle.
Every one of them reads the KV and GDN state the prefill wrote, so they carry
the prefill's error; spec 5's A2 (prompt-logit relative L2) would need an
all-positions logits dump the engine does not have.

Exit 0 when both bars pass on every prompt, 1 when one fails (the prompt and
the bar are printed), 2 when a log cannot be parsed.
"""
import re
import sys

A2_MEAN_TOL = 0.002
A2_MIN_TOL = 0.01

PREFIX = re.compile(r"^\d+: ")   # ctest -V
SECTION = re.compile(
    r"^=+ (\S+): (\d+) prompt ids PREFILLED at chunk (\d+) \((\d+) chunk\(s\)\), "
    r"(\d+) generated, golden vocab \d+ =+$")
TABLE = re.compile(r"^\s*greedy (\d+):\s+step\s+engine\s+golden\s+det\?\s+logit-cos\s")
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(yes|TIE)\s+([0-9.eE+-]+)\s+\d+\s+\d+\s+\d+")
VERDICT = re.compile(
    r"^\s*(\S+): (\d+) determined-exact / (\d+) tie-agreements / (\d+) tie-set-members"
    r"\s+\((\d+) determined \+ (\d+) undetermined = (\d+)\)(.*)$")
BACKEND = re.compile(r"^prefill backend: (\S+)$")
NEAR = re.compile(r"^\s*(\S+): (\d+) determined row\(s\) accepted as bf16 near-ties")


def die(path, msg):
    sys.stderr.write(f"gate_compare.py: {path}: {msg}\n")
    sys.exit(2)


def parse(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [PREFIX.sub("", l.rstrip("\n")) for l in f]
    except OSError as e:
        die(path, f"cannot read: {e}")
    backend = None
    prompts = {}
    order = []
    cur = None          # the prompt whose section we are in
    in_table = False
    for n, line in enumerate(lines, 1):
        m = BACKEND.match(line)
        if m:
            backend = m.group(1)
            continue
        m = SECTION.match(line)
        if m:
            name = m.group(1)
            if name in prompts:
                die(path, f"line {n}: prompt '{name}' appears twice")
            cur = {"name": name, "ids": int(m.group(2)), "chunk": int(m.group(3)),
                   "chunks": int(m.group(4)), "gen": int(m.group(5)), "cos": [],
                   "table": False, "verdict": None}
            prompts[name] = cur
            order.append(name)
            in_table = False
            continue
        m = NEAR.match(line)
        if m and m.group(1) in prompts and prompts[m.group(1)].get("verdict"):
            # prefill_gate_test argv[7]: determined rows accepted as bf16 near-ties
            # (operator ruling 2026-09-24). They count toward A3 as exact.
            v = prompts[m.group(1)]["verdict"]
            v["near"] = int(m.group(2))
            v["pass"] = (v["det_exact"] + v["near"] == v["n_det"] and v["tie_ok"] == v["n_tie"]
                         and not v["failure_text"])
            continue
        if cur is None:
            continue
        m = TABLE.match(line)
        if m:
            if int(m.group(1)) != cur["gen"]:
                die(path, f"line {n}: table says greedy {m.group(1)}, section says "
                          f"{cur['gen']} generated")
            cur["table"] = True
            in_table = True
            continue
        if in_table:
            m = ROW.match(line)
            if m:
                step = int(m.group(1))
                if step != len(cur["cos"]):
                    die(path, f"line {n}: step {step} out of order in '{cur['name']}' "
                              f"(expected {len(cur['cos'])})")
                cur["cos"].append(float(m.group(5)))
                continue
        m = VERDICT.match(line)
        if m and m.group(1) == cur["name"]:
            det_exact, tie_agree, tie_member, n_det, n_tie, total = map(int, m.groups()[1:7])
            if total != cur["gen"] or n_det + n_tie != total:
                die(path, f"line {n}: verdict counts do not add up to {cur['gen']}")
            cur["verdict"] = {
                "det_exact": det_exact, "n_det": n_det, "n_tie": n_tie,
                "tie_ok": tie_agree + tie_member, "near": 0,
                "failure_text": "GATE FAILURE" in m.group(8),
                "pass": det_exact == n_det and tie_agree + tie_member == n_tie
                        and "GATE FAILURE" not in m.group(8),
            }
            in_table = False
            cur = None
    if backend is None:
        die(path, "no 'prefill backend: ...' line - is this a prefill_gate_test log?")
    if not order:
        die(path, "no '================ <prompt>: ... PREFILLED at chunk ...' section found")
    for name in order:
        p = prompts[name]
        if not p["table"]:
            die(path, f"'{name}': no 'greedy N:  step  engine  golden  det?  logit-cos' header")
        if len(p["cos"]) != p["gen"]:
            die(path, f"'{name}': {len(p['cos'])} logit-cos rows, expected {p['gen']}")
        if p["verdict"] is None:
            die(path, f"'{name}': no '<name>: N determined-exact / ...' verdict line")
    return backend, order, prompts


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    b0, order0, p0 = parse(argv[1])
    b1, order1, p1 = parse(argv[2])
    if order0 != order1:
        die(argv[2], f"prompts {order1} differ from the l0 log's {order0}")
    for name in order0:
        for key in ("ids", "chunk", "gen"):
            if p0[name][key] != p1[name][key]:
                die(argv[2], f"'{name}': {key} {p1[name][key]} != the l0 log's {p0[name][key]}")
    print(f"gate_compare: {argv[1]} ({b0})  vs  {argv[2]} ({b1})")
    if b0 != "l0" or b1 != "l0-int8":
        print(f"  note: backends are {b0} / {b1}, not l0 / l0-int8")
    print(f"  A2: mean >= l0 - {A2_MEAN_TOL}, min >= l0 - {A2_MIN_TOL};"
          f"  A3: int8 PASS where l0 PASS, same determined-exact")
    print("  prompt   ids  chunk  n   l0 mean      l0 min       int8 mean    int8 min"
          "     l0 gate          int8 gate        A2    A3")
    failed = []
    for name in order0:
        a, b = p0[name], p1[name]
        va, vb = a["verdict"], b["verdict"]
        ma, mb = sum(a["cos"]) / len(a["cos"]), sum(b["cos"]) / len(b["cos"])
        na, nb = min(a["cos"]), min(b["cos"])
        a2 = mb >= ma - A2_MEAN_TOL and nb >= na - A2_MIN_TOL
        a3 = (not va["pass"]) or (vb["pass"] and vb["det_exact"] + vb["near"] == va["det_exact"])

        def gate(v):
            near = f"+{v['near']}nt" if v["near"] else ""
            return (f"{'PASS' if v['pass'] else 'FAIL'} {v['det_exact']:2d}{near}/{v['n_det']:2d}"
                    f" t{v['tie_ok']}/{v['n_tie']}")
        print(f"  {name:7s} {a['ids']:5d} {a['chunk']:5d} {len(a['cos']):3d}"
              f"  {ma:.9f}  {na:.9f}  {mb:.9f}  {nb:.9f}  {gate(va):15s}  {gate(vb):15s}"
              f"  {'PASS' if a2 else 'FAIL'}  {'PASS' if a3 else 'FAIL'}")
        if not a2:
            failed.append(f"A2 FAIL on {name}: int8 mean {mb:.9f} vs l0 {ma:.9f} - {A2_MEAN_TOL},"
                          f" int8 min {nb:.9f} vs l0 {na:.9f} - {A2_MIN_TOL}")
        if not a3:
            failed.append(f"A3 FAIL on {name}: l0 {gate(va)}, int8 {gate(vb)}")
    all_a2 = not any(f.startswith("A2") for f in failed)
    all_a3 = not any(f.startswith("A3") for f in failed)
    print(f"  A2 {'PASS' if all_a2 else 'FAIL'}   A3 {'PASS' if all_a3 else 'FAIL'}")
    for f in failed:
        print(f"  {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
