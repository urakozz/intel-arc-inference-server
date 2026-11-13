#!/usr/bin/env python3
"""G0's baseline build and its recorded gate outputs, kept between runs (stdlib only: the box's host python3).

  baseline.py build-check --base DIR [--cmake-args S]
        exit 0 and "BASELINE reused: ..." when DIR holds a complete build of the baseline commit
        (DIR/.b70-baseline-ref) with the same identity as the one stamped after it was built; exit 1 and
        "BASELINE rebuild: <why>" otherwise (the stamp is moved aside first: a build that is interrupted
        leaves none). A reused baseline's directory is touched: tools/box_tidy.sh keeps the newest ones.
  baseline.py build-stamp --base DIR [--cmake-args S]       the build in DIR finished: record its identity
  baseline.py gates-check --base DIR --re ERE --snap DIR --out FILE [--cmake-args S]
        exit 0 and the baseline's recorded gate output (g0.bitwise's JUnit) copied to FILE when the build
        identity, the gate regex and the data the gates read (the golden sets, the snapshot; not the card - the
        cards are equal, operator 2026-11-13) are
        what they were when it was recorded; exit 1 and "BASELINE gates rerun: <why>" otherwise
  baseline.py gates-stamp --base DIR --re ERE --snap DIR --xml FILE [--cmake-args S]

The identity of a build: the baseline commit, the CMake arguments, tools/box_validate/reuse.py's build_id
(the tree's sources, the build's cache settings, the compiler's version) and machine() (the kernel series),
and ocloc's version (it compiles every device binary). A complete build also has every kernel binary the
stamp counted, the test list and b70-decode.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse  # noqa: E402

STAMP = ".b70-baseline-build.json"
GATES = ".b70-baseline-gates.json"
GATES_XML = ".b70-baseline-gates.xml"


def ocloc_version():
    try:
        r = subprocess.run(["ocloc", "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "no ocloc"
    return (r.stdout + r.stderr).strip()


def kernel_bins(base):
    n = 0
    for d, _, files in os.walk(os.path.join(base, "build")):
        if os.path.basename(d) == "kernels":
            n += sum(1 for f in files if f.endswith(".bin"))
    return n


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def baseline_ref(base):
    try:
        with open(os.path.join(base, ".b70-baseline-ref"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def identity(base, cmake_args):
    return {"sha": baseline_ref(base), "cmake_args": cmake_args, "build_id": reuse.build_id(base),
            "machine": reuse.machine(), "ocloc": ocloc_version()}


def differs(was, now):
    """-> the first part of the identity that is not the stamp's, in words, or None"""
    for k, what in (("sha", "the baseline commit"), ("cmake_args", "the CMake arguments"),
                    ("build_id", "the sources, the build's cache settings or the compiler"),
                    ("machine", "the machine (kernel series)"), ("ocloc", "ocloc's version")):
        if was.get(k) != now.get(k):
            if k in ("machine", "cmake_args"):
                return f"{what} changed ({was.get(k)!r} -> {now.get(k)!r})"
            return f"{what} changed"
    return None


def complete(base, stamp):
    n = kernel_bins(base)
    if n != stamp.get("kernel_bins"):
        return f"the tree is incomplete: {n} kernel binaries, the stamp counted {stamp.get('kernel_bins')}"
    for rel in ("build/CTestTestfile.cmake", "build/src/cli/b70-decode"):
        if not os.path.exists(os.path.join(base, rel)):
            return f"the tree is incomplete: no {rel}"
    return None


def cmd_build_check(a):
    stamp_path = os.path.join(a.base, STAMP)
    stamp = read_json(stamp_path)
    why = None
    if not baseline_ref(a.base):
        why = "no .b70-baseline-ref (the tree was not extracted by box_validate.sh)"
    elif stamp is None:
        why = "no build stamp (never built here, or a build did not finish)"
    elif not os.path.isdir(os.path.join(a.base, "build")):
        why = "no build directory"
    else:
        now = identity(a.base, a.cmake_args)
        why = differs(stamp["identity"], now) or complete(a.base, stamp)
    if why is None:
        os.utime(a.base)   # in use: the newest baseline tree is the one tools/box_tidy.sh keeps
        print(f"BASELINE reused: {baseline_ref(a.base)[:12]} built {stamp['built']}, same sources, build settings, "
              f"compiler, ocloc and kernel series {stamp['identity']['machine'].get('kernel')}; "
              f"{stamp['kernel_bins']} kernel binaries")
        return 0
    if stamp is not None:
        os.replace(stamp_path, stamp_path + ".prev")
    for p in (GATES, GATES_XML):   # recorded on a build that is going away
        if os.path.exists(os.path.join(a.base, p)):
            os.replace(os.path.join(a.base, p), os.path.join(a.base, p + ".prev"))
    print(f"BASELINE rebuild: {why}")
    return 1


def cmd_build_stamp(a):
    ident = identity(a.base, a.cmake_args)
    write_json(os.path.join(a.base, STAMP), {"identity": ident, "kernel_bins": kernel_bins(a.base),
                                             "built": time.strftime("%Y-%m-%d %H:%M")})
    print(f"BASELINE stamped: {ident['sha'][:12]}, {kernel_bins(a.base)} kernel binaries")
    return 0


def tree_fingerprint(paths):
    """the files under PATHS (symbolic links followed): relative name, size, mtime - what the gates read"""
    h = hashlib.sha256()
    for top in paths:
        h.update(f"T {os.path.basename(top)}\n".encode())
        for d, dirs, files in os.walk(top, followlinks=True):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                try:
                    st = os.stat(p)
                except OSError:
                    h.update(f"M {os.path.relpath(p, top)}\n".encode())
                    continue
                h.update(f"F {os.path.relpath(p, top)} {st.st_size} {st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def gates_key(a):
    golden = sorted(os.path.join(a.base, n) for n in os.listdir(a.base) if n.startswith("oracle-out"))
    return {"identity": identity(a.base, a.cmake_args), "re": a.re,
            "snap": a.snap, "data": tree_fingerprint(golden + ([a.snap] if os.path.isdir(a.snap) else []))}


def cmd_gates_check(a):
    rec = read_json(os.path.join(a.base, GATES))
    xml = os.path.join(a.base, GATES_XML)
    why = None
    if rec is None or not os.path.exists(xml):
        why = "no recorded gate output"
    elif not os.path.exists(os.path.join(a.base, STAMP)):
        why = "the baseline build has no stamp"
    else:
        now = gates_key(a)
        why = differs(rec["key"]["identity"], now["identity"])
        for k, what in (("re", "the gate list"), ("snap", "the snapshot"),
                        ("data", "the golden sets or the snapshot's files")):
            if why is None and rec["key"].get(k) != now[k]:
                why = f"{what} changed"
    if why is not None:
        print(f"BASELINE gates rerun: {why}")
        return 1
    shutil.copyfile(xml, a.out)
    print(f"BASELINE gates reused: the baseline's own gate output recorded {rec['recorded']} "
          "(same build, gate list and data; the cards are equal, operator 2026-11-13)")
    return 0


def cmd_gates_stamp(a):
    shutil.copyfile(a.xml, os.path.join(a.base, GATES_XML))
    write_json(os.path.join(a.base, GATES), {"key": gates_key(a), "recorded": time.strftime("%Y-%m-%d %H:%M")})
    print("BASELINE gates recorded for the next run")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build-check", cmd_build_check), ("build-stamp", cmd_build_stamp),
                     ("gates-check", cmd_gates_check), ("gates-stamp", cmd_gates_stamp)):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True)
        p.add_argument("--cmake-args", default=os.environ.get("CMAKE_ARGS", ""))
        if name.startswith("gates"):
            p.add_argument("--re", required=True); p.add_argument("--snap", required=True)
        if name == "gates-check":
            p.add_argument("--out", required=True)
        if name == "gates-stamp":
            p.add_argument("--xml", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.base = os.path.abspath(a.base)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
