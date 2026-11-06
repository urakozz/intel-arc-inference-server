#!/usr/bin/env python3
"""A test's PASS kept while nothing it read has changed (stdlib only: the box's host python3 and the Mac).

  reuse.py record --tests JSON --junit FILE --trace DIR --since FILE [--tree DIR] [--store DIR] [--from NAME]
        every PASS in FILE whose trace (tools/test_launcher.sh, B70_TRACE_DIR=DIR) was written in this run
        (after FILE --since) becomes a store entry: the test's command, properties and environment, and
        every file it - or a process it started - opened, stat'ed, exec'ed or found missing, each with its
        state now: files of the tree by content, files elsewhere (model snapshots, reference outputs, the
        driver's libraries) by real path, size and mtime, missing ones as missing
  reuse.py claim  --tests JSON --re ERE --out FILE [--tree DIR] [--store DIR] [--state DIR]
        every selected test with a store entry whose inputs all still match goes into FILE (JUnit, the old
        output behind a REUSED line), so junit.py todo leaves it out of the ctest run and pick counts it;
        nothing claimed with BV_NO_REUSE=1 (box_validate.sh --no-reuse) or a file named no-reuse in --state
  reuse.py build-id [--tree DIR]
        the build's identity: the tree's sources, the build's cache settings, the compiler

What counts as the same build output: a file under build/ matches when its bytes do (the same tree, a
later commit that left that binary alone; a device binary always, a new driver compiler changes those
with no source changed), or - a host program only - when the entry came from another tree built from the
same sources with the same settings and compiler (box_check's tree vs the validation tree: the programs
differ only by the tree path compiled into them). A test is never recorded when it changed directory, used a path
the trace cannot resolve, or read a file that changed while the run was going; files a test created
itself are its outputs, not inputs. Only PASS is kept: a failure or a skip always runs again.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

STORE = os.path.expanduser("~/b70/test-results")
KEEP = 4    # entries per test: the newest first
IGNORED = ("/proc/", "/sys/", "/dev/", "/run/", "/tmp/", "/var/tmp/")
SKIP_DIRS = {".git", "__pycache__", ".idea", ".cache"}
TREE_TAG = "T:"

LINE = re.compile(r"^(\d+)\s+(.*)$")
CALL = re.compile(r"^([a-z0-9_]+)\((.*)\)\s+=\s+(-?\d+|\?)(?:\s+([A-Z0-9_]+))?")
STR = re.compile(r'"((?:[^"\\]|\\.)*)"(\.\.\.)?')
WRITES = {"truncate", "rename", "renameat", "renameat2", "unlink", "unlinkat", "mkdir", "mkdirat",
          "symlink", "symlinkat", "link", "linkat", "creat"}
DIRFD_CALLS = {"openat", "openat2", "execveat", "newfstatat", "statx", "faccessat", "faccessat2", "readlinkat",
               "renameat", "renameat2", "unlinkat", "mkdirat", "linkat"}


class Unusable(Exception):
    pass


def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(c); i += 1; continue
        n = s[i + 1] if i + 1 < len(s) else ""
        simple = {"n": "\n", "t": "\t", "r": "\r", "v": "\v", "f": "\f", '"': '"', "\\": "\\"}
        if n in simple:
            out.append(simple[n]); i += 2
        elif n == "x":
            out.append(chr(int(s[i + 2:i + 4], 16))); i += 4
        elif n.isdigit():
            j = i + 1
            while j < len(s) and j < i + 4 and s[j] in "01234567":
                j += 1
            out.append(chr(int(s[i + 1:j], 8))); i = j
        else:
            out.append(n); i += 2
    return "".join(out)


def sha_file(path, cache={}):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)     # ctime: an edit that kept the mtime
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        cache[key] = h.hexdigest()
    return cache[key]


def under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


# ---- the trace -----------------------------------------------------------------------------------------
def parse_trace(path, cwd):
    """-> ordered [(event, abs_path)], the files the run touched: open (opened to read: a directory
    opened is listed), read (exec'ed, stat'ed, checked, readlink'ed), missing (looked for, not there),
    write (created, truncated, renamed, removed, linked: an output). Raises Unusable when a path cannot
    be resolved."""
    pending, fds, events = {}, {}, []
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for raw in f:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            pid, rest = m.group(1), m.group(2)
            if rest.endswith("<unfinished ...>"):
                pending[pid] = rest[:-len("<unfinished ...>")].rstrip()
                continue
            r = re.match(r"^<\.\.\. ([a-z0-9_]+) resumed>(.*)$", rest)
            if r:
                rest = pending.pop(pid, r.group(1) + "(") + r.group(2)
            if rest.startswith(("+++", "---")):
                continue
            c = CALL.match(rest)
            if not c:
                continue
            name, args, ret, err = c.group(1), c.group(2), c.group(3), c.group(4)
            strs = [(unescape(s.group(1)), bool(s.group(2))) for s in STR.finditer(args)]
            if not strs:
                continue
            if any(trunc for _, trunc in strs[:2]):
                raise Unusable(f"a path longer than the trace keeps ({name})")
            dirfd = None
            if name in DIRFD_CALLS:
                dirfd = args.split(",", 1)[0].strip()
            elif name == "symlinkat":                # symlinkat(target, newdirfd, linkpath)
                dirfd = args.split('"')[2].strip(" ,")
            ok = ret != "?" and int(ret) >= 0

            def resolve(p):
                if p == "":
                    return None                      # statx(fd, "", AT_EMPTY_PATH): the fd's own file
                if p.startswith("/"):
                    return os.path.normpath(p)
                if dirfd is None or dirfd == "AT_FDCWD":
                    return os.path.normpath(os.path.join(cwd, p))
                base = fds.get((pid, dirfd))
                if base is None:
                    raise Unusable(f"{name} relative to fd {dirfd} not opened in the trace")
                return os.path.normpath(os.path.join(base, p))

            if name in ("chdir", "fchdir"):
                raise Unusable("the test changed directory")
            if name in ("rename", "renameat", "renameat2", "link", "linkat"):
                for p, _ in strs[:2]:
                    a = resolve(p)
                    if a:
                        events.append(("write", a))
                continue
            if name in ("symlink", "symlinkat"):
                a = resolve(strs[-1][0])
                if a:
                    events.append(("write", a))
                continue
            p = resolve(strs[0][0])
            if p is None:
                continue
            if name in WRITES:
                events.append(("write", p))
                continue
            if name in ("open", "openat", "openat2"):
                flags = args.split('"')[-1]
                creates = any(fl in flags for fl in ("O_WRONLY", "O_CREAT", "O_TRUNC"))
                if ok:
                    fds[(pid, ret)] = p
                    events.append(("write" if creates else "open", p))
                elif not creates:
                    events.append(("missing", p))
                continue
            # exec, stat, access, readlink
            events.append(("read" if ok else "missing", p))
    return events


def inputs(events, ignore):
    """-> {path: 'read' | 'open' | 'missing'}: what the run needed of each path before it wrote it (a path it
    created first is its output, not an input); an open beats a read (a directory opened is listed)."""
    first, written = {}, set()
    for ev, p in events:
        if any(under(p, i.rstrip("/")) for i in ignore) or p.startswith(IGNORED):
            continue
        if ev == "write":
            if p not in first:
                written.add(p)
            continue
        if p in written or any(under(p, w) for w in written):
            continue
        if p not in first or (ev == "open" and first[p] == "read"):
            first[p] = ev
    return first


# ---- the state of one input ----------------------------------------------------------------------------
def lexical(p, tree):
    return TREE_TAG + os.path.relpath(p, tree) if under(p, tree) else p


def resolve_lex(lex, tree):
    return os.path.normpath(os.path.join(tree, lex[len(TREE_TAG):])) if lex.startswith(TREE_TAG) else lex


def state_of(lex, tree, listed=False):
    """the input's state now: [kind, ...]; a directory by its entries when it was listed (opened)"""
    p = resolve_lex(lex, tree)
    if not os.path.lexists(p):
        return ["missing"]
    real = os.path.realpath(p)
    treal = os.path.realpath(tree)
    if not os.path.exists(real):
        return ["dangling", os.readlink(p)]
    if os.path.isdir(real):
        if not listed:
            return ["dir"]
        return ["dirlist", hashlib.sha256("\n".join(sorted(os.listdir(real))).encode()).hexdigest()]
    if under(real, treal):
        rel = os.path.relpath(real, treal)
        kind = "build" if rel.split(os.sep)[0].startswith("build") else "src"
        return [kind, rel, sha_file(real)]
    st = os.stat(real)
    return ["ext", real, st.st_size, st.st_mtime_ns]


BUILD_INPUTS = ("CMakeLists.txt", "cmake", "src", "tests", "third_party", "tools")


def is_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def build_id(tree):
    """what the build is made from: the files of BUILD_INPUTS (the documentation and anything else at the top -
    a stray note, the run's own files - left out), the build's cache settings, the compiler's version"""
    tree = os.path.realpath(tree)
    h = hashlib.sha256()
    walk = []
    for top in BUILD_INPUTS:
        p = os.path.join(tree, top)
        if os.path.isfile(p):
            h.update(f"F {top} {sha_file(p)}\n".encode())
        elif os.path.isdir(p) and not os.path.islink(p):
            walk.append(p)
    for d, dirs, files in (w for top in walk for w in os.walk(top)):
        rel_d = os.path.relpath(d, tree)
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not os.path.islink(os.path.join(d, x)))
        for fn in sorted(files):
            fp = os.path.join(d, fn)
            if fn.endswith((".pyc", ".md")):
                continue
            rel = os.path.normpath(os.path.join(rel_d, fn))
            if os.path.islink(fp):
                h.update(f"L {rel} {os.readlink(fp)}\n".encode())
            elif os.path.isfile(fp):
                h.update(f"F {rel} {sha_file(fp)}\n".encode())
    cache = os.path.join(tree, "build", "CMakeCache.txt")
    compiler = ""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8", errors="replace") as f:
            for line in f:
                if re.match(r"^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_EXE_LINKER_FLAGS|"
                            r"CMAKE_CACHE_(MAJOR|MINOR|PATCH)_VERSION|B70_|OCLOC|ocloc)", line):
                    h.update(line.replace(tree, "<T>").encode())
                    if line.startswith("CMAKE_CXX_COMPILER:"):
                        compiler = line.split("=", 1)[1].strip()
    if compiler:
        try:
            h.update(subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.encode())
        except OSError:
            pass
    return h.hexdigest()


def machine():
    return {"kernel": platform.release()}


# ---- the test's own settings ---------------------------------------------------------------------------
def test_defs(tests_json, tree):
    with open(tests_json, encoding="utf-8") as f:
        data = json.load(f)
    tree_r = os.path.realpath(tree)
    out = {}
    for t in data.get("tests", []):
        cmd = [c.replace(tree_r, "<T>").replace(tree, "<T>") for c in t.get("command", []) or []]
        if cmd and cmd[0].endswith("tools/test_launcher.sh"):
            cmd = cmd[1:]
        props, env = {}, {}
        for p in t.get("properties", []) or []:
            n, v = p.get("name"), p.get("value")
            if n == "LABELS":
                continue
            if n == "ENVIRONMENT":
                for kv in (v if isinstance(v, list) else [v]):
                    k, _, val = kv.partition("=")
                    if k != "B70_TEST_NAME":
                        env[k] = val.replace(tree_r, "<T>").replace(tree, "<T>")
                continue
            props[n] = json.loads(json.dumps(v).replace(tree_r, "<T>").replace(tree, "<T>"))
        out[t["name"]] = {"command": cmd, "props": props, "env": env}
    return out


ENV_RX = re.compile(r"^(B70_|ZE_|HF_|ONEAPI_|SYCL_|IGC_|NEO|OCL_|LD_LIBRARY_PATH$|LD_PRELOAD$)")
# B70_GIT_SHA only labels b70-decode's printed bench rows (remote.sh exports the commit): never a test's result.
ENV_DROP = {"B70_TRACE_DIR", "B70_TEST_NAME", "B70_GIT_SHA"}


def env_norm(e, tree):
    """tree paths as <T>; ZE_AFFINITY_MASK as the number of cards it shows (card 0 or card 1 alone is the same
    test, one card or two is not)"""
    out = {k: v.replace(os.path.realpath(tree), "<T>").replace(tree, "<T>") for k, v in e.items()
           if ENV_RX.match(k) and k not in ENV_DROP}
    if "ZE_AFFINITY_MASK" in out:
        out["ZE_AFFINITY_MASK"] = f"{len([c for c in out['ZE_AFFINITY_MASK'].split(',') if c.strip()])} card(s)"
    return out


def env_now(props_env, tree):
    """the environment the test would see, as the launcher records it"""
    e = dict(os.environ)
    for k, v in props_env.items():
        e[k] = v.replace("<T>", tree)
    return env_norm(e, tree)


def env_recorded(path, tree):
    e = {}
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            if k:
                e[k] = v
    return env_norm(e, tree)


# ---- the store -----------------------------------------------------------------------------------------
def store_path(store, name):
    return os.path.join(store, name + ".json")


def load(store, name):
    try:
        with open(store_path(store, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


class Lock:
    def __init__(self, store):
        os.makedirs(store, exist_ok=True)
        self.f = open(os.path.join(store, ".lock"), "w")

    def __enter__(self):
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)


def save(store, name, entry):
    entries = [entry] + [e for e in load(store, name) if e.get("inputs") != entry["inputs"]
                         or e.get("build_id") != entry["build_id"]]
    tmp = store_path(store, name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries[:KEEP], f)
    os.replace(tmp, store_path(store, name))


# ---- record --------------------------------------------------------------------------------------------
def cmd_record(a):
    tree = os.path.abspath(a.tree)
    defs = test_defs(a.tests, tree)
    since = os.path.getmtime(a.since)
    try:
        root = ET.parse(a.junit).getroot()
    except (ET.ParseError, OSError) as e:
        print(f"REUSE nothing recorded: cannot read {a.junit}: {e}")
        return 0
    bid, mach = build_id(tree), machine()
    ignore = [os.path.abspath(a.trace)] + [os.path.abspath(x) for x in a.ignore or []]
    n, why = 0, {}
    with Lock(a.store):
        for tc in root.iter("testcase"):
            name = tc.get("name")
            if tc.get("status") != "run" or tc.find("failure") is not None or name not in defs:
                continue
            base = os.path.join(a.trace, name)
            try:
                for ext in (".trace", ".env", ".cwd"):
                    if not os.path.exists(base + ext) or os.path.getmtime(base + ext) < since:
                        raise Unusable("no trace from this run")
                started = os.path.getmtime(base + ".cwd")
                with open(base + ".cwd", encoding="utf-8") as f:
                    cwd = f.read().strip()
                ins = inputs(parse_trace(base + ".trace", cwd), ignore)
                states = {}
                for p, ev in sorted(ins.items()):
                    lex = lexical(p, tree)
                    st = state_of(lex, tree, listed=ev == "open")
                    if ev == "missing" and st[0] != "missing":
                        raise Unusable(f"{p} appeared during the run")
                    if ev == "read" and st[0] == "missing":
                        raise Unusable(f"{p} went away during the run")
                    if st[0] != "missing" and os.path.lexists(p):
                        real = os.path.realpath(p)
                        if os.path.exists(real) and not os.path.isdir(real) \
                                and os.stat(real).st_mtime >= started - 1:
                            raise Unusable(f"{p} changed during the run")
                    states[lex] = [ev == "open"] + st
                d = defs[name]
                entry = {"name": name, "command": d["command"], "props": d["props"],
                         "env": env_recorded(base + ".env", tree), "machine": mach, "build_id": bid,
                         "inputs": states, "junit": ET.tostring(tc, encoding="unicode"),
                         "from": a.source or os.path.basename(os.path.dirname(os.path.abspath(a.junit))),
                         "tree": tree, "date": time.strftime("%Y-%m-%d %H:%M"), "time": tc.get("time")}
                save(a.store, name, entry)
                n += 1
            except Unusable as e:
                why[name] = str(e)
            except OSError as e:
                why[name] = f"{type(e).__name__}: {e}"
    print(f"REUSE recorded {n} PASS(es) from {os.path.basename(a.junit)}; not recordable {len(why)}")
    for name, w in sorted(why.items())[:20]:
        print(f"REUSE   not recorded {name}: {w}")
    return 0


# ---- claim ---------------------------------------------------------------------------------------------
def matches(entry, d, tree, bid, mach):
    """-> None when the entry stands for this test now, else what differs"""
    if entry.get("machine") != mach:
        return f"the machine ({entry.get('machine')} -> {mach})"
    if entry.get("command") != d["command"]:
        return "the command line"
    if entry.get("props") != d["props"]:
        return "the test's properties"
    if entry.get("env") != env_now(d["env"], tree):
        return "the environment"
    # A host program of another tree built from the same sources differs only by that tree's path compiled in:
    # it counts as the same. Everything else under build/ - the device binaries, which carry no path and which
    # a new driver compiler changes without any source changing - must match byte for byte.
    path_only = entry.get("build_id") == bid and os.path.realpath(entry.get("tree", "")) != os.path.realpath(tree)
    for lex, (listed, *old) in entry.get("inputs", {}).items():
        now = state_of(lex, tree, listed=listed)
        if now != old and path_only and old[0] == "build" and now[0] == "build" and old[1] == now[1] \
                and not old[1].split(os.sep)[1:2] == ["kernels"] and is_elf(resolve_lex(lex, tree)):
            continue
        if now != old:
            return f"{lex[len(TREE_TAG):] if lex.startswith(TREE_TAG) else lex} ({old[0]} -> {now[0]})"
    return None


def cmd_claim(a):
    tree = os.path.abspath(a.tree)
    if os.environ.get("BV_NO_REUSE") == "1" or (a.state and os.path.exists(os.path.join(a.state, "no-reuse"))):
        print("REUSE off in this run (--no-reuse)")
        return 0
    defs = test_defs(a.tests, tree)
    rx = re.compile(a.re)
    want = [n for n in defs if rx.search(n)]
    if not want:
        return 0
    bid, mach = build_id(tree), machine()
    claimed, misses = [], {}
    for name in want:
        entries = load(a.store, name)
        if not entries:
            misses[name] = "never recorded"
            continue
        why = None
        for e in entries:
            why = matches(e, defs[name], tree, bid, mach)
            if why is None:
                claimed.append((name, e))
                break
        if why is not None:
            misses[name] = why
    if claimed:
        suite = ET.Element("testsuite", name="reused", tests=str(len(claimed)), failures="0")
        for name, e in claimed:
            tc = ET.fromstring(e["junit"])
            out = tc.find("system-out")
            if out is None:
                out = ET.SubElement(tc, "system-out")
            out.text = (f"REUSED: every input identical to the PASS of {e['date']} ({e['from']})\n"
                        + (out.text or ""))
            suite.append(tc)
        ET.ElementTree(suite).write(a.out, encoding="utf-8", xml_declaration=True)
    print(f"REUSE claimed {len(claimed)} of {len(want)} (their inputs are unchanged since a PASS); "
          f"{len(misses)} run")
    for name, e in claimed:
        print(f"REUSE   reused {name}: {e['from']} {e['date']} ({e.get('time')} s saved)")
    for name, w in sorted(misses.items()):
        print(f"REUSE   runs {name}: {w}")
    return 0


def main(argv=None):
    sha_file.__defaults__[0].clear()            # one call's view of the files (the tests call main in-process)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--tests", required=True); p.add_argument("--junit", required=True)
    p.add_argument("--trace", required=True); p.add_argument("--since", required=True)
    p.add_argument("--ignore", action="append"); p.add_argument("--from", dest="source")
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("claim")
    p.add_argument("--tests", required=True); p.add_argument("--re", required=True)
    p.add_argument("--out", required=True); p.add_argument("--state")
    p.set_defaults(fn=cmd_claim)
    p = sub.add_parser("build-id")
    p.set_defaults(fn=lambda a: print(build_id(os.path.abspath(a.tree))) or 0)
    for p in sub.choices.values():
        p.add_argument("--tree", default=".")
        p.add_argument("--store", default=STORE)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
