#!/usr/bin/env python3
"""A content-addressed cache in front of ocloc, shared by every tree on the box (cmake/ocloc.cmake runs each kernel
compile through it).

    ocloc_cache.py <toolchain stamp> <ocloc> compile -file F ... -out_dir D -options "..."
    ocloc_cache.py --stats | --zero-stats | --prune [GiB]   (evicted entries move to ~/b70/trash/<date>/ocloc-cache)

The key is the sha256 of: the compiler's identity (the stamp cmake/ocloc_stamp.cmake writes, plus the size and mtime
of ocloc, libocloc and the two IGC libraries), every argument but the output directory, the IGC_* environment, and
the source with every file an #include can reach - each name tried in the including file's directory and in every -I
directory, a missing candidate counted as missing, so a header that appears later changes the key. #if is ignored
(a header named under a branch not taken is still hashed). The build passes tree-relative paths and runs in the
tree's root (cmake/ocloc.cmake), so two trees with the same sources share every entry.

A hit copies the stored outputs into -out_dir and prints ocloc's stored stdout / stderr (its build log, spill
warnings included). A miss runs ocloc into a private directory, stores outputs + log on success and copies them out.
Whatever the key cannot be computed for (an #include by macro, __has_include, a -include style option, NEO debug
keys in the environment, no writable cache) runs ocloc as it is, uncached. B70_OCLOC_CACHE=0 does the same;
B70_OCLOC_CACHE_DIR moves the cache (default ~/b70/cache/ocloc). Entries are written to a temporary directory and
renamed into place, so concurrent jobs never see a half-written one.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

CACHE = os.environ.get("B70_OCLOC_CACHE_DIR") or os.path.expanduser("~/b70/cache/ocloc")
INCLUDE = re.compile(rb'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
DIRECTIVE = re.compile(rb'^\s*#\s*(include|include_next|import|embed)\b')


class Uncacheable(Exception):
    pass


def file_id(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.realpath(path)} {st.st_size} {st.st_mtime_ns}"


def toolchain(stamp: str, ocloc: str) -> bytes:
    with open(stamp, "rb") as f:
        text = f.read()
    ids = [file_id(shutil.which(ocloc) or ocloc)]
    # The stamp names libigc / libigdfcl by real path; libocloc is what the ocloc binary loads.
    paths = re.findall(rb": (/\S+)", text)
    ld = subprocess.run(["ldconfig", "-p"], capture_output=True).stdout
    paths += re.findall(rb"libocloc\.so[^\n]*=> (\S+)", ld)[:1]
    for p in paths:
        ids.append(file_id(p.decode()))
    return text + b"\n" + "\n".join(ids).encode()


def include_dirs(options: str) -> list:
    dirs, toks = [], options.split()
    for i, t in enumerate(toks):
        if t.startswith(("-include", "-imacros", "-iquote", "-isystem", "-idirafter", "-iprefix")):
            raise Uncacheable(t)
        if t == "-I":
            if i + 1 >= len(toks):
                raise Uncacheable("-I without a directory")
            dirs.append(toks[i + 1])
        elif t.startswith("-I"):
            dirs.append(t[2:])
    return dirs


def hash_sources(h, source: str, dirs: list) -> None:
    seen, todo = set(), [source]
    while todo:
        path = todo.pop()
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        with open(path, "rb") as f:
            data = f.read()
        h.update(b"file %s %s\n" % (path.encode(), hashlib.sha256(data).hexdigest().encode()))
        if b"__has_include" in data:
            raise Uncacheable(f"__has_include in {path}")
        for line in data.splitlines():
            if not DIRECTIVE.match(line):
                continue
            m = INCLUDE.match(line)
            if not m:   # #include_next / #import / #embed, or an #include by macro
                raise Uncacheable(f"{path}: {line.decode(errors='replace').strip()}")
            name = m.group(2).decode()
            if os.path.isabs(name):
                cands = [name]
            else:
                cands = [os.path.join(os.path.dirname(path), name)] + [os.path.join(d, name) for d in dirs]
            for c in cands:
                if os.path.isfile(c):
                    todo.append(c)
                else:
                    h.update(b"missing %s\n" % c.encode())


def compute_key(stamp: str, ocloc: str, args: list) -> str:
    if any(k == "NEOReadDebugKeys" for k in os.environ):
        raise Uncacheable("NEOReadDebugKeys is set")

    def value(flag):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise Uncacheable(f"{flag} not given exactly once")
        return args[args.index(flag) + 1]

    source, options = value("-file"), value("-options")
    value("-out_dir")
    h = hashlib.sha256(b"b70-ocloc-cache v1\n")
    h.update(toolchain(stamp, ocloc) + b"\n")
    keyed = list(args)
    keyed[keyed.index("-out_dir") + 1] = "<out>"
    h.update(b"args %s\n" % "\0".join([ocloc] + keyed).encode())
    for k in sorted(k for k in os.environ if k.startswith("IGC_")):
        h.update(b"env %s=%s\n" % (k.encode(), os.environ[k].encode()))
    hash_sources(h, source, include_dirs(options))
    return h.hexdigest()


def copy_out(src_dir: str, out_dir: str) -> None:
    for name in sorted(os.listdir(src_dir)):
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f".{name}.")
        os.close(fd)
        shutil.copyfile(os.path.join(src_dir, name), tmp)
        os.replace(tmp, os.path.join(out_dir, name))


def replay(entry: str) -> None:
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        with open(os.path.join(entry, name), "rb") as f:
            stream.flush()
            stream.buffer.write(f.read())
            stream.flush()


def count(what: str) -> None:
    try:
        fd = os.open(os.path.join(CACHE, "stats"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.write(fd, what.encode() + b"\n")
        os.close(fd)
    except OSError:
        pass


def compile_cached(stamp: str, ocloc: str, args: list) -> int:
    key = compute_key(stamp, ocloc, args)
    out_dir = args[args.index("-out_dir") + 1]
    entry = os.path.join(CACHE, key[:2], key)
    if os.path.isdir(entry):
        try:
            copy_out(os.path.join(entry, "out"), out_dir)
            replay(entry)
            os.utime(entry)   # the prune's LRU order
            count("hit")
            return 0
        except OSError:
            pass   # pruned under us: compile
    os.makedirs(os.path.join(CACHE, "tmp"), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.join(CACHE, "tmp"), prefix=key[:16] + ".")
    try:
        os.mkdir(os.path.join(tmp, "out"))
        run = list(args)
        run[run.index("-out_dir") + 1] = os.path.join(tmp, "out")
        p = subprocess.run([ocloc] + run, capture_output=True)
        for name, data in (("stdout", p.stdout), ("stderr", p.stderr)):
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
        replay(tmp)
        if p.returncode != 0:
            return p.returncode
        copy_out(os.path.join(tmp, "out"), out_dir)
        os.makedirs(os.path.dirname(entry), exist_ok=True)
        try:
            os.rename(tmp, entry)
            count("miss")
        except OSError:
            pass   # another job stored the same key first
        return 0
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def entries():
    for top in os.listdir(CACHE):
        d = os.path.join(CACHE, top)
        if len(top) == 2 and os.path.isdir(d):
            for key in os.listdir(d):
                e = os.path.join(d, key)
                size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(e) for f in fs)
                yield os.path.getmtime(e), size, e


def main() -> int:
    if sys.argv[1:2] == ["--stats"]:
        try:
            with open(os.path.join(CACHE, "stats")) as f:
                lines = f.read().split()
        except OSError:
            lines = []
        hits, misses = lines.count("hit"), lines.count("miss")
        ents = list(entries()) if os.path.isdir(CACHE) else []
        print(f"ocloc cache {CACHE}: {hits} hits, {misses} stored misses since the last --zero-stats; "
              f"{len(ents)} entries, {sum(e[1] for e in ents) / 2**30:.2f} GiB")
        return 0
    if sys.argv[1:2] == ["--zero-stats"]:
        open(os.path.join(CACHE, "stats"), "w").close()
        return 0
    if sys.argv[1:2] == ["--prune"]:
        # Nothing is deleted: evicted entries and stale temp folders move to the trash, which the owner empties.
        cap = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        trash = os.environ.get("B70_TRASH_DIR") or os.path.expanduser(f"~/b70/trash/{time.strftime('%Y-%m-%d')}")
        trash = os.path.join(trash, "ocloc-cache")
        os.makedirs(trash, exist_ok=True)

        def evict(path):
            dest = os.path.join(trash, os.path.basename(path))
            n = 1
            while os.path.exists(dest):
                dest, n = os.path.join(trash, f"{os.path.basename(path)}.{n}"), n + 1
            shutil.move(path, dest)

        tmp = os.path.join(CACHE, "tmp")
        for t in os.listdir(tmp) if os.path.isdir(tmp) else []:   # left by a killed compile
            if os.path.getmtime(os.path.join(tmp, t)) < time.time() - 86400:
                evict(os.path.join(tmp, t))
        ents = sorted(entries())
        total = sum(e[1] for e in ents)
        for _, size, e in ents:
            if total <= cap * 2**30:
                break
            evict(e)
            total -= size
        print(f"ocloc cache {CACHE}: {total / 2**30:.2f} GiB after pruning to {cap} GiB; evicted entries in {trash}")
        return 0
    stamp, ocloc, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    if os.environ.get("B70_OCLOC_CACHE", "1") != "0":
        try:
            os.makedirs(CACHE, exist_ok=True)
            return compile_cached(stamp, ocloc, args)
        except (Uncacheable, OSError) as e:
            print(f"ocloc cache: not cached ({e}), running ocloc directly", file=sys.stderr)
    os.execvp(ocloc, [ocloc] + args)


if __name__ == "__main__":
    sys.exit(main())
