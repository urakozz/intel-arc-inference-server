#!/usr/bin/env python3
"""Hot / cold / archive tiers of the box's Hugging Face model cache (tools/hf_tier.sh runs this).

  hot      ~/.cache/huggingface/hub           the NVMe: the working set, real folders
  cold     ~/b70/hdd/hf-hub                   the 8 TB disk: every model, always
  archive  ~/b70/hdd3/hf-hub                  the 3 TB disk: parked models (instead of cold)

The cache entry `<hot>/models--X` is a real folder (hot) or a symlink to `<cold>/models--X` or
`<archive>/models--X`. Every command copies first, verifies, then swaps the entry; what it
replaces goes to that disk's trash (`<disk>/trash/<date>/hf-tier/`) for the operator to delete.
Nothing is ever deleted here, except an empty folder (a former bind-mount point) and the entry's
own symlink when it is replaced.

Commands (a model is `models--Org--Name`, `datasets--Org--Name` or `Org/Name`):
  status            every model: tier, size, where it lives, which stores hold a copy; free space
  hot <model>       copy the live copy (cold or archive) to the NVMe, verify, entry = real folder
  cold <model>      make sure the 8 TB copy holds everything the live copy has (copy, verify),
                    entry = link to it; a hot folder goes to the NVMe trash
  archive <model>   the same with the 3 TB disk; the 8 TB copy (if any) goes to the 8 TB trash
  sync <model>      only the copy + verify into the 8 TB store; the entry is not touched

Copies are self-contained: a per-repo blob that is a link into a cache-wide shared blob store
(huggingface_hub >= 2.x, `<cache>/blobs/<2 hex>/<xet hash>` + `.refs` manifests) is copied as
a regular file; links inside the model folder (snapshots -> ../../blobs/<etag>) stay links.
When a hot folder leaves, the shared payloads nothing else on the NVMe still links to go to the
trash too (with their .refs / .lock), so no link dangles and the space really can be freed.

Verify: every file of the source resolves to a file of the same size at the same path in the
target; a blob named by its sha256 (LFS etag, 64 hex) - or a snapshot path linking to one - is
hashed on the target and must equal its name; any other file is compared byte for byte. Every
link in the target must resolve, inside the target's own disk store. Re-reads skip the page
cache (fsync + POSIX_FADV_DONTNEED) so the check reads the disk, not the copy's memory.

Merging into a copy that already exists (the old USB copies): files the target lacks are added;
a differing file under 64 MiB (refs/main, a local config edit) takes the live copy's version and
the target's old one goes to the target disk's trash; a differing larger file stops the command.
A blob the target lacks is skipped when every path linking to it is a real file in the target
already (the USB copies keep snapshots as real files).

Refuses a model any process has open (/proc/<pid>/fd and /proc/<pid>/maps; mmap'd shards count).
One command at a time (flock on <hot>/.hf-tier.lock). Interrupted copies resume: staging folders
(<store>/.hf-tier-staging/<model>) are reused and files are written as *.hf-tier-part then renamed.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import os
import re
import shutil
import stat
import sys
import time

SMALL = 64 << 20
SHA_RE = re.compile(r"[0-9a-f]{64}")
PART = ".hf-tier-part"
STAGING = ".hf-tier-staging"


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def die(msg, code=1):
    print("hf_tier: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def gb(n):
    return "%.1f GB" % (n / 1e9)


def model_name(arg):
    a = arg.rstrip("/").split("/")[-1] if arg.startswith(("models--", "datasets--")) else arg
    if a.startswith(("models--", "datasets--")):
        return a
    if a.startswith("datasets/"):
        return "datasets--" + a[len("datasets/"):].replace("/", "--")
    return "models--" + a.replace("/", "--")


def inside(path, root):
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


class Stores:
    def __init__(self, a):
        home = os.path.expanduser("~")
        self.hot = os.path.abspath(a.hot or os.path.join(home, ".cache/huggingface/hub"))
        self.cold = os.path.abspath(a.cold or os.path.join(home, "b70/hdd/hf-hub"))
        self.archive = os.path.abspath(a.archive or os.path.join(home, "b70/hdd3/hf-hub"))
        date = a.date or datetime.date.today().isoformat()
        hot_trash = a.hot_trash or os.path.join(home, "b70/trash")
        self.trash = {
            "hot": os.path.join(hot_trash, date, "hf-tier"),
            "cold": os.path.join(os.path.dirname(self.cold), "trash", date, "hf-tier"),
            "archive": os.path.join(os.path.dirname(self.archive), "trash", date, "hf-tier"),
        }
        self.min_free = int(a.min_free_gb * 1e9)
        self.jobs = a.jobs

    def root(self, tier):
        return {"hot": self.hot, "cold": self.cold, "archive": self.archive}[tier]

    def tier_of(self, path):
        """Which store a real path lives in (None: elsewhere)."""
        rp = os.path.realpath(path)
        for t in ("cold", "archive", "hot"):
            r = os.path.realpath(self.root(t)) if os.path.isdir(self.root(t)) else None
            if r and inside(rp, r):
                return t
        return None


# ---------------------------------------------------------------- inspecting a model folder

def walk(top):
    """(rel, kind) for every entry under top, not following directory links. kind: d / f / l."""
    for root, dirs, files in os.walk(top):
        rel_root = os.path.relpath(root, top)
        rel_root = "" if rel_root == "." else rel_root
        keep = []
        for d in dirs:
            p = os.path.join(root, d)
            if os.path.islink(p):
                files.append(d)
            else:
                keep.append(d)
                yield os.path.join(rel_root, d), "d"
        dirs[:] = keep
        for f in files:
            p = os.path.join(root, f)
            yield os.path.join(rel_root, f), ("l" if os.path.islink(p) else "f")


def describe(folder, store_root):
    """Facts about one model folder: files (logical path -> real path), dangling links, sizes.

    real size counts each distinct real file once; `shared` the part that lives in the store's
    cache-wide blob folder; `outside` links that leave the store altogether.
    """
    info = {"files": {}, "dangling": [], "incomplete": 0, "size": 0, "shared": 0,
            "shared_paths": set(), "outside": [], "empty": True, "ref": None}
    if not os.path.isdir(folder):
        return info
    real_folder = os.path.realpath(folder)
    shared_root = os.path.join(os.path.realpath(store_root), "blobs")
    seen = set()
    for rel, kind in walk(folder):
        if kind == "d":
            continue
        info["empty"] = False
        p = os.path.join(folder, rel)
        if rel.endswith(".incomplete") or rel.endswith(PART):
            info["incomplete"] += 1
            continue
        rp = os.path.realpath(p)
        if not os.path.isfile(rp):
            info["dangling"].append(rel)
            continue
        info["files"][rel] = rp
        if rp in seen:
            continue
        seen.add(rp)
        sz = os.path.getsize(rp)
        info["size"] += sz
        if inside(rp, shared_root):
            info["shared"] += sz
            info["shared_paths"].add(rp)
        elif not inside(rp, real_folder) and not inside(rp, os.path.realpath(store_root)):
            info["outside"].append(rel)
    try:
        with open(os.path.join(folder, "refs", "main")) as f:
            info["ref"] = f.read().strip()
    except OSError:
        pass
    return info


def entry_state(st, name):
    """(state, live folder or None). state: hot / cold / archive / empty / missing / broken."""
    e = os.path.join(st.hot, name)
    if os.path.islink(e):
        t = os.path.realpath(e)
        if not os.path.isdir(t):
            return "broken", None
        return (st.tier_of(t) or "elsewhere"), t
    if os.path.isdir(e):
        if any(True for _ in os.scandir(e)):
            return "hot", e
        return "empty", None
    return "missing", None


def nonempty_dir(p):
    return os.path.isdir(p) and not os.path.islink(p) and any(True for _ in os.scandir(p))


def copies(st, name):
    out = {}
    for t in ("hot", "cold", "archive"):
        p = os.path.join(st.root(t), name)
        if nonempty_dir(p):
            out[t] = p
    return out


# ---------------------------------------------------------------- open-file check

def open_users(paths):
    """pids (with command names) that hold any file under one of `paths` open or mapped."""
    prefixes = [os.path.realpath(p) for p in paths if p]
    hits = {}
    if not os.path.isdir("/proc/self/fd"):
        log("warning: no /proc here - the open-file check is skipped")
        return hits
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        found = None
        try:
            fd_dir = "/proc/%s/fd" % pid
            for fd in os.listdir(fd_dir):
                try:
                    t = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if any(inside(t, p) for p in prefixes):
                    found = t
                    break
            if not found:
                with open("/proc/%s/maps" % pid) as f:
                    for line in f:
                        parts = line.split(None, 5)
                        if len(parts) == 6:
                            t = parts[5].strip()
                            if any(inside(t, p) for p in prefixes):
                                found = t
                                break
        except (OSError, PermissionError):
            continue
        if found:
            try:
                with open("/proc/%s/comm" % pid) as f:
                    comm = f.read().strip()
            except OSError:
                comm = "?"
            hits[int(pid)] = (comm, found)
    return hits


def refuse_if_open(paths):
    hits = open_users(paths)
    if hits:
        for pid, (comm, f) in sorted(hits.items()):
            print("  pid %d (%s) has %s open" % (pid, comm, f), file=sys.stderr)
        die("refused: the model's files are in use (see above); stop the job or wait", 3)


# ---------------------------------------------------------------- copying and verifying

def drop_cache(fd):
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


_SHA_CACHE = {}


def sha256_file(path):
    st_ = os.stat(path)
    key = (os.path.realpath(path), st_.st_size, st_.st_mtime_ns, st_.st_ino)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = _sha256_read(path)
    return _SHA_CACHE[key]


def _sha256_read(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        drop_cache(f.fileno())
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def same_bytes(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        drop_cache(fa.fileno())
        drop_cache(fb.fileno())
        while True:
            x = fa.read(16 << 20)
            y = fb.read(16 << 20)
            if x != y:
                return False
            if not x:
                return True


def copy_file(src_real, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + PART
    shutil.copyfile(src_real, tmp)
    shutil.copystat(src_real, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
        drop_cache(f.fileno())
    os.replace(tmp, dst)


def sha_name_of(rel, src_folder):
    """The sha256 a logical path's content must have, from its blob name, else None."""
    p = os.path.join(src_folder, rel)
    name = os.path.basename(rel)
    if rel.startswith("blobs/") and SHA_RE.fullmatch(name):
        return name
    if os.path.islink(p):
        t = os.readlink(p)
        tn = os.path.basename(t)
        if "/blobs/" in "/" + t and SHA_RE.fullmatch(tn) and not os.path.isabs(t):
            # snapshot -> ../../blobs/<etag>: the etag is the sha256 for LFS files
            if inside(os.path.normpath(os.path.join(os.path.dirname(p), t)),
                      os.path.normpath(src_folder)):
                return tn
    return None


def internal_link(src_folder, rel):
    """The link text if `rel` is a relative link that stays inside the model folder."""
    p = os.path.join(src_folder, rel)
    if not os.path.islink(p):
        return None
    t = os.readlink(p)
    if os.path.isabs(t):
        return None
    target = os.path.normpath(os.path.join(os.path.dirname(p), t))
    if inside(target, os.path.normpath(src_folder)) and os.path.isfile(os.path.realpath(p)):
        return t
    return None


def blob_of(src, rel, link):
    return os.path.relpath(os.path.normpath(os.path.join(src, os.path.dirname(rel), link)), src)


def plan_and_copy(st, src, dst, dst_tier):
    """Make dst hold everything src has. Returns the logical paths to verify."""
    trash = os.path.join(st.trash[dst_tier], "replaced", os.path.basename(dst.rstrip("/")))
    entries = list(walk(src))
    blob_users = {}   # blob rel -> logical paths that link to it
    for rel, kind in entries:
        if kind == "l":
            t = internal_link(src, rel)
            if t:
                blob_users.setdefault(blob_of(src, rel, t), []).append(rel)
    needed = set()
    check = []
    big_conflicts = []
    copied = 0
    t0 = time.time()

    def handle_existing(rel, sreal):
        d = os.path.join(dst, rel)
        dreal = os.path.realpath(d)
        if not os.path.isfile(dreal):
            return False
        ss, ds = os.path.getsize(sreal), os.path.getsize(dreal)
        if ss <= SMALL and ds <= SMALL:
            if ss == ds and same_bytes(sreal, dreal):
                return True
            dest = os.path.join(trash, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.rename(d, dest)
            log("replaced %s (the target's old version -> %s)" % (rel, dest))
            return False
        if ss != ds:
            big_conflicts.append("%s: %d bytes here, %d bytes in the target" % (rel, ss, ds))
        return True

    # Pass 1: everything except blobs/ (blobs follow from what links to them).
    for rel, kind in entries:
        if rel.startswith("blobs/") or rel.endswith(".incomplete") or rel.endswith(PART):
            continue
        s = os.path.join(src, rel)
        d = os.path.join(dst, rel)
        if kind == "d":
            os.makedirs(d, exist_ok=True)
            continue
        sreal = os.path.realpath(s)
        if not os.path.isfile(sreal):
            die("the source has a dangling link: %s" % s)
        link = internal_link(src, rel) if kind == "l" else None
        if link and os.path.islink(d) and os.readlink(d) == link:
            needed.add(blob_of(src, rel, link))  # the same link (a resumed copy): its blob follows
            check.append(rel)
            continue
        if os.path.islink(d) and not os.path.isfile(os.path.realpath(d)):
            to_trash(st, dst_tier, d, os.path.join("replaced", os.path.basename(dst.rstrip("/")), os.path.dirname(rel)))
        if os.path.lexists(d) and handle_existing(rel, sreal):
            check.append(rel)
            continue
        if link:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.symlink(link, d)
            needed.add(blob_of(src, rel, link))
        else:
            copy_file(sreal, d)
            copied += os.path.getsize(sreal)
        check.append(rel)
    # Pass 2: blobs. A blob the target holds already under another name (an older snapshot kept
    # as real files) is hard-linked, not copied: same size and a sha256 equal to the blob's name.
    by_size = {}
    for rel, kind in walk(dst):
        if kind == "f" and not rel.endswith(PART):
            by_size.setdefault(os.path.getsize(os.path.join(dst, rel)), []).append(os.path.join(dst, rel))

    def link_existing(rel, size, d):
        name = os.path.basename(rel)
        if not SHA_RE.fullmatch(name):
            return False
        for c in by_size.get(size, []):
            if sha256_file(c) == name:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if os.path.lexists(d + PART):
                    os.unlink(d + PART)
                os.link(c, d + PART)
                os.replace(d + PART, d)
                log("%s: the target has it as %s - hard-linked" % (rel, os.path.relpath(c, dst)))
                return True
        return False

    for rel, kind in entries:
        if not rel.startswith("blobs/") or kind == "d" or rel.endswith(".incomplete") or rel.endswith(PART):
            continue
        users = blob_users.get(rel, [])
        if rel not in needed and users and all(
                os.path.isfile(os.path.join(dst, u)) and not os.path.islink(os.path.join(dst, u)) for u in users):
            continue  # the target holds this content as real snapshot files already
        s = os.path.join(src, rel)
        d = os.path.join(dst, rel)
        sreal = os.path.realpath(s)
        if not os.path.isfile(sreal):
            die("the source has a dangling link: %s" % s)
        if os.path.lexists(d) and handle_existing(rel, sreal):
            check.append(rel)
            continue
        if link_existing(rel, os.path.getsize(sreal), d):
            check.append(rel)
            continue
        log("copy %s (%s)" % (rel, gb(os.path.getsize(sreal))))
        copy_file(sreal, d)
        copied += os.path.getsize(sreal)
        check.append(rel)
    if big_conflicts:
        for c in big_conflicts:
            print("  differs: " + c, file=sys.stderr)
        die("refused: the target already has different large files (above); nothing swapped")
    dt = max(time.time() - t0, 1e-3)
    log("copied %s in %.0f s (%.0f MB/s)" % (gb(copied), dt, copied / dt / 1e6))
    return check


def verify(st, src, dst, rels, dst_store_root):
    """Every logical path: same size, sha256 = name for LFS blobs, else bytes equal."""
    bad = []
    dst_real = os.path.realpath(dst)
    store_real = os.path.realpath(dst_store_root)
    jobs = []
    for rel in rels:
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        sreal, dreal = os.path.realpath(s), os.path.realpath(d)
        if not os.path.isfile(dreal):
            bad.append("%s: missing in the target" % rel)
            continue
        if not inside(dreal, store_real):
            bad.append("%s: resolves outside the target's store (%s)" % (rel, dreal))
            continue
        if os.path.getsize(sreal) != os.path.getsize(dreal):
            bad.append("%s: size %d vs %d" % (rel, os.path.getsize(sreal), os.path.getsize(dreal)))
            continue
        jobs.append((rel, sreal, dreal, sha_name_of(rel, src)))
    done = {}

    def one(job):
        rel, sreal, dreal, sha = job
        key = (dreal, sha)
        if sha:
            got = sha256_file(dreal)
            return rel, (None if got == sha else "%s: sha256 %s, its name says %s" % (rel, got, sha))
        if os.path.samefile(sreal, dreal):
            return rel, None
        return rel, (None if same_bytes(sreal, dreal) else "%s: bytes differ" % rel)

    uniq = {}
    for j in jobs:
        uniq.setdefault((j[2], j[3]), j)
    total = sum(os.path.getsize(j[2]) for j in uniq.values())
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=st.jobs) as ex:
        for rel, err in ex.map(one, uniq.values()):
            if err:
                bad.append(err)
    dt = max(time.time() - t0, 1e-3)
    # every link in the target resolves inside the target's store
    for rel, kind in walk(dst):
        if kind != "l":
            continue
        r = os.path.realpath(os.path.join(dst, rel))
        if not os.path.isfile(r):
            bad.append("%s: dangling link in the target" % rel)
        elif not inside(r, store_real):
            bad.append("%s: link leaves the target's store (%s)" % (rel, r))
    log("verified %d files (%s read in %.0f s, %.0f MB/s): %s" % (
        len(uniq), gb(total), dt, total / dt / 1e6, "OK" if not bad else "%d FAILED" % len(bad)))
    if bad:
        for b in bad[:50]:
            print("  " + b, file=sys.stderr)
        die("verify failed; nothing swapped")


def ensure_copy(st, name, src, tier):
    """Copy + verify src into <tier>/<name> (merging into an existing copy). Returns its path."""
    root = st.root(tier)
    if not os.path.isdir(root):
        die("no %s store at %s (is the disk mounted?)" % (tier, root))
    final = os.path.join(root, name)
    if not os.path.islink(final) and os.path.realpath(src) == os.path.realpath(final):
        return final
    need = describe(src, st.root(st.tier_of(src) or "hot"))["size"]
    if nonempty_dir(final):
        dst = final
        log("%s: merging into the existing %s copy %s" % (name, tier, final))
    else:
        dst = os.path.join(root, STAGING, name)
        os.makedirs(dst, exist_ok=True)
        free = shutil.disk_usage(root).free
        have = describe(dst, root)["size"]
        reserve = st.min_free if tier == "hot" else 0
        if free + have < need + reserve:
            die("refused: %s needs %s, the %s disk has %s free (+%s staged)%s" % (
                name, gb(need), tier, gb(free), gb(have),
                (" and keeps %s free" % gb(reserve)) if reserve else ""), 4)
        log("%s: copying %s to %s" % (name, gb(need), dst))
    rels = plan_and_copy(st, src, dst, tier)
    verify(st, src, dst, rels, root)
    if dst != final:
        if os.path.lexists(final):
            # an empty folder or a stray link at the final name
            if os.path.islink(final):
                os.unlink(final)
            else:
                os.rmdir(final)
        os.rename(dst, final)
        try:
            os.rmdir(os.path.join(root, STAGING))
        except OSError:
            pass
    return final


# ---------------------------------------------------------------- swapping the cache entry

def to_trash(st, tier, path, sub):
    dest_dir = os.path.join(st.trash[tier], sub)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path))
    n = 1
    while os.path.lexists(dest):
        dest = os.path.join(dest_dir, "%s.%d" % (os.path.basename(path), n))
        n += 1
    os.rename(path, dest)
    log("moved %s -> %s" % (path, dest))
    return dest


def point_entry(st, name, target):
    """The cache entry becomes a symlink to target; a hot folder goes to the NVMe trash."""
    e = os.path.join(st.hot, name)
    tmp = os.path.join(st.hot, ".%s.hf-tier-link" % name)
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(target, tmp)
    retired = None
    if os.path.islink(e):
        os.unlink(e)
    elif os.path.isdir(e):
        if any(True for _ in os.scandir(e)):
            retired = to_trash(st, "hot", e, "nvme")
        else:
            os.rmdir(e)
    os.rename(tmp, e)
    log("%s -> %s" % (e, target))
    return retired


def hot_shared_refs(st):
    """Real paths of shared-store payloads that some folder in the hot store still links to."""
    refs = set()
    shared = os.path.join(os.path.realpath(st.hot), "blobs")
    for n in os.listdir(st.hot):
        p = os.path.join(st.hot, n)
        if not n.startswith(("models--", "datasets--")) or os.path.islink(p) or not os.path.isdir(p):
            continue
        b = os.path.join(p, "blobs")
        if not os.path.isdir(b):
            continue
        for f in os.listdir(b):
            q = os.path.join(b, f)
            if os.path.islink(q):
                r = os.path.realpath(q)
                if inside(r, shared):
                    refs.add(r)
    return refs


def sweep_shared(st, candidates):
    """Move shared payloads (and their .refs/.lock) nothing on the NVMe links to any more."""
    live = hot_shared_refs(st)
    freed = 0
    for p in sorted(candidates):
        if p in live or not os.path.isfile(p):
            continue
        sz = os.path.getsize(p)
        sub = os.path.join("nvme-shared-blobs", os.path.basename(os.path.dirname(p)))
        for extra in (p + ".refs", p + ".lock"):
            if os.path.lexists(extra):
                to_trash(st, "hot", extra, sub)
        to_trash(st, "hot", p, sub)
        freed += sz
    if freed:
        log("shared blob store: %s no longer linked from the NVMe -> trash" % gb(freed))


# ---------------------------------------------------------------- commands

def live_copy(st, name):
    state, live = entry_state(st, name)
    if live:
        return state, live
    c = copies(st, name)
    for t in ("cold", "archive"):
        if t in c:
            return state, c[t]
    die("%s: no copy anywhere (cache entry: %s)" % (name, state))


def cmd_place(st, name, tier):
    state, src = live_copy(st, name)
    sinfo = describe(src, st.root(st.tier_of(src) or "hot"))
    if sinfo["dangling"]:
        die("%s: the live copy %s has dangling links: %s" % (name, src, ", ".join(sinfo["dangling"][:5])))
    refuse_if_open([src] + sorted(sinfo["shared_paths"]))
    if tier == "hot":
        if state == "hot":
            log("%s: already hot" % name)
            return
        dst = ensure_copy(st, name, src, "hot")  # staging under <hot>/.hf-tier-staging, renamed in
        # ensure_copy renamed the staged folder onto the entry (it removed the old link/empty dir)
        log("%s: hot (%s)" % (name, dst))
        return
    if tier == "sync":
        ensure_copy(st, name, src, "cold")
        log("%s: the 8 TB copy is complete" % name)
        return
    if state == tier:
        log("%s: already %s" % (name, tier))
        return
    dst = ensure_copy(st, name, src, tier)
    shared = describe(os.path.join(st.hot, name), st.hot)["shared_paths"] if state == "hot" else set()
    point_entry(st, name, dst)
    if shared:
        sweep_shared(st, shared)
    if tier == "archive":
        c = os.path.join(st.cold, name)
        if os.path.isdir(c) and not os.path.islink(c):
            to_trash(st, "cold", c, "8tb")
    log("%s: %s (%s)" % (name, tier, dst))


def cmd_status(st):
    names = set()
    for t in ("hot", "cold", "archive"):
        r = st.root(t)
        if os.path.isdir(r):
            names.update(n for n in os.listdir(r) if n.startswith(("models--", "datasets--")))
    rows = []
    for n in sorted(names):
        state, live = entry_state(st, n)
        c = copies(st, n)
        size, note = 0, []
        if live:
            info = describe(live, st.root(st.tier_of(live) or "hot"))
            size = info["size"]
            if info["dangling"]:
                state = "broken"
                note.append("%d dangling" % len(info["dangling"]))
            if info["shared"]:
                note.append("%s in the shared blob store" % gb(info["shared"]))
            if info["incomplete"]:
                note.append("%d *.incomplete" % info["incomplete"])
            if info["outside"]:
                note.append("%d links leave the store" % len(info["outside"]))
        elif state in ("empty", "missing") and c:
            note.append("not visible: `cold`/`archive` links it")
            any_copy = c.get("cold") or c.get("archive")
            size = describe(any_copy, st.root(st.tier_of(any_copy)))["size"]
        if state == "hot" and "cold" not in c:
            note.append("no 8 TB copy")
        rows.append((n, state, size, ",".join(t for t in ("hot", "cold", "archive") if t in c), "; ".join(note)))
    w = max([len(r[0]) for r in rows] + [5])
    print("%-*s  %-8s %10s  %-17s %s" % (w, "model", "tier", "size", "copies", "notes"))
    for r in rows:
        print("%-*s  %-8s %10s  %-17s %s" % (w, r[0], r[1], gb(r[2]) if r[2] else "-", r[3] or "-", r[4]))
    for t in ("hot", "cold", "archive"):
        r = st.root(t)
        if os.path.isdir(r):
            u = shutil.disk_usage(r)
            print("%-8s %s: %s free of %s" % (t, r, gb(u.free), gb(u.total)))
        else:
            print("%-8s %s: not there (disk not mounted?)" % (t, r))
    for t in ("hot", "cold", "archive"):
        if os.path.isdir(st.trash[t]):
            print("trash    %s waits for the operator" % st.trash[t])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["status", "hot", "cold", "archive", "sync"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--hot", help="the NVMe cache (default ~/.cache/huggingface/hub)")
    ap.add_argument("--cold", help="the 8 TB store (default ~/b70/hdd/hf-hub)")
    ap.add_argument("--archive", help="the 3 TB store (default ~/b70/hdd3/hf-hub)")
    ap.add_argument("--hot-trash", help="the NVMe trash (default ~/b70/trash)")
    ap.add_argument("--date", help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=40.0,
                    help="NVMe space `hot` leaves free (default 40)")
    ap.add_argument("--jobs", type=int, default=3, help="parallel verify readers (default 3)")
    a = ap.parse_args()
    st = Stores(a)
    if not os.path.isdir(st.hot):
        die("no cache at %s" % st.hot)
    if a.cmd == "status":
        cmd_status(st)
        return
    if not a.models:
        die("%s needs a model" % a.cmd, 2)
    lock = open(os.path.join(st.hot, ".hf-tier.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        die("another hf_tier command is running", 2)
    for m in a.models:
        cmd_place(st, model_name(m), a.cmd)


if __name__ == "__main__":
    main()
