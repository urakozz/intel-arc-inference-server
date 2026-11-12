#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the bf16 original completely downloaded? (tools/quantize/quantize_qwen4exp.sh check-download)

    check_download.py [--repo Qwen/Qwen3.8-Flash-Next] [--sha] [--jobs 4]

Compares the HF cache's snapshot of `--repo` (refs/main, as `hf download` leaves it) with the hub's own file
list at that revision: every file present, every size equal; with --sha also every LFS file's sha256 (the
hub's) - ~360 GB read, ~15-30 min on the box. Needs the hub (the file list and sizes); nothing is
downloaded. Prints `DOWNLOAD OK ...` (exit 0) or `DOWNLOAD NOT COMPLETE ...` with what is missing or wrong
(exit 1).
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--sha", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    from huggingface_hub.constants import HF_HUB_CACHE
    entry = os.path.join(HF_HUB_CACHE, "models--" + a.repo.replace("/", "--"))
    ref = os.path.join(entry, "refs", "main")
    if not os.path.exists(ref):
        sys.exit(f"DOWNLOAD NOT COMPLETE: {entry} has no refs/main - nothing downloaded yet")
    with open(ref, encoding="utf-8") as f:
        rev = f.read().strip()
    snap = os.path.join(entry, "snapshots", rev)
    info = HfApi().model_info(a.repo, revision=rev, files_metadata=True)
    want = {s.rfilename: s for s in info.siblings}
    errs, todo = [], []
    for name, s in sorted(want.items()):
        p = os.path.join(snap, name)
        if not os.path.exists(p):
            errs.append(f"missing {name}")
        elif os.path.getsize(p) != s.size:
            errs.append(f"{name}: {os.path.getsize(p)} bytes, the hub has {s.size}")
        elif a.sha and s.lfs:
            todo.append((name, p, s.lfs.sha256))
    if todo:
        print(f"hashing {len(todo)} files ({sum(want[n].size for n, _, _ in todo) / 1e9:.1f} GB)...", flush=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            for (name, _, ref_sha), got in zip(todo, ex.map(lambda t: sha256(t[1]), todo)):
                if got != ref_sha:
                    errs.append(f"{name}: sha256 {got[:16]}..., the hub has {ref_sha[:16]}...")
    total = sum(s.size or 0 for s in want.values())
    if errs:
        for e in errs[:20]:
            print("  " + e)
        print(f"DOWNLOAD NOT COMPLETE: {len(errs)} of {len(want)} files missing or wrong ({snap})")
        sys.exit(1)
    print(f"DOWNLOAD OK: {len(want)} files, {total / 1e9:.2f} GB, sizes{' and sha256' if a.sha else ''} equal "
          f"the hub's at {rev[:12]} ({snap})")


if __name__ == "__main__":
    main()
