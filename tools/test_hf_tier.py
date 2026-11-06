#!/usr/bin/env python3
"""Tests of tools/hf_tier.py on a fake hub in a temp dir: python3 tools/test_hf_tier.py

The fake NVMe cache has a cache-wide shared blob store (huggingface_hub 2.x: <hub>/blobs/<2 hex>/
<xet hash> + .refs, per-repo blobs linking into it), a repo-local model, an empty former bind
mount, and a model whose 8 TB copy keeps its snapshot as real files (the old USB copies).
Runs on the Mac and on the box (the open-file refusal only where /proc exists).
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "hf_tier.py")
SNAP = "0123456789abcdef0123456789abcdef01234567"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(p, b):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b)


def link(p, target):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    os.symlink(target, p)


def make_model(hub, name, files, shared=None, ref=SNAP):
    """files: {snapshot path: bytes}; LFS-style blobs named by sha256. shared: paths whose blob
    lives in <hub>/blobs/<xx>/<xet> (a fake xet hash) with a .refs manifest."""
    m = os.path.join(hub, name)
    write(os.path.join(m, "refs", "main"), ref.encode())
    for rel, data in files.items():
        etag = sha(data)
        blob = os.path.join(m, "blobs", etag)
        if not os.path.lexists(blob):
            if shared and rel in shared:
                xet = sha(b"xet" + data)
                store = os.path.join(hub, "blobs", xet[:2], xet)
                write(os.path.join(hub, "blobs", ".huggingface-shared-blobs"), b"1\n")
                if not os.path.exists(store):
                    write(store, data)
                with open(store + ".refs", "a") as f:
                    f.write("%s/blobs/%s\n" % (name, etag))
                write(store + ".lock", b"")
                link(blob, os.path.relpath(store, os.path.dirname(blob)))
            else:
                write(blob, data)
        depth = rel.count("/")
        link(os.path.join(m, "snapshots", ref, rel), "../../" + "../" * depth + "blobs/" + etag)
    return m


def dangling(top):
    out = []
    for root, dirs, files in os.walk(top):
        for f in files + dirs:
            p = os.path.join(root, f)
            if os.path.islink(p) and not os.path.exists(p):
                out.append(p)
    return out


def read(p):
    with open(p, "rb") as f:
        return f.read()


class HfTierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hf_tier_test_")
        t = self.tmp
        self.hot = os.path.join(t, "nvme", "hub")
        self.cold = os.path.join(t, "hdd", "hf-hub")
        self.arch = os.path.join(t, "hdd3", "hf-hub")
        self.htrash = os.path.join(t, "nvme", "trash")
        for d in (self.hot, self.cold, self.arch):
            os.makedirs(d)
        self.big = os.urandom(3 << 20)
        # A: shared-store model (its weight and a sibling's share one payload)
        make_model(self.hot, "models--o--A", {"model.safetensors": self.big, "config.json": b"{}",
                                              "sub/tok.json": b"tok"}, shared={"model.safetensors"})
        make_model(self.hot, "models--o--A2", {"model.safetensors": self.big}, shared={"model.safetensors"})
        # B: repo-local model, no cold copy yet
        make_model(self.hot, "models--o--B", {"w.bin": b"B" * 1000, "config.json": b"{b}"})
        # D: empty former bind mount; the 8 TB copy has it
        os.makedirs(os.path.join(self.hot, "models--o--D"))
        make_model(self.cold, "models--o--D", {"w.bin": b"D" * 500})
        # E: hot, and an 8 TB copy whose snapshot holds real files and an older refs/main
        make_model(self.hot, "models--o--E", {"w.bin": b"E" * 4000, "config.json": b"{new}"})
        e = os.path.join(self.cold, "models--o--E")
        write(os.path.join(e, "refs", "main"), b"old")
        write(os.path.join(e, "snapshots", SNAP, "w.bin"), b"E" * 4000)
        write(os.path.join(e, "snapshots", SNAP, "config.json"), b"{new}")

    def run_tool(self, *args, ok=True):
        cmd = [sys.executable, TOOL] + list(args) + [
            "--hot", self.hot, "--cold", self.cold, "--archive", self.arch,
            "--hot-trash", self.htrash, "--date", "D", "--min-free-gb", "0"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout)
        else:
            self.assertNotEqual(r.returncode, 0, r.stdout)
        return r.stdout

    def snap(self, root, name, rel):
        return os.path.join(root, name, "snapshots", SNAP, rel)

    def test_status(self):
        out = self.run_tool("status")
        self.assertRegex(out, r"models--o--A\s+hot .*shared blob store.*no 8 TB copy")
        self.assertRegex(out, r"models--o--D\s+empty .*cold\s+not visible")
        self.assertRegex(out, r"models--o--E\s+hot .*hot,cold")

    def test_cold_shared_then_hot(self):
        self.run_tool("cold", "o/A")
        e = os.path.join(self.hot, "models--o--A")
        self.assertTrue(os.path.islink(e))
        self.assertEqual(os.path.realpath(e), os.path.realpath(os.path.join(self.cold, "models--o--A")))
        self.assertEqual(read(self.snap(self.hot, "models--o--A", "model.safetensors")), self.big)
        self.assertEqual(read(self.snap(self.hot, "models--o--A", "sub/tok.json")), b"tok")
        # the cold copy is self-contained: blobs are real files, snapshots links into them
        cb = os.path.join(self.cold, "models--o--A", "blobs", sha(self.big))
        self.assertFalse(os.path.islink(cb))
        self.assertTrue(os.path.islink(self.snap(self.cold, "models--o--A", "model.safetensors")))
        # A2 still links the payload: it stays; nothing dangles on the NVMe
        self.assertEqual(dangling(self.hot), [])
        self.assertEqual(read(self.snap(self.hot, "models--o--A2", "model.safetensors")), self.big)
        self.assertTrue(os.path.isdir(os.path.join(self.htrash, "D", "hf-tier", "nvme", "models--o--A")))
        # A2 leaves too: now the payload goes to the trash with its .refs / .lock
        self.run_tool("cold", "models--o--A2")
        self.assertEqual(dangling(self.hot), [])
        shared = os.path.join(self.hot, "blobs")
        left = [f for _, _, fs in os.walk(shared) for f in fs if not f.startswith(".")]
        self.assertEqual(left, [])
        trashed = os.path.join(self.htrash, "D", "hf-tier", "nvme-shared-blobs")
        self.assertEqual(len([f for _, _, fs in os.walk(trashed) for f in fs]), 3)
        # and back: hot again, a real folder, cold keeps its copy
        self.run_tool("hot", "o/A")
        self.assertFalse(os.path.islink(e))
        self.assertTrue(os.path.isdir(e))
        self.assertEqual(read(self.snap(self.hot, "models--o--A", "model.safetensors")), self.big)
        self.assertTrue(os.path.islink(self.snap(self.hot, "models--o--A", "model.safetensors")))
        self.assertTrue(os.path.isdir(os.path.join(self.cold, "models--o--A")))
        self.assertEqual(dangling(self.hot), [])
        self.assertFalse(os.path.exists(os.path.join(self.hot, ".hf-tier-staging")))

    def test_empty_bind_point_becomes_link(self):
        self.run_tool("cold", "o/D")
        e = os.path.join(self.hot, "models--o--D")
        self.assertTrue(os.path.islink(e))
        self.assertEqual(read(self.snap(self.hot, "models--o--D", "w.bin")), b"D" * 500)

    def test_merge_into_real_file_copy(self):
        self.run_tool("cold", "o/E")
        c = os.path.join(self.cold, "models--o--E")
        self.assertEqual(read(os.path.join(c, "refs", "main")), SNAP.encode())
        self.assertEqual(read(os.path.join(self.cold, "..", "trash", "D", "hf-tier", "replaced",
                                           "models--o--E", "refs", "main")), b"old")
        # the content was there as real snapshot files: no blob duplicated
        self.assertEqual(os.listdir(os.path.join(c, "blobs")) if os.path.isdir(os.path.join(c, "blobs")) else [], [])
        self.assertEqual(read(self.snap(self.hot, "models--o--E", "w.bin")), b"E" * 4000)

    def test_new_snapshot_hardlinks_old_real_file(self):
        make_model(self.hot, "models--o--F", {"w.bin": b"F" * 5000})
        f = os.path.join(self.cold, "models--o--F")
        write(os.path.join(f, "refs", "main"), b"older")
        write(os.path.join(f, "snapshots", "older", "w.bin"), b"F" * 5000)
        out = self.run_tool("cold", "o/F")
        self.assertIn("hard-linked", out)
        blob = os.path.join(f, "blobs", sha(b"F" * 5000))
        self.assertTrue(os.path.samefile(blob, os.path.join(f, "snapshots", "older", "w.bin")))
        self.assertEqual(read(self.snap(self.hot, "models--o--F", "w.bin")), b"F" * 5000)

    def test_sync_and_archive(self):
        self.run_tool("sync", "o/B")
        self.assertTrue(os.path.isdir(os.path.join(self.cold, "models--o--B")))
        self.assertFalse(os.path.islink(os.path.join(self.hot, "models--o--B")))
        self.run_tool("archive", "o/B")
        e = os.path.join(self.hot, "models--o--B")
        self.assertEqual(os.path.realpath(e), os.path.realpath(os.path.join(self.arch, "models--o--B")))
        self.assertFalse(os.path.exists(os.path.join(self.cold, "models--o--B")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "hdd", "trash", "D", "hf-tier", "8tb", "models--o--B")))
        self.assertEqual(read(self.snap(self.hot, "models--o--B", "w.bin")), b"B" * 1000)
        self.assertIn("archive", self.run_tool("status"))

    def test_corrupt_blob_refused(self):
        b = os.path.join(self.hot, "models--o--B", "blobs", sha(b"B" * 1000))
        write(b, b"X" * 1000)  # same size, wrong content: its name no longer matches
        out = self.run_tool("cold", "o/B", ok=False)
        self.assertIn("sha256", out)
        self.assertFalse(os.path.islink(os.path.join(self.hot, "models--o--B")))

    def test_hot_space_refused(self):
        cmd = [sys.executable, TOOL, "hot", "o/D", "--hot", self.hot, "--cold", self.cold,
               "--archive", self.arch, "--hot-trash", self.htrash, "--min-free-gb", "1e9"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(r.returncode, 4, r.stdout)

    @unittest.skipUnless(os.path.isdir("/proc/self/fd"), "needs /proc")
    def test_open_file_refused(self):
        p = self.snap(self.hot, "models--o--B", "w.bin")
        holder = subprocess.Popen([sys.executable, "-c", "import sys,time; f=open(sys.argv[1]); time.sleep(30)", p])
        try:
            import time
            time.sleep(0.5)
            r = subprocess.run([sys.executable, TOOL, "cold", "o/B", "--hot", self.hot, "--cold", self.cold,
                                "--archive", self.arch, "--hot-trash", self.htrash],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(r.returncode, 3, r.stdout)
        finally:
            holder.kill()


if __name__ == "__main__":
    unittest.main()
