#!/usr/bin/env python3
"""Tests of tools/box_validate that need no box: python3 tools/box_validate/test_box_validate.py

The JUnit / test-list logic, G0's normalised comparison and binary checksum compare, the
interleaved medians, the summary, the orchestrator's resume / redo / stop-the-line / needs /
after rules and need_ok (remote.sh over a stub stage file, with a stand-in flock), the server
rows' client (serve_run.sh + serve_client.py against a stand-in b70-serve: run, table,
compare), long_ids.py, k2_oracle.sh's RAM guard, tok_diff.py, the driver's --list, --dry-run
and selection (never printing the box address), and the registry's consistency. Runs on the
Mac and on the box.
"""
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import junit  # noqa: E402
import summary  # noqa: E402


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def junit_xml(cases):
    """cases: [(name, status, message or None, output)]"""
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<testsuite name="(empty)" tests="%d">' % len(cases)]
    for name, st, msg, text in cases:
        out.append(f'\t<testcase name="{name}" classname="{name}" time="0.1" status="{st}">')
        if st == "fail":
            out.append(f'\t\t<failure message="{msg or "Failed"}"/>')
        elif st == "notrun":
            out.append(f'\t\t<skipped message="{msg}"/>')
        esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f"\t\t<system-out>{esc}</system-out>")
        out.append("\t</testcase>")
    out.append("</testsuite>")
    return "\n".join(out) + "\n"


def tests_json(tests):
    """tests: [(name, [labels])]"""
    return json.dumps({"kind": "ctestInfo", "tests": [
        {"name": n, "properties": [{"name": "LABELS", "value": l}] if l else []} for n, l in tests]})


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bv-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)


class JunitTest(Tmp):
    def test_statuses(self):
        write(self.p("a.junit.xml"), junit_xml([
            ("passer", "run", None, "ok"),
            ("failer", "fail", "Failed", "bad"),
            ("timeouter", "fail", "Timeout", ""),
            ("skipper", "notrun", "SKIP_RETURN_CODE=77", "no oracle"),
            ("missing", "notrun", "Unable to find executable", ""),
            ("disabled", "disabled", None, ""),
        ]))
        r = junit.parse_junit(self.p("a.junit.xml"))
        self.assertEqual(r["passer"][0], "PASS")
        self.assertEqual(r["failer"][0], "FAIL")
        self.assertEqual(r["timeouter"][:2], ("FAIL", "Timeout"))
        self.assertEqual(r["skipper"][0], "SKIP")
        self.assertEqual(r["missing"][0], "FAIL")          # an absent executable is not a skip
        self.assertEqual(r["disabled"][0], "SKIP")

    def test_todo_pick_labels_also_in(self):
        write(self.p("tests.json"), tests_json([
            ("a_test", []), ("b_test", ["checkpoint"]), ("c_kv8_test", ["checkpoint", "kv8"]),
            ("d_agnes_test", ["agnes"]), ("new_test", [])]))
        write(self.p("base.json"), tests_json([("a_test", []), ("b_test", []), ("c_kv8_test", [])]))
        write(self.p("g0.junit.xml"), junit_xml([("a_test", "run", None, "")]))
        run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "junit.py"), *a],
                                        capture_output=True, text=True)
        r = run("todo", self.p("tests.json"), "--also-in", self.p("base.json"), "--no-label", "kv8",
                "--done", self.p("*.junit.xml"))
        self.assertEqual(r.stdout.strip(), "^(b_test)$")
        r = run("todo", self.p("tests.json"), "--re", "test", "--label", "kv8", "--done", self.p("*.junit.xml"))
        self.assertEqual(r.stdout.strip(), "^(c_kv8_test)$")
        r = run("todo", self.p("tests.json"), "--re", "^a_test$", "--done", self.p("*.junit.xml"))
        self.assertEqual(r.stdout.strip(), "")             # already has a result
        # pick: a passes, b has no result -> fail
        r = run("pick", self.p("tests.json"), "--re", "^(a_test|b_test)$", "--junit", self.p("*.junit.xml"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("b_test: no result in this run", r.stdout)
        write(self.p("r.junit.xml"), junit_xml([("b_test", "notrun", "SKIP_RETURN_CODE=77", "")]))
        r = run("pick", self.p("tests.json"), "--re", "^b_test$", "--junit", self.p("*.junit.xml"))
        self.assertEqual(r.returncode, 77)                 # only skips: a skip
        r = run("pick", self.p("tests.json"), "--re", "^(a_test|b_test)$", "--junit", self.p("*.junit.xml"))
        self.assertEqual(r.returncode, 0)
        r = run("pick", self.p("tests.json"), "--re", "^nothing$", "--junit", self.p("*.junit.xml"))
        self.assertEqual(r.returncode, 1)                  # selecting nothing is an error

    def test_latest_result_wins(self):
        write(self.p("tests.json"), tests_json([("a_test", [])]))
        write(self.p("1.junit.xml"), junit_xml([("a_test", "fail", "Failed", "")]))
        os.utime(self.p("1.junit.xml"), (1000, 1000))
        write(self.p("2.junit.xml"), junit_xml([("a_test", "run", None, "")]))
        self.assertEqual(junit.main(["pick", self.p("tests.json"), "--re", "a", "--junit", self.p("*.junit.xml")]), 0)

    def test_compare_normalises(self):
        a = "load: 13.6 s\nTree /home/u/base/build/x\ncos 0.999999123\nptr 0xdeadbeef\nkernel_count 774\n"
        b = "load: 12.1 s\nTree /home/u/head/build/x\ncos 0.999999123\nptr 0x1234\nkernel_count 774\n"
        write(self.p("a.xml"), junit_xml([("g", "run", None, a), ("only_a", "run", None, "")]))
        write(self.p("b.xml"), junit_xml([("g", "run", None, b)]))
        self.assertEqual(junit.main(["compare", self.p("a.xml"), self.p("b.xml"), "--strip-a", "/home/u/base",
                                     "--strip-b", "/home/u/head", "--out", self.p("d.diff")]), 0)
        write(self.p("b.xml"), junit_xml([("g", "run", None, b.replace("0.999999123", "0.999999124"))]))
        self.assertEqual(junit.main(["compare", self.p("a.xml"), self.p("b.xml"), "--strip-a", "/home/u/base",
                                     "--strip-b", "/home/u/head", "--out", self.p("d.diff")]), 1)
        with open(self.p("d.diff")) as f:
            self.assertIn("+cos 0.999999124", f.read())

    def test_timing_filter_keeps_numbers_that_matter(self):
        keep = ["  mean of the 64 per-layer upper-medians: 0.999931742   (GDN 48: 0.9999, FA 16: 0.9999)",
                "golden_gate_test OK: 3 prompts x 32 greedy tokens - 93/93 determined rows",
                "      min cos 0.999 (L3), max cos 1.0, max relL2 1.2e-03, 0/48 below 0.990",
                "x 8 x 17408 x 2 = 278528 B"]
        drop = ["prefill: 4096 ids in 1927.3 ms", "decode 256 ids at 29.45 t/s",
                "loaded in 13.6 s", "Total Test time (real) = 12.00 sec", "achieved 590.1 GB/s"]
        for line in keep:
            self.assertFalse(junit.TIMING.search(line), line)
        for line in drop:
            self.assertTrue(junit.TIMING.search(line), line)

    def test_grep(self):
        write(self.p("x.junit.xml"), junit_xml([("replay_determinism_test", "run", None,
                                                 "kernel_count 774\nother\n")]))
        r = subprocess.run([sys.executable, os.path.join(HERE, "junit.py"), "grep", "--junit",
                            self.p("*.junit.xml"), "--test", "replay", "--pattern", "kernel_count",
                            "--label", "replay"], capture_output=True, text=True)
        self.assertEqual(r.stdout, "METRIC replay\t[replay_determinism_test] kernel_count 774\n")


class G0CompareTest(Tmp):
    def mk(self, build, files):
        for rel, data in files.items():
            write(self.p(build, rel), data)

    def run_cmp(self, env=None):
        e = dict(os.environ, **(env or {}))
        return subprocess.run([os.path.join(HERE, "g0_compare.sh"), self.p("base"), self.p("head"), self.p("out")],
                              capture_output=True, text=True, env=e)

    def test_identical_and_added(self):
        self.mk("base", {"kernels/a.bin": "A", "kernels/b.bin": "B", "other/c.bin": "C"})
        self.mk("head", {"kernels/a.bin": "A", "kernels/b.bin": "B", "kernels/new.bin": "N"})
        r = self.run_cmp()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("common 2, identical 2, differ 0, added 1, removed 0", r.stdout)
        with open(self.p("out", "added.txt")) as f:
            self.assertEqual(f.read(), "kernels/new.bin\n")

    def test_differ_and_removed(self):
        self.mk("base", {"kernels/a.bin": "A", "kernels/b.bin": "B", "kernels/gone.bin": "G"})
        self.mk("head", {"kernels/a.bin": "A", "kernels/b.bin": "B2"})
        r = self.run_cmp()
        self.assertEqual(r.returncode, 1)
        self.assertIn("differ 1", r.stdout)
        self.assertIn("kernels/b.bin", r.stdout)
        self.mk("head", {"kernels/b.bin": "B"})
        r = self.run_cmp()
        self.assertEqual(r.returncode, 1)                  # removed fails by default
        r = self.run_cmp({"G0_ALLOW_REMOVED": "1"})
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_nothing_in_common(self):
        self.mk("base", {"kernels/a.bin": "A"})
        self.mk("head", {"kernels/z.bin": "Z"})
        self.assertEqual(self.run_cmp().returncode, 2)


class DataTest(Tmp):
    def test_link_and_resolve_and_complete(self):
        write(self.p("data", "oracle-out-primary", "x"), "1")
        write(self.p("data", "oracle-out-agnes", "x"), "1")
        os.makedirs(self.p("tree"))
        write(self.p("tree2", "oracle-out-agnes", "own"), "1")
        sh = os.path.join(HERE, "data.sh")
        r = subprocess.run([sh, "link", self.p("data"), self.p("tree"), self.p("tree2")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.islink(self.p("tree", "oracle-out-primary")))
        self.assertFalse(os.path.islink(self.p("tree2", "oracle-out-agnes")))   # kept, not replaced
        r = subprocess.run([sh, "link", self.p("data"), self.p("tree")], capture_output=True, text=True)
        self.assertIn("kept", r.stdout)
        # an HF cache with one complete and one partial snapshot
        hub = self.p("hf", "hub")
        for repo, files in (("models--o--full", ["a.safetensors", "b.safetensors"]),
                            ("models--o--part", ["a.safetensors"])):
            write(os.path.join(hub, repo, "refs", "main"), "abc")
            snap = os.path.join(hub, repo, "snapshots", "abc")
            write(os.path.join(snap, "config.json"), "{}")
            write(os.path.join(snap, "model.safetensors.index.json"),
                  json.dumps({"weight_map": {"w1": "a.safetensors", "w2": "b.safetensors"}}))
            for f in files:
                write(os.path.join(snap, f), "")
        env = dict(os.environ, HF_HOME=self.p("hf"), SNAP_QWEN="o/full", SNAP_AGNES="o/part",
                   SNAP_ORNITH="o/absent", SNAP_K2="o/absent")
        r = subprocess.run([sh, "resolve", "o/full"], capture_output=True, text=True, env=env)
        self.assertEqual(r.stdout.strip(), os.path.join(hub, "models--o--full", "snapshots", "abc"))
        r = subprocess.run([sh, "have"], capture_output=True, text=True, env=env, cwd=self.p("tree"))
        self.assertIn("HAVE_qwen=1", r.stdout)
        self.assertIn("HAVE_agnes=0", r.stdout)
        self.assertIn("HAVE_ornith=0", r.stdout)
        self.assertIn("HAVE_k2=0", r.stdout)
        self.assertIn("HAVE_oracle_k2=0", r.stdout)
        self.assertIn("HAVE_oracle_ornith_mtp=0", r.stdout)
        for p in ("prose", "code", "cjk"):                  # oracle_k2: all three prompts complete
            write(self.p("tree", "oracle-out-k2", p + ".ids"), "0 1\n")
            write(self.p("tree", "oracle-out-k2", p + ".golden.safetensors"), "x")
        r = subprocess.run([sh, "have"], capture_output=True, text=True, env=env, cwd=self.p("tree"))
        self.assertIn("HAVE_oracle_k2=1", r.stdout)
        self.assertIn("HAVE_oracle_qwen=1", r.stdout)      # the link made above


class MediansTest(unittest.TestCase):
    ROWS = textwrap.dedent("""\
        WARM a | b70-decode abc | 4096 | 256 | 1.00 | 1.0 |
        ROW a | b70-decode abc | 4096 | 256 | 30.00 | 33.3 |
        ROW a | b70-decode abc l0-int8 pp | 4096 | 2048 | 1900.0 | 2000.00 |
        ROW b | b70-decode abc int8-head | 4096 | 256 | 33.00 | 30.3 |
        ROW a | b70-decode abc | 4096 | 256 | 29.00 | 34.5 |
        ROW a | b70-decode abc l0-int8 pp | 4096 | 2048 | 1950.0 | 2100.00 |
        ROW b | b70-decode abc int8-head | 4096 | 256 | 32.00 | 31.2 |
        ROW a | b70-decode abc | 4096 | 256 | 31.00 | 32.3 |
        ROW a | b70-decode abc l0-int8 pp | 4096 | 2048 | 1800.0 | 2200.00 |
        ROW b | b70-decode abc int8-head | 4096 | 256 | 34.00 | 29.4 |
        ROW k1 BENCH k=1 mode=greedy head=int8 dv=off prompt=golden/prose@4096 depth=4096 ids=256 ms=1 tps=40.0 acc=0.5000 per_iter=1.5
        ROW k1 BENCH k=1 mode=greedy head=int8 dv=off prompt=golden/prose@4096 depth=4096 ids=256 ms=1 tps=42.0 acc=0.5200 per_iter=1.5
        ROW k1d BENCH k=1 mode=greedy head=int8 dv=128k prompt=golden/prose@4096 depth=4096 ids=256 ms=1 tps=44.1 acc=0.5100 per_iter=1.5
        """).splitlines()

    def test_medians_and_ratio(self):
        out = "\n".join(summary.medians_md(self.ROWS, ["b/a", "k1d/k1"]))
        self.assertIn("| a | 30.00 | 33.30 | 6.67% | 2100.00 | 9.52% | 3 |", out)
        self.assertIn("| b | 33.00 |", out)
        self.assertIn("RATIO b/a tg 1.1000", out)          # 33 / 30; the warm-up is not counted
        self.assertIn("RATIO k1d/k1 golden/prose@4096 1.0756", out)   # 44.1 / median(40, 42) = 44.1 / 41


class InterleaveTest(Tmp):
    def test_rounds_order_and_medians(self):
        rows = self.p("x.rows")
        arm = "echo '| b70-decode s{tag} | 4096 | 256 | {tps} | 33.3 |'; echo {tag} >> " + self.p("order")
        r = subprocess.run([os.path.join(HERE, "interleave.sh"), "-o", rows, "-r", "3", "--ratio", "b/a", "--",
                            "a", arm.format(tag="a", tps="30.00"), "b", arm.format(tag="b", tps="33.00")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.p("order")) as f:
            self.assertEqual(f.read().split(), ["a", "b", "a", "b", "b", "a", "a", "b"])  # warm-up, A B, B A, A B
        with open(rows) as f:
            lines = f.read().splitlines()
        self.assertEqual(sum(l.startswith("WARM ") for l in lines), 2)
        self.assertEqual(sum(l.startswith("ROW ") for l in lines), 6)
        self.assertIn("RATIO b/a tg 1.1000", r.stdout)
        r = subprocess.run([os.path.join(HERE, "interleave.sh"), "-o", rows, "-r", "1", "--", "a", "false"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("ARM FAILED a", r.stdout)


class OrchestratorTest(Tmp):
    """remote.sh over a stub stage file: resume, redo, stop-the-line, needs, after."""
    STUB = textwrap.dedent("""\
        row 0 "gate"
        row 1 "one"
        stage pre 0 always cpu - - "pre"
        st_pre() { chk "echo 'HAVE_a=1  # a' > $STATE/have.env" "have"; finish; }
        stage g0.a 0 g0 cpu - - "g0 a"
        st_g0_a() { chk "test -f $STATE/../g0_ok" "the g0 marker"; finish; }
        stage r1.a 1 default gpu a - "needs a"
        st_r1_a() { chk "echo ran >> $STATE/count-r1a; echo 'value 42'" "count"; grab m 'value'; finish; }
        stage r1.b 1 default cpu b - "needs b"
        st_r1_b() { chk true "never"; finish; }
        stage r1.c 1 default cpu - r1.a "after r1.a"
        st_r1_c() { chk true "fine"; finish; }
        stage r1.d 1 default cpu - - "fails"
        st_r1_d() { chk false "always false"; finish; }
        stage r1.e 1 default cpu - - "skips"
        st_r1_e() { skip "nothing to do"; finish; }
        stage r1.f 1 default cpu - r1.d "after a failure"
        st_r1_f() { chk true "fine"; finish; }
        stage r1.g 1 default cpu - - "need_ok over a PASS and a SKIP"
        st_r1_g() { need_ok r1.a r1.e r1.z; finish; }
        stage r1.h 1 default cpu - - "need_ok over a FAIL"
        st_r1_h() { need_ok r1.a r1.d; finish; }
        """)

    def setUp(self):
        super().setUp()
        write(self.p("stages.sh"), self.STUB)
        # macOS has no flock: a stand-in that always gets the lock (the box has util-linux's)
        write(self.p("bin", "flock"), "#!/bin/sh\nwhile [ $# -gt 0 ]; do case \"$1\" in -*) shift ;; *) break ;; esac; done\n"
              "shift; [ $# -gt 0 ] && exec \"$@\"; exit 0\n", 0o755)
        os.makedirs(self.p("home"))
        os.makedirs(self.p("base"))
        self.state = self.p("runs", "s1")

    def run_remote(self, ids, **env):
        e = dict(os.environ, PATH=self.p("bin") + os.pathsep + os.environ["PATH"], HOME=self.p("home"),
                 BV_STAGES_FILE=self.p("stages.sh"), BV_BASE=self.p("base"), B70_GIT_SHA="test")
        e.update(env)
        r = subprocess.run([os.path.join(HERE, "remote.sh"), self.state, *ids], capture_output=True, text=True, env=e)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def status(self, sid):
        with open(os.path.join(self.state, sid + ".status")) as f:
            return f.read().rstrip("\n").split("\t")

    def count(self):
        with open(os.path.join(self.state, "count-r1a")) as f:
            return len(f.readlines())

    def test_run(self):
        ids = ["pre", "g0.a", "r1.a", "r1.b", "r1.c", "r1.d", "r1.e", "r1.f", "r1.g", "r1.h"]
        self.run_remote(ids)
        self.assertEqual(self.status("g0.a")[1], "FAIL")
        self.assertIn("the g0 marker", self.status("g0.a")[6])
        for s in ("r1.a", "r1.b", "r1.c", "r1.d"):
            self.assertEqual(self.status(s)[1], "SKIP")
            self.assertIn("blocked: G0", self.status(s)[6])

        write(self.p("runs", "g0_ok"), "")
        self.run_remote(ids)
        self.assertEqual(self.status("g0.a")[1], "PASS")
        self.assertEqual(self.status("r1.a")[1], "PASS")
        self.assertEqual(self.status("r1.b")[1:3], ["SKIP", "77"])
        self.assertIn("missing data: b", self.status("r1.b")[6])
        self.assertEqual(self.status("r1.c")[1], "PASS")
        self.assertEqual(self.status("r1.d")[1], "FAIL")
        self.assertIn("always false", self.status("r1.d")[6])
        self.assertEqual(self.status("r1.e")[1], "SKIP")
        self.assertIn("nothing to do", self.status("r1.e")[6])
        self.assertEqual(self.status("r1.f")[1], "SKIP")
        self.assertIn("r1.d=FAIL", self.status("r1.f")[6])
        self.assertEqual(self.status("r1.g")[1], "PASS")    # a SKIP / not-run is noted, not failed
        with open(os.path.join(self.state, "r1.g.log")) as f:
            g = f.read()
        self.assertIn("NOTE r1.e: SKIP - not evidence", g)
        self.assertIn("NOTE r1.z: not run", g)
        self.assertEqual(self.status("r1.h")[1], "FAIL")
        self.assertIn("r1.d is FAIL", self.status("r1.h")[6])
        with open(os.path.join(self.state, "r1.a.log")) as f:
            self.assertIn("METRIC m\tvalue 42", f.read())
        self.assertEqual(self.count(), 1)

        out = self.run_remote(ids)                          # resume: PASS kept
        self.assertIn("r1.a PASS (kept", out)
        self.assertEqual(self.count(), 1)
        self.run_remote(["pre", "r1.a"], BV_REDO="r1.a")    # redo
        self.assertEqual(self.count(), 2)
        self.assertTrue(os.path.exists(os.path.join(self.state, "r1.a.log.prev")))

        os.remove(self.p("runs", "g0_ok"))                  # G0 breaks again
        self.run_remote(["pre", "g0.a", "r1.a"], BV_REDO="all")
        self.assertEqual(self.status("r1.a")[1], "SKIP")
        self.run_remote(["pre", "r1.a"], BV_REDO="r1.a", BV_FORCE="1")
        self.assertEqual(self.status("r1.a")[1], "PASS")    # --force runs past it
        self.assertEqual(self.count(), 3)

        # the summary over this state
        reg = self.p("registry.tsv")
        write(reg, "\n".join([
            "row\t0\tgate", "row\t1\tone", "note\t1\tsomething no stage runs",
            "stage\tpre\t0\talways\tcpu\t-\t-\tpre", "stage\tg0.a\t0\tg0\tcpu\t-\t-\tg0 a",
            "stage\tr1.a\t1\tdefault\tgpu\ta\t-\tneeds a", "stage\tr1.d\t1\tdefault\tcpu\t-\t-\tfails",
            "stage\tr1.z\t1\toptin\tgpu\t-\t-\tan opt-in", "cmd\tr1.z\ttools/box_validate.sh --with r1.z",
            "stage\tr1.m\t1\tmanual\t-\t-\t-\ta manual one", "cmd\tr1.m\t$ do it by hand"]) + "\n")
        md = self.p("summary.md")
        self.assertEqual(summary.main(["report", "--state", self.state, "--registry", reg, "--out", md,
                                       "--date", "2026-10-05"]), 0)
        with open(md) as f:
            text = f.read()
        self.assertIn("# Box validation - 2026-10-05", text)
        self.assertIn("**G0 (bitwise neutrality, stop-the-line): NOT PASSED**", text)
        self.assertIn("| `r1.a` | needs a | **PASS** |", text)
        self.assertIn("| `r1.d` | fails | **FAIL** | always false", text)
        self.assertIn("| `r1.z` | an opt-in | **OPT-IN** |", text)
        self.assertIn("- m: `value 42`", text)
        self.assertIn("something no stage runs", text)
        self.assertIn("$ do it by hand", text)


FAKE_SERVE = textwrap.dedent("""\
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    a = sys.argv[1:]
    port = int(a[a.index("--port") + 1])
    logdir = a[a.index("--log-requests") + 1] if "--log-requests" in a else None
    drift = "--drift" in a              # an arm whose output differs from the others
    n = [0]
    sys.stderr.write("max_len: 16384, mtp: %s\\n" % (a[a.index("--mtp") + 1] if "--mtp" in a else "off"))
    sys.stderr.flush()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass
        def reply(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        def do_GET(self):
            self.reply({"data": [{"id": "b70"}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path.endswith("/chat/completions")
            assert body["temperature"] == 0 and body["return_token_ids"] is True
            ids = [(7 * i + (1 if drift and i == 5 else 0)) % 1000 for i in range(body["max_tokens"])]
            t0 = time.monotonic(); time.sleep(0.01); t1 = time.monotonic(); time.sleep(0.02)
            ch = {"index": 0, "finish_reason": "length", "token_ids": ids}
            if chat:
                assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["tools"]
                ch["message"] = {"role": "assistant", "content": "ok"}
            else:
                ch["text"] = "ok"
            self.reply({"choices": [ch], "usage": {"prompt_tokens": 3, "completion_tokens": len(ids)},
                        "prompt_token_ids": [1, 2, 3]})
            if logdir:
                n[0] += 1
                with open(os.path.join(logdir, "%06d.json" % n[0]), "w") as f:
                    json.dump({"t_start": t0, "t_first_token": t1, "t_end": time.monotonic(), "out_ids": ids}, f)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeClientTest(Tmp):
    """serve_run.sh + serve_client.py against a stand-in b70-serve (no box, no model)."""

    def setUp(self):
        super().setUp()
        write(self.p("fake-serve"), "#!/bin/sh\nexec " + sys.executable + " " + self.p("fake_serve.py") + ' "$@"\n', 0o755)
        write(self.p("fake_serve.py"), FAKE_SERVE)
        # a chat set: manifest + one scenario + the ids the server is expected to encode
        write(self.p("set", "manifest.json"), json.dumps([{"name": "t1_x"}]))
        write(self.p("set", "t1_x.json"), json.dumps({"name": "t1_x", "enable_thinking": False,
              "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}))
        write(self.p("set", "t1_x.ids"), "1 2 3\n")

    def arm(self, label, rnd, *serve_flags):
        env = dict(os.environ, B70_SERVE=self.p("fake-serve"), PORT=str(free_port()), WAIT="30")
        r = subprocess.run([os.path.join(HERE, "serve_run.sh"), self.p("runs", f"{label}.r{rnd}"), label, str(rnd),
                            "/snap", *serve_flags, "--", "--set", "golden", "--set", self.p("set"),
                            "--max-tokens", "8"], capture_output=True, text=True, env=env, timeout=120)
        return r

    def client(self, *a):
        return subprocess.run([sys.executable, os.path.join(HERE, "serve_client.py"), *a],
                              capture_output=True, text=True)

    def test_arms_table_compare(self):
        for rnd in (1, 2):
            for label, flags in (("off", ["--mtp", "off"]), ("k3", ["--mtp", "3"])):
                r = self.arm(label, rnd, *flags)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SERVE_RUN [k3 r2] rc=0", r.stdout)
        self.assertIn("max_len: 16384, mtp: 3", r.stdout)          # the server's startup lines
        with open(self.p("runs", "k3.r2", "requests.jsonl")) as f:
            recs = [json.loads(line) for line in f]
        self.assertEqual([x["name"] for x in recs], ["prose", "code", "cjk", "t1_x"])
        chat = recs[-1]
        self.assertEqual(chat["endpoint"], "/v1/chat/completions")
        self.assertIs(chat["prompt_ids_match"], True)
        self.assertIs(recs[0]["prompt_ids_match"], False)          # golden .ids are not [1, 2, 3]
        self.assertEqual(chat["out_ids"], [0, 7, 14, 21, 28, 35, 42, 49])
        self.assertIsNotNone(chat["gen_tps"])                       # from the server's own record
        self.assertGreater(chat["gen_s"], 0.0)
        self.assertIsNone(chat["accepted"])
        r = self.client("compare", "--dir", self.p("runs"), "--ref", "off")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("IDENTICAL k3 r1 vs off: 4/4", r.stdout)
        r = self.client("table", "--dir", self.p("runs"), "--ref", "off", "--ratio", "k3/off")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("| golden/prose |", r.stdout)
        self.assertRegex(r.stdout, r"RATIO k3/off golden geomean [0-9.]+ over 3 prompts")
        r = self.arm("drift", 1, "--drift")                         # an arm that differs at index 5
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.client("compare", "--dir", self.p("runs"), "--ref", "off")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("DIFFER drift r1 vs off golden/prose at index 5", r.stdout)

    def test_server_down(self):
        env = dict(os.environ, B70_SERVE="/usr/bin/false", PORT=str(free_port()), WAIT="5")
        r = subprocess.run([os.path.join(HERE, "serve_run.sh"), self.p("down"), "x", "1", "/snap", "--", "--set", "golden"],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("the server did not come up", r.stdout)


class LongIdsAndOracleTest(Tmp):
    def test_long_ids(self):
        out = self.p("long.ids")
        r = subprocess.run([sys.executable, os.path.join(HERE, "long_ids.py"), "70000", out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out) as f:
            ids = [int(x) for x in f.read().split()]
        with open(os.path.join(ROOT, "tests", "golden", "prompts", "prose.ids")) as f:
            tail = [int(x) for x in f.read().split()]
        with open(os.path.join(ROOT, "tests", "golden", "prompts", "long32k.ids")) as f:
            filler = [int(x) for x in f.read().split()]
        self.assertEqual(len(ids), 70000)
        self.assertEqual(ids[-len(tail):], tail)
        self.assertEqual(ids[:len(filler)], filler)                  # the file, then the file again
        self.assertEqual(ids[len(filler):2 * len(filler)][:100], filler[:100])

    def test_k2_oracle_ram_guard(self):
        if os.path.exists("/proc/meminfo"):
            self.skipTest("a Linux box: the guard reads the real MemAvailable")
        r = subprocess.run([os.path.join(HERE, "k2_oracle.sh"), self.p("data")], capture_output=True, text=True,
                           env=dict(os.environ, K2_REF_MIN_GB="1"))
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("SKIP_REASON k2_oracle: MemAvailable", r.stdout)
        self.assertFalse(os.path.exists(self.p("data", "oracle-out-k2")))


class TokDiffTest(Tmp):
    """tok_diff.py: two tokenizer.json files beyond their added tokens (row 16, r16.tokdiff)."""

    def tok(self, name, vocab, merges, added, decoder=None):
        doc = {"model": {"type": "BPE", "vocab": vocab, "merges": merges}, "decoder": decoder or {"type": "ByteLevel"},
               "added_tokens": [{"id": i, "content": c} for i, c in added]}
        write(self.p(name, "tokenizer.json"), json.dumps(doc))
        return self.p(name, "tokenizer.json")

    def run_diff(self, a, b):
        return subprocess.run([sys.executable, os.path.join(HERE, "tok_diff.py"), a, b], capture_output=True, text=True)

    def test_added_only_and_beyond(self):
        v = {"a": 0, "b": 1, "ab": 2}
        a = self.tok("qwen", dict(v, **{"<x>": 3, "<y>": 4}), ["a b"], [(3, "<x>"), (4, "<y>")])
        b = self.tok("ornith", dict(v, **{"<x>": 3}), [["a", "b"]], [(3, "<x>")])   # one added token fewer
        r = self.run_diff(a, b)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("TOKDIFF model.merges: equal (1)", r.stdout)   # "a b" and ["a", "b"] are one merge
        self.assertIn("only in a (4, '<y>')", r.stdout)
        self.assertIn("TOKDIFF beyond the added tokens: none", r.stdout)
        c = self.tok("other", {"a": 0, "b": 2, "ba": 1}, ["b a"], [(3, "<x>")], decoder={"type": "Metaspace"})
        r = self.run_diff(a, c)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("only in a 1, only in b 1, at another id 1", r.stdout)
        self.assertIn("TOKDIFF beyond the added tokens: model.vocab, model.merges, decoder", r.stdout)
        self.assertEqual(self.run_diff(a, self.p("absent.json")).returncode, 2)


class DriverTest(unittest.TestCase):
    """tools/box_validate.sh without a box: --list, --dry-run, --registry, selection."""

    def run_driver(self, *args, bash=None, ok=True):
        env = {k: v for k, v in os.environ.items() if k not in ("BOX", "REMOTE_DIR", "BOX_SUFFIX")}
        cmd = ([bash] if bash else []) + [os.path.join(ROOT, "tools", "box_validate.sh"), *args]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        return r

    def box_address(self):
        """BOX from tools/box.env - this tree's, else the main checkout's (a worktree has none)."""
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, cwd=ROOT).stdout.strip()
        for d in (ROOT, os.path.dirname(common) if common else None):
            try:
                with open(os.path.join(d, "tools", "box.env")) as f:
                    m = re.search(r"^\s*(?:export\s+)?BOX=(\S+)", f.read(), re.M)
            except (OSError, TypeError):
                continue
            if m and m.group(1).strip("'\""):
                return m.group(1).strip("'\"")
        return None

    def test_dry_run_full(self):
        out = self.run_driver("--dry-run").stdout
        self.assertIn("tools/box.sh sync", out)
        self.assertIn("tools/probe/detach.sh", out)
        self.assertIn("--- g0.sha", out)
        self.assertIn("g0_compare.sh", out)
        self.assertIn("-j44", out)
        self.assertNotIn("--- r11.passkey262k", out)     # opt-in: not by default
        self.assertNotIn("--- r14.oracle", out)
        self.assertNotIn("--- r6.passkey120k", out)
        # rows 13 and 14 (spec 15d, 18b): R0 / K0 first, the no-checkpoint kernels before the gates
        self.assertIn("--- r13.kernels", out)
        self.assertIn("--- r14.k1", out)
        self.assertIn("env B70_LONGCTX_TESTS=1 ctest", out)             # r6.mtp_long's variants
        self.assertIn("--mtp 3 > ", out)                                 # r6.greedy_long: b70-decode --mtp
        self.assertIn("probe_mtp_steps", out)
        self.assertIn(" 120000 32 3 int8 off 131072", out)
        addr = self.box_address()
        if addr:
            self.assertNotIn(addr, out)
            self.assertNotIn(addr, self.run_driver("--dry-run", "--with", "optin").stdout)
        # G0 first, row 11's kernels before its gate twins
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order[:5], ["pre", "g0.build", "g0.sha", "g0.bitwise", "g0.suite"])
        self.assertLess(order.index("r11.kernels"), order.index("r11.q2"))
        self.assertEqual(order[-1], "x.rest")
        for first, later in (("r13.r0", "r13.kernels"), ("r13.kernels", "r13.prefill"), ("r13.kernels", "r13.gates"),
                             ("r14.k0", "r14.k1"), ("r14.k1", "r14.load"), ("r14.load", "r14.k3"),
                             ("r14.k3", "r14.golden"), ("r12.d2", "r12.a4"),
                             ("r14.cli", "r15.k0"), ("r15.k0", "r15.k1"), ("r15.k1", "r15.prefill"),
                             ("r15.prefill", "r15.golden"), ("r15.golden", "r15.cli"), ("r15.cli", "r16.r0"),
                             ("r16.r0", "r16.kernels"), ("r16.kernels", "r16.mtp"), ("r16.mtp", "r16.cost"),
                             ("r16.tokdiff", "r17.host"), ("r17.k0", "x.rest")):
            self.assertLess(order.index(first), order.index(later), (first, later))
        for optin in ("r15.p0", "r15.speed", "r15.golden_eager", "r16.mtp_rows", "r16.passkey", "r16.benchy"):
            self.assertNotIn(optin, order)
        # the server rows: one serve_run.sh per arm and round, round 2 in reversed order
        arms = re.findall(r"serve_run\.sh \S+/r2\.auto_rows/(\S+) ", out)
        self.assertEqual(arms, ["off.r1", "k1.r1", "k3.r1", "auto.r1", "auto.r2", "k3.r2", "k1.r2", "off.r2",
                                "off.r3", "k1.r3", "k3.r3", "auto.r3"])
        self.assertIn("serve_client.py compare --dir", out)
        self.assertRegex(out, r"serve_run\.sh \S+/r12\.a4/lookup2\.r1 lookup2 1 \S+ --max-len 16384 --spec lookup --spec-min-match 2 -- ")
        # Ornith prefills since 15d and is served since 15e: nothing expects those refusals
        self.assertNotIn("printing /prefill of a mixture-of-experts model/", out)
        self.assertNotIn("spec 15e/", out)
        self.assertRegex(out, r"b70-serve \S+ --port \d+ --max-len 16384 --pp-backend sycl-tla\n"
                              r" +\(must exit non-zero without a crash, printing /prefills on the L0 backends only/\)")
        # K2: b70-serve's refusal as main prints it; 18c's CLI rejects in row 15, 18b's in row 14
        self.assertIn("printing /is not served yet: spec 18d.s engine side/", out)
        self.assertIn("--re '^cli_reject_(k2_kv8|mtp_k2)$'", out)
        self.assertIn("--re '^cli_reject_k2_(pp_int8|prefill_sycl)$'", out)
        self.assertNotIn("cli_reject_k2_prefill,", out)
        # Ornith's greedy argmax binaries (66a8923) in the binary checks of rows 10, 13 and 16
        for stage, names in (("r10.r0", ["argmax_stage1_M1_V248070", "moe_M1_E256_T8_D2048_I512"]),
                             ("r13.r0", ["argmax_stage1_M1_V248070", "pf_moe_E256_T8_D2048_I512"]),
                             ("r16.r0", ["argmax_stage1_M%d_V248070" % m for m in (1, 2, 3, 4)])):
            body = out.split("--- " + stage + " ")[1].split("\n--- ")[0]
            line = next(x for x in body.splitlines() if "build/kernels/" in x)
            for n in names:
                self.assertRegex(line, r" %s[ ;]" % n, (stage, n))
        # row 16: the startup note that compared tokenizer and descriptor must stay silent for Ornith
        self.assertIn("note: tokenizer[.]json defines", out)

    def test_selection(self):
        out = self.run_driver("--dry-run", "--only", "r11", "--with", "r11.passkey120k", "--skip", "r11.speed").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r11.bf16", "r11.kernels", "r11.q2", "r11.q5", "r11.memory",
                                 "r11.passkey120k"])
        out = self.run_driver("--dry-run", "--only", "r14", "--with", "r14.oracle").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r14.k0", "r14.host", "r14.k1", "r14.load", "r14.k3", "r14.oracle",
                                 "r14.golden", "r14.golden_eager", "r14.cli"])
        self.assertIn("tools/box_validate/k2_oracle.sh $HOME/b70-inference-server", out)
        out = self.run_driver("--dry-run", "--only", "r13").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r13.r0", "r13.host", "r13.kernels", "r13.prefill", "r13.gates", "r13.split",
                                 "r13.cli"])
        out = self.run_driver("--dry-run", "--only", "r15", "--with", "r15.speed").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r15.k0", "r15.host", "r15.k1", "r15.prefill", "r15.split", "r15.golden",
                                 "r15.cli", "r15.speed"])
        self.assertIn("pp4096-eager 'B70_K2_ATTN=eager B70_GIT_SHA=", out)
        out = self.run_driver("--dry-run", "--only", "r16").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r16.r0", "r16.host", "r16.kernels", "r16.mtp", "r16.m1", "r16.serve",
                                 "r16.golden_server", "r16.prefix", "r16.cost", "r16.tokdiff"])
        self.assertIn("build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode "
                      "tests/golden/prompts urakozz/Ornith-", out)
        self.assertIn("tools/box_validate/tok_diff.py", out)
        out = self.run_driver("--dry-run", "--only", "r17").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r17.host", "r17.k0"])
        r = self.run_driver("--dry-run", "--only", "r99", ok=False)
        self.assertEqual(r.returncode, 2)
        out = self.run_driver("--dry-run", "--redo", "r1").stdout
        self.assertIn("# redo: r1.host r1.load", out)
        self.assertIn("BV_REDO=r1.host\\ r1.load", out)

    def test_list_and_bash32(self):
        out = self.run_driver("--list").stdout
        self.assertIn("r11.kernels*", out)
        if os.path.exists("/bin/bash"):
            v = subprocess.run(["/bin/bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout
            if v.strip() == "3":                            # the Mac's /bin/bash: the driver must work there
                self.assertEqual(self.run_driver("--list", bash="/bin/bash").stdout, out)
                a = self.run_driver("--dry-run", bash="/bin/bash").stdout
                b = self.run_driver("--dry-run").stdout
                self.assertEqual(a, b)

    def test_registry_consistent(self):
        reg = tempfile.mktemp(suffix=".tsv")
        try:
            self.run_driver("--registry", reg)
            rows, notes, stages, cmds = summary.read_registry(reg)
        finally:
            if os.path.exists(reg):
                os.remove(reg)
        with open(os.path.join(HERE, "data.sh")) as f:
            keys = set(re.findall(r"^\s*have (\w+) ", f.read(), re.M))
        with open(os.path.join(HERE, "stages.sh")) as f:
            body = f.read()
        seen = []
        for sid, s in stages.items():
            self.assertIn(s["row"], rows, sid)
            self.assertIn(s["sel"], ("always", "g0", "default", "optin", "manual"), sid)
            self.assertIn(s["kind"], ("cpu", "gpu", "gpu-self", "-"), sid)
            fn = "st_" + sid.replace(".", "_").replace("-", "_")
            self.assertRegex(body, r"(?m)^" + re.escape(fn) + r"\(\) \{", sid)
            for k in s["needs"].split(","):
                if k != "-":
                    self.assertIn(k, keys, f"{sid}: unknown data key {k}")
            for d in s["after"].split(","):
                if d != "-":
                    self.assertIn(d, seen, f"{sid}: after {d}, which does not run before it")
            if s["sel"] in ("optin", "manual"):
                self.assertTrue(cmds.get(sid), f"{sid}: no commands")
            seen.append(sid)
        self.assertEqual(len(seen), len(set(seen)))
        g0 = [sid for sid, s in stages.items() if s["sel"] == "g0"]
        first = [sid for sid, s in stages.items() if s["sel"] != "always"][: len(g0)]
        self.assertEqual(g0, first, "the G0 stages come first")


if __name__ == "__main__":
    unittest.main()
