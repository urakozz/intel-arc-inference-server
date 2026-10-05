#!/usr/bin/env python3
"""Tests of tools/box_validate that need no box: python3 tools/box_validate/test_box_validate.py

The JUnit / test-list logic, G0's normalised comparison and binary checksum compare, the
interleaved medians, the summary, the orchestrator's resume / redo / stop-the-line / needs /
after rules (remote.sh over a stub stage file, with a stand-in flock), the driver's --list,
--dry-run and selection, and the registry's consistency. Runs on the Mac and on the box.
"""
import json
import os
import re
import shutil
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
                   SNAP_ORNITH="o/absent")
        r = subprocess.run([sh, "resolve", "o/full"], capture_output=True, text=True, env=env)
        self.assertEqual(r.stdout.strip(), os.path.join(hub, "models--o--full", "snapshots", "abc"))
        r = subprocess.run([sh, "have"], capture_output=True, text=True, env=env, cwd=self.p("tree"))
        self.assertIn("HAVE_qwen=1", r.stdout)
        self.assertIn("HAVE_agnes=0", r.stdout)
        self.assertIn("HAVE_ornith=0", r.stdout)
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
        ids = ["pre", "g0.a", "r1.a", "r1.b", "r1.c", "r1.d", "r1.e", "r1.f"]
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
        try:
            with open(os.path.join(ROOT, "tools", "box.env")) as f:
                m = re.search(r"^BOX=(\S+)", f.read(), re.M)
                return m.group(1).strip("'\"") if m else None
        except OSError:
            return None

    def test_dry_run_full(self):
        out = self.run_driver("--dry-run").stdout
        self.assertIn("tools/box.sh sync", out)
        self.assertIn("tools/probe/detach.sh", out)
        self.assertIn("--- g0.sha", out)
        self.assertIn("g0_compare.sh", out)
        self.assertIn("-j44", out)
        self.assertNotIn("--- r11.passkey262k", out)     # opt-in: not by default
        addr = self.box_address()
        if addr:
            self.assertNotIn(addr, out)
        # G0 first, row 11's kernels before its gate twins
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order[:5], ["pre", "g0.build", "g0.sha", "g0.bitwise", "g0.suite"])
        self.assertLess(order.index("r11.kernels"), order.index("r11.q2"))
        self.assertEqual(order[-1], "x.rest")

    def test_selection(self):
        out = self.run_driver("--dry-run", "--only", "r11", "--with", "r11.passkey120k", "--skip", "r11.speed").stdout
        order = re.findall(r"^--- (\S+)", out, re.M)
        self.assertEqual(order, ["pre", "r11.bf16", "r11.kernels", "r11.q2", "r11.q5", "r11.memory",
                                 "r11.passkey120k"])
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
