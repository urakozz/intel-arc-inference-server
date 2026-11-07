#!/usr/bin/env python3
"""Checks for serve_judge.py on hand-made requests and references (V = 8; no box, no model, no numpy needed).

    python3 tools/box_validate/test_serve_judge.py [test_name ...]

The key against qwen_forced.py's hash and forced_judge.h's name rule; then the N6 cases: identical, a difference
without its references (SKIP, dumped), with them - no more speculative misses than plain (PASS), more but every one a
near-miss (PASS), one beyond 8 ulp (FAIL; PASS when spec_judge.h's kReviewedRows names it, exactly or by tag / row /
ids / gap), the one-ulp tie rule, a length-only difference (FAIL), a missing request (FAIL), an unknown prompt (SKIP).
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_judge as sj  # noqa: E402

V = 8
PROMPT = [1, 2, 3]
OFF = [4, 5, 6, 7]
SPEC = [4, 5, 1, 1]          # differs at index 2: context PROMPT + [4, 5], suffixes [6, 7] / [1, 1]
CTX = PROMPT + [4, 5]
TOP = 10.0                   # one bf16 ulp at 10.0 is 2^-4 = 0.0625; 8 ulp 0.5


def load_qwen_forced():
    spec = importlib.util.spec_from_file_location("qwen_forced_t", os.path.join(HERE, "..", "oracle", "qwen_forced.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_judge(path, rows, force):
    """rows: [(top id, {id: value below TOP})] -> `logits` [len(rows)][V], every other id 5.0 below."""
    flat = []
    for top, near in rows:
        r = [TOP - 5.0] * V
        r[top] = TOP
        for i, below in near.items():
            r[i] = TOP - below
        flat += r
    lg = struct.pack(f"<{len(flat)}f", *flat)
    tk = struct.pack(f"<{len(force)}i", *force)
    h = {"logits": {"dtype": "F32", "shape": [len(rows), V], "data_offsets": [0, len(lg)]},
         "tokens": {"dtype": "I32", "shape": [len(force)], "data_offsets": [len(lg), len(lg) + len(tk)]}}
    hb = json.dumps(h).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + lg + tk)


class Case:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="serve_judge_test_")
        self.dir = os.path.join(self.tmp, "runs")
        self.judge = os.path.join(self.tmp, "judge")
        self.dump = os.path.join(self.tmp, "dump")
        os.makedirs(self.judge)
        self.header = os.path.join(self.tmp, "spec_judge.h")
        self.reviewed([])

    def arm(self, label, rnd, out, prompt=PROMPT, name="prose", match=None):
        d = os.path.join(self.dir, f"{label}.r{rnd}")
        os.makedirs(d, exist_ok=True)
        rec = {"label": label, "round": rnd, "set": "golden", "name": name, "out_ids": out,
               "prompt_ids": prompt, "prompt_ids_match": match}
        with open(os.path.join(d, "requests.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def reviewed(self, entries):
        body = "".join(f'    {{"{k}", {r}, {a}, {b}, {g}, "2026-11-05", kReviewedReason}},\n' for k, r, a, b, g in entries)
        with open(self.header, "w") as f:
            f.write("inline constexpr ReviewedRow kReviewedRows[] = {\n    // a comment {\"x\", 1, 2, 3, 4.0, \"d\"}\n"
                    + body + "};\n")

    def keys(self, tag="serve_int8_golden_prose"):
        return sj.forced_key(tag, CTX, OFF[2:]), sj.forced_key(tag, CTX, SPEC[2:])

    def refs(self, plain_rows, spec_rows):
        kp, ks = self.keys()
        write_judge(os.path.join(self.judge, kp + ".golden.safetensors"), plain_rows, OFF[2:])
        write_judge(os.path.join(self.judge, ks + ".golden.safetensors"), spec_rows, SPEC[2:])

    def run(self):
        r = subprocess.run([sys.executable, os.path.join(HERE, "serve_judge.py"), "--dir", self.dir, "--ref", "off",
                            "--dump", self.dump, "--judge", self.judge, "--header", self.header, "--vocab-used", str(V)],
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def test_key():
    qf = load_qwen_forced()
    for p, f in (([1, 2, 3], [4]), ([248068, 0, 7], []), (list(range(300)), [5, 6])):
        assert sj.fnv_key_hash(p, f) == qf.fnv_key_hash(p, f)
    k = sj.forced_key("serve_int8_toolcall/t1 x:y,z", [1], [2])
    assert k.startswith("serve_int8_toolcall_t1_x_y_z_") and len(k.rsplit("_", 1)[1]) == 16
    assert sj.key_tag(k) == "serve_int8_toolcall_t1_x_y_z"
    import re
    assert re.search(qf.I8HEAD_RE, k)                      # the int8 head's references (b70-serve's head)


def test_identical():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("k3", 1, OFF)
    rc, out = c.run()
    assert rc == 0, out
    assert "JUDGED k3 r1 vs off: 1 requests - 1 identical, 0 differ and pass" in out


def test_awaiting_is_a_skip():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("k3", 1, SPEC)
    c.arm("k3", 2, SPEC)                                   # the same context twice: one pair of keys
    rc, out = c.run()
    assert rc == 77, out
    kp, ks = c.keys()
    assert f"SKIP k3 r1 vs off golden/prose at index 2 (4 vs 4 ids): no f32 reference for {kp}, {ks}" in out, out
    assert "AWAITING f32 reference (not judged, not passed)" in out
    assert "SKIP_REASON serve_judge: 2 differing request(s) not judged (2 context(s) await" in out
    with open(os.path.join(c.dump, ks + ".prompt.ids")) as f:
        assert f.read() == "1 2 3 4 5\n"
    with open(os.path.join(c.dump, ks + ".force.ids")) as f:
        assert f.read() == "1 1\n"
    qf = load_qwen_forced()
    assert set(qf.load_contexts([c.dump])) == {kp, ks}     # qwen_forced.py reads the dumps (key hash checked)


def test_judged_pass_and_near_miss():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("k3", 1, SPEC)
    # plain: rejects 7 at row 1 (one miss); spec: rejects 1 at row 1 (one miss) -> 1 <= 1: PASS
    c.refs([(6, {}), (3, {7: 2.0}), (0, {})], [(1, {}), (3, {1: 2.0}), (0, {})])
    rc, out = c.run()
    assert rc == 0, out
    assert "the f32 judge rejects 1 speculative / 1 plain ids from there (worst speculative miss 32.00 ulp) - PASS" in out
    # more misses than plain, each a near-miss (row 0: 0.25 below = 4 ulp) -> PASS
    c.refs([(6, {}), (7, {}), (0, {})], [(6, {1: 0.25}), (1, {}), (0, {})])
    rc, out = c.run()
    assert rc == 0, out
    assert "rejects 1 speculative / 0 plain ids from there (worst speculative miss 4.00 ulp) - PASS" in out


def test_tie_rule():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("k3", 1, SPEC)
    # row 0: id 1 sits 0.05 (< one ulp 0.0625) below a unique top: the one-ulp tie set holds it
    c.refs([(6, {}), (7, {}), (0, {})], [(6, {1: 0.05}), (1, {}), (0, {})])
    rc, out = c.run()
    assert rc == 0, out
    assert "rejects 0 speculative / 0 plain ids" in out


def test_beyond_near_miss_fails_unless_reviewed():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("k3", 1, SPEC)
    c.refs([(6, {}), (7, {}), (0, {})], [(1, {}), (3, {1: 1.0}), (0, {})])   # row 1: 16 ulp
    rc, out = c.run()
    assert rc == 1, out
    assert "rejects 1 speculative / 0 plain ids from there (worst speculative miss 16.00 ulp) - FAIL" in out
    assert "row 1: ids 3 (f32 top) / 1 (speculative): f32 gap 16.00 ulp; no bf16 reference - the strict bar 8: FAIL" in out
    kp, ks = c.keys()
    c.reviewed([(ks, 1, 3, 1, 16.0)])
    rc, out = c.run()
    assert rc == 0, out
    assert "PASS as REVIEWED" in out
    c.reviewed([("serve_int8_golden_prose_0123456789abcdef", 1, 3, 1, 16.0)])   # a variant's key: tag / row / ids / gap
    rc, out = c.run()
    assert rc == 0, out
    assert "matched by tag / row / ids / f32 gap" in out
    c.reviewed([("serve_int8_golden_prose_0123456789abcdef", 1, 3, 1, 15.5)])   # the gap is not the entry's
    rc, out = c.run()
    assert rc == 1, out
    c.reviewed([("serve_int8_golden_code_0123456789abcdef", 1, 3, 1, 16.0)])    # another prompt's tag
    rc, out = c.run()
    assert rc == 1, out


def test_structural_failures_and_unknown_prompt():
    c = Case()
    c.arm("off", 1, OFF)
    c.arm("off", 1, OFF, name="code")
    c.arm("k3", 1, OFF[:3])                                # a prefix: a length difference
    rc, out = c.run()
    assert rc == 1, out
    assert "one stream is a prefix of the other" in out
    assert "MISSING k3 r1 golden/code: FAIL" in out
    c = Case()
    c.arm("off", 1, OFF, prompt=None, match=False)
    c.arm("k3", 1, SPEC, prompt=None, match=False)
    rc, out = c.run()
    assert rc == 77, out
    assert "the context is unknown, not judged" in out


def test_reviewed_rows_of_the_header():
    rows = sj.reviewed_rows(sj.HEADER)
    assert rows and all(len(e) == 6 for e in rows), rows
    assert rows[0][0].startswith("mtp_gpu_") and rows[0][1] == 0


def test_decode_records():
    """serve_client.py record: b70-decode's prompt / output ids as records serve_judge.py judges like a server's."""
    import serve_client as sc
    tmp = tempfile.mkdtemp(prefix="serve_record_test_")
    files = {}
    for n, v in (("p", PROMPT), ("off", OFF), ("spec", SPEC)):
        files[n] = os.path.join(tmp, n + ".ids")
        with open(files[n], "w") as f:
            f.write("\n".join(map(str, v)) + "\n")
    for label, k in (("off", "off"), ("mtp", "spec")):
        rc = sc.main(["record", "--out", os.path.join(tmp, "runs", label + ".r1", "requests.jsonl"), "--label", label,
                      "--set", "golden", "--name", "long", "--prompt", files["p"], "--ids", files[k]])
        assert rc == 0
    rc = sc.main(["record", "--out", os.path.join(tmp, "runs", "mtp.r1", "requests.jsonl"), "--label", "mtp",
                  "--set", "golden", "--name", "long", "--prompt", files["p"], "--ids", files["spec"]])
    assert rc == 0                                               # a re-run replaces its record
    recs = sc.records(os.path.join(tmp, "runs"))
    assert len(recs) == 2, recs
    assert sorted(r["label"] for r in recs) == ["mtp", "off"], recs
    assert all(r["prompt_ids"] == PROMPT for r in recs), recs
    assert sc.main(["record", "--out", os.path.join(tmp, "x.jsonl"), "--label", "x", "--set", "g", "--name", "n",
                    "--prompt", os.path.join(tmp, "absent.ids"), "--ids", files["off"]]) == 1
    out = subprocess.run([sys.executable, os.path.join(HERE, "serve_judge.py"), "--dir", os.path.join(tmp, "runs"),
                          "--ref", "off", "--dump", os.path.join(tmp, "dump"), "--judge", os.path.join(tmp, "judge")],
                         capture_output=True, text=True)
    assert out.returncode == 77, out.stdout + out.stderr          # a difference, no reference: SKIP, never PASS
    assert "AWAITING f32 reference" in out.stdout, out.stdout


def main(argv):
    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    names = argv or sorted(tests)
    for n in names:
        tests[n]()
        print(f"ok {n}")
    print(f"{len(names)} serve_judge tests passed")


if __name__ == "__main__":
    main(sys.argv[1:])
