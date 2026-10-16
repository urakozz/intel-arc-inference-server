"""Tests for muse_p0_project.py (spec 23f Task 5). Dependency-free but for dflash_accept's pooling (pure Python).

    python3 tools/spec/test_muse_p0_project.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muse_p0_project as mp  # noqa: E402
import test_lookup_accept as tla  # noqa: E402


def _accept(path, arm, srcs):
    with open(path, "w") as f:
        json.dump({"arm": arm, "variants": {arm: srcs}, "agree": {}, "meta": {}}, f)


def _entry(alphas, rows=10):
    """One source's per-K stats whose pooled alpha_j are `alphas` (counted 100 at every depth)."""
    e = {}
    for K in range(1, 16):
        e[str(K)] = {"rows": rows, "censored": 0, "hist": [0] * (K + 1),
                     "counted": [100] * K, "accepted": [round(100 * alphas[j]) for j in range(K)]}
    e["sampled"] = {"raw": {"counted": [100] * 15, "sum": [100 * a for a in alphas]},
                    "capped": {"counted": [100] * 15, "sum": [50 * a for a in alphas]}}
    return e


def test_derived_table():
    assert len(mp.DERIVED_VERIFY) == 16 and mp.DERIVED_VERIFY[:4] == [1.00, 1.18, 1.56, 1.79]
    assert abs(mp.DERIVED_VERIFY[15] - (1.79 + 12 * 0.305)) < 1e-6
    assert mp.DERIVED_DRAFT["int4g64"] < mp.DERIVED_DRAFT["int8"] < mp.DERIVED_DRAFT["bf16"]
    assert abs(mp.DERIVED_DRAFT["int4g64"] - (1.358 + 1.346) / 14.86) < 1e-3
    assert mp.arm_format("int4g64h") == "int4g64" and mp.arm_format("int8h") == "int8" and mp.arm_format("bf16") == "bf16"


def test_rate_and_best_k():
    E = [1 + 0.9 * k for k in range(1, 16)]          # E_K grows 0.9 a draft
    verify = [1.0 + 0.1 * m for m in range(16)]      # +0.1 a row
    rows, best = mp.dflash_rate(E, verify, 0.2)
    assert rows[0] == (1, E[0], verify[1] + 0.2, E[0] / (verify[1] + 0.2))
    assert best[0] == 15 and abs(best[3] - E[14] / (verify[15] + 0.2)) < 1e-12
    rows, best = mp.dflash_rate([1.5] * 15, verify, 0.2)   # nothing past depth 1: K = 1 is best
    assert best[0] == 1


def test_verdict():
    good = {"code": {"dflash": 2.0, "lookup": 1.2}, "a4": {"dflash": 1.5, "lookup": 1.3}, "agentic": {"dflash": 1.3, "lookup": 1.1}}
    assert mp.verdict(good) == "GO"
    bad = dict(good, a4={"dflash": 1.5, "lookup": 1.4})   # 1.5 < 1.10 x 1.4
    assert mp.verdict(bad) == "NO GO"
    assert mp.verdict({"code": {"dflash": 2.0, "lookup": 1.0}}).startswith("UNDECIDED")


def test_end_to_end_and_probe():
    with tempfile.TemporaryDirectory() as d:
        acc = os.path.join(d, "accept")
        os.makedirs(acc)
        _accept(os.path.join(acc, "accept.int4g64h.json"), "int4g64h",
                {"code/a": _entry([0.9] * 15), "a4/x": _entry([0.8] * 15), "agentic/s": _entry([0.7] * 15)})
        dumps = os.path.join(d, "dumps")
        os.makedirs(dumps)
        rep = list(range(100, 140))
        tla._write_dump(os.path.join(dumps, "code__a.taps.safetensors"), "code/a", rep + rep[:30], 40)
        tla._write_dump(os.path.join(dumps, "a4__x.taps.safetensors"), "a4/x", list(range(60)), 30)
        tla._write_dump(os.path.join(dumps, "agentic__s.taps.safetensors"), "agentic/s", list(range(80)), 40)
        out = os.path.join(d, "p.json")
        assert mp.main(["--accept", acc, "--dumps", dumps, "--json", out]) == 0
        r = json.load(open(out))
        code = r["results"]["int4g64h"]["code"]
        assert code["best_k"] >= 1 and code["dflash"] > 1.0 and code["lookup"] > 1.0   # the copied span pays lookup
        assert r["verdicts"]["int4g64h"] in ("GO", "NO GO")
        assert mp.main(["--accept", acc, "--sampled", "capped", "--json", out]) == 0
        assert json.load(open(out))["results"]["int4g64h"]["code"]["E"] < code["E"]
        # a probe log in probe_draft_cost --shapes muse's table form
        log = os.path.join(d, "probe.log")
        with open(log, "w") as f:
            f.write("| arm | GEMVs + head + argmax ms | ratio to M = 1 (pairs) | attn est. at 4096 ms | verify at 4096 ms | x M = 1 |\n")
            for M in range(1, 17):
                f.write(f"| verify M {M} | {20 + M:.3f} | {1 + 0.05 * M:.3f} | 0.230 | {20.23 + M:.3f} | {1 + 0.06 * (M - 1):.3f} |\n")
                if M > 8:
                    f.write(f"| verify M {M} (two passes: 8 + {M - 8}) | {30 + M:.3f} | 1.5 | 0.230 | 31.0 | {1 + 0.05 * (M - 1):.3f} |\n")
            f.write("| drafter block int4-g64 + int8 head x15 | 4.200 | 0.200 | 2700 | 640 |\n")
        verify, draft = mp.parse_probe(log)
        assert len(verify) == 16 and verify[0] == 1.0
        assert abs(verify[15] - (1 + 0.05 * 15)) < 1e-9 and abs(verify[3] - (1 + 0.06 * 3)) < 1e-9   # the better form
        assert abs(draft["int4g64"] - 4.2 / 21.23) < 1e-9
        assert mp.main(["--accept", acc, "--probe", log, "--json", out]) == 0
        assert json.load(open(out))["source"].startswith("measured")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
