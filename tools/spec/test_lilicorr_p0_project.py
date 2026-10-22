"""Tests for lilicorr_p0_project.py (spec 26a Task 5). Dependency-free but for dflash_accept's pooling (pure Python).

    python3 tools/spec/test_lilicorr_p0_project.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lilicorr_p0_project as lp  # noqa: E402
import test_lookup_accept as tla  # noqa: E402


def _accept(path, arm, srcs):
    with open(path, "w") as f:
        json.dump({"arm": arm, "variants": {arm: srcs}, "agree": {}, "meta": {}}, f)


def _entry(alphas, lp_scale=1.0):
    """One source's per-K stats whose pooled alpha_j are `alphas` (counted 100 at every depth); sampled sums for the two
    q forms (lattice+lp at lp_scale of lattice's)."""
    e = {}
    for K in range(1, 8):
        e[str(K)] = {"rows": 10, "censored": 0, "hist": [0] * (K + 1),
                     "counted": [100] * K, "accepted": [round(100 * alphas[j]) for j in range(K)]}
    e["sampled"] = {"lattice": {"counted": [100] * 7, "sum": [100 * a for a in alphas]},
                    "lattice+lp": {"counted": [100] * 7, "sum": [100 * a * lp_scale for a in alphas]}}
    e["sampled_censored"] = [0] * 7
    e["sampled_short"] = [0] * 7
    return e


def _probe_log(path, depths=(4096, 65536)):
    """probe_draft_cost --shapes bolt-lilicorr's output, in its printf formats (tools/probe/probe_draft_cost.cc)."""
    L = ["the experts: 9 blocks a MoE block 3.100 ms, 72 blocks 24.500 ms (x 7.90 for x 8 the blocks): 0.0085 ms a "
         "2.507 MB block (from the 72) - the union-aware form at u distinct experts a MoE block = 40 x u x that (DERIVED)"]
    h = "| M | GEMVs + plain experts + head ms | x M = 1 (pairs) | state slots (1 + M) x 4 MiB ms | replay 3 x 4 MiB ms |"
    for d in depths:
        h += f" attn est. at {d} ms | verify at {d} ms | x M = 1 |"
    L += [h, "|---:|---:|---:|---:|---:|" + "---:|---:|---:|" * len(depths)]
    for M in range(1, 9):
        g = 8.0 + 1.5 * (M - 1)
        slots = 0.3 * (1 + M)
        row = f"| {M} | {g:.3f} | {g / 8.0:.3f} | {slots:.3f} | {0.9:.3f} |"
        for d in depths:
            attn = 0.0139 * d / 1000
            st = 0.6 if M == 1 else slots
            row += f" {attn:.3f} | {g + attn + st:.3f} | {(g + attn + st) / (8.0 + attn + 0.6):.3f} |"
        L.append(row)
    L.append("| M | Mamba GEMVs ms | x M = 1 | attention GEMVs ms | x M = 1 | MoE dense + routers ms | x M = 1 | head + argmax 1 ms "
             "| x M = 1 | plain experts (full - parts) ms |")
    L.append("| 1 | 4.000 | 1.000 | 0.300 | 1.000 | 2.000 | 1.000 | 1.000 | 1.000 | 0.700 |")
    L.append("| arm | ms | x the M = 1 verify's GEMVs (pairs) | weight MB | GB/s |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| the LiLiCorr head alone (bf16, 56 / 7 / 1 rows) | 0.150 | 0.019 | 74 | 490 |")
    L.append("| LiLiCorr block, drafter int4-g64 + int8 head x7 + LiLiCorr head bf16 | 2.000 | 0.250 | 1180 | 590 |")
    L.append("| LiLiCorr block, drafter int8 + int8 head x7 + LiLiCorr head bf16 | 2.900 | 0.360 | 1690 | 580 |")
    L.append("| LiLiCorr block, drafter bf16 + int8 head x7 + LiLiCorr head bf16 | 4.700 | 0.590 | 2760 | 587 |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def test_derived():
    v = lp.derived_verify()
    assert len(v) == 8 and abs(v[0] - 1.0) < 1e-12 and all(a < b for a, b in zip(v, v[1:]))
    pg = lp.plain_gb()
    assert 5.4 < pg < 5.6, pg                                   # spec 25 §3's ~5.48 GB a step (+ KV at 4096)
    assert abs(v[7] - (lp.NONEXPERT_GB + 72 * 40 * 2.5 / 1e3 + 40 * 9 * lp.STATE_MB / 1e3 + 8192 * 4096 / 1e9) / pg) < 1e-12
    b = lp.derived_block()
    assert abs(b["exact"] - 0.50) < 0.01 and abs(b["int8"] - 0.31) < 0.01 and abs(b["int4g64"] - 0.22) < 0.01  # spec 26 §4
    u = {M: (9 + 3 * (M - 1), 9 + 4 * (M - 1)) for M in range(2, 9)}
    vu = lp.union_verify(v, u, 0, None)
    assert vu[0] == v[0] and all(x < y for x, y in zip(vu[1:], v[1:]))
    print(f"derived: a plain step {pg:.3f} GB, verify M = 8 {v[7]:.3f} plain steps (plain experts), {vu[7]:.3f} with a "
          f"union of {u[8][0]}; blocks exact {b['exact']:.3f} / int8 {b['int8']:.3f} / int4 g64 {b['int4g64']:.3f}")


def test_rate():
    E = [1 + 0.8 * k for k in range(1, 8)]
    verify = [1.0 + 0.1 * m for m in range(8)]
    rows, best = lp.rate(E, verify, 0.3)
    assert rows[0] == (1, E[0], verify[1] + 0.3, E[0] / (verify[1] + 0.3)) and best[0] == 7
    rows, best = lp.rate([1.4] * 7, verify, 0.3)
    assert best[0] == 1


def test_probe_parse():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "probe.log")
        _probe_log(log)
        p = lp.parse_probe(log)
        assert p["depth"] == 4096 and p["t_block_ms"] == 0.0085 and not p["plain_measured"]
        r1, r8 = p["rows"][1], p["rows"][8]
        assert abs(p["verify"][7] - (1 + (r8["total_ms"] - r1["total_ms"]) / r1["total_ms"])) < 1e-12
        assert abs(p["block"]["int4g64"] - 2.0 / r1["total_ms"]) < 1e-12 and set(p["block"]) == {"int4g64", "int8", "exact"}
        q = lp.parse_probe(log, 65536, plain_ms=20.0)
        assert q["depth"] == 65536 and q["plain_measured"] and abs(q["block"]["int8"] - 2.9 / 20.0) < 1e-12
        assert abs(q["verify"][3] - (1 + (q["rows"][4]["total_ms"] - q["rows"][1]["total_ms"]) / 20.0)) < 1e-12


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        acc = os.path.join(d, "accept")
        os.makedirs(acc)
        for arm, a, s in (("exacth", 0.85, 0.9), ("int8h", 0.85, 0.9), ("int4g64h", 0.80, 0.9)):
            _accept(os.path.join(acc, f"accept.{arm}.json"), arm,
                    {"code/a": _entry([a] * 7, s), "a4/x": _entry([a - 0.1] * 7, s), "agentic/s": _entry([a - 0.2] * 7, s),
                     "prose/p": _entry([0.5] + [0.2] * 6, s)})
        dumps = os.path.join(d, "dumps")
        os.makedirs(dumps)
        rep = list(range(100, 140))
        tla._write_dump(os.path.join(dumps, "code__a.taps.safetensors"), "code/a", rep + rep[:30], 40)
        tla._write_dump(os.path.join(dumps, "a4__x.taps.safetensors"), "a4/x", list(range(60)), 30)
        tla._write_dump(os.path.join(dumps, "agentic__s.taps.safetensors"), "agentic/s", list(range(80)), 40)
        tla._write_dump(os.path.join(dumps, "prose__p.taps.safetensors"), "prose/p", list(range(50)), 20)
        log = os.path.join(d, "probe.log")
        _probe_log(log)
        uj = os.path.join(d, "union.json")
        with open(uj, "w") as f:
            json.dump({"M": {str(M): {"u_p50": 9 + 3 * (M - 1), "u_p90": 9 + 4 * (M - 1)} for M in range(1, 9)}}, f)
        out, md = os.path.join(d, "p.json"), os.path.join(d, "p.md")
        assert lp.main(["--accept", acc, "--dumps", dumps, "--probe", log, "--union", uj, "--json", out, "--md", md]) == 0
        r = json.load(open(out))
        code = r["results"]["int8h"]["code"]
        assert code["best_k"] >= 1 and code["lilicorr"] > 1.0 and code["lookup"] > 1.0
        assert r["union_results"]["int8h"]["code"]["lilicorr"] > code["lilicorr"]
        text = open(md).read()
        assert "drafter format (Decision 2): int8" in text            # int4 g64 0.05+ below exact at the chosen K
        assert "q form (Decision 6): lattice -" in text and "5.26 (coding)" in text and "AdaptiveK's cost row" in text
        assert "replay lever (Decision 4)" in text and "union MoE (Decision 10): verify at M" in text
        # int4 within 0.05 of exact -> int4 g64; lattice+lp higher -> lattice+lp
        for arm in ("int4g64h",):
            _accept(os.path.join(acc, f"accept.{arm}.json"), arm,
                    {"code/a": _entry([0.85] * 7, 1.2), "a4/x": _entry([0.75] * 7, 1.2), "agentic/s": _entry([0.65] * 7, 1.2),
                     "prose/p": _entry([0.5] + [0.2] * 6, 1.2)})
        assert lp.main(["--accept", acc, "--md", md]) == 0
        text = open(md).read()
        assert "drafter format (Decision 2): int4g64" in text and "q form (Decision 6): lattice+lp" in text
        assert "DERIVED" in text and "PENDING the route records" in text
    print("end to end: the table per arm and corpus, lookup on the same continuations, the union column, the defaults")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
