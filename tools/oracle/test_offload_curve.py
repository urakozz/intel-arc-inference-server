#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for offload_curve.py (spec 22 P0.6 / P0.8 / P0.7, plan 22a Tasks 3-4). CPU, seconds (the tiny-model test:
a minute).

    test_offload_curve.py [test_name ...]          in agnes-ref-img; PYTHONPATH=<the 5.19.0 site> for the tiny test

Synthetic traces built here with known answers (21a's TRACE_FORMAT, written with safetensors):
- h(f) analytic: layer l routes only experts [0, 10 + l), every one equally often -> the uniform fill holds [0, n)
  (ties to the lower id), h = mean_l min(n, 10 + l) / (10 + l);
- a hold-out split whose halves see disjoint experts scores h = 0 held out and 1 in sample;
- a swap simulation whose copy never lands (--copy-ms inf) equals the static fill exactly;
- M = K + 1 unions: repeated rows give 10 a layer, disjoint rows 10 M;
- prefill: chunks of one repeated row stream 10 experts a layer;
- counts sum to T x L x k; the profile round-trips;
- the tiny model's own traces (qwen4exp_ref.py trace on a 16-expert / top-4 tiny made by qwen4exp_make_tiny.py over
  q4exp_agentic and code, at most 512 ids each): every subcommand runs.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OC = _load("offload_curve")
REPO = os.path.dirname(os.path.dirname(_HERE))
TMP = tempfile.mkdtemp(prefix="offload_curve_test_")


class Skip(Exception):
    pass


def write_trace(d, name, ids, p=None, onorm=None, mtp_ids=None, experts=64, mtp=True):
    """ids int [T][L][k] (each row's k distinct ids, sorted here); p / onorm default 1 / k and 1."""
    os.makedirs(d, exist_ok=True)
    ids = np.sort(np.asarray(ids, dtype=np.int32), axis=-1)
    T, L, k = ids.shape
    p = np.full((T, L, k), 1.0 / k, np.float32) if p is None else np.asarray(p, np.float32)
    onorm = np.ones((T, L, k), np.float32) if onorm is None else np.asarray(onorm, np.float32)
    t = {"ids": ids, "p": p, "onorm": onorm}
    if mtp:
        m = np.sort(np.asarray(mtp_ids if mtp_ids is not None else ids[:, 0, :], np.int32), axis=-1).copy()
        m[-1] = -1
        t["mtp_ids"] = m
        t["mtp_p"] = np.where(m >= 0, 1.0 / k, 0.0).astype(np.float32)
        t["mtp_onorm"] = np.where(m >= 0, 1.0, 0.0).astype(np.float32)
    meta = {"model": "synthetic", "checkpoint": "/synthetic", "checkpoint_revision": "synth0", "source": name,
            "T": str(T), "layers": str(L), "top_k": str(k), "experts": str(experts), "format": "test"}
    save_file(t, os.path.join(d, f"{name}.routes.safetensors"), metadata=meta)


def cycling_rows(T, L, k, span_of_layer, offset=0):
    """Row t of layer l: k consecutive ids (mod span) starting at t k - every id of [offset, offset + span) equally
    often when T k is a multiple of span."""
    ids = np.zeros((T, L, k), np.int64)
    for l in range(L):
        s = span_of_layer(l)
        for t in range(T):
            ids[t, l] = [offset + (t * k + j) % s for j in range(k)]
    return ids


def test_curve_analytic():
    d = os.path.join(TMP, "analytic")
    L, k, E = 4, 2, 64
    # T k a multiple of every span 10 .. 13: T = 10 * 11 * 12 * 13 / 2 = 8580
    T = 8580
    write_trace(d, "q4exp_agentic", cycling_rows(T, L, k, lambda l: 10 + l), experts=E)
    tr = OC.load_traces(d)
    assert len(tr) == 1 and tr[0].cls == "agentic"
    counts = OC.counts_of(tr, L, E)
    assert int(counts.sum()) == T * L * k
    for f in (0.0625, 0.125, 0.15625, 0.5):
        n = int(round(f * E))
        want = sum(min(n, 10 + l) / (10 + l) for l in range(L)) / L
        res = OC.resident_uniform(counts, f)
        assert int(res.sum(1)[0]) == n
        if n < 10:
            assert bool(res[:, :n].all()) and not bool(res[:, n:].any()), "ties go to the lower id"
        hits, total = OC.hit_rate(tr, res)
        assert total == T * L * k and abs(hits / total - want) < 1e-12, (f, hits / total, want)
        # per layer from the global greedy: never below the uniform fill on its own training data
        hp, tp = OC.hit_rate(tr, OC.resident_perlayer(counts, f))
        assert hp >= hits and tp == total, (f, hp, hits)
    r = OC.curve(tr, L, E, OC.frange("0.0625:0.5:0.0625"), split="none")
    row = [x for x in r["rows"] if abs(x["f"] - 0.125) < 1e-9][0]
    assert abs(row["uniform_in"] - sum(min(8, 10 + l) / (10 + l) for l in range(L)) / L) < 1e-12
    print(f"  h(f) analytic at f = 1/16, 1/8, 5/32, 1/2; per-layer >= uniform; T x L x k = {T * L * k} routes counted")


def _two_halves():
    names = [f"src{i}" for i in range(64)]
    h = OC.halves(names)
    a = next(n for n in names if h[n] == 0)
    b = next(n for n in names if h[n] == 1)
    return a, b


def test_holdout_disjoint():
    d = os.path.join(TMP, "disjoint")
    a, b = _two_halves()
    L, k, E, T = 3, 2, 64, 400
    write_trace(d, a, cycling_rows(T, L, k, lambda l: 8, offset=0), experts=E)
    write_trace(d, b, cycling_rows(T, L, k, lambda l: 8, offset=32), experts=E)
    tr = OC.load_traces(d)
    r = OC.curve(tr, L, E, [8 / 64], split="hash")
    row = r["rows"][0]
    assert row["uniform_held"] == 0.0 and row["uniform_in"] == 1.0, row
    assert row["perlayer_held"] == 0.0 and row["perlayer_in"] == 1.0, row
    # A4 scenarios split across both halves, balanced
    a4 = [f"t{i}_task-{j}" for i in range(1, 7) for j in "abcdef"]
    hh = OC.halves(a4 + ["code", "prose"])
    assert sum(hh[n] for n in a4) == 18, "the 36 A4 scenarios split 18 / 18"
    assert OC.source_class("t3_rename-box") == "A4" and OC.source_class("code") == "code"
    assert OC.source_class("q4exp_agentic") == "agentic" and OC.source_class("prose") == "prose"
    assert OC.source_class("2026-10-01_turn3") == "opencode"
    print(f"  {a} / {b} in different halves: held out 0, in sample 1 (uniform and per layer); A4 18 / 18")


def test_swap_never_lands_is_static():
    d = os.path.join(TMP, "swap")
    L, k, E, T = 3, 4, 64, 600
    rng = np.random.default_rng(3)
    ids = np.stack([np.stack([rng.choice(E, k, replace=False) for _ in range(L)]) for _ in range(T)])
    write_trace(d, "q4exp_agentic", ids, experts=E)
    tr = OC.load_traces(d)
    counts = OC.counts_of(tr, L, E)
    res = OC.resident_uniform(counts, 0.25)
    static, total = OC.hit_rate(tr, res)
    for S in (1, 4, 16):
        for W in (8, 64):
            h, t = OC.swap_sim(tr[0], res, S=S, B=10 * 2611200, W=W, copy_ms=math.inf, step_ms=10.0,
                               expert_bytes=2611200)
            assert t == total and h == static, (S, W, h, static)
    # and a copy that lands can only change h by replacing residents (sanity: it runs, h in [0, 1])
    h, t = OC.swap_sim(tr[0], res, S=4, B=4 * 2611200, W=64, copy_ms=0.1, step_ms=10.0, expert_bytes=2611200)
    assert 0 <= h <= t == total
    print(f"  copy never lands: swap == static ({static}/{total}) at S 1 / 4 / 16, W 8 / 64; a landing copy: "
          f"{h}/{t}")


def test_mtp_unions():
    d = os.path.join(TMP, "mtp")
    L, k, E, T = 2, 10, 512, 50
    rep = np.tile(np.arange(10), (T, L, 1))
    write_trace(d, "q4exp_agentic", rep, experts=E)
    dis = np.stack([np.stack([np.arange(10 * t, 10 * t + 10) % E for _ in range(L)]) for t in range(T)])
    write_trace(d, "code", dis, experts=E)
    tr = {t.name: t for t in OC.load_traces(d)}
    none = np.zeros((L, E), bool)
    for K in (1, 2, 3, 4):
        M = K + 1
        u = OC.mtp_unions(tr["q4exp_agentic"], K, none)
        assert u["distinct_per_layer"] == 10.0, (K, u)
        u = OC.mtp_unions(tr["code"], K, none)
        assert u["distinct_per_layer"] == 10.0 * M, (K, u)
        assert u["misses_per_step"] == 10.0 * M * L
    print("  M = K + 1 unions: repeated rows 10 a layer, disjoint rows 10 M (K = 1..4)")


def test_prefill_repeated_row():
    d = os.path.join(TMP, "prefill")
    L, k, E, T = 3, 10, 512, 4096
    write_trace(d, "q4exp_agentic", np.tile(np.arange(100, 110), (T, L, 1)), experts=E)
    tr = OC.load_traces(d)
    none = np.zeros((L, E), bool)
    for C in (2048, 4096):
        r = OC.prefill_bytes(tr[0], C, none, expert_bytes=2611200)
        assert r["chunks"] == T // C and r["unique_per_layer"] == 10.0, r
        assert r["streamed_bytes_per_chunk"] == L * 10 * 2611200, r
    print("  prefill: chunks of one repeated row stream 10 experts a layer (2048, 4096)")


def test_counts_and_profile_roundtrip():
    d = os.path.join(TMP, "profile")
    L, k, E, T = 4, 3, 32, 200
    rng = np.random.default_rng(5)
    ids = np.stack([np.stack([rng.choice(E, k, replace=False) for _ in range(L)]) for _ in range(T)])
    write_trace(d, "q4exp_agentic", ids, experts=E)
    write_trace(d, "t1_define-box", ids[:50], experts=E)
    tr = OC.load_traces(d)
    c = OC.counts_of(tr, L, E)
    assert int(c.sum()) == 250 * L * k
    out = os.path.join(d, "profile.safetensors")
    OC.write_profile(out, tr, L, E)
    prof = OC.read_profile(out)
    assert np.array_equal(prof["counts"], c.astype(np.uint32)) and prof["counts"].dtype == np.uint32
    mc = np.zeros(E, np.int64)
    for t in tr:
        m = t.mtp_ids[t.mtp_ids >= 0]
        np.add.at(mc, m, 1)
    assert np.array_equal(prof["mtp_counts"], mc.astype(np.uint32))
    meta = prof["meta"]
    assert meta["format"] == "spec22-profile-1" and meta["tokens"] == "250"
    tl = json.loads(meta["traces"])
    assert {x["name"] for x in tl} == {"q4exp_agentic", "t1_define-box"}
    for x in tl:
        h = hashlib.sha256(open(os.path.join(d, x["name"] + ".routes.safetensors"), "rb").read()).hexdigest()
        assert x["sha256"] == h
    print(f"  counts sum to T x L x k = {250 * L * k}; the profile round-trips (counts, mtp_counts, metadata)")


def test_cli_synthetic():
    """Every Task 3 subcommand through the CLI on synthetic traces, JSON beside the text."""
    d = os.path.join(TMP, "cli")
    L, k, E, T = 3, 4, 64, 300
    rng = np.random.default_rng(7)
    for name in ("q4exp_agentic", "code", "prose", "t1_define-box", "t2_explain-step"):
        zipf = 1.0 / np.arange(1, E + 1) ** 1.1
        ids = np.stack([np.stack([rng.choice(E, k, replace=False, p=zipf / zipf.sum()) for _ in range(L)])
                        for _ in range(T)])
        write_trace(d, name, ids, experts=E)
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py")]
    for cmd in (["counts"], ["curve", "--f", "0.1:1.0:0.1"], ["swap", "--f", "0.5", "--S", "1,4", "--B", "2.6,26",
                "--copy-ms", "0.25", "--step-ms", "12", "--W", "16,64"], ["mtp", "--K", "1,2,3", "--f", "0.5"],
                ["prefill", "--chunk", "128,256", "--f", "0.6:0.9:0.1"], ["profile", "--out", os.path.join(d, "p.safetensors")]):
        js = os.path.join(d, cmd[0] + ".json")
        r = subprocess.run(py + [cmd[0], "--traces", d, "--json", js] + cmd[1:], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stdout[-1500:], r.stderr[-1500:])
        assert os.path.exists(js) and json.load(open(js)), cmd
    print("  counts / curve / swap / mtp / prefill / profile: exit 0, JSON written")


def test_tiny_model_traces():
    """qwen4exp_ref.py trace on a 16-expert / top-4 tiny (the downloaded tiny has 4 experts / top-2), over
    q4exp_agentic and code (at most 512 ids each), then every subcommand; counts sum to T x L x k; the profile round-trips."""
    try:
        import transformers
        if transformers.__version__ != "5.19.0":
            raise ImportError(transformers.__version__)
    except ImportError as e:
        raise Skip(f"transformers 5.19.0 is not on PYTHONPATH ({e}): tools/oracle/qwen4exp_env.sh")
    mk = _load("qwen4exp_make_tiny")
    snap = os.path.join(TMP, "tiny16")
    mk.make(snap, hidden=128, layer_types="lllq", experts=16, topk=4, form="tiny", mtp=True, seed=11)
    out = os.path.join(TMP, "tiny16-traces")
    srcs = []
    for n in ("q4exp_agentic", "code"):
        srcs += ["--source", f"{n}:{os.path.join(REPO, 'tests', 'golden', 'prompts', n + '.ids')}"]
    r = subprocess.run([sys.executable, os.path.join(_HERE, "qwen4exp_ref.py"), "trace", snap, "--out", out,
                        "--max-tokens", "512"] + srcs, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    tr = OC.load_traces(out)
    assert {t.name for t in tr} == {"q4exp_agentic", "code"}
    L, E, k = tr[0].layers, tr[0].experts, tr[0].top_k
    assert (L, E, k) == (4, 16, 4), (L, E, k)
    T = sum(t.T for t in tr)
    assert int(OC.counts_of(tr, L, E).sum()) == T * L * k
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py")]
    for cmd in (["counts"], ["curve", "--f", "0.25:1.0:0.125"], ["swap", "--f", "0.5", "--S", "1,4", "--B", "2.6",
                "--copy-ms", "0.25", "--step-ms", "12", "--W", "16"], ["mtp", "--K", "1,2,3,4", "--f", "0.5"],
                ["prefill", "--chunk", "128", "--f", "0.5:0.75:0.25"], ["profile", "--out", os.path.join(out, "profile.safetensors")]):
        r = subprocess.run(py + [cmd[0], "--traces", out] + cmd[1:], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stdout[-1500:], r.stderr[-1500:])
    prof = OC.read_profile(os.path.join(out, "profile.safetensors"))
    assert np.array_equal(prof["counts"], OC.counts_of(tr, L, E).astype(np.uint32))
    assert int(prof["mtp_counts"].sum()) == sum(int((t.mtp_ids >= 0).sum()) for t in tr) > 0
    print(f"  tiny 16 x top-4, {T} rows over 2 sources: every subcommand ran; counts = T x L x k; profile round-trips")


TESTS = [test_curve_analytic, test_holdout_disjoint, test_swap_never_lands_is_static, test_mtp_unions,
         test_prefill_repeated_row, test_counts_and_profile_roundtrip, test_cli_synthetic, test_tiny_model_traces]


def main() -> None:
    import traceback
    only = sys.argv[1:]
    skipped, failed = 0, []
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
        except Exception:      # noqa: BLE001 - report every test, then fail
            traceback.print_exc()
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
