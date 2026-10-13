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
    for cmd in (["counts"], ["curve", "--f", "0.25:1.0:0.125", "--split", "none"], ["swap", "--f", "0.5", "--S", "1,4", "--B", "2.6",
                "--copy-ms", "0.25", "--step-ms", "12", "--W", "16"], ["mtp", "--K", "1,2,3,4", "--f", "0.5"],
                ["prefill", "--chunk", "128", "--f", "0.5:0.75:0.25"], ["profile", "--out", os.path.join(out, "profile.safetensors")]):
        r = subprocess.run(py + [cmd[0], "--traces", out] + cmd[1:], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stdout[-1500:], r.stderr[-1500:])
    prof = OC.read_profile(os.path.join(out, "profile.safetensors"))
    assert np.array_equal(prof["counts"], OC.counts_of(tr, L, E).astype(np.uint32))
    assert int(prof["mtp_counts"].sum()) == sum(int((t.mtp_ids >= 0).sum()) for t in tr) > 0
    print(f"  tiny 16 x top-4, {T} rows over 2 sources: every subcommand ran; counts = T x L x k; profile round-trips")


# --- P0.8 (Task 4): REAP masks -----------------------------------------------------------------------------------------
def _reap_set(d, E=16, L=2, k=2, T=320):
    """Every expert routed equally often (cycling rows), p = 1 / k; onorm of expert e = e + 1 except experts 9 and 10,
    whose onorm are equal (11.0): REAP's score is (e + 1) x count / k, with an exact tie between 9 and 10. The A4
    scenarios route the low ids only (so calibrating without them moves the frequency mask)."""
    ids = cycling_rows(T, L, k, lambda l: E)
    on = (ids + 1).astype(np.float32)
    on[ids == 9] = 11.0
    on[ids == 10] = 11.0
    write_trace(d, "q4exp_agentic", ids, onorm=on, experts=E)
    a4 = [f"t{i}_define-{f}" for i in (1, 2) for f in ("box", "step", "loader", "openai")]
    for n in a4:
        lo = cycling_rows(64, L, k, lambda l: 4)
        write_trace(d, n, lo, onorm=np.ones(lo.shape, np.float32), experts=E)
    return a4


def test_reap_masks():
    d = os.path.join(TMP, "reap")
    a4 = _reap_set(d)
    E, L, k = 16, 2, 2
    tr = OC.load_traces(d)
    s = OC.reap_scores([t for t in tr if t.cls == "agentic"], L, E, "reap")
    assert np.allclose(s[0], np.array([(e + 1 if e not in (9, 10) else 11.0) * 40 * 0.5 for e in range(E)])), s[0]
    keep = OC.mask_keep(s, 0.375)                       # round(16 x 0.375) = 6: the top 6 by score
    assert keep.sum(1).tolist() == [6, 6]
    assert sorted(np.nonzero(keep[0])[0].tolist()) == [9, 11, 12, 13, 14, 15], np.nonzero(keep[0])
    keep7 = OC.mask_keep(s, 0.4375)                     # 7: the tie (9, 10) at the cut -> both in; at 5 neither
    assert sorted(np.nonzero(keep7[0])[0].tolist()) == [9, 10, 11, 12, 13, 14, 15]
    s5 = s.copy()
    s5[:, 15] = 0.0                                     # now 9 and 10 tie for the 6th of 6 kept: the lower id wins
    assert sorted(np.nonzero(OC.mask_keep(s5, 0.375)[0])[0].tolist()) == [9, 10, 11, 12, 13, 14]
    s6 = s.copy()
    s6[:, 14] = 0.0
    s6[:, 15] = 0.0
    assert sorted(np.nonzero(OC.mask_keep(s6, 0.375)[0])[0].tolist()) == [8, 9, 10, 11, 12, 13]
    s7 = np.zeros((L, E))
    s7[:, 3] = s7[:, 7] = 5.0
    assert np.nonzero(OC.mask_keep(s7, 1 / 16)[0])[0].tolist() == [3], "an exact tie at the cut goes to the lower id"
    try:
        OC.mask_keep(s, 1 / 16, top_k=k)
    except SystemExit as e:
        assert "keeps 1 experts, fewer than top_k 2" in str(e), e
    else:
        raise AssertionError("a mask below top_k was written")
    out = os.path.join(d, "masks")
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py")]
    for extra in ([], ["--holdout-a4"]):
        r = subprocess.run(py + ["reap", "--traces", d, "--sources", "A4,agentic", "--kept", "0.375,0.5",
                                 "--criterion", "reap,freq", "--out", out] + extra, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    m_in = OC.read_mask(os.path.join(out, "mask-reap-0.375.safetensors"))
    m_ho = OC.read_mask(os.path.join(out, "mask-reap-0.375-noa4.safetensors"))
    f_in = OC.read_mask(os.path.join(out, "mask-freq-0.375.safetensors"))
    f_ho = OC.read_mask(os.path.join(out, "mask-freq-0.375-noa4.safetensors"))
    for m in (m_in, m_ho, f_in, f_ho):
        assert m["keep"].dtype == np.uint8 and m["keep"].shape == (L, E) and int(m["mtp_keep"].sum()) == E
        assert m["meta"]["format"] == "spec22-mask-1" and m["meta"]["per_layer"] == "uniform"
        assert float(m["meta"]["kept_fraction"]) == 0.375
    cal_in = {x["name"] for x in json.loads(m_in["meta"]["calibration"])}
    cal_ho = {x["name"] for x in json.loads(m_ho["meta"]["calibration"])}
    assert cal_in == set(a4) | {"q4exp_agentic"} and cal_ho == {"q4exp_agentic"}, (cal_in, cal_ho)
    assert m_in["meta"]["criterion"] == m_ho["meta"]["criterion"] == "reap"
    for key in ("model", "checkpoint_revision", "kept_fraction", "per_layer", "format"):
        assert m_in["meta"][key] == m_ho["meta"][key], key            # --holdout-a4 changes the calibration only
    assert sorted(np.nonzero(f_ho["keep"][0])[0].tolist()) == [0, 1, 2, 3, 4, 5]          # all equal: lower ids
    assert sorted(np.nonzero(f_in["keep"][0])[0].tolist()) == [0, 1, 2, 3, 4, 5]
    js = json.load(open(os.path.join(out, "reap.json")))
    assert "overlap" in js and js["overlap"]["0.375"]["all"][0] == len({9, 11, 12, 13, 14, 15} & {0, 1, 2, 3, 4, 5})
    print("  REAP keeps the analytic top 6 (ties to the lower id), freq the lower ids; --holdout-a4 drops the A4 "
          "traces from the calibration and nothing else; below top_k refused; the overlap written")


def test_reap_folds():
    """Decision 9 (operator, 2026-10-10): --kfold 4 --fold i calibrates without fold i's A4 scenarios (keeping every
    non-A4 source); the 4 folds partition the A4 set; --fold all writes the 4 masks."""
    d = os.path.join(TMP, "folds")
    E, L, k = 16, 2, 2
    write_trace(d, "q4exp_agentic", cycling_rows(320, L, k, lambda l: E), experts=E)
    a4 = [f"t{i}_{t}-{f}" for i, t in enumerate(("define", "explain", "rename", "comment", "test", "fix"), 1)
          for f in ("box", "linear_l0", "loader", "openai", "rotation", "step")]
    for n in a4:
        write_trace(d, n, cycling_rows(16, L, k, lambda l: 4), experts=E)
    write_trace(d, "2026-10-01_turn3", cycling_rows(16, L, k, lambda l: 8), experts=E)     # an opencode recording
    folds = OC.a4_folds(a4, 4)
    assert sorted(sum(1 for n in a4 if folds[n] == i) for i in range(4)) == [9, 9, 9, 9]
    out = os.path.join(d, "masks")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "offload_curve.py"), "reap", "--traces", d, "--sources",
                        "A4,agentic,opencode", "--kept", "0.5", "--kfold", "4", "--fold", "all", "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    seen = set()
    for i in range(4):
        m = OC.read_mask(os.path.join(out, f"mask-reap-0.5-fold{i}.safetensors"))
        cal = {x["name"] for x in json.loads(m["meta"]["calibration"])}
        held = {n for n in a4 if folds[n] == i}
        assert cal == (set(a4) - held) | {"q4exp_agentic", "2026-10-01_turn3"}, i
        assert json.loads(m["meta"]["held_out"]) == sorted(held) and m["meta"]["fold"] == f"{i}/4"
        seen |= held
    assert seen == set(a4)
    assert os.path.exists(os.path.join(out, "folds.json"))
    print("  --kfold 4 --fold all: 4 masks, each calibrated without its 9 A4 scenarios (agentic + opencode kept)")


def test_prune_report():
    """prune-report: the curve table from maskeval JSONs and the A4 outputs - in sample (the all-traces mask) and the
    4-fold held-out accuracy (each scenario on its fold's mask), the gap, the experts' bytes and the two-card fit."""
    d = os.path.join(TMP, "prune")
    ev, a4d = os.path.join(d, "maskeval"), os.path.join(d, "a4")
    os.makedirs(ev, exist_ok=True)
    os.makedirs(a4d, exist_ok=True)
    names = [f"t{i}_define-{f}" for i in range(1, 3) for f in ("box", "step", "loader", "openai")]
    # the folds reap wrote (folds.json) - deliberately NOT a4_folds' order over these names: prune-report must read them
    folds = {n: i % 4 for i, n in enumerate(sorted(names))}
    assert folds != OC.a4_folds(names, 4)
    fj = os.path.join(d, "folds.json")
    json.dump({"kfold": 4, "folds": folds}, open(fj, "w"))
    call ="<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
    other = "<tool_call>\n<function=grep>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>"
    for n in names:
        open(os.path.join(a4d, f"{n}.bf16.txt"), "w").write(call)
        open(os.path.join(a4d, f"{n}.mask-reap-0.5.txt"), "w").write(call)                  # in sample: all match
        bad = folds[n] == 0                                                                    # held out: fold 0 fails
        open(os.path.join(a4d, f"{n}.mask-reap-0.5-fold{folds[n]}.txt"), "w").write(other if bad else call)
    for kept, kl in ((0.5, 0.25), (0.875, 0.01)):
        json.dump({"prompt": "q4exp_short", "set": "golden", "kept": kept, "criterion": "reap", "tag": f"mask-reap-{kept}",
                   "kl_mean": kl, "kl_p99": kl * 4, "argmax": 1 - kl, "rows": 100},
                  open(os.path.join(ev, f"q4exp_short.mask-reap-{kept}.json"), "w"))
    budget = os.path.join(d, "budget.json")
    open(budget, "w").write(json.dumps({"max_len": 32768, "devices": 2, "mtp": True, "prefill": True, "dense": "int4",
                                        "cache_total": 50764655176, "routed_total": 64172851200, "f": 0.791061,
                                        "experts": 512, "layers": 48, "expert_bytes": 2611200}) + "\n")
    cmd = [sys.executable, os.path.join(_HERE, "offload_curve.py"), "prune-report", "--dir", ev, "--a4", a4d,
           "--budget", budget, "--folds", fj, "--json", os.path.join(d, "prune.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    # a scenario the folds do not list is refused (its held-out run would be graded on a mask that saw it)
    fj2 = os.path.join(d, "folds-short.json")
    json.dump({"kfold": 4, "folds": {n: v for n, v in folds.items() if n != names[0]}}, open(fj2, "w"))
    r2 = subprocess.run(cmd[:cmd.index("--folds") + 1] + [fj2] + cmd[cmd.index("--folds") + 2:], capture_output=True,
                        text=True)
    assert r2.returncode != 0 and names[0] in (r2.stdout + r2.stderr), r2.stdout[-800:] + r2.stderr[-800:]
    j = json.load(open(os.path.join(d, "prune.json")))
    row = {x["kept"]: x for x in j["rows"]}[0.5]
    n0 = sum(1 for n in names if folds[n] == 0)
    assert row["a4_in_sample"] == 1.0 and abs(row["a4_held_out"] - (len(names) - n0) / len(names)) < 1e-12, row
    assert abs(row["a4_gap"] - n0 / len(names)) < 1e-12 and row["kl_mean_golden"] == 0.25
    assert row["experts_gb"] == round(48 * 256 * 2611200 / 1e9, 3) and row["fits_two_cards"] is True
    assert {x["kept"]: x for x in j["rows"]}[0.875]["fits_two_cards"] is False
    print(f"  prune-report: in sample 1.0, held out {row['a4_held_out']:.3f}, gap {row['a4_gap']:.3f}; 50 % "
          f"({row['experts_gb']} GB) fits two cards, 87.5 % does not")


# --- P0.7 (Task 4 Step 4): the projection ------------------------------------------------------------------------------
def test_project_spec_table():
    """P = 12 GB/s, BW = 590 GB/s, D = 3.712 GB, R = 1.253 GB and spec 22 §2's h reproduce its serial and overlapped
    t/s columns to 0.1; mix at alpha 0 / 1 is overlapped / serial."""
    table = {1.00: (118.8, 118.8), 0.99: (106.0, 119.1), 0.95: (73.9, 86.9), 0.90: (53.6, 59.8), 0.80: (34.6, 36.8),
             0.50: (16.8, 17.1)}
    for h, (ser, ovl) in table.items():
        ts = 1e3 / OC.decode_ms(3.712e9, 1.253e9, 590.0, 12.0, h, "serial")
        to = 1e3 / OC.decode_ms(3.712e9, 1.253e9, 590.0, 12.0, h, "overlapped")
        assert round(ts, 1) == ser and round(to, 1) == ovl, (h, ts, to)
        assert OC.decode_ms(3.712e9, 1.253e9, 590, 12, h, "mix", 1.0) == OC.decode_ms(3.712e9, 1.253e9, 590, 12, h, "serial")
        assert OC.decode_ms(3.712e9, 1.253e9, 590, 12, h, "mix", 0.0) == OC.decode_ms(3.712e9, 1.253e9, 590, 12, h, "overlapped")
    print("  spec 22 §2's table reproduced to 0.1 t/s (serial and overlapped, h 1.00 .. 0.50)")


def _project_inputs(d, h):
    os.makedirs(d, exist_ok=True)
    rr = 48 * 10 * 2611200
    lines = []
    for dense in ("int4", "bf16"):
        for devices in (1, 2):
            for L in (4096, 32768, 131072):
                first, end = ([0], [48]) if devices == 1 else ([0, 24], [24, 48])
                cache = [26e9] if devices == 1 else [25.9e9, 24.9e9]
                lines.append({"max_len": L, "devices": devices, "mtp": True, "prefill": True, "int8_head": True,
                              "dense": dense, "layers": 48, "experts": 512, "top_k": 10, "expert_bytes": 2611200,
                              "read_routed": rr, "read_non_routed": 3.712e9 if dense == "int4" else 8.0e9,
                              "read_mtp_step": 0.845e9, "first": first, "end": end, "cache": cache,
                              "cache_total": sum(cache), "routed_total": 48 * 512 * 2611200,
                              "f": sum(cache) / (48 * 512 * 2611200)})
    open(os.path.join(d, "budget.json"), "w").write("".join("budget-json " + json.dumps(x) + "\n" for x in lines))
    rows = [{"f": f, "uniform_held": h, "perlayer_held": h - 0.01, "uniform_in": h, "perlayer_in": h}
            for f in OC.frange("0.30:1.00:0.05")]
    json.dump({"split": "hash", "rows": rows}, open(os.path.join(d, "curve.json"), "w"))
    po = ["po info mode=zc dev=0 pairs=4", "po zc dev=0 card=0 P=26 law_summary law=sum alpha=1.000 tA_us=1 tC_us=2 "
          "P_gbps=12.000 P_marginal_gbps=13.0", "po zc dev=1 card=1 P=26 law_summary law=sum alpha=1.000 tA_us=1 tC_us=2 "
          "P_gbps=12.000 P_marginal_gbps=13.0", "po zc dev=0 card=0 P=7 law_summary law=max alpha=0.000 tA_us=1 tC_us=2 "
          "P_gbps=99.000 P_marginal_gbps=99", "po copy dev=both card=all list=regular ordinal=copy shape=256_experts_one "
          "bytes=1336934400 us=60000.0 aggregate_gbps=22.000"]
    open(os.path.join(d, "p0.txt"), "w").write("\n".join(po) + "\n")
    json.dump({"K": [{"K": K, "M": K + 1, "distinct_per_layer": 10.0 * (K + 1) * 0.8} for K in (1, 2, 3, 4)]},
              open(os.path.join(d, "mtp.json"), "w"))
    json.dump({"rows": [{"C": C, "f": f, "streamed_gb_per_chunk": (1 - f) * 64.17} for C in (2048, 4096)
                        for f in OC.frange("0.60:0.90:0.05")]}, open(os.path.join(d, "prefill.json"), "w"))


def test_project_stop_flips():
    """The last line is STOP below 40 t/s and GO-CANDIDATE above (two cards, 32k, ours, the best fill, the measured law
    - the largest P's - and P per card): h 0.83 projects ~38.6 t/s, h 0.85 ~41.9 (serial, P 12, BW 590)."""
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py"), "project"]
    for h, want in ((0.83, "STOP"), (0.85, "GO-CANDIDATE")):
        d = os.path.join(TMP, f"project-{h}")
        _project_inputs(d, h)
        js = os.path.join(d, "project.json")
        r = subprocess.run(py + ["--budget", os.path.join(d, "budget.json"), "--p0", os.path.join(d, "p0.txt"), "--curve",
                                 os.path.join(d, "curve.json"), "--mtp", os.path.join(d, "mtp.json"), "--prefill",
                                 os.path.join(d, "prefill.json"), "--bw", "590", "--json", js],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        last = r.stdout.strip().splitlines()[-1]
        assert last.startswith(want + ":"), last
        j = json.load(open(js))
        assert j["verdict"] == want and j["law"] == "serial" and j["P"] == {"0": 12.0, "1": 12.0}, (j["law"], j["P"])
        D = 3.712e9 + OC.attn_bytes(32768)
        tps = 1e3 / OC.decode_ms(D, 48 * 10 * 2611200, 590.0, 12.0, h, "serial")
        assert abs(j["stop_inputs"]["tps"] - tps) < 1e-9, (j["stop_inputs"], tps)
        assert j["stop_inputs"]["policy"] == "uniform (pooled curve)"      # the better fill; no per-device curve here
        assert {m["K"] for m in j["mtp"]} == {1, 2, 3} and all("PROPOSED" in m["accept_source"] for m in j["mtp"])
        assert [p["C"] for p in j["prefill"]] == [2048, 4096] and j["prefill"][1]["derived"]
        print(f"  h {h}: {last[:90]}...")


def test_project_per_device_and_swaps():
    """Each card's h is scored on its own layers at its own f (curve --budget writes per_device); the swap simulation's
    gain applies only where its f matches the card's, with its copy traffic charged to the link."""
    d = os.path.join(TMP, "project-pd")
    tr = os.path.join(d, "traces")
    L, k, E, T = 4, 2, 64, 200
    rng = np.random.default_rng(9)
    for i in range(8):                       # sources in both halves: layers 0-1 always experts 0-1, layers 2-3 uniform
        ids = np.zeros((T, L, k), np.int64)
        ids[:, :2, :] = [0, 1]
        for t in range(T):
            for l in (2, 3):
                ids[t, l] = rng.choice(E, k, replace=False)
        write_trace(tr, f"src{i}", ids, experts=E)
    eb, P = 2611200, 12.0
    routed_dev = 2 * E * eb                  # two layers a card
    lines = []
    for L_ctx, f_card in ((4096, 0.9), (32768, 0.5)):
        cache = [f_card * routed_dev, f_card * routed_dev]
        lines.append({"max_len": L_ctx, "devices": 2, "mtp": True, "prefill": True, "int8_head": True, "dense": "int4",
                      "layers": L, "experts": E, "top_k": k, "expert_bytes": eb, "read_routed": L * k * eb,
                      "read_non_routed": 1.0e9, "read_mtp_step": 0.2e9, "first": [0, 2], "end": [2, 4], "cache": cache,
                      "cache_total": sum(cache), "routed_total": L * E * eb, "f": f_card})
    budget = os.path.join(d, "budget.json")
    open(budget, "w").write("".join("budget-json " + json.dumps(x) + "\n" for x in lines))
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py")]
    cj = os.path.join(d, "curve.json")
    r = subprocess.run(py + ["curve", "--traces", tr, "--f", "0.1:1.0:0.1", "--budget", budget, "--json", cj],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    c = json.load(open(cj))
    b32 = next(x for x in c["budget"] if x["max_len"] == 32768)
    pd = b32["per_device"]
    assert len(pd) == 2 and pd[0]["h_uniform"] == 1.0 and pd[1]["h_uniform"] < 0.75, pd
    sw = os.path.join(d, "swap.json")
    json.dump({"f": 0.5, "fill": "uniform", "copy_ms": 0.2, "step_ms": 25, "static": 0.6,
               "rows": [{"S": 4, "B_mb": 2.6112, "W": 64, "h": 0.7, "gain": 0.1, "moved_per_step": 0.25}]},
              open(sw, "w"))
    p0 = os.path.join(d, "p0.txt")
    open(p0, "w").write("po zc dev=0 card=0 P=26 law_summary law=sum alpha=1.000 tA_us=1 tC_us=2 P_gbps=12.000 "
                        "P_marginal_gbps=1\npo zc dev=1 card=1 P=26 law_summary law=sum alpha=1.000 tA_us=1 tC_us=2 "
                        "P_gbps=12.000 P_marginal_gbps=1\n")
    js = os.path.join(d, "project.json")
    r = subprocess.run(py + ["project", "--budget", budget, "--p0", p0, "--curve", cj, "--swap", sw, "--bw", "590",
                             "--json", js], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    j = json.load(open(js))
    rows = {x["max_len"]: x for x in j["rows"]}
    assert "swaps" not in rows[4096]["policy"], rows[4096]           # f 0.9: not where the swaps were simulated
    assert "swaps" in rows[32768]["policy"], rows[32768]
    assert rows[32768]["h"][0] == 1.0                                 # card 0's own layers, not the pooled curve
    # the time: per card, the static (or swapped) h on its own layers, + the swap copies on the link
    want = 0.0
    for i in range(2):
        hs = pd[i]
        h_static = max(hs["h_uniform"], hs["h_perlayer"])
        h_sw = min(1.0, hs["h_uniform"] + 0.1)
        Ri = 2 * k * eb
        Di = (1.0e9 + OC.attn_bytes(32768)) * 0.5
        t_static = OC.decode_ms(Di, Ri, 590.0, P, h_static, "serial")
        t_swap = OC.decode_ms(Di, Ri, 590.0, P, h_sw, "serial") + 0.25 * 0.5 * eb / P / 1e6
        want += min(t_static, t_swap)
    assert abs(rows[32768]["ms"] - want) < 1e-9, (rows[32768]["ms"], want)
    print(f"  per card h {rows[32768]['h']} on its own layers; swaps applied at f 0.5 only, their copies charged")


TESTS = [test_curve_analytic, test_holdout_disjoint, test_swap_never_lands_is_static, test_mtp_unions,
         test_prefill_repeated_row, test_counts_and_profile_roundtrip, test_cli_synthetic, test_tiny_model_traces,
         test_reap_masks, test_reap_folds, test_prune_report, test_project_spec_table, test_project_stop_flips,
         test_project_per_device_and_swaps]


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
