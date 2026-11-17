#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Checks for offload_curve.py (spec 22 P0.6 / P0.7, plan 22a Tasks 3-4). CPU, seconds (the tiny-model test:
a minute).

    test_offload_curve.py [test_name ...]          in agnes-ref-img; PYTHONPATH=<the 5.19.0 site> for the tiny test

Synthetic traces built here with known answers (21-QFN-v1a's TRACE_FORMAT, written with safetensors):
- h(f) analytic: layer l routes only experts [0, 10 + l), every one equally often -> the uniform fill holds [0, n)
  (ties to the lower id), h = mean_l min(n, 10 + l) / (10 + l);
- a hold-out split whose halves see disjoint experts scores h = 0 held out and 1 in sample;
- a swap simulation whose copy never lands (--copy-ms inf) equals the static fill exactly;
- M = K + 1 unions: repeated rows give 10 a layer, disjoint rows 10 M;
- prefill: chunks of one repeated row stream 10 experts a layer;
- counts sum to T x L x k; the profile round-trips;
- Laguna's shape (spec 30): §7.1's decode table (+-1 t/s) and §8's drafter table (+-0.02) from the formulas; prompt and
  answer rows scored apart; verify unions to 8 rows (u(M)); `project --shape laguna` ends in the stop rule's verdict;
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


def write_trace(d, name, ids, p=None, onorm=None, mtp_ids=None, experts=64, mtp=True, answer_from=None):
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
    if answer_from is not None:
        meta["answer_from"] = str(answer_from)
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


def test_tp_spec_table():
    """Spec 21-QFN-v2 §5's table (BW 457.8, P 12): (b) tp at D 2.38 / R 0.627 a card + 0.34 ms, and (a) pp at the D the
    table was built with (3.46 GB: non-routed 3.04 + the 0.42 fixed; the text's 3.76 gives ~4 t/s less), to 1 t/s;
    tp_card_bytes halves all but the hyper-connections."""
    table = {1.00: (97, 145), 0.97: (75, 118), 0.95: (65, 106), 0.90: (49, 83), 0.85: (39, 69)}
    for h, (pp, tp) in table.items():
        assert abs(1e3 / OC.decode_ms(3.46e9, 1.253e9, 457.8, 12.0, h) - pp) <= 1.0, (h, pp)
        assert abs(1e3 / OC.tp_ms(2.38e9, 0.627e9, 457.8, 12.0, h) - tp) <= 1.0, (h, tp)
    Dc, Rc = OC.tp_card_bytes(3.47e9, 1.254e9)
    assert abs(Dc - 2.38e9) < 1e6 and Rc == 0.627e9, (Dc, Rc)
    print("  spec 21-QFN-v2 §5's (a) and (b) columns reproduced to 1 t/s (h 1.00 .. 0.85)")


def test_swap_policies():
    """basalt's and saber's placements: a copy that never lands is the static fill (h and 0 copies); saber promotes an
    expert on its second miss within 8 steps (a layer alternating between two cold experts gains), and reads through
    when a layer misses on more than half its slots."""
    L, k, E, T = 2, 2, 16, 200
    ids = np.zeros((T, L, k), np.int64)
    ids[:, :, 0] = 0
    ids[:, :, 1] = 8 + (np.arange(T) % 2)[:, None]       # 8, 9, 8, 9 ...: both cold under the fill below
    d = os.path.join(TMP, "policies")
    write_trace(d, "q4exp_agentic", ids, experts=E, mtp=False)
    t = OC.load_traces(d)[0]
    prof = np.tile(np.arange(E)[::-1], (L, 1))            # expert 0 the most routed
    res = OC.resident_uniform(prof, 0.25)                # experts 0-3 resident
    static, total = OC.hit_rate([t], res)
    assert static == T * L
    for w in (1, 8):
        h, n, c = OC.policy_basalt(t, res, w, math.inf, 25.0)
        assert (h, n, c) == (static, total, 0), (w, h, c)
    h, n, c = OC.policy_saber(t, res, prof, math.inf, 25.0)
    assert (h, n, c) == (static, total, 0)
    h, n, c = OC.policy_saber(t, res, prof, 0.2, 25.0)
    assert c == 2 * L and h == total - 4 * L, (h, c, total)   # misses at steps 0-3; 8 and 9 land for steps 4 and 5
    # reads through: both slots miss (more than half) -> nothing promoted
    ids2 = np.stack([8 + (np.arange(T) % 2), 10 + (np.arange(T) % 2)], -1)[:, None, :].repeat(L, 1)
    write_trace(os.path.join(TMP, "policies2"), "q4exp_agentic", ids2, experts=E, mtp=False)
    t2 = OC.load_traces(os.path.join(TMP, "policies2"))[0]
    h, n, c = OC.policy_saber(t2, res, prof, 0.2, 25.0)
    assert (h, c) == (0, 0), (h, c)
    h, n, c = OC.policy_basalt(t, res, 1, 0.2, 25.0)
    assert h > static and c > 0, (h, c)
    print(f"  never landing = static; saber: second chance on 8 / 9, read-through; basalt gains ({h}/{n}, {c} copies)")


def test_project_stop_flips():
    """The last line is STOP only when (a) pp < 30 and (b) tp < 50 t/s (two cards, 32k, ours, the best fill, the
    measured law - the largest P's - and P per card): h 0.60 projects pp ~20 / tp ~38 (STOP), h 0.75 pp ~29 / tp ~54
    (GO: tp passes) (serial, P 12, BW 590)."""
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py"), "project"]
    for h, want in ((0.60, "STOP"), (0.75, "GO-CANDIDATE")):
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
        Dc, Rc = OC.tp_card_bytes(D, 48 * 10 * 2611200)
        assert abs(j["stop_inputs"]["tp_tps"] - 1e3 / OC.tp_ms(Dc, Rc, 590.0, 12.0, h)) < 1e-9, j["stop_inputs"]
        assert ((tps < 30) and (j["stop_inputs"]["tp_tps"] < 50)) == (want == "STOP"), j["stop_inputs"]
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


SPEC30_DECODE = {   # spec 30 §7.1: h -> layer split, tensor parallel, the same with spec 21-QFN-v2's constants (t/s)
    1.00: (85, 128, 101, 185), 0.99: (73, 113, 84, 156), 0.976: (60, 98, 67, 127), 0.95: (45, 77, 50, 95),
    0.92: (36, 62, 38, 74), 0.90: (31, 55, 33, 64), 0.88: (27, 50, 29, 57), 0.85: (24, 43, 25, 48),
    0.80: (19, 35, 20, 39)}
SPEC30_DRAFTER = {  # spec 30 §8: (tau, u) -> speedup at h 1.00, 0.976, 0.95, 0.90 (layer split, 3 ms a draft)
    (3.5, 4.9): (1.27, 1.03, 0.93, 0.85), (4.5, 4.9): (1.63, 1.32, 1.20, 1.09), (4.5, 3.5): (2.02, 1.73, 1.60, 1.48),
    (6.0, 4.9): (2.17, 1.77, 1.59, 1.45)}


def test_laguna_decode_table():
    sh = OC.SHAPES["laguna"]
    for h, want in SPEC30_DECODE.items():
        got = [OC.two_constant_tps(sh, h, tp, up) for up in (False, True) for tp in (False, True)]
        assert all(abs(g - w) <= 1.0 for g, w in zip(got, want)), (h, got, want)
    # §7.1 at depth: h 0.976, the layer split 58 / 50 / 42 / 33 t/s at 4k / 32k / 64k / 128k
    for ctx, want in ((4096, 58), (32768, 50), (65536, 42), (131072, 33)):
        got = OC.two_constant_tps(sh, 0.976, ctx=ctx)
        assert abs(got - want) <= 1.0, (ctx, got, want)
    print("  spec 30 §7.1's table (9 h x 4) and its depth row within 1 t/s")


def test_laguna_drafter_table():
    sh = OC.SHAPES["laguna"]
    for (tau, u), want in SPEC30_DRAFTER.items():
        got = [OC.drafter_speedup(sh, h, u, tau) for h in (1.0, 0.976, 0.95, 0.90)]
        assert all(abs(g - w) <= 0.02 for g, w in zip(got, want)), (tau, u, got, want)
    print("  spec 30 §8's drafter table (4 x 4) within 0.02")


def test_answer_split():
    """Prompt rows route to experts [0, 128), answer rows to [128, 256): a fill at f 0.5 from both halves' counts
    holds the busier set, so the two parts score apart; answer_from absent -> no parts."""
    d = os.path.join(TMP, "answer")
    L, k, E = 47, 10, 256
    for j, (Tp, Ta) in enumerate(((600, 100), (580, 120), (620, 80), (590, 110))):
        prompt = cycling_rows(Tp, L, k, lambda l: 128)
        answer = cycling_rows(Ta, L, k, lambda l: 128, offset=128)
        write_trace(d, f"t{j}_task-f{j}", np.concatenate([prompt, answer]), experts=E, mtp=False, answer_from=Tp)
    tr = OC.load_traces(d, shape="laguna")
    assert tr[0].answer_from == 600
    r = OC.curve(tr, L, E, [0.5], "hash")
    row = r["rows"][0]
    assert r["parts"] == ["prompt", "answer"]
    assert row["uniform_held_prompt"] == 1.0 and row["uniform_held_answer"] == 0.0, row
    res = OC.resident_uniform(OC.counts_of(tr, L, E), 0.5)
    a, b = OC.hit_rate(tr, res, "answer")
    assert (a, b) == (0, sum(t.T - t.answer_from for t in tr) * L * k)
    d2 = os.path.join(TMP, "answer-none")
    write_trace(d2, "code", cycling_rows(50, 2, 3, lambda l: 12), experts=12, mtp=False)
    assert OC.curve(OC.load_traces(d2), 2, 12, [0.5], "none")["parts"] == []
    try:
        OC.load_traces(d2, shape="laguna")
        raise AssertionError("a 2-layer trace passed as Laguna's shape")
    except SystemExit:
        pass
    print("  prompt rows h 1, answer rows h 0 at f 0.5; no answer_from -> no parts; a wrong shape is refused")


def test_mtp_unions_to_8():
    d = os.path.join(TMP, "mtp8")
    L, k, E, T = 3, 10, 256, 64
    dis = np.stack([np.stack([np.arange(10 * t, 10 * t + 10) % E for _ in range(L)]) for t in range(T)])
    write_trace(d, "code", dis, experts=E, mtp=False)
    t = OC.load_traces(d)[0]
    res = np.zeros((L, E), bool)
    res[:, :128] = True
    for K in range(1, 8):
        u = OC.mtp_unions(t, K, res)
        M = K + 1
        assert u["distinct_per_layer"] == 10.0 * M and u["u"] == float(M), u
        want = 0          # the brute-force misses: per window and layer, the non-resident of the union
        for s0 in range(T - M + 1):
            for l in range(L):
                want += int((~res[l, np.unique(dis[s0:s0 + M, l])]).sum())
        assert u["misses_per_step"] == want / (T - M + 1), (K, u["misses_per_step"], want)
    print("  disjoint rows: u(M) = M for M 2..8; misses equal the per-window brute force")


def test_project_laguna_cli():
    """Laguna-shaped traces (47 layers, 256 experts, top-10) that route only [0, 200): at f 0.80..0.88 every routed
    expert is resident (h 1) -> GO; curve, mtp --K 7 and project chain through their JSON."""
    d = os.path.join(TMP, "laguna-cli")
    L, k, E = 47, 10, 256
    for j in range(4):
        write_trace(d, f"t{j}_task-f{j}", cycling_rows(400, L, k, lambda l: 200), experts=E, mtp=False,
                    answer_from=300)
    cj, mj = os.path.join(TMP, "lc.curve.json"), os.path.join(TMP, "lc.mtp.json")
    py = [sys.executable, os.path.join(_HERE, "offload_curve.py")]
    for cmd in (["curve", "--traces", d, "--shape", "laguna", "--f", "0.70:0.90:0.01", "--json", cj],
                ["mtp", "--traces", d, "--shape", "laguna", "--K", "1,2,3,4,5,6,7", "--f", "0.88", "--json", mj]):
        r = subprocess.run(py + cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    r = subprocess.run(py + ["project", "--shape", "laguna", "--curve", cj, "--mtp", mj], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("GO: answer-position h 1.0000"), last
    u8 = next(x["u"] for x in json.load(open(mj))["K"] if x["M"] == 8)
    assert u8 == 8.0, u8                     # cycling rows of 10 consecutive ids: 8 rows touch 80 distinct
    want = "KEEP" if OC.drafter_speedup(OC.SHAPES["laguna"], 1.0, u8, 5.0) >= 1.25 else "STRUCK"
    assert last.endswith(f"against 1.25x)") and f"DRAFTER: {want}" in last, last
    print(f"  project --shape laguna: {last}")


TESTS = [test_curve_analytic, test_holdout_disjoint, test_swap_never_lands_is_static, test_mtp_unions,
         test_prefill_repeated_row, test_counts_and_profile_roundtrip, test_cli_synthetic, test_tiny_model_traces,
         test_project_spec_table, test_tp_spec_table, test_swap_policies, test_project_stop_flips,
         test_project_per_device_and_swaps, test_laguna_decode_table, test_laguna_drafter_table, test_answer_split,
         test_mtp_unions_to_8, test_project_laguna_cli]


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
