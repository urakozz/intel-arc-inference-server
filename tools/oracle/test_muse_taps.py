#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for muse_taps.py (spec 23f Task 2) on 23a's tiny l8 variant - transformers 5.19.0 (spec 21a's site) first
on PYTHONPATH; SKIPs without it or the tiny model.

    test_muse_taps.py [test_name ...]

  - `run`: the dump's taps are the OUTPUT of layers 1 and 5 - bitwise transformers' own un-streamed model with
    `output_hidden_states=[1, 5]` and muse_ref.Recorder's taps; greedy is the raw lm_head row's argmax; top_ids /
    top_logits / lse are the capped row's; load_dump round-trips (ids, n_prompt, tap_from, layers); a 2100-id source
    keeps rows from tap_from = n_prompt - 2048 on; a re-run skips the dumped source.
  - `from-golden`: muse_ref.py run --taps --gen 12 converted - ids = prompt + tokens, the taps the golden's rows,
    greedy the raw rows' argmax; dflash_accept.anchors_of / accept_rows read it.
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _load("muse_taps")
dt = _load("dump_taps")


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


_ENV: dict = {}


def env():
    if "t" in _ENV:
        return _ENV["t"]
    try:
        import transformers
    except ImportError:
        _skip("transformers is not importable (spec 21a's 5.19.0 site first on PYTHONPATH)")
    if transformers.__version__ != "5.19.0":
        _skip(f"transformers {transformers.__version__} (5.19.0 needed)")
    t = _load("test_muse_ref")
    try:
        t.variant("l8")
    except t.Skip as s:
        _skip(str(s))
    _ENV["t"] = t
    return t


TMP = tempfile.mkdtemp(prefix="muse_taps_test_")


def _write_ids(path, ids):
    with open(path, "w") as f:
        f.write(" ".join(map(str, ids)) + "\n")


def test_run_taps_bitwise():
    t = env()
    snap = t.variant("l8")
    ids = t.rand_ids(100, seed=41)
    p = os.path.join(TMP, "a.ids")
    _write_ids(p, ids)
    out = os.path.join(TMP, "dumps")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_taps.py"), "run", snap, "--source",
                        f"code/a:{p}:60", "--out", out, "--layers", "1,5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = dt.load_dump(os.path.join(out, "code__a.taps.safetensors"))
    assert d["name"] == "code/a" and d["n_prompt"] == 60 and d["n"] == 100 and d["tap_from"] == 0
    assert d["layers"] == [1, 5] and d["ids"].tolist() == ids
    hf = t.hf_model(snap, torch.bfloat16)
    raws = []
    hk = hf.lm_head.register_forward_hook(lambda _m, _a, o: raws.append(o.detach()[0]))
    try:
        with torch.no_grad():
            o = hf(input_ids=torch.tensor([ids]), output_hidden_states=[1, 5])
    finally:
        hk.remove()
    for i in (1, 5):
        assert torch.equal(d["taps"][i], o.hidden_states[i][0]), f"tap {i} differs from transformers' hidden_states[{i}]"
    raw = raws[0].float()
    best = raw.max(-1, keepdim=True).values
    want = torch.where(raw == best, torch.arange(raw.shape[1])[None], raw.shape[1]).min(-1).values
    assert torch.equal(d["greedy"], want)
    cap = o.logits[0].float()
    assert torch.equal(d["top_ids"][:, 0], cap.argmax(-1)) and torch.allclose(d["lse"], torch.logsumexp(cap, -1))
    assert torch.equal(d["top_logits"], cap.gather(1, d["top_ids"]))
    # Recorder's taps (the streamed port) are the same tensors
    mr = t.ref
    model, _pf, tc = t.port_model(snap, torch.bfloat16)
    rec = mr.Recorder(model, tc, acts=False, taps=(1, 5))
    try:
        mr.forward_chunks(model, ids, 0, rec)
        rt = rec.tensors()["taps"]
    finally:
        rec.remove()
    assert torch.equal(rt[:, 0], d["taps"][1]) and torch.equal(rt[:, 1], d["taps"][5])
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_taps.py"), "run", snap, "--source",
                        f"code/a:{p}:60", "--out", out, "--layers", "1,5"], capture_output=True, text=True)
    assert r.returncode == 0 and "1 already dumped" in r.stdout, r.stdout[-1500:]
    print("  run: taps = transformers' hidden_states[1] / [5] = Recorder's, BITWISE; greedy / top-64 / lse; resumable")


def test_tap_from():
    t = env()
    snap = t.variant("l8")
    ids = t.rand_ids(2100, seed=42)
    p = os.path.join(TMP, "long.ids")
    _write_ids(p, ids)
    out = os.path.join(TMP, "dumps_long")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_taps.py"), "run", snap, "--source",
                        f"agentic/long:{p}:2090", "--out", out, "--layers", "1,5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = dt.load_dump(os.path.join(out, "agentic__long.taps.safetensors"))
    assert d["tap_from"] == 42 and d["taps"][1].shape[0] == 2058 and d["greedy"].shape[0] == 2058
    print("  tap_from = n_prompt - 2048 (42 of 2100): 2058 rows")


def test_from_golden():
    t = env()
    snap = t.variant("l8")
    ids = t.rand_ids(40, seed=43)
    gdir = os.path.join(TMP, "golden")
    os.makedirs(gdir, exist_ok=True)
    p = os.path.join(gdir, "g.ids")
    _write_ids(p, ids)
    gf = os.path.join(gdir, "g.golden.safetensors")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_ref.py"), "run", snap, "--prompt", p, "--out", gf,
                        "--gen", "12", "--taps", "--act-tail", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = os.path.join(TMP, "dumps_g")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_taps.py"), "from-golden", gf, "--name", "prose/g",
                        "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    from safetensors import safe_open
    with safe_open(gf, "pt") as h:
        g = {k: h.get_tensor(k) for k in h.keys()}
    d = dt.load_dump(os.path.join(out, "prose__g.taps.safetensors"))
    assert d["ids"].tolist() == ids + g["tokens"].tolist() and d["n_prompt"] == 40 and d["layers"] == [1]
    assert torch.equal(d["taps"][1], g["taps"][:, 0])
    raw = g["logits.raw"]
    best = raw.max(-1, keepdim=True).values
    assert torch.equal(d["greedy"], torch.where(raw == best, torch.arange(raw.shape[1])[None], raw.shape[1]).min(-1).values)
    acc = _load("dflash_accept")
    acc._load_deps()
    assert acc.anchors_of(d["n_prompt"], d["n"])[:1] == [40]
    print(f"  from-golden: ids = prompt + tokens ({d['n']}), taps = the golden's rows, greedy = the raw argmax")


TESTS = [test_run_taps_bitwise, test_tap_from, test_from_golden]


def main() -> None:
    only = sys.argv[1:]
    skipped = 0
    for fn in TESTS:
        if only and fn.__name__ not in only:
            continue
        print(fn.__name__, flush=True)
        try:
            fn()
        except Skip as s:
            skipped += 1
            print(f"SKIP {fn.__name__}: {s}")
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
