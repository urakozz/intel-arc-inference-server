#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Plan 30a Task 3: laguna_ref.py `trace` - the layer-streamed teacher-forced pass over several sources at once -
against the vendor's model loaded plainly and run on each source alone, on the tiny model.

    python3 tools/oracle/test_laguna_routes.py [test_name ...]     (pytest collects it too)

BITWISE: every MoE layer's chosen experts and their weights, and the last layer's output, for sources of different
lengths (one past the 512-key window). Then the CLI's files: offload_curve.py's reader takes them, the metadata
carries answer_from and the MoE layer list, a second run skips what is written.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load("test_laguna_ref")
ref = tr.ref
curve = _load("offload_curve")
LENGTHS = (37, 700, 5)


@torch.no_grad()
def plain_routes(m, ids):
    """The plain model on one source, no cache: (ids ascending [T][L][k], weights aligned, last layer's output)."""
    cfg = m.config
    rec = {}
    hooks = [m.model.layers[i].mlp.gate.register_forward_hook(lambda _m, _a, out, i=i: rec.__setitem__(i, out))
             for i in ref.moe_layers(cfg)]
    last = {}
    hooks.append(m.model.layers[cfg.num_hidden_layers - 1].register_forward_hook(
        lambda _m, _a, out: last.__setitem__("h", out[0].clone())))
    try:
        m(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    sel = torch.stack([rec[i][2] for i in ref.moe_layers(cfg)], 1).to(torch.int32)
    w = torch.stack([rec[i][1] for i in ref.moe_layers(cfg)], 1).float()
    o = sel.argsort(-1)
    return sel.gather(-1, o), w.gather(-1, o), last["h"]


def test_trace_equals_plain():
    snap = tr.tiny()
    srcs = [(f"s{n}", tr.prompt(n, seed=n)) for n in LENGTHS]
    model, pf = ref.build_streamed(snap)
    res = ref.trace_routes(model, pf, srcs, log=lambda *a, **k: None)
    m = tr.plain_model(snap)
    for name, ids in srcs:
        ids_p, w_p, last_p = plain_routes(m, ids)
        o = res[name]
        assert o["ids"].shape == (len(ids), 7, 4) and o["ids"].dtype == torch.int32
        assert torch.equal(o["ids"], ids_p), name
        assert torch.equal(o["p"], w_p), name
        assert torch.equal(o["last"], last_p), name
        assert bool((o["ids"][..., 1:] > o["ids"][..., :-1]).all()), "ids not ascending"


def test_cli_trace():
    snap = tr.tiny()
    d = os.path.join(tr.TMP, "srcs")
    os.makedirs(d, exist_ok=True)
    lines = []
    for n, af in ((20, 12), (9, 9)):
        with open(os.path.join(d, f"t{n}.ids"), "w") as f:
            f.write(" ".join(map(str, tr.prompt(n, seed=n))))
        lines.append(f"t{n}\tt{n}.ids\t{af}")
    lst = os.path.join(d, "sources.tsv")
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")
    out = os.path.join(tr.TMP, "traces")
    cmd = [sys.executable, os.path.join(_HERE, "laguna_ref.py"), "trace", snap, "--list", lst, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "layer 7:" in r.stdout and "peak RSS" in r.stdout, r.stdout
    traces = {t.name: t for t in curve.load_traces(out)}
    assert sorted(traces) == ["t20", "t9"]
    t = traces["t20"]
    assert (t.T, t.layers, t.top_k, t.experts) == (20, 7, 4, 16)
    assert t.meta["answer_from"] == "12" and t.meta["moe_layers"] == "1,2,3,4,5,6,7"
    assert traces["t9"].meta["answer_from"] == "9"
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "2 already written" in r.stdout, r.stdout + r.stderr


TESTS = [test_trace_equals_plain, test_cli_trace]


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
        except tr.Skip as s:
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
