#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for k2_kv8_probe.py (spec 18e Task 1). CPU, tiny random K2 weights (test_k2_ref.py's
tiny config at head_dim 128), seconds; no checkpoint.

    test_k2_kv8_probe.py [test_name ...]

  * the signs are torch's hadamard(128, 0) (the engine's kSignWords[0..3]); the FWHT, rotate_kv /
    rotate_q / unrotate against the fp64 orthonormal R (x sqrt(2), / sqrt(2), / sqrt(2)), q.k kept;
  * quant_dequant is kv8.h's quantiser (bounds, +-127 at the amax, fp16 scale, zero rows);
  * the probe's bf16 variant IS k2_ref (logits bitwise, prompt and greedy steps through the cache);
    rotkv / pt / f32attn run end to end and stay close; the routing diagnostic and the capture;
  * the replay over a tiny capture at a real and a tiled depth.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402


def _mod(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


tk = _mod("test_k2_ref")     # registers k2_ref, gives the tiny config and random weights
ref = sys.modules["k2_ref"]
pr = _mod("k2_kv8_probe")


def hadamard64(n=128):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * pr.signs(n).double()[None, :] / math.sqrt(n)


def test_signs():
    assert torch.equal(pr.signs(), pr.torch_signs(128, 0)), "the engine's signs are not torch's hadamard(128, 0)"
    assert int((pr.signs() < 0).sum()) == 58
    g = torch.Generator().manual_seed(0)
    s256 = torch.randint(0, 2, (256,), generator=g).float() * 2 - 1
    assert torch.equal(s256[:128], pr.signs()), "not the first 128 of 12a's hadamard(256, 0)"


def test_rotations():
    R = hadamard64()
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(64, 128, generator=g) * torch.where(torch.arange(128) % 37 == 5, 30.0, 1.0)).to(torch.bfloat16).float()
    q = torch.randn(64, 128, generator=g)
    e0 = torch.zeros(128)
    e0[0] = 1
    assert torch.equal(pr.rotate_kv(e0), pr.signs() * 0.125)
    assert torch.equal(pr.rotate_q(e0), pr.signs() * 0.0625)
    r2 = math.sqrt(2)
    nx = x.double().norm(dim=-1, keepdim=True)
    assert float(((pr.rotate_kv(x).double() - r2 * x.double() @ R).abs() / nx).max()) < 1e-6
    assert float(((pr.rotate_q(q).double() - q.double() @ R / r2).abs() / q.double().norm(dim=-1, keepdim=True)).max()) < 1e-6
    assert float(((pr.unrotate(pr.rotate_kv(x)).double() - x.double()).abs() / nx).max()) < 1e-6
    d0 = (q.double() * x.double()).sum(-1)
    d1 = (pr.rotate_q(q).double() * pr.rotate_kv(x).double()).sum(-1)
    assert float(((d0 - d1).abs() / (q.double().norm(dim=-1) * x.double().norm(dim=-1))).max()) < 1e-6


def test_quantiser():
    g = torch.Generator().manual_seed(2)
    y = torch.randn(500, 128, generator=g) * torch.logspace(-5, 3, 500)[:, None]
    d = pr.quant_dequant(y)
    s = (y.abs().amax(-1, keepdim=True) / 127).half().float()
    q = d / s
    assert torch.equal(q, torch.round(q)) and float(q.abs().max()) <= 127
    normal = s[:, 0] >= 2 ** -14
    assert bool((q.abs().amax(-1)[normal] == 127).all())
    inside = (y / s).abs() <= 127   # off the clamp: within half a scale
    assert bool((((d - y).abs() <= s / 2 * (1 + 1e-6) + 1e-30) | ~inside).all())
    assert torch.equal(pr.quant_dequant(torch.zeros(3, 128)), torch.zeros(3, 128))


def tiny_model(seed=0):
    d, c, sd = tk.tiny(seed, head_dim=128, rope_head_dim=128)
    return c, ref.K2Ref(c, ref.DictSource(sd), mode="bf16", prefetch=False)


def test_probe_bf16_is_k2_ref():
    c, r = tiny_model()
    ids = [0, 5, 9, 17, 33, 65, 2, 3, 44, 7, 101, 12]
    want = r.forward(ids, 0, r.new_cache())
    lg, toks, routes, n_prompt = pr.run_probe(ref, r, ids, pr.VARIANTS, log=lambda s: None)
    assert torch.equal(lg["bf16"], want.float()), "the probe's bf16 element is not k2_ref's forward"
    for v in ("rotkv", "pt", "f32attn"):
        cos = torch.nn.functional.cosine_similarity(lg[v].double(), want.double(), dim=-1)
        assert float(cos.min()) > 0.99, (v, float(cos.min()))
    # greedy steps through the caches: the bf16 element is k2_ref.generate's rows and tokens
    c2, r2 = tiny_model()
    wl, wt = r2.generate(ids, 3)
    lg2, toks2, routes2, _ = pr.run_probe(ref, r2, ids, ("bf16", "rotkv"), gen=3, log=lambda s: None)
    assert toks2 == wt and torch.equal(lg2["bf16"], wl.float())
    diag = pr.routing_diag(routes2, ("bf16", "rotkv"))
    assert diag["rotkv"]["rows_x_layers"] > 0 and set(diag["rotkv"]) == {"mova", "moe", "rows_x_layers", "first"}
    met = pr.logits_metrics(lg2, ("bf16", "rotkv"))
    assert float(met["rotkv"]["cos"].min()) > 0.99 and "ctl_bf16out" in met


def test_capture_and_replay():
    from safetensors.torch import save_file
    c, r = tiny_model(3)
    ids = list(range(1, 65))
    cap = {}
    pr.run_probe(ref, r, ids, ("bf16", "rotkv"), capture=cap, log=lambda s: None)
    assert sorted(cap) == list(range(c.num_hidden_layers))
    q, k, v = cap[1]
    assert q.shape == (c.num_attention_heads, 64, 128) and k.shape == (c.num_key_value_heads, 64, 128)
    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "cap.safetensors")
        tens = {}
        for i, (a, b, e) in cap.items():
            tens[f"q.L{i}"], tens[f"k.L{i}"], tens[f"v.L{i}"] = a.contiguous(), b.contiguous(), e.contiguous()
        save_file(tens, path, metadata={"n_prompt": "64", "fa_layers": ",".join(map(str, sorted(cap)))})

        class A:
            capture, depths, tiles, nq, window, out = path, "48", "128", 8, 16, os.path.join(t, "r.json")
        pr.cmd_replay(A)
        import json
        rows = json.load(open(A.out))["rows"]
        assert rows["48/rotkv"]["cos_min"] > 0.999 and rows["128 tiled/rotkv"]["cos_min"] > 0.999
        assert rows["48/ctl_bf16out"]["cos_min"] > 0.9999


def test_cli_run():
    """`run` on a tiny on-disk checkpoint (the real layout) end to end: the output file, its
    summary, the capture (the bf16 element is held to k2_ref in-process above)."""
    import subprocess
    d, c, sd = tk.tiny(4, head_dim=128, rope_head_dim=128)
    ids = [0, 3, 8, 21, 40, 77, 5, 6]
    with tempfile.TemporaryDirectory() as t:
        tk.write_checkpoint(t, d, sd)
        for name, xs in (("p.ids", ids), ("c.ids", [9, 10, 11])):
            with open(os.path.join(t, name), "w") as f:
                f.write(" ".join(map(str, xs)))
        out = os.path.join(t, "p.pt")
        r = subprocess.run([sys.executable, os.path.join(_HERE, "k2_kv8_probe.py"), "run", t, "--ids",
                            os.path.join(t, "p.ids"), "--cont-ids", os.path.join(t, "c.ids"), "--gen-cont", "2",
                            "--gen", "1", "--capture", os.path.join(t, "cap.safetensors"), "--out", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "| decision" in r.stdout and "rotkv" in r.stdout, r.stdout
        res = torch.load(out, weights_only=False)
        assert res["meta"]["n"] == 11 and res["meta"]["n_prompt"] == 8
        assert set(res["routing"]) == {"rotkv", "pt", "f32attn"}
        assert os.path.exists(os.path.join(t, "cap.safetensors"))


TESTS = [test_signs, test_rotations, test_quantiser, test_probe_bf16_is_k2_ref, test_capture_and_replay,
         test_cli_run]


def main() -> None:
    torch.manual_seed(0)
    only = sys.argv[1:]
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        t()
    print("ok")


if __name__ == "__main__":
    main()
