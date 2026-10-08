#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for k2_taps.py (the EAGLE3 K2 P0). CPU, a tiny K2-shaped model (test_k2_ref.py's),
seconds.

    test_k2_taps.py [test_name ...]

- the tap points, BITWISE, against the checkpoint's own modeling_k2_horizon.py (vendored) in
  bf16 with the plugin's capture emulated literally (k2_horizon_vllm/model.py:340-345: index 0
  = the embedding before layer 0, index idx + 1 after layer idx): aux.{a} == that index ==
  layer a - 1's output, and != layer a's (the off-by-one the DFlash convention would give);
  `final` == the last layer's output (pre-norm); greedy == argmax of the HF logits;
- the routes == k2_ref's Recorder (which test_k2_ref.py holds to HF), sliced to the head rows;
- the fp32-GEMM arithmetic within a few bf16 ulps of mode bf16, same greedy;
- the CLI over a tiny checkpoint directory: sources (incl. --a4-ref), keep-ctx slicing, the
  dump round trip, resume, golden-ids, --dry-run.
"""
import json
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


TK = _load("test_k2_ref")          # tiny(), hf_model(), write_checkpoint(); loads k2_ref
ref = TK.ref
KT = _load("k2_taps")


def hf_capture(d, sd, ids):
    """The vendored model in bf16: the plugin's aux list (index 0 before layer 0, idx + 1 after
    layer idx) and the logits."""
    model = TK.hf_model(d, sd, torch.bfloat16)
    aux = {}

    def emb_fn(_m, _a, out):              # returns None: a hook's return value replaces the output
        aux.setdefault(0, out[0].float().clone())
    emb_hook = model.model.embed_tokens.register_forward_hook(emb_fn)

    def cap(i):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            aux.setdefault(i + 1, t[0].float().clone())
        return fn
    hooks = [layer.register_forward_hook(cap(i)) for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    emb_hook.remove()
    for h in hooks:
        h.remove()
    return aux, logits


def test_tap_points_bitwise():
    d, c, sd = TK.tiny(seed=21)
    L = c.num_hidden_layers
    n, n_p = 14, 9
    ids = torch.randint(0, c.vocab_size, (n,), generator=torch.Generator().manual_seed(21)).tolist()
    aux_hf, logits = hf_capture(d, sd, ids)
    r = KT.build(ref.DictSource(sd), c, mode="bf16", matmul="bf16", prefetch=False)
    aux_ids = list(range(L))
    t, meta = KT.forward_taps(r, ids, n_p, aux_ids, keep_ctx=4, topk=8)
    af, hf_ = meta["aux_from"], meta["head_from"]
    assert (af, hf_) == (n_p - 4 - 1, n_p - 1)
    for a in aux_ids:
        assert torch.equal(t[f"aux.{a}"].float(), aux_hf[a][af:]), a
    for a in range(1, L):                       # not the output of layer a (DFlash's id convention)
        assert not torch.equal(t[f"aux.{a}"].float(), aux_hf[a + 1][af:]), a
    assert torch.equal(t["final"].float(), aux_hf[L][hf_:])
    assert torch.equal(t["greedy"].long(), logits[hf_:].argmax(-1))
    top = logits[hf_:].topk(8, -1)
    assert torch.equal(t["top_logits"], top.values) and torch.equal(t["lse"], torch.logsumexp(logits[hf_:].double(), -1).float())
    # the routes, against the plain reference's recorder
    r0 = ref.K2Ref(c, ref.DictSource(sd), prefetch=False)
    rec = ref.Recorder(n)
    r0.forward(ids, 0, r0.new_cache(), rec)
    sparse = [i for i in range(L) if c.is_sparse(i)]
    want = torch.stack([rec.t[f"route.moe.ids.L{i}"][0][hf_:] for i in sparse], 1)
    assert torch.equal(t["moe_ids"].long(), want)
    want_v = torch.stack([rec.t[f"route.mova.ids.L{i}"][0][hf_:] for i in sparse if c.is_mova(i)], 1)
    assert torch.equal(t["mova_ids"].long(), want_v)
    assert t["moe_ids"].shape == (n - hf_, len(sparse), c.num_experts_per_tok)
    print(f"  aux.0..{L - 1} == the plugin's capture of the vendored model (index a = layer a-1's output) "
          f"bitwise, != layer a's; final, greedy, top-k, lse bitwise; routes == k2_ref's ({len(sparse)} sparse layers)")


def test_fp32_matmul_close():
    d, c, sd = TK.tiny(seed=22)
    ids = torch.randint(0, c.vocab_size, (16,), generator=torch.Generator().manual_seed(22)).tolist()
    a = KT.forward_taps(KT.build(ref.DictSource(sd), c, matmul="bf16", prefetch=False), ids, 10, [1, 2, 3])[0]
    b = KT.forward_taps(KT.build(ref.DictSource(sd), c, matmul="fp32", prefetch=False), ids, 10, [1, 2, 3])[0]
    worst = 0.0
    for k in ("aux.1", "aux.2", "aux.3", "final"):
        x, y = a[k].float(), b[k].float()
        rel = float(((x - y).abs() / x.abs().clamp_min(1e-2)).max())
        worst = max(worst, rel)
    assert worst < 0.1, worst
    agree = float((a["greedy"] == b["greedy"]).float().mean())
    assert agree >= 0.8, agree
    nbit = sum(int(torch.equal(a[k], b[k])) for k in ("aux.1", "aux.2", "aux.3"))
    print(f"  fp32 GEMMs + one bf16 rounding vs bf16 GEMMs: worst relative diff {worst:.3g} "
          f"({nbit}/3 aux bitwise), greedy agree {agree:.2f}")


def test_cli_end_to_end():
    d, c, sd = TK.tiny(seed=23)
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap")
        os.makedirs(snap)
        TK.write_checkpoint(snap, d, sd)
        g = torch.Generator().manual_seed(23)
        # two plain sources and an A4-style reference directory with one of two scenarios done
        files = {}
        for nm, (np_, nc) in {"p1": (12, 5), "p2": (7, 4), "s1": (9, 3), "s2": (8, 3)}.items():
            files[nm] = (torch.randint(0, c.vocab_size, (np_,), generator=g).tolist(),
                         torch.randint(0, c.vocab_size, (nc,), generator=g).tolist())
        for nm in ("p1", "p2"):
            for part, v in zip(("prompt", "cont"), files[nm]):
                with open(os.path.join(tmp, f"{nm}.{part}.ids"), "w") as f:
                    f.write(" ".join(map(str, v)))
        a4 = os.path.join(tmp, "a4")
        os.makedirs(os.path.join(a4, "set"))
        with open(os.path.join(a4, "set", "manifest.json"), "w") as f:
            json.dump([{"name": "s1", "ids": 9}, {"name": "s2", "ids": 8}], f)
        for nm in ("s1", "s2"):
            with open(os.path.join(a4, "set", f"{nm}.ids"), "w") as f:
                f.write(" ".join(map(str, files[nm][0])))
        with open(os.path.join(a4, "s1.bf16.ids"), "w") as f:
            f.write(" ".join(map(str, files["s1"][1])))
        with open(os.path.join(a4, "s1.bf16.txt"), "w") as f:
            f.write("text")
        out = os.path.join(tmp, "dumps")
        cmd = [sys.executable, os.path.join(_HERE, "k2_taps.py"), "run", snap, "--out-dir", out, "--aux", "1,2,3",
               "--keep-ctx", "3", "--topk", "4", "--a4-ref", a4,
               "--source", f"golden/p1:{tmp}/p1.prompt.ids:{tmp}/p1.cont.ids",
               "--source", f"golden/p2:{tmp}/p2.prompt.ids:{tmp}/p2.cont.ids"]
        dry = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True)
        assert dry.returncode == 0 and "estimate" in dry.stdout and not os.path.exists(os.path.join(out, "golden__p1.k2taps.safetensors")), dry.stdout + dry.stderr
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "1 of 2 scenarios" in p.stdout, p.stdout
        names = sorted(os.listdir(out))
        assert names == ["a4__s1.k2taps.safetensors", "golden__p1.k2taps.safetensors", "golden__p2.k2taps.safetensors",
                         "sources.json"], names
        dmp = KT.load_dump(os.path.join(out, "golden__p1.k2taps.safetensors"))
        assert (dmp["n_prompt"], dmp["n"], dmp["aux_from"], dmp["head_from"]) == (12, 17, 8, 11)
        assert dmp["aux"][1].shape == (17 - 8, c.hidden_size) and dmp["greedy"].shape == (17 - 11,)
        assert dmp["ids"].tolist() == files["p1"][0] + files["p1"][1]
        # the dump equals the in-memory forward (fp32 matmul, the CLI default)
        want = KT.forward_taps(KT.build(ref.DictSource(sd), c, prefetch=False), dmp["ids"].tolist(), 12, [1, 2, 3], 3, 4)[0]
        assert torch.equal(want["aux.2"], dmp["aux"][2]) and torch.equal(want["greedy"].long(), dmp["greedy"])
        assert KT.aux_cat(dmp, [1, 2, 3]).shape == (9, 3 * c.hidden_size)
        again = subprocess.run(cmd, capture_output=True, text=True)
        assert again.returncode == 0 and "3 already dumped, 0 to do" in again.stdout, again.stdout
        # golden-ids: k2_ref run's `tokens`
        from safetensors.torch import save_file
        save_file({"tokens": torch.tensor([5, 6, 7], dtype=torch.int32)}, os.path.join(tmp, "g.safetensors"))
        q = subprocess.run([sys.executable, os.path.join(_HERE, "k2_taps.py"), "golden-ids",
                            os.path.join(tmp, "g.safetensors"), os.path.join(tmp, "g.ids")], capture_output=True, text=True)
        assert q.returncode == 0 and open(os.path.join(tmp, "g.ids")).read().split() == ["5", "6", "7"], q.stderr
    print("  CLI: --source + --a4-ref (1 of 2 done), keep-ctx slicing, dump == in-memory forward, resume, "
          "golden-ids, --dry-run writes nothing")


TESTS = [test_tap_points_bitwise, test_fp32_matmul_close, test_cli_end_to_end]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            print(t.__name__)
            t()
    print("ok")


if __name__ == "__main__":
    main()
