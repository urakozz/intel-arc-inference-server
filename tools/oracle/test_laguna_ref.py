#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Plan 30a Task 2: laguna_ref.py's `run` against the vendor's model loaded plainly (from_pretrained), and
laguna_facts.py's facts, on the tiny model laguna_make_tiny.py writes.

    python3 tools/oracle/test_laguna_ref.py [test_name ...]       (pytest collects it too)

The tiny model is made from the original's small files (config, the vendor's code, the tokenizer): $LAGUNA_VENDOR,
else the box's cold-disk snapshot, else the HF cache; every test SKIPs when none is there. The streamed reference
must be BITWISE the plainly loaded model: logits and every layer's output, prompt and cached decode.
"""
import glob
import os
import subprocess
import sys
import tempfile

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


ref = _load("laguna_ref")
mk = _load("laguna_make_tiny")
facts = _load("laguna_facts")
SNAPSHOTS = "models--poolside--Laguna-S-2.1/snapshots/*"


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def vendor_dir() -> str:
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cands = [os.environ.get("LAGUNA_VENDOR", "")]
    for root in (os.path.expanduser("~/b70/hdd/hf-hub"), os.path.join(hf, "hub"), "/hf/hub"):
        cands += sorted(glob.glob(os.path.join(root, SNAPSHOTS)))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "modeling_laguna.py")):
            return c
    _skip("the original's small files are absent (LAGUNA_VENDOR=<dir with config.json, modeling_laguna.py, ...>)")


TMP = tempfile.mkdtemp(prefix="laguna_test_")
_CACHE: dict = {}


def tiny() -> str:
    if "tiny" not in _CACHE:
        out = os.path.join(TMP, "tiny")
        mk.make(out, vendor_dir())
        _CACHE["tiny"] = out
    return _CACHE["tiny"]


def plain_model(snap: str):
    """The vendor's model loaded plainly: from_pretrained, f32, sdpa (the reference's attention)."""
    key = ("plain", snap)
    if key not in _CACHE:
        ref.require_version()
        from transformers import AutoModelForCausalLM
        m, info = AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True, dtype=torch.float32,
                                                       attn_implementation="sdpa", output_loading_info=True)
        bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
        assert not bad, f"from_pretrained: {bad}"
        _CACHE[key] = m.eval()
    return _CACHE[key]


def prompt(n: int, seed: int = 1) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return [2] + torch.randint(100, 100000, (n - 1,), generator=g).tolist()


def test_tiny_is_vendor_code():
    m = plain_model(tiny())
    cfg = m.config
    assert type(m).__module__.startswith("transformers_modules."), type(m).__module__
    assert cfg.layer_types[:4] == ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
    assert ref.moe_layers(cfg) == list(range(1, 8))
    assert [m.model.layers[i].self_attn.num_heads for i in range(8)] == [4, 6, 6, 6, 4, 6, 6, 6]
    bias = m.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.abs().sum() > 0, "the selection bias was not loaded onto the router"


@torch.no_grad()
def test_generates_deterministically():
    m = plain_model(tiny())
    ids = torch.tensor([prompt(24)])
    a = m.generate(ids, max_new_tokens=8, do_sample=False)
    b = m.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(a, b) and a.shape[1] == 32


@torch.no_grad()
def test_run_equals_plain():
    """Streamed `run` = the plain model, bitwise: prompt logits, every layer's output, the 8 greedy tokens and the
    cached decode steps' logits."""
    from transformers import DynamicCache
    snap = tiny()
    ids = prompt(40)
    t, meta = ref.run(snap, ids, 8, logits_tail=-1, act_tail=-1)
    m = plain_model(snap)
    acts = {i: [] for i in range(8)}
    hooks = [layer.register_forward_hook(lambda _m, _a, out, i=i: acts[i].append(out.float().clone()))
             for i, layer in enumerate(m.model.layers)]
    try:
        out = m(input_ids=torch.tensor([ids]), past_key_values=DynamicCache(config=m.config), use_cache=True)
        rows = [out.logits[0].float()]
        toks = []
        for _ in range(8):
            toks.append(int(rows[-1][-1].argmax()))
            out = m(input_ids=torch.tensor([[toks[-1]]]), past_key_values=out.past_key_values, use_cache=True)
            rows.append(out.logits[0].float())
    finally:
        for h in hooks:
            h.remove()
    assert t["tokens"].tolist() == toks
    assert torch.equal(t["logits"], torch.cat(rows))
    for i in range(8):
        assert torch.equal(t[f"H.L{i}"], torch.cat(acts[i], 1)[0]), f"layer {i}"
    gen = m.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):].tolist()
    assert gen == toks, (gen, toks)
    assert t["route.ids.L1"].shape == (48, 4) and "route.ids.L0" not in t
    assert meta["moe_layers"] == "1,2,3,4,5,6,7"


def test_cli_run():
    snap = tiny()
    p = os.path.join(TMP, "p.ids")
    with open(p, "w") as f:
        f.write(" ".join(map(str, prompt(12))))
    out = os.path.join(TMP, "p.golden.safetensors")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "laguna_ref.py"), "run", snap, "--prompt", p, "--out", out,
                        "--gen", "4", "--act-tail", "3", "--logits-tail", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from safetensors import safe_open
    with safe_open(out, "pt") as f:
        assert f.get_tensor("tokens").shape == (4,)
        assert f.get_tensor("logits").shape[0] == 2 + 4
        assert f.get_tensor("H.L7").shape == (3 + 4, 128)
        assert f.metadata()["act_row0"] == "9"


def test_facts():
    """laguna_facts.py's assertions hold on the tiny model (they raise otherwise)."""
    snap = tiny()
    lines = facts.check(snap, plain_model(snap))
    assert any(ln.startswith("sliding window:") for ln in lines)


TESTS = [test_tiny_is_vendor_code, test_generates_deterministically, test_run_equals_plain, test_cli_run, test_facts]


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
