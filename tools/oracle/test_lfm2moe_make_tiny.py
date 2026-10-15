#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""lfm2moe_make_tiny.py's variants (spec 24a Task 2) against transformers 5.19.0.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_lfm2moe_make_tiny.py [test_name ...]   (pytest collects it too)

Each variant loads in 5.19.0 (from_pretrained, eager; the int4 export through its dequantised twin - transformers
cannot read an auto-round export without the auto-round package); pattern8's layer types and dense / MoE split
are the real model's layers 0..7; `ties` really ties (the planted pair's `sel` bit-equal and at the 4th / 5th
place on > 50 % of 64 random tokens); the int4 export's dequant is (q - 8) x scale (stream.dequant_t), its
quantised set is every nn.Linear, its expert names follow --expert-names, and the twin holds those values.
Variants are written under $LFM_TINY_DIR (default <repo>/oracle-out-lfm2moe-tiny, git-ignored).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mk = _load("lfm2moe_make_tiny")
stream = _load("stream")
REPO = os.path.dirname(os.path.dirname(_HERE))
TINY = os.environ.get("LFM_TINY_DIR", os.path.join(REPO, "oracle-out-lfm2moe-tiny"))
# The real model's config.json layer_types[:8] and num_dense_layers (spec 24 §1; read from the snapshot when present)
REAL8 = ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv"]


def variant(name: str, names: str = "w") -> str:
    """The variant's directory (written once per seed-0 content; int4 also writes its twin)."""
    sub = name if name != "int4-g64" else f"int4-g64-{names}"
    d = os.path.join(TINY, sub)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        mk.make(d, name, 0, names=names, twin=os.path.join(TINY, "int4-g64-twin") if name == "int4-g64" else None)
    return d


def hf(d: str, dtype=torch.bfloat16):
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    return m.Lfm2MoeForCausalLM.from_pretrained(d, dtype=dtype, attn_implementation="eager").eval()


def test_variants_load():
    for name, dt in (("pattern8", torch.bfloat16), ("pattern8-fp32", torch.float32), ("ties", torch.bfloat16)):
        model = hf(variant(name), dt)
        assert model.config.layer_types == REAL8, (name, model.config.layer_types)
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), name
        assert model.model.embed_tokens.weight.dtype == dt
        b = model.model.layers[2].feed_forward.expert_bias
        assert b.dtype == torch.float32 and bool((b != 0).any()), name
    variant("int4-g64")
    model = hf(os.path.join(TINY, "int4-g64-twin"))
    assert model.config.num_hidden_layers == 8


def test_pattern8_is_the_real_pattern():
    cfg = json.load(open(os.path.join(variant("pattern8"), "config.json"), encoding="utf-8"))
    real = REAL8
    snap = mk.default_tokenizer()
    if snap and os.path.exists(os.path.join(snap, "config.json")):
        rc = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
        real = rc["layer_types"][:8]
        assert rc["num_dense_layers"] == cfg["num_dense_layers"] == 2
        assert rc["hidden_size"] // rc["num_attention_heads"] == cfg["hidden_size"] // cfg["num_attention_heads"] == 64
    assert cfg["layer_types"] == real
    model = hf(variant("pattern8"))
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    kinds = [type(layer.feed_forward).__name__ for layer in model.model.layers]
    assert kinds == ["Lfm2MoeMLP"] * 2 + ["Lfm2MoeSparseMoeBlock"] * 6, kinds
    assert [layer.is_attention_layer for layer in model.model.layers] == [t == "full_attention" for t in real]
    attn = model.model.layers[2].self_attn
    assert attn.head_dim == 64 and attn.num_key_value_groups == 4 and isinstance(attn, m.Lfm2MoeAttention)


def test_ties_really_tie():
    model = hf(variant("ties"))
    caught = {}
    hooks = [model.model.layers[i].feed_forward.gate.register_forward_hook(
        lambda mod, a, out, i=i: caught.__setitem__(i, (a[0].detach(), out))) for i in mk.TIES]
    ids = torch.randint(0, 124000, (1, 64), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    for layer, (a, c) in mk.TIES.items():
        x, (logits, _w, sel_ids) = caught[layer]
        s = logits.sigmoid()
        sel = s + model.model.layers[layer].feed_forward.expert_bias
        top5 = sel.topk(5, dim=-1)
        at_cut = (top5.values[:, 3] == top5.values[:, 4])
        pair = torch.stack([top5.indices[:, 3], top5.indices[:, 4]], -1).sort(-1).values
        is_pair = (pair == torch.tensor(sorted((a, c)))).all(-1)
        frac = float((at_cut & is_pair).float().mean())
        assert frac > 0.5, (layer, frac)
        print(f"  ties layer {layer}: experts {a} / {c} tied at the 4th / 5th sel on {frac:.0%} of 64 tokens")


def test_int4_export():
    from safetensors import safe_open
    for names in ("w", "gate_up_down"):
        d = variant("int4-g64", names)
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        qc = cfg["quantization_config"]
        assert (qc["packing_format"], qc["bits"], qc["group_size"], qc["sym"]) == ("auto_round:auto_gptq", 4, 64, True)
        with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            q = {k[: -len(".qweight")] for k in keys if k.endswith(".qweight")}
            spelled = {k.rsplit(".", 1)[-1] for k in q if ".experts." in k}
            assert spelled == set(mk.EXPERT_NAMES[names]), (names, spelled)
            plain = {k for k in keys if k.endswith(".weight")}
            assert not any(mk.is_linear(k) for k in plain), sorted(k for k in plain if mk.is_linear(k))[:3]
            assert all(any(k.endswith(s) for s in ("gate.weight", "conv.conv.weight", "norm.weight",
                                                    "layernorm.weight", "embed_tokens.weight")) for k in plain), plain
            assert f.get_tensor("model.layers.2.feed_forward.expert_bias").dtype == torch.float32
            base = f"model.layers.3.feed_forward.experts.5.{mk.EXPERT_NAMES[names][0]}"
            qw, sc, qz = f.get_tensor(base + ".qweight"), f.get_tensor(base + ".scales"), f.get_tensor(base + ".qzeros")
            assert bool((qz == 0x77777777).all()) and sc.dtype == torch.float16
            got = stream.dequant_t(qw, sc, 64)
            nib = torch.stack([(qw >> (4 * i)) & 0xF for i in range(8)], 1).reshape(-1, qw.shape[1])  # [K][N]
            want = ((nib.float() - 8.0) * sc.float().repeat_interleave(64, 0)).to(torch.bfloat16).t()
            assert torch.equal(got, want)
        with safe_open(os.path.join(TINY, "int4-g64-twin", "model.safetensors"), framework="pt") as t:
            assert torch.equal(t.get_tensor("model.layers.3.feed_forward.experts.5.w1.weight"), got)
    # rtn.h's rule at the edges: an all-zero group is q = 8 / scale 0; +amax rounds to 8 + 8 -> 15 and -amax to
    # 8 - 8 = 0 (s16 = f16(2 / 15) < 2 / 15, so |w| / s16 = 7.5018 rounds to 8; clamped)
    w = torch.zeros(8, 64)
    w[1, 0] = 1.0
    w[2, 3] = -1.0
    qw, _, sc = mk.rtn_int4(w)
    q = torch.stack([(qw >> (4 * i)) & 0xF for i in range(8)], 1).reshape(64, 8)
    assert sc[0, 0] == 0 and bool((q[:, 0] == 8).all())
    assert int(q[0, 1]) == 15 and int(q[3, 2]) == 0 and float(sc[0, 1]) == float(torch.tensor(2 / 15).half())


if __name__ == "__main__":
    want = sys.argv[1:]
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and (not want or n in want)]
    for n, f in tests:
        f()
        print(f"ok  {n}", flush=True)
    print(f"{len(tests)} tests passed")
