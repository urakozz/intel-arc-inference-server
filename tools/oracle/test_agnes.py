#!/usr/bin/env python3
"""Checks for agnes.py (spec 14 §3.4). CPU, seconds; no weights.

    test_agnes.py <agnes-snapshot>   (config.json + model.safetensors.index.json suffice)

1. config translation: 72 layers, 54 linear_attention + 18 full_attention, full
   attention exactly at l % 4 == 3, parallel FFN 2048, every other key kept;
2. the name map on every real checkpoint key: no `delta_attn.` / `global_attn.`
   survives, mapped names are unique, and the per-layer key set equals Qwen3.5's
   plus exactly the 12 `mlp.parallel_ffn.*` tensors (qweight/scales/qzeros/g_idx
   x 3); the MTP head's 15 keys map onto Qwen3.5's `mtp.layers.0.self_attn.*`;
3. AgnesMLP (random bf16 weights) against a hand-written two-branch SwiGLU, bitwise;
4. if transformers has Qwen3_5 (the oracle container; NOT the x86_64 Mac, where
   torch stops at 2.2): build_reference on meta at the real config, and the strict
   state-dict key set against the mapped checkpoint names. Otherwise SKIPPED, named.
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

_spec = importlib.util.spec_from_file_location("agnes", os.path.join(_HERE, "agnes.py"))
agnes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agnes)


class _Mlp(nn.Module):   # Qwen3.5's MLP shape, for the arithmetic check
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def main() -> None:
    snap = sys.argv[1]
    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(snap, "model.safetensors.index.json"), encoding="utf-8") as f:
        keys = list(json.load(f)["weight_map"])

    # 1. config
    assert agnes.is_agnes(cfg)
    kw, par = agnes.translate_text_config(cfg)
    assert par == 2048 and kw["num_hidden_layers"] == 72 and kw["intermediate_size"] == 17408
    lt = kw["layer_types"]
    assert len(lt) == 72 and lt.count("linear_attention") == 54 and lt.count("full_attention") == 18
    assert all((t == "full_attention") == (i % 4 == 3) for i, t in enumerate(lt))
    for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "linear_num_value_heads", "linear_num_key_heads", "rope_parameters", "vocab_size"):
        assert kw[k] == cfg["text_config"][k], k
    assert not set(agnes.DROP_KEYS) & set(kw)
    print(f"1. config: 72 layers (54 GDN + 18 FA at l % 4 == 3), parallel FFN {par}")

    # 2. names
    lm = [k for k in keys if k.startswith("model.language_model.")]
    mapped = [agnes.map_name(k) for k in lm]
    assert len(set(mapped)) == len(mapped)
    assert not any("delta_attn." in m or "global_attn." in m for m in mapped)
    per_layer = {}
    for m in mapped:
        r = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", m)
        if r:
            per_layer.setdefault(int(r.group(1)), set()).add(r.group(2))
    assert sorted(per_layer) == list(range(72))
    for i, names in per_layer.items():
        par_names = {n for n in names if n.startswith("mlp.parallel_ffn.")}
        assert len(par_names) == 12, (i, sorted(par_names))
        mixer = "self_attn." if i % 4 == 3 else "linear_attn."
        assert any(n.startswith(mixer) for n in names) and not any(
            n.startswith("linear_attn." if mixer == "self_attn." else "self_attn.") for n in names), i
    mtp = [agnes.map_name(k) for k in keys if k.startswith("mtp.")]
    assert len(mtp) == 15 and sum(m.startswith("mtp.layers.0.self_attn.") for m in mtp) == 6, mtp
    print(f"2. names: {len(lm)} language-model keys mapped, unique; 12 parallel_ffn tensors per "
          f"layer; MTP head 15 keys -> self_attn")

    # 3. AgnesMLP arithmetic
    torch.manual_seed(0)
    main_mlp, branch = _Mlp(64, 96).to(torch.bfloat16), _Mlp(64, 32).to(torch.bfloat16)
    m = agnes.AgnesMLP(main_mlp, branch)
    x = torch.randn(5, 64).to(torch.bfloat16)
    with torch.no_grad():
        def swiglu(mod, v):
            return mod.down_proj(torch.nn.functional.silu(mod.gate_proj(v)) * mod.up_proj(v))
        want = swiglu(main_mlp, x) + swiglu(branch, x)
        got = m(x)
    assert torch.equal(got, want)
    sd = set(m.state_dict())
    assert {"gate_proj.weight", "up_proj.weight", "down_proj.weight",
            "parallel_ffn.gate_proj.weight", "parallel_ffn.up_proj.weight",
            "parallel_ffn.down_proj.weight"} == sd, sd
    print("3. AgnesMLP == two-branch SwiGLU, bitwise; state-dict names as the checkpoint's")

    # 4. the real build, where transformers can
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: F401
    except Exception as e:  # noqa: BLE001 - any import failure means "not here"
        print(f"4. SKIPPED: transformers' Qwen3_5 is not importable here ({type(e).__name__}); "
              f"run in the oracle container (validation checklist, spec 14)")
        print("test_agnes OK (3 of 4 checks; 4 pending)")
        return
    model, _ = agnes.build_reference(cfg, device="meta")
    want_keys = set(model.state_dict())
    have = set()
    for k in lm:
        if k.endswith((".qzeros", ".g_idx", ".scales")):
            continue
        base = k[: -len(".qweight")] + ".weight" if k.endswith(".qweight") else k
        have.add("model." + agnes.map_name(base)[len("model.language_model."):])
    have.add("lm_head.weight")
    missing, unexpected = sorted(want_keys - have), sorted(have - want_keys)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    print(f"4. build_reference: {len(want_keys)} parameters, strict key match with the checkpoint")
    print("test_agnes OK")


if __name__ == "__main__":
    main()
