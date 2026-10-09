#!/usr/bin/env python3
"""Spec 21a Task 1: the tiny qwen4_exp model end to end in the pinned transformers (5.19.0).

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/qwen4exp_tiny_check.py <tiny snapshot>

The snapshot is `qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next` (fp32, 4 layers: 3 GDN + 1 QSA, 4
experts top-2, hidden 16, model_type qwen4_exp_text, saved by transformers 5.16.1, layer type
`qwen_sparse_attention`). What it checks and prints (the facts sheet records the lines):
  - the transformers version (refused unless 5.19.0) and `layer_types` as the config remapped them;
  - Qwen4ExpForCausalLM.from_pretrained in bf16 and fp32, attn eager; the experts implementation
    5.19.0 picks for this class by default; that the tiny's per-expert mlp.experts.N.{gate,up,down}_proj
    and its 128 ngram_embedding.shard_N were fused / concatenated on load (conversion_mapping.py);
  - a 24-id prompt: generate(max_new_tokens=8, greedy) with the cache, then the same 32 ids as ONE
    uncached forward: the last 8 rows' logits against the cached steps' (max |diff| printed per dtype;
    whether fp32 is bitwise is recorded, not required: the GDN's cached decode is the recurrent form,
    the uncached forward the chunked form - two different fp32 op orders);
  - a 2100-id random prompt forward completes and the QSA selection differs from causal for at least
    one row past position 2051 (the count is printed).
Exit 0 only if all hold ("tiny check: OK").
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import json  # noqa: E402

import torch  # noqa: E402

PINNED = "5.19.0"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    snap = sys.argv[1]
    import transformers
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    print(f"torch {torch.__version__}, transformers {transformers.__version__} ({os.path.dirname(transformers.__file__)})")
    if transformers.__version__ != PINNED:
        print(f"FAIL: transformers {transformers.__version__}, the reference is {PINNED}")
        return 1
    ok = True
    raw = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
    cfg = AutoConfig.from_pretrained(snap)
    print(f"config: {type(cfg).__name__} model_type {cfg.model_type}; layer_types on disk {raw['layer_types']} "
          f"-> loaded {cfg.layer_types}; ple_layer_ids {cfg.ple_layer_ids}; "
          f"number_of_conv_states {cfg.number_of_conv_states}; output_gate_type {cfg.output_gate_type}")
    if cfg.layer_types[3] != "indexed_attention":
        print("FAIL: the QSA layer was not remapped to indexed_attention")
        ok = False

    from safetensors import safe_open
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as h:
        g0 = h.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        u0 = h.get_tensor("model.layers.0.mlp.experts.0.up_proj.weight")
        d3 = h.get_tensor("model.layers.0.mlp.experts.3.down_proj.weight")
        shards = [h.get_tensor(f"model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight")
                  for i in range(raw["split_ngram_parts"])]

    models = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        mdl = m.Qwen4ExpForCausalLM.from_pretrained(snap, dtype=dt, attn_implementation="eager").eval()
        models[name] = mdl
        ex = mdl.model.layers[0].mlp.experts
        impl = getattr(mdl.config, "_experts_implementation", None)
        fused = (torch.equal(ex.gate_up_proj[0].float(), torch.cat([g0, u0]).to(dt).float())
                 and torch.equal(ex.down_proj[3].float(), d3.to(dt).float()))
        emb = mdl.model.layers[1].ple.ple_embedding.ngram_embedding.weight
        cat = torch.equal(emb.float(), torch.cat(shards).to(dt).float())
        print(f"{name}: experts implementation {impl!r}; per-expert gate/up/down fused into gate_up_proj "
              f"[{ex.gate_up_proj.shape[0]}, {ex.gate_up_proj.shape[1]}, {ex.gate_up_proj.shape[2]}] "
              f"(gate rows first): {fused}; ngram_embedding = cat of {len(shards)} shards "
              f"{list(emb.shape)}: {cat}")
        ok &= fused and cat

    g = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, 248320, (1, 24), generator=g)
    for name, mdl in models.items():
        with torch.no_grad():
            out = mdl.generate(prompt, max_new_tokens=8, do_sample=False, return_dict_in_generate=True,
                               output_logits=True)
            seq = out.sequences
            steps = torch.stack([x[0].float() for x in out.logits])             # [8, V]
            full = mdl(input_ids=seq, use_cache=False).logits[0, 23:31].float()  # rows 23..30
        same_tok = torch.equal(full.argmax(-1), steps.argmax(-1))
        diff = (full - steps).abs().max().item()
        print(f"{name}: generate 8 -> {seq[0, 24:].tolist()}; uncached 32-id forward rows 23..30 vs the "
              f"cached steps: bitwise {torch.equal(full, steps)}, max |diff| {diff:.3e}, argmax equal {same_tok}")
        ok &= same_tok and seq.shape[1] == 32

    mdl = models["bf16"]
    attn = mdl.model.layers[3].self_attn
    cap = {}
    hook = attn.indexer.register_forward_hook(lambda _m, _a, o: cap.__setitem__("sel", o.detach()))
    ids = torch.randint(0, 248320, (1, 2100), generator=g)
    with torch.no_grad():
        logits = mdl(input_ids=ids, use_cache=False).logits
    hook.remove()
    sel = cap["sel"][0, 0]                                            # [T, T] 0 = selected, min = not
    T = sel.shape[0]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    chosen = sel == 0
    differs = (chosen != causal).any(-1)
    rows = differs.nonzero().flatten()
    n_sel = chosen.sum(-1)
    print(f"bf16 2100-id forward: finite {bool(torch.isfinite(logits).all())}; rows whose QSA selection differs "
          f"from causal: {int(differs.sum())} (first {int(rows[0]) if rows.numel() else None}); selected count "
          f"row 2050 {int(n_sel[2050])}, 2051 {int(n_sel[2051])}, 2099 {int(n_sel[2099])}; any row <= 2050 differs: "
          f"{bool(differs[:2051].any())}")
    ok &= bool(torch.isfinite(logits).all()) and int(differs.sum()) > 0 and not bool(differs[:2051].any())
    print("tiny check: OK" if ok else "tiny check: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
