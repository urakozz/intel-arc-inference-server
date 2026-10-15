#!/usr/bin/env python3
"""Spec 24a Task 1: the tiny lfm2_moe model end to end in the pinned transformers (5.19.0).

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/lfm2moe_tiny_check.py <tiny snapshot>

The snapshot is `yujiepan/lfm2-moe-tiny-random` (bf16, 3 layers conv, conv, full_attention - the real
model's first three -, 2 dense layers, 32 experts top-4, hidden 64, 2 q / 1 kv heads of 32, expert_bias
all zero, vocab 65536; `tie_word_embeddings` absent from config.json, so the config class's default True).
What it checks and prints (the facts sheet records the lines):
  - the transformers version (refused unless 5.19.0); layer_types, tie_word_embeddings, num_dense_layers,
    use_expert_bias as the config class read them;
  - Lfm2MoeForCausalLM.from_pretrained in bf16 and fp32, attn eager; `config._experts_implementation`
    (grouped_mm when torch's grouped_mm dispatches, modeling_utils.py:1887-1909, else eager); whether
    causal_conv1d_update / causal_conv1d_fn resolved to the torch functions (hub_kernels.py:984-1019:
    the `causal_conv1d` package is absent on CPU, so `is_new_implementation` is false); that expert_bias
    stays an fp32 buffer under dtype=bf16;
  - a 24-id prompt: generate(max_new_tokens=8, greedy) with the cache, then the same 32 ids as ONE
    uncached forward: the last 8 rows' argmax equal (required); whether the logits are bitwise is printed
    per dtype with the max |diff| and, where they differ, the first module whose output differs (recorded,
    not required: a cached step is an M = 1 product, the uncached forward an M = 32 one - transformers'
    own two paths; F1 compares the streamed reference with transformers on the SAME path);
  - the head is the embedding (lm_head.weight shares embed_tokens.weight's storage);
  - an sdpa twin of the bf16 model on the same 32 ids: max |logit(eager) - logit(sdpa)| (the facts
    sheet's sdpa row; the reference never uses sdpa - spec 24 §2.6, decision 13).
Exit 0 only if all hold ("tiny check: OK").
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import inspect  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

PINNED = "5.19.0"


def resolved_torch(fn) -> bool:
    """True when a use_kernel_func_from_hub_with_fallback wrapper runs the torch function itself."""
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        cv = inspect.getclosurevars(fn).nonlocals
        if "is_new_implementation" in cv:
            return not cv["is_new_implementation"]
        fn = getattr(fn, "__wrapped__", None)
    return True


def first_difference(mdl, seq) -> str:
    """The first module (in execution order) whose output for the LAST row differs between the uncached forward of
    `seq` and a cached step that feeds the last id after the rest."""
    outs: dict = {}
    names = {mod: n for n, mod in mdl.named_modules() if n and not list(mod.children())}

    def hook(tag):
        def fn(mod, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.dim() >= 2:
                outs.setdefault(tag, []).append((names[mod], t.detach().reshape(-1, t.shape[-1])[-1].clone()))
        return fn
    with torch.no_grad():
        hs = [mod.register_forward_hook(hook("full")) for mod in names]
        mdl(input_ids=seq, use_cache=False)
        for h in hs:
            h.remove()
        out = mdl(input_ids=seq[:, :-1], use_cache=True)
        hs = [mod.register_forward_hook(hook("step")) for mod in names]
        mdl(input_ids=seq[:, -1:], past_key_values=out.past_key_values, use_cache=True)
        for h in hs:
            h.remove()
    for (n, a), (_, b) in zip(outs["full"], outs["step"]):
        if a.shape == b.shape and not torch.equal(a, b):
            return f"{n} (max |diff| {(a.float() - b.float()).abs().max().item():.3e})"
    return "none"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    snap = sys.argv[1]
    import importlib.util
    import transformers
    from transformers import AutoConfig
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    print(f"torch {torch.__version__}, transformers {transformers.__version__} ({os.path.dirname(transformers.__file__)})")
    if transformers.__version__ != PINNED:
        print(f"FAIL: transformers {transformers.__version__}, the reference is {PINNED}")
        return 1
    ok = True
    raw = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
    cfg = AutoConfig.from_pretrained(snap)
    print(f"config: {type(cfg).__name__} model_type {cfg.model_type}; layer_types {cfg.layer_types}; "
          f"tie_word_embeddings {cfg.tie_word_embeddings} (in config.json: {'tie_word_embeddings' in raw}); "
          f"num_dense_layers {cfg.num_dense_layers}; use_expert_bias {cfg.use_expert_bias}; "
          f"num_experts {cfg.num_experts} top-{cfg.num_experts_per_tok}; hidden {cfg.hidden_size}; heads "
          f"{cfg.num_attention_heads} / kv {cfg.num_key_value_heads}; norm_eps {cfg.norm_eps}; conv_L_cache "
          f"{cfg.conv_L_cache}; conv_bias {cfg.conv_bias}")
    pkg = importlib.util.find_spec("causal_conv1d") is not None
    upd, fn = resolved_torch(m.causal_conv1d_update), resolved_torch(m.causal_conv1d_fn)
    print(f"conv path: causal_conv1d package installed {pkg}; causal_conv1d_update runs the torch function {upd}; "
          f"causal_conv1d_fn runs the torch function {fn}")
    ok &= upd and fn

    models = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        mdl = m.Lfm2MoeForCausalLM.from_pretrained(snap, dtype=dt, attn_implementation="eager").eval()
        models[name] = mdl
        impl = getattr(mdl.config, "_experts_implementation", None)
        bias = mdl.model.layers[2].feed_forward.expert_bias
        tied = mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr()
        print(f"{name}: experts implementation {impl!r}; attn {mdl.config._attn_implementation!r}; expert_bias "
              f"{bias.dtype} {list(bias.shape)} all zero {bool((bias == 0).all())}; lm_head is embed_tokens "
              f"(same storage) {tied}; gate_up_proj {list(mdl.model.layers[2].feed_forward.experts.gate_up_proj.shape)}")
        ok &= tied and bias.dtype == torch.float32

    g = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, raw["vocab_size"], (1, 24), generator=g)
    full_rows = {}
    for name, mdl in models.items():
        with torch.no_grad():
            out = mdl.generate(prompt, max_new_tokens=8, do_sample=False, return_dict_in_generate=True,
                               output_logits=True)
            seq = out.sequences
            steps = torch.stack([x[0].float() for x in out.logits])             # [8, V]
            full = mdl(input_ids=seq, use_cache=False).logits[0, 23:31].float()  # rows 23..30
        full_rows[name] = (seq, mdl(input_ids=seq, use_cache=False).logits[0].float())
        same_tok = torch.equal(full.argmax(-1), steps.argmax(-1))
        bitwise = torch.equal(full, steps)
        diff = (full - steps).abs().max().item()
        where = "" if bitwise else f"; first differing module output on row 31: {first_difference(mdl, seq)}"
        print(f"{name}: generate 8 -> {seq[0, 24:].tolist()}; uncached 32-id forward rows 23..30 vs the cached "
              f"steps: bitwise {bitwise}, max |diff| {diff:.3e}, argmax equal {same_tok}{where}")
        ok &= same_tok and seq.shape[1] == 32

    seq, eager = full_rows["bf16"]
    sd = m.Lfm2MoeForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    with torch.no_grad():
        sdpa = sd(input_ids=seq, use_cache=False).logits[0].float()
    print(f"sdpa twin (bf16, 32 ids): attn {sd.config._attn_implementation!r}; max |logit(eager) - logit(sdpa)| "
          f"{(eager - sdpa).abs().max().item():.3e}; bitwise {torch.equal(eager, sdpa)}; argmax equal "
          f"{torch.equal(eager.argmax(-1), sdpa.argmax(-1))} (recorded only: the reference is eager)")
    print("tiny check: OK" if ok else "tiny check: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
