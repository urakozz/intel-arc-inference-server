#!/usr/bin/env python3
"""Spec 23a Task 1: the tiny Muse-Glimmer model end to end in the pinned transformers (5.19.0).

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/muse_tiny_check.py <tiny snapshot>

The snapshot is `optimum-intel-internal-testing/tiny-random-muse-glimmer` at cb70af19 (bf16, 4 layers
[s, s, s, f], hidden 64, 4 q / 2 kv heads x 16, FF 256, vocab 202048, tied embeddings, the GPTQ repo's
older template). What it checks and prints (the facts sheet records the lines):
  - the transformers version (refused unless 5.19.0); AutoConfig's `layer_types`, `layer_rope_theta`,
    `tie_word_embeddings`, `sliding_window`, the softcap constants;
  - MuseGlimmerForConditionalGeneration.from_pretrained in bf16 and fp32, attn eager;
  - a 24-id prompt: generate(max_new_tokens=8, greedy) with the cache, then the same 32 ids as ONE
    uncached forward: the last 8 rows' logits against the cached steps' (bitwise / max |diff| printed per
    dtype; the greedy ids must agree);
  - a 40-id forward with output_hidden_states=[1, 3] returns exactly two non-None entries at indices 1 and 3:
    entry 1 is decoder layer 1's OUTPUT (a forward hook on the layer, bitwise); entry 3 - layer 3 is the
    LAST layer of this tiny - is replaced by the final-normed `last_hidden_state` (capture_outputs'
    tie_last_hidden_states, OC:294-305): a tap list whose last id is the model's last layer does not see
    that layer's raw output. The real taps [1, 13, 25, 37, 49] of 52 layers are not affected (the list's
    last entry, layer 51's, is None, so nothing is replaced).
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
    from transformers.models.muse_glimmer import modeling_muse_glimmer as m
    print(f"torch {torch.__version__}, transformers {transformers.__version__} ({os.path.dirname(transformers.__file__)})")
    if transformers.__version__ != PINNED:
        print(f"FAIL: transformers {transformers.__version__}, the reference is {PINNED}")
        return 1
    ok = True
    raw = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
    cfg = AutoConfig.from_pretrained(snap)
    tc = cfg.text_config
    print(f"config: {type(cfg).__name__} model_type {cfg.model_type} / {tc.model_type}; layer_types {tc.layer_types}; "
          f"layer_rope_theta {tc.layer_rope_theta}; rope_parameters {tc.rope_parameters}; sliding_window "
          f"{tc.sliding_window}; tie_word_embeddings top {cfg.tie_word_embeddings} text {tc.tie_word_embeddings}; "
          f"qk_scale_factor {tc.qk_scale_factor}; output_multiplier {tc.output_multiplier}; final_logit_softcapping "
          f"{tc.final_logit_softcapping}; rms_norm_eps {tc.rms_norm_eps}; post_norm_eps {tc.post_norm_eps}; "
          f"image_token_id {cfg.image_token_id}; video_token_id {cfg.video_token_id}")
    if raw["text_config"]["layer_types"] != tc.layer_types:
        print("FAIL: layer_types changed on load")
        ok = False

    from safetensors import safe_open
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as h:
        names = list(h.keys())
    print(f"checkpoint: {len(names)} tensors, lm_head.weight present: {'lm_head.weight' in names}; "
          f"text {sum(n.startswith('model.language_model.') for n in names)}, "
          f"vision {sum(not n.startswith('model.language_model.') and n != 'lm_head.weight' for n in names)}")

    models = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        mdl = m.MuseGlimmerForConditionalGeneration.from_pretrained(snap, dtype=dt, attn_implementation="eager").eval()
        models[name] = mdl
        tied = mdl.lm_head.weight.data_ptr() == mdl.model.language_model.embed_tokens.weight.data_ptr()
        print(f"{name}: loaded; lm_head shares embed_tokens' storage: {tied}; attn {mdl.config._attn_implementation}")

    g = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, 200000, (1, 24), generator=g)
    for name, mdl in models.items():
        with torch.no_grad():
            out = mdl.generate(prompt, max_new_tokens=8, min_new_tokens=8, do_sample=False, top_k=None, top_p=None,
                               temperature=None, return_dict_in_generate=True, output_logits=True)
            seq = out.sequences
            steps = torch.stack([x[0].float() for x in out.logits])             # [8, V]
            full = mdl(input_ids=seq, use_cache=False).logits[0, 23:31].float()  # rows 23..30
        same_tok = torch.equal(full.argmax(-1), steps.argmax(-1))
        diff = (full - steps).abs().max().item()
        print(f"{name}: generate 8 -> {seq[0, 24:].tolist()}; uncached 32-id forward rows 23..30 vs the cached steps: "
              f"bitwise {torch.equal(full, steps)}, max |diff| {diff:.3e}, argmax equal {same_tok}")
        ok &= same_tok and seq.shape[1] == 32

    for name, mdl in models.items():
        ids = torch.randint(0, 200000, (1, 40), generator=g)
        cap = {}
        hooks = [mdl.model.language_model.layers[i].register_forward_hook(
            lambda _m, _a, o, i=i: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().clone()))
            for i in range(4)]
        with torch.no_grad():
            out = mdl(input_ids=ids, use_cache=False, output_hidden_states=[1, 3])
        for hk in hooks:
            hk.remove()
        hs = out.hidden_states
        present = [i for i, x in enumerate(hs) if x is not None]
        last = mdl.model.language_model.norm(cap[3])
        e1 = torch.equal(hs[1], cap[1])
        e3_raw, e3_norm = torch.equal(hs[3], cap[3]), torch.equal(hs[3], last)
        print(f"{name}: output_hidden_states=[1, 3] on 40 ids -> {len(hs)} entries, non-None at {present}; entry 1 == "
              f"layer 1's output: {e1}; entry 3 == layer 3's raw output: {e3_raw}, == norm(layer 3's output) "
              f"(tie_last_hidden_states): {e3_norm}")
        ok &= present == [1, 3] and e1 and e3_norm and not e3_raw
    print("tiny check: OK" if ok else "tiny check: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
