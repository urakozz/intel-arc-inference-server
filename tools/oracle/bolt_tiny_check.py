#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a Task 1: Bolt's remote code end to end in the pinned transformers (5.19.0) on a tiny puzzle config.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/bolt_tiny_check.py <snapshot>

<snapshot> is our export's pinned revision 8932aa8d in the HF cache (only its small files are read). The tiny config is
built in code from the snapshot's llm_config (TINY_OVERRIDES: the real model's first 12 block types
`M E M E M E M * E M E M`, hidden 256, Mamba 8 heads x 32 (d_inner 256), n_groups 2, state 16, conv 4, attention
16 q / 1 kv x 16, MoE E per layer {16, 20, 24} with I {64, 128, 192}, top-9, latent 64, shared 128, vocab 512),
random weights from R's own init. For fp32 and bf16, and for each Mamba path - R's torch fallback (the unpatched
dispatch on a CPU) and `cuda_kernels_forward` (the dispatch patch, bolt_remote) - a 24-id prompt is generated 8 ids
greedy with our fp32 cache (R:1664-1667 would build it in the model's dtype), then the 32 ids run as ONE uncached
forward: the last 8 rows' argmax must equal the generated ids; the logits' max |diff| is printed (recorded: plan 24a
found cached vs uncached not bitwise in transformers itself). The stub's call counts per path are printed; the kernel
path must call all five functions and rmsnorm_fn, the fallback only rmsnorm_fn. A save_pretrained / from_pretrained
round trip in bf16 records whether transformers keeps the router's F32 `e_score_correction_bias` in fp32.
Exit 0 only if every check holds ("tiny check: OK").
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402

if "bolt_remote" in sys.modules:          # one bolt_remote per process (it holds the imported remote code)
    br = sys.modules["bolt_remote"]
else:
    spec = importlib.util.spec_from_file_location("bolt_remote", os.path.join(_HERE, "bolt_remote.py"))
    br = importlib.util.module_from_spec(spec)
    sys.modules["bolt_remote"] = br
    spec.loader.exec_module(br)

PATTERN12 = "MEMEMEM*EMEM"
E_I = [(16, 64), (20, 128), (24, 192)]          # cycled over the MoE layers


def tiny_config(snapshot: str, pattern: str = PATTERN12, vocab: int = 512) -> dict:
    """The tiny puzzle config: the snapshot's llm_config with the sizes overridden and `pattern`'s blocks."""
    d = dict(br.llm_config(snapshot))
    real = "".join({"mamba": "M", "moe": "E", "attention": "*"}[b["block_type"]] for b in d["block_configs"])
    if not real.startswith(pattern[: min(len(pattern), 12)]) and pattern == PATTERN12:
        raise RuntimeError(f"the real pattern starts {real[:12]}, not {PATTERN12}")
    blocks, e = [], 0
    for ch in pattern:
        if ch == "E":
            E, I = E_I[e % len(E_I)]
            e += 1
            blocks.append({"block_type": "moe", "moe_intermediate_size": I, "n_routed_experts": E,
                           "num_experts_per_tok": 9})
        else:
            blocks.append({"block_type": {"M": "mamba", "*": "attention"}[ch]})
    d.update(block_configs=blocks, hidden_size=256, mamba_num_heads=8, mamba_head_dim=32, n_groups=2,
             ssm_state_size=16, conv_kernel=4, num_attention_heads=16, num_key_value_heads=1, head_dim=16,
             moe_latent_size=64, moe_shared_expert_intermediate_size=128, vocab_size=vocab, intermediate_size=128,
             max_position_embeddings=4096)
    d.pop("layers_block_type", None)
    return d


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    snap = sys.argv[1]
    import transformers
    print(f"torch {torch.__version__}, transformers {transformers.__version__} ({os.path.dirname(transformers.__file__)})")
    if transformers.__version__ != br.PINNED_TRANSFORMERS:
        print(f"FAIL: transformers {transformers.__version__}, the reference is {br.PINNED_TRANSFORMERS}")
        return 1
    ns = br.load_remote(snap)
    print(f"remote code: {snap}; modeling_nemotron_h.py sha256 {br.sha256(os.path.join(snap, 'modeling_nemotron_h.py'))}, "
          f"git blob {br.git_blob_sha1(os.path.join(snap, 'modeling_nemotron_h.py'))} (pinned {br.REMOTE_BLOB})")
    ok = br.git_blob_sha1(os.path.join(snap, "modeling_nemotron_h.py")) == br.REMOTE_BLOB
    R = ns.R
    print(f"R at import on this CPU: is_fast_path_available {R.is_fast_path_available}; rmsnorm_fn is the stub's "
          f"{R.rmsnorm_fn is ns.ops._rmsnorm_fn}; the five kernel globals set to the stub's "
          f"{all(getattr(R, n) is f for n, f in ns.ops.KERNEL_FUNCTIONS.items())}")
    cfgd = tiny_config(snap)
    cfg = br.make_config(cfgd)
    print(f"tiny config: {cfg.num_hidden_layers} blocks {cfg.hybrid_override_pattern}, hidden {cfg.hidden_size}, mamba "
          f"{cfg.mamba_num_heads} x {cfg.mamba_head_dim} groups {cfg.n_groups} state {cfg.ssm_state_size} conv "
          f"{cfg.conv_kernel} chunk {cfg.chunk_size}, time_step_limit {cfg.time_step_limit}, time_step_min "
          f"{cfg.time_step_min}; attention {cfg.num_attention_heads} / {cfg.num_key_value_heads} x {cfg.head_dim}; "
          f"MoE (E, I) {[(b.n_routed_experts, b.moe_intermediate_size) for b in cfg.block_configs if b.block_type == 'moe']}"
          f" top-9, latent {cfg.moe_latent_size}, shared {cfg.moe_shared_expert_intermediate_size}; vocab "
          f"{cfg.vocab_size}; residual_in_fp32 {cfg.residual_in_fp32}")
    g = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, cfgd["vocab_size"], (1, 24), generator=g)
    for dname, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        for path in ("torch", "kernel"):
            model = br.build_lm(cfgd, dt, path=path, seed=0)
            ns.ops.CALLS.clear()
            with torch.no_grad():
                out = model.generate(prompt, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                                     past_key_values=br.fp32_cache(model, 1), use_cache=True, return_dict_in_generate=True,
                                     output_logits=True, top_k=None, top_p=None, temperature=None)
                calls_gen = dict(ns.ops.CALLS)
                seq = out.sequences
                gen = seq[0, 24:].tolist()
                step_logits = torch.stack([lg[0] for lg in out.logits])            # [8][V] fp32
                full = model(input_ids=seq[:, :32], use_cache=False).logits[0, 23:31].float()
            arg = full.argmax(-1).tolist()
            diff = (full - step_logits.float()).abs().max().item()
            # a row whose ids differ is UNDETERMINED (docs/14) when the uncached row's own top-2 margin is within that
            # row's cached-vs-uncached |diff|: two independently rounded paths may take either id
            undet = []
            for r, (a_, g_) in enumerate(zip(arg, gen)):
                if a_ != g_:
                    top = full[r].topk(2).values
                    rd = (full[r] - step_logits[r].float()).abs().max().item()
                    undet.append((r, a_, g_, float(top[0] - top[1]), rd, float(full[r, a_] - full[r, g_]) <= rd))
            same = all(u[-1] for u in undet)
            if undet:
                print(f"  rows whose argmax differs (row, uncached, cached, uncached top-2 margin, row |diff|, "
                      f"undetermined): {undet}; logits row scale (max |logit|) {full.abs().max().item():.3f}")
            bias_dt = {str(m.e_score_correction_bias.dtype) for m in model.modules()
                       if isinstance(m, R.NemotronHTopkRouter)}
            print(f"{dname} {path:6s}: generated {gen}; uncached argmax of the last 8 rows {'EQUAL' if arg == gen else f'{arg} ({len(undet)} undetermined)' if same else f'{arg} DIFFERENT'}; "
                  f"logits cached vs uncached max |diff| {diff:.3e} ({'bitwise' if diff == 0 else 'not bitwise'}); "
                  f"router bias {bias_dt}; stub calls during generate {calls_gen}")
            ok &= same
            five = {"selective_state_update", "mamba_chunk_scan_combined", "causal_conv1d_fn", "causal_conv1d_update",
                    "rmsnorm_fn"}
            if path == "kernel":
                ran = "cuda_kernels_forward (the dispatch patch)"
                ok &= five <= set(calls_gen)
            else:
                ran = "torch_forward (R's fallback, unpatched dispatch)"
                ok &= set(calls_gen) == {"rmsnorm_fn"}
            print(f"  mamba path that ran: {ran}")
            if dname == "bf16" and path == "kernel":
                with tempfile.TemporaryDirectory() as d:
                    model.save_pretrained(d)
                    re = ns.LM.from_pretrained(d, dtype=torch.bfloat16).eval()
                    rb = [m.e_score_correction_bias for m in re.modules() if isinstance(m, R.NemotronHTopkRouter)]
                    ob = [m.e_score_correction_bias for m in model.modules() if isinstance(m, R.NemotronHTopkRouter)]
                    print(f"  save_pretrained / from_pretrained(dtype=bf16): the router bias loads as "
                          f"{sorted({str(b.dtype) for b in rb})}, values equal to the saved fp32 "
                          f"{all(torch.equal(a.float(), b) for a, b in zip(rb, ob))}; A_log / D / dt_bias load as "
                          f"{sorted({str(p.dtype) for n, p in re.named_parameters() if n.endswith(('A_log', '.D', 'dt_bias'))})}")
    print("tiny check: OK" if ok else "tiny check: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
