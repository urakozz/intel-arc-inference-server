#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Stage `quant` of tools/quantize/quantize_ripple.sh: AutoRound int4 g64 of Ripple-65B-Preview.

    autoround_run.py --model <bf16 snapshot> --out <dir> [--device xpu|cpu] [--attn int4|bf16] [--iters 200]
                     [--nsamples 128] [--seqlen 2048] [--batch-size 8] [--gas 1]
                     [--dataset nvidia/OpenCodeInstruct:apply_chat_template] [--print-only]

W4A16, group 64, symmetric, export `auto_round:auto_gptq`, tuned sign-SGD, the Flash-Next recipe's sizes and
calibration data (tools/quantize/qwen4exp/autoround_run.py: all 50 cached OpenCodeInstruct shards or refuse).
Quantised: every linear of the 80 blocks - self_attn q/k/v/o/g_proj and mlp gate/up/down_proj - or with
`--attn bf16` only the MLP (self_attn is AutoRound's ignore list, exported as bits 16: the fallback if int4
attention costs too much quality). Kept bf16: the norms, q/k norms, value_res_lambda, the embedding and lm_head
(no quant_lm_head: the engine builds its int8 head at load, as for Qwen3.8; and a quantised layer outside the
blocks would turn low CPU memory and the per-block resume off - see the Flash-Next script).

Ripple-specific:
- Remote code: AutoRound loads `RicursiveForCausalLM` through the snapshot's `auto_map` (trust_remote_code=True)
  and copies modeling_ricursive.py / configuration_ricursive.py into the export.
- Low CPU memory: 130 GB of bf16 does not fit the test machine's 121 GB, and AutoRound only builds its meta skeleton by
  itself for fused-MoE checkpoints - a dense model would be loaded whole. AR_DISK_STREAM_MODEL=1 (set here) forces
  the skeleton for any model: embedding, final norm and head are materialised (~2 GB), each block is read from the
  checkpoint when its turn comes, tuned, written to the export and freed. Checked after load: the run refuses if
  AutoRound did not take that path.
- Value residuals: every layer's V input is lerp(e, x, sigmoid(lambda_l)), e = the scaled embedding rows, handed
  to every layer as the `value_residual` keyword. AutoRound caches the first block's keyword tensors per sample
  (calibration/llm.py `_make_block_forward_func`: a tensor keyword not in SHARED_CACHE_KEYS is split per sample)
  and replays them, sliced to the drawn samples, into every block it tunes (orchestrator: `input_others` from the
  one pre-cache, only the hidden states chained) - exactly Ripple's contract, since e is the same tensor for all
  80 layers. On a resume `input_others` is re-captured from block 0's fresh forward. No shim is needed; a guard is
  installed instead: a forward pre-hook on every decoder layer refuses a call without `value_residual` (the
  remote code would silently use the normed input), and counts the layers seen with it outside torch.compile;
  a run that saw none refuses its export (the hooks never fired, so nothing proved the keyword arrived), and
  check.py refuses a recipe whose count is 0.
- Masks: AutoRound masks the last token of each row, so Ripple's `_update_causal_mask` builds a 4-D additive
  causal mask (a per-sample keyword, cached like `value_residual`); the sliding layers cut it to the window
  (1024 of the 2048 rows), the full layers (NoPE) use it as is; SDPA with that mask, no flash-attention path.

Resumable through AutoRound's per-block checkpoint (AR_RESUME_DIR, set by the script).
"""
import argparse
import filecmp
import glob
import importlib.util
import json
import os
import resource
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("qwen4exp_autoround_run",
                                               os.path.join(HERE, "..", "qwen4exp", "autoround_run.py"))
Q4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q4)          # require_opencode, autoround_info, OPENCODE: the Flash-Next run's own

KEEP_FILES = ("generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "configuration_ricursive.py", "modeling_ricursive.py")


def settings(a) -> dict:
    return {
        "scheme": "W4A16", "group_size": 64, "sym": True,
        "iters": a.iters, "nsamples": a.nsamples, "seqlen": a.seqlen, "batch_size": a.batch_size,
        "gradient_accumulate_steps": a.gas, "dataset": a.dataset, "low_gpu_mem_usage": True,
        "low_cpu_mem_usage": True, "device_map": a.device, "ignore_layers": "self_attn" if a.attn == "bf16" else "",
        "seed": a.seed, "disable_model_free": True, "trust_remote_code": True,
    }


def check_files(snap: str) -> None:
    """Every shard of the index present, and the headers' tensor bytes add up to the index's total_size."""
    import struct
    with open(os.path.join(snap, "model.safetensors.index.json"), encoding="utf-8") as f:
        idx = json.load(f)
    total = 0
    for fn in sorted(set(idx["weight_map"].values())):
        p = os.path.join(snap, fn)
        if not os.path.exists(p):
            raise SystemExit(f"download: {fn} is missing in {snap}")
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        size = os.path.getsize(p)
        for k, t in hdr.items():
            if k != "__metadata__":
                total += t["data_offsets"][1] - t["data_offsets"][0]
                if 8 + n + t["data_offsets"][1] > size:
                    raise SystemExit(f"download: {fn} is truncated ({size} bytes)")
    want = idx.get("metadata", {}).get("total_size")
    if want is not None and total != want:
        raise SystemExit(f"download: the shards hold {total} tensor bytes, the index says {want}")
    print(f"download: {len(set(idx['weight_map'].values()))} shards, {total / 1e9:.2f} GB of tensors, complete",
          flush=True)


def guard_value_residual(model) -> set:
    """A pre-hook on every RicursiveDecoderLayer: no `value_residual`, no forward. Returns the set of layers that
    were called with one outside torch.compile. One hook function for all layers, and nothing recorded while
    compiling: under torch.compile the check is traced into the block's graph (a missing keyword fails its guard
    and raises on the retrace), and a per-layer closure would make every block a new graph and use up dynamo's
    recompile limit after a few blocks."""
    import torch
    seen: set = set()
    layers = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "RicursiveDecoderLayer"]
    if not layers:
        raise SystemExit("value residual guard: no RicursiveDecoderLayer in the model - not Ripple's remote code?")

    def pre(mod, args, kwargs):
        if kwargs.get("value_residual") is None:
            raise RuntimeError("a decoder layer was called without value_residual - AutoRound dropped the keyword; "
                               "the block would be tuned on the wrong V input")
        if not torch.compiler.is_compiling():
            seen.add(mod.ripple_layer_name)
    for n, m in layers:
        m.ripple_layer_name = n
        m.register_forward_pre_hook(pre, with_kwargs=True)
    return seen


def cpu_blocks_in_amp_dtype(ar) -> None:
    """DEVICE=cpu (the selftest): a block read from the checkpoint comes out bf16 while AutoRound computes in fp32
    on a CPU without bf16 (disk_stream_util.materialize_module takes the checkpoint's dtype over a declared fp32),
    so the first matmul fails ("expected m1 and m2 to have the same dtype"). Cast each materialised block to the
    compute dtype, as AutoRound's whole-model CPU load does; bf16_same.py restores the unquantised tensors to the
    source's bf16. On the card the compute dtype is bf16 and nothing changes."""
    import torch
    import auto_round.compressors.orchestrator as orch
    amp = ar.model_context.amp_dtype
    if amp == torch.bfloat16:
        return
    orig = orch.materialize_model_

    def ours(m):
        orig(m)
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point() and t.dtype != amp:
                t.data = t.data.to(amp)
    orch.materialize_model_ = ours


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--attn", choices=("int4", "bf16"), default="int4",
                    help="bf16: self_attn q/k/v/o/g stay bf16, only the MLP is int4")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gas", type=int, default=1, help="gradient_accumulate_steps (batch 4 + gas 2 = batch 8)")
    ap.add_argument("--dataset", default=Q4.OPENCODE + ":apply_chat_template")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    kw = settings(a)
    call = f"AutoRound({a.model!r}, " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + \
        f").quantize_and_save({a.out!r}, format='auto_round:auto_gptq')"
    print(call, flush=True)
    if a.print_only:
        return
    check_files(a.model)
    if a.dataset.split(":")[0] in (Q4.OPENCODE, "opencode-instruct"):
        Q4.require_opencode()
    os.environ["AR_DISK_STREAM_MODEL"] = "1"
    import torch
    from auto_round import AutoRound
    t0 = time.time()
    ar = AutoRound(a.model, **kw)
    if getattr(ar.model_context, "_disk_stream_index", None) is None:
        raise SystemExit("AutoRound loaded the whole model instead of the per-block meta skeleton - refusing "
                         "(130 GB of bf16 does not fit the RAM)")
    print("low CPU memory: meta skeleton, blocks read from the checkpoint one at a time", flush=True)
    if a.device == "cpu":
        cpu_blocks_in_amp_dtype(ar)
    seen = guard_value_residual(ar.model_context.model)
    ar.quantize_and_save(a.out, format="auto_round:auto_gptq")
    n_layers = ar.model_context.model.config.num_hidden_layers
    if not seen:
        raise SystemExit(f"value residual guard: no decoder-layer call reached the pre-hook (0 of {n_layers} layers "
                         "seen outside torch.compile) - AutoRound ran the blocks past the hooks, so nothing shows "
                         "they got their value_residual; refusing the export")
    print(f"value residual: present on every decoder-layer call; {len(seen)} of "
          f"{n_layers} layers seen with it outside torch.compile", flush=True)
    export = os.path.dirname(min(glob.glob(os.path.join(a.out, "**", "config.json"), recursive=True), key=len))
    # AutoRound writes generation_config.json from config.json (no sampling defaults) and re-serialises the
    # tokenizer files: the source's bytes win, for these and for the remote code.
    copied = []
    for fn in KEEP_FILES:
        src, dst = os.path.join(a.model, fn), os.path.join(export, fn)
        if os.path.exists(src) and not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)):
            shutil.copyfile(src, dst)
            copied.append(fn)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    recipe = {"autoround": Q4.autoround_info(), "call": call, "settings": kw, "attn": a.attn,
              "env": {"AR_DISK_STREAM_MODEL": "1"}, "value_residual_seen_layers": len(seen),
              "copied_from_source": copied, "source": os.path.abspath(a.model), "torch": torch.__version__,
              "seconds": round(time.time() - t0), "peak_rss_gib": round(peak, 1),
              "made": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    with open(os.path.join(export, "ripple_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=1)
    print(f"exported to {export} in {recipe['seconds']} s, peak RSS {peak:.1f} GiB (copied from the source: "
          f"{', '.join(copied) or 'nothing'})", flush=True)


if __name__ == "__main__":
    main()
