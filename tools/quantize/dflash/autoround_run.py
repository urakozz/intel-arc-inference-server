#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `quant` of tools/quantize/quantize_dflash.sh: AutoRound on the real non-causal DFlash 2 drafter.

    autoround_run.py --drafter <snapshot> --target <quantised target snapshot> --taps <dir> --out <dir>
                     [--device xpu|cpu] [--iters 200] [--nsamples 128] [--anchors 32] [--batch-size 8]
                     [--seed 42] [--context 2304] [--print-only]

Settings (docs/open-work.md, "DFlash drafter quant"): tuned W4A16, group 64, symmetric, export
`auto_round:auto_gptq`, 200 iterations, 128 samples, seed 42; no `quant_lm_head` (the drafter has no head: the
embedding and head are the target's). The model is model.py's DFlash2Calib (dflash_ref.py's forward, batched),
its three block lists (`encoder` = fc + hidden_norm, the five layers, `candidate_selector`) tuned in order on
calib.py's samples: the quantised target's taps (the `taps` stage), blocks of 8 at anchors of the training
recordings. Every drafter linear is quantised: fc, q/k/v/o, gate/up/down, both convs' kernel_projection and the
selector's hidden_projection. AutoRound writes the raw export to <out>; finalize.py turns it into the checkpoint.

`<out>/dflash_recipe.json`: the pinned AutoRound (commit from AR_COMMIT, version, patch state), the AutoRound call,
the calibration data (taps dir, recordings, requests and samples, what is held out), sources, torch, time.
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib as C  # noqa: E402
import model as M  # noqa: E402

HELD_OUT = ["opencode-session-2 (the whole recording)", "A4", "golden"]


class NoTokenizer:
    """AutoRound wants a tokenizer for a model object; the samples are tensors, every row valid (no pad id)."""
    pad_token = pad_token_id = eos_token_id = None


def settings(a, rows: int, n_layers: int) -> dict:
    return {
        "scheme": "W4A16", "group_size": 64, "sym": True,
        "iters": a.iters, "nsamples": a.nsamples, "seqlen": rows, "batch_size": a.batch_size,
        "low_gpu_mem_usage": True, "device_map": a.device, "seed": a.seed, "disable_model_free": True,
        "to_quant_block_names": M.blocks(n_layers), "enable_torch_compile": a.torch_compile,
    }


def autoround_info() -> dict:
    import auto_round
    return {"version": auto_round.__version__, "path": os.path.dirname(auto_round.__file__),
            "commit": os.environ.get("AR_COMMIT", "?"), "patched": os.environ.get("AR_PATCH", "?"),
            "repo": os.environ.get("AR_REPO", "?")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--anchors", type=int, default=32, help="blocks (anchor + 7 mask rows) per sample")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=C.context_rows(2048, 256))
    ap.add_argument("--torch-compile", action="store_true", help="AutoRound's torch.compile of the blocks")
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    import torch
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    data = C.CalibData(a.taps, a.drafter, a.target, a.nsamples, a.anchors, a.seed, a.context, dtype)
    kw = settings(a, data.rows, data.cfg.n_layers)
    call = "AutoRound(DFlash2Calib(" + repr(a.drafter) + "), tokenizer=NoTokenizer(), dataset=CalibData(...), " + \
        ", ".join(f"{k}={v!r}" for k, v in kw.items()) + f").quantize_and_save({a.out!r}, format='auto_round:auto_gptq')"
    summary = data.summary()
    print(call, flush=True)
    print("calibration: " + json.dumps(summary), flush=True)
    if a.print_only:
        return
    from auto_round import AutoRound
    t0 = time.time()
    m = M.load(a.drafter, dtype=dtype)
    ar = AutoRound(m, tokenizer=NoTokenizer(), dataset=data, **kw)
    ar.quantize_and_save(a.out, format="auto_round:auto_gptq")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    recipe = {
        "autoround": autoround_info(), "call": call, "settings": {**kw, "format": "auto_round:auto_gptq",
                                                                  "quant_lm_head": False},
        "model": "tools/quantize/dflash/model.py DFlash2Calib (dflash_ref.py's forward, non-causal, batched)",
        "calibration": {**summary, "taps": os.path.abspath(a.taps), "seed": a.seed,
                        "target": os.path.abspath(a.target), "target_layer_ids": data.cfg.target_layer_ids,
                        "requests": sorted({e["name"] for e in data.entries}), "held_out": HELD_OUT},
        "kept_source_precision": ["norms (hidden_norm, norm, input/post_attention_layernorm, q_norm, k_norm)",
                                  "attention_conv / mlp_conv base_kernel", "candidate_selector codebooks",
                                  "mask embedding (if shipped)"],
        "source": os.path.abspath(a.drafter), "torch": torch.__version__, "seconds": round(time.time() - t0),
        "peak_rss_gib": round(peak, 1), "made": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    with open(os.path.join(a.out, "dflash_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=1)
    print(f"exported to {a.out} in {recipe['seconds']} s, peak RSS {peak:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
