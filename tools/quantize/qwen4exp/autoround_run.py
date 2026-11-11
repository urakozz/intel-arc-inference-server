#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `quant` of tools/quantize/quantize_qwen4exp.sh: AutoRound with spec 21 v2 §2's settings.

    autoround_run.py --model <bf16 snapshot> --out <dir> [--device xpu|cpu] [--iters 200]
                     [--nsamples 128] [--seqlen 2048] [--batch-size 8] [--dataset NeelNanda/pile-10k]
                     [--num-layers N] [--print-only]

Quantised, int4 g64 symmetric, export `auto_round:auto_gptq`: the routed experts (AutoRound unfuses the
original's `mlp.experts.gate_up_proj` / `down_proj` into per-expert linears and exports them per expert),
the shared experts, QSA `q/k/v/o_proj`, GDN `in_proj_qkv` / `in_proj_z` / `out_proj`, and `lm_head`
(`quant_lm_head`). Kept bf16: the routers (`mlp.gate`) and the hyper-connections (AutoRound's own Qwen4 rule),
`shared_expert_gate`, GDN `in_proj_a` / `in_proj_b`, the QSA indexer, the PLE layer (`.ple.`: its table and
projections; the table is converted to int8 by ple_int8.py), every norm, the embedding, the vision tower.
AutoRound's Qwen4 rule also keeps `shared_expert` in bf16 (Intel's form); ours quantises the shared experts
(spec 21 §5 "Ours"), so that one entry is dropped from its predefined list here.

The MTP head (`mtp.*`) is not a transformers module, so it is never tuned: at export AutoRound
(`missing_tensors.py` of the pinned commit) copies the source's `mtp.*` tensors,
splits the fused experts per expert and quantises every linear not in the ignore list by RTN int4 g64. Ours
keeps the head's attention and shared expert bf16 (spec 21 v2 §2, `loader::q4_mtp_desc`), so those two are
in the ignore list; its 512 routed experts come out int4 g64 per expert, the rest bf16, declared in
`extra_config`.

Resumable through AutoRound's own per-block checkpoint (`AR_RESUME_DIR`, set by the script): a rerun with
the same settings skips the blocks already written. `--num-layers N` loads and quantises only the first N layers
(AutoRound's `AR_DEBUG_LAYER_NUM`, the CLI's `--num_hidden_layers`; as an API keyword it is ignored) - the sizing
measurement.
"""
import argparse
import glob
import json
import os
import resource
import time

IGNORE = ["shared_expert_gate", "in_proj_a", "in_proj_b", "indexer", ".ple.", "mtp.layers.0.self_attn",
          "mtp.layers.0.mlp.shared_expert"]


def settings(a) -> dict:
    kw = {
        "scheme": "W4A16", "group_size": 64, "sym": True, "quant_lm_head": True,
        "iters": a.iters, "nsamples": a.nsamples, "seqlen": a.seqlen, "batch_size": a.batch_size,
        "dataset": a.dataset, "low_gpu_mem_usage": True, "low_cpu_mem_usage": True, "device_map": a.device,
        "ignore_layers": ",".join(IGNORE), "seed": a.seed, "disable_model_free": True,
    }
    return kw


def drop_shared_expert_rule() -> None:
    """AutoRound's predefined Qwen4 ignore list keeps `shared_expert` in bf16; ours quantises it."""
    import auto_round.compressors.base as base
    orig = base.get_predefined_ignore_layers

    def ours(model):
        return [n for n in orig(model) if n != "shared_expert"]
    base.get_predefined_ignore_layers = ours


def autoround_info() -> dict:
    import auto_round
    info = {"version": auto_round.__version__, "path": os.path.dirname(auto_round.__file__),
            "commit": os.environ.get("AR_COMMIT", "?"), "patched": os.environ.get("AR_PATCH", "?")}
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dataset", default="NeelNanda/pile-10k")
    ap.add_argument("--num-layers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    kw = settings(a)
    call = f"AutoRound({a.model!r}, " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + \
        f").quantize_and_save({a.out!r}, format='auto_round:auto_gptq')"
    print(call, flush=True)
    if a.print_only:
        return
    if a.num_layers:
        os.environ["AR_DEBUG_LAYER_NUM"] = str(a.num_layers)
    import torch
    from auto_round import AutoRound
    drop_shared_expert_rule()
    t0 = time.time()
    ar = AutoRound(a.model, **kw)
    ar.quantize_and_save(a.out, format="auto_round:auto_gptq")
    export = os.path.dirname(min(glob.glob(os.path.join(a.out, "**", "config.json"), recursive=True), key=len))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    recipe = {"autoround": autoround_info(), "call": call, "settings": kw, "ignore_layers_ours": IGNORE,
              "source": os.path.abspath(a.model), "torch": torch.__version__, "seconds": round(time.time() - t0),
              "peak_rss_gib": round(peak, 1), "made": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    with open(os.path.join(export, "qwen4exp_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=1)
    print(f"exported to {export} in {recipe['seconds']} s, peak RSS {peak:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
