#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Stage `quant` of tools/quantize/quantize_nemotron.sh: AutoRound int4 g64 of NVIDIA Nemotron 3.5 Super (VL).

    autoround_run.py --model <bf16 snapshot> --out <dir> [--device xpu|cpu] [--iters 200]
                     [--nsamples 128] [--seqlen 2048] [--batch-size 8]
                     [--dataset nvidia/OpenCodeInstruct:apply_chat_template] [--print-only]

The checkpoint is NVIDIA's omni wrapper (`nemotron_h_omni`, remote code) around transformers' native
`NemotronHForCausalLM` and a RADIO vision tower. AutoRound (our fork, `auto_round/modeling/nemotron_h_omni.py`)
loads it as such: the trunk `language_model.backbone.*` renamed to `language_model.model.*`, the routed experts
(per expert on disk, fused [E][I][L] in the native class) unfused into per-expert linears again, A_log / D / dt_bias
and the router bias restored in fp32 after the load, a meta skeleton with each block materialised from the
checkpoint (low CPU memory: one block of the 249 GB original in RAM at a time).

Quantised, int4 g64 symmetric, export `auto_round:auto_gptq`, tuned: every block's routed experts (512 a MoE
block, up 1024 -> 2688, down 2688 -> 1024), the shared expert, `fc1_latent_proj` / `fc2_latent_proj`, Mamba
`in_proj` / `out_proj`, attention `q/k/v/o_proj` - the set Nemotron 3.5 Bolt's int4 checkpoints quantised. Kept bf16
by construction (not `nn.Linear`): the router (`mixer.gate`, fp32 logits) and its F32 bias, `conv1d`, the norms
(the gated Mamba norm included), A_log / D / dt_bias, the embedding; `lm_head` is not AutoRound's (no
`quant_lm_head`: a quantised layer outside the blocks turns low CPU memory and the per-block resume off) - the
engine builds its int8 head at load; the vision tower and projector stay bf16 here and the `strip` stage drops
them (the engine never loads them).

The MTP layer (`language_model.mtp.layers.{0,1}.*`: eh_proj, enorm / hnorm, an attention layer, a 512-expert MoE
layer, final_layernorm; it reuses the trunk's embedding and lm_head) is not a transformers module - the omni
wrapper and the native class both drop `mtp.*` on load - so it is never TUNED. At export AutoRound's
`missing_tensors.py` copies every source tensor the export lacks, and quantises each 2-D linear among them by RTN
int4 g64 unless the config excludes it. IGNORE below keeps the MTP layer's eh_proj, attention, shared expert and
latent projections bf16 (declared bits 16 in `extra_config`), so only its 512 routed experts are int4 (RTN): the
experts are 2.82 B of its 2.94 B parameters, and they are what NVIDIA quantised in their own MTP release. The
`check` stage refuses an export without every MTP tensor.

Calibration: nvidia/OpenCodeInstruct through the checkpoint's chat template (each row a user turn and the
assistant's answer), from the HF cache - all of its `data/train-*.parquet` shards or a refusal, never a
fallback dataset. Resumable through AutoRound's per-block checkpoint (`AR_RESUME_DIR`, set by the script).
"""
import argparse
import glob
import json
import os
import resource
import shutil
import time

OPENCODE = "nvidia/OpenCodeInstruct"
OPENCODE_SHARDS = 50
IGNORE = ["mtp.layers.0.", "mtp.layers.1.mixer.shared_experts", "mtp.layers.1.mixer.fc1_latent_proj",
          "mtp.layers.1.mixer.fc2_latent_proj"]


def settings(a) -> dict:
    return {
        "scheme": "W4A16", "group_size": 64, "sym": True,
        "iters": a.iters, "nsamples": a.nsamples, "seqlen": a.seqlen, "batch_size": a.batch_size,
        "dataset": a.dataset, "low_gpu_mem_usage": True, "low_cpu_mem_usage": True, "device_map": a.device,
        "ignore_layers": ",".join(IGNORE), "seed": a.seed, "disable_model_free": True, "trust_remote_code": True,
    }


def require_opencode() -> None:
    """The cached OpenCodeInstruct parquet shards, all of them, or exit (no silent fallback)."""
    from huggingface_hub import snapshot_download
    try:
        snap = snapshot_download(OPENCODE, repo_type="dataset", local_files_only=True)
    except Exception as e:  # noqa: BLE001 - the hub's errors for a missing cache entry vary
        raise SystemExit(f"calibration data: {OPENCODE} is not in the HF cache ({e}) - refusing to run")
    shards = sorted(glob.glob(os.path.join(snap, "data", "train-*.parquet")))
    if len(shards) != OPENCODE_SHARDS:
        raise SystemExit(f"calibration data: {snap} has {len(shards)} of {OPENCODE_SHARDS} train parquet shards")
    print(f"calibration data: {OPENCODE} from {snap} ({len(shards)} parquet shards)", flush=True)


def autoround_info() -> dict:
    import auto_round
    return {"version": auto_round.__version__, "path": os.path.dirname(auto_round.__file__),
            "commit": os.environ.get("AR_COMMIT", "?"), "base": os.environ.get("AR_BASE", "?"),
            "ref": os.environ.get("AR_REF", "?")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dataset", default=OPENCODE + ":apply_chat_template")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    kw = settings(a)
    call = f"AutoRound({a.model!r}, " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + \
        f").quantize_and_save({a.out!r}, format='auto_round:auto_gptq')"
    print(call, flush=True)
    if a.print_only:
        return
    if a.dataset.split(":")[0] in (OPENCODE, "opencode-instruct"):
        require_opencode()
    import torch
    from auto_round import AutoRound
    t0 = time.time()
    ar = AutoRound(a.model, **kw)
    ar.quantize_and_save(a.out, format="auto_round:auto_gptq")
    export = os.path.dirname(min(glob.glob(os.path.join(a.out, "**", "config.json"), recursive=True), key=len))
    # AutoRound writes a generation_config.json made from config.json (eos 11 only, no sampling defaults); the
    # original's (eos [2, 11], T 1.0, top-p 0.95) replaces it.
    shutil.copyfile(os.path.join(a.model, "generation_config.json"), os.path.join(export, "generation_config.json"))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    recipe = {"autoround": autoround_info(), "call": call, "settings": kw, "ignore_layers_ours": IGNORE,
              "source": os.path.abspath(a.model), "torch": torch.__version__, "seconds": round(time.time() - t0),
              "peak_rss_gib": round(peak, 1), "made": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    with open(os.path.join(export, "nemotron_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=1)
    print(f"exported to {export} in {recipe['seconds']} s, peak RSS {peak:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
