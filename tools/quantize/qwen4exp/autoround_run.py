#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `quant` of tools/quantize/quantize_qwen4exp.sh: AutoRound with spec 21 v2 §2's settings.

    autoround_run.py --model <bf16 snapshot> --out <dir> [--device xpu|cpu] [--iters 200]
                     [--nsamples 128] [--seqlen 2048] [--batch-size 8]
                     [--dataset nvidia/OpenCodeInstruct:apply_chat_template]
                     --ple-dir <dir on NVMe> [--print-only]

Quantised, int4 g64 symmetric, export `auto_round:auto_gptq`: the routed experts (AutoRound unfuses the
original's `mlp.experts.gate_up_proj` / `down_proj` into per-expert linears and exports them per expert),
the shared experts, QSA `q/k/v/o_proj`, GDN `in_proj_qkv` / `in_proj_z` / `out_proj`. Kept bf16: the routers
(`mlp.gate`) and the hyper-connections (AutoRound's own Qwen4 rule), `shared_expert_gate`, GDN `in_proj_a` /
`in_proj_b`, the QSA indexer, the PLE layer (`.ple.`: its table and projections; the table is converted to int8
by ple_int8.py), every norm, the embedding, the vision tower. AutoRound's Qwen4 rule also keeps `shared_expert`
in bf16 (Intel's form); ours quantises the shared experts (spec 21 §5 "Ours"), so that one entry is dropped from
its predefined list here.

`lm_head` is NOT AutoRound's (no `quant_lm_head`): a quantised layer outside the blocks makes it switch low CPU
memory off ("`low_cpu_mem_usage` is not fully supported when there are quantized layers outside blocks"), which
keeps every finished block resident - the whole 360 GB model by the end - and disables the per-block resume. It
would only RTN the head anyway; head_rtn.py packs it afterwards by the engine's rule.

The MTP head (`mtp.*`) is not a transformers module, so it is never tuned: at export AutoRound
(`missing_tensors.py` of the pinned commit) copies the source's `mtp.*` tensors, splits the fused experts per
expert and quantises every linear not in the ignore list by RTN int4 g64. Ours keeps the head's attention and
shared expert bf16 (spec 21 v2 §2, `loader::q4_mtp_desc`), so those two are in the ignore list; its 512 routed
experts come out int4 g64 per expert, the rest bf16, declared in `extra_config`.

Low CPU memory: the fused-MoE original is built as a meta skeleton and each block materialised from the
checkpoint, tuned, written to the export and freed - except the PLE table, see ple_table_from_file. Calibration (operator, 2026-11-12): nvidia/OpenCodeInstruct through the checkpoint's chat template (each row a
user turn and the assistant's answer, ending in <|im_end|>; the per-sample BOS/EOS reset is upstream, #2591) -
the model is for agentic coding. The pinned commit is the operator's fork commit whose `calib_dataset.py` reads the
dataset's cached parquet shards (`_load_cached_parquet_dataset`); its fallback is a hub download, which the offline
environment turns into an error. Before AutoRound starts, the cache must hold all of the dataset's
`data/train-*.parquet` shards (OPENCODE_SHARDS), or this refuses - the run can never fall back to pile-10k.

Resumable through AutoRound's per-block checkpoint
(`AR_RESUME_DIR`, set by the script): a rerun with the same settings skips the blocks already written.
"""
import argparse
import glob
import json
import os
import resource
import time

OPENCODE = "nvidia/OpenCodeInstruct"
OPENCODE_SHARDS = 50
IGNORE = ["shared_expert_gate", "in_proj_a", "in_proj_b", "indexer", ".ple.", "mtp.layers.0.self_attn",
          "mtp.layers.0.mlp.shared_expert"]


def settings(a) -> dict:
    return {
        "scheme": "W4A16", "group_size": 64, "sym": True,
        "iters": a.iters, "nsamples": a.nsamples, "seqlen": a.seqlen, "batch_size": a.batch_size,
        "dataset": a.dataset, "low_gpu_mem_usage": True, "low_cpu_mem_usage": True, "device_map": a.device,
        "ignore_layers": ",".join(IGNORE), "seed": a.seed, "disable_model_free": True,
    }


def drop_shared_expert_rule() -> None:
    """AutoRound's predefined Qwen4 ignore list keeps `shared_expert` in bf16; ours quantises it."""
    import auto_round.compressors.base as base
    orig = base.get_predefined_ignore_layers

    def ours(model):
        return [n for n in orig(model) if n != "shared_expert"]
    base.get_predefined_ignore_layers = ours


def ple_table_from_file(ple_dir: str) -> None:
    """The 95.4 GiB PLE table as a file-backed (mmap) tensor instead of AutoRound's in-memory concatenation.

    At the pinned commit, materialising layer 1 reads the 128 `ngram_embedding.shard_*` tensors and
    `torch.cat`s them (`disk_stream_util._assemble_sharded_tensor`): ~190 GiB at the peak on a 121 GiB box. Here
    the shards are streamed once into one raw bf16 file in `ple_dir` (reused on a resume) and the table is
    `torch.from_file` of it: pages are read when a lookup touches them and the kernel may drop them again, so only
    the rows the calibration tokens hit stay resident. The file must be on a fast disk (NVMe): the first lookups
    are millions of small random reads. The values are the shards' bytes in order - the `bf16` stage checks the
    exported table bitwise against the source."""
    import torch
    import auto_round.utils.disk_stream_util as ds
    orig = ds._assemble_sharded_tensor

    def ours(index, full_name, device):
        if not full_name.endswith("ngram_embedding.weight"):
            return orig(index, full_name, device)
        prefix = full_name[: -len("weight")] + "shard_"
        names = sorted((n for n in index.weight_map if n.startswith(prefix) and n.endswith(".weight")),
                       key=lambda n: int(n[len(prefix):-len(".weight")]))
        path = os.path.join(ple_dir, full_name + ".bf16")
        meta_path = path + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                rows, dim = json.load(f)["shape"]
        else:
            os.makedirs(ple_dir, exist_ok=True)
            rows, dim = 0, None
            with open(path + ".tmp", "wb") as f:
                for n in names:
                    t = index.read_tensors([n], device="cpu")[n]
                    assert t.dtype == torch.bfloat16, f"{n} is {t.dtype}"
                    rows, dim = rows + t.shape[0], t.shape[1]
                    t.contiguous().view(torch.int16).numpy().tofile(f)
                    del t
            os.replace(path + ".tmp", path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"shape": [rows, dim], "shards": len(names)}, f)
        print(f"PLE table {full_name}: {len(names)} shards -> {path} [{rows}][{dim}] bf16, file-backed", flush=True)
        return torch.from_file(path, shared=False, size=rows * dim, dtype=torch.bfloat16).view(rows, dim), "cpu"
    ds._assemble_sharded_tensor = ours


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
            "commit": os.environ.get("AR_COMMIT", "?"), "patched": os.environ.get("AR_PATCH", "?"),
            "base": os.environ.get("AR_BASE", "?"), "ref": os.environ.get("AR_REF", "?")}


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
    ap.add_argument("--ple-dir", required=True, help="where the PLE table's raw bf16 file goes (fast disk, ~102 GB)")
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
    drop_shared_expert_rule()
    ple_table_from_file(a.ple_dir)
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
