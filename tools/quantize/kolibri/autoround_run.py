#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stages `rtn` and `tune` of tools/quantize_kolibri1.sh: AutoRound with spec 20 §3.1's settings.

    autoround_run.py --shim <dir> --calib <calib_final.safetensors> --out <dir> --iters 0|200
                     --attn int4|bf16 [--device cpu|xpu|cuda] [--nsamples N] [--print-only]

The model is loaded by AutoRound from the shim (the bf16 snapshot + third_party/kolibri1 through
`auto_map`, trust_remote_code), so AutoRound sees one nn.Linear per routed expert projection
(`mlp.experts.{e}.{gate,up,down}_proj`, no fused 3-D expert tensors for its MoE unfusing to rewrite)
and never sees the router as a linear at all (`mlp.gate` is a plain parameter holder).

Settings (§3.1): scheme W4A16, group 64, symmetric (`sym=True`, AutoRound's default; the engine's
`w = (q - 8) * scale`, qzeros 0x77777777), format `auto_round:auto_gptq` (what
loader::QuantConfig::parse accepts), model dtype bf16 (every tensor left unquantised is written bf16).
Kept in bf16: `shared_experts` (ignore_layers), all of `self_attn` when --attn bf16, the router and
`expert_bias` (not nn.Linear), the norms, `embed_tokens`, `lm_head` (no quant_lm_head).
`--iters 0` is plain RTN (disable_opt_rtn: no data-driven scale search - the baseline); `--iters 200`
is AutoRound's sign-SGD tuning on the calibration rows (nsamples x seqlen), low_gpu_mem_usage.
The calibration rows go in pre-tokenised (one [1, seqlen] tensor each), so AutoRound neither
re-tokenises nor re-packs them.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def autoround_commit() -> dict:
    import auto_round
    info = {"version": getattr(auto_round, "__version__", "?"), "path": os.path.dirname(auto_round.__file__)}
    src = os.environ.get("KOLIBRI_AR_SRC_DIR")
    if src and os.path.isdir(os.path.join(src, ".git")):
        try:
            info["commit"] = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True,
                                            text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if "commit" not in info and os.environ.get("KOLIBRI_AR_COMMIT"):
        info["commit"] = os.environ["KOLIBRI_AR_COMMIT"] + " (as installed by tools/quantize_kolibri1.sh)"
    return info


def settings(a, nsamples: int) -> dict:
    ignore = ["shared_experts"] + (["self_attn"] if a.attn == "bf16" else [])
    kw = {
        "scheme": "W4A16", "group_size": 64, "sym": True,
        "iters": a.iters, "nsamples": nsamples, "seqlen": a.seqlen, "batch_size": a.batch_size,
        "low_gpu_mem_usage": True, "device_map": a.device, "ignore_layers": ",".join(ignore),
        "model_dtype": "bf16", "seed": a.seed, "disable_model_free": True,
    }
    if a.iters == 0:
        kw["disable_opt_rtn"] = True
    if a.torch_compile != "auto":
        kw["enable_torch_compile"] = a.torch_compile == "on"
    return kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="the bf16 model (HF id or snapshot); (re)makes --shim from it")
    ap.add_argument("--shim", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--attn", choices=("int4", "bf16"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nsamples", type=int, default=0, help="0 = every row of --calib")
    ap.add_argument("--seqlen", type=int, default=C.SEQLEN)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", help="the bf16 snapshot whose bytes the unquantised tensors must equal "
                                     "(default: the shim, which links to it)")
    ap.add_argument("--torch-compile", choices=("auto", "on", "off"), default="auto",
                    help="AutoRound's torch.compile of each block (auto = its own default)")
    ap.add_argument("--print-only", action="store_true", help="print the AutoRound call and exit")
    a = ap.parse_args()

    import torch
    if a.print_only and not os.path.exists(a.calib):
        ids, cats, meta, n = None, None, None, f"<rows of {a.calib}>"
    else:
        ids, cats, meta = C.load_rows(a.calib)
        if ids.shape[1] != a.seqlen:
            raise SystemExit(f"{a.calib}: rows are {ids.shape[1]} tokens, --seqlen {a.seqlen}")
        n = ids.shape[0] if a.nsamples <= 0 else min(a.nsamples, ids.shape[0])
    kw = settings(a, n)
    call = (f"AutoRound({a.shim!r}, dataset=<{n} rows of {a.calib}>, "
            + ", ".join(f"{k}={v!r}" for k, v in kw.items())
            + f").quantize_and_save({a.out!r}, format='auto_round:auto_gptq')")
    print(call, flush=True)
    if a.print_only:
        return
    if a.model:
        source = C.resolve_model(a.model)
        C.make_shim(source, a.shim)
        a.source = a.source or source
    from auto_round import AutoRound
    data = [ids[i:i + 1].clone() for i in range(n)]
    ar = AutoRound(a.shim, dataset=data, **kw)
    ar.quantize_and_save(a.out, format="auto_round:auto_gptq")
    export = C.find_export(a.out)
    restored = C.restore_bf16(export, a.source or a.shim)
    C.log(f"bf16 tensors: {restored}")
    recipe = {"autoround": autoround_commit(), "call": call, "settings": kw, "calib": os.path.abspath(a.calib),
              "calib_sha256": C.sha256_file(a.calib), "calib_rows": n, "calib_meta": meta,
              "attn": a.attn, "iters": a.iters, "torch": torch.__version__, "bf16_restore": restored}
    C.write_json(os.path.join(export, "kolibri_recipe.json"), recipe)
    C.log(f"exported to {export}")


if __name__ == "__main__":
    main()
