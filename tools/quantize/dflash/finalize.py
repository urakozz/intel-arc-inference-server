#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `export` of tools/quantize/quantize_dflash.sh: AutoRound's raw export -> the drafter checkpoint.

    finalize.py <raw export dir> <source drafter snapshot> <checkpoint dir>

- tensor names back to the checkpoint's (model.py's `encoder.fc.*` / `encoder.hidden_norm.*` -> `fc.*` /
  `hidden_norm.*`), in the tensors and in quantization_config (extra_config keys, block_name_to_quantize);
- every unpacked tensor is the source's, bit for bit: the norms, the conv base kernels, the selector codebooks.
  AutoRound on a CPU without bf16 writes them as fp32 widenings: those are checked to be exactly the widened source
  and replaced by the source's bytes; anything else is refused (exit 1);
- every source tensor is either packed (`.qweight` / `.scales` / `.qzeros`) or shipped unpacked; no embedding,
  no head (the target's are shared);
- config.json = the source's (architectures DFlash2DraftModel, dflash_config) + AutoRound's quantization_config;
  `mask_embedding.pt` / `mask_embedding.safetensors` copied byte for byte when the source ships one;
- dflash_recipe.json (autoround_run.py) copied, and README.md with the provenance.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

INT4 = (".qweight", ".scales", ".qzeros", ".g_idx")
MASK_FILES = ("mask_embedding.pt", "mask_embedding.safetensors")


def raw_dir(root: str) -> str:
    hits = sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True), key=len)
    if not hits:
        sys.exit(f"finalize: no config.json under {root}")
    return os.path.dirname(hits[0])


def read_all(d: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(p, "pt") as f:
            for n in f.keys():
                out[n] = f.get_tensor(n)
    return out


def rename_qc(qc: dict) -> dict:
    qc = dict(qc)
    if isinstance(qc.get("extra_config"), dict):
        qc["extra_config"] = {M.export_name(k): v for k, v in qc["extra_config"].items()}
    def fc(b):
        if isinstance(b, list):
            return [fc(x) for x in b]
        return ",".join("fc" if s == "encoder" else s for s in b.split(","))
    if qc.get("block_name_to_quantize") is not None:
        qc["block_name_to_quantize"] = fc(qc["block_name_to_quantize"])
    return qc


def readme(recipe: dict, source: str) -> str:
    ar, s, c = recipe["autoround"], recipe["settings"], recipe["calibration"]
    return f"""# Qwen3.8-27B-DFlash2, W4A16 g64 (AutoRound, tuned)

`z-lab/Qwen3.8-27B-DFlash2` (`{os.path.basename(os.path.normpath(source))}`) quantised by
`tools/quantize/quantize_dflash.sh`: int4 symmetric, group 64, `auto_round:auto_gptq`
(`w = (q - 8) * scale`, qzeros 0x77777777, no act-order).

## Quantisation provenance

- tool: auto-round {ar['version']} at commit `{ar['commit']}` (AR_PATCH={ar['patched']}, from {ar['repo']})
- recipe: scheme {s['scheme']}, group_size {s['group_size']}, sym {s['sym']}, iters {s['iters']}, nsamples
  {s['nsamples']}, batch_size {s['batch_size']}, seed {s['seed']}, format {s['format']}, quant_lm_head
  {s['quant_lm_head']} (the embedding and head are the target's); the full call is in dflash_recipe.json
- quantised: fc, every layer's q/k/v/o_proj, gate/up/down_proj, attention_conv / mlp_conv kernel_projection,
  candidate_selector.hidden_projection; kept at source precision (bitwise): {', '.join(recipe['kept_source_precision'])}
- calibration: the non-causal DFlash2DraftModel forward (block size {c['block_size']}, {c['anchors_per_sample']}
  blocks per sample, {c['context_rows']} context rows) on the quantised target's taps at layers
  {c['target_layer_ids']} (`{c['target']}`), {c['samples']} samples over {c['requests_used']} requests of
  {', '.join(c['recordings'])}; held out: {', '.join(c['held_out'])}
- made {recipe['made']} in {recipe['seconds']} s (torch {recipe['torch']})
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("source")
    ap.add_argument("out")
    a = ap.parse_args()
    raw = raw_dir(a.raw)
    with open(os.path.join(raw, "config.json")) as f:
        qc = rename_qc(json.load(f)["quantization_config"])
    with open(os.path.join(a.source, "config.json")) as f:
        cfg = json.load(f)
    src_map = M.ref._weight_map(a.source)
    handles = {}

    def src(n):
        if src_map[n] not in handles:
            handles[src_map[n]] = safe_open(src_map[n], "pt")
        return handles[src_map[n]].get_tensor(n)

    t = {M.export_name(n): v for n, v in read_all(raw).items()}
    errs, restored = [], 0
    for n in list(t):
        if n.endswith(INT4):
            continue
        if n not in src_map:
            errs.append(f"{n}: not a source tensor")
            continue
        s = src(n)
        if t[n].dtype == s.dtype and torch.equal(t[n], s):
            continue
        if t[n].dtype == torch.float32 and s.dtype != torch.float32 and torch.equal(t[n], s.float()):
            t[n] = s
            restored += 1
        else:
            errs.append(f"{n}: {t[n].dtype} {list(t[n].shape)} is not the source's {s.dtype} {list(s.shape)}")
    for n in src_map:
        if n not in t and n[: -len(".weight")] + ".qweight" not in t:
            errs.append(f"{n}: in the source, neither shipped nor packed")
    if errs:
        for e in errs[:20]:
            print("REFUSED:", e)
        sys.exit(f"finalize: {len(errs)} problem(s)")
    os.makedirs(a.out, exist_ok=True)
    save_file(t, os.path.join(a.out, "model.safetensors.tmp"), metadata={"format": "pt"})
    os.replace(os.path.join(a.out, "model.safetensors.tmp"), os.path.join(a.out, "model.safetensors"))
    cfg["quantization_config"] = qc
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for m in MASK_FILES:
        if os.path.isfile(os.path.join(a.source, m)):
            shutil.copyfile(os.path.join(a.source, m), os.path.join(a.out, m))
    recipe_path = os.path.join(a.raw, "dflash_recipe.json")
    with open(recipe_path) as f:
        recipe = json.load(f)
    shutil.copyfile(recipe_path, os.path.join(a.out, "dflash_recipe.json"))
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(readme(recipe, a.source))
    packed = sum(1 for n in t if n.endswith(".qweight"))
    print(f"finalize: {packed} packed linears, {len(t) - 3 * packed} source tensors bitwise "
          f"({restored} restored from fp32) -> {a.out}")


if __name__ == "__main__":
    main()
