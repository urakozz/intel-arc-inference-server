#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `check` of tools/quantize_kolibri1.sh: would the engine accept this export?

    check.py <export-dir> --source <bf16 snapshot> [--attn int4|bf16] [--sample 64]

src/loader/quant.cc's rules, in Python (loader::QuantConfig::parse, LinearSrc::classify,
assert_quant_invariants), plus what spec 20 §3.1 fixes for Kolibri:

  quantization_config  bits 4, group_size 64, sym true, desc_act false or absent; quant_method
                       "gptq" | "auto-round"; packing_format "auto_round:auto_gptq" when present;
                       `dynamic` rules only "-:" exclusions; every `extra_config` entry bits 16, or
                       bits 4 at g64 / g128 symmetric
  int4 linears         every routed expert's gate/up/down_proj (and q/k/v/o_proj with --attn int4):
                       qweight I32 [K/8, N], scales F16 [K/64, N] (g64 exactly), finite (subnormals
                       counted), qzeros I32 all 0x77777777 when shipped, g_idx (if any) the identity
                       k // 64 - no activation-order permutation
  bf16 tensors         router `mlp.gate.weight`, `moe.router.expert_bias`, `shared_experts.*`, every
                       norm, `embed_tokens`, `lm_head` (no lm_head.qweight), attention with --attn
                       bf16: BF16 `.weight`, and BITWISE the bf16 source's tensor
  names                exactly kolibri_ref.expected_names (a .weight may be replaced by
                       .qweight/.scales/.qzeros[/.g_idx]); nothing else
  dequant sanity       --sample int4 linears dequantised by dequant.py's rule ((q - 8) * scale)
                       against the source weight: relative error < 0.2 (a wrong nibble order or
                       a transposed pack is ~1.4)

Exit 0 and "ACCEPTED" or exit 1 with every violation listed.
"""
import argparse
import json
import os
import random
import re
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

QZ = 0x77777777
INT4_SUFFIXES = (".qweight", ".scales", ".qzeros", ".g_idx")


def headers(d: str) -> dict:
    """name -> (file, dtype, shape) from the safetensors headers (no tensor is read)."""
    out = {}
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    for fn in files:
        with open(os.path.join(d, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            if k in out:
                raise SystemExit(f"{k} appears in two shards")
            out[k] = (fn, v["dtype"], tuple(v["shape"]))
    return out


def parse_quant_config(cfg: dict, errs: list) -> dict:
    q = cfg.get("quantization_config")
    if not q:
        errs.append("config.json has no quantization_config")
        return {}
    if q.get("bits") != 4 or q.get("group_size") != 64 or q.get("sym") is not True:
        errs.append(f"quantization_config: need bits 4, group_size 64, sym true; got bits {q.get('bits')}, "
                    f"group_size {q.get('group_size')}, sym {q.get('sym')}")
    if q.get("desc_act") not in (None, False):
        errs.append(f"quantization_config.desc_act {q.get('desc_act')}: activation order is not supported")
    if q.get("quant_method") not in (None, "gptq", "auto-round"):
        errs.append(f"quant_method {q.get('quant_method')!r} is neither 'gptq' nor 'auto-round'")
    if q.get("packing_format") not in (None, "auto_round:auto_gptq"):
        errs.append(f"packing_format {q.get('packing_format')!r} is not 'auto_round:auto_gptq'")
    for rule in (q.get("dynamic") or {}):
        if not rule.startswith("-:"):
            errs.append(f"dynamic rule {rule!r} is not a '-:' exclusion")
    excluded = []
    for mod, rule in (q.get("extra_config") or {}).items():
        if not isinstance(rule, dict):
            errs.append(f"extra_config[{mod!r}] is not an object")
            continue
        b = rule.get("bits", q.get("bits"))
        if b == 16:
            excluded.append(mod)
        elif b == 4:
            g, s = rule.get("group_size", q.get("group_size")), rule.get("sym", q.get("sym"))
            if g not in (64, 128) or not s:
                errs.append(f"extra_config[{mod!r}] is int4 g{g} sym={s}: the loader implements g64 / g128 sym")
        else:
            errs.append(f"extra_config[{mod!r}] bits={b}: the loader implements 4 and 16")
    return {"excluded": excluded, "q": q}


def check(export: str, source: str, attn: str | None, sample: int) -> tuple[list, dict]:
    ref = C.kolibri_ref()
    errs, info = [], {}
    cfg = C.read_json(os.path.join(export, "config.json"))
    qc = parse_quant_config(cfg, errs)
    if cfg.get("model_type") != "kolibri1":
        errs.append(f"model_type {cfg.get('model_type')!r}")
    c = ref.KConfig.from_dict({k: v for k, v in cfg.items() if k != "quantization_config"})
    hdr = headers(export)
    recipe_path = os.path.join(export, "kolibri_recipe.json")
    if attn is None and os.path.exists(recipe_path):
        attn = C.read_json(recipe_path).get("attn")
    if attn is None:
        attn = "int4" if any(k.endswith("self_attn.q_proj.qweight") for k in hdr) else "bf16"
    info["attn"] = attn
    int4, bf16 = [], []
    for name in ref.expected_names(c):
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        is_expert = ".mlp.experts." in name
        is_attn = re.search(r"self_attn\.[qkvo]_proj\.weight$", name) is not None
        (int4 if (is_expert or (is_attn and attn == "int4")) else bf16).append((name, base))
    seen = set()
    n_sub, int4_bytes, total_bytes = 0, 0, 0
    for name, base in int4:
        qw, sc = base + ".qweight", base + ".scales"
        if name in hdr:
            errs.append(f"{name} is shipped unquantised; expected {qw}")
            seen.add(name)
            continue
        if qw not in hdr or sc not in hdr:
            errs.append(f"{base}: missing {'qweight' if qw not in hdr else 'scales'}")
            continue
        seen.update({qw, sc})
        (_, dq, sq), (_, ds, ss) = hdr[qw], hdr[sc]
        if dq != "I32" or ds != "F16" or len(sq) != 2 or len(ss) != 2:
            errs.append(f"{base}: qweight {dq} {sq}, scales {ds} {ss}; need I32 [K/8, N] and F16 [K/64, N]")
            continue
        K, N = sq[0] * 8, sq[1]
        if ss != (K // 64, N) or K % 64:
            errs.append(f"{base}: scales {ss} for K={K}, N={N}: need g64 [{K // 64}, {N}]")
        for suf in (".qzeros", ".g_idx"):
            if base + suf in hdr:
                seen.add(base + suf)
    for name, _ in bf16:
        if name not in hdr:
            b = name[:-len(".weight")]
            errs.append(f"{name}: missing" + (" (shipped as int4, must stay bf16)" if b + ".qweight" in hdr else ""))
            continue
        seen.add(name)
        if hdr[name][1] != "BF16":
            errs.append(f"{name}: dtype {hdr[name][1]}, must be BF16")
    extra = sorted(set(hdr) - seen)
    if extra:
        errs.append(f"{len(extra)} unexpected tensors, e.g. {extra[:5]}")
    if any(k.startswith("lm_head.") and k.endswith(INT4_SUFFIXES) for k in hdr):
        errs.append("lm_head is quantised: spec 20 keeps it bf16 (the engine builds the int8 head at load)")

    # the data: qzeros, g_idx, scales; bf16 tensors against the source; a dequant sample
    from safetensors import safe_open
    handles = {fn: safe_open(os.path.join(export, fn), framework="pt") for fn in {v[0] for v in hdr.values()}}
    get = (lambda k: handles[hdr[k][0]].get_tensor(k))
    for k, (fn, dt, shp) in sorted(hdr.items()):
        n_el = 1
        for s in shp:
            n_el *= s
        nbytes = n_el * {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}.get(dt, 4)
        total_bytes += nbytes
        if k.endswith(INT4_SUFFIXES):
            int4_bytes += nbytes
        if k.endswith(".qzeros"):
            t = get(k)
            bad = (t != QZ).nonzero()
            if bad.numel():
                errs.append(f"{k}{bad[0].tolist()} = {int(t[tuple(bad[0])]) & 0xFFFFFFFF:#x}, expected 0x77777777")
        elif k.endswith(".g_idx"):
            t = get(k).long()
            if not torch.equal(t, torch.arange(t.numel()) // 64):
                errs.append(f"{k} is not the identity k // 64 (an activation-order permutation)")
        elif k.endswith(".scales"):
            t = get(k)
            bits = t.view(torch.int16).int() & 0xFFFF
            e, m = (bits >> 10) & 0x1F, bits & 0x3FF
            if bool((e == 0x1F).any()):
                errs.append(f"{k}: NaN / Inf scale")
            n_sub += int(((e == 0) & (m != 0)).sum())
    src = ref.Checkpoint(source) if source else None
    if src is not None:
        if src.cfg.group_size is not None:
            errs.append(f"--source {source} is quantised; it must be the bf16 snapshot")
        else:
            for name, _ in bf16:
                if name in hdr and not torch.equal(get(name).float(), src.tensor(name).float()):
                    errs.append(f"{name}: differs from the bf16 source (a tensor that should be untouched)")
            rng = random.Random(0)
            picks = rng.sample(int4, min(sample, len(int4)))
            worst = 0.0
            for name, base in picks:
                if base + ".qweight" not in hdr:
                    continue
                w = src.tensor(name).float()
                deq = ref._dequant.dequant_gptq(get(base + ".qweight"), get(base + ".scales"), 64).t().float()
                rel = float((deq - w).norm() / w.norm().clamp(min=1e-12))
                worst = max(worst, rel)
                if rel > 0.2:
                    errs.append(f"{base}: dequantised weight is {rel:.3f} relative error from the source "
                                f"(nibble order / transposition?)")
            info["dequant_rel_err_max"] = worst
            info["dequant_sampled"] = len(picks)
    info.update({"tensors": len(hdr), "int4_linears": len(int4), "bf16_tensors": len(bf16),
                 "excluded_extra_config": len(qc.get("excluded", [])), "subnormal_scales": n_sub,
                 "bytes_total": total_bytes, "bytes_int4": int4_bytes,
                 "quantization_config": {k: v for k, v in qc.get("q", {}).items() if k != "extra_config"}})
    return errs, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--source", help="the bf16 snapshot (bitwise check of the bf16 tensors, dequant sample)")
    ap.add_argument("--attn", choices=("int4", "bf16"))
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--json", help="write the result here")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.export, "config.json")):
        a.export = C.find_export(a.export)          # an arm directory: AutoRound's export is below it
    errs, info = check(a.export, C.resolve_model(a.source) if a.source else None, a.attn, a.sample)
    if a.json:
        C.write_json(a.json, {"export": a.export, "accepted": not errs, "errors": errs, **info})
    print(json.dumps(info, indent=1))
    if errs:
        print(f"REJECTED: {len(errs)} problem(s)", file=sys.stderr)
        for e in errs[:200]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print(f"ACCEPTED: {a.export} (int4 g64 sym, auto_round:auto_gptq, attention {info['attn']}, "
          f"{info['bytes_total'] / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
