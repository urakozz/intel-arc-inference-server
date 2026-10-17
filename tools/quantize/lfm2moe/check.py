#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would the engine accept this LFM2.5-24B-A2B export? (spec 24b Task 1, on kolibri's check.py's pattern.)

    check.py <export-dir> [--source <bf16 snapshot>] [--expert-names w|proj] [--sample 64]

src/loader/quant.cc's rules (loader::QuantConfig::parse, LinearSrc::classify, assert_quant_invariants) plus
spec 24 §5's fixes for LFM2.5-24B-A2B:

  quantization_config  bits 4, group_size 64, sym true, desc_act false or absent; quant_method "gptq" |
                       "auto-round"; packing_format "auto_round:auto_gptq" when present
  int4 linears         every routed expert's w1/w3/w2 (or gate_proj/up_proj/down_proj), the dense w1/w2/w3,
                       conv in_proj/out_proj, attention q/k/v/out_proj: qweight I32 [K/8, N], scales F16
                       [K/64, N] (g64 exactly), finite, qzeros I32 0x77777777 when shipped, g_idx (if any) the
                       identity k // 64
  bf16 / fp32 tensors  router feed_forward.gate.weight, conv.conv.weight, every norm, embed_tokens: BF16;
                       expert_bias: F32 (the module's buffer is fp32 whatever the model dtype)
  names                exactly lfm2moe_ref.expected_names(cfg, "int4", spelling) for ONE spelling (all or
                       nothing: a checkpoint mixing w1/w3/w2 and gate_proj/up_proj/down_proj is rejected); an
                       optional lm_head.weight must be bitwise embed_tokens.weight (the tie) or is rejected
                       naming both
  dequant sanity       --sample int4 linears dequantised by lfm2moe_make_tiny.dequant against --source: relative
                       error < 0.2

Exit 0 and "ACCEPTED" or exit 1 with every violation listed.
"""
import argparse
import json
import os
import random
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

INT4_SUFFIXES = (".qweight", ".scales", ".qzeros", ".g_idx")


def headers(d: str) -> dict:
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
    if q.get("extra_config"):
        errs.append("quantization_config.extra_config is set: spec 24 §5 has no excluded nn.Linear "
                    "(the router / conv.conv / norms / embedding are never nn.Linear)")
    return {"q": q}


def detect_spelling(hdr: dict) -> str | None:
    if any(".experts." in k and k.endswith(("w1.qweight", "w1.weight")) for k in hdr):
        return "w"
    if any(".experts." in k and k.endswith(("gate_proj.qweight", "gate_proj.weight")) for k in hdr):
        return "proj"
    return None


def check(export: str, source: str | None, expert_names: str | None, sample: int) -> tuple[list, dict]:
    ref = C.lfm2moe_ref()
    errs, info = [], {}
    cfg = C.read_json(os.path.join(export, "config.json"))
    qc = parse_quant_config(cfg, errs)
    if cfg.get("model_type") != "lfm2_moe":
        errs.append(f"model_type {cfg.get('model_type')!r}, expected lfm2_moe")
    hdr = headers(export)
    spelling = expert_names or detect_spelling(hdr)
    if spelling is None:
        errs.append("no experts.*.w1 nor experts.*.gate_proj found: neither expert spelling")
        spelling = "w"
    info["expert_names"] = spelling
    c = ref.config(None, raw={k: v for k, v in cfg.items() if k != "quantization_config"})
    want = ref.expected_names(c, "int4", spelling)
    other = ref.expected_names(c, "int4", "proj" if spelling == "w" else "w")

    int4_bases, bf16_names, fp32_names = [], [], []
    for name in want:
        if name.endswith(".qweight"):
            int4_bases.append(name[: -len(".qweight")])
        elif name.endswith("expert_bias"):
            fp32_names.append(name)
        elif not name.endswith((".scales", ".qzeros", ".g_idx")):
            bf16_names.append(name)
    seen = set()
    for base in int4_bases:
        qw, sc, qz = base + ".qweight", base + ".scales", base + ".qzeros"
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
        if qz in hdr:
            seen.add(qz)
        gi = base + ".g_idx"
        if gi in hdr:
            seen.add(gi)
    for name in bf16_names:
        if name not in hdr:
            errs.append(f"{name}: missing")
            continue
        seen.add(name)
        if hdr[name][1] != "BF16":
            errs.append(f"{name}: dtype {hdr[name][1]}, must be BF16")
    for name in fp32_names:
        if name not in hdr:
            errs.append(f"{name}: missing")
            continue
        seen.add(name)
        if hdr[name][1] not in ("F32", "FP32"):
            errs.append(f"{name}: dtype {hdr[name][1]}, must be F32 (expert_bias stays fp32, M:205)")
    if any(n in hdr for n in other if n not in want):
        errs.append(f"the checkpoint mixes expert spellings: names of both 'w' and 'proj' are present")
    lm_head = "lm_head.weight"
    if lm_head in hdr:
        seen.add(lm_head)
        from safetensors import safe_open
        emb_fn = hdr["model.embed_tokens.weight"][0]
        lm_fn = hdr[lm_head][0]
        with safe_open(os.path.join(export, emb_fn), framework="pt") as h1, \
             safe_open(os.path.join(export, lm_fn), framework="pt") as h2:
            emb = h1.get_tensor("model.embed_tokens.weight")
            lm = h2.get_tensor(lm_head)
        if emb.shape != lm.shape or not torch.equal(emb, lm):
            errs.append(f"lm_head.weight is shipped and differs from model.embed_tokens.weight: LFM2.5 ties "
                        f"the head (M:632) - refusing to guess which is right")
    extra = sorted(set(hdr) - seen)
    if extra:
        errs.append(f"{len(extra)} unexpected tensor(s), e.g. {extra[:5]}")

    from safetensors import safe_open
    handles = {fn: safe_open(os.path.join(export, fn), framework="pt") for fn in {v[0] for v in hdr.values()}}
    get = (lambda k: handles[hdr[k][0]].get_tensor(k))
    n_sub, int4_bytes, total_bytes = 0, 0, 0
    for k, (fn, dt, shp) in sorted(hdr.items()):
        n_el = 1
        for s in shp:
            n_el *= s
        nbytes = n_el * {"BF16": 2, "F16": 2, "F32": 4, "FP32": 4, "I32": 4}.get(dt, 4)
        total_bytes += nbytes
        if k.endswith(INT4_SUFFIXES):
            int4_bytes += nbytes
        if k.endswith(".qzeros"):
            t = get(k)
            bad = (t != C.QZ).nonzero()
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

    src = C.resolve_model(source) if source else None
    if src is not None:
        MT = C.lfm2moe_make_tiny()
        srcC = ref.Ckpt(src)
        for name in bf16_names + fp32_names:
            if name in hdr and name in srcC:
                want_t = srcC.get(name)
                got = get(name)
                if want_t.shape != got.shape or not torch.equal(want_t.float(), got.float()):
                    errs.append(f"{name}: differs from the bf16 source (a tensor that should be untouched)")
        rng = random.Random(0)
        picks = rng.sample(int4_bases, min(sample, len(int4_bases)))
        worst = 0.0
        for base in picks:
            if base + ".qweight" not in hdr:
                continue
            src_name = base.replace(".gate_proj", ".w1").replace(".up_proj", ".w3").replace(".down_proj", ".w2")
            src_name += ".weight"
            if src_name not in srcC:
                continue
            w = srcC.get(src_name).float()
            deq = MT.dequant(get(base + ".qweight"), get(base + ".scales")).float()
            rel = float((deq - w).norm() / w.norm().clamp(min=1e-12))
            worst = max(worst, rel)
            if rel > 0.2:
                errs.append(f"{base}: dequantised weight is {rel:.3f} relative error from the source")
        info["dequant_rel_err_max"] = worst
        info["dequant_sampled"] = len(picks)
    info.update({"tensors": len(hdr), "int4_linears": len(int4_bases), "bf16_tensors": len(bf16_names),
                 "subnormal_scales": n_sub, "bytes_total": total_bytes, "bytes_int4": int4_bytes,
                 "quantization_config": qc.get("q", {})})
    return errs, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--source", help="the bf16 snapshot (bitwise check of the untouched tensors, dequant sample)")
    ap.add_argument("--expert-names", choices=("w", "proj"))
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--json", help="write the result here")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.export, "config.json")):
        a.export = C.find_export(a.export)
    errs, info = check(a.export, a.source, a.expert_names, a.sample)
    if a.json:
        C.write_json(a.json, {"export": a.export, "accepted": not errs, "errors": errs, **info})
    print(json.dumps(info, indent=1))
    if errs:
        print(f"REJECTED: {len(errs)} problem(s)", file=sys.stderr)
        for e in errs[:200]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print(f"ACCEPTED: {a.export} (int4 g64 sym, auto_round:auto_gptq, expert names '{info['expert_names']}', "
          f"{info['bytes_total'] / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
