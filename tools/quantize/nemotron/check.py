#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would the engine accept this Nemotron 3.5 Super int4 export? (Nemotron 3.5 Bolt's check at 55e8923f, for Super.)

    check.py <export dir>

The int4 rules of src/loader/quant.cc (QuantConfig::parse, LinearSrc::classify, assert_quant_invariants) and the
names, dtypes and shapes the reference reads (tools/oracle/nemotron_ref.py expected_names - one list, not a copy):

  config         model_type nemotron_h_omni over llm_config.model_type nemotron_h, num_nextn_predict_layers 1 with
                 mtp_layers_block_type; quantization_config bits 4, group_size 64, sym true, quant_method auto-round,
                 packing_format auto_round:auto_gptq, no desc_act; every extra_config entry bits 16 (left in fp - the
                 MTP layer's dense linears, the routers); no `dynamic` rule
  names          exactly expected_names(llm_config, "int4") - the trunk (`language_model.backbone.*`), the untied
                 lm_head, and the WHOLE MTP layer (`language_model.mtp.layers.{0,1}.*`) - and nothing else: no
                 vision tower, no projector (the `strip` stage), no leftover `.weight` beside a `.qweight`
  int4 linears   qweight I32 [K/8][N], scales F16 [K/64][N] (finite; negative and zero counted, not refused -
                 AutoRound's signed symmetric scale), qzeros I32 [K/64][N/8] = 0x77777777, no g_idx
  kept tensors   as the original ships them: router BF16 [E][H] + F32 bias [E], A_log / D / dt_bias BF16 [heads],
                 conv1d BF16 [C][1][4] + bias, every norm BF16, embedding / lm_head BF16 [V][H], the MTP layer's
                 eh_proj / attention / shared expert / latent projections BF16 `.weight`

Exit 0 and "ACCEPTED" or exit 1 with every violation listed (the first 40).
"""
import argparse
import json
import os
import struct
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE = os.path.abspath(os.path.join(_HERE, "..", "..", "oracle"))


def headers(d: str) -> dict:
    with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    out = {}
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


def scale_stats(t: torch.Tensor) -> dict:
    f = t.float()
    w = t.view(torch.int16)
    return {"count": f.numel(), "negative": int((f < 0).sum()),
            "subnormal": int((((w & 0x7C00) == 0) & ((w & 0x3FF) != 0)).sum()),
            "zero": int((f == 0).sum()), "nonfinite": int((~torch.isfinite(f)).sum())}


def parse_quant(cfg: dict, errs: list) -> None:
    q = cfg.get("quantization_config")
    if not q:
        errs.append("config.json has no quantization_config")
        return
    if q.get("bits") != 4 or q.get("sym") is not True or q.get("group_size") != 64:
        errs.append(f"quantization_config: need bits 4, sym true, group_size 64; got bits {q.get('bits')}, "
                    f"sym {q.get('sym')}, group_size {q.get('group_size')}")
    if q.get("desc_act") not in (None, False):
        errs.append(f"quantization_config.desc_act {q.get('desc_act')}: activation order is not supported")
    if q.get("quant_method") != "auto-round" or q.get("packing_format") != "auto_round:auto_gptq":
        errs.append(f"quant_method {q.get('quant_method')!r} / packing_format {q.get('packing_format')!r}: need "
                    f"auto-round / auto_round:auto_gptq")
    if q.get("dynamic"):
        errs.append("quantization_config.dynamic is set (our export declares exclusions in extra_config)")
    for mod, rule in (q.get("extra_config") or {}).items():
        if not isinstance(rule, dict) or rule.get("bits") != 16:
            errs.append(f"extra_config[{mod!r}] = {rule}: only bits-16 (left in fp) entries are expected")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    a = ap.parse_args()
    errs: list = []
    with open(os.path.join(a.dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "nemotron_h_omni":
        errs.append(f"model_type {cfg.get('model_type')!r}, expected nemotron_h_omni")
    parse_quant(cfg, errs)
    sys.path.insert(0, _ORACLE)
    import nemotron_ref  # noqa: E402  (the names are the reference's, not a second copy)
    llm = nemotron_ref.config(a.dir)
    if not llm["mtp_kinds"]:
        errs.append("llm_config has no MTP layer (num_nextn_predict_layers / mtp_layers_block_type)")
    hdr = headers(a.dir)
    trunk = "backbone" if any(k.startswith("language_model.backbone.") for k in hdr) else "model"
    want = nemotron_ref.expected_names(llm, "int4", trunk)
    for n in sorted(set(want) - set(hdr))[:10]:
        errs.append(f"missing tensor {n}")
    for n in sorted(set(hdr) - set(want))[:10]:
        errs.append(f"unexpected tensor {n}")

    def expect(name, dtype, shape):
        if name in hdr and (hdr[name][1] != dtype or hdr[name][2] != tuple(shape)):
            errs.append(f"{name}: {hdr[name][1]} {list(hdr[name][2])}, expected {dtype} {list(shape)}")
    for name, v in want.items():
        if v is not None:
            expect(name, *v)
    g = 64
    by_file: dict = {}
    n_lin = n_mtp = 0
    kinds = [("language_model." + trunk + f".layers.{i}.", k) for i, k in enumerate(llm["kinds"])]
    kinds += [(f"language_model.mtp.layers.{j}.", k) for j, k in enumerate(llm["mtp_kinds"])]
    for p, kind in kinds:
        for suf, (K, N) in nemotron_ref.linear_shapes(llm, kind).items():
            base = p + "mixer." + suf
            if base + ".qweight" not in want:
                continue
            n_lin += 1
            n_mtp += base.startswith("language_model.mtp.")
            if K % g or N % 8:
                errs.append(f"{base}: K {K} N {N} at group {g}")
                continue
            expect(base + ".qweight", "I32", (K // 8, N))
            expect(base + ".scales", "F16", (K // g, N))
            expect(base + ".qzeros", "I32", (K // g, N // 8))
            for s in (".qzeros", ".scales"):
                if base + s in hdr:
                    by_file.setdefault(hdr[base + s][0], []).append(base + s)

    from safetensors import safe_open
    neg = tot = sub = zero = 0
    for fn, names in sorted(by_file.items()):
        with safe_open(os.path.join(a.dir, fn), framework="pt") as h:
            for name in names:
                t = h.get_tensor(name)
                if name.endswith(".qzeros"):
                    if not bool((t == 0x77777777).all()):
                        errs.append(f"{name} is not 0x77777777 (symmetric zero point 8)")
                    continue
                st = scale_stats(t)
                if st["nonfinite"]:
                    errs.append(f"{name} has NaN / Inf")
                neg, tot, sub, zero = neg + st["negative"], tot + st["count"], sub + st["subnormal"], zero + st["zero"]
        if len(errs) > 40:
            break
    for e in errs[:40]:
        print("REFUSED:", e)
    if errs:
        print(f"check: {len(errs)} violation(s) in {a.dir}")
        return 1
    print(f"check: {a.dir}: {len(llm['kinds'])} blocks + the MTP layer ({'/'.join(llm['mtp_kinds'])}), {n_lin} int4 "
          f"linears (group {g}; {n_mtp} of them the MTP layer's routed experts), {len(hdr)} tensors, no vision; "
          f"scales {tot}: {neg} negative ({100.0 * neg / max(tot, 1):.1f} %), {sub} subnormal, {zero} zero")
    print("ACCEPTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
