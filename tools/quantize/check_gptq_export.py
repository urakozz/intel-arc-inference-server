#!/usr/bin/env python3
"""Refuse an AutoRound export unless the engine's loader would take it, byte for byte.

What `loader::QuantConfig::parse`, `LinearSrc::classify` and
`assert_quant_invariants` (src/loader/quant.{h,cc}, docs/13-loader.md) accept,
checked offline on the safetensors headers plus the few tensors whose *values*
matter (every .qzeros word, every .scales f16, every .g_idx):

  config   quantization_config: bits 4, group_size 64 (or --group), sym true,
           desc_act false or absent, quant_method gptq | auto-round,
           packing_format auto_round:auto_gptq when present, `dynamic` only
           "-:" exclusion rules, every `extra_config` module either bits 16
           (left in fp) or bits 4 g64/g128 sym.
  tensors  every <prefix>.qweight has .qzeros and .scales beside it, no
           <prefix>.weight left over; qweight I32 [K/8][N], scales F16
           [K/g][N] and finite, qzeros I32 [K/g][N/8] and every word 0x77777777
           (GPTQ v1 zeros, symmetric zero point 8), g_idx absent or the identity
           k / g.
  lm_head  --lm-head quant: lm_head.{qweight,qzeros,scales} shipped and declared
           (extra_config["lm_head"] bits 4) - auto-round after 0.14.2 drops
           exactly this packing on its default RTN path; --lm-head keep:
           lm_head.weight shipped, no lm_head.qweight.
  coverage --source DIR (the unquantised model): every 2-D `.weight` of the
           source except the embeddings is either packed in the export or
           excluded by the config (extra_config bits 16 / dynamic "-:" / an
           lm_head left in fp) - so a silently skipped linear is a FAIL. Also
           WARNs (not fatal) for every unpacked tensor whose dtype differs from
           the source's: AutoRound on a CPU without bf16 writes them as F32.

numpy is the only dependency (no torch): any venv with numpy, or on the Mac
`uv run --with numpy`. Exit 0 and one VERIFIED line, or exit 1 with every
failure listed.

  python3 tools/quantize/check_gptq_export.py OUT_DIR --lm-head quant [--source SRC_DIR]
  uv run --with numpy tools/quantize/check_gptq_export.py OUT_DIR ...
"""

import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np

PACKING = "auto_round:auto_gptq"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensor_bytes(path, data_start, t):
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        return f.read(b - a)


def index_dir(d):
    """{name: (tensor_info, file, data_start)} over every *.safetensors in d."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h, start = read_header(p)
        for k, t in h.items():
            out[k] = (t, p, start)
    return out


def find_config(root):
    hits = sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True),
                  key=lambda p: p.count(os.sep))
    hits = [p for p in hits if "quantization_config" in json.load(open(p))]
    if not hits:
        sys.exit(f"FAIL: no config.json with a quantization_config under {root}")
    return hits[0]


def excluded_by_config(name, qc):
    """True when the config says module `name` is left in fp."""
    mod = name[: -len(".weight")]
    ec = qc.get("extra_config") or {}
    for key, rule in ec.items():
        if not isinstance(rule, dict) or rule.get("bits", qc["bits"]) != 16:
            continue
        if key == mod or (any(c in key for c in "*^$[]()+?\\") and re.fullmatch(key, mod)):
            return True
    for rule in (qc.get("dynamic") or {}):
        if rule.startswith("-:") and re.match(rule[2:], mod):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="the export directory (config.json is searched recursively)")
    ap.add_argument("--group", type=int, default=64, help="the group size the recipe asked for (64)")
    ap.add_argument("--lm-head", choices=["quant", "keep"], default="quant")
    ap.add_argument("--source", help="the unquantised model directory, for the coverage check")
    a = ap.parse_args()

    fails = []
    cfg_path = find_config(a.out)
    d = os.path.dirname(cfg_path)
    qc = json.load(open(cfg_path))["quantization_config"]

    # --- config, as QuantConfig::parse reads it ---
    if qc.get("bits") != 4 or qc.get("group_size") != a.group or qc.get("sym") is not True:
        fails.append(f"config: bits={qc.get('bits')} group_size={qc.get('group_size')} sym={qc.get('sym')}, "
                     f"need 4 / {a.group} / true")
    if qc.get("desc_act") not in (None, False):
        fails.append(f"config: desc_act={qc.get('desc_act')}")
    if qc.get("quant_method") not in (None, "gptq", "auto-round"):
        fails.append(f"config: quant_method {qc.get('quant_method')!r}")
    if qc.get("packing_format") not in (None, PACKING):
        fails.append(f"config: packing_format {qc.get('packing_format')!r}, need {PACKING!r}")
    for rule in (qc.get("dynamic") or {}):
        if not rule.startswith("-:"):
            fails.append(f"config: dynamic rule {rule!r} is not a '-:' exclusion")
    ec = qc.get("extra_config") or {}
    for mod, rule in ec.items():
        if not isinstance(rule, dict):
            fails.append(f"config: extra_config[{mod!r}] is not an object")
            continue
        mb = rule.get("bits", qc.get("bits"))
        if mb == 16:
            continue
        mg, ms = rule.get("group_size", qc.get("group_size")), rule.get("sym", qc.get("sym"))
        if mb != 4 or mg not in (64, 128) or ms is not True:
            fails.append(f"config: extra_config[{mod!r}] bits={mb} group_size={mg} sym={ms}")

    # --- tensors, as classify + assert_quant_invariants read them ---
    ts = index_dir(d)
    prefixes = sorted(k[: -len(".qweight")] for k in ts if k.endswith(".qweight"))
    if not prefixes:
        fails.append("tensors: no .qweight anywhere - nothing was packed")
    zeros_words = 0
    for p in prefixes:
        qw = ts[p + ".qweight"][0]
        missing = [s for s in (".qzeros", ".scales") if p + s not in ts]
        if missing:
            fails.append(f"{p}: qweight without {missing}")
            continue
        if p + ".weight" in ts:
            fails.append(f"{p}: both .qweight and an unpacked .weight shipped")
        sc, sc_file, sc_start = ts[p + ".scales"]
        qz, qz_file, qz_start = ts[p + ".qzeros"]
        if qw["dtype"] != "I32" or sc["dtype"] != "F16" or qz["dtype"] != "I32":
            fails.append(f"{p}: dtypes qweight {qw['dtype']} scales {sc['dtype']} qzeros {qz['dtype']}, "
                         f"need I32 / F16 / I32")
            continue
        if len(qw["shape"]) != 2 or len(sc["shape"]) != 2 or len(qz["shape"]) != 2:
            fails.append(f"{p}: non rank-2 qweight/scales/qzeros")
            continue
        K, N = qw["shape"][0] * 8, qw["shape"][1]
        g = a.group
        if sc["shape"] != [K // g, N] or qz["shape"] != [K // g, N // 8] or K % g:
            fails.append(f"{p}: K={K} N={N}: scales {sc['shape']} qzeros {qz['shape']}, "
                         f"need [{K // g}, {N}] / [{K // g}, {N // 8}] (group {g})")
            continue
        z = np.frombuffer(tensor_bytes(qz_file, qz_start, qz), dtype="<u4")
        bad = np.nonzero(z != 0x77777777)[0]
        if bad.size:
            fails.append(f"{p}.qzeros[{bad[0]}] = {hex(int(z[bad[0]]))}, need 0x77777777 "
                         f"({bad.size} of {z.size} words differ)")
        zeros_words += z.size
        s = np.frombuffer(tensor_bytes(sc_file, sc_start, sc), dtype="<f2")
        if not np.isfinite(s).all():
            fails.append(f"{p}.scales: non-finite f16 at {int(np.nonzero(~np.isfinite(s))[0][0])}")
        if p + ".g_idx" in ts:
            gi, gf, gs = ts[p + ".g_idx"]
            v = np.frombuffer(tensor_bytes(gf, gs, gi), dtype="<i4")
            if not np.array_equal(v, np.arange(K, dtype=np.int32) // g):
                fails.append(f"{p}.g_idx is not the identity k / {g} (an activation-order permutation)")

    # --- lm_head ---
    lm_keys = sorted(k for k in ts if k.startswith("lm_head."))
    if a.lm_head == "quant":
        if "lm_head.qweight" not in ts:
            fails.append(f"lm_head NOT packed (asked for --quant_lm_head); shipped only {lm_keys}")
        lm_rule = ec.get("lm_head")
        if not isinstance(lm_rule, dict) or lm_rule.get("bits", qc.get("bits")) != 4:
            fails.append(f"config: extra_config['lm_head'] = {lm_rule!r}, need a bits-4 entry")
    else:
        if "lm_head.weight" not in ts or "lm_head.qweight" in ts:
            fails.append(f"lm_head should stay in fp; shipped {lm_keys}")

    # --- coverage against the source model ---
    covered = None
    if a.source:
        src = index_dir(a.source)
        covered = 0
        for k, (t, _, _) in sorted(src.items()):
            if not k.endswith(".weight") or len(t["shape"]) != 2 or "embed" in k:
                continue
            p = k[: -len(".weight")]
            if p + ".qweight" in ts:
                covered += 1
            elif not (excluded_by_config(k, qc) or (p == "lm_head" and a.lm_head == "keep")):
                fails.append(f"coverage: source linear {p} is neither packed nor excluded by the config")
        widened = sorted(f"{k} {src[k][0]['dtype']}->{t['dtype']}" for k, (t, _, _) in ts.items()
                         if k in src and t["dtype"] != src[k][0]["dtype"])
        if widened:
            print(f"WARN: {len(widened)} unpacked tensor(s) changed dtype from the source "
                  f"(e.g. {widened[:3]}) - AutoRound on a CPU without bf16 writes them as F32")

    if fails:
        print(f"FAIL: {len(fails)} problem(s) in {d}:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    excl = sum(1 for r in ec.values() if isinstance(r, dict) and r.get("bits") == 16)
    print(f"VERIFIED B70-READY: int4 g{a.group} sym, {qc.get('quant_method')} / {qc.get('packing_format')}, "
          f"{len(prefixes)} packed linears{f' (all {covered} source linears accounted for)' if covered is not None else ''}, "
          f"{zeros_words} qzeros words all 0x77777777, lm_head "
          f"{'int4' if a.lm_head == 'quant' else 'fp'}, {excl} extra_config fp exclusions, "
          f"autoround_version {qc.get('autoround_version')} -> {d}")


if __name__ == "__main__":
    main()
