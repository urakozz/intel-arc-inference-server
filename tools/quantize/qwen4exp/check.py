#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would the engine accept this Qwen3.8-Flash-Next export? (spec 21b Task 1; 21q's `check` stage.)

    check.py <checkpoint> [--ple <int8 dir>] [--sample 64]

src/loader/quant.cc's rules in Python (loader::QuantConfig::parse, LinearSrc::classify,
assert_quant_invariants) plus what spec 21 §5 / plan 21b fix for this family:

  quantization_config  bits 4, group_size 64 (ours) or 128 (Intel's), sym, desc_act false or absent,
                       quant_method auto-round | gptq, packing_format auto_round:auto_gptq; every
                       extra_config entry bits 16, or bits 4 at g64 / g128 symmetric
  names                exactly qwen4exp_ref.expected_names(text config, form, mtp) - the form read from
                       the names (layer 0's linear_attn.in_proj_qkv.qweight: ours, .weight: intel); visual
                       tensors ignored (the engine skips the tower)
  dtypes / shapes      every tensor against make_synth's table of the real widths; int4 linears qweight
                       I32 [K/8, N], scales F16 [K/g, N] (g 64; 128 for Intel's routed experts), qzeros I32
                       [K/g, N/8] all 0x77777777, scales finite; no g_idx
  PLE                  the three I64 tensors equal the formula (qwen4exp_facts) for the config's
                       ngram_vocab_size_base / seed / vocab
  --ple <dir>          ple_int8.py's file: per head ple.h<h>.q I8 [prime_h][160] and ple.h<h>.s
                       (one scale dtype), the I64 tensors = the formula; --sample rows a head (and each
                       head's first and last row) re-quantised from the checkpoint's shards by spec 9's
                       rule: q and s equal bit for bit

Exit 0 and "ACCEPTED", or exit 1 with every violation listed (the first 20).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_synth as MS  # noqa: E402
import ple_int8 as PI  # noqa: E402

QZ = 0x77777777


def parse_quant_config(cfg: dict, form: str, errs: list) -> None:
    q = cfg.get("quantization_config")
    if not q:
        errs.append("config.json has no quantization_config")
        return
    want_g = 64 if form == "ours" else 128
    if q.get("bits") != 4 or q.get("group_size") != want_g or q.get("sym") is not True:
        errs.append(f"quantization_config: the {form} form needs bits 4, group_size {want_g}, sym true; got bits "
                    f"{q.get('bits')}, group_size {q.get('group_size')}, sym {q.get('sym')}")
    if q.get("desc_act") not in (None, False):
        errs.append(f"quantization_config.desc_act {q.get('desc_act')}: activation order is not supported")
    if q.get("quant_method") not in (None, "gptq", "auto-round"):
        errs.append(f"quant_method {q.get('quant_method')!r} is neither 'gptq' nor 'auto-round'")
    if q.get("packing_format") not in (None, "auto_round:auto_gptq"):
        errs.append(f"packing_format {q.get('packing_format')!r} is not 'auto_round:auto_gptq'")
    for mod, rule in (q.get("extra_config") or {}).items():
        b = rule.get("bits", q.get("bits"))
        if b == 16:
            continue
        if b != 4 or rule.get("group_size", q.get("group_size")) not in (64, 128) or rule.get("sym", q.get("sym")) is not True:
            errs.append(f"extra_config[{mod!r}] = {rule}: only bits 16, or int4 g64 / g128 sym")


def spec_table(t: dict, s: MS.Shape, mtp: bool) -> dict:
    """name -> shape of every `.weight` / I64 tensor of the text model (before int4 packing)."""
    out = {}
    P = MS.P
    for n, sh in MS.hc_specs(P + "hyper_connection_mixer", s, False):
        out[n] = sh
    out[P + "embed_tokens.weight"] = (s.V, s.H)
    out["lm_head.weight"] = (s.V, s.H)
    ple_layers = {x - 1 for x in t.get("ple_layer_ids") or []}
    sizes, offsets, total, padded, _ = MS.ple_tables(t)
    parts = t.get("split_ngram_parts", 512)

    def layer(b, kind, ple):
        sp = MS.hc_specs(b + "attn_hyper_connection", s, True) + MS.hc_specs(b + "mlp_hyper_connection", s, True)
        sp += MS.moe_dense_specs(b, s) + (MS.gdn_specs(b, s) if kind == "gdn" else MS.qsa_specs(b, s))
        for e in range(s.E):
            sp += MS.expert_specs(b, s, e)
        if ple:
            sp += MS.ple_specs(b, s, padded // parts, parts)
            for k in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                sp.append((b + "ple.ple_embedding." + k, (3 if k == "layer_multipliers" else len(sizes),)))
        for n, sh in sp:
            out[n] = sh

    for i, lt in enumerate(t["layer_types"]):
        layer(f"{P}layers.{i}.", "gdn" if lt == "linear_attention" else "qsa", i in ple_layers)
    if mtp:
        for n, sh in [("mtp.fc_embedding.weight", (s.H, s.H)), ("mtp.fc_hidden.weight", (s.H, s.H)),
                      ("mtp.pre_fc_norm_embedding.weight", (s.H,)), ("mtp.pre_fc_norm_hidden.weight", (s.HC,))]:
            out[n] = sh
        for n, sh in MS.hc_specs("mtp.hyper_connection_mixer", s, False):
            out[n] = sh
        layer("mtp.layers.0.", "qsa", False)
    return out


def check(ckpt: str, ple_dir: str | None = None, sample: int = 64, seed: int = 0) -> list:
    errs: list = []
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    t = MS.text_of(cfg)
    tm = PI.tensor_map(ckpt)
    names = {n for n in tm if not n.startswith("model.visual.")}
    form = "ours" if MS.P + "layers.0.linear_attn.in_proj_qkv.qweight" in names else "intel"
    mtp = any(n.startswith("mtp.") for n in names)
    parse_quant_config(cfg, form, errs)
    r = MS.ref()
    tc = r.text_config(ckpt, raw=cfg)
    want = set(r.expected_names(tc, form, mtp=mtp))
    for n in sorted(want - names)[:10]:
        errs.append(f"missing {n}")
    for n in sorted(names - want)[:10]:
        errs.append(f"unexpected {n} (not in qwen4exp_ref.expected_names(tc, {form!r}))")
    s = MS.Shape(t)
    shapes = spec_table(t, s, mtp)
    for n in sorted(names & want):
        e = tm[n][1]
        if n.endswith((".qweight", ".scales", ".qzeros")):
            base = n.rsplit(".", 1)[0]
            N, K = shapes[base + ".weight"]
            g = 128 if (form == "intel" and ".mlp.experts." in n) else 64
            exp = {".qweight": ("I32", [K // 8, N]), ".scales": ("F16", [K // g, N]),
                   ".qzeros": ("I32", [K // g, N // 8])}["." + n.rsplit(".", 1)[1]]
            if [e["dtype"], e["shape"]] != list(exp):
                errs.append(f"{n}: {e['dtype']} {e['shape']}, expected {exp[0]} {exp[1]} (g{g})")
                continue
            arr = PI.mmap_of(tm[n])
            if n.endswith(".qzeros") and not bool((np.asarray(arr).view(np.uint32) == QZ).all()):
                errs.append(f"{n}: not all 0x77777777 (symmetric zero point 8)")
            if n.endswith(".scales") and not bool(np.isfinite(np.asarray(arr, dtype=np.float32)).all()):
                errs.append(f"{n}: a non-finite f16 scale")
        elif n.endswith(".g_idx"):
            errs.append(f"{n}: a g_idx (activation order) - the family's one format has none")
        else:
            sh = shapes.get(n)
            dt = "I64" if "ple_embedding." in n and "ngram_embedding" not in n else "BF16"
            if sh is None or e["shape"] != list(sh) or e["dtype"] != dt:
                errs.append(f"{n}: {e['dtype']} {e['shape']}, expected {dt} {list(sh) if sh else '?'}")
    # The I64 tensors against the formula.
    ids = t.get("ple_layer_ids") or []
    if ids:
        base = f"{MS.P}layers.{ids[0] - 1}.ple.ple_embedding."
        try:
            PI.check_constants(t, base, tm)
        except SystemExit as x:
            errs.append(str(x))
    if ple_dir:
        errs += check_ple(ckpt, ple_dir, t, sample, seed)
    return errs


def check_ple(ckpt: str, ple_dir: str, t: dict, sample: int, seed: int) -> list:
    errs = []
    pm = PI.tensor_map(ple_dir)
    sizes, offsets, _, _, mult = MS.ple_tables(t)
    for n, want in (("ple.layer_multipliers", mult), ("ple.ngram_heads_vocab_sizes", sizes),
                    ("ple.ngram_heads_offsets", offsets)):
        if n not in pm or [int(x) for x in PI.mmap_of(pm[n])] != list(want):
            errs.append(f"{ple_dir}: {n} is absent or differs from the formula")
    _, base, shards, tm = PI.ple_source(ckpt)
    rows_per = int(tm[shards[0]][1]["shape"][0])
    sdts = {pm[f"ple.h{h}.s"][1]["dtype"] for h in range(len(sizes)) if f"ple.h{h}.s" in pm}
    if len(sdts) != 1 or not sdts <= {"BF16", "F32"}:
        errs.append(f"{ple_dir}: scale dtypes {sorted(sdts)} - one of BF16 / F32 expected")
        return errs
    scale = "bf16" if sdts == {"BF16"} else "f32"
    rng = random.Random(seed)
    for h, (n, o) in enumerate(zip(sizes, offsets)):
        qe, se = pm.get(f"ple.h{h}.q"), pm.get(f"ple.h{h}.s")
        if qe is None or se is None or qe[1]["shape"] != [n, 160] or qe[1]["dtype"] != "I8" or se[1]["shape"] != [n]:
            errs.append(f"{ple_dir}: ple.h{h}.q / .s absent or not I8 [{n}][160] / [{n}]")
            continue
        q, sv = PI.mmap_of(qe), PI.mmap_of(se)
        rows = sorted({0, n - 1} | {rng.randrange(n) for _ in range(sample)})
        for r in rows:
            g = o + r
            src = tm[shards[g // rows_per]]
            w = PI.as_f32(PI.mmap_of(src)[g % rows_per: g % rows_per + 1], src[1]["dtype"])
            wq, ws = PI.quantise_rows(w, scale)
            got_s = torch.from_numpy(np.asarray(sv[r:r + 1]).copy())
            got_s = got_s.view(torch.bfloat16) if scale == "bf16" else got_s
            if not torch.equal(torch.from_numpy(np.asarray(q[r]).copy()), wq[0]) or not torch.equal(got_s, ws):
                errs.append(f"{ple_dir}: head {h} row {r} (global {g}, shard {g // rows_per} row {g % rows_per}) "
                            f"is not spec 9's quantisation of the source row")
                break
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--ple")
    ap.add_argument("--sample", type=int, default=64)
    a = ap.parse_args()
    errs = check(a.ckpt, a.ple, a.sample)
    if errs:
        for e in errs[:20]:
            print("REFUSED:", e)
        print(f"{len(errs)} violation(s)")
        sys.exit(1)
    print(f"ACCEPTED {a.ckpt}" + (f" with the PLE file {a.ple}" if a.ple else ""))


if __name__ == "__main__":
    main()
