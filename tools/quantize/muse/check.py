#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would the engine accept this Muse-Glimmer int4 checkpoint? (spec 23b Task 4)

    check.py <checkpoint> [--sample N] [--seed S]

src/loader/muse_repack.cc's rules in Python (loader::MuseCheckpoint over QuantConfig::parse, LinearSrc::classify and
assert_quant_invariants), against the reference's own names:

  quantization_config  bits 4, group_size 64, sym true, desc_act false or absent; quant_method "gptq" | "auto-round";
                       packing_format "auto_round:auto_gptq" when present
  names                exactly muse_ref.expected_names(text_config(<checkpoint>), form) - form "gptq" when any text
                       `.g_idx` is shipped, else "autoround" (either Vishva007 spelling) - plus vision tensors
                       (`model.vision_{tower,adapter,projection}.*`, skipped by the engine, counted); nothing else
  int4 linears         qweight I32 [K/8, N], scales F16 [K/64, N] finite and > 0, qzeros I32 [K/64, N/8] all
                       0x77777777, g_idx I32 [K] = i // 64 when shipped
  bf16 tensors         norms, embed_tokens, lm_head, the final norm: BF16 at the text config's shapes
  the twins            a make_synth.py checkpoint (muse_synth.json beside it): every sampled int4 linear (--sample, 0 =
                       all) dequantises ((q - 8) x scale) to within RTN's bound of the bf16 twin make_synth regenerates
                       from the recipe: |w - deq| <= scale / 2 + |w| 2^-8 (the f16 scale's own rounding at the clamp)

Prints ACCEPTED (exit 0) or every violation, the first one first (exit 1). Needs transformers 5.19.0 (muse_ref's
text_config): the oracle image with spec 21a's site first on PYTHONPATH.
"""
import argparse
import json
import os
import random
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_synth as MS  # noqa: E402

QZ = 0x77777777
VISION = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")
_ref = None


def ref():
    global _ref
    if _ref is None:
        _ref = MS._load(os.path.join(MS.ORACLE, "muse_ref.py"), "muse_check_ref")
    return _ref


def headers(d: str) -> dict:
    """name -> (file, dtype, shape) from the safetensors headers (no tensor read)."""
    out = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
        with open(os.path.join(d, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            if k in out:
                raise SystemExit(f"{k} appears in two shards")
            out[k] = (fn, v["dtype"], tuple(v["shape"]))
    return out


def parse_quant_config(cfg: dict, errs: list) -> None:
    q = cfg.get("quantization_config")
    if not q:
        errs.append("config.json has no quantization_config")
        return
    if q.get("bits") != 4 or q.get("group_size") != 64 or q.get("sym") is not True:
        errs.append(f"quantization_config: need bits 4, group_size 64, sym true; got bits {q.get('bits')}, "
                    f"group_size {q.get('group_size')}, sym {q.get('sym')}")
    if q.get("desc_act") not in (None, False):
        errs.append(f"quantization_config.desc_act {q.get('desc_act')}: activation order is not supported")
    if q.get("quant_method") not in ("gptq", "auto-round"):
        errs.append(f"quantization_config.quant_method {q.get('quant_method')!r} is neither 'gptq' nor 'auto-round'")
    if q.get("packing_format") not in (None, "auto_round:auto_gptq"):
        errs.append(f"quantization_config.packing_format {q.get('packing_format')!r} is not 'auto_round:auto_gptq'")


class Reader:
    def __init__(self, d: str, hdr: dict):
        from safetensors import safe_open
        self.h = {fn: safe_open(os.path.join(d, fn), framework="pt") for fn in sorted({v[0] for v in hdr.values()})}
        self.hdr = hdr

    def get(self, name: str) -> torch.Tensor:
        return self.h[self.hdr[name][0]].get_tensor(name)


def check(ckpt: str, sample: int = 0, seed: int = 0) -> tuple[list, dict]:
    errs, info = [], {}
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    parse_quant_config(cfg, errs)
    tc = ref().text_config(ckpt, raw=cfg)
    t = cfg.get("text_config", cfg)
    hdr = headers(ckpt)
    form = "gptq" if any(k.endswith(".g_idx") and ref().is_text(k) for k in hdr) else "autoround"
    want = set(ref().expected_names(tc, form))
    vision = [k for k in hdr if k.startswith(VISION)]
    text = [k for k in hdr if not k.startswith(VISION)]
    info.update(form=form, text=len(text), vision=len(vision), layers=tc.num_hidden_layers)
    for k in sorted(want - set(text)):
        errs.append(f"missing tensor {k} ({form} spelling)")
    for k in sorted(set(text) - want):
        errs.append(f"unexpected tensor {k} (not muse_ref.expected_names(tc, {form!r}), not vision)")
    rd = Reader(ckpt, hdr)
    H, V = t["hidden_size"], t["vocab_size"]
    bf16_shapes = {MS.LM + "embed_tokens.weight": (V, H), "lm_head.weight": (V, H), MS.LM + "norm.weight": (H,)}
    lin = MS.linear_shapes(t)
    for i in range(tc.num_hidden_layers):
        p = f"{MS.LM}layers.{i}."
        for n in MS.NORMS:
            bf16_shapes[p + n + ".weight"] = (H,)
    for name, shape in bf16_shapes.items():
        if name in hdr and (hdr[name][1] != "BF16" or hdr[name][2] != shape):
            errs.append(f"{name}: {hdr[name][1]} {list(hdr[name][2])}, expected BF16 {list(shape)}")
    bases = [(f"{MS.LM}layers.{i}.{ln}", s) for i in range(tc.num_hidden_layers) for ln, s in lin.items()]
    recipe = None
    if os.path.exists(os.path.join(ckpt, MS.RECIPE)):
        with open(os.path.join(ckpt, MS.RECIPE), encoding="utf-8") as f:
            recipe = json.load(f)
    twins = bases if sample <= 0 else random.Random(seed).sample(bases, min(sample, len(bases)))
    twins = set(b for b, _ in twins)
    worst = 0.0
    for base, (N, K) in bases:
        if base + ".qweight" not in hdr or base + ".scales" not in hdr:
            continue
        spec = {".qweight": ("I32", (K // 8, N)), ".scales": ("F16", (K // 64, N)), ".qzeros": ("I32", (K // 64, N // 8)),
                ".g_idx": ("I32", (K,))}
        bad = False
        for sfx, (dt, sh) in spec.items():
            if base + sfx in hdr and (hdr[base + sfx][1] != dt or hdr[base + sfx][2] != sh):
                errs.append(f"{base}{sfx}: {hdr[base + sfx][1]} {list(hdr[base + sfx][2])}, expected {dt} {list(sh)}")
                bad = True
        if bad:
            continue
        if base + ".qzeros" in hdr:
            qz = rd.get(base + ".qzeros").to(torch.int64) & 0xFFFFFFFF
            off = (qz != QZ).nonzero()
            if len(off):
                r, c = off[0].tolist()
                errs.append(f"{base}.qzeros[{r}][{c}] = {int(qz[r, c]):#010x}, expected 0x77777777 (symmetric int4)")
        if base + ".g_idx" in hdr:
            g = rd.get(base + ".g_idx").to(torch.int64)
            off = (g != torch.arange(K, dtype=torch.int64) // 64).nonzero()
            if len(off):
                j = int(off[0])
                errs.append(f"{base}.g_idx[{j}] = {int(g[j])}, expected {j // 64} (no activation order)")
        sc = rd.get(base + ".scales")
        if not bool(torch.isfinite(sc.float()).all()) or not bool((sc.float() > 0).all()):
            errs.append(f"{base}.scales: a non-finite or non-positive f16 scale")
        if recipe is not None and base in twins:
            w = MS.source(base + ".weight", (N, K), recipe["seed"]).float().t()      # [K][N]
            qw = rd.get(base + ".qweight")
            deq = _deq(qw, sc)
            bound = sc.float().repeat_interleave(64, dim=0) / 2 + w.abs() * 2 ** -8
            over = (w - deq).abs() > bound
            if bool(over.any()):
                k, n = over.nonzero()[0].tolist()
                errs.append(f"{base}: weight [{n}][{k}] dequantises to {float(deq[k, n]):.6g}, its make_synth twin is "
                            f"{float(w[k, n]):.6g} - beyond RTN's bound {float(bound[k, n]):.3g}")
            worst = max(worst, float(((w - deq).abs() / sc.float().repeat_interleave(64, dim=0)).max()))
    info["twins"] = len(twins) if recipe is not None else 0
    info["worst_err_in_scales"] = worst
    return errs, info


def _deq(qw: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """[K/8, N] I32 + [K/64, N] F16 -> [K, N] fp32: (q - 8) x scale, nibble j of word r is k = 8r + j."""
    rows, n = qw.shape
    shifts = torch.arange(8, dtype=torch.int32) * 4
    q = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(rows * 8, n).float()
    return (q - 8.0) * sc.float().repeat_interleave(64, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--sample", type=int, default=0, help="int4 linears checked against their twins (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    errs, info = check(a.checkpoint, a.sample, a.seed)
    print(json.dumps(info))
    if errs:
        for e in errs:
            print("REFUSED: " + e)
        sys.exit(1)
    print(f"ACCEPTED: {a.checkpoint} - {info['layers']} layers, the {info['form']} spelling, {info['text']} text tensors "
          f"(= muse_ref.expected_names), {info['vision']} vision skipped, {info['twins']} linears within RTN's bound of "
          f"their twins (worst {info['worst_err_in_scales']:.3f} scales)")


if __name__ == "__main__":
    main()
