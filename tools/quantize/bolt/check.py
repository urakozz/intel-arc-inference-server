#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would the engine accept this Nemotron 3.5 Bolt int4 checkpoint? (spec 25b Task 1; 25q merges its own check with this.)

    check.py <dir> [--expect gptq-g128|ar-g64]

src/loader/quant.cc's rules (QuantConfig::parse, LinearSrc::classify, assert_quant_invariants, with spec 25 §5.2's
relaxations) and spec 25 §1.3 / §5 for Bolt:

  quantization_config  bits 4, sym true, desc_act false or absent, group_size 64 or 128, quant_method gptq | auto-round,
                       packing_format absent | auto_round:auto_gptq; `dynamic` holds `-:` exclusions and `+:` rules that
                       RESTATE the top level (exactly {bits, group_size, sym}, each equal to it) - nothing else; no
                       extra_config
  names                exactly bolt_ref.expected_names(llm_config, form) for the language model (every linear
                       .qweight / .qzeros / .scales [+ .g_idx in the g128 GPTQ form]), plus tensors under the vision
                       prefixes (vision_model. / vision_projector. / mlp1.), which are only counted
  int4 linears         qweight I32 [K/8][N], scales F16 [K/g][N] (finite, non-zero; NEGATIVE allowed - AutoRound's signed
                       scale - and counted), qzeros I32 [K/g][N/8] = 0x77777777, g_idx I32 [K] = i // g (g128 form; absent
                       in the g64 form), K x N the descriptor's shapes
  kept tensors         router gate.weight BF16 [E][4096], e_score_correction_bias F32 [E], A_log / D / dt_bias F32 [128],
                       conv1d.weight BF16 [10240][1][4], conv1d.bias BF16 [10240], mixer.norm.weight BF16 [8192], every
                       block norm.weight BF16 [4096], embeddings / lm_head BF16 [131072][4096], norm_f BF16 [4096]
  config               model_type nemotron_h_omni over llm_config.model_type nemotron_h_puzzle; block_configs and
                       layers_block_type agree

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
VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")


def headers(d: str) -> dict:
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


def parse_quant(cfg: dict, errs: list) -> dict:
    q = cfg.get("quantization_config")
    if not q:
        errs.append("config.json has no quantization_config")
        return {}
    if q.get("bits") != 4 or q.get("sym") is not True or q.get("group_size") not in (64, 128):
        errs.append(f"quantization_config: need bits 4, sym true, group_size 64 or 128; got bits {q.get('bits')}, "
                    f"sym {q.get('sym')}, group_size {q.get('group_size')}")
    if q.get("desc_act") not in (None, False):
        errs.append(f"quantization_config.desc_act {q.get('desc_act')}: activation order is not supported")
    if q.get("quant_method") not in (None, "gptq", "auto-round"):
        errs.append(f"quant_method {q.get('quant_method')!r} is neither 'gptq' nor 'auto-round'")
    if q.get("packing_format") not in (None, "auto_round:auto_gptq"):
        errs.append(f"packing_format {q.get('packing_format')!r} is not 'auto_round:auto_gptq'")
    if q.get("extra_config"):
        errs.append("quantization_config.extra_config is set (no excluded nn.Linear exists in Bolt)")
    for rule, val in (q.get("dynamic") or {}).items():
        if rule.startswith("-:"):
            continue
        if not rule.startswith("+:"):
            errs.append(f"dynamic rule {rule!r}: neither '-:' nor '+:'")
        elif not isinstance(val, dict) or set(val) != {"bits", "group_size", "sym"} or any(
                val[k] != q.get(k) for k in ("bits", "group_size", "sym")):
            errs.append(f"dynamic rule {rule!r} = {val}: a '+:' rule must restate the top level exactly "
                        f"(the broken-MTP-config guard)")
    return q


def linear_shapes(llm: dict, block: dict) -> dict:
    """suffix -> (K, N) of every int4 linear of one block (R's names, mixer-relative)."""
    H = llm["hidden_size"]
    t = block["block_type"]
    if t == "mamba":
        nh, hd, ng, st = llm["mamba_num_heads"], llm["mamba_head_dim"], llm["n_groups"], llm["ssm_state_size"]
        d_inner = nh * hd
        return {"in_proj": (H, d_inner + d_inner + 2 * ng * st + nh), "out_proj": (d_inner, H)}
    if t == "attention":
        qn, kvn = llm["num_attention_heads"] * llm["head_dim"], llm["num_key_value_heads"] * llm["head_dim"]
        return {"q_proj": (H, qn), "k_proj": (H, kvn), "v_proj": (H, kvn), "o_proj": (qn, H)}
    L, S, I, E = (llm["moe_latent_size"], llm["moe_shared_expert_intermediate_size"], block["moe_intermediate_size"],
                  block["n_routed_experts"])
    out = {"fc1_latent_proj": (H, L), "fc2_latent_proj": (L, H), "shared_experts.up_proj": (H, S),
           "shared_experts.down_proj": (S, H)}
    for e in range(E):
        out[f"experts.{e}.up_proj"] = (L, I)
        out[f"experts.{e}.down_proj"] = (I, L)
    return out


def kept_shapes(llm: dict, block: dict) -> dict:
    """suffix (mixer-relative) -> (dtype, shape) of a block's kept tensors."""
    t = block["block_type"]
    nh, hd, ng, st = llm["mamba_num_heads"], llm["mamba_head_dim"], llm["n_groups"], llm["ssm_state_size"]
    if t == "mamba":
        cd = nh * hd + 2 * ng * st
        return {"A_log": ("F32", (nh,)), "D": ("F32", (nh,)), "dt_bias": ("F32", (nh,)),
                "conv1d.weight": ("BF16", (cd, 1, llm["conv_kernel"])), "conv1d.bias": ("BF16", (cd,)),
                "norm.weight": ("BF16", (nh * hd,))}
    if t == "moe":
        E = block["n_routed_experts"]
        return {"gate.weight": ("BF16", (E, llm["hidden_size"])), "gate.e_score_correction_bias": ("F32", (E,))}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--expect", choices=("gptq-g128", "ar-g64"), help="refuse the other form")
    a = ap.parse_args()
    errs: list = []
    with open(os.path.join(a.dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "nemotron_h_omni":
        errs.append(f"model_type {cfg.get('model_type')!r}, expected nemotron_h_omni")
    llm = cfg.get("llm_config", {})
    if llm.get("model_type") != "nemotron_h_puzzle":
        errs.append(f"llm_config.model_type {llm.get('model_type')!r}, expected nemotron_h_puzzle")
    blocks = llm.get("block_configs", [])
    if llm.get("layers_block_type") not in (None, [b["block_type"] for b in blocks]):
        errs.append("layers_block_type disagrees with block_configs")
    q = parse_quant(cfg, errs)
    g = q.get("group_size", 0)
    form = "gptq-g128" if g == 128 else "ar-g64"
    if a.expect and a.expect != form:
        errs.append(f"expected {a.expect}, the config says group {g}")
    hdr = headers(a.dir)
    sys.path.append(_ORACLE)
    import bolt_ref  # noqa: E402  (names are 25a's, not a second copy)
    names_form = form
    want = set(bolt_ref.expected_names(llm, names_form))
    if g == 128 and not any(k.endswith(".g_idx") for k in hdr):
        errs.append("a g128 checkpoint without any g_idx (the GPTQ form ships it, spec 25 §5.1)")
    got = {k for k in hdr if not k.startswith(VISION_PREFIXES)}
    vision = len(hdr) - len(got)
    for n in sorted(want - got)[:10]:
        errs.append(f"missing tensor {n}")
    for n in sorted(got - want)[:10]:
        errs.append(f"unexpected tensor {n}")

    # dtypes and shapes
    def expect(name, dtype, shape):
        if name not in hdr:
            return
        _, dt, sh = hdr[name]
        if dt != dtype or sh != tuple(shape):
            errs.append(f"{name}: {dt} {list(sh)}, expected {dtype} {list(shape)}")
    V, H = llm["vocab_size"], llm["hidden_size"]
    p = "language_model.model."
    expect(p + "embeddings.weight", "BF16", (V, H))
    expect("language_model.lm_head.weight", "BF16", (V, H))
    expect(p + "norm_f.weight", "BF16", (H,))
    n_lin = 0
    zero_files: dict = {}
    for i, b in enumerate(blocks):
        lp = f"{p}layers.{i}."
        expect(lp + "norm.weight", "BF16", (H,))
        for suf, (dt, sh) in kept_shapes(llm, b).items():
            expect(lp + "mixer." + suf, dt, sh)
        for suf, (K, N) in linear_shapes(llm, b).items():
            base = lp + "mixer." + suf
            n_lin += 1
            if K % g or K % 8 or N % 8:
                errs.append(f"{base}: K {K} N {N} at group {g}")
                continue
            expect(base + ".qweight", "I32", (K // 8, N))
            expect(base + ".scales", "F16", (K // g, N))
            expect(base + ".qzeros", "I32", (K // g, N // 8))
            if g == 128:
                expect(base + ".g_idx", "I32", (K,))
            for s in (".qweight", ".scales", ".qzeros", ".g_idx"):
                if base + s in hdr:
                    zero_files.setdefault(hdr[base + s][0], []).append((base, s, K, N))

    # values: every qzeros, every g_idx, every scale
    from safetensors import safe_open
    neg = tot = sub = 0
    for fn, items in sorted(zero_files.items()):
        with safe_open(os.path.join(a.dir, fn), framework="pt") as h:
            for base, s, K, N in items:
                if s == ".qzeros":
                    t = h.get_tensor(base + s)
                    if not bool((t == 0x77777777).all()):
                        errs.append(f"{base}.qzeros is not 0x77777777 (symmetric zero point 8)")
                elif s == ".g_idx":
                    t = h.get_tensor(base + s)
                    if not torch.equal(t, torch.arange(K, dtype=torch.int32) // g):
                        errs.append(f"{base}.g_idx is not i // {g}")
                elif s == ".scales":
                    t = h.get_tensor(base + s)
                    f = t.float()
                    if not bool(torch.isfinite(f).all()):
                        errs.append(f"{base}.scales has NaN / Inf")
                    if bool((f == 0).any()):
                        errs.append(f"{base}.scales has a zero scale")
                    neg += int((f < 0).sum())
                    tot += f.numel()
                    sub += int(((t.view(torch.int16) & 0x7C00) == 0).logical_and((t.view(torch.int16) & 0x3FF) != 0).sum())
        if len(errs) > 40:
            break
    for e in errs[:40]:
        print("REFUSED:", e)
    if errs:
        print(f"check: {len(errs)} violation(s) in {a.dir}")
        return 1
    print(f"check: {a.dir}: {len(blocks)} blocks, {n_lin} int4 linears (group {g}, form {form}), {len(got)} language "
          f"tensors + {vision} vision / projector tensors (skipped by prefix); scales {tot}: {neg} negative "
          f"({100.0 * neg / max(tot, 1):.1f} %), {sub} subnormal")
    print("ACCEPTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
