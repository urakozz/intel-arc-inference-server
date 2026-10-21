#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a Task 2: tiny Nemotron 3.5 Bolt (`nemotron_h_puzzle`) checkpoints for the tests (no public tiny exists).

    bolt_make_tiny.py <out dir> --variant NAME [--seed 0] [--snapshot <pinned snapshot>] [--signed-scales]
                      [--trunk model|backbone] [--twin <dir>]

Random weights on the real model's structure, written `save_pretrained`-shaped (model.safetensors + an index,
config.json, generation_config.json, the real tokenizer / template copied from the HF cache at run time) with the
CHECKPOINT's tensor names: `language_model.model.layers.<l>.{norm.weight, mixer.*}`, `language_model.model.embeddings.
weight`, `language_model.model.norm_f.weight`, `language_model.lm_head.weight` (SergiioB's spelling; `--trunk backbone`
writes NVIDIA's `language_model.backbone.*`). config.json is the omni wrapper's shape - `model_type nemotron_h_omni`
with the `llm_config` built from the cached real one at tiny widths (bolt_tiny_check.tiny_config, vocab 131072 so the
real tokenizer's ids are valid). `A_log` / `D` / `dt_bias` / `e_score_correction_bias` are F32 (the first three holding
bf16 values, as the checkpoint ships them); everything else bf16 (fp32 in `pattern12-fp32`).

  pattern12       blocks `M E M E M E M * E M E M` (the real model's first 12), hidden 256, Mamba 8 x 32, n_groups 2,
                  state 16, attention 16 / 1 x 16, MoE (E, I) cycling (16, 64) (20, 128) (24, 192), top-9, latent 64,
                  shared 128; `e_score_correction_bias` ~ 4.5 + N(0, 0.02) (the checkpoint's scale), `dt_bias` in
                  [-10.5, 8.5], `A_log` in [-4.7, 4.4] (the checkpoint's ranges), conv bias non-zero, norms 1 + N(0, 0.1)
  pattern12-fp32  the same in fp32, except `dt_bias` in [-2.5, 8.5]: every head's softplus(dt) stays above R's fallback
                  clamp (0.001) on the tests' prompts, so the fallback is the same function as the port
  ties            pattern12 with exact `sel` ties AT THE CUT: in layer 3 experts 4 and 13, in layer 8 experts 2 and 9,
                  identical router rows and biases; eight other experts of the layer +3 on the bias and the pair +1.5,
                  so the pair is always 9th / 10th
  dtclamp         pattern12 with half the heads of every Mamba layer at `dt_bias` in [-10.5, -7.5] (softplus < 0.001
                  near dt_raw 0): where R's fallback clamp changes the decay
  int4-g64        pattern12 with every linear (routed experts' up / down, shared up / down, fc1 / fc2 latent, Mamba
                  in_proj / out_proj, attention q / k / v / o) RTN int4 g64 sym in the `auto_round:auto_gptq` bytes
                  (qweight I32 [K/8][N], scales F16 [K/64][N], qzeros 0x77777777; src/loader/rtn.h's rule
                  s16 = f16(2 amax / 15)); `--signed-scales` writes AutoRound's signed form s16 = f16(-e / 8), e the
                  group's largest-magnitude value (it maps to q = 0); router, bias, conv, norms, embedding, head kept
  int4-g128-gptq  SergiioB's spelling on `pattern12-k128` (latent 128, I cycling 128 / 256 / 384: every K a multiple
                  of 128): quant_method gptq, group 128, `g_idx` = i // 128, qzeros 0x77777777, F16 SIGNED scales, the
                  cached checkpoint's quantization_config verbatim (the `dynamic` object with its restating `+:`
                  out_proj rule)
--twin DIR (int4 variants) also writes the dequantised weights (bf16(float(q - 8) x float(scale))) as a bf16
checkpoint in the plain names: the same model as plain tensors.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

VARIANTS = ("pattern12", "pattern12-fp32", "ties", "dtclamp", "int4-g64", "int4-g128-gptq")
TIES = {3: (4, 13), 8: (2, 9)}                  # layer -> the planted pair
TIE_TOP8 = (0, 1, 3, 5, 6, 7, 8, 10)            # +3 in a tie layer (the pair is never among them)
SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
AUTOROUND_VERSION = "0.17.0"
F32_NAMES = (".A_log", ".D", ".dt_bias", ".e_score_correction_bias")
INT4 = (".qweight", ".qzeros", ".scales")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("bolt_remote")
tc = _load("bolt_tiny_check")


def tiny_llm_config(snapshot: str, variant: str) -> dict:
    d = tc.tiny_config(snapshot, vocab=131072)
    d["dtype"] = "float32" if variant == "pattern12-fp32" else "bfloat16"
    if variant == "int4-g128-gptq":
        e = 0
        for b in d["block_configs"]:
            if b["block_type"] == "moe":
                b["moe_intermediate_size"] = (128, 256, 384)[e % 3]
                e += 1
        d["moe_latent_size"] = 128
    return d


def is_linear(name: str) -> bool:
    """The linears every int4 form packs (spec 25 §1.3): not the router, the conv, norms, embedding or head."""
    return name.endswith((".up_proj.weight", ".down_proj.weight", ".fc1_latent_proj.weight", ".fc2_latent_proj.weight",
                          ".in_proj.weight", ".out_proj.weight", ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
                          ".o_proj.weight")) and ".layers." in name


def shapes(llm: dict) -> dict[str, tuple]:
    """R's state-dict names and shapes for `llm` (the model built on meta)."""
    ns = br._require()
    cfg = br.make_config(llm)
    with torch.device("meta"):
        m = ns.LM(cfg)
    out = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    for k, v in m.named_buffers():
        out[k] = tuple(v.shape)
    return out


def random_state(llm: dict, seed: int, variant: str) -> dict[str, torch.Tensor]:
    """fp32 tensors holding bf16 values (except the router bias: fp32) for every name of R's state dict."""
    g = torch.Generator().manual_seed(seed)

    def bf(t):
        return t.to(torch.bfloat16).float()

    def uni(n, lo, hi):
        """uniform in [lo, hi] with both ends present (the checkpoint's range is reached)"""
        t = lo + (hi - lo) * torch.rand(n, generator=g)
        if n >= 2:
            t[0], t[-1] = lo, hi
        return t
    sd = {}
    for name, shp in sorted(shapes(llm).items()):
        if name.endswith("e_score_correction_bias"):
            sd[name] = 4.5 + 0.02 * torch.randn(shp, generator=g)
            continue
        if name.endswith(".dt_bias"):
            t = uni(shp[0], -2.5, 8.5) if variant == "pattern12-fp32" else uni(shp[0], -10.5, 8.5)
            if variant == "dtclamp":
                half = torch.randperm(shp[0], generator=g)[: shp[0] // 2]
                t[half] = uni(len(half), -10.5, -7.5)
        elif name.endswith(".A_log"):
            t = uni(shp[0], -4.7, 4.4)
        elif name.endswith(".D"):
            t = 1.0 + 0.3 * torch.randn(shp, generator=g)
        elif name.endswith(".conv1d.weight"):
            t = 0.5 * torch.randn(shp, generator=g)
        elif name.endswith(".conv1d.bias"):
            t = 0.2 * torch.randn(shp, generator=g)
        elif name.endswith("norm.weight") or name.endswith("norm_f.weight"):
            t = 1.0 + 0.1 * torch.randn(shp, generator=g)
        elif name.endswith("embeddings.weight"):
            t = torch.randn(shp, generator=g)
        elif name.endswith(".gate.weight"):
            t = 2.0 * torch.randn(shp, generator=g) / math.sqrt(shp[1])
        elif len(shp) == 2:
            t = torch.randn(shp, generator=g) / math.sqrt(shp[1])
        else:
            raise RuntimeError(f"no rule for {name} {shp}")
        sd[name] = bf(t)
    if variant == "ties":
        for layer, (a, c) in TIES.items():
            p = f"model.layers.{layer}.mixer.gate."
            if p + "weight" not in sd:
                raise RuntimeError(f"layer {layer} is not a MoE layer")
            sd[p + "weight"][c] = sd[p + "weight"][a]
            b = sd[p + "e_score_correction_bias"]
            b[list(TIE_TOP8)] += 3.0
            b[a] = b[c] = 4.5 + 1.5
    return sd


# ------------------------------------------------------------------------------------------------
# int4

def pack_int4(w: torch.Tensor, g: int, signed: bool):
    """w [N][K] (bf16 values) -> (qweight I32 [K/8][N], qzeros I32 [K/g][N/8] 0x77777777, scales F16 [K/g][N]).
    unsigned (rtn.h): s16 = f16(2 amax / 15); signed (AutoRound): s16 = f16(-e / 8), e the group's value of largest
    magnitude (the first such on a tie). q = clamp(rint(w / f32(s16)) + 8, 0, 15); q = 8 where s16 = 0."""
    N, K = w.shape
    if K % g or K % 8 or N % 8:
        raise ValueError(f"pack_int4: [{N}][{K}] at group {g}")
    wf = w.to(torch.bfloat16).float().reshape(N, K // g, g)
    if signed:
        idx = wf.abs().argmax(-1, keepdim=True)
        s16 = (-torch.gather(wf, -1, idx)[..., 0] / 8.0).to(torch.float16)
    else:
        s16 = ((2.0 * wf.abs().amax(-1)) / 15.0).to(torch.float16)
    s = s16.float().unsqueeze(-1)
    nz = s != 0
    q = torch.where(nz, torch.round(wf / torch.where(nz, s, torch.ones_like(s))) + 8.0, torch.full_like(wf, 8.0))
    q = q.clamp(0, 15).to(torch.int64).reshape(N, K).t()                                        # [K][N]
    r = q.reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= r[:, i, :] << (4 * i)
    qweight = (packed - ((packed >> 31) << 32)).to(torch.int32)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int64)
    return qweight, (qz - ((qz >> 31) << 32)).to(torch.int32), s16.t().contiguous()


def dequant(qweight: torch.Tensor, scales: torch.Tensor, g: int) -> torch.Tensor:
    """bf16(float(q - 8) x float(scale)) as the nn.Linear weight [N][K] (dequant.py's rule, transposed)."""
    return _load("dequant").dequant_gptq(qweight, scales, g).t().contiguous()


# ------------------------------------------------------------------------------------------------
# writing

def ckpt_name(r_name: str, trunk: str) -> str:
    """R's state-dict name -> the checkpoint's (`language_model.` + R's, `model.` -> `backbone.` for --trunk backbone)."""
    if trunk == "backbone" and r_name.startswith("model."):
        r_name = "backbone." + r_name[len("model."):]
    return "language_model." + r_name


def omni_config(snapshot: str, llm: dict, quant: dict | None) -> dict:
    from transformers.configuration_utils import PreTrainedConfig
    c = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni", "llm_config": llm,
         "tie_word_embeddings": False, "eos_token_id": 11, "bolt_tiny": True}
    if quant is not None:
        c["quantization_config"] = quant
    return PreTrainedConfig._encode_special_floats(c)


def _write_dir(out: str, snapshot: str, cfg: dict, t: dict) -> list[str]:
    os.makedirs(out, exist_ok=True)
    t = {k: v.contiguous() for k, v in sorted(t.items())}
    save_file(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in t}}, f, indent=1)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": [2, 11], "pad_token_id": 0}, f, indent=1)
    for fn in SMALL_FILES:
        src = os.path.join(snapshot, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, fn))
    return list(t)


def plain_tensors(sd: dict, dtype: torch.dtype, trunk: str) -> dict:
    return {ckpt_name(k, trunk): (v.float() if k.endswith(F32_NAMES) else v.to(dtype)) for k, v in sd.items()}


def make(out: str, variant: str, seed: int = 0, snapshot: str | None = None, signed: bool = False,
         trunk: str = "model", twin: str | None = None) -> dict:
    """Write the variant; returns its llm_config dict."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    snapshot = snapshot or br.find_snapshot()
    if snapshot is None:
        raise RuntimeError("the pinned Bolt snapshot is not in the HF cache (its config.json and remote code are read)")
    br.load_remote(snapshot)
    llm = tiny_llm_config(snapshot, variant)
    sd = random_state(llm, seed, variant)
    if not variant.startswith("int4"):
        dtype = torch.float32 if variant == "pattern12-fp32" else torch.bfloat16
        _write_dir(out, snapshot, omni_config(snapshot, llm, None), plain_tensors(sd, dtype, trunk))
        return llm
    g, gptq = (128, True) if variant == "int4-g128-gptq" else (64, False)
    signed = signed or gptq
    t, deq = {}, {}
    for k, v in sd.items():
        if is_linear(k):
            qw, qz, sc = pack_int4(v, g, signed)
            base = ckpt_name(k, trunk)[: -len(".weight")]
            t[base + ".qweight"], t[base + ".qzeros"], t[base + ".scales"] = qw, qz, sc
            if gptq:
                t[base + ".g_idx"] = (torch.arange(v.shape[1], dtype=torch.int32) // g).contiguous()
            deq[k] = dequant(qw, sc, g)
        else:
            t[ckpt_name(k, trunk)] = v.float() if k.endswith(F32_NAMES) else v.to(torch.bfloat16)
            deq[k] = v
    if gptq:
        with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
            quant = json.load(f)["quantization_config"]
    else:
        quant = {"autoround_version": AUTOROUND_VERSION, "bits": 4, "data_type": "int", "group_size": g, "iters": 0,
                 "packing_format": "auto_round:auto_gptq", "quant_method": "auto-round", "sym": True}
    _write_dir(out, snapshot, omni_config(snapshot, llm, quant), t)
    if twin:
        _write_dir(twin, snapshot, omni_config(snapshot, llm, None), plain_tensors(deq, torch.bfloat16, trunk))
    return llm


def read_state(d: str) -> tuple[dict, dict]:
    """(llm_config, R-named state dict) of a variant directory: int4 linears dequantised (dequant.py's rule), the
    trunk spelling mapped back, F32 tensors as stored. The tests' plain loader (bolt_ref streams instead)."""
    from safetensors import safe_open
    from transformers.configuration_utils import PreTrainedConfig
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        llm = PreTrainedConfig._decode_special_floats(json.load(f))["llm_config"]
    with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as h:
        raw = {k: h.get_tensor(k) for k in h.keys()}
    sd = {}
    for k, v in raw.items():
        if k.endswith((".qzeros", ".scales", ".g_idx")):
            continue
        r = k[len("language_model."):]
        if r.startswith("backbone."):
            r = "model." + r[len("backbone."):]
        if k.endswith(".qweight"):
            sc = raw[k[: -len(".qweight")] + ".scales"]
            sd[r[: -len(".qweight")] + ".weight"] = dequant(v, sc, v.shape[0] * 8 // sc.shape[0])
        else:
            sd[r] = v
    return llm, sd


def load_model(d: str, dtype: torch.dtype, path: str = "kernel"):
    """R's model of a variant directory, un-streamed: build_lm + a strict load (the router bias kept fp32)."""
    llm, sd = read_state(d)
    model = br.build_lm(llm, dtype, path=path)
    sd = {k: (v.float() if k.endswith("e_score_correction_bias") else v.to(dtype)) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", help="the pinned Bolt snapshot (default: the HF cache's)")
    ap.add_argument("--signed-scales", action="store_true", help="int4-g64: AutoRound's signed scales")
    ap.add_argument("--trunk", default="model", choices=["model", "backbone"])
    ap.add_argument("--twin", help="int4 variants: also write the dequantised bf16 twin here")
    a = ap.parse_args()
    if a.twin and not a.variant.startswith("int4"):
        ap.error("--twin is the int4 variants'")
    make(a.out, a.variant, a.seed, a.snapshot, a.signed_scales, a.trunk, a.twin)
    print(f"wrote {a.out} ({a.variant}, trunk {a.trunk})" + (f" and its dequantised twin {a.twin}" if a.twin else ""))


if __name__ == "__main__":
    main()
