#!/usr/bin/env python3
"""Spec 24a Task 2: tiny lfm2_moe checkpoints the downloaded tiny model cannot give (tests only).

    lfm2moe_make_tiny.py <out dir> --variant pattern8|pattern8-fp32|ties|int4-g64 [--seed 0]
                         [--tokenizer <LFM2.5 snapshot>] [--expert-names w|gate_up_down] [--twin <dir>]

Random weights on the real model's structure, written with the REAL checkpoint's tensor names (per expert
`model.layers.L.feed_forward.experts.E.{w1,w2,w3}.weight` as the original stores them, `expert_bias` F32,
`conv.conv.weight` [H][1][3], no `lm_head.weight` - the head is tied):
  pattern8       8 layers `conv, conv, full_attention, conv, conv, conv, full_attention, conv` (the real model's
                 first eight), num_dense_layers 2, hidden 256, 4 q / 1 kv heads of 64 (GQA 4, the real head_dim),
                 16 experts top-4, moe_intermediate_size 128, intermediate_size 384, vocab 128000 (the real
                 tokenizer files copied when --tokenizer names the LFM2.5 snapshot), expert_bias ~ N(0, 0.05)
                 fp32 (non-zero: the bias reorders), theta 1e7, max_position_embeddings 131072, bf16
  pattern8-fp32  the same weights in fp32
  ties           pattern8 with exact `sel` ties AT THE CUT: in layer 2 experts 3 and 11 get identical router
                 rows and biases, in layer 4 experts 5 and 6 - and in both layers three other experts get a bias
                 of +3 and the pair +1.5, the others keep |bias| < 0.5 (a bf16 sigmoid is in [0, 1]), so the pair
                 is always 4th / 5th: the 4th / 5th `sel` are bit-equal on every token
  int4-g64       pattern8 with every nn.Linear (experts, dense w1/w2/w3, conv in_proj / out_proj, attention
                 q/k/v/out_proj) packed RTN int4 g64 sym in the `auto_round:auto_gptq` bytes (qweight int32
                 [K/8][N], scales F16 [K/64][N], qzeros 0x77777777; src/loader/rtn.h:20-27's rule:
                 s16 = f16(2 amax / 15), q = clamp(rint(w / s16) + 8, 0, 15)) and quantization_config as
                 AutoRound 0.17.0 writes it. The router (`feed_forward.gate.weight`), `expert_bias`,
                 `conv.conv.weight`, the norms and the embedding stay as shipped.
--expert-names gate_up_down writes the int4 experts as `experts.E.{gate_proj,up_proj,down_proj}` (AutoRound's
unfused spelling, spec 24 §5); the default keeps `w1 / w3 / w2`. --twin DIR also writes the int4 variant's
weights DEQUANTISED (bf16(float(q - 8) x float(scale)), dequant.py's rule) as a bf16 checkpoint in the
original's names: what transformers' from_pretrained reads (it cannot load an auto-round export without the
auto-round package).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

VARIANTS = ("pattern8", "pattern8-fp32", "ties", "int4-g64")
PATTERN8 = ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv"]
TIES = {2: (3, 11), 4: (5, 6)}          # layer -> the planted pair
TIE_TOP3 = (0, 1, 2)                     # bias +3 in the tie layers: always ranks 1-3, so the pair is 4th / 5th
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
GROUP = 64
AUTOROUND_VERSION = "0.17.0"
EXPERT_NAMES = {"w": ("w1", "w3", "w2"), "gate_up_down": ("gate_proj", "up_proj", "down_proj")}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"lmt_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(variant: str) -> dict:
    """config.json of the variant: the real model's fields at tiny widths."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    return {
        "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "bos_token_id": 124894,
        "eos_token_id": 124900, "pad_token_id": 124893, "conv_L_cache": 3, "conv_bias": False,
        "dtype": "float32" if variant == "pattern8-fp32" else "bfloat16", "hidden_size": 256,
        "initializer_range": 0.02, "intermediate_size": 384, "layer_types": list(PATTERN8),
        "max_position_embeddings": 131072, "moe_intermediate_size": 128, "norm_eps": 1e-05, "norm_topk_prob": True,
        "num_attention_heads": 4, "num_dense_layers": 2, "num_experts": 16, "num_experts_per_tok": 4,
        "num_hidden_layers": 8, "num_key_value_heads": 1,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}, "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True, "use_cache": True, "use_expert_bias": True, "vocab_size": 128000}


def random_state(cfg: dict, seed: int, ties: bool = False) -> dict[str, torch.Tensor]:
    """fp32 random weights for every tensor of the checkpoint, by the REAL checkpoint's names (per-expert
    w1 / w3 / w2, expert_bias fp32)."""
    g = torch.Generator().manual_seed(seed)
    H, I, Ie, E = cfg["hidden_size"], cfg["intermediate_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh

    def lin(n, k):
        return torch.randn(n, k, generator=g) / math.sqrt(k)

    def norm(n):
        return 1.0 + 0.1 * torch.randn(n, generator=g)
    sd = {"model.embed_tokens.weight": torch.randn(cfg["vocab_size"], H, generator=g),
          "model.embedding_norm.weight": norm(H)}
    for i, t in enumerate(cfg["layer_types"]):
        b = f"model.layers.{i}."
        sd[b + "operator_norm.weight"] = norm(H)
        sd[b + "ffn_norm.weight"] = norm(H)
        if t == "conv":
            sd[b + "conv.in_proj.weight"] = lin(3 * H, H)
            sd[b + "conv.conv.weight"] = 0.5 * torch.randn(H, 1, cfg["conv_L_cache"], generator=g)
            sd[b + "conv.out_proj.weight"] = lin(H, H)
        else:
            sd[b + "self_attn.q_proj.weight"] = lin(nh * hd, H)
            sd[b + "self_attn.k_proj.weight"] = lin(nkv * hd, H)
            sd[b + "self_attn.v_proj.weight"] = lin(nkv * hd, H)
            sd[b + "self_attn.out_proj.weight"] = lin(H, nh * hd)
            sd[b + "self_attn.q_layernorm.weight"] = norm(hd)
            sd[b + "self_attn.k_layernorm.weight"] = norm(hd)
        f = b + "feed_forward."
        if i < cfg["num_dense_layers"]:
            sd[f + "w1.weight"], sd[f + "w3.weight"], sd[f + "w2.weight"] = lin(I, H), lin(I, H), lin(H, I)
            continue
        gate = 3.0 * torch.randn(E, H, generator=g) / math.sqrt(H)
        bias = 0.05 * torch.randn(E, generator=g)
        if ties and i in TIES:
            a, c = TIES[i]
            gate[c] = gate[a]
            bias = bias.clamp(-0.45, 0.45)
            bias[list(TIE_TOP3)] = 3.0
            bias[a] = bias[c] = 1.5
        sd[f + "gate.weight"] = gate
        sd[f + "expert_bias"] = bias
        for e in range(E):
            sd[f"{f}experts.{e}.w1.weight"] = lin(Ie, H)
            sd[f"{f}experts.{e}.w3.weight"] = lin(Ie, H)
            sd[f"{f}experts.{e}.w2.weight"] = lin(H, Ie)
    return sd


def rtn_int4(w: torch.Tensor, g: int = GROUP):
    """src/loader/rtn.h's rule on w [N][K] (its values as bf16): per output column n and group of g k,
    amax = max |w|, s16 = f16(2 amax / 15) (RNE), q = clamp(rint(w / f32(s16)) + 8, 0, 15) (q = 8 when
    s16 = 0). Returns (qweight int32 [K/8][N] - word (r, n) holds k = 8r..8r+7, 8r in the lowest nibble -,
    qzeros int32 [K/g][N/8] = 0x77777777, scales f16 [K/g][N])."""
    N, K = w.shape
    if K % g or K % 8 or N % 8:
        raise ValueError(f"rtn_int4: [{N}][{K}] at group {g}")
    wf = w.to(torch.bfloat16).float().reshape(N, K // g, g)
    amax = wf.abs().amax(-1)
    s16 = ((2.0 * amax) / 15.0).to(torch.float16)
    s = s16.float().unsqueeze(-1)
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))) + 8.0,
                    torch.full_like(wf, 8.0)).clamp(0, 15).to(torch.int64).reshape(N, K).t()      # [K][N]
    r = q.reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= r[:, i, :] << (4 * i)
    qweight = (packed - ((packed >> 31) << 32)).to(torch.int32)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int64)
    qzeros = (qz - ((qz >> 31) << 32)).to(torch.int32)
    return qweight, qzeros, s16.t().contiguous()


def dequant(qweight: torch.Tensor, scales: torch.Tensor, g: int = GROUP) -> torch.Tensor:
    """bf16(float(q - 8) x float(scale)) as the nn.Linear weight [N][K] (dequant.py's rule, transposed)."""
    return _load("dequant").dequant_gptq(qweight, scales, g).t().contiguous()


def is_linear(name: str) -> bool:
    """Every nn.Linear weight of the model (what our export quantises, spec 24 §5)."""
    return name.endswith((".w1.weight", ".w2.weight", ".w3.weight", ".in_proj.weight", ".out_proj.weight",
                          ".q_proj.weight", ".k_proj.weight", ".v_proj.weight"))


def expert_rename(name: str, names: str) -> str:
    """experts.E.w1/w3/w2 -> the spelling `names` picks (w | gate_up_down); other names unchanged."""
    if ".experts." not in name or names == "w":
        return name
    for src, dst in zip(EXPERT_NAMES["w"], EXPERT_NAMES[names]):
        if name.endswith(f".{src}.weight"):
            return name[: -len(f"{src}.weight")] + f"{dst}.weight"
    return name


def write(out: str, cfg: dict, sd: dict, dtype: torch.dtype, tokenizer: str | None) -> list[str]:
    os.makedirs(out, exist_ok=True)
    t = {k: (v.to(dtype) if v.is_floating_point() and not k.endswith("expert_bias") else v).contiguous()
         for k, v in sd.items()}
    save_file(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(t)}}, f, indent=1)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": cfg["bos_token_id"], "eos_token_id": [cfg["eos_token_id"]],
                   "pad_token_id": cfg["pad_token_id"]}, f, indent=1)
    if tokenizer:
        for fn in TOKENIZER_FILES:
            src = os.path.join(tokenizer, fn)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out, fn))
    return sorted(t)


def write_int4(out: str, cfg: dict, sd: dict, tokenizer: str | None, names: str = "w",
               twin: str | None = None) -> list[str]:
    """The int4 export of `sd` (bf16 values packed RTN g64) and, with `twin`, its dequantised bf16 twin."""
    os.makedirs(out, exist_ok=True)
    t, deq = {}, {}
    for k, v in sd.items():
        if is_linear(k):
            qw, qz, sc = rtn_int4(v)
            base = expert_rename(k, names)[: -len(".weight")]
            t[base + ".qweight"], t[base + ".qzeros"], t[base + ".scales"] = qw, qz, sc
            deq[k] = dequant(qw, sc)
        else:
            t[k] = v if k.endswith("expert_bias") else v.to(torch.bfloat16)
            deq[k] = v
    t = {k: v.contiguous() for k, v in t.items()}
    save_file(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(t)}}, f, indent=1)
    c = dict(cfg)
    c["quantization_config"] = {
        "autoround_version": AUTOROUND_VERSION, "bits": 4, "data_type": "int", "group_size": GROUP, "iters": 0,
        "packing_format": "auto_round:auto_gptq", "quant_method": "auto-round", "sym": True}
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, indent=1)
    with open(os.path.join(out, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": cfg["bos_token_id"], "eos_token_id": [cfg["eos_token_id"]],
                   "pad_token_id": cfg["pad_token_id"]}, f, indent=1)
    if tokenizer:
        for fn in TOKENIZER_FILES:
            src = os.path.join(tokenizer, fn)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out, fn))
    if twin:
        write(twin, cfg, deq, torch.bfloat16, tokenizer)
    return sorted(t)


def default_tokenizer() -> str | None:
    """The LFM2.5 snapshot's small files in the HF cache (the container's /hf or ~/.cache), when present."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for root in (hf, "/hf", os.path.expanduser("~/.cache/huggingface")):
        for c in sorted(glob.glob(os.path.join(root, "hub", "models--LiquidAI--LFM2.5-24B-A2B", "snapshots", "*"))):
            if os.path.exists(os.path.join(c, "tokenizer.json")):
                return c
    return None


def make(out: str, variant: str, seed: int = 0, tokenizer: str | None = None, names: str = "w",
         twin: str | None = None) -> dict:
    """Write the variant; returns its config dict."""
    cfg = config(variant)
    tokenizer = tokenizer if tokenizer is not None else default_tokenizer()
    sd = random_state(cfg, seed, ties=variant == "ties")
    if variant == "int4-g64":
        write_int4(out, cfg, sd, tokenizer, names, twin)
    else:
        write(out, cfg, sd, torch.float32 if variant == "pattern8-fp32" else torch.bfloat16, tokenizer)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", help="the LFM2.5 snapshot (its tokenizer.json, tokenizer_config.json, "
                                        "chat_template.jinja are copied); default: the HF cache's, when present")
    ap.add_argument("--expert-names", default="w", choices=sorted(EXPERT_NAMES))
    ap.add_argument("--twin", help="int4-g64: also write the dequantised bf16 twin here")
    a = ap.parse_args()
    if a.twin and a.variant != "int4-g64":
        ap.error("--twin is the int4-g64 variant's")
    make(a.out, a.variant, a.seed, a.tokenizer, a.expert_names, a.twin)
    print(f"wrote {a.out} ({a.variant}" + (f", experts {a.expert_names}" if a.variant == "int4-g64" else "") + ")"
          + (f" and its dequantised twin {a.twin}" if a.twin else ""))


if __name__ == "__main__":
    main()
