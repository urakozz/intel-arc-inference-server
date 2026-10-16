#!/usr/bin/env python3
"""Spec 23a: tiny Muse-Glimmer checkpoints the downloaded tiny model cannot give (tests only).

    muse_make_tiny.py <tiny snapshot> <out dir> --variant untied|l8|gqa16|win16|int4|assist|assist16
                      [--original <bf16 snapshot>] [--gptq <gptq snapshot>] [--seed 0] [--twin <dir>]

The base is `optimum-intel-internal-testing/tiny-random-muse-glimmer` (4 layers [s, s, s, f], hidden 64, 4 q / 2 kv
heads x 16, FF 256, vocab 202048, bf16, TIED - no lm_head tensor). Every variant is bf16, seeded, keeps the tiny's
vision tensors and config (so from_pretrained loads it cleanly), is UNTIED (an independent random lm_head;
`tie_word_embeddings` false at both levels), and is written with the ORIGINAL's chat_template.jinja, tokenizer.json,
tokenizer_config.json and generation_config.json (`--original`, default: the HF cache's meta-models/Muse-Glimmer-30B):
  untied  the tiny's own text weights + a random lm_head
  l8      8 layers ([s, s, s, f] x 2: NoPE at 3 and 7), random text weights
  gqa16   4 layers, 16 q / 1 kv heads x 16 (GQA 16, the real model's ratio), random text weights
  win16   l8 with sliding_window 16 (window crossings at tiny lengths)
  int4    untied's weights with every linear RTN-packed at g64 in the GPTQ names (`qweight` I32 [K/8][N], `scales`
          F16 [K/64][N], `qzeros` I32 0x77777777, `g_idx` I32 i // 64) and the real GPTQ repo's quantization_config
          (`--gptq`, default its HF cache snapshot); --twin DIR also writes the dequantised bf16 form
          (bf16(float(q - 8) x float(scale)), dequant.py's rule) - the checkpoint from_pretrained reads
  assist  a 2-layer MuseGlimmerAssistantModel for 23f (hidden 64, 4 q / 2 kv x 16, FF 256, block 16,
          target_layer_ids [1, 5] - the l8 target's layers -, mask_token_id 201818, window 2048), random bf16
  assist16  assist's weights (the same generator) with sliding_window 16 (spec 23f: window crossings at tiny lengths)
Weights: linears N(0, 1/sqrt(fan_in)), centered-norm weights N(0, 0.1) (they act as 1 + w), the final norm and the
drafter's norms N(1, 0.1), lm_head N(0, 1) (raw logits of a few units, so the softcap is not the identity).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

LM = "model.language_model."
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
VARIANTS = ("untied", "l8", "gqa16", "win16", "int4", "assist", "assist16")
SMALL_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
GROUP = 64
BF = torch.bfloat16


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"mmt_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hf_snapshot(repo: str) -> str | None:
    org, name = repo.split("/")
    for hub in (os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"), "/hf/hub"):
        c = sorted(glob.glob(os.path.join(hub, f"models--{org}--{name}", "snapshots", "*")))
        if c and os.path.exists(os.path.join(c[0], "config.json")):
            return c[0]
    return None


def text_shapes(t: dict) -> dict[str, tuple]:
    """Every text tensor's shape for a text config dict (names as the checkpoint stores them)."""
    H, I, L = t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"]
    nq, nkv, hd, V = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"], t["vocab_size"]
    out = {LM + "embed_tokens.weight": (V, H), LM + "norm.weight": (H,), "lm_head.weight": (V, H)}
    lin = {"self_attn.q_proj": (nq * hd, H), "self_attn.k_proj": (nkv * hd, H), "self_attn.v_proj": (nkv * hd, H),
           "self_attn.gate_proj": (nq * hd, H), "self_attn.o_proj": (H, nq * hd), "mlp.gate_proj": (I, H),
           "mlp.up_proj": (I, H), "mlp.down_proj": (H, I)}
    for i in range(L):
        b = f"{LM}layers.{i}."
        for n in NORMS:
            out[b + n + ".weight"] = (H,)
        for k, s in lin.items():
            out[b + k + ".weight"] = s
    return out


def rand_tensor(name: str, shape: tuple, g: torch.Generator) -> torch.Tensor:
    if len(shape) == 2:
        std = 1.0 if name == "lm_head.weight" else shape[1] ** -0.5
        return (torch.randn(shape, generator=g) * std).to(BF)
    if name.endswith(tuple(n + ".weight" for n in NORMS)):
        return (torch.randn(shape, generator=g) * 0.1).to(BF)              # centered: the module uses 1 + w
    return (1.0 + torch.randn(shape, generator=g) * 0.1).to(BF)            # the final norm (plain w)


def pack_int4(w: torch.Tensor, g: int = GROUP):
    """RTN sym int4 at group g: w [N][K] -> qweight I32 [K/8][N], qzeros I32 [K/g][N/8] (0x77777777), scales F16
    [K/g][N], g_idx I32 [K] (i // g); dequant (q - 8) x scale."""
    N, K = w.shape
    wf = w.float().reshape(N, K // g, g)
    s16 = (wf.abs().amax(-1) / 7.0).clamp_min(1e-8).to(torch.float16)
    q = (torch.round(wf / s16.float().unsqueeze(-1)) + 8).clamp(0, 15).to(torch.int64).reshape(N, K).t()   # [K][N]
    r = q.reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= r[:, i, :] << (4 * i)
    qweight = (packed - ((packed >> 31) << 32)).to(torch.int32)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int64)
    qzeros = (qz - ((qz >> 31) << 32)).to(torch.int32)
    g_idx = (torch.arange(K, dtype=torch.int32) // g).contiguous()
    return qweight, qzeros, s16.t().contiguous(), g_idx


def read_all(snap: str) -> dict[str, torch.Tensor]:
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as h:
        return {k: h.get_tensor(k) for k in h.keys()}


def write(out: str, cfg: dict, sd: dict, original: str, fmt: str = "pt") -> None:
    os.makedirs(out, exist_ok=True)
    for fn in SMALL_FILES:
        shutil.copyfile(os.path.join(original, fn), os.path.join(out, fn))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    save_file({k: v.contiguous() for k, v in sorted(sd.items())}, os.path.join(out, "model.safetensors"),
              metadata={"format": fmt})


def make(tiny: str, out: str, variant: str, original: str, gptq: str | None, seed: int = 0, twin: str | None = None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    # int4 is untied quantised: the same generator, so its lm_head is untied's (gate Q's tiny pair)
    g = torch.Generator().manual_seed(seed * 1000 + VARIANTS.index({"int4": "untied", "assist16": "assist"}.get(variant, variant)))
    cfg = json.load(open(os.path.join(tiny, "config.json"), encoding="utf-8"))
    src = read_all(tiny)
    if variant in ("assist", "assist16"):
        return make_assist(out, original, g, window=16 if variant == "assist16" else 2048)
    t = cfg["text_config"]
    cfg["tie_word_embeddings"] = t["tie_word_embeddings"] = False
    if variant in ("l8", "win16"):
        t["num_hidden_layers"] = 8
        t["layer_types"] = ["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + ["full_attention"]
        th = t["rope_parameters"]["rope_theta"]
        t["layer_rope_theta"] = [th, th, th, 0, th, th, th, 0]
    if variant == "win16":
        t["sliding_window"] = 16
    if variant == "gqa16":
        t["num_attention_heads"], t["num_key_value_heads"] = 16, 1
    sd = {k: v for k, v in src.items() if not (k.startswith(LM) or k == "lm_head.weight")}    # the vision tensors
    shapes = text_shapes(t)
    for name, shape in shapes.items():
        keep = variant in ("untied", "int4") and name in src and tuple(src[name].shape) == shape
        sd[name] = src[name].to(BF) if keep else rand_tensor(name, shape, g)
    sd[LM + "embed_tokens.weight"] = src[LM + "embed_tokens.weight"].to(BF)      # the tiny's table in every variant
    if variant == "int4":
        dq = _load("dequant")
        qc = json.load(open(os.path.join(gptq, "config.json"), encoding="utf-8"))["quantization_config"]
        cfg["quantization_config"] = qc
        twin_sd = dict(sd)
        for name in list(sd):
            if name.startswith(LM + "layers.") and name.endswith("_proj.weight"):
                base = name[: -len(".weight")]
                qw, qz, sc, gi = pack_int4(sd.pop(name))
                sd.update({base + ".qweight": qw, base + ".qzeros": qz, base + ".scales": sc, base + ".g_idx": gi})
                twin_sd[name] = dq.dequant_gptq(qw, sc, GROUP).t().contiguous()
        if twin:
            tcfg = json.loads(json.dumps(cfg))
            tcfg.pop("quantization_config")
            write(twin, tcfg, twin_sd, original)
    write(out, cfg, sd, original)
    return out


def make_assist(out: str, original: str, g: torch.Generator, window: int = 2048) -> str:
    H, I, nq, nkv, hd, L, taps = 64, 256, 4, 2, 16, 2, [1, 5]
    cfg = {"architectures": ["MuseGlimmerAssistantModel"], "model_type": "muse_glimmer_assistant",
           "attention_dropout": 0, "block_size": 16, "bos_token_id": 200000, "dtype": "bfloat16", "eos_token_id": 200001,
           "head_dim": hd, "hidden_act": "silu", "hidden_size": H, "intermediate_size": I,
           "layer_types": ["sliding_attention"] * L, "mask_token_id": 201818, "max_position_embeddings": 131072,
           "num_attention_heads": nq, "num_hidden_layers": L, "num_key_value_heads": nkv, "pad_token_id": 200018,
           "rms_norm_eps": 1e-5, "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
           "sliding_window": window, "target_layer_ids": taps}
    shapes = {"encoder.fc.weight": (H, H * len(taps)), "encoder.output_norm_enc.weight": (H,), "norm.weight": (H,)}
    for i in range(L):
        b = f"layers.{i}."
        shapes.update({b + "input_layernorm.weight": (H,), b + "post_attention_layernorm.weight": (H,),
                       b + "self_attn.q_proj.weight": (nq * hd, H), b + "self_attn.k_proj.weight": (nkv * hd, H),
                       b + "self_attn.v_proj.weight": (nkv * hd, H), b + "self_attn.o_proj.weight": (H, nq * hd),
                       b + "self_attn.q_norm.weight": (hd,), b + "self_attn.k_norm.weight": (hd,),
                       b + "mlp.gate_proj.weight": (I, H), b + "mlp.up_proj.weight": (I, H),
                       b + "mlp.down_proj.weight": (H, I)})
    sd = {}
    for name, shape in shapes.items():
        if len(shape) == 2:
            sd[name] = (torch.randn(shape, generator=g) * shape[1] ** -0.5).to(BF)
        else:
            sd[name] = (1.0 + torch.randn(shape, generator=g) * 0.1).to(BF)     # the drafter's norms are plain w
    write(out, cfg, sd, original)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiny")
    ap.add_argument("out")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--original", default=None, help="the bf16 original's snapshot (its template / tokenizer files)")
    ap.add_argument("--gptq", default=None, help="the GPTQ repo's snapshot (its quantization_config, for int4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twin")
    a = ap.parse_args()
    original = a.original or hf_snapshot("meta-models/Muse-Glimmer-30B")
    gptq = a.gptq or hf_snapshot("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
    if not original:
        sys.exit("FATAL: the original's small files are needed (--original or the HF cache)")
    if a.variant == "int4" and not gptq:
        sys.exit("FATAL: int4 needs the GPTQ repo's config.json (--gptq or the HF cache)")
    print(make(a.tiny, a.out, a.variant, original, gptq, a.seed, a.twin))


if __name__ == "__main__":
    main()
