#!/usr/bin/env python3
"""Spec 21a: tiny qwen4_exp checkpoints the downloaded tiny model cannot give (tests only).

    qwen4exp_make_tiny.py <out dir> [--hidden 128] [--layer-types lllq] [--experts 16] [--topk 4]
        [--form tiny|bf16|intel|ours] [--group 64] [--tie-router A,B] [--mtp] [--indexer-budget 2048]
        [--ple-base 1000] [--seed 0] [--dtype bf16|f32] [--twin <dir>]

Random weights at a small hidden size on the real model's structure (GDN 16 / 48 heads of 128, QSA 4
q heads of 256 on 2 kv heads in the tiny config's shape, indexer 4 x 128, MoE intermediate 640, 4
hyper-connection streams, PLE 16 heads x 160 at one-indexed layer 2) written in one of the forms
qwen4exp_ref.expected_names knows:
  tiny   `model.layers.*`, per-expert `.weight` experts (the downloaded tiny's naming)
  bf16   the original's: `model.language_model.*`, fused `mlp.experts.gate_up_proj` / `down_proj`
  intel  Intel's: per-expert int4 `qweight` / `qzeros` (0x77777777) / `scales` F16 at --group (128), rest bf16
  ours   spec 21 §5: int4 at --group (64) for the experts, shared experts, QSA q/k/v/o, GDN qkv / z / out
--tie-router A,B makes expert B's router row a copy of expert A's (exact ties: test_route_ties).
--mtp adds the head's `mtp.*` tensors (the original's names; per-expert `.weight` experts except in bf16's
fused form). --twin DIR also writes the SAME weights as the dequantised `tiny` form (int4 linears
dequantised by dequant.py's rule), the checkpoint transformers' from_pretrained reads.
`write_ple_int8` writes 21b Task 1's int8 PLE file (per head `ple.h<h>.q` I8 / `ple.h<h>.s`).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"mt_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("qwen4exp_ref")
_dequant = _load("dequant")
TINY_CONFIG = {   # qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next's config.json, the fields that matter
    "architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", "attention_bias": False,
    "attention_dropout": 0.0, "bos_token_id": 248044, "eos_token_id": 248044, "hc_count": 4, "hc_lowrank": 320,
    "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": 16, "indexer_budget": 2048,
    "indexer_compress_ratio": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
    "initializer_range": 0.02, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
    "max_position_embeddings": 262144, "moe_intermediate_size": 640, "ngram_size": 3, "ngram_vocab_size_base": 8000,
    "norm_topk_prob": True, "num_attention_heads": 4, "num_experts": 4, "num_experts_per_tok": 2,
    "num_hidden_layers": 4, "num_key_value_heads": 2, "output_gate_type": "sigmoid", "pad_token_id": None,
    "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560, "ple_layer_ids": [2],
    "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                               "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                               "rope_type": "default"},
    "router_aux_loss_coef": 0.001, "seed": 1234, "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
    "tie_word_embeddings": False, "use_cache": True, "vocab_size": 248320, "mtp_num_hidden_layers": 1,
}
KINDS = {"l": "linear_attention", "q": "qwen_sparse_attention"}


def config(hidden=128, layer_types="lllq", experts=16, topk=4, indexer_budget=2048, ple_base=1000, **over) -> dict:
    d = json.loads(json.dumps(TINY_CONFIG))
    d.update(hidden_size=hidden, num_hidden_layers=len(layer_types), layer_types=[KINDS[c] for c in layer_types],
             num_experts=experts, num_experts_per_tok=topk, indexer_budget=indexer_budget,
             ngram_vocab_size_base=ple_base)
    d.update(over)
    return d


def pack_int4(w: torch.Tensor, g: int):
    """RTN sym int4 at group g: w [N][K] -> (qweight int32 [K/8][N], qzeros int32 [K/g][N/8] = 0x77777777,
    scales f16 [K/g][N]); dequant (q - 8) x scale (dequant.py's rule)."""
    N, K = w.shape
    wf = w.float().reshape(N, K // g, g)
    s16 = (wf.abs().amax(-1) / 7.0).clamp_min(1e-8).to(torch.float16)
    q = (torch.round(wf / s16.float().unsqueeze(-1)) + 8).clamp(0, 15).to(torch.int64).reshape(N, K).t()   # [K][N]
    r = q.reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= r[:, i, :] << (4 * i)
    qweight = (packed - ((packed >> 31) << 32)).to(torch.int32)
    qzeros = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int64)
    qzeros = (qzeros - ((qzeros >> 31) << 32)).to(torch.int32)
    return qweight, qzeros, s16.t().contiguous()


def unpack_int4(qweight, scales, g) -> torch.Tensor:
    return _dequant.dequant_gptq(qweight, scales, g).t().contiguous()


def random_state(tc, gen: torch.Generator, tie=None, mtp: bool = False) -> dict[str, torch.Tensor]:
    """fp32 random weights for every tensor of Qwen4ExpForCausalLM(tc) (model names, fused experts), plus
    `mtp.*` (fused experts) when asked."""
    m = ref.mq()
    with torch.device("meta"):
        model = m.Qwen4ExpForCausalLM(tc)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    sizes, offsets, total, padded = ref.facts.head_table(tc.to_dict(), 0)
    mult = ref.facts.layer_multipliers(tc.vocab_size, tc.ngram_size, 0, tc.seed)
    if mtp:
        hc = tc.hc_count * tc.hidden_size
        H = tc.hidden_size
        L0 = [k for k in shapes if k.startswith("model.layers.") and ".self_attn." in k]
        qsa_layer = L0[0].split(".")[2]
        for k, s in list(shapes.items()):
            pre = f"model.layers.{qsa_layer}."
            if k.startswith(pre) and ".ple." not in k:
                shapes["mtp.layers.0." + k[len(pre):]] = s
        shapes.update({"mtp.fc_embedding.weight": (H, H), "mtp.fc_hidden.weight": (H, H),
                       "mtp.pre_fc_norm_embedding.weight": (H,), "mtp.pre_fc_norm_hidden.weight": (hc,),
                       "mtp.hyper_connection_mixer.hc_norm.weight": (hc,),
                       "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (tc.hc_lowrank, hc),
                       "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (hc, tc.hc_lowrank)})
    sd = {}
    for k, s in sorted(shapes.items()):
        if k.endswith("layer_multipliers"):
            sd[k] = torch.tensor(mult, dtype=torch.long)
        elif k.endswith("ngram_heads_vocab_sizes"):
            sd[k] = torch.tensor(sizes, dtype=torch.long)
        elif k.endswith("ngram_heads_offsets"):
            sd[k] = torch.tensor(offsets, dtype=torch.long)
        elif k.endswith("ngram_embedding.weight"):
            sd[k] = torch.randn(s, generator=gen)
        elif k.endswith("A_log"):
            sd[k] = torch.empty(s).uniform_(1.0, 16.0, generator=gen).log()
        elif k.endswith("dt_bias"):
            sd[k] = 1.0 + 0.1 * torch.randn(s, generator=gen)
        elif k.endswith("linear_attn.norm.weight"):
            sd[k] = 1.0 + 0.1 * torch.randn(s, generator=gen)
        elif len(s) == 1:                                    # (1 + w) norms
            sd[k] = 0.1 * torch.randn(s, generator=gen)
        elif k.endswith("conv1d.weight"):
            sd[k] = 0.5 * torch.randn(s, generator=gen)
        elif k.endswith("embed_tokens.weight"):
            sd[k] = torch.randn(s, generator=gen)
        elif k.endswith("mlp.gate.weight"):
            sd[k] = 3.0 * torch.randn(s, generator=gen) / math.sqrt(s[-1])
        else:
            sd[k] = torch.randn(s, generator=gen) / math.sqrt(s[-1])
    if tie:
        a, b = tie
        for k in sd:
            if k.endswith("mlp.gate.weight"):
                sd[k][b] = sd[k][a]
    return sd


def write_form(out: str, raw_cfg: dict, tc, sd: dict, form: str, group: int, dtype: torch.dtype,
               ple_parts: int) -> list[str]:
    """Write `sd` (model names, fused experts, fp32) in `form`; returns the tensor names written."""
    P = "model." if form == "tiny" else "model.language_model."
    E, I = tc.num_experts, tc.moe_intermediate_size
    out_sd: dict[str, torch.Tensor] = {}

    def put(name, t):
        out_sd[name] = t.to(dtype) if t.is_floating_point() else t

    def lin(name, w, quant):
        if quant:
            qw, qz, sc = pack_int4(w, group)
            out_sd[name + ".qweight"], out_sd[name + ".qzeros"], out_sd[name + ".scales"] = qw, qz, sc
        else:
            put(name + ".weight", w)

    def is_q_dense(k):
        return form == "ours" and (any(f"self_attn.{s}." in k for s in ref.QSA_LINEARS)
                                   or any(f"linear_attn.{s}." in k for s in ref.GDN_LINEARS)
                                   or ".mlp.shared_expert." in k)
    for k, t in sd.items():
        is_mtp = k.startswith("mtp.")
        name = k if is_mtp else (P + k[len("model."):] if k.startswith("model.") else k)
        if k.endswith("ngram_embedding.weight"):
            base = name[: -len(".weight")]
            rows = t.shape[0] // ple_parts
            for j in range(ple_parts):
                put(f"{base}.shard_{j}.weight", t[j * rows:(j + 1) * rows].clone())
            continue
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
            if k.endswith("down_proj"):
                continue
            b = name[: -len("gate_up_proj")]
            dn = sd[k[: -len("gate_up_proj")] + "down_proj"]
            if form == "bf16":
                put(b + "gate_up_proj", t)
                put(b + "down_proj", dn)
                continue
            quant = form in ("intel", "ours") and not is_mtp
            for e in range(E):
                lin(f"{b}{e}.gate_proj", t[e, :I], quant)
                lin(f"{b}{e}.up_proj", t[e, I:], quant)
                lin(f"{b}{e}.down_proj", dn[e], quant)
            continue
        if k.endswith(".weight") and t.dim() == 2 and is_q_dense(k) and not is_mtp:
            lin(name[: -len(".weight")], t, True)
            continue
        put(name, t)
    os.makedirs(out, exist_ok=True)
    save_file(out_sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(out_sd)}}, f)
    cfg = dict(raw_cfg)
    if form != "tiny":
        text = dict(cfg)
        text["model_type"] = "qwen4_exp_text"
        text.pop("architectures", None)
        cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
               "tie_word_embeddings": False, "text_config": text}
        if form in ("intel", "ours"):
            cfg["quantization_config"] = {"bits": 4, "group_size": group, "sym": True, "data_type": "int",
                                          "packing_format": "auto_round:auto_gptq", "quant_method": "auto-round"}
    cfg["dtype"] = "bfloat16" if dtype == torch.bfloat16 else "float32"
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return sorted(out_sd)


def dequant_twin(sd: dict, form: str, group: int) -> dict:
    """The weights a quantised `form` stores, dequantised (dequant.py's rule) - what transformers must see."""
    if form not in ("intel", "ours"):
        return sd
    out = {}
    I = None
    for k, t in sd.items():
        if k.startswith("mtp."):
            out[k] = t
            continue
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
            if I is None:
                I = sd[k].shape[1] // 2 if k.endswith("gate_up_proj") else sd[k].shape[2]
            out[k] = torch.stack([_rt(t[e], group) for e in range(t.shape[0])])
            continue
        quant = form == "ours" and t.dim() == 2 and (
            any(f"self_attn.{s}." in k for s in ref.QSA_LINEARS) or any(f"linear_attn.{s}." in k for s in ref.GDN_LINEARS)
            or ".mlp.shared_expert." in k)
        out[k] = _rt(t, group) if quant else t
    return out


def _rt(w: torch.Tensor, g: int) -> torch.Tensor:
    """w -> pack -> dequant (bf16): the values an int4 checkpoint carries. A fused gate_up [2I][H] packs as
    its two halves (gate, up) - per row, so packing the whole is the same as packing the halves."""
    qw, _, sc = pack_int4(w, g)
    return unpack_int4(qw, sc, g)


def make(out: str, hidden=128, layer_types="lllq", experts=16, topk=4, form="tiny", group=None, tie=None,
         mtp=False, indexer_budget=2048, ple_base=1000, seed=0, dtype=torch.bfloat16, twin: str | None = None,
         **over) -> tuple:
    """Write the variant (and its dequantised `tiny` twin); returns (text config, the fp32 state dict)."""
    group = group or (128 if form == "intel" else 64)
    raw = config(hidden, layer_types, experts, topk, indexer_budget, ple_base, **over)
    tc = ref.text_config(out, raw=raw, dtype=dtype)
    gen = torch.Generator().manual_seed(seed)
    sd = random_state(tc, gen, tie, mtp)
    write_form(out, raw, tc, sd, form, group, dtype, tc.split_ngram_parts)
    if twin:
        write_form(twin, raw, tc, dequant_twin(sd, form, group), "tiny", group, dtype, tc.split_ngram_parts)
    return tc, sd


def write_ple_int8(table: torch.Tensor, tc, out_dir: str, scale: str = "f32") -> None:
    """21b Task 1's file for a table [padded][160] (any float dtype): per head h, `ple.h<h>.q` I8
    [prime_h][160] = clamp(rne(w / s), -127, 127) with s = max|w| / 127 in fp32 (spec 9's row rule; a zero
    row has s = 0, q = 0) and `ple.h<h>.s` F32 or BF16 [prime_h]; the I64 constants; an index json."""
    sizes, offsets, _, _ = ref.facts.head_table(tc.to_dict(), 0)
    out = {}
    for h, (n, o) in enumerate(zip(sizes, offsets)):
        w = table[o:o + n].float()
        s = w.abs().amax(-1) / 127.0
        if scale == "bf16":
            s = s.to(torch.bfloat16).float()
        q = torch.where(s.unsqueeze(-1) > 0, torch.round(w / torch.where(s > 0, s, 1.0).unsqueeze(-1)),
                        torch.zeros_like(w)).clamp(-127, 127).to(torch.int8)
        out[f"ple.h{h}.q"] = q.contiguous()
        out[f"ple.h{h}.s"] = s.to(torch.bfloat16 if scale == "bf16" else torch.float32).contiguous()
    out["ple.layer_multipliers"] = torch.tensor(ref.facts.layer_multipliers(tc.vocab_size, tc.ngram_size, 0, tc.seed))
    out["ple.ngram_heads_vocab_sizes"] = torch.tensor(sizes)
    out["ple.ngram_heads_offsets"] = torch.tensor(offsets)
    os.makedirs(out_dir, exist_ok=True)
    fn = "ple_int8-00001-of-00001.safetensors"
    save_file(out, os.path.join(out_dir, fn), metadata={"rule": "row-int8-spec9", "scale": scale,
                                                         "source": "qwen4exp_make_tiny.write_ple_int8"})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: fn for k in sorted(out)}}, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layer-types", default="lllq")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--form", default="tiny", choices=["tiny", "bf16", "intel", "ours"])
    ap.add_argument("--group", type=int)
    ap.add_argument("--tie-router")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--indexer-budget", type=int, default=2048)
    ap.add_argument("--ple-base", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    ap.add_argument("--twin")
    a = ap.parse_args()
    tie = tuple(int(x) for x in a.tie_router.split(",")) if a.tie_router else None
    make(a.out, a.hidden, a.layer_types, a.experts, a.topk, a.form, a.group, tie, a.mtp, a.indexer_budget,
         a.ple_base, a.seed, torch.bfloat16 if a.dtype == "bf16" else torch.float32, a.twin)
    print(f"wrote {a.out} ({a.form})" + (f" and its dequantised twin {a.twin}" if a.twin else ""))


if __name__ == "__main__":
    main()
