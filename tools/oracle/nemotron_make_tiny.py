#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A tiny random NVIDIA Nemotron 3.5 Super (VL, 120B-A12B) in the original checkpoint's own layout.

    nemotron_make_tiny.py <out> --source <the real BF16 snapshot> [--seed 0] [--experts 16] [--top-k 4]

What quantize_nemotron.sh's `selftest` quantises and test_nemotron_ref.py runs: the real snapshot's config.json with
the language model shrunk (hidden 256; layers mamba / moe / attention / moe - one of each kind, the MoE twice; Mamba-2
64 heads x 8, 2 groups, state 32, conv 4, chunk 16; latent MoE 1024 -> 128 latent, 16 relu^2 experts x 64, top-4, a
shared expert x 128, the fp32 sigmoid router with its F32 bias, scaling 5.0; attention 4 q / 2 kv heads x 64, NoPE)
and a one-block RADIO vision tower of width 64, every other key (the native `nemotron_h` spelling, the MTP
`mtp_layers_block_type`, the token ids) as the real file has it. Every multiple a g64 quantiser needs holds: each
linear's K and N are multiples of 64 (Mamba-2 64 heads, so in_proj's N = 2 x 512 + 2 x 2 x 32 + 64 = 1216).

The weights: the model is built from that config with the checkpoint's own remote code (trust_remote_code), its
random init kept except where a value matters - norms 1 + noise, the router bias random, dt_bias in [-2, 0] (so
softplus(dt_raw + dt_bias) stays above the native class's 0.001 floor and the reference, which has no floor, agrees
with it) - and saved by save_pretrained, which writes the original's names (`language_model.backbone.*`, routed
experts per expert, the projector as `mlp1.*`, RADIO's `vision_model.radio_model.*`). Then what save_pretrained
cannot write: the router bias as F32 (the real dtype), RADIO's input conditioner (F32 [3][1][1]), and the MTP layer
`language_model.mtp.layers.{0,1}.*` (eh_proj [H][2H], enorm / hnorm / norm, attention; norm, final_layernorm, the
router and bias, latent projections, shared and routed experts) in a second shard, as the real checkpoint keeps it in
its last shards. Every name is checked against the real index's names (digits generalised): the tiny has no tensor
the real model lacks and lacks none of its patterns. The tokenizer, chat template, generation and processor files
and the remote code come from the real snapshot.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

LLM = {"hidden_size": 256, "layers_block_type": ["mamba", "moe", "attention", "moe"], "num_hidden_layers": 4,
       "mamba_num_heads": 64, "mamba_head_dim": 8, "n_groups": 2, "ssm_state_size": 32, "chunk_size": 16,
       "moe_intermediate_size": 64, "intermediate_size": 64, "moe_latent_size": 128,
       "moe_shared_expert_intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64}
VISION = {"num_hidden_layers": 1, "hidden_size": 64, "num_attention_heads": 2}
COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
        "preprocessor_config.json", "processor_config.json", "processing_nemotron_h_omni.py",
        "image_processing_nemotron_h_omni.py", "configuration_nemotron_h_omni.py", "modeling_nemotron_h_omni.py",
        "configuration_radio.py", "modeling_radio.py")
MAIN, MTP = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"


def pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def tiny_config(source: str, experts: int, top_k: int) -> dict:
    with open(os.path.join(source, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["llm_config"].update(LLM, n_routed_experts=experts, num_experts_per_tok=top_k)
    cfg["vision_config"].update(VISION)
    cfg["vit_hidden_size"] = VISION["hidden_size"]
    cfg["projector_hidden_size"] = 128
    return cfg


def mtp_tensors(sd: dict, llm: dict, gen: torch.Generator) -> dict:
    """The MTP layer pair, shaped from the trunk's attention block (2) and MoE block (1), fresh random values."""
    H = llm["hidden_size"]

    def rnd(shape, std=0.02):
        return (std * torch.randn(shape, generator=gen)).to(torch.bfloat16)

    def norm():
        return (1 + 0.1 * torch.randn(H, generator=gen)).to(torch.bfloat16)
    t = "language_model.backbone.layers."
    m0, m1 = "language_model.mtp.layers.0.", "language_model.mtp.layers.1."
    out = {m0 + "eh_proj.weight": rnd((H, 2 * H)), m0 + "enorm.weight": norm(), m0 + "hnorm.weight": norm(),
           m0 + "norm.weight": norm(), m1 + "norm.weight": norm(), m1 + "final_layernorm.weight": norm()}
    for k, v in sd.items():
        if k.startswith(t + "2.mixer."):
            out[m0 + k[len(t + "2."):]] = rnd(tuple(v.shape))
        elif k.startswith(t + "1.mixer."):
            name = m1 + k[len(t + "1."):]
            if k.endswith("e_score_correction_bias"):
                out[name] = (0.05 * torch.randn(tuple(v.shape), generator=gen)).float()
            else:
                out[name] = rnd(tuple(v.shape))
    return out


def make(out: str, source: str, seed: int = 0, experts: int = 16, top_k: int = 4) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM
    os.makedirs(out, exist_ok=True)
    cfg = tiny_config(source, experts, top_k)
    for fn in COPY:
        if os.path.exists(os.path.join(source, fn)):
            shutil.copyfile(os.path.join(source, fn), os.path.join(out, fn))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out, trust_remote_code=True),
                                             trust_remote_code=True).to(torch.bfloat16)
    llm = cfg["llm_config"]
    with torch.no_grad():
        for name, p in model.language_model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm_f.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=gen))
            elif name.endswith("dt_bias"):
                p.copy_(-2 * torch.rand(p.shape, generator=gen))
        for name, b in model.language_model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.05 * torch.randn(b.shape, generator=gen))
    tmp = os.path.join(out, "save_pretrained.tmp")
    model.save_pretrained(tmp)
    sd = {}
    for fn in sorted(os.listdir(tmp)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(tmp, fn), "pt") as f:
                sd.update({k: f.get_tensor(k) for k in f.keys()})
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    for k in sd:
        if k.endswith("e_score_correction_bias"):
            sd[k] = sd[k].float()
    sd["vision_model.radio_model.input_conditioner.norm_mean"] = torch.tensor(cfg["norm_mean"]).view(3, 1, 1)
    sd["vision_model.radio_model.input_conditioner.norm_std"] = torch.tensor(cfg["norm_std"]).view(3, 1, 1)
    mtp = mtp_tensors(sd, llm, gen)

    with open(os.path.join(source, "model.safetensors.index.json"), encoding="utf-8") as f:
        real = {pattern(k) for k in json.load(f)["weight_map"]}
    # save_pretrained spells this one buffer under radio_model; the original ships it one level up
    sd["vision_model.summary_idxs"] = sd.pop("vision_model.radio_model.summary_idxs")
    # and RADIO's layer scales, which the original does not ship at all (the model initialises them)
    for k in [k for k in sd if re.search(r"\.layer_scale[12]\.lambda1$", k)]:
        del sd[k]
    tiny = {pattern(k) for k in list(sd) + list(mtp)}
    extra, lacking = sorted(tiny - real), sorted(real - tiny)
    if extra or lacking:
        raise SystemExit(f"the tiny's names differ from the real checkpoint's: extra {extra[:8]}, lacking {lacking[:8]}")
    save_file(sd, os.path.join(out, MAIN), metadata={"format": "pt"})
    save_file(mtp, os.path.join(out, MTP), metadata={"format": "pt"})
    wm = {**{k: MAIN for k in sd}, **{k: MTP for k in mtp}}
    total = sum(t.numel() * t.element_size() for t in list(sd.values()) + list(mtp.values()))
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=1)
    print(f"wrote {out}: {len(sd)} + {len(mtp)} MTP tensors ({total / 2**20:.1f} MiB); "
          f"layers {llm['layers_block_type']}, {experts} experts top-{top_k}, hidden {llm['hidden_size']}, "
          f"vocab {llm['vocab_size']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--source", required=True, help="the real BF16 snapshot (config, remote code, tokenizer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()
    make(a.out, a.source, a.seed, a.experts, a.top_k)


if __name__ == "__main__":
    main()
