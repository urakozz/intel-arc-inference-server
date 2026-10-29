#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3.5 Bolt 74B-A9B's config.json written from code (spec 25b Task 1, Decision 20: optional beside the vendored
copy tests/model/bolt/config.json).

    make_config.py <out config.json> [--layers N] [--form gptq-g128|ar-g64|none]
    make_config.py --check                      # every typed value against the vendored config.json

Every value below is typed from the config of the served checkpoint, our export
urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ @ 8932aa8d (its `llm_config` is NVIDIA's); the key it
comes from is the same key of `llm_config` (or of the top level, or `quantization_config`) - the comment on a group names
where it was read. The per-layer pattern and the per-MoE-layer
(E, I) are the facts sheet's (docs/probe-bolt-2026-10-21.md §2), which 25a read from the same file. `--layers N` keeps the
first N blocks (`block_configs`, `layers_block_type`); nothing else moves (the model has no `num_hidden_layers` key: the
count is the length of `block_configs`). `--check` is the test that the typed values ARE the export's (exit 1 naming the
first key that differs); it needs no torch, no transformers and no network. The `gptq-g128` form is the GPTQ g128
spelling (the loader's exact g128 -> g64 expansion, a format capability the synthetic checkpoints test); no served file
carries it, so `--check` holds only its arithmetic keys to the export's.

The wrapper is the language-relevant part of the omni config (architectures, model_type, dtype, eos, tie flag) - the
vision tower's own keys (`vision_config`, the image / video keys) are not written: nothing here reads them.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED = os.path.join(HERE, "config.json")

INF = {"__float__": "Infinity"}          # transformers' special-float encoding, as published (time_step_limit)

# llm_config.layers_block_type / block_configs: M mamba, E moe, * attention (88 blocks: 40 / 40 / 8)
PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
# the facts sheet's per MoE layer (E, I): layer:(n_routed_experts, moe_intermediate_size)
MOE_EI = ("1:(240,2048) 3:(240,1536) 5:(240,2048) 8:(304,2304) 10:(240,2304) 12:(240,2048) 14:(240,2048) 17:(304,2304) "
          "19:(304,2048) 21:(432,2560) 23:(432,2304) 26:(240,2048) 28:(432,2304) 30:(304,2560) 32:(304,2304) "
          "34:(432,2304) 37:(368,2304) 39:(368,2048) 41:(368,2304) 43:(432,2560) 45:(368,2560) 48:(496,2560) "
          "50:(432,2560) 52:(240,2304) 54:(368,2304) 56:(368,2560) 59:(304,2048) 61:(240,2304) 63:(240,2304) "
          "65:(240,2048) 67:(240,2048) 70:(368,2048) 72:(368,2304) 74:(432,2304) 76:(432,2560) 79:(496,2560) "
          "81:(432,2560) 83:(432,2560) 85:(496,2560) 87:(368,2560)")
BLOCK = {"M": "mamba", "E": "moe", "*": "attention"}
LAYERS = len(PATTERN)


def moe_ei() -> dict:
    out = {}
    for tok in MOE_EI.split():
        l, rest = tok.split(":")
        e, i = rest.strip("()").split(",")
        out[int(l)] = (int(e), int(i))
    return out


def block_configs(layers: int) -> list:
    ei = moe_ei()
    if len(PATTERN) != 88 or sorted(ei) != [i for i, c in enumerate(PATTERN) if c == "E"]:
        raise SystemExit("make_config: the typed pattern and the typed (E, I) layers disagree")
    out = []
    for l, c in enumerate(PATTERN[:layers]):
        b = {"block_type": BLOCK[c]}
        if c == "E":
            b.update(moe_intermediate_size=ei[l][1], n_routed_experts=ei[l][0], num_experts_per_tok=9)
        out.append(b)
    return out


def llm_config(layers: int = LAYERS) -> dict:
    if not 1 <= layers <= LAYERS:
        raise SystemExit(f"--layers {layers}: 1..{LAYERS}")
    bc = block_configs(layers)
    return {
        # llm_config.* as published (the Puzzle config, RP:, over NemotronH's, RC:)
        "_megatron_bridge_hf_prefix": "model",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "attention_bias": False, "attention_dropout": 0.0,
        "auto_map": {"AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
                     "AutoModelForCausalLM": "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM"},
        "aux_loss_alpha": 0.0,
        "blockwise_members": ["block_type", "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok"],
        "bos_token_id": 1, "chunk_size": 128, "conv_kernel": 4, "dtype": "bfloat16", "eos_token_id": 2,
        "expand": 2, "head_dim": 128, "hidden_act": "relu2", "hidden_dropout": 0.0, "hidden_size": 4096,
        "initializer_range": 0.02, "intermediate_size": 2688, "layer_norm_epsilon": 1e-05,
        "mamba_head_dim": 64, "mamba_hidden_act": "silu", "mamba_num_heads": 128, "mamba_proj_bias": False,
        "mamba_ssm_cache_dtype": "float32", "max_position_embeddings": 262144, "mlp_bias": False,
        "mlp_hidden_act": "relu2", "model_type": "nemotron_h_puzzle", "moe_latent_size": 1024,
        "moe_shared_expert_intermediate_size": 5376, "moe_shared_expert_overlap": True,
        "mtp_block_configs": [], "mtp_layers_block_type": [], "mtp_num_hidden_layers": None,
        "n_group": 1, "n_groups": 8, "n_shared_experts": 1, "norm_eps": 1e-05, "norm_topk_prob": True,
        "num_attention_heads": 32, "num_experts": 512, "num_key_value_heads": 2, "num_local_experts": 512,
        "num_logits_to_keep": 1, "num_nextn_predict_layers": 0, "pad_token_id": 0,
        "rescale_prenorm_residual": True, "residual_in_fp32": False, "rms_norm_eps": 1e-05,
        "routed_scaling_factor": 5.0, "router_aux_loss_coef": 0.0, "scoring_func": "sigmoid",
        "sliding_window": None, "ssm_state_size": 128, "tie_word_embeddings": False,
        "time_step_floor": 0.0001, "time_step_limit": [0.0, dict(INF)], "time_step_max": 0.1, "time_step_min": 0.001,
        "topk_group": 1, "use_bias": False, "use_cache": True, "use_conv_bias": True, "use_mamba_kernels": True,
        "use_qk_norm": False, "vocab_size": 131072,
        "block_configs": bc,
        "layers_block_type": [b["block_type"] for b in bc],
    }


def quant_gptq_g128() -> dict:
    """The GPTQ g128 spelling (AutoRound 0.17.0's GPTQ writer at group 128): gptq, provider auto-round, the 44 `-:`
    exclusions (embeddings, lm_head, vision_model, vision_projector and the router of each of the 40 MoE layers - ALL 40,
    whatever --layers) and the ONE restating `+:` rule (spec 25 §5.2)."""
    dyn = {"+:.*mixer\\.out_proj$": {"bits": 4, "group_size": 128, "sym": True},
           "-:.*embeddings.*": {}, "-:.*lm_head.*": {}, "-:.*vision_model.*": {}, "-:.*vision_projector.*": {}}
    for l in moe_ei():
        dyn[f"-:.*language_model\\.model\\.layers\\.{l}\\.mixer\\.gate.*"] = {}
    return {"autoround_version": "0.17.0", "bits": 4, "damp_percent": 0.01, "data_type": "int", "desc_act": False,
            "dynamic": dict(sorted(dyn.items())), "group_size": 128, "lm_head": False, "low_gpu_mem_usage": True,
            "provider": "auto-round", "quant_method": "gptq", "sym": True, "true_sequential": False}


def quant_ar_g64() -> dict:
    """The synthetics' native g64: AutoRound's own writer's spelling (as tools/oracle/bolt_make_tiny.py's int4-g64: quant_method
    auto-round, packing_format auto_round:auto_gptq, no desc_act, no dynamic) - the export's arithmetic keys (`--check`), without
    its per-router `extra_config` (check.py refuses one; the routers are kept bf16 either way)."""
    return {"autoround_version": "0.17.0", "bits": 4, "data_type": "int", "group_size": 64, "iters": 0,
            "packing_format": "auto_round:auto_gptq", "quant_method": "auto-round", "sym": True}


ARITH = ("autoround_version", "bits", "data_type", "group_size", "packing_format", "quant_method", "sym")


def quant_export() -> dict:
    """quantization_config.* of our export (the served checkpoint): AutoRound 0.17.0, auto-round / auto_round:auto_gptq, g64
    sym, and one `extra_config` exclusion {bits 16, data_type float} for the router of each of the 40 MoE layers (ALL 40,
    whatever --layers; written in the `language_model.model.` spelling though the shards spell the trunk `backbone.`)."""
    ec = {f".*language_model\\.model\\.layers\\.{l}\\.mixer\\.gate.*": {"bits": 16, "data_type": "float"}
          for l in sorted(moe_ei())}
    return {"bits": 4, "data_type": "int", "group_size": 64, "sym": True, "low_gpu_mem_usage": True,
            "autoround_version": "0.17.0", "block_name_to_quantize": "language_model.backbone.layers",
            "quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "extra_config": ec}


def config(layers: int = LAYERS, form: str = "none") -> dict:
    c = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "dtype": "bfloat16", "eos_token_id": 11,
         "model_type": "nemotron_h_omni", "tie_word_embeddings": False, "llm_config": llm_config(layers)}
    if form == "gptq-g128":
        c["quantization_config"] = quant_gptq_g128()
    elif form == "ar-g64":
        c["quantization_config"] = quant_ar_g64()
    elif form != "none":
        raise SystemExit(f"--form {form}: gptq-g128 | ar-g64 | none")
    return c


def check() -> int:
    with open(PUBLISHED, encoding="utf-8") as f:
        pub = json.load(f)
    mine = config(LAYERS, "none")
    bad = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    bad.append(f"{path}.{k}: published, not typed")
                elif k not in b:
                    bad.append(f"{path}.{k}: typed, not published")
                else:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                bad.append(f"{path}: {len(a)} typed entries, {len(b)} published")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a != b or type(a) is not type(b):
            bad.append(f"{path}: typed {a!r}, published {b!r}")
    llm_pub = pub["llm_config"]
    walk(mine["llm_config"], llm_pub, "llm_config")
    walk(quant_export(), pub["quantization_config"], "quantization_config")
    for form, q, keys in (("ar-g64", quant_ar_g64(), ARITH),
                          ("gptq-g128", quant_gptq_g128(), ("autoround_version", "bits", "data_type", "sym"))):
        for k in keys:   # the g128 spelling's own: group 128, quant_method gptq, no packing_format
            walk(q.get(k), pub["quantization_config"][k], f"{form}.{k}")
    for k in ("architectures", "dtype", "eos_token_id", "model_type", "tie_word_embeddings"):
        walk(mine[k], pub[k], k)
    n_ex = len(pub["quantization_config"]["extra_config"])
    for line in bad[:20]:
        print("DIFF", line)
    if bad:
        print(f"make_config --check: FAIL, {len(bad)} difference(s)")
        return 1
    print(f"make_config --check: OK - llm_config ({len(llm_pub)} keys, {len(llm_pub['block_configs'])} blocks), "
          f"quantization_config ({n_ex} extra_config router exclusions), and the wrapper keys equal the export's "
          f"config.json key by key; the synthetic forms' arithmetic keys equal its")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--form", default="none")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    if not a.out:
        ap.error("an output path, or --check")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(config(a.layers, a.form), f, indent=1)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
