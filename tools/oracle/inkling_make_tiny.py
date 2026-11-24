# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Write a deterministic 4-layer Inkling text checkpoint in release tensor names."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .inkling_ref import ROOT, STREAMS


def make_tiny(out, seed=41):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    hidden, heads, kv, dim, rel, window, ff, experts, vocab = 256, 4, 2, 64, 8, 16, 64, 16, 512
    rank, markov, conv = 16, 16, 4
    text = {
        "model_type": "inkling2_loop_text", "hidden_size": hidden, "vocab_size": vocab, "unpadded_vocab_size": 507,
        "eos_token_id": 506, "num_hidden_layers": 4, "local_layer_ids": [0, 2], "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": dim, "swa_num_attention_heads": heads,
        "swa_num_key_value_heads": kv, "swa_head_dim": dim, "d_rel": rel, "rel_extent": window,
        "sliding_window_size": window, "sconv_kernel_size": conv, "rms_norm_eps": 1e-6,
        "log_scaling_n_floor": 8, "log_scaling_alpha": .1, "qk_l2_clip_threshold": 5.,
        "rel_l2_clip_threshold": 2., "n_layers_per_attn_res_block": 1, "n_routed_experts": experts,
        "num_experts_per_tok": 4, "n_shared_experts": 0, "intermediate_size": ff, "route_scale": 4.,
        "mlp_gate_softcap": 4., "mlp_up_softcap": 25., "final_logit_softcapping": 30.,
        "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": rank, "is_varloop": False},
    }
    mtp = {
        "hidden_size": hidden, "num_hidden_layers": 3, "num_attention_heads": heads, "num_key_value_heads": kv,
        "head_dim": dim, "intermediate_size": 2 * hidden, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
        "local_layer_ids": [0, 2], "sliding_window_size": window, "block_size": 8, "markov_rank": markov,
        "markov_head_type": "rnn_hidden", "target_layer_ids": [0, 3, 5, 8],
    }
    (out / "config.json").write_text(json.dumps({"model_type": "inkling2_loop", "text_config": text, "mtp_config": mtp},
                                                indent=2) + "\n")

    weights = {}

    def rnd(key, shape, gain=.08, bias=0, dtype=torch.bfloat16):
        weights[key] = (torch.randn(shape, generator=rng) * gain + bias).to(dtype)

    rnd(ROOT + "embed.weight", (vocab, hidden))
    rnd(ROOT + "embed_norm.weight", (hidden,), bias=1)
    rnd(ROOT + "norm.weight", (hidden,), bias=1)
    rnd(ROOT + "unembed.weight", (vocab, hidden), gain=.035)
    rnd(ROOT + "attn_res.output_query", (hidden,), dtype=torch.float32)
    for key in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
        rnd(ROOT + key, (4, hidden), dtype=torch.float32)

    groups = []
    for i in range(4):
        layer = ROOT + f"layers.{i}."
        local = i in text["local_layer_ids"]
        if i:
            rnd(ROOT + f"attn_res.attn_queries.{i - 1}", (hidden,) if local else (3, hidden), dtype=torch.float32)
        rnd(ROOT + f"attn_res.mlp_queries.{i}", (hidden,), dtype=torch.float32)
        for key in ("attn_norm", "mlp_norm") if local else ("q_input_norm", "k_input_norm", "v_input_norm", "mlp_norm"):
            rnd(layer + key + ".weight", (hidden,), bias=1)
        for key, shape in {
            "wq_du.weight": (heads * dim, hidden), "wk_dv.weight": (kv * dim, hidden),
            "wv_dv.weight": (kv * dim, hidden), "wo_ud.weight": (hidden, heads * dim),
            "wr_du.weight": (heads * rel, hidden), "gate_attn.weight": (heads * dim, hidden),
            "attn_sink_proj.weight": (heads, hidden), "rel_logits_proj.proj": (rel, window),
        }.items():
            rnd(layer + "attn." + key, shape)
        rnd(layer + "attn.q_l2_clip.scale", (heads, dim), bias=1)
        rnd(layer + "attn.rel_l2_clip.scale", (heads, rel), bias=1)
        for key in ("k", "v"):
            base = layer + f"attn.{key}_sconv."
            if local:
                rnd(base + "weight", (kv * dim, 1, conv))
            else:
                rnd(base + "static_w", (conv, kv * dim))
                rnd(base + "weight_proj", (kv * conv, hidden), gain=.025)
        for key in ("attn_sconv", "mlp_sconv"):
            rnd(layer + key + ".weight", (hidden, 1, conv))
        if 1 <= i < 3:
            for stream in STREAMS:
                base = layer + f"controllers.{stream}."
                rnd(base + "down_weight", (rank, 2 * hidden), gain=.035)
                rnd(base + "gain_weight", (hidden, rank))
                rnd(base + "forget_weight", (hidden, rank))
        rnd(layer + "mlp.gate.weight", (experts, hidden), gain=.045)
        rnd(layer + "mlp.gate.bias", (experts,), gain=.015, dtype=torch.float32)

        nv = i < 3
        group = {"weight_transform": "nvfp4_nearest" if nv else "mxfp8_nearest",
                 "weight_block_shape": [1, 16] if nv else [32, 32], "tensors": []}
        if nv:
            group["weight_scale_format"] = "amax_denominator"
        for part, n, k in (("w13", 2 * ff, hidden), ("w2", hidden, ff)):
            name = layer + f"mlp.experts.{part}_weight"
            group["tensors"].append(name)
            if nv:
                weights[name] = torch.randint(256, (experts, n, k // 2), dtype=torch.uint8, generator=rng)
                scale = torch.randint(35, 52, (experts, n, k // 16), dtype=torch.uint8, generator=rng)
                weights[name + "_scale"] = scale.view(torch.float8_e4m3fn)
                weights[name + "_amax"] = .4 + .2 * torch.rand(experts, generator=rng)
                weights[name + "_outer_scale_denominator"] = torch.tensor([9.])
            else:
                code = torch.randint(256, (experts, n, k), dtype=torch.uint8, generator=rng)
                code[(code & 127) == 127] ^= 1
                weights[name] = code.view(torch.float8_e4m3fn)
                scales = torch.randint(114, 118, (experts, n // 32, k // 32), dtype=torch.uint8, generator=rng)
                weights[name + "_scale"] = scales.repeat_interleave(32, dim=1)
        groups.append(group)
    (out / "hf_quant_config.json").write_text(json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}},
                                                         indent=2) + "\n")
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, out / shard, metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": sum(w.nbytes for w in weights.values())},
                     "weight_map": dict.fromkeys(weights, shard)}) + "\n")

    weights.clear()
    rnd("fuse_weight", (4, 4), gain=.3)
    rnd("mask_embed", (hidden,), gain=.3)
    rnd("norm.weight", (hidden,), bias=1)
    for i in range(3):
        layer = f"layers.{i}."
        for key in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            rnd(layer + key + ".weight", (hidden,), bias=1)
        for key in ("q_norm", "k_norm", "k_norm_ctx"):
            rnd(layer + "self_attn." + key + ".weight", (dim,), bias=1)
        for key, shape in {
            "q_proj": (heads * dim, hidden), "k_proj": (kv * dim, hidden), "v_proj": (kv * dim, hidden),
            "k_proj_ctx": (kv * dim, hidden), "v_proj_ctx": (kv * dim, hidden), "o_proj": (hidden, heads * dim),
        }.items():
            rnd(layer + "self_attn." + key + ".weight", shape)
        for key, shape in (("gate_proj", (2 * hidden, hidden)), ("up_proj", (2 * hidden, hidden)),
                           ("down_proj", (hidden, 2 * hidden))):
            rnd(layer + "mlp." + key + ".weight", shape)
    rnd("markov_head.dflare_norm.weight", (hidden,), bias=1)
    rnd("markov_head.markov_w1.weight", (vocab, markov))
    rnd("markov_head.rnn_proj.weight", (3 * markov, 2 * markov + 2 * hidden), gain=.04)
    rnd("markov_head.rnn_proj.bias", (3 * markov,))
    rnd("markov_head.hidden_out.weight", (hidden, markov), gain=.04)
    rnd("markov_head.hidden_out.bias", (hidden,))
    save_file(weights, out / "mtp.safetensors")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()
    make_tiny(args.out, args.seed)
