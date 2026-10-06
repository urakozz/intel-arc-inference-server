#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A synthetic Kolibri-1 checkpoint at the REAL widths, in the int4 export's form (spec 20c Task 1).

    make_synth.py <out-dir> --layers N --attn int4|bf16 --tokenizer <dir> [--seed 0]

Spec 20c needs a checkpoint the engine and the reference both read before 20b's real one exists
(spec 20 decision 1): every tensor named, typed and shaped as `tools/quantize_kolibri1.sh` (AutoRound
0.17.0 @ 6afaecdb, `--format auto_round:auto_gptq`) exports the real model, but with random weights
and only the first N layers (1..50; the default 5 holds both kinds: 4 sliding + the full layer 4):

  config.json   the vendored third_party/kolibri1/config.json with num_hidden_layers = N and
                layer_types = the real pattern[:N], every other key unchanged; plus
                quantization_config as AutoRound writes it (quant_method auto-round, packing_format
                auto_round:auto_gptq, bits 4, group_size 64, sym, extra_config bits 16 for the
                shared experts and - the bf16 arm - attention)
  int4          every routed expert's gate/up/down_proj (and q/k/v/o_proj in the int4 arm), packed
                by RTN (pack_rtn_g64): per 64-row group of K, scale = max|w| / 8 rounded to f16,
                q = clamp(round(w / scale) + 8, 0, 15), nibbles low-first along K; qweight I32
                [K/8, N], scales F16 [K/64, N], qzeros I32 [K/64, N/8] all 0x77777777 - the bytes of
                AutoRound's auto_round:auto_gptq export (dequant.py: w = (q - 8) x scale)
  bf16          router mlp.gate, moe.router.expert_bias, shared_experts.*, every norm, embed_tokens,
                lm_head (and attention in the bf16 arm)

The draws are make_tiny.py's at the real widths: norms 1 + 0.2 N(0,1), expert_bias 0.5 N(0,1),
linears N(0,1) / sqrt(K), the router 3 N(0,1) / sqrt(K), embedding N(0,1), lm_head N(0,1) / sqrt(2560).
One layer is generated, packed and written at a time (one shard per layer; shard 1 holds the
embedding, lm_head and the final norm), so the peak is about two layers' packed bytes (~2 GB),
not the model. Size: 2.0 GB + 0.83 GB per layer (int4 arm; + 0.05 GB per layer in the bf16 arm).

Afterwards `check.py <out-dir> --attn <arm>` must print ACCEPTED (names, dtypes, shapes, qzeros, g64
scales, quantization_config); the packer's arithmetic is test_kolibri_quant.py's test_synth_*.
The tokenizer files (tokenizer.json, tokenizer_config.json, generation_config.json) are copied from
--tokenizer, falling back to the vendored port directory; a missing tokenizer.json is reported, not
fatal (the engine reads none; tools/oracle/tokenize.py needs it).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

QZ = 0x77777777
AR_VERSION = "0.17.0"


# --------------------------------------------------------------------------------------------
# the packer (dequant.py's rule, inverted by RTN)

def pack_rtn_g64(w: torch.Tensor):
    """w [N, K] (bf16 or fp32) -> (qweight I32 [K/8, N], scales F16 [K/64, N], qzeros I32 [K/64, N/8]).

    Per (column n, 64-row group g of K): scale = f16(max|w| / 8); q = clamp(round(w / f32(scale)) + 8,
    0, 15) (round half to even); element k of column n is nibble k % 8 of word [k // 8][n], low nibble
    first - dequant.py's layout. A group of zeros gets scale 0 and q = 8."""
    N, K = w.shape
    if K % 64 or N % 8:
        raise ValueError(f"pack_rtn_g64: K={K} must be whole 64-groups and N={N} whole 8-column words")
    wf = w.float().reshape(N, K // 64, 64)
    sc16 = (wf.abs().amax(dim=-1) / 8.0).to(torch.float16)                 # [N, K/64]
    sc = sc16.float()
    div = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.clamp(torch.round(wf / div[..., None]) + 8, 0, 15).to(torch.int64).reshape(N, K)
    qt = q.t().reshape(K // 8, 8, N)                                         # [K/8, 8, N]
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 8, 1)
    words = (qt << shifts).sum(dim=1)                                        # [K/8, N] in [0, 2^32)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    qz = torch.full((K // 64, N // 8), QZ, dtype=torch.int32)               # 0x77777777 < 2^31
    return words.contiguous(), sc16.t().contiguous(), qz.contiguous()


def dequant(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[K/8, N] I32 + [K/64, N] F16 -> [N, K] fp32: w = (q - 8) x scale, dequant.py's rule with the
    product left in fp32 (dequant.py rounds it to bf16 once; the test compares the fp32 product)."""
    rows, n = qweight.shape
    k = rows * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF               # [K/8, 8, N]
    q = nib.reshape(k, n).to(torch.float32)
    w = (q - 8.0) * scales.to(torch.float32).repeat_interleave(64, dim=0)   # [K, N]
    return w.t().contiguous()


# --------------------------------------------------------------------------------------------
# the config

def synth_config(layers: int, attn: str) -> dict:
    """The vendored config.json at `layers` layers (layer_types the real pattern's prefix) with the
    quantization_config AutoRound 0.17.0 writes for the arm."""
    if attn not in ("int4", "bf16"):
        raise ValueError(f"attn {attn!r}: int4 or bf16")
    d = C.read_json(os.path.join(C.PORT_DIR, "config.json"))
    real = list(d["layer_types"])
    if not 1 <= layers <= len(real):
        raise ValueError(f"--layers {layers}: 1..{len(real)}")
    d["num_hidden_layers"] = layers
    d["layer_types"] = real[:layers]
    extra = {}
    for i in range(layers):
        p = f"model.layers.{i}."
        for n in ("gate_proj", "up_proj", "down_proj"):
            extra[p + "mlp.shared_experts." + n] = {"bits": 16, "data_type": "float"}
        if attn == "bf16":
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                extra[p + "self_attn." + n] = {"bits": 16, "data_type": "float"}
    d["quantization_config"] = {
        "quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "bits": 4,
        "group_size": 64, "sym": True, "data_type": "int", "iters": 0,
        "autoround_version": AR_VERSION, "extra_config": extra,
    }
    return d


# --------------------------------------------------------------------------------------------
# the weights

def _randn(g: torch.Generator, shape, scale: float) -> torch.Tensor:
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)


def _int4(out: dict, base: str, w: torch.Tensor) -> None:
    qw, sc, qz = pack_rtn_g64(w)
    out[base + ".qweight"], out[base + ".scales"], out[base + ".qzeros"] = qw, sc, qz


def layer_tensors(c, i: int, attn: str, seed: int) -> dict:
    """Layer i's tensors (checkpoint names) - generated and packed here, nothing kept."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + 17 + i)
    D, hd = c.hidden_size, c.head_dim
    qn, kvn = c.num_attention_heads * hd, c.num_key_value_heads * hd
    I, SI, E = c.moe_intermediate_size, c.shared_expert_intermediate_size, c.num_experts
    p = f"model.layers.{i}."
    out = {}
    for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
        out[p + n + ".weight"] = (1 + 0.2 * torch.randn(D, generator=g)).to(torch.bfloat16)
    for n in ("q_norm", "k_norm"):
        out[p + f"self_attn.{n}.weight"] = (1 + 0.2 * torch.randn(hd, generator=g)).to(torch.bfloat16)
    for n, (rows, cols) in (("q_proj", (qn, D)), ("k_proj", (kvn, D)), ("v_proj", (kvn, D)), ("o_proj", (D, qn))):
        w = _randn(g, (rows, cols), 1 / math.sqrt(cols))
        if attn == "int4":
            _int4(out, p + "self_attn." + n, w)
        else:
            out[p + f"self_attn.{n}.weight"] = w
    out[p + "mlp.gate.weight"] = _randn(g, (E, D), 3 / math.sqrt(D))
    out[p + "moe.router.expert_bias"] = (0.5 * torch.randn(E, generator=g)).to(torch.bfloat16)
    for n, (rows, cols) in (("gate_proj", (SI, D)), ("up_proj", (SI, D)), ("down_proj", (D, SI))):
        out[p + f"mlp.shared_experts.{n}.weight"] = _randn(g, (rows, cols), 1 / math.sqrt(cols))
    for e in range(E):
        for n, (rows, cols) in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
            _int4(out, p + f"mlp.experts.{e}.{n}", _randn(g, (rows, cols), 1 / math.sqrt(cols)))
    return out


def top_tensors(c, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed * 1_000_003 + 5)
    D, V = c.hidden_size, c.vocab_size
    return {"model.embed_tokens.weight": _randn(g, (V, D), 1.0),
            "lm_head.weight": _randn(g, (V, D), D ** -0.5),
            "model.norm.weight": (1 + 0.2 * torch.randn(D, generator=g)).to(torch.bfloat16)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--attn", choices=("int4", "bf16"), required=True)
    ap.add_argument("--tokenizer", required=True, help="a directory with tokenizer.json / tokenizer_config.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from safetensors.torch import save_file
    ref = C.kolibri_ref()
    cfg = synth_config(a.layers, a.attn)
    c = ref.KConfig.from_dict({k: v for k, v in cfg.items() if k != "quantization_config"})
    os.makedirs(a.out, exist_ok=True)
    n_shards = 1 + c.num_hidden_layers
    wm, total = {}, 0
    t0 = time.time()

    def write(k: int, sd: dict) -> None:
        nonlocal total
        fn = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(sd, os.path.join(a.out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in sd})
        total += sum(t.numel() * t.element_size() for t in sd.values())

    write(0, top_tensors(c, a.seed))
    for i in range(c.num_hidden_layers):
        write(1 + i, layer_tensors(c, i, a.attn, a.seed))
        print(f"  layer {i} ({c.layer_types[i]}) written, {time.time() - t0:.0f}s", flush=True)
    want = set()
    for name in ref.expected_names(c):
        base = name[:-len(".weight")]
        packed = ".mlp.experts." in name or (a.attn == "int4" and ".self_attn." in name and
                                              name.endswith(("q_proj.weight", "k_proj.weight",
                                                             "v_proj.weight", "o_proj.weight")))
        want |= {base + s for s in (".qweight", ".scales", ".qzeros")} if packed else {name}
    if want != set(wm):
        raise SystemExit(f"make_synth: names differ from kolibri_ref.expected_names: missing "
                         f"{sorted(want - set(wm))[:5]}, extra {sorted(set(wm) - want)[:5]}")
    C.write_json(os.path.join(a.out, "model.safetensors.index.json"),
                 {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))})
    C.write_json(os.path.join(a.out, "config.json"), cfg)
    C.write_json(os.path.join(a.out, "kolibri_recipe.json"),
                 {"attn": a.attn, "synthetic": True, "layers": a.layers, "seed": a.seed,
                  "maker": "tools/quantize/kolibri/make_synth.py (RTN of random weights)"})
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(a.tokenizer, fn)
        if not os.path.exists(src):
            src = os.path.join(C.PORT_DIR, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, fn))
        else:
            print(f"make_synth: no {fn} in {a.tokenizer} or {C.PORT_DIR} - not copied "
                  f"(tokenize.py needs it; the engine does not)", file=sys.stderr)
    print(f"wrote {a.out}: {len(wm)} tensors in {n_shards} shards, {total / 1e9:.2f} GB, "
          f"{a.layers} layers ({c.layer_types.count('full_attention')} full), attention {a.attn}, "
          f"{time.time() - t0:.0f}s")
    print(json.dumps({"layers": a.layers, "attn": a.attn, "bytes": total}))


if __name__ == "__main__":
    main()
