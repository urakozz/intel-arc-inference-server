#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A synthetic Muse-Glimmer-30B checkpoint at the REAL widths, in the served int4 export's form (spec 23b Task 4).

    make_synth.py <out-dir> --layers N --tokenizer <bf16 snapshot> [--gptq <int4 -GPTQ snapshot>] [--seed 0]

The engine's loader (loader::MuseCheckpoint) and the reference (tools/oracle/muse_ref.py) both read it before the
real 18.96 GB checkpoint is on a machine: every text tensor named, typed and shaped as
Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ ships it (muse_ref.expected_names(tc, "gptq")), random weights, the
first N layers (1..52; 4 holds both kinds: 3 sliding + the full layer 3):

  config.json   the -GPTQ repo's config.json (--gptq, default its HF cache snapshot: its quantization_config -
                quant_method gptq, bits 4, group_size 64, sym, desc_act false - its vision_config) with
                text_config.num_hidden_layers = N and layer_types / layer_rope_theta truncated to N
  int4          the eight linears of every layer (q, k, v, the attention gate_proj, o; gate, up, down), RTN-packed
                by 23a's packer (tools/oracle/muse_make_tiny.pack_int4: per 64-row group of K scale = f16(max|w| / 7),
                q = clamp(round(w / scale) + 8, 0, 15), nibbles low-first along K): qweight I32 [K/8, N], scales F16
                [K/64, N], qzeros I32 [K/64, N/8] all 0x77777777, g_idx I32 [K] = i // 64
  bf16          the four centered norms a layer, embed_tokens, lm_head and the final norm at the real shapes
  small files   the ORIGINAL's tokenizer.json, tokenizer_config.json, chat_template.jinja (spec 23 decision 1) and
                generation_config.json (--tokenizer: the bf16 snapshot - its small files suffice)
  muse_synth.json   the recipe (layers, seed, vocab): check.py regenerates the bf16 twins from it

No vision tensors (the text path; the loader counts 0 skipped). The draws (spec 20b's scales, the centered norms
around 0): linears N(0, 1) / sqrt(K), the centered norms' w 0.2 N(0, 1) (they act as 1 + w), the final norm
1 + 0.2 N(0, 1) (plain w), embed_tokens N(0, 1), lm_head N(0, 1) / sqrt(hidden). Every tensor has its own seeded
generator (seed and name), so check.py regenerates any one of them alone. One layer is generated, packed and written
at a time (one shard a layer; shard 1 holds the embedding, lm_head and the final norm): the peak is the top shard's
~5.4 GB of fp32 draws, not the model. Size (derived): 5.38 GB + 0.267 GB a layer (2 layers ~5.9 GB, 4 ~6.4 GB).
`vocab` (make() only, test_muse_quant.py) shrinks embed / lm_head and text_config.vocab_size - a checkpoint the
engine refuses by design (check_muse_config holds vocab_size), the tests' size knob.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ORACLE = os.path.join(REPO, "tools", "oracle")
LM = "model.language_model."
NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
RECIPE = "muse_synth.json"
GROUP = 64
BF = torch.bfloat16


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tiny = None


def tiny():
    """23a's muse_make_tiny (the packer, hf_snapshot) - no transformers needed."""
    global _tiny
    if _tiny is None:
        _tiny = _load(os.path.join(ORACLE, "muse_make_tiny.py"), "muse_synth_make_tiny")
    return _tiny


def gen(seed: int, name: str) -> torch.Generator:
    """The tensor's own generator: the draws of `name` do not depend on any other tensor (check.py's twins)."""
    return torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))


def linear_shapes(t: dict) -> dict[str, tuple]:
    """layer-relative linear -> [out N, in K] for a text config dict."""
    H, I = t["hidden_size"], t["intermediate_size"]
    q, kv = t["num_attention_heads"] * t["head_dim"], t["num_key_value_heads"] * t["head_dim"]
    return {"self_attn.q_proj": (q, H), "self_attn.k_proj": (kv, H), "self_attn.v_proj": (kv, H),
            "self_attn.gate_proj": (q, H), "self_attn.o_proj": (H, q), "mlp.gate_proj": (I, H),
            "mlp.up_proj": (I, H), "mlp.down_proj": (H, I)}


def source(name: str, shape: tuple, seed: int) -> torch.Tensor:
    """The bf16 weight `name` (a full checkpoint name; a linear's is `<base>.weight`) - the twin check.py compares."""
    g = gen(seed, name)
    if len(shape) == 2:
        if name.endswith("embed_tokens.weight"):
            std = 1.0
        elif name == "lm_head.weight":
            std = shape[1] ** -0.5
        else:
            std = shape[1] ** -0.5                                           # a linear: N(0, 1) / sqrt(K)
        return (torch.randn(shape, generator=g) * std).to(BF)
    if name.endswith(tuple(n + ".weight" for n in NORMS)):
        return (0.2 * torch.randn(shape, generator=g)).to(BF)               # centered: the module uses 1 + w
    return (1.0 + 0.2 * torch.randn(shape, generator=g)).to(BF)             # the final norm (plain w)


def synth_config(gptq_cfg: dict, layers: int, vocab: int | None = None) -> dict:
    cfg = json.loads(json.dumps(gptq_cfg))
    t = cfg["text_config"]
    n = t["num_hidden_layers"]
    if not 1 <= layers <= n:
        raise ValueError(f"--layers {layers}: 1..{n}")
    t["num_hidden_layers"] = layers
    for k in ("layer_types", "layer_rope_theta"):
        t[k] = list(t[k][:layers])
    if vocab is not None:
        t["vocab_size"] = vocab
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "gptq" or q.get("bits") != 4 or q.get("group_size") != GROUP or q.get("sym") is not True:
        raise ValueError(f"--gptq: its quantization_config is not int4 g64 sym gptq: {q}")
    return cfg


def layer_tensors(t: dict, i: int, seed: int) -> dict:
    """Layer i's tensors (checkpoint names), generated and packed here, nothing kept."""
    p = f"{LM}layers.{i}."
    out = {}
    for n in NORMS:
        out[p + n + ".weight"] = source(p + n + ".weight", (t["hidden_size"],), seed)
    for lin, shape in linear_shapes(t).items():
        qw, qz, sc, gi = tiny().pack_int4(source(p + lin + ".weight", shape, seed), GROUP)
        out.update({p + lin + ".qweight": qw, p + lin + ".qzeros": qz, p + lin + ".scales": sc, p + lin + ".g_idx": gi})
    return out


def top_tensors(t: dict, seed: int) -> dict:
    V, H = t["vocab_size"], t["hidden_size"]
    return {LM + "embed_tokens.weight": source(LM + "embed_tokens.weight", (V, H), seed),
            "lm_head.weight": source("lm_head.weight", (V, H), seed),
            LM + "norm.weight": source(LM + "norm.weight", (H,), seed)}


def default_snapshot(repo: str) -> str | None:
    return tiny().hf_snapshot(repo)


def make(out: str, layers: int, tokenizer: str, gptq: str, seed: int = 0, vocab: int | None = None,
         verbose: bool = True) -> dict:
    from safetensors.torch import save_file
    with open(os.path.join(gptq, "config.json"), encoding="utf-8") as f:
        cfg = synth_config(json.load(f), layers, vocab)
    t = cfg["text_config"]
    os.makedirs(out, exist_ok=True)
    n_shards = 1 + layers
    wm, total, t0 = {}, 0, time.time()

    def write(k: int, sd: dict) -> None:
        nonlocal total
        fn = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file({n: v.contiguous() for n, v in sd.items()}, os.path.join(out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in sd})
        total += sum(v.numel() * v.element_size() for v in sd.values())

    write(0, top_tensors(t, seed))
    if verbose:
        print(f"  embed / lm_head / norm written, {time.time() - t0:.0f}s", flush=True)
    for i in range(layers):
        write(1 + i, layer_tensors(t, i, seed))
        if verbose:
            print(f"  layer {i} ({t['layer_types'][i]}) written, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=2)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    recipe = {"synthetic": True, "layers": layers, "seed": seed, "vocab": t["vocab_size"], "bytes": total,
              "maker": "tools/quantize/muse/make_synth.py (RTN of random weights, 23a's packer)"}
    with open(os.path.join(out, RECIPE), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
    for fn in SMALL_FILES:
        shutil.copyfile(os.path.join(tokenizer, fn), os.path.join(out, fn))
    if verbose:
        full = t["layer_types"].count("full_attention")
        print(f"wrote {out}: {len(wm)} tensors in {n_shards} shards, {total / 1e9:.2f} GB, {layers} layers ({full} full), "
              f"{time.time() - t0:.0f}s")
    return recipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokenizer", default=None, help="the bf16 original's snapshot (its small files)")
    ap.add_argument("--gptq", default=None, help="the -AutoRound-GPTQ snapshot (its config.json)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = a.tokenizer or default_snapshot("meta-models/Muse-Glimmer-30B")
    gptq = a.gptq or default_snapshot("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
    if not tok or not gptq:
        sys.exit("FATAL: make_synth needs the original's small files (--tokenizer) and the -GPTQ repo's config.json "
                 "(--gptq), or both in the HF cache")
    print(json.dumps(make(a.out, a.layers, tok, gptq, a.seed)))


if __name__ == "__main__":
    main()
