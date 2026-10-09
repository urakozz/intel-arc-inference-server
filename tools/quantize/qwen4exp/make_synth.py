#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A synthetic Qwen3.8-Flash-Next (`qwen4_exp`) checkpoint at the REAL widths (spec 21b Task 1).

    make_synth.py <out-dir> --form ours|intel --tokenizer <dir> [--config <config.json>] [--layers 4]
                  [--ple-base 1000] [--mtp] [--seed 0]

Spec 21b's loader, planner and (21c) engine are built before any real checkpoint is on the box, and
21c's golden gates need a checkpoint the engine and the reference both read. This writes one: every
tensor named, typed and shaped as the export of `form` names it (`tools/oracle/qwen4exp_ref.py`
`expected_names`), random weights, and only the first N layers:

  config.json   the original's config.json (`--config`, default `<tokenizer dir>/config.json` - the
                original snapshot's small files) with text_config.num_hidden_layers = N, layer_types[:N]
                (N in 2..48; the default 4 holds three GDN layers - one of them the PLE layer 1 - and the
                QSA layer 3), ngram_vocab_size_base = --ple-base (the reduced-prime PLE table: the same
                hash and gather, small primes; every other width real), and quantization_config as the
                form's export writes it
  --form ours   spec 21 §5 (21q): routed experts, shared experts, QSA q/k/v/o and GDN in_proj_qkv /
                in_proj_z / out_proj int4 g64 in the auto_round:auto_gptq packing; everything else bf16
  --form intel  Intel/Qwen3.8-Flash-Next-W4A16-AutoRound's form: routed experts int4 **g128** (F16
                scales, per-expert names), dense / shared / MTP bf16 (its extra_config's bits-16 rules)
  PLE           the table as 128 bf16 shards `ngram_embedding.shard_K.weight` like the original's (rows =
                the padded total for the reduced base, split_ngram_parts 128) and the three I64 tensors
                computed by the formula (qwen4exp_facts.layer_multipliers / head_table)
  --mtp         the head's `mtp.*` (bf16, per-expert `.weight` experts: both forms ship it so)

The packer is tools/quantize/kolibri/make_synth.py's pack_rtn_g64 at group g: per (column n, g-row group
of K) scale = f16(max|w| / 8), q = clamp(round(w / f32(scale)) + 8, 0, 15) (round half to even), element
k of column n is nibble k % 8 of word [k // 8][n], low nibble first; qweight I32 [K/8, N], scales F16
[K/g, N], qzeros I32 [K/g, N/8] all 0x77777777 - dequant.py's rule w = (q - 8) x scale. The draws are
21a's tools/oracle/qwen4exp_make_tiny.py's at the real widths: (1 + w) norms 0.1 N(0,1), the GDN gated
norm and dt_bias 1 + 0.1 N(0,1), A_log = log U(1, 16), conv taps 0.5 N(0,1), routers 3 N(0,1) / sqrt(K),
the embedding and the PLE rows N(0,1), every other linear N(0,1) / sqrt(K).

One layer is generated, packed and written at a time (one shard per layer, one expert's matrices at a
time), so the peak is about one layer's tensors (~3 GB at N = 4, ESTIMATED), not the model. Size: ~1.40 GB
a layer (intel; ~0.75 GB ours) + 2.54 GB of embedding and head (+ 0.19 GB of the head's dense part and
5.03 GB of bf16 experts with --mtp; derived). Afterwards `check.py <out-dir> [--ple <dir>]` must print
ACCEPTED, and `ple_int8.py <out-dir> <out-dir>-ple-int8` makes the PLE file the engine reads.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ORACLE = os.path.join(REPO, "tools", "oracle")

import importlib.util  # noqa: E402

import torch  # noqa: E402

QZ = 0x77777777
P = "model.language_model."
AR_OURS = "0.17.0"
AR_INTEL = "0.15.0"
GDN_LINEARS = ("in_proj_qkv", "in_proj_z", "out_proj")
QSA_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
INT4 = (".qweight", ".qzeros", ".scales")


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


facts = _load(os.path.join(ORACLE, "qwen4exp_facts.py"), "q4synth_facts")
_ref = None


def ref():
    """tools/oracle/qwen4exp_ref.py (needs transformers 5.19.0 for text_config; imported on use)."""
    global _ref
    if _ref is None:
        _ref = _load(os.path.join(ORACLE, "qwen4exp_ref.py"), "q4synth_ref")
    return _ref


# ------------------------------------------------------------------------------------------------
# the packer

def pack(w: torch.Tensor, g: int):
    """w [N, K] -> (qweight I32 [K/8, N], scales F16 [K/g, N], qzeros I32 [K/g, N/8]) at group g."""
    N, K = w.shape
    if K % g or K % 8 or N % 8:
        raise ValueError(f"pack: K={K} must be whole {g}-groups and N={N} whole 8-column words")
    wf = w.float().reshape(N, K // g, g)
    sc16 = (wf.abs().amax(dim=-1) / 8.0).to(torch.float16)                  # [N, K/g]
    sc = sc16.float()
    div = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.clamp(torch.round(wf / div[..., None]) + 8, 0, 15).to(torch.int64).reshape(N, K)
    qt = q.t().reshape(K // 8, 8, N)
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 8, 1)
    words = (qt << shifts).sum(dim=1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    qz = torch.full((K // g, N // 8), QZ, dtype=torch.int32)
    return words.contiguous(), sc16.t().contiguous(), qz.contiguous()


def dequant(qweight: torch.Tensor, scales: torch.Tensor, g: int) -> torch.Tensor:
    """[K/8, N] I32 + [K/g, N] F16 -> [N, K] fp32: (q - 8) x scale, the product left in fp32."""
    rows, n = qweight.shape
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    q = nib.reshape(rows * 8, n).to(torch.float32)
    return ((q - 8.0) * scales.to(torch.float32).repeat_interleave(g, dim=0)).t().contiguous()


# ------------------------------------------------------------------------------------------------
# the config

def text_of(cfg: dict) -> dict:
    return cfg["text_config"] if "text_config" in cfg else cfg


def extra_bf16(form: str) -> dict:
    """The extra_config bits-16 rules of each export (Intel's patterns, 21a's facts; ours adds the dense
    rows it keeps bf16 and drops the ones it quantises)."""
    keep = [".*embed_tokens.*", ".*hyper_connection.*", ".*in_proj_a.*", ".*in_proj_b.*", ".*indexer.*",
            ".*mlp\\.gate.*", ".*shared_expert_gate.*", ".*ple.*", ".*mtp.*", ".*visual.*", ".*lm_head.*"]
    if form == "intel":
        keep += [".*linear_attn.*", ".*self_attn.*", ".*shared_expert.*"]
    return {k: {"bits": 16, "data_type": "float"} for k in keep}


def synth_config(src: dict, layers: int, form: str, ple_base: int) -> dict:
    if form not in ("ours", "intel"):
        raise ValueError(f"--form {form!r}: ours or intel")
    d = json.loads(json.dumps(src))
    t = text_of(d)
    real = list(t["layer_types"])
    if not 2 <= layers <= len(real):
        raise ValueError(f"--layers {layers}: 2..{len(real)} (the PLE layer is one-indexed layer 2: N >= 2)")
    t["num_hidden_layers"] = layers
    t["layer_types"] = real[:layers]
    t["ngram_vocab_size_base"] = int(ple_base)
    g = 64 if form == "ours" else 128
    d["quantization_config"] = {
        "quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "bits": 4, "group_size": g,
        "sym": True, "data_type": "int", "iters": 0, "block_name_to_quantize": "model.language_model.layers",
        "autoround_version": AR_OURS if form == "ours" else AR_INTEL, "extra_config": extra_bf16(form)}
    return d


class Shape:
    """The real model's widths from a text config dict."""

    def __init__(self, t: dict, experts: int | None = None, vocab: int | None = None):
        self.H = t["hidden_size"]
        self.hc = t["hc_count"]
        self.HC = self.hc * self.H
        self.low = t["hc_lowrank"]
        self.V = vocab or t["vocab_size"]
        self.E = experts or t["num_experts"]
        self.I = t["moe_intermediate_size"]
        self.SI = t["shared_expert_intermediate_size"]
        self.kh, self.vh, self.hd = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"]
        self.conv = (2 * self.kh + self.vh) * self.hd                     # 10240
        self.z = self.vh * t["linear_value_head_dim"]                      # 6144
        self.taps = t["linear_conv_kernel_dim"]
        self.qh, self.kvh, self.ad = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.idx = (t["indexer_n_heads"] + t["indexer_kv_heads"]) * t["indexer_head_dim"]   # 640
        self.idd = t["indexer_head_dim"]
        self.ple_e = t["ple_embed_dim"]
        self.heads = (t["ngram_size"] - 1) * t["heads_per_ngram"]
        self.pdim = self.ple_e // self.heads
        self.ptaps = t["ple_conv_kernel_size"]


def hc_specs(base: str, s: Shape, inject: bool) -> list:
    out = [(base + ".hc_norm.weight", (s.HC,)), (base + ".input_mix_weight_down.weight", (s.low, s.HC)),
           (base + ".input_mix_weight_up.weight", (s.HC, s.low))]
    return out + ([(base + ".block_inject_weight.weight", (s.hc, s.HC))] if inject else [])


def qsa_specs(b: str, s: Shape) -> list:
    return [(b + "self_attn.q_norm.weight", (s.ad,)), (b + "self_attn.k_norm.weight", (s.ad,)),
            (b + "self_attn.indexer.index_qk_proj.weight", (s.idx, s.H)),
            (b + "self_attn.indexer.q_layernorm.weight", (s.idd,)),
            (b + "self_attn.indexer.k_layernorm.weight", (s.idd,)),
            (b + "self_attn.q_proj.weight", (2 * s.qh * s.ad, s.H)), (b + "self_attn.k_proj.weight", (s.kvh * s.ad, s.H)),
            (b + "self_attn.v_proj.weight", (s.kvh * s.ad, s.H)), (b + "self_attn.o_proj.weight", (s.H, s.qh * s.ad))]


def gdn_specs(b: str, s: Shape) -> list:
    return [(b + "linear_attn.A_log", (s.vh,)), (b + "linear_attn.dt_bias", (s.vh,)),
            (b + "linear_attn.conv1d.weight", (s.conv, 1, s.taps)), (b + "linear_attn.norm.weight", (s.hd,)),
            (b + "linear_attn.in_proj_a.weight", (s.vh, s.H)), (b + "linear_attn.in_proj_b.weight", (s.vh, s.H)),
            (b + "linear_attn.in_proj_qkv.weight", (s.conv, s.H)), (b + "linear_attn.in_proj_z.weight", (s.z, s.H)),
            (b + "linear_attn.out_proj.weight", (s.H, s.z))]


def moe_dense_specs(b: str, s: Shape) -> list:
    return [(b + "mlp.gate.weight", (s.E, s.H)), (b + "mlp.shared_expert_gate.weight", (1, s.H)),
            (b + "mlp.shared_expert.gate_proj.weight", (s.SI, s.H)), (b + "mlp.shared_expert.up_proj.weight", (s.SI, s.H)),
            (b + "mlp.shared_expert.down_proj.weight", (s.H, s.SI))]


def expert_specs(b: str, s: Shape, e: int) -> list:
    return [(f"{b}mlp.experts.{e}.gate_proj.weight", (s.I, s.H)), (f"{b}mlp.experts.{e}.up_proj.weight", (s.I, s.H)),
            (f"{b}mlp.experts.{e}.down_proj.weight", (s.H, s.I))]


def ple_specs(b: str, s: Shape, rows_per: int, parts: int) -> list:
    p = b + "ple."
    out = [(p + "conv1d.weight", (s.HC, 1, s.ptaps)), (p + "key_proj.weight", (s.HC, s.ple_e)),
           (p + "value_proj.weight", (s.H, s.ple_e)), (p + "norm_conv.weight", (s.HC,)),
           (p + "norm_key.weight", (s.HC,)), (p + "norm_query.weight", (s.HC,))]
    return out + [(f"{p}ple_embedding.ngram_embedding.shard_{k}.weight", (rows_per, s.pdim)) for k in range(parts)]


def quantised(name: str, form: str) -> bool:
    """Whether `form`'s export packs this `.weight` (routed experts always; ours also the dense rows)."""
    if name.startswith("mtp."):
        return False
    if ".mlp.experts." in name:
        return True
    if form != "ours":
        return False
    return (any(f"self_attn.{n}." in name for n in QSA_LINEARS) or any(f"linear_attn.{n}." in name for n in GDN_LINEARS)
            or ".mlp.shared_expert." in name)


# ------------------------------------------------------------------------------------------------
# the weights

def draw(name: str, shape, g: torch.Generator) -> torch.Tensor:
    """21a's qwen4exp_make_tiny.random_state rules, fp32."""
    if name.endswith("A_log"):
        return torch.empty(shape).uniform_(1.0, 16.0, generator=g).log()
    if name.endswith("dt_bias") or name.endswith("linear_attn.norm.weight"):
        return 1.0 + 0.1 * torch.randn(shape, generator=g)
    if len(shape) == 1:
        return 0.1 * torch.randn(shape, generator=g)
    if name.endswith("conv1d.weight"):
        return 0.5 * torch.randn(shape, generator=g)
    if name.endswith("embed_tokens.weight") or ".ngram_embedding." in name:
        return torch.randn(shape, generator=g)
    if name.endswith("mlp.gate.weight"):
        return 3.0 * torch.randn(shape, generator=g) / math.sqrt(shape[-1])
    return torch.randn(shape, generator=g) / math.sqrt(shape[-1])


def put(out: dict, name: str, w: torch.Tensor, form: str) -> None:
    if quantised(name, form):
        qw, sc, qz = pack(w.to(torch.bfloat16), 64 if form == "ours" or ".mlp.experts." not in name else 128)
        base = name[: -len(".weight")]
        out[base + ".qweight"], out[base + ".scales"], out[base + ".qzeros"] = qw, sc, qz
    else:
        out[name] = w.to(torch.bfloat16)


def ple_tables(t: dict):
    sizes, offsets, total, padded = facts.head_table(t, 0)
    mult = facts.layer_multipliers(t["vocab_size"], t["ngram_size"], 0, t.get("seed", 1234))
    return sizes, offsets, total, padded, mult


def layer_tensors(t: dict, s: Shape, i: int, form: str, seed: int, prefix: str = P, mtp: bool = False) -> dict:
    """Layer i's tensors (checkpoint names), generated and packed here, nothing kept. `mtp`: the head's layer."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + 17 + (100 + i if mtp else i))
    b = f"{prefix}layers.{i}."
    kind = "qsa" if mtp else ("gdn" if t["layer_types"][i] == "linear_attention" else "qsa")
    out: dict = {}
    specs = hc_specs(b + "attn_hyper_connection", s, True) + hc_specs(b + "mlp_hyper_connection", s, True)
    specs += moe_dense_specs(b, s) + (gdn_specs(b, s) if kind == "gdn" else qsa_specs(b, s))
    ple_layers = {x - 1 for x in t.get("ple_layer_ids") or []}
    if not mtp and i in ple_layers:
        sizes, offsets, total, padded, mult = ple_tables(t)
        parts = t.get("split_ngram_parts", 512)
        if padded % parts:
            raise ValueError(f"the PLE table's {padded} rows do not split into {parts} shards")
        specs += ple_specs(b, s, padded // parts, parts)
        pe = b + "ple.ple_embedding."
        out[pe + "layer_multipliers"] = torch.tensor(mult, dtype=torch.int64)
        out[pe + "ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
        out[pe + "ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    for name, shape in specs:
        if mtp:
            out[name] = draw(name, shape, g).to(torch.bfloat16)
        else:
            put(out, name, draw(name, shape, g), form)
    for e in range(s.E):                               # one expert's three matrices at a time
        for name, shape in expert_specs(b, s, e):
            w = draw(name, shape, g)
            if mtp:
                out[name] = w.to(torch.bfloat16)
            else:
                put(out, name, w, form)
    return out


def rows_bf16(name: str, shape, g: torch.Generator, chunk: int = 16384) -> torch.Tensor:
    """A [V][H] draw made 16384 rows at a time (the fp32 vocab-sized temporary would be 2.5 GB)."""
    out = torch.empty(shape, dtype=torch.bfloat16)
    for a in range(0, shape[0], chunk):
        b = min(a + chunk, shape[0])
        out[a:b] = draw(name, (b - a, shape[1]), g).to(torch.bfloat16) if not name.startswith("lm_head") else \
            (torch.randn((b - a, shape[1]), generator=g) / math.sqrt(shape[1])).to(torch.bfloat16)
    return out


def top_tensors(s: Shape, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed * 1_000_003 + 5)
    out = {P + "embed_tokens.weight": rows_bf16("embed_tokens.weight", (s.V, s.H), g),
           "lm_head.weight": rows_bf16("lm_head.weight", (s.V, s.H), g)}
    for name, shape in hc_specs(P + "hyper_connection_mixer", s, False):
        out[name] = draw(name, shape, g).to(torch.bfloat16)
    return out


def mtp_top_tensors(s: Shape, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed * 1_000_003 + 7)
    specs = [("mtp.fc_embedding.weight", (s.H, s.H)), ("mtp.fc_hidden.weight", (s.H, s.H)),
             ("mtp.pre_fc_norm_embedding.weight", (s.H,)), ("mtp.pre_fc_norm_hidden.weight", (s.HC,))]
    specs += hc_specs("mtp.hyper_connection_mixer", s, False)
    return {n: draw(n, sh, g).to(torch.bfloat16) for n, sh in specs}


def make(out: str, form: str, tokenizer: str | None = None, config: str | None = None, layers: int = 4,
         ple_base: int = 1000, mtp: bool = False, seed: int = 0, experts: int | None = None,
         vocab: int | None = None, verbose: bool = True) -> dict:
    """Write the checkpoint; returns {layers, form, bytes, tensors}. `experts` / `vocab` shrink the
    checkpoint for the tests only (the written config says so, so nothing mistakes it for the model)."""
    from safetensors.torch import save_file
    src = config or os.path.join(tokenizer, "config.json")
    with open(src, encoding="utf-8") as f:
        cfg = synth_config(json.load(f), layers, form, ple_base)
    t = text_of(cfg)
    if experts:
        t["num_experts"] = experts
    if vocab:
        t["vocab_size"] = vocab
    s = Shape(t)
    os.makedirs(out, exist_ok=True)
    n_shards = 1 + layers + (2 if mtp else 0)
    wm, total = {}, 0
    t0 = time.time()

    def write(k: int, sd: dict) -> None:
        nonlocal total
        fn = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(sd, os.path.join(out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in sd})
        total += sum(x.numel() * x.element_size() for x in sd.values())

    write(0, top_tensors(s, seed))
    for i in range(layers):
        write(1 + i, layer_tensors(t, s, i, form, seed))
        if verbose:
            print(f"  layer {i} ({t['layer_types'][i]}) written, {time.time() - t0:.0f}s", flush=True)
    if mtp:
        write(1 + layers, mtp_top_tensors(s, seed))
        write(2 + layers, layer_tensors(t, s, 0, form, seed, prefix="mtp.", mtp=True))
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=1)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "qwen4exp_synth.json"), "w", encoding="utf-8") as f:
        json.dump({"form": form, "synthetic": True, "layers": layers, "ple_base": ple_base, "mtp": mtp, "seed": seed,
                   "experts": s.E, "vocab": s.V, "maker": "tools/quantize/qwen4exp/make_synth.py (RTN of random weights)"},
                  f, indent=1)
    if tokenizer:
        for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            p = os.path.join(tokenizer, fn)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(out, fn))
            elif verbose:
                print(f"make_synth: no {fn} in {tokenizer} - not copied (the engine reads none of them)", file=sys.stderr)
    return {"layers": layers, "form": form, "bytes": total, "tensors": len(wm), "seconds": time.time() - t0}


def check_names(out: str, mtp: bool) -> None:
    """The written names against qwen4exp_ref.expected_names, both ways (needs transformers 5.19.0)."""
    r = ref()
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(out, "qwen4exp_synth.json"), encoding="utf-8") as f:
        form = json.load(f)["form"]
    tc = r.text_config(out, raw=raw)
    with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
        have = set(json.load(f)["weight_map"])
    want = set(r.expected_names(tc, form, mtp=mtp))
    if want != have:
        raise SystemExit(f"make_synth: names differ from qwen4exp_ref.expected_names: missing {sorted(want - have)[:5]}, "
                         f"extra {sorted(have - want)[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--form", choices=("ours", "intel"), required=True)
    ap.add_argument("--tokenizer", required=True, help="the original snapshot's small files (tokenizer.json, config.json, ...)")
    ap.add_argument("--config", help="the original's config.json (default: <tokenizer>/config.json)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ple-base", type=int, default=1000)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = make(a.out, a.form, a.tokenizer, a.config, a.layers, a.ple_base, a.mtp, a.seed)
    check_names(a.out, a.mtp)
    print(f"wrote {a.out}: {r['tensors']} tensors, {r['bytes'] / 1e9:.2f} GB, {a.layers} layers, form {a.form}, "
          f"PLE base {a.ple_base}{', MTP' if a.mtp else ''}, {r['seconds']:.0f}s; names = qwen4exp_ref.expected_names")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
