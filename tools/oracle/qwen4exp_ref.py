#!/usr/bin/env python3
"""Spec 21a: the CPU reference of Qwen3.8-Flash-Next (`qwen4_exp`), F1.

The model is transformers 5.19.0's own `Qwen4ExpForCausalLM` (decision 1; refused on any other
version) built on `meta` from the checkpoint's text config and LAYER-STREAMED (stream.py: the
embedding, lm_head and final mixer resident, each decoder layer materialised by a forward pre-hook
and dropped after). Three pieces Ornith's reference did not need:

  1. the PLE n-gram table is never materialised (102.4 GB bf16): `ple.ple_embedding.ngram_embedding`
     is replaced by a lookup that reads the 16 rows a token by mmap - from the checkpoint's shards
     (`ngram_embedding.shard_K.weight`) or from 21b's int8 file (`--ple int8:<dir>`, the engine-format
     reference). transformers' own hash computes the row ids (the buffers come from the checkpoint);
     a gather has no rounding, so the forward is the materialised model's.
  2. the QSA indexer caches each complete block's compressed key (fp32 mean of its 4 raw keys ->
     bf16 -> k_layernorm -> RoPE at the block's first position, M:733-742) once per cache instead of
     recomputing every block for every query (transformers' per-query loop, O(T^2) per layer). The ops
     per block are the same ops, and the per-query scores are the same matmul at the same shapes, so
     the selection and the attention are transformers' bit for bit (test_qwen4exp_ref.py).
  3. routed experts are dequantised on demand (Ornith's LazyExperts) from any of the forms: the
     original's fused bf16 `mlp.experts.gate_up_proj` / `down_proj` (sliced per expert), Intel's /
     ours per-expert int4 `qweight` / `scales` (g128 or g64, stream.dequant_t: bf16(float(q - 8) x
     float(scale))), the tiny model's per-expert `.weight`.

Both name forms are read: the real checkpoints' `model.language_model.*` (Qwen4ExpForConditional-
Generation) and the tiny model's `model.layers.*` (qwen4_exp_text); `mtp.*` and `model.visual.*` are
not part of the text model (the MTP head is qwen4exp_mtp.py).

    qwen4exp_ref.py run <snapshot> --prompt <ids> --out <p>.golden.safetensors [--gen 32] [--layers N]
                        [--ple bf16|bf16:<snapshot>|int8:<dir>] [--chunk C] [--logits-tail L] [--act-tail A]
    qwen4exp_ref.py ppl <snapshot> [--text NAME=FILE ...] [--ids NAME=FILE ...] [--layers N] [--max-tokens T]
    qwen4exp_ref.py trace <snapshot> --source NAME:IDS [...] [--opencode DIR] --out <dir> [--layers N]
    qwen4exp_ref.py hfcheck <snapshot> --layers N [--prompt <ids>] [--n 2100] [--gen 4]
    qwen4exp_ref.py facts <snapshot> [--qwen38-tokenizer FILE]

`run`'s golden layout (bf16 activations are the rows from `act_row0`, metadata, to the end; logits from
`logits_row0`):
    H.L<l>        bf16 [A][hc*H]   the 4-stream residual after layer l (prompt tail + every generated row)
    mixer.L<l>    bf16 [A][H]      the GDN / QSA output
    moe.L<l>      bf16 [A][H]      the MoE block's output (routed + gated shared)
    route.ids.L<l> i32 [T+gen][k]  the router's top-k in ITS order (descending p: the combine's slot order)
    route.w.L<l>  f32  [T+gen][k]  the renormalised weights as the experts get them (bf16, widened)
    route.gap.L<l> f32 [T+gen]     p(k) - p(k+1) of the fp32 softmax (0 = an exact tie at the cut)
    qsa.sel.L<l>  i32  [R][512]    QSA layers: rows p >= 2051 (qsa_row0), the selected block ids ascending
    qsa.gap.L<l>  f32  [T+gen]     the 512th - 513th block score (+inf where <= 512 blocks are complete)
    ple.ids       i64  [T+gen][16] the 16 n-gram rows each token read
    logits        f32  [Lr][V]     rows logits_row0 .. T+gen-1
    tokens        i32  [gen]       greedy;   nll f32 [T-1] the prompt's teacher-forced NLL
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
DEFAULT_EXPERTS = "grouped_mm"       # 5.19.0's default for Qwen4ExpTextExperts (Task 1, qwen4exp_tiny_check.py)
QSA_DENSE_ROWS = 2051                # rows p <= 2050 see <= 2051 positions: QSA selects all of them (spec 21 §2.3)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"q4exp_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stream = _load("stream")
facts = _load("qwen4exp_facts")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def require_version() -> None:
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(f"qwen4exp_ref restates transformers {PINNED_TRANSFORMERS}'s qwen4_exp; this is "
                           f"{transformers.__version__} ({os.path.dirname(transformers.__file__)}) - put the 5.19.0 "
                           f"site first on PYTHONPATH (tools/oracle/qwen4exp_env.sh)")


def mq():
    require_version()
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    return m


# ------------------------------------------------------------------------------------------------
# the config

def read_config(snapshot: str) -> dict:
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def text_config_dict(raw: dict) -> dict:
    d = dict(raw["text_config"] if "text_config" in raw else raw)
    d.pop("quantization_config", None)
    return d


def text_config(snapshot: str, layers: int | None = None, experts: str = DEFAULT_EXPERTS,
                dtype: torch.dtype = torch.bfloat16, raw: dict | None = None):
    """config.json's text_config (or the tiny's top level) -> the Qwen4ExpTextConfig the reference builds.

    layers=N truncates num_hidden_layers and layer_types[:N] (spec 21's --layers N: the engine loads
    layers [0, N), then the final mixer and the head). ple_layer_ids are ONE-indexed (M:1260), so N must
    keep them inside: N >= max(ple_layer_ids) (C:239-245)."""
    require_version()
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    d = text_config_dict(raw if raw is not None else read_config(snapshot))
    if layers is not None:
        n = d["num_hidden_layers"]
        ple = d.get("ple_layer_ids") or []
        if not 1 <= layers <= n:
            die(f"--layers {layers}: the checkpoint has {n}")
        if ple and layers < max(ple):
            die(f"--layers {layers} drops the PLE layer: ple_layer_ids {ple} are one-indexed (layer_idx "
                f"{max(ple) - 1}), so N >= {max(ple)}")
        d["num_hidden_layers"] = layers
        if d.get("layer_types") is not None:
            d["layer_types"] = list(d["layer_types"][:layers])
    tc = Qwen4ExpTextConfig(**d)
    if tc.model_type != "qwen4_exp_text":
        die(f"text config model_type {tc.model_type!r}, expected qwen4_exp_text")
    if tc.tie_word_embeddings:
        die("tie_word_embeddings is true; Qwen3.8-Flash-Next ships a separate lm_head")
    tc._attn_implementation = "eager"          # fp32 softmax over the masked row: the contract (doc 03)
    tc._experts_implementation = experts
    tc.dtype = dtype
    return tc


def require_cacheable(tc) -> None:
    """A cached run (run / ppl / hfcheck / trace) needs an indexed_attention layer among the first N:
    transformers 5.19.0's DynamicCache takes the sequence length from an attention layer
    (cache_utils.py:1566), so --layers N < 4 can only be run uncached."""
    if "indexed_attention" not in tc.layer_types:
        die(f"--layers {tc.num_hidden_layers}: no indexed_attention layer among them - transformers' DynamicCache "
            f"needs one for the sequence length (cache_utils.py:1566); a cached run needs N past the first QSA "
            f"layer (N >= 4 on this model)")


def lm_prefix(names) -> str:
    """The text model's checkpoint prefix: the real checkpoints' `model.language_model.`, the tiny's `model.`."""
    return "model.language_model." if any(n.startswith("model.language_model.") for n in names) else "model."


def model_name(ckpt: str) -> str:
    return "model." + ckpt[len("model.language_model."):] if ckpt.startswith("model.language_model.") else ckpt


# ------------------------------------------------------------------------------------------------
# expected names: what 21b's synthetic checkpoints write and the loader requires

GDN_LINEARS = ("in_proj_qkv", "in_proj_z", "out_proj")
QSA_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
SHARED_LINEARS = ("gate_proj", "up_proj", "down_proj")
INT4 = (".qweight", ".qzeros", ".scales")


def expected_names(tc, form: str, mtp: bool = True) -> list[str]:
    """Every text-model tensor of a checkpoint in `form` (vision excluded):
      "bf16"  the original: fused routed experts `mlp.experts.gate_up_proj` / `down_proj`, all bf16
      "intel" Intel's AutoRound export: per-expert int4 `qweight` / `qzeros` / `scales` (g128), the rest bf16
              as shipped; its MTP head bf16 with per-expert `.weight` experts
      "ours"  spec 21 §5 (21q): per-expert int4 g64 experts, and int4 for the shared experts, QSA q/k/v/o and
              GDN in_proj_qkv / in_proj_z / out_proj; everything else bf16 (ple ignored); MTP as Intel's
              (decision 6 open)
      "tiny"  qikp/tiny-random-...: `model.layers.*` prefix, per-expert `.weight` experts
    `mtp` adds the head's `mtp.*` (the original's and Intel's sets; the tiny has none)."""
    if form not in ("bf16", "intel", "ours", "tiny"):
        raise ValueError(form)
    P = "model." if form == "tiny" else "model.language_model."
    E = tc.num_experts
    parts = getattr(tc, "split_ngram_parts", 512)

    def lin(base: str, quant: bool) -> list[str]:
        return [base + s for s in INT4] if quant else [base + ".weight"]

    def hc(base: str, inject: bool) -> list[str]:
        out = [base + ".hc_norm.weight", base + ".input_mix_weight_down.weight", base + ".input_mix_weight_up.weight"]
        return out + ([base + ".block_inject_weight.weight"] if inject else [])

    def moe(base: str, experts_form: str, shared_q: bool) -> list[str]:
        out = [base + "mlp.gate.weight", base + "mlp.shared_expert_gate.weight"]
        for s in SHARED_LINEARS:
            out += lin(f"{base}mlp.shared_expert.{s}", shared_q)
        if experts_form == "fused":
            out += [base + "mlp.experts.gate_up_proj", base + "mlp.experts.down_proj"]
        else:
            for e in range(E):
                for s in SHARED_LINEARS:
                    out += lin(f"{base}mlp.experts.{e}.{s}", experts_form == "int4")
        return out

    def qsa(base: str, quant: bool) -> list[str]:
        out = [base + "self_attn.q_norm.weight", base + "self_attn.k_norm.weight",
               base + "self_attn.indexer.index_qk_proj.weight", base + "self_attn.indexer.q_layernorm.weight",
               base + "self_attn.indexer.k_layernorm.weight"]
        for s in QSA_LINEARS:
            out += lin(f"{base}self_attn.{s}", quant)
        return out

    q_dense = form == "ours"
    ex_form = {"bf16": "fused", "intel": "int4", "ours": "int4", "tiny": "weight"}[form]
    names = [P + "embed_tokens.weight", "lm_head.weight"] + hc(P + "hyper_connection_mixer", False)
    ple_layers = {i - 1 for i in tc.ple_layer_ids}
    for i, t in enumerate(tc.layer_types):
        b = f"{P}layers.{i}."
        names += hc(b + "attn_hyper_connection", True) + hc(b + "mlp_hyper_connection", True)
        names += moe(b, ex_form, q_dense)
        if t == "linear_attention":
            names += [b + "linear_attn.A_log", b + "linear_attn.dt_bias", b + "linear_attn.conv1d.weight",
                      b + "linear_attn.norm.weight", b + "linear_attn.in_proj_a.weight", b + "linear_attn.in_proj_b.weight"]
            for s in GDN_LINEARS:
                names += lin(f"{b}linear_attn.{s}", q_dense)
        else:
            names += qsa(b, q_dense)
        if i in ple_layers:
            p = b + "ple."
            names += [p + "conv1d.weight", p + "key_proj.weight", p + "value_proj.weight", p + "norm_conv.weight",
                      p + "norm_key.weight", p + "norm_query.weight", p + "ple_embedding.layer_multipliers",
                      p + "ple_embedding.ngram_heads_vocab_sizes", p + "ple_embedding.ngram_heads_offsets"]
            names += [f"{p}ple_embedding.ngram_embedding.shard_{k}.weight" for k in range(parts)]
    if mtp and form != "tiny":
        b = "mtp.layers.0."
        names += ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
                  "mtp.pre_fc_norm_hidden.weight"] + hc("mtp.hyper_connection_mixer", False)
        names += hc(b + "attn_hyper_connection", True) + hc(b + "mlp_hyper_connection", True)
        names += moe(b, "fused" if form == "bf16" else "weight", False) + qsa(b, False)
    return names


# ------------------------------------------------------------------------------------------------
# the checkpoint

class Ckpt:
    """Name -> tensor over a snapshot's shards (index or a single model.safetensors), plus raw mmaps."""

    def __init__(self, snapshot: str):
        from safetensors import safe_open
        self.dir = snapshot
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                self.wm = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(snapshot, "model.safetensors"), framework="pt") as h:
                self.wm = {k: "model.safetensors" for k in h.keys()}
        self.h = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
                  for fn in sorted(set(self.wm.values()))}
        self._hdr: dict[str, dict] = {}

    def __contains__(self, k: str) -> bool:
        return k in self.wm

    def keys(self):
        return self.wm.keys()

    def get(self, k: str) -> torch.Tensor:
        return self.h[self.wm[k]].get_tensor(k)

    def get_rows(self, k: str, a: int, b: int) -> torch.Tensor:
        return self.h[self.wm[k]].get_slice(k)[a:b]

    def mmap(self, k: str) -> np.ndarray:
        """The tensor's bytes as a read-only numpy memmap (bf16 as uint16): rows by fancy indexing."""
        fn = self.wm[k]
        if fn not in self._hdr:
            with open(os.path.join(self.dir, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[fn] = dict(json.loads(f.read(n)), __start__=8 + n)
        e, s = self._hdr[fn][k], self._hdr[fn]["__start__"]
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I8": np.int8, "I64": np.int64}[e["dtype"]]
        a, _ = e["data_offsets"]
        return np.memmap(os.path.join(self.dir, fn), dtype=dt, mode="r", offset=s + a, shape=tuple(e["shape"])), e["dtype"]


def _cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype) if t.is_floating_point() and t.dtype != dtype else t


def _dequant(ck: Ckpt, base: str, checked: set) -> torch.Tensor:
    """`base`.{qweight, scales, qzeros} -> the nn.Linear weight [N][K] bf16: bf16(float(q - 8) x float(scale)),
    the group size from the scales' shape (64 ours, 128 Intel's experts). qzeros must be the sym constant."""
    qw, sc = ck.get(base + ".qweight"), ck.get(base + ".scales")
    g = qw.shape[0] * 8 // sc.shape[0]
    if base not in checked:
        qz = ck.get(base + ".qzeros")
        if not bool((qz == 0x77777777).all()):
            die(f"{base}.qzeros is not 0x77777777: not symmetric int4 (the engine's rule is (q - 8) x scale)")
    out = _stream.dequant_t(qw, sc, g, check=base not in checked)
    checked.add(base)
    return out


def layer_state(ck: Ckpt, keys, dtype: torch.dtype, checked: set) -> dict[str, torch.Tensor]:
    """Checkpoint names -> model tensors (int4 dequantised, floats cast to the model's dtype)."""
    sd = {}
    for k in keys:
        if k.endswith((".scales", ".qzeros")):
            continue
        if k.endswith(".qweight"):
            b = k[: -len(".qweight")]
            sd[model_name(b) + ".weight"] = _cast(_dequant(ck, b, checked), dtype)
        else:
            sd[model_name(k)] = _cast(ck.get(k), dtype)
    return sd


class LazyExperts:
    """ornith_ref.LazyExperts for this family: one shared [E][2I][H] / [E][H][I] pair, a layer's routed
    experts filled on demand by a pre-hook on `mlp.experts` (it receives top_k_index). Neither experts
    implementation reads an expert no token routes to, so the result is the materialised model's.
    Forms per layer: fused bf16 (sliced), per-expert int4 (dequant_t), per-expert `.weight` (the tiny)."""

    def __init__(self, ck: Ckpt, tc, prefix: str, keep=(), dtype=None):
        self.ck, self.tc, self.prefix = ck, tc, prefix
        self.dtype = dtype or tc.dtype
        E, I, H = tc.num_experts, tc.moe_intermediate_size, tc.hidden_size
        self.I = I
        self.shape = ((E, 2 * I, H), (E, H, I))
        self.keep = set(keep)
        self.gate_up = torch.empty(*self.shape[0], dtype=self.dtype)
        self.down = torch.empty(*self.shape[1], dtype=self.dtype)
        self.own: dict[int, tuple] = {}
        self.lock = threading.Lock()
        self.filled = 0
        self.checked: set[str] = set()
        self.layer: int | None = None
        self.have: set[int] = set()

    def tensors(self, layer: int):
        if layer not in self.keep:
            return self.gate_up, self.down
        with self.lock:
            if layer not in self.own:
                self.own[layer] = (torch.empty(*self.shape[0], dtype=self.dtype),
                                   torch.empty(*self.shape[1], dtype=self.dtype), set())
            return self.own[layer][:2]

    def buffers(self, layer: int):
        if layer in self.keep:
            self.tensors(layer)
            return self.own[layer]
        if layer != self.layer:
            self.layer, self.have = layer, set()
        return self.gate_up, self.down, self.have

    def expert(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [2I][H] gate rows first, down [H][I]) of expert e, in the model's dtype."""
        b = f"{self.prefix}{layer}.mlp.experts."
        ck, I = self.ck, self.I
        if b + "gate_up_proj" in ck:                      # the original: fused, gate then up (M:943)
            gu = ck.get_rows(b + "gate_up_proj", e, e + 1)[0]
            dn = ck.get_rows(b + "down_proj", e, e + 1)[0]
            return _cast(gu, self.dtype), _cast(dn, self.dtype)
        if f"{b}{e}.gate_proj.qweight" in ck:
            g, u, d = (_dequant(ck, f"{b}{e}.{s}", self.checked) for s in SHARED_LINEARS)
        elif f"{b}{e}.gate_proj.weight" in ck:
            g, u, d = (ck.get(f"{b}{e}.{s}.weight") for s in SHARED_LINEARS)
        else:
            die(f"{b}{e}.gate_proj: neither fused, int4 nor .weight in the checkpoint")
        if g.shape[0] != I:
            die(f"{b}{e}.gate_proj has {g.shape[0]} rows, expected {I}")
        return _cast(torch.cat([g, u], 0), self.dtype), _cast(d, self.dtype)

    def fill(self, layer: int, ids) -> None:
        gate_up, down, have = self.buffers(layer)
        for e in ids:
            if e in have:
                continue
            have.add(e)
            gate_up[e], down[e] = self.expert(layer, e)
            self.filled += 1

    def hook(self, layer: int):
        def fn(_mod, args):
            self.fill(layer, sorted(set(int(x) for x in args[1].reshape(-1).tolist())))
        return fn


# ------------------------------------------------------------------------------------------------
# PLE: the table read by mmap, and the hash restated in exact integers

def ple_hash_ids(tokens: torch.Tensor, multipliers, sizes, offsets, eos: int, ngram: int = 3,
                 heads_per_ngram: int = 8, prev: torch.Tensor | None = None,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """The 16 n-gram row ids of each of `tokens` (int64 [T] -> [T, 16]), restated from M:1107-1166.

    History: token t's predecessor at distance s (1, 2) is used only if it exists and lies AFTER the last
    EOS strictly before t; otherwise it reads as EOS (M:1107-1121). `prev` are the ids before `tokens`
    (a cached decode passes them; transformers keeps the last 2, which is equivalent: an EOS further back
    can only hide predecessors that are themselves further back). `mask` [T] (0 = padding) maps padded
    ids to EOS first, as Qwen4ExpTextModel does (M:1464-1468). mixed = (t0 m0) XOR (t1 m1) [XOR (t2 m2)]
    in int64 (no overflow: m <= (2^63 - 1) // vocab), id = mixed mod prime_h + offset_h."""
    tokens = tokens.long()
    if mask is not None:
        tokens = torch.where(mask.bool(), tokens, torch.full_like(tokens, eos))
    seq = tokens if prev is None else torch.cat([prev.long(), tokens])
    n0 = seq.numel() - tokens.numel()
    pos = torch.arange(seq.numel())
    eos_pos = torch.where(seq == eos, pos, torch.full_like(pos, -1))
    last_incl = torch.cummax(eos_pos, 0).values
    last_before = torch.cat([torch.full((1,), -1, dtype=torch.long), last_incl[:-1]])
    shifted = []
    for s in range(ngram):
        src = pos - s
        ok = (src >= 0) & (src > last_before)
        shifted.append(torch.where(ok, seq[src.clamp_min(0)], torch.full_like(seq, eos)))
    m = torch.as_tensor(multipliers, dtype=torch.long)
    sizes = torch.as_tensor(sizes, dtype=torch.long)
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    blocks = []
    for n in range(2, ngram + 1):
        a = (n - 2) * heads_per_ngram
        mixed = shifted[0] * m[0]
        for p in range(1, n):
            mixed = torch.bitwise_xor(mixed, shifted[p] * m[p])
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[a:a + heads_per_ngram]) + offsets[a:a + heads_per_ngram])
    return torch.cat(blocks, -1)[n0:]


class PleTable:
    """16 rows a token by mmap. `source`: a snapshot directory whose PLE layer stores the table as
    `ngram_embedding.shard_K.weight` (bf16, or the tiny's fp32), or "int8:<dir>" - 21b Task 1's file:
    per head h `ple.h<h>.q` I8 [prime_h][160] and `ple.h<h>.s` F32 or BF16 [prime_h], an index json; a row
    dequantises to bf16(float(q) x float(s)) (spec 9's row rule, the engine-format reference)."""

    def __init__(self, source: str, tc):
        if len(tc.ple_layer_ids) != 1:
            raise ValueError(f"one PLE layer expected, got ple_layer_ids {tc.ple_layer_ids}")
        d = tc.to_dict()
        self.sizes, self.offsets, self.total, self.padded = facts.head_table(d, 0)
        self.multipliers = facts.layer_multipliers(tc.vocab_size, tc.ngram_size, 0, tc.seed)
        self.eos = tc.eos_token_id[0] if isinstance(tc.eos_token_id, list) else tc.eos_token_id
        self.heads = (tc.ngram_size - 1) * tc.heads_per_ngram
        self.ngram, self.heads_per_ngram = tc.ngram_size, tc.heads_per_ngram
        self.dim = tc.ple_embed_dim // self.heads
        self.reads = 0
        if source.startswith("int8:"):
            self.kind = "int8"
            ck = Ckpt(source[5:])
            self.q = [ck.mmap(f"ple.h{h}.q")[0] for h in range(self.heads)]
            self.s = [ck.mmap(f"ple.h{h}.s") for h in range(self.heads)]
            for h in range(self.heads):
                if self.q[h].shape != (self.sizes[h], self.dim):
                    raise ValueError(f"ple.h{h}.q is {self.q[h].shape}, expected ({self.sizes[h]}, {self.dim})")
            self.out_dtype = torch.bfloat16
        else:
            self.kind = "shards"
            ck = Ckpt(source)
            pre = lm_prefix(ck.keys())
            base = f"{pre}layers.{tc.ple_layer_ids[0] - 1}.ple.ple_embedding.ngram_embedding.shard_"
            parts = tc.split_ngram_parts
            self.shards = []
            for k in range(parts):
                arr, dt = ck.mmap(f"{base}{k}.weight")
                self.shards.append(arr)
            self.rows_per = self.padded // parts
            if any(a.shape != (self.rows_per, self.dim) for a in self.shards):
                raise ValueError(f"PLE shards are not {parts} x [{self.rows_per}, {self.dim}]")
            self.src_dtype = dt
            self.out_dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[dt]

    def ids(self, history: torch.Tensor, multipliers=None, sizes=None, offsets=None, prev=None, mask=None):
        """ple_hash_ids with this table's constants unless given (the checkpoint's I64 tensors)."""
        return ple_hash_ids(history, self.multipliers if multipliers is None else multipliers,
                            self.sizes if sizes is None else sizes, self.offsets if offsets is None else offsets,
                            self.eos, self.ngram, self.heads_per_ngram, prev=prev, mask=mask)

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """int64 [..., 16] global row ids -> [..., 16, dim] (the shards' dtype, or bf16 from int8)."""
        flat = ids.reshape(-1, self.heads).long()
        out = torch.empty(flat.shape[0], self.heads, self.dim, dtype=self.out_dtype)
        self.reads += flat.numel()
        if self.kind == "shards":
            g = flat.reshape(-1).numpy()
            sh, lo = g // self.rows_per, g % self.rows_per
            res = np.empty((g.size, self.dim), dtype=self.shards[0].dtype)
            for k in np.unique(sh):
                sel = np.nonzero(sh == k)[0]
                res[sel] = self.shards[int(k)][lo[sel]]
            t = torch.from_numpy(res)
            if self.src_dtype == "BF16":
                t = t.view(torch.bfloat16)
            out.copy_(t.view(-1, self.heads, self.dim))
        else:
            for h in range(self.heads):
                local = (flat[:, h] - self.offsets[h]).numpy()
                if (local < 0).any() or (local >= self.sizes[h]).any():
                    raise ValueError(f"PLE head {h}: a row id outside [offset, offset + prime)")
                q = torch.from_numpy(np.asarray(self.q[h][local])).float()
                s_arr, s_dt = self.s[h]
                s = torch.from_numpy(np.asarray(s_arr[local]))
                s = s.view(torch.bfloat16).float() if s_dt == "BF16" else s.float()
                out[:, h] = (q * s.unsqueeze(-1)).to(torch.bfloat16)
        return out.view(*ids.shape, self.dim)


class PleLookup(torch.nn.Module):
    """Stands in for `ngram_embedding` (nn.Embedding [320001536, 160]): rows from a PleTable, cast to
    the model's dtype. `weight` is a plain attribute (not a parameter), read by M:1169 for its device."""

    def __init__(self, table: PleTable, dtype: torch.dtype):
        super().__init__()
        self.table, self.dtype = table, dtype
        self.weight = torch.empty(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table.rows(ids).to(self.dtype)


# ------------------------------------------------------------------------------------------------
# the QSA indexer with cached compressed block keys

def _causal_visible(q_len: int, kv: int) -> torch.Tensor:
    past = kv - q_len
    return torch.arange(kv).unsqueeze(0) <= (torch.arange(q_len) + past).unsqueeze(1)


def block_keys(indexer, raw: torch.Tensor, starts_cos: torch.Tensor, starts_sin: torch.Tensor) -> torch.Tensor:
    """Complete blocks' compressed keys, M:733-742: raw [n*C, D] -> mean of C in fp32 -> raw dtype ->
    k_layernorm -> RoPE at each block's first position (cos / sin [n, rot] already gathered)."""
    m = mq()
    C, D = indexer.compress_ratio, indexer.index_head_dim
    groups = raw.view(-1, C, D)
    pooled = groups.float().mean(dim=1).to(raw.dtype)
    pooled = indexer.k_layernorm(pooled)
    return m.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=starts_cos, sin=starts_sin).squeeze(1)


def cached_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Qwen4ExpTextQSAIndexer.forward (M:685-771) for batch 1 with causal visibility, each complete
    block's compressed key computed once and kept beside the cache. Optional per-instance attributes:
    `_q4_rec(layer_idx, pos, blocks, gap)` records each row; `_q4_force` (int32 [q][W] token lists)
    replaces the selection (the MTP head's skip_topk); `_q4_last` receives this call's token lists."""
    m = mq()
    B, q_len, _ = hidden_states.shape
    if B != 1:
        raise RuntimeError("the cached indexer is batch 1 (one sequence per item)")
    D, C = self.index_head_dim, self.compress_ratio
    full_cos, full_sin = position_embeddings
    cur_cos, cur_sin = full_cos[:, -q_len:, :], full_sin[:, -q_len:, :]
    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [self.index_n_heads * D, self.index_kv_heads * D], dim=-1)
    q, raw_keys = q.reshape(B, q_len, -1, D), token_k.reshape(B, q_len, -1, D).squeeze(2)
    q = self.q_layernorm(q)
    q = m.apply_rotary_pos_emb(q, cos=cur_cos, sin=cur_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        store = past_key_values.layers[self.layer_idx]
    else:
        store = types.SimpleNamespace()
    vis = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    kv = vis.shape[-1]
    past = kv - q_len
    if raw_keys.shape[1] != kv or not torch.equal(vis[0, 0], _causal_visible(q_len, kv)):
        raise RuntimeError("the cached indexer supports causal visibility only (no padding, no packing)")
    blocks = getattr(store, "_q4_blocks", None)
    have = 0 if blocks is None else blocks.shape[0]
    n_max = kv // C
    if n_max > have:
        starts = torch.arange(have, n_max) * C
        new = block_keys(self, raw_keys[0, have * C:n_max * C], full_cos[0].index_select(0, starts),
                         full_sin[0].index_select(0, starts))
        blocks = new if blocks is None else torch.cat([blocks, new], 0)
        store._q4_blocks = blocks
    W = self.token_budget + C - 1
    selected = torch.full((B, q_len, W), -1, dtype=torch.int32, device=hidden_states.device)
    force = getattr(self, "_q4_force", None)
    rec = getattr(self, "_q4_rec", None)
    for r in range(q_len):
        p = past + r
        nvis = p + 1
        n = nvis // C
        if force is not None:
            row = force[r]
            selected[0, r, :] = row
            continue
        if n > 0:
            scores = torch.matmul(q[0, r].float(), blocks[:n].float().transpose(-1, -2)).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(D)
            sel = scores.topk(min(self.block_topk, n), dim=0).indices
            toks = torch.arange(n * C).view(n, C).index_select(0, sel).flatten()
        else:
            sel = None
            toks = torch.tensor([], device=hidden_states.device)
        tail = torch.arange(n * C, nvis)
        toks = torch.cat([toks, tail]).to(torch.int32)
        selected[0, r, : toks.numel()] = toks
        if rec is not None:
            if n > self.block_topk:
                v = scores.topk(self.block_topk + 1, dim=0).values
                gap = float(v[-2] - v[-1])
                blk = sel.sort().values.to(torch.int32)
            else:
                gap, blk = math.inf, None
            rec(self.layer_idx, p, blk, gap)
    self._q4_last = selected[0].clone()
    kv_length = attention_mask.shape[-1]
    mask = torch.zeros((*selected.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool)
    scatter = torch.where(selected >= 0, selected, kv_length)
    mask = mask.scatter(-1, scatter.long(), True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        mask = torch.where(mask, attention_mask.new_zeros(()), min_dtype)
    return mask


def patch_indexers(model_or_layers) -> list:
    """Replace every QSA indexer's forward with cached_indexer_forward; returns the indexers."""
    m = mq()
    out = []
    for mod in model_or_layers.modules():
        if isinstance(mod, m.Qwen4ExpTextQSAIndexer):
            mod.forward = types.MethodType(cached_indexer_forward, mod)
            out.append(mod)
    return out


# ------------------------------------------------------------------------------------------------
# the streamed model

def build_streamed(snapshot: str, tc, ple: "PleTable | None" = None, keep_dense=(), keep_experts=(),
                   own_indexer: bool = False):
    """Qwen4ExpForCausalLM on meta; embed / lm_head / final mixer resident; decoder layers streamed;
    routed experts lazy; the PLE table by mmap; QSA indexers cached (own_indexer: transformers' own
    per-query recomputation instead - hfcheck's other arm). Returns (model, prefetcher, lazy experts)."""
    m = mq()
    ck = Ckpt(snapshot)
    pre = lm_prefix(ck.keys())
    with torch.device("meta"):
        model = m.Qwen4ExpForCausalLM(tc)
    ple = ple or PleTable(snapshot, tc)
    for layer in model.model.layers:
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = PleLookup(ple, tc.dtype)
    if not own_indexer:
        patch_indexers(model)
    n = tc.num_hidden_layers
    per: dict[int, list[str]] = {i: [] for i in range(n)}
    rest = []
    for key in sorted(ck.keys()):
        if key.startswith(("mtp.", "model.visual.")):
            continue
        mm = re.match(re.escape(pre) + r"layers\.(\d+)\.", key)
        if mm:
            i = int(mm.group(1))
            if i >= n or ".mlp.experts." in key or ".ngram_embedding.shard_" in key:
                continue
            per[i].append(key)
        else:
            rest.append(key)
    checked: set = set()
    resident = layer_state(ck, rest, tc.dtype, checked)
    lazy = LazyExperts(ck, tc, pre + "layers.", keep_experts)

    def layer_sd(i: int) -> dict[str, torch.Tensor]:
        sd = layer_state(ck, per[i], tc.dtype, checked)
        lp = f"model.layers.{i}."
        bad = [k for k in sd if not k.startswith(lp)]
        if bad:
            raise RuntimeError(f"layer {i}: {bad[:3]} outside {lp}")
        out = {k[len(lp):]: v for k, v in sd.items()}
        out["mlp.experts.gate_up_proj"], out["mlp.experts.down_proj"] = lazy.tensors(i)
        return out

    layers = list(model.model.layers)
    pf = _stream.attach(model, layers, layer_sd, resident,
                        rebuild=[(model.model, "rotary_emb", lambda: type(model.model.rotary_emb)(tc))],
                        keep=keep_dense)
    for i, layer in enumerate(layers):
        layer.mlp.experts.register_forward_pre_hook(lazy.hook(i))
    model.q4_ple, model.q4_ckpt = ple, ck
    return model.eval(), pf, lazy


def hf_reference(snapshot: str, dtype: torch.dtype, num_layers: int | None = None, experts: str = DEFAULT_EXPERTS):
    """transformers' own from_pretrained model, un-streamed (the tiny / test sizes only)."""
    m = mq()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot)
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers
        cfg.layer_types = cfg.layer_types[:num_layers]
    model = m.Qwen4ExpForCausalLM.from_pretrained(snapshot, config=cfg, dtype=dtype, attn_implementation="eager")
    model.config._experts_implementation = experts
    return model.eval()


# ------------------------------------------------------------------------------------------------
# the restated ops (the rounding chain the facts sheet pins; 21c's fixture imports these)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """M:144-169: fp32, x * rsqrt(mean(x^2) + eps), times (1 + w) in fp32, ONE cast to x's dtype."""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        out = out.flatten(-2)
    return (out * (1.0 + w.float())).type_as(x)


def gated_residual(H: torch.Tensor, norm_w, down_w, up_w, inject_w, hc: int, hidden: int, eps: float):
    """M:1006-1023 restated: xn = grouped (1 + w) norm (group = hidden); g = sigmoid(up(silu(down(xn) / hc)))
    with a rounding after the linear, the division and each activation; mixed = mean_s(g * xn) (the
    product rounded, the mean in fp32 rounded once); inj = 2 * sigmoid(inject(xn) / hc). Returns mixed,
    or (mixed, H, inj) with the inject rows."""
    xn = rms_norm(H, norm_w, eps, group=hidden)
    g = F.silu(F.linear(xn, down_w) / hc)
    g = torch.sigmoid(F.linear(g, up_w))
    mixed = (g.unflatten(-1, (hc, hidden)) * xn.unflatten(-1, (hc, hidden))).mean(dim=-2)
    if inject_w is None:
        return mixed
    return mixed, H, 2 * torch.sigmoid(F.linear(xn, inject_w) / hc)


def hc_combine(H0: torch.Tensor, y: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
    """M:1294-1301: H0 + y (x) inj - the product rounded to the dtype, then the add."""
    return H0 + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


def gdn_gated_norm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, act: str = "sigmoid"):
    """M:179-188: x-hat in fp32, cast to x's dtype, times w in that dtype (rounded), times act(z) in fp32,
    one cast. act = config.output_gate_type (sigmoid for this model, M:492)."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = w * xf.to(dt)
    y = y * (torch.sigmoid(z.float()) if act == "sigmoid" else F.silu(z.float()))
    return y.to(dt)


def router(x: torch.Tensor, w: torch.Tensor, k: int, norm_topk: bool = True):
    """M:961-970: logits = x W^T (dtype), softmax over E in fp32, topk, renormalised in fp32, cast."""
    logits = F.linear(x, w)
    p = F.softmax(logits, dtype=torch.float, dim=-1)
    v, i = torch.topk(p, k, dim=-1)
    if norm_topk:
        v = v / v.sum(dim=-1, keepdim=True)
    return logits, v.to(logits.dtype), i


# ------------------------------------------------------------------------------------------------
# recording

class Recorder:
    """Per layer and row (golden layout, module docstring); `act_from` = the first absolute position whose
    activations are kept. Rows are bucketed by `key` (layer_major's item) - 0 for a sequential run."""

    def __init__(self, model, tc, act_from: int = 0, acts: bool = True, want_p: bool = False):
        self.tc, self.k, self.n = tc, tc.num_experts_per_tok, tc.num_hidden_layers
        self.act_from, self.acts, self.want_p = act_from, acts, want_p
        self.key = 0
        self.base: dict = {}             # key -> the absolute position of the current forward's row 0
        self.rows: dict = {}
        self.qsa: dict = {}
        self.hooks = []
        self.indexers = []
        for i, layer in enumerate(model.model.layers):
            mixer = layer.linear_attn if tc.layer_types[i] == "linear_attention" else layer.self_attn
            if acts:
                self.hooks += [layer.register_forward_hook(self._act(f"H.L{i}")),
                               mixer.register_forward_hook(self._act(f"mixer.L{i}")),
                               layer.mlp.register_forward_hook(self._act(f"moe.L{i}"))]
            self.hooks.append(layer.mlp.gate.register_forward_hook(self._router(i)))
            if layer.ple is not None:
                self.hooks.append(layer.ple.ple_embedding.ngram_embedding.register_forward_hook(
                    lambda _m, a, _o: self._put("ple.ids", a[0].detach().reshape(-1, a[0].shape[-1]).clone())))
            if tc.layer_types[i] != "linear_attention":
                layer.self_attn.indexer._q4_rec = self._qsa
                self.indexers.append(layer.self_attn.indexer)

    def remove(self):
        for h in self.hooks:
            h.remove()
        for ix in self.indexers:
            if getattr(ix, "_q4_rec", None) == self._qsa:
                ix._q4_rec = None

    def _bucket(self):
        return self.rows.setdefault(self.key, {})

    def _put(self, name, t):
        self._bucket().setdefault(name, []).append(t)

    def _act(self, name):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach()[0]
            skip = max(0, self.act_from - self.base.get(self.key, 0))
            if skip < t.shape[0]:
                self._put(name, t[skip:].to(torch.bfloat16).clone())
        return fn

    def _router(self, i):
        def fn(_m, _a, out):
            logits, scores, idx = out
            full = F.softmax(logits.detach(), dtype=torch.float, dim=-1)          # M:964
            p = full.sort(-1, descending=True).values
            self._put(f"route.ids.L{i}", idx.detach().to(torch.int32).clone())
            self._put(f"route.w.L{i}", scores.detach().float().clone())
            self._put(f"route.gap.L{i}", (p[:, self.k - 1] - p[:, self.k]).clone())
            if self.want_p:                                                        # M:965-967 before the cast
                top = full.gather(-1, idx.detach())
                if self.tc.norm_topk_prob:
                    top /= top.sum(dim=-1, keepdim=True)
                self._put(f"route.p.L{i}", top.clone())
        return fn

    def _qsa(self, layer, pos, blocks, gap):
        b = self.qsa.setdefault(self.key, {}).setdefault(layer, {"pos": [], "gap": [], "sel": []})
        b["pos"].append(pos)
        b["gap"].append(gap)
        if blocks is not None:
            row = torch.full((self.tc.indexer_budget // self.tc.indexer_compress_ratio,), -1, dtype=torch.int32)
            row[: blocks.numel()] = blocks
            b["sel"].append((pos, row))

    def tensors(self, key=0) -> dict[str, torch.Tensor]:
        out = {k: torch.cat(v, 0).contiguous() for k, v in self.rows.get(key, {}).items()}
        for layer, b in self.qsa.get(key, {}).items():
            order = sorted(range(len(b["pos"])), key=lambda j: b["pos"][j])
            out[f"qsa.gap.L{layer}"] = torch.tensor([b["gap"][j] for j in order], dtype=torch.float32)
            sel = sorted(b["sel"], key=lambda x: x[0])
            W = self.tc.indexer_budget // self.tc.indexer_compress_ratio
            out[f"qsa.sel.L{layer}"] = (torch.stack([r for _, r in sel]) if sel
                                        else torch.empty(0, W, dtype=torch.int32))
        return out


def gap_line(g: torch.Tensor, what: str) -> str:
    g = g[torch.isfinite(g)].double()
    if g.numel() == 0:
        return f"{what}: no rows past the cut"
    qs = torch.quantile(g, torch.tensor([0.0, 0.0001, 0.001, 0.01, 0.5], dtype=torch.float64)).tolist()
    return (f"{what} over {g.numel()} (row, layer): ==0 {int((g == 0).sum())}, <1e-4 {int((g < 1e-4).sum())}, "
            f"<1e-3 {int((g < 1e-3).sum())}; min/p0.01%/p0.1%/p1%/median " + " / ".join(f"{x:.3g}" for x in qs))


def gap_summary(t: dict, tc) -> list[str]:
    n = tc.num_hidden_layers
    lines = [gap_line(torch.cat([t[f"route.gap.L{i}"] for i in range(n)]),
                      f"MoE {tc.num_experts_per_tok}th/{tc.num_experts_per_tok + 1}th gap")]
    q = [i for i in range(n) if tc.layer_types[i] != "linear_attention" and f"qsa.gap.L{i}" in t]
    if q:
        lines.append(gap_line(torch.cat([t[f"qsa.gap.L{i}"] for i in q]), "QSA 512th/513th gap (all QSA layers)"))
        for i in q:
            lines.append("  " + gap_line(t[f"qsa.gap.L{i}"], f"QSA 512th/513th gap L{i}"))
    return lines


# ------------------------------------------------------------------------------------------------
# layer-major batches (oracle_generate --batch's shape; trace's engine)

LAYER_MAJOR_TRANSFORMERS = "5.19.0"


def layer_major(model, pf, rec: "Recorder | None" = None, tap=None):
    """-> step_many(items) for build_streamed's model: items = [(ids, pos, state)], state a dict whose
    "cache" is None before the first chunk. Returns per item the pre-mixer 4-stream hidden [T, hc*H]
    (the last layer's materialised output) - logits via `final_logits`. BITWISE what
    `model(input_ids=[ids], past_key_values=cache, use_cache=True)` computes: Qwen4ExpTextModel.forward's
    prologue (M:1408-1472) restated with the module's own functions, each decoder layer called with the
    arguments its loop passes (M:1474-1483). Layer-major: every item through layer 0, then layer 1, ...
    with stream.py's hold mode, so one pass reads each layer's weights once for all items. `rec.key` (and
    `tap.key`) is set to the item's key - keys[j], default j - before each of its calls."""
    m = mq()
    tm = model.model
    cfg = tm.config
    layers = list(tm.layers[: cfg.num_hidden_layers])
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id

    def prologue(ids, pos, st):
        x = torch.tensor([list(ids)])
        h = tm.embed_tokens(x)
        ple_ids = x if cfg.ple_layer_ids else None
        if st.get("cache") is None:
            st["cache"] = m.DynamicCache(config=cfg)
        cache = st["cache"]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a cache of {past} tokens")
        position_ids = torch.arange(h.shape[1], device=h.device) + past
        position_ids = position_ids.view(1, 1, -1).expand(4, h.shape[0], -1)
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
        if hasattr(cache, "position_ids"):
            position_ids = torch.cat([cache.position_ids, position_ids], dim=-1)
        cache.position_ids = position_ids
        mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": cache,
              "position_ids": text_position_ids, "allow_is_causal_skip": False}
        masks = {"indexed_attention": m.create_causal_mask(**mk),
                 "linear_attention": m.create_recurrent_attention_mask(**mk)}
        conv_mask = masks.get("linear_attention")
        if cfg.ple_layer_ids and conv_mask is not None:
            ple_ids = torch.where(conv_mask.bool(), ple_ids, eos)
        pe = tm.rotary_emb(h, position_ids)
        return {"h": h.repeat(1, 1, cfg.hc_count), "pe": pe, "masks": masks, "conv": conv_mask,
                "ple": ple_ids, "cache": cache, "past": past}

    def select(key, past):
        if rec is not None:
            rec.key = key
            rec.base[key] = past
        if tap is not None:
            tap.key = key

    @torch.no_grad()
    def step_many(items, keys=None):
        import transformers
        if transformers.__version__ != LAYER_MAJOR_TRANSFORMERS:
            raise RuntimeError(f"layer_major restates transformers {LAYER_MAJOR_TRANSFORMERS}'s Qwen4ExpTextModel."
                               f"forward; this is {transformers.__version__}")
        keys = list(range(len(items))) if keys is None else list(keys)
        ps = []
        for j, (ids, pos, st) in enumerate(items):
            select(keys[j], pos)
            ps.append(prologue(ids, pos, st))
        pf.hold = True
        try:
            for layer in layers:
                for j, p in enumerate(ps):
                    select(keys[j], p["past"])
                    p["h"] = layer(p["h"], position_embeddings=p["pe"], attention_mask=p["masks"]["indexed_attention"],
                                   conv_mask=p["conv"], past_key_values=p["cache"], ple_input_ids=p["ple"])
                pf.release()
        finally:
            pf.hold = False
            pf.release()
        out = []
        for p in ps:
            out.append(p["h"][0])
            p.clear()
        return out
    return step_many


@torch.no_grad()
def final_logits(model, R: torch.Tensor) -> torch.Tensor:
    """The final mixer (no inject) and lm_head over pre-mixer rows R [T, hc*H] (M:1485, M:1656)."""
    return model.lm_head(model.model.hyper_connection_mixer(R[None]))[0]


# ------------------------------------------------------------------------------------------------
# a sequential run (prompt in chunks, then greedy decode)

def read_ids(path: str) -> list[int]:
    return [int(x) for x in open(path, encoding="utf-8").read().split()]


@torch.no_grad()
def forward_chunks(model, ids: list[int], chunk: int, cache=None, rec: "Recorder | None" = None,
                   on_logits=None, logits_to_keep: int = 0):
    """model(input_ids=chunk, past_key_values=cache) over `ids` in chunks (whole when chunk <= 0): the
    reference's prefill. on_logits(row0, logits bf16 [c][V]) sees every chunk's logits. Returns the cache."""
    pos = 0 if cache is None else cache.get_seq_length()
    c = chunk if chunk > 0 else len(ids)
    for a in range(0, len(ids), c):
        part = ids[a:a + c]
        if rec is not None:
            rec.base[rec.key] = pos + a
        out = model(input_ids=torch.tensor([part]), past_key_values=cache, use_cache=True,
                    logits_to_keep=logits_to_keep)
        cache = out.past_key_values
        if on_logits is not None:
            on_logits(pos + a + len(part) - out.logits.shape[1], out.logits[0])
    return cache


def auto_chunk(n: int, chunk: int) -> int:
    if chunk >= 0:
        return chunk
    return 0 if n <= 4096 else 2048


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, transformers {PINNED_TRANSFORMERS}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    tc = text_config(a.snapshot, a.layers, a.experts)
    require_cacheable(tc)
    ids = read_ids(a.prompt)
    T = len(ids)
    if not 1 <= T <= a.max_prompt:
        die(f"{a.prompt}: {T} ids (1..{a.max_prompt})")
    ple = PleTable(ple_source(a.snapshot, a.ple), tc)
    model, pf, lazy = build_streamed(a.snapshot, tc, ple)
    chunk = auto_chunk(T, a.chunk)
    act_from = max(0, T - a.act_tail) if a.act_tail >= 0 else 0
    log_from = max(0, T - max(1, a.logits_tail)) if a.logits_tail >= 0 else 0   # the last prompt row: greedy
    print(f"config: {tc.num_hidden_layers} layers ({sum(t != 'linear_attention' for t in tc.layer_types)} QSA), "
          f"{tc.num_experts} experts top-{tc.num_experts_per_tok}, hidden {tc.hidden_size} x {tc.hc_count} streams, "
          f"attn eager, experts {tc._experts_implementation}; PLE {ple.kind}; prompt {T} ids, chunk {chunk or 'whole'}; "
          f"activations from row {act_from}, logits from row {log_from}; built {time.time() - t0:.1f}s", flush=True)
    rec = Recorder(model, tc, act_from=act_from)
    rows, nll = [], torch.empty(max(T - 1, 0))
    ids_t = torch.tensor(ids)

    def on_logits(r0, lg):
        for b0 in range(0, lg.shape[0], 256):             # 256 rows at a time: a 2048-row f32 block is 2 GB
            lf = lg[b0:b0 + 256].float()
            ra = r0 + b0
            r1 = ra + lf.shape[0]
            a_, b_ = ra, min(r1, T - 1)
            if b_ > a_:
                nll[a_:b_] = -torch.log_softmax(lf[: b_ - a_], -1).gather(-1, ids_t[a_ + 1:b_ + 1, None])[:, 0]
            if r1 > log_from:
                rows.append(lf[max(0, log_from - ra):].clone())
    t1 = time.time()
    cache = forward_chunks(model, ids, chunk, rec=rec, on_logits=on_logits)
    print(f"prompt forward {time.time() - t1:.1f}s ({lazy.filled} expert dequants, {ple.reads} PLE rows)", flush=True)
    tokens: list[int] = []
    t2 = time.time()
    last = rows[-1][-1]
    for step in range(a.gen):
        nxt = int(torch.argmax(last).item())
        tokens.append(nxt)
        rec.base[0] = T + step
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last = out.logits[0, -1].float()
        rows.append(out.logits[0].float())
        if step == 0:
            print(f"  first decode step {time.time() - t2:.1f}s", flush=True)
    print(f"greedy {a.gen}: {time.time() - t2:.1f}s -> {tokens}", flush=True)
    rec.remove()
    t = rec.tensors()
    t["logits"] = torch.cat(rows, 0).contiguous()
    t["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    t["nll"] = nll.float().contiguous()
    n_rows = T + a.gen
    for i in range(tc.num_hidden_layers):
        for k in ("route.ids", "route.w", "route.gap"):
            if t[f"{k}.L{i}"].shape[0] != n_rows:
                die(f"{k}.L{i} has {t[f'{k}.L{i}'].shape[0]} rows, expected {n_rows}")
    if t["ple.ids"].shape[0] != n_rows:
        die(f"ple.ids has {t['ple.ids'].shape[0]} rows, expected {n_rows}")
    if not torch.isfinite(t[f"H.L{tc.num_hidden_layers - 1}"].float()).all():
        die("the last layer's output has NaN/Inf - the weights are wrong")
    for line in gap_summary(t, tc):
        print(line)
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt": os.path.abspath(a.prompt), "n_prompt": str(T),
            "gen": str(a.gen), "layers": str(tc.num_hidden_layers), "chunk": str(chunk), "ple": ple.kind,
            "act_row0": str(act_from), "logits_row0": str(log_from), "qsa_row0": str(QSA_DENSE_ROWS),
            "attn_implementation": "eager", "experts_implementation": tc._experts_implementation,
            "transformers": PINNED_TRANSFORMERS, "torch": torch.__version__,
            "ppl": f"{math.exp(float(nll.double().mean())) if T > 1 else float('nan'):.6g}",
            "model": "Qwen3.8-Flash-Next (Qwen4ExpForCausalLM, tools/oracle/qwen4exp_ref.py, layer-streamed)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(t, a.out, metadata=meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wrote {a.out}: {len(t)} tensors; prompt ppl {meta['ppl']}")
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB, prefetch wait {pf.seconds:.1f}s")


def ple_source(snapshot: str, spec: str) -> str:
    if spec in ("bf16", "", None):
        return snapshot
    if spec.startswith("bf16:"):
        return spec[5:]
    if spec.startswith("int8:"):
        return spec
    die(f"--ple {spec}: bf16, bf16:<snapshot> or int8:<dir>")


# ------------------------------------------------------------------------------------------------
# ppl

def tokenize_text(snapshot: str, path: str) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    text = open(path, encoding="utf-8").read().rstrip("\n")
    return tok.encode(text, add_special_tokens=False).ids


def cmd_ppl(a) -> None:
    t0 = time.time()
    tc = text_config(a.snapshot, a.layers, a.experts)
    require_cacheable(tc)
    model, pf, lazy = build_streamed(a.snapshot, tc, PleTable(ple_source(a.snapshot, a.ple), tc))
    srcs = [(n, tokenize_text(a.snapshot, p)) for n, p in (x.split("=", 1) for x in a.text)]
    srcs += [(n, read_ids(p)) for n, p in (x.split("=", 1) for x in a.ids)]
    if not srcs:
        die("ppl: no --text / --ids")
    for name, ids in srcs:
        ids = ids[: a.max_tokens]
        tot, n = 0.0, 0
        ids_t = torch.tensor(ids)

        def on_logits(r0, lg):
            nonlocal tot, n
            for b0 in range(0, lg.shape[0], 256):
                ra = r0 + b0
                b_ = min(ra + min(256, lg.shape[0] - b0), len(ids) - 1)
                if b_ > ra:
                    lp = torch.log_softmax(lg[b0:b0 + b_ - ra].float(), -1).gather(-1, ids_t[ra + 1:b_ + 1, None])[:, 0]
                    tot -= float(lp.double().sum())
                    n += b_ - ra
        t1 = time.time()
        forward_chunks(model, ids, auto_chunk(len(ids), a.chunk), on_logits=on_logits)
        print(f"ppl {name}: {math.exp(tot / max(n, 1)):.4f} over {n} predicted tokens ({len(ids)} ids, "
              f"{tc.num_hidden_layers} layers, {time.time() - t1:.0f}s)", flush=True)
    print(f"wall {time.time() - t0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB")


# ------------------------------------------------------------------------------------------------
# hfcheck: transformers' own indexer against the port's cache, on any weights (--layers N)

def cmd_hfcheck(a) -> None:
    t0 = time.time()
    tc = text_config(a.snapshot, a.layers, a.experts)
    require_cacheable(tc)
    ple = PleTable(ple_source(a.snapshot, a.ple), tc)
    if a.prompt:
        ids = read_ids(a.prompt)[: a.n]
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, tc.vocab_size, (a.n,), generator=g).tolist()
    res = {}
    for arm, own in (("port", False), ("transformers", True)):
        model, pf, _ = build_streamed(a.snapshot, tc, ple, own_indexer=own)
        rec = Recorder(model, tc, act_from=0)
        rows = []
        cache = forward_chunks(model, ids, 0, rec=rec, on_logits=lambda r0, lg: rows.append(lg), logits_to_keep=64)
        toks = []
        for step in range(a.gen):
            nxt = int(rows[-1][-1].float().argmax())
            toks.append(nxt)
            rec.base[0] = len(ids) + step
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0])
        rec.remove()
        t = {k: v for k, v in rec.tensors().items() if k.startswith(("H.", "mixer.", "moe.", "route."))}
        t["logits"] = torch.cat(rows)
        res[arm] = (t, toks)
        print(f"hfcheck {arm}: {len(ids)} ids + {a.gen} steps, {time.time() - t0:.0f}s", flush=True)
        del model, pf
    (pt, ptok), (ht, htok) = res["port"], res["transformers"]
    bad = [k for k in ht if not torch.equal(pt[k], ht[k])]
    print(f"hfcheck: {len(ht)} tensors, greedy {ptok} vs {htok}; differing: {bad[:8] if bad else 'none'}")
    if bad or ptok != htok:
        print("hfcheck: FAIL (the cached indexer is not transformers' arithmetic on these weights)")
        sys.exit(1)
    print(f"hfcheck: BITWISE ({tc.num_hidden_layers} layers, {len(ids)} ids: the QSA selection active past "
          f"{QSA_DENSE_ROWS - 1})")


# ------------------------------------------------------------------------------------------------
# trace (Task 7): teacher-forced routes for spec 22's P0.6 / P0.8

class OnormTap:
    """The routed experts' output norm per (row, slot) - REAP's ||f_e(x)|| - from grouped_mm's own down
    projection (transformers.integrations.moe._grouped_linear's second call, sorted by expert), un-permuted
    by the same torch.sort the implementation does. Values are only read, never changed."""

    def __init__(self, model_or_layers):
        import transformers.integrations.moe as moe
        layers = list(model_or_layers.model.layers) if hasattr(model_or_layers, "model") else list(model_or_layers)
        self.moe, self.orig = moe, moe._grouped_linear
        self.calls: list = []
        self.active = False
        self.rows: dict = {}

        def wrapped(*args, **kw):
            out = self.orig(*args, **kw)
            if self.active:
                self.calls.append(out)
            return out
        moe._grouped_linear = wrapped
        self.hooks = []
        for i, layer in enumerate(layers):
            self.hooks.append(layer.mlp.experts.register_forward_pre_hook(self._pre))
            self.hooks.append(layer.mlp.experts.register_forward_hook(self._post(i)))
        self.key = 0

    def _pre(self, _m, _a):
        self.calls, self.active = [], True

    def _post(self, i):
        def fn(mod, args, _out):
            self.active = False
            if mod.config._experts_implementation != "grouped_mm" or len(self.calls) != 2:
                raise RuntimeError("the onorm tap reads grouped_mm's two grouped linears")
            idx = args[1]
            _, perm = torch.sort(idx.reshape(-1))
            norm_sorted = self.calls[1].float().norm(dim=-1)
            onorm = torch.empty_like(norm_sorted)
            onorm[perm] = norm_sorted
            self.rows.setdefault(self.key, {}).setdefault(i, []).append(onorm.view(idx.shape).clone())
        return fn

    def close(self):
        self.moe._grouped_linear = self.orig
        for h in self.hooks:
            h.remove()


def opencode_sources(d: str) -> list[tuple[str, list[int]]]:
    """mtp_accept.py's reader of a recorded opencode session (spec 7 plan 7a's request log): each turn's
    prompt ids followed by its output ids, teacher-forced. The log's ids are Qwen3.8's tokenizer's, which
    this model's is (the facts sheet's token section)."""
    acc = _load("mtp_accept")
    turns, _ = acc.opencode_sources(d, 10**12, 10**12)
    return [(re.sub(r"[^A-Za-z0-9_.-]", "_", name).removesuffix(".json"), list(ctx) + list(cont))
            for name, ctx, cont in turns]


def trace_routes(model, pf, tc, sources: list[tuple[str, list[int]]], chunk: int, head=None) -> dict:
    """Teacher-forced routes for `sources` (name, ids) in ONE layer-major pass per chunk round (no
    generation): {name: {"ids": i32 [T][L][k] ascending, "p": f32 renormalised fp32 probabilities,
    "onorm": f32 ||f_e(x)||, and with `head` the MTP head's step-1 routes "mtp_ids" / "mtp_p" /
    "mtp_onorm" [T][k] (row T-1: -1 / 0)}}."""
    rec = Recorder(model, tc, acts=False, want_p=True)
    tap = OnormTap(model)
    step = layer_major(model, pf, rec, tap)
    states = [{} for _ in sources]
    Rs = [[] for _ in sources]
    longest = max(len(ids) for _, ids in sources)
    c = chunk if chunk > 0 else longest
    try:
        for c0 in range(0, longest, c):
            items, where = [], []
            for j, (_, ids) in enumerate(sources):
                if c0 < len(ids):
                    items.append((ids[c0:c0 + c], c0, states[j]))
                    where.append(j)
            for j, R in zip(where, step(items, keys=where)):
                Rs[j].append(R)
    finally:
        rec.remove()
        tap.close()
    L = tc.num_hidden_layers
    out = {}
    for j, (name, ids) in enumerate(sources):
        r = rec.rows[j]
        ids_t = torch.stack([torch.cat(r[f"route.ids.L{i}"]) for i in range(L)], 1)
        p_t = torch.stack([torch.cat(r[f"route.p.L{i}"]) for i in range(L)], 1)
        on_t = torch.stack([torch.cat(tap.rows[j][i]) for i in range(L)], 1)
        order = ids_t.argsort(-1)
        o = {"ids": ids_t.gather(-1, order).contiguous(), "p": p_t.gather(-1, order).contiguous(),
             "onorm": on_t.gather(-1, order).contiguous()}
        if o["ids"].shape[0] != len(ids):
            raise RuntimeError(f"{name}: {o['ids'].shape[0]} rows, expected {len(ids)}")
        if head is not None:
            o.update(head.trace_routes(torch.cat(Rs[j]), ids, chunk=c))
        out[name] = o
    return out


def cmd_trace(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    tc = text_config(a.snapshot, a.layers, a.experts)
    require_cacheable(tc)
    srcs = []
    for s in a.source:
        name, _, path = s.partition(":")
        srcs.append((name, read_ids(path)[: a.max_tokens]))
    if a.opencode:
        srcs += opencode_sources(a.opencode)
    if not srcs:
        die("trace: no --source")
    os.makedirs(a.out, exist_ok=True)
    todo = [(n, ids) for n, ids in srcs if not os.path.exists(os.path.join(a.out, f"{n}.routes.safetensors"))]
    print(f"trace: {len(srcs)} sources, {len(srcs) - len(todo)} already written, chunk {a.chunk}, batch {a.batch}")
    model, pf, lazy = build_streamed(a.snapshot, tc, PleTable(ple_source(a.snapshot, a.ple), tc))
    head = None
    if a.mtp and any(k.startswith("mtp.") for k in model.q4_ckpt.keys()):
        head = _load("qwen4exp_mtp").MtpHead(a.snapshot, tc, model=model)
    L, k = tc.num_hidden_layers, tc.num_experts_per_tok
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0:b0 + a.batch]
        t1 = time.time()
        res = trace_routes(model, pf, tc, batch, a.chunk if a.chunk > 0 else 2048, head)
        for name, ids in batch:
            o = res[name]
            meta = {"model": "Qwen3.8-Flash-Next", "checkpoint": os.path.abspath(a.snapshot),
                    "checkpoint_revision": os.path.basename(os.path.normpath(a.snapshot)),
                    "transformers": PINNED_TRANSFORMERS, "source": name, "T": str(len(ids)), "layers": str(L),
                    "top_k": str(k), "experts": str(tc.num_experts), "chunk": str(a.chunk),
                    "format": TRACE_FORMAT}
            save_file(o, os.path.join(a.out, f"{name}.routes.safetensors"), metadata=meta)
            distinct = [int(torch.unique(o["ids"][:, i]).numel()) for i in range(L)]
            print(f"trace {name}: {len(ids)} tokens; distinct experts per layer min {min(distinct)} median "
                  f"{sorted(distinct)[L // 2]} max {max(distinct)}", flush=True)
        print(f"trace batch of {len(batch)}: {time.time() - t1:.0f}s ({lazy.filled} expert dequants so far)", flush=True)
    print(f"wall {time.time() - t0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB")


TRACE_FORMAT = ("ids i32 [T][L][k]: each row's top-k per layer, ascending expert id; p f32 [T][L][k]: the router's "
                "renormalised fp32 probabilities (before the bf16 cast) aligned with ids; onorm f32 [T][L][k]: the L2 "
                "norm of each routed expert's output (bf16, before the routing weight: REAP's ||f_e(x)||); mtp_ids / "
                "mtp_p / mtp_onorm [T][k]: the MTP head's step-1 routes on (R_t, t+1) - row T-1 has no next token "
                "(-1 / 0)")


# ------------------------------------------------------------------------------------------------
# torch's CPU topk on exact ties (Task 2 Step 4)

def topk_ties() -> list[str]:
    """Which index torch.topk returns on planted exact ties at the cut: fp32 [600] at 512 / 513 (the QSA
    selection, M:749) and fp32 softmax [512] at 10 / 11 (the router, M:965)."""
    out = []
    g = torch.Generator().manual_seed(0)
    for n, k, what in ((600, 512, "QSA top-512 of 600 blocks"), (512, 10, "router top-10 of 512")):
        lower, stable, trials = 0, 0, 50
        for t in range(trials):
            v = torch.rand(n, generator=g)
            order = v.argsort(descending=True)
            a, b = sorted(order[k - 1:k + 1].tolist())       # the k-th and (k+1)-th: make them an exact tie
            v[b] = v[a]
            if what.startswith("router"):
                v = torch.softmax(torch.log(v) * 5, -1)       # probabilities with the tie kept exact
                if v[a] != v[b]:
                    continue
            idx = v.topk(k).indices.tolist()
            lower += (a in idx) and (b not in idx)
            stable += idx == v.topk(k).indices.tolist()
        out.append(f"{what}: the lower index wins the tie at the cut in {lower}/{trials} trials; "
                   f"repeat calls identical {stable}/{trials} (torch {torch.__version__})")
    return out


def cmd_facts(a) -> None:
    facts.main([a.snapshot] + (["--qwen38-tokenizer", a.qwen38_tokenizer] if a.qwen38_tokenizer else []))
    tc = text_config(a.snapshot)
    ck = Ckpt(a.snapshot)
    names = {k for k in ck.keys() if not k.startswith("model.visual.")}
    for form in ("bf16", "intel", "ours", "tiny"):
        want = set(expected_names(tc, form))
        if want == names:
            print(f"expected_names(tc, {form!r}) == the checkpoint's {len(names)} text tensors")
            break
    else:
        best = min(("bf16", "intel", "ours", "tiny"), key=lambda f: len(set(expected_names(tc, f)) ^ names))
        want = set(expected_names(tc, best))
        print(f"no form matches; nearest {best}: missing {sorted(want - names)[:5]}, extra {sorted(names - want)[:5]}")
    for line in topk_ties():
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("snapshot")
        p.add_argument("--layers", type=int)
        p.add_argument("--experts", default=DEFAULT_EXPERTS, choices=["grouped_mm", "eager"])
        p.add_argument("--ple", default="bf16", help="bf16 (the snapshot's shards), bf16:<snapshot>, int8:<dir>")
        p.add_argument("--chunk", type=int, default=-1, help="prefill chunk (-1: whole up to 4096 ids, else 2048)")
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=65536)
    r.add_argument("--logits-tail", type=int, default=1024, help="keep logits from row T - L (-1: all)")
    r.add_argument("--act-tail", type=int, default=256, help="keep H / mixer / moe from row T - A (-1: all)")
    p = sub.add_parser("ppl")
    common(p)
    p.add_argument("--text", action="append", default=[], help="NAME=FILE (tokenized by the snapshot's tokenizer.json)")
    p.add_argument("--ids", action="append", default=[], help="NAME=FILE")
    p.add_argument("--max-tokens", type=int, default=4096)
    h = sub.add_parser("hfcheck")
    common(h)
    h.add_argument("--prompt")
    h.add_argument("--n", type=int, default=2100)
    h.add_argument("--gen", type=int, default=4)
    t = sub.add_parser("trace")
    common(t)
    t.add_argument("--source", action="append", default=[], help="NAME:IDS_FILE")
    t.add_argument("--opencode", help="a recorded opencode session dir (mtp_accept.py's reader)")
    t.add_argument("--out", required=True)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--max-tokens", type=int, default=32768)
    t.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True)
    f = sub.add_parser("facts")
    f.add_argument("snapshot")
    f.add_argument("--qwen38-tokenizer")
    a = ap.parse_args()
    {"run": cmd_run, "ppl": cmd_ppl, "hfcheck": cmd_hfcheck, "trace": cmd_trace, "facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
