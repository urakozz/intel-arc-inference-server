#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K2-Horizon (IFM K2-Horizon-MoVA-36B-A4B) CPU reference - spec 18a.

A plain-torch port of the checkpoint's own `modeling_k2_horizon.py` (vendored, unmodified, under
third_party/k2_horizon/; Apache-2.0), layer at a time, float32 compute, deterministic. It reads
the bf16 checkpoint or the operator's int4 AutoRound GPTQ checkpoint (dequantised by dequant.py's
rule) by tensor name, and dumps what spec 18 §6 K2 needs: per-layer residuals, MoE and MoVA
routing (ids ascending, weights, the selection-score gap at the cut) per layer per position,
every logits row, the greedy ids.

    k2_ref.py run <snapshot> --prompt <ids-file> --out <file.safetensors> [--gen 32] [--mode bf16]
    k2_ref.py hfcheck <snapshot> --prompt <ids-file> [--against <file.safetensors>]
    k2_ref.py facts <snapshot-or-config.json>

Semantics (file:line = third_party/k2_horizon/modeling_k2_horizon.py):

  layer        h += Attn(GNorm(h)); h += (dense ? MLP : MoE)(GNorm(h))            :705-728
  GNorm        groups of hidden/G (G = layernorm_num_groups = 2), each x*rsqrt(mean(x^2)+eps)
               on its own mean of squares, then PLAIN w * x (not (1 + w) * x); fp32 inside,
               one rounding to the activation dtype                                :625-636
  kinds        sparse = i not in mlp_only_layers and (i + 1) % decoder_sparse_step == 0;
               sparse -> MoVA attention (if mova_num_experts) + MoE; else plain attention + MLP :647-658
  attention    q, k = W_q x, W_k x (+ per-head grouped q/k norm if query_key_norm); RoPE
               rotate_half, cos/sin = cat(f, f), f = pos * theta^(-2i/rope_dim), rounded to the
               activation dtype; partial RoPE (rope_head_dim < head_dim) rotates the channel pairs
               (i, i + head_dim/2), i < rope_head_dim/2, and passes the rest     :258-289, :773-800
               GQA head h reads kv head h // (heads / kv_heads); eager softmax in fp32  :86-133
               o = attn * softplus(W_g x; beta = ln 2, threshold 20) - x itself where beta*x > 20 :312-321
  MoVA         v = sum over the top-k of mova_num_experts value experts, ascending id, of
               w_e * SiLU(V_e x); THIS v is what the KV cache stores and attention reads  :424-444
  routers      s = sigmoid(fp32(W_r x)) (logits rounded to the activation dtype first); the bias
               is added to s for SELECTION ONLY; w = s[ids] / sum(s[ids]) * scaling (2.5), cast to
               the activation dtype; MoVA always normalises (top_k > 1), MoE when norm_topk_prob
                                                                                   :136-166, :560-579
  combine      routed outputs summed in ascending expert id into a zero tensor in the activation
               dtype (index_add_, one rounding per add), then the shared expert added  :169-194, :581-607
  head         logits = lm_head(GNorm(h)); no embedding or logit scaling           :902, :1069

Rounding (mode "bf16", the default): values are kept in float32 tensors and rounded to bf16
exactly where the reference running in bf16 rounds (every linear output, every elementwise op,
norms once, routing weights once, each index_add_ add, attention scores/probabilities/output as
transformers' eager path does); the matmuls are torch's own bf16 GEMMs (fp32 accumulation, one
rounding), because an fp32 matmul rounded afterwards accumulates in another order and differs by
an ulp in places. On the same torch this mode is bit-identical to the vendored model in bf16 with
eager attention (test_k2_ref.py: every residual and logit, prompt and cached decode). Mode "f32"
never rounds and uses fp32 matmuls (a higher-precision control; = the vendored model in fp32).
Deterministic: two runs on one machine and thread count are bitwise equal.

Departures from the HF code, both deliberate and recorded in spec 18a's facts doc:
  * top-k ties go to the LOWER expert id (a stable descending sort). torch.topk leaves ties to
    the implementation (measured on torch 2.14 CPU: 100 equal scores, top 8 -> 69, 63, 64, ...),
    and the engine needs one rule. The routing dump's gap column is 0 exactly where this matters.
  * attention is never sliding-window: the checkpoint has use_sliding_window false; a config
    that sets it is refused rather than half-implemented.

The weights are read per layer: a background thread loads the next layer's attention, router,
norm and shared/dense tensors (stream.py's _Prefetch), and experts are read on first use within
a layer and dropped with it, so a decode step reads only the 8 + 4 experts it routes to.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

THIRD_PARTY = os.path.join(_HERE, "third_party", "k2_horizon")
SOFTPLUS_BETA = math.log(2)
SOFTPLUS_THRESHOLD = 20.0   # F.softplus's default, which the modeling code relies on


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"k2_oracle_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dequant = _load("dequant")
_stream = None


def stream_mod():
    global _stream
    if _stream is None:
        _stream = _load("stream")
    return _stream


# --------------------------------------------------------------------------------------------
# config

@dataclass
class K2Config:
    """config.json, with configuration_k2_horizon.py's defaults for absent keys."""
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_head_dim: int | None = None
    layernorm_num_groups: int = 1
    mlp_only_layers: list = field(default_factory=list)
    decoder_sparse_step: int = 1
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    num_shared_experts: int = 0
    norm_topk_prob: bool = False
    router_score_func: str = "softmax"
    router_scaling_factor: float = 1.0
    moe_gate_bias: bool = False
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0
    attention_gate_func: str | None = None
    query_key_norm: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_ids: list = field(default_factory=list)
    group_size: int | None = None     # GPTQ group size when the checkpoint is quantised

    @classmethod
    def from_dict(cls, d: dict) -> "K2Config":
        if d.get("model_type") not in (None, "k2_horizon"):
            raise ValueError(f"model_type {d.get('model_type')!r} is not k2_horizon")
        if d.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {d.get('hidden_act')!r}: the modeling code's experts are SiLU only")
        if d.get("use_sliding_window") and d.get("sliding_window"):
            raise ValueError("use_sliding_window: K2-Horizon ships without; not implemented here")
        rp = d.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rp.get('rope_type')!r}: only 'default' is implemented")
        theta = rp.get("rope_theta", d.get("rope_theta", 10000.0))
        if d.get("router_score_func", "softmax") != "sigmoid":
            raise ValueError(f"router_score_func {d.get('router_score_func')!r}: only sigmoid is implemented")
        if d.get("attention_gate_func") not in (None, "softplus", "silu"):
            raise ValueError(f"attention_gate_func {d.get('attention_gate_func')!r}")
        if d.get("attention_bias"):
            raise ValueError("attention_bias: K2-Horizon ships without; not implemented here")
        names = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in d.items() if k in names}
        kw["rope_theta"] = float(theta)
        eos = d.get("eos_token_id")
        kw["eos_token_ids"] = list(eos) if isinstance(eos, list) else ([] if eos is None else [eos])
        if kw.get("router_scaling_factor") is None:
            kw["router_scaling_factor"] = 1.0
        if kw.get("mlp_only_layers") is None:
            kw["mlp_only_layers"] = []
        q = d.get("quantization_config")
        if q:
            if q.get("bits") != 4 or not q.get("sym") or q.get("desc_act"):
                raise ValueError(f"quantization_config: only int4 symmetric without desc_act, got {q}")
            if q.get("quant_method") not in ("gptq", "auto-round"):
                raise ValueError(f"quant_method {q.get('quant_method')!r}")
            kw["group_size"] = int(q["group_size"])
        c = cls(**kw)
        if c.hidden_size % c.layernorm_num_groups:
            raise ValueError("hidden_size % layernorm_num_groups != 0")
        if c.num_attention_heads % c.num_key_value_heads:
            raise ValueError("num_attention_heads % num_key_value_heads != 0")
        rd = c.rope_dim
        if rd % 2 or rd > c.head_dim:
            raise ValueError(f"rope_head_dim {rd} must be even and <= head_dim {c.head_dim}")
        return c

    @classmethod
    def from_file(cls, path: str) -> "K2Config":
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def rope_dim(self) -> int:
        return self.head_dim if self.rope_head_dim is None else self.rope_head_dim

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_sparse(self, i: int) -> bool:
        return (i not in self.mlp_only_layers and self.num_experts > 0
                and (i + 1) % self.decoder_sparse_step == 0)

    def is_mova(self, i: int) -> bool:
        return self.is_sparse(i) and self.mova_num_experts > 0

    def layer_kind(self, i: int) -> str:
        return ("mova+moe" if self.is_mova(i) else "moe") if self.is_sparse(i) else "dense"


def expected_names(c: K2Config) -> list[str]:
    """Every tensor the bf16 checkpoint ships, by name (the modeling code's state dict)."""
    out = ["model.embed_tokens.weight", "model.norm.weight"]
    if not c.tie_word_embeddings:
        out.append("lm_head.weight")
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        out += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        a = p + "self_attn."
        out += [a + "q_proj.weight", a + "k_proj.weight", a + "o_proj.weight"]
        if c.attention_gate_func is not None:
            out.append(a + "gate_proj.weight")
        if c.query_key_norm:
            out += [a + "q_norm.weight", a + "k_norm.weight"]
        if c.is_mova(i):
            out.append(a + "v_router.weight")
            if c.moe_gate_bias:
                out.append(a + "v_router.bias")
            out += [a + f"v_experts.{e}.weight" for e in range(c.mova_num_experts)]
        else:
            out.append(a + "v_proj.weight")
        m = p + "mlp."
        if c.is_sparse(i):
            out.append(m + "gate.weight")
            if c.moe_gate_bias:
                out.append(m + "gate.bias")
            for e in range(c.num_experts):
                out += [m + f"experts.{e}.{n}_proj.weight" for n in ("gate", "up", "down")]
            if c.num_shared_experts:
                out += [m + f"shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")]
        else:
            out += [m + f"{n}_proj.weight" for n in ("gate", "up", "down")]
    return out


# --------------------------------------------------------------------------------------------
# arithmetic: fp32 values, rounded where the bf16 reference rounds

class Arith:
    def __init__(self, mode: str = "bf16"):
        if mode not in ("bf16", "f32"):
            raise ValueError(f"mode {mode!r}: bf16 or f32")
        self.mode = mode

    def r(self, x: torch.Tensor) -> torch.Tensor:
        """Round float32 values to the activation dtype (bf16, RNE) and back; identity in f32."""
        return x.to(torch.bfloat16).float() if self.mode == "bf16" else x

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """y = x W^T with fp32 accumulation, rounded once.

        bf16 mode runs torch's own bf16 GEMM (fp32 accumulate, one RNE rounding) on the bf16
        values: that is the reference's F.linear, and it is NOT bitwise the same as an fp32
        matmul rounded afterwards - the accumulation order differs (measured on torch 2.14 CPU:
        109 of 1008 shapes differ in 1-4 elements by one bf16 ulp)."""
        if self.mode == "bf16":
            return F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16)).float()
        return x @ w.float().t()

    def mm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a @ b, as linear(): torch's bf16 matmul in bf16 mode (eager attention's two matmuls)."""
        if self.mode == "bf16":
            return (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float()
        return a @ b


def grouped_rms_norm(ar: Arith, x: torch.Tensor, w: torch.Tensor, groups: int, eps: float) -> torch.Tensor:
    """K2HorizonRMSNorm.forward (:625-636): each group on its own mean of squares, then PLAIN w * x."""
    shape = x.shape
    xg = x.float().reshape(*shape[:-1], groups, -1)
    var = xg.pow(2).mean(-1, keepdim=True)
    xg = xg * torch.rsqrt(var + eps)
    return ar.r(w.float() * xg.reshape(shape))


def rope_cos_sin(ar: Arith, positions: torch.Tensor, rope_dim: int, theta: float):
    """K2HorizonRotaryEmbedding (:771-800): [T, rope_dim] each, cat(freqs, freqs), rounded."""
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return ar.r(emb.cos()), ar.r(emb.sin())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(ar: Arith, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [heads, T, head_dim] in the checkpoint's split-halves layout; cos/sin [T, rope_dim].

    Full: x*cos + rotate_half(x)*sin, each op rounded (:79-83). Partial (rope_dim < head_dim):
    the modeling code's interleave dance (:269-289) rotates exactly the channel pairs
    (i, i + head_dim/2) for i < rope_dim/2, with rope_dim's frequencies, and leaves the others."""
    d, rd = x.shape[-1], cos.shape[-1]
    if rd == d:
        return ar.r(ar.r(x * cos) + ar.r(rotate_half(x) * sin))
    idx = torch.cat([torch.arange(rd // 2), d // 2 + torch.arange(rd // 2)])
    out = x.clone()
    xr = x[..., idx]
    out[..., idx] = ar.r(ar.r(xr * cos) + ar.r(rotate_half(xr) * sin))
    return out


def attention(ar: Arith, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor,
              k_pos: torch.Tensor, n_rep: int) -> torch.Tensor:
    """eager_attention_forward (:110-133): q [H, T, d], k/v [Hkv, P, d] -> [T, H, d].

    Causal by absolute position; head h reads kv head h // n_rep (repeat_kv). Scores and
    probabilities rounded as transformers' bf16 eager path rounds them; the masked entries are
    exactly 0 after the fp32 softmax either way."""
    H, d = q.shape[0], q.shape[-1]
    kk = k.repeat_interleave(n_rep, dim=0)
    vv = v.repeat_interleave(n_rep, dim=0)
    s = ar.mm(q, kk.transpose(1, 2))
    s = ar.r(s * (d ** -0.5))
    mask = k_pos[None, :] > q_pos[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    p = ar.r(torch.softmax(s, dim=-1))
    o = ar.mm(p, vv)
    return o.transpose(0, 1)


def attn_gate(ar: Arith, g: torch.Tensor, func: str | None) -> torch.Tensor:
    if func == "softplus":   # beta = ln 2; returns g itself where beta * g > 20 (:319)
        return ar.r(F.softplus(g, beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD))
    if func == "silu":
        return ar.r(F.silu(g))
    raise ValueError(func)


@dataclass
class Route:
    ids: torch.Tensor        # [T, k] int64, ascending expert id
    weights: torch.Tensor    # [T, k] fp32, the multiplier each id gets (rounded in bf16 mode)
    gap: torch.Tensor        # [T] fp32, selection score of the k-th minus the (k+1)-th (0 = a tie at the cut)
    sel_order: torch.Tensor  # [T, k] int64, the ids in selection order (torch.topk's order)


def route(ar: Arith, logits: torch.Tensor, bias: torch.Tensor | None, k: int, scale: float,
          normalise: bool) -> Route:
    """calc_router_weights (:136-166) / the MoE block's inline copy (:560-579).

    s = sigmoid(fp32 logits); selection on s + bias; weights from s alone; normalised over the
    selected (sum in selection order, as torch.topk returns them), times `scale`, rounded once.
    Ties at equal selection score go to the lower id (stable sort)."""
    s = torch.sigmoid(logits.float())
    sel = s + bias.float() if bias is not None else s
    vals, order = torch.sort(sel, dim=-1, descending=True, stable=True)
    top = order[..., :k]
    gap = (vals[..., k - 1] - vals[..., k]) if sel.shape[-1] > k else torch.full(sel.shape[:-1], float("inf"))
    w_sel = torch.gather(s, -1, top)
    if normalise:
        w_sel = w_sel / w_sel.sum(dim=-1, keepdim=True)
    w_sel = ar.r(w_sel * scale)
    ids, perm = torch.sort(top, dim=-1)
    return Route(ids=ids, weights=torch.gather(w_sel, -1, perm), gap=gap, sel_order=top)


def combine_ascending(ar: Arith, rt: Route, x: torch.Tensor, expert_fn, out_dim: int) -> torch.Tensor:
    """sum_e w_e * expert_e(x) over each row's selected experts, ascending expert id.

    The modeling code's loop (:186-192, :590-603): experts in ascending id; each expert runs on
    the rows that selected it (in torch.where's slot-major order over the selection-order ids),
    its output times the rounded weight is rounded, then index_add_ into a zero tensor in the
    activation dtype - one rounding per add, so the order is part of the result."""
    T = x.shape[0]
    acc = torch.zeros(T, out_dim)
    for e in torch.unique(rt.ids).tolist():          # torch.unique is sorted ascending
        slot, rows = torch.where((rt.sel_order == e).t())
        y = expert_fn(e, x[rows])
        w = rt.weights[rows][rt.ids[rows] == e]
        y = ar.r(y * w[:, None])
        acc[rows] = ar.r(acc[rows] + y)
    return acc


def mlp(ar: Arith, x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """K2HorizonMLP (:526-528): down(silu(gate x) * up x), each op rounded."""
    return ar.linear(ar.r(ar.r(F.silu(ar.linear(x, gate))) * ar.linear(x, up)), down)


# --------------------------------------------------------------------------------------------
# weights

class DictSource:
    """Tensors by checkpoint name from an in-memory dict (tests)."""

    def __init__(self, sd: dict):
        self.sd = sd

    def names(self):
        return self.sd.keys()

    def has(self, name: str) -> bool:
        return name in self.sd

    def tensor(self, name: str) -> torch.Tensor:
        return self.sd[name]

    def linear(self, base: str) -> torch.Tensor:
        return self.sd[base + ".weight"]


class Checkpoint:
    """Tensors by name over model.safetensors.index.json's shards (bf16 or int4 GPTQ).

    `linear(base)` returns the [out, in] bf16 weight: `base.weight` as stored, or
    `base.qweight` + `.scales` dequantised by dequant.py's rule ((q - 8) * scale, fp32 product,
    one RNE cast). GPTQ preconditions are checked per tensor: sequential g_idx (k // group) and
    v1 qzeros of 0x77777777 (zero point 8). `fast` uses stream.py's C++ dequant_t (bit-identical,
    self-checking, needs g++ and ninja)."""
    QZ = 0x77777777                # v1 qzeros: eight nibbles of 7 (zero point 8 stored minus 1)

    def __init__(self, snapshot: str, fast: bool | None = None):
        from safetensors import safe_open
        self.snapshot = snapshot
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                self.wm = json.load(f)["weight_map"]
        else:
            fn = "model.safetensors"
            with safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu") as h:
                self.wm = {k: fn for k in h.keys()}
        self.handles = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
                        for fn in sorted(set(self.wm.values()))}
        self.cfg = K2Config.from_file(snapshot)
        if fast is None:
            import shutil
            fast = self.cfg.group_size is not None and bool(shutil.which("g++") and shutil.which("ninja"))
        self.fast = fast

    def names(self):
        return self.wm.keys()

    def has(self, name: str) -> bool:
        return name in self.wm

    def tensor(self, name: str) -> torch.Tensor:
        return self.handles[self.wm[name]].get_tensor(name)

    def linear(self, base: str) -> torch.Tensor:
        if base + ".weight" in self.wm:
            return self.tensor(base + ".weight")
        if base + ".qweight" not in self.wm:
            raise KeyError(f"{base}: neither .weight nor .qweight in the checkpoint")
        g = self.cfg.group_size
        qw, sc = self.tensor(base + ".qweight"), self.tensor(base + ".scales")
        K = qw.shape[0] * 8
        if base + ".g_idx" in self.wm:
            gi = self.tensor(base + ".g_idx")
            if not torch.equal(gi.long(), torch.arange(K) // g):
                raise ValueError(f"{base}.g_idx is not sequential (desc_act ordering is not supported)")
        if base + ".qzeros" in self.wm:
            qz = self.tensor(base + ".qzeros")
            if not bool((qz == self.QZ).all()):
                raise ValueError(f"{base}.qzeros is not all 0x77777777 (symmetric, zero point 8)")
        if self.fast:
            return stream_mod().dequant_t(qw, sc, g)
        return _dequant.dequant_gptq(qw, sc, g).t().contiguous()


class LayerWeights:
    """One decoder layer: the dense part loaded up front, experts read on first use."""

    def __init__(self, src, c: K2Config, i: int, dense: dict):
        self.src, self.c, self.i, self.d = src, c, i, dense
        self.cache: dict = {}

    def __getitem__(self, k: str) -> torch.Tensor:
        return self.d[k]

    def get(self, k: str):
        return self.d.get(k)

    def expert(self, e: int):
        key = ("moe", e)
        if key not in self.cache:
            b = f"model.layers.{self.i}.mlp.experts.{e}."
            self.cache[key] = tuple(self.src.linear(b + n + "_proj") for n in ("gate", "up", "down"))
        return self.cache[key]

    def v_expert(self, e: int) -> torch.Tensor:
        key = ("mova", e)
        if key not in self.cache:
            self.cache[key] = self.src.linear(f"model.layers.{self.i}.self_attn.v_experts.{e}")
        return self.cache[key]


def load_dense_layer(src, c: K2Config, i: int) -> dict:
    """Layer i's non-expert tensors, keys relative to the layer (stream.py's layer_sd shape)."""
    p = f"model.layers.{i}."
    d = {"input_layernorm": src.tensor(p + "input_layernorm.weight"),
         "post_attention_layernorm": src.tensor(p + "post_attention_layernorm.weight")}
    for n in ("q_proj", "k_proj", "o_proj"):
        d[n] = src.linear(p + "self_attn." + n)
    if c.attention_gate_func is not None:
        d["gate_proj"] = src.linear(p + "self_attn.gate_proj")
    if c.query_key_norm:
        d["q_norm"] = src.tensor(p + "self_attn.q_norm.weight")
        d["k_norm"] = src.tensor(p + "self_attn.k_norm.weight")
    if c.is_mova(i):
        d["v_router"] = src.linear(p + "self_attn.v_router")
        d["v_router.bias"] = src.tensor(p + "self_attn.v_router.bias") if c.moe_gate_bias else None
    else:
        d["v_proj"] = src.linear(p + "self_attn.v_proj")
    if c.is_sparse(i):
        d["gate"] = src.linear(p + "mlp.gate")
        d["gate.bias"] = src.tensor(p + "mlp.gate.bias") if c.moe_gate_bias else None
        if c.num_shared_experts:
            for n in ("gate", "up", "down"):
                d["shared." + n] = src.linear(p + f"mlp.shared_experts.{n}_proj")
    else:
        for n in ("gate", "up", "down"):
            d["mlp." + n] = src.linear(p + f"mlp.{n}_proj")
    return d


# --------------------------------------------------------------------------------------------
# the model

class Recorder:
    """What a forward leaves behind, in the golden layout (plan 8d's names, spec 18 §6 K2)."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len
        self.t: dict[str, list] = {}

    def add(self, name: str, x: torch.Tensor) -> None:
        self.t.setdefault(name, []).append(x.detach().clone())

    def tensors(self, ar: Arith) -> dict:
        act = torch.bfloat16 if ar.mode == "bf16" else torch.float32
        out = {}
        for k, v in self.t.items():
            x = torch.cat(v, dim=0)
            if k.startswith(("resid.", "mixer.", "mlp.")):
                out[k] = x[: self.prompt_len].to(act).contiguous()
            elif ".ids." in k:
                out[k] = x.to(torch.int32).contiguous()
            elif k.startswith("route.") and ".w." in k or k.startswith("rope."):
                out[k] = x.to(act).contiguous()
            else:
                out[k] = x.float().contiguous()
        return out


class K2Ref:
    """The reference, layer at a time. `src` is a Checkpoint or a DictSource."""

    def __init__(self, c: K2Config, src, mode: str = "bf16", prefetch: bool = True, keep_dense=(),
                 keep_experts=()):
        """keep_dense: layers whose dense tensors stay loaded after their first forward;
        keep_experts: layers whose dequantised experts (MoE and MoVA) are kept once read, instead
        of being dropped with the layer - each read once per run. Neither changes a value."""
        self.c, self.src, self.ar = c, src, Arith(mode)
        self.embed = src.tensor("model.embed_tokens.weight")
        self.norm_w = src.tensor("model.norm.weight")
        self.lm_head = self.embed if c.tie_word_embeddings else src.tensor("lm_head.weight")
        self.keep_dense, self.keep_experts = set(keep_dense), set(keep_experts)
        self.dense: dict[int, dict] = {}
        self.experts: dict[int, dict] = {i: {} for i in self.keep_experts}
        self.pf = None
        if prefetch:
            self.pf = stream_mod()._Prefetch(lambda i: load_dense_layer(src, c, i), c.num_hidden_layers,
                                             keep=self.keep_dense)
        self.load_seconds = 0.0

    def layer(self, i: int) -> LayerWeights:
        t = time.time()
        d = self.dense.get(i)
        if d is None:
            d = self.pf.get(i) if self.pf else load_dense_layer(self.src, self.c, i)
            if i in self.keep_dense:
                self.dense[i] = d
        self.load_seconds += time.time() - t
        lw = LayerWeights(self.src, self.c, i, d)
        if i in self.keep_experts:
            lw.cache = self.experts[i]
        return lw

    def new_cache(self) -> list:
        return [None] * self.c.num_hidden_layers

    # one decoder layer --------------------------------------------------------------------
    def attn(self, i: int, lw: LayerWeights, x: torch.Tensor, pos: torch.Tensor, cos, sin, cache: list,
             rec: Recorder | None) -> torch.Tensor:
        c, ar = self.c, self.ar
        T, d, H, Hkv = x.shape[0], c.head_dim, c.num_attention_heads, c.num_key_value_heads
        if c.is_mova(i):
            logits = ar.linear(x, lw["v_router"])
            rt = route(ar, logits, lw.get("v_router.bias"), c.mova_num_experts_per_tok,
                       c.router_scaling_factor, normalise=c.mova_num_experts_per_tok > 1)
            v = combine_ascending(ar, rt, x, lambda e, xs: ar.r(F.silu(ar.linear(xs, lw.v_expert(e)))), Hkv * d)
            if rec:
                rec.add(f"route.mova.ids.L{i}", rt.ids)
                rec.add(f"route.mova.w.L{i}", rt.weights)
                rec.add(f"route.mova.gap.L{i}", rt.gap)
        else:
            v = ar.linear(x, lw["v_proj"])
        q = ar.linear(x, lw["q_proj"])
        k = ar.linear(x, lw["k_proj"])
        if c.query_key_norm:
            q = grouped_rms_norm(ar, q, lw["q_norm"], H, c.rms_norm_eps)
            k = grouped_rms_norm(ar, k, lw["k_norm"], Hkv, c.rms_norm_eps)
        q = apply_rope(ar, q.view(T, H, d).transpose(0, 1), cos, sin)
        k = apply_rope(ar, k.view(T, Hkv, d).transpose(0, 1), cos, sin)
        v = v.view(T, Hkv, d).transpose(0, 1)
        if cache[i] is not None:                       # the KV cache: post-RoPE K, the (MoVA) V
            pk, pv, pp = cache[i]
            k, v, kpos = torch.cat([pk, k], 1), torch.cat([pv, v], 1), torch.cat([pp, pos])
        else:
            kpos = pos
        cache[i] = (k, v, kpos)
        o = attention(ar, q, k, v, pos, kpos, c.n_rep)          # [T, H, d]
        if c.attention_gate_func is not None:
            g = attn_gate(ar, ar.linear(x, lw["gate_proj"]).view(T, H, d), c.attention_gate_func)
            o = ar.r(o * g)
        return ar.linear(o.reshape(T, H * d), lw["o_proj"])

    def ffn(self, i: int, lw: LayerWeights, x: torch.Tensor, rec: Recorder | None) -> torch.Tensor:
        c, ar = self.c, self.ar
        if not c.is_sparse(i):
            return mlp(ar, x, lw["mlp.gate"], lw["mlp.up"], lw["mlp.down"])
        logits = ar.linear(x, lw["gate"])
        rt = route(ar, logits, lw.get("gate.bias"), c.num_experts_per_tok, c.router_scaling_factor,
                   normalise=c.norm_topk_prob)
        if rec:
            rec.add(f"route.moe.ids.L{i}", rt.ids)
            rec.add(f"route.moe.w.L{i}", rt.weights)
            rec.add(f"route.moe.gap.L{i}", rt.gap)
        y = combine_ascending(ar, rt, x, lambda e, xs: mlp(ar, xs, *lw.expert(e)), c.hidden_size)
        if c.num_shared_experts:
            y = ar.r(y + mlp(ar, x, lw["shared.gate"], lw["shared.up"], lw["shared.down"]))
        return y

    def decoder_layer(self, i: int, lw: LayerWeights, h: torch.Tensor, pos, cos, sin, cache: list,
                      rec: Recorder | None = None) -> torch.Tensor:
        c, ar = self.c, self.ar
        G, eps = c.layernorm_num_groups, c.rms_norm_eps
        a = self.attn(i, lw, grouped_rms_norm(ar, h, lw["input_layernorm"], G, eps), pos, cos, sin, cache, rec)
        h = ar.r(h + a)
        m = self.ffn(i, lw, grouped_rms_norm(ar, h, lw["post_attention_layernorm"], G, eps), rec)
        h = ar.r(h + m)
        if rec:
            rec.add(f"mixer.L{i}", a)
            rec.add(f"mlp.L{i}", m)
            rec.add(f"resid.L{i}", h)
        return h

    # the whole model ----------------------------------------------------------------------
    def _embed(self, ids, pos0: int, rec: Recorder | None = None):
        c, ar = self.c, self.ar
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
        pos = torch.arange(pos0, pos0 + ids.numel())
        cos, sin = rope_cos_sin(ar, pos, c.rope_dim, c.rope_theta)
        if rec:
            rec.add("rope.cos", cos)
            rec.add("rope.sin", sin)
        return self.embed[ids].float(), pos, cos, sin

    def _final(self, h: torch.Tensor) -> torch.Tensor:
        c = self.c
        return self.head(grouped_rms_norm(self.ar, h, self.norm_w, c.layernorm_num_groups, c.rms_norm_eps))

    @torch.no_grad()
    def forward(self, ids, pos0: int, cache: list, rec: Recorder | None = None) -> torch.Tensor:
        """Tokens `ids` at positions pos0.. through every layer; returns fp32 logits [T, vocab]."""
        h, pos, cos, sin = self._embed(ids, pos0, rec)
        for i in range(self.c.num_hidden_layers):
            lw = self.layer(i)
            h = self.decoder_layer(i, lw, h, pos, cos, sin, cache, rec)
            del lw
        return self._final(h)

    @torch.no_grad()
    def forward_many(self, items: list) -> list[torch.Tensor]:
        """Layer-major forward() over several sequences: items = [(ids, pos0, cache)], each its own
        sequence with its own cache (ragged lengths and positions). Every item goes through layer
        0, then every item through layer 1, ...: a layer's weights - its dense tensors and each
        routed expert, dequantised once - are read once for all items. Each item's arithmetic is
        forward()'s - the same calls on the same tensors, one item at a time, nothing concatenated
        across items - so its logits are bitwise forward()'s whatever else is in the batch.
        Returns each item's LAST logits row (fp32 [vocab]): the full [T, vocab] head is computed
        (as forward() does) and dropped item by item, so a batch of prompts never holds B of them."""
        st = []
        for ids, pos0, cache in items:
            h, pos, cos, sin = self._embed(ids, pos0)
            st.append([h, pos, cos, sin, cache])
        for i in range(self.c.num_hidden_layers):
            lw = self.layer(i)
            for s in st:
                s[0] = self.decoder_layer(i, lw, s[0], s[1], s[2], s[3], s[4])
            del lw
        out = []
        for s in st:
            out.append(self._final(s[0])[-1].clone())
            s.clear()
        return out

    def head(self, x: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        """lm_head: one bf16 GEMM in bf16 mode (as the reference); in f32 mode in vocab-row chunks
        (an fp32 copy of the 250624 x 2560 head would be 2.6 GB)."""
        if self.ar.mode == "bf16":
            return self.ar.linear(x, self.lm_head)
        return torch.cat([self.ar.linear(x, self.lm_head[a:a + chunk])
                          for a in range(0, self.lm_head.shape[0], chunk)], dim=-1)

    def generate(self, ids, n: int, rec: Recorder | None = None, log=None):
        """Prompt forward, then n greedy steps through the cache. Returns (logits [T + n, V], tokens [n]),
        dump.py's convention: tokens[0] = argmax(logits[T - 1]); logits[T + j] consumed tokens[j]."""
        cache = self.new_cache()
        ids = list(ids)
        t = time.time()
        rows = [self.forward(ids, 0, cache, rec)]
        if log:
            log(f"prompt forward ({len(ids)} ids): {time.time() - t:.1f}s")
        toks: list[int] = []
        for step in range(n):
            nxt = int(torch.argmax(rows[-1][-1]))
            toks.append(nxt)
            t = time.time()
            rows.append(self.forward([nxt], len(ids) + step, cache, rec))
            if log:
                log(f"  step {step}: token {nxt}, {time.time() - t:.1f}s")
        return torch.cat(rows, dim=0), toks


# --------------------------------------------------------------------------------------------
# the checkpoint's own modeling code (vendored), for cross-checks

def hf_module():
    """third_party/k2_horizon/modeling_k2_horizon.py, imported as a package (its relative import)."""
    name = "k2_horizon_vendored"
    if name + ".modeling_k2_horizon" in sys.modules:
        return sys.modules[name + ".modeling_k2_horizon"]
    pkg = types.ModuleType(name)
    pkg.__path__ = [THIRD_PARTY]
    sys.modules[name] = pkg
    return importlib.import_module(name + ".modeling_k2_horizon")


def build_hf(config: dict, device: str = "cpu", dtype=torch.bfloat16):
    """K2HorizonForCausalLM from a config.json dict, eager attention, in `dtype`."""
    m = hf_module()
    conf_mod = sys.modules["k2_horizon_vendored.configuration_k2_horizon"]
    keep = {k: v for k, v in config.items() if k not in ("auto_map", "architectures", "quantization_config",
                                                           "transformers_version", "model_type", "torch_dtype")}
    keep["dtype"] = str(dtype).replace("torch.", "")
    cfg = conf_mod.K2HorizonConfig(**keep)
    cfg._attn_implementation = "eager"
    with torch.device(device):
        model = m.K2HorizonForCausalLM(cfg).to(dtype)
    if device != "meta":
        # .to(dtype) also casts the rotary inv_freq buffer; from_pretrained keeps it fp32 (it is
        # built from an explicit float arange), so rebuild it. On meta, stream.attach rebuilds it.
        model.model.rotary_emb = m.K2HorizonRotaryEmbedding(cfg)
    model.eval()
    return model, cfg


def hf_layer_sd(src, c: K2Config, i: int) -> dict:
    """Layer i's full state dict for the vendored model (keys relative to the layer)."""
    p = f"model.layers.{i}."
    out = {}
    for name in expected_names(c):
        if not name.startswith(p):
            continue
        rel = name[len(p):]
        if name.endswith(".weight") and src.has(name[:-len(".weight")] + ".qweight"):
            out[rel] = src.linear(name[:-len(".weight")])
        else:
            out[rel] = src.tensor(name)
    return out


def hf_streamed(src, config: dict, dtype=torch.bfloat16):
    """The vendored model on meta, layers streamed by stream.attach (the spec 14 pattern)."""
    st = stream_mod()
    c = K2Config.from_dict(config)
    model, cfg = build_hf(config, device="meta", dtype=dtype)
    m = hf_module()
    resident = {"model.embed_tokens.weight": src.tensor("model.embed_tokens.weight").to(dtype),
                "model.norm.weight": src.tensor("model.norm.weight").to(dtype)}
    if not c.tie_word_embeddings:
        resident["lm_head.weight"] = src.tensor("lm_head.weight").to(dtype)
    # Biases keep their stored dtype (the int4 checkpoint's v_router.bias is f16; assign=True keeps
    # it, and the modeling code widens it to fp32 before the add, as the port does).
    pf = st.attach(model, model.model.layers,
                   lambda i: {k: v if k.endswith(".bias") else v.to(dtype) for k, v in hf_layer_sd(src, c, i).items()},
                   resident, rebuild=[(model.model, "rotary_emb", lambda: m.K2HorizonRotaryEmbedding(cfg))])
    return model, pf


# --------------------------------------------------------------------------------------------
# CLI

def _read_ids(path: str) -> list[int]:
    with open(path, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    if not ids:
        raise SystemExit(f"{path} is empty")
    return ids


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    print(f"torch {torch.__version__}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})", flush=True)
    src = Checkpoint(a.snapshot, fast=None if not a.slow_dequant else False)
    c = src.cfg
    missing = [n for n in expected_names(c) if not (src.has(n) or src.has(n[:-len('.weight')] + ".qweight"))]
    if missing:
        raise SystemExit(f"checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")
    ids = _read_ids(a.prompt)
    if len(ids) > a.max_prompt:
        raise SystemExit(f"prompt is {len(ids)} ids, --max-prompt is {a.max_prompt}")
    print(f"{a.snapshot}: {c.num_hidden_layers} layers ({sum(map(c.is_mova, range(c.num_hidden_layers)))} MoVA+MoE), "
          f"{'int4 g%d dequantised' % c.group_size if c.group_size else 'bf16'}; mode {a.mode}; "
          f"prompt {len(ids)} ids; gen {a.gen}", flush=True)
    ref = K2Ref(c, src, mode=a.mode)
    rec = Recorder(len(ids))
    logits, toks = ref.generate(ids, a.gen, rec, log=lambda s: print(s, flush=True))
    out = rec.tensors(ref.ar)
    out["logits"] = logits.float().contiguous()
    out["tokens"] = torch.tensor(toks, dtype=torch.int32)
    last = out[f"resid.L{c.num_hidden_layers - 1}"]
    if not torch.isfinite(last.float()).all():
        raise SystemExit("the last residual is not finite - the weights are wrong")
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt_ids": " ".join(map(str, ids)),
            "n_prompt": str(len(ids)), "gen": str(a.gen), "mode": a.mode,
            "reference": "tools/oracle/k2_ref.py (port of modeling_k2_horizon.py, layer at a time)",
            "weights": f"int4 g{c.group_size} dequant.py rule" if c.group_size else "bf16 as stored",
            "torch": torch.__version__, "threads": str(torch.get_num_threads())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(out, a.out, metadata=meta)
    gaps = torch.cat([v.flatten() for k, v in out.items() if k.startswith("route.moe.gap.")]) if c.num_experts else None
    if gaps is not None and gaps.numel():
        print(f"MoE 8th/9th selection gap: min {float(gaps.min()):.3e}, "
              f"#<1e-4 {int((gaps < 1e-4).sum())} of {gaps.numel()}, #==0 {int((gaps == 0).sum())}")
    mg = [v.flatten() for k, v in out.items() if k.startswith("route.mova.gap.")]
    if mg:
        mg = torch.cat(mg)
        print(f"MoVA 4th/5th selection gap: min {float(mg.min()):.3e}, "
              f"#<1e-4 {int((mg < 1e-4).sum())} of {mg.numel()}, #==0 {int((mg == 0).sum())}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"tokens {toks}")
    print(f"wrote {a.out}: {len(out)} tensors; layer-load wait {ref.load_seconds:.1f}s; "
          f"wall {time.time() - t0:.1f}s; peak RSS {rss:.1f} GiB")


def cmd_hfcheck(a) -> None:
    """Prompt forward through the vendored HF model (layer-streamed) against the port."""
    src = Checkpoint(a.snapshot, fast=None if not a.slow_dequant else False)
    with open(os.path.join(a.snapshot, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    ids = _read_ids(a.prompt)
    t = time.time()
    model, _ = hf_streamed(src, config)
    with torch.no_grad():
        hf = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    print(f"HF (vendored, streamed, eager bf16) prompt forward: {time.time() - t:.1f}s", flush=True)
    if a.against:
        from safetensors import safe_open
        with safe_open(a.against, framework="pt") as h:
            ours = h.get_tensor("logits")[: len(ids)]
    else:
        ref = K2Ref(src.cfg, src, mode="bf16")
        ours = ref.forward(ids, 0, ref.new_cache())
    diff = (hf - ours).abs()
    agree = (hf.argmax(-1) == ours.argmax(-1)).float().mean()
    cos = F.cosine_similarity(hf, ours, dim=-1)
    print(f"logits vs port: max |diff| {float(diff.max()):.4g}, bitwise rows "
          f"{int((diff.amax(-1) == 0).sum())}/{len(ids)}, argmax agree {float(agree):.3f}, "
          f"min cosine {float(cos.min()):.6f}")


def cmd_facts(a) -> None:
    c = K2Config.from_file(a.path)
    L = c.num_hidden_layers
    kinds = [c.layer_kind(i) for i in range(L)]
    runs, start = [], 0
    for i in range(1, L + 1):
        if i == L or kinds[i] != kinds[start]:
            runs.append(f"{start}-{i - 1} {kinds[start]}" if i - 1 > start else f"{start} {kinds[start]}")
            start = i
    print(f"layers {L}: " + ", ".join(runs))
    print(f"hidden {c.hidden_size}, norm groups {c.layernorm_num_groups} x {c.hidden_size // c.layernorm_num_groups}, "
          f"eps {c.rms_norm_eps}; heads {c.num_attention_heads}/{c.num_key_value_heads} (GQA {c.n_rep}) x "
          f"{c.head_dim}; rope dim {c.rope_dim} theta {c.rope_theta:g}; qk norm {c.query_key_norm}; "
          f"gate {c.attention_gate_func}")
    print(f"MoE {c.num_experts} x {c.moe_intermediate_size} top {c.num_experts_per_tok} + {c.num_shared_experts} shared, "
          f"norm {c.norm_topk_prob}, scale {c.router_scaling_factor}, bias {c.moe_gate_bias}; "
          f"MoVA {c.mova_num_experts} x ({c.hidden_size} -> {c.num_key_value_heads * c.head_dim}) top "
          f"{c.mova_num_experts_per_tok}; dense MLP {c.intermediate_size}")
    snap = a.path if os.path.isdir(a.path) else os.path.dirname(a.path)
    gen_eos = None
    if os.path.exists(os.path.join(snap, "generation_config.json")):
        with open(os.path.join(snap, "generation_config.json"), encoding="utf-8") as f:
            gen_eos = json.load(f).get("eos_token_id")
    print(f"vocab {c.vocab_size}, BOS {c.bos_token_id}, EOS {c.eos_token_ids} (config.json), "
          f"{gen_eos} (generation_config.json), tied {c.tie_word_embeddings}, "
          f"quant {'int4 g%d' % c.group_size if c.group_size else 'none'}")
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        want = set(expected_names(c))
        have = {n[:-len(".qweight")] + ".weight" if n.endswith(".qweight") else n for n in wm
                if not n.endswith((".scales", ".qzeros", ".g_idx"))}
        print(f"index: {len(wm)} entries; expected {len(want)} tensors; missing {sorted(want - have)[:5]}, "
              f"unexpected {sorted(have - want)[:5]}")
        quant = sorted({n.split("layers.")[1].split(".", 1)[1].rsplit(".", 1)[0] if "layers." in n else n
                        for n in wm if n.endswith(".qweight")})
        plain = sorted({n.split("layers.")[1].split(".", 1)[1] if "layers." in n else n
                        for n in wm if not n.endswith((".qweight", ".scales", ".qzeros", ".g_idx"))})
        import re
        squash = (lambda s: re.sub(r"experts\.\d+", "experts.E", s))
        print("int4: " + ", ".join(sorted(set(map(squash, quant)))) if quant else "int4: none")
        print("plain: " + ", ".join(sorted(set(map(squash, plain)))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--prompt", required=True, help="file of whitespace-separated token ids")
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=64)
    r.add_argument("--mode", choices=("bf16", "f32"), default="bf16")
    r.add_argument("--slow-dequant", action="store_true", help="dequant.py's torch path instead of the C++ one")
    h = sub.add_parser("hfcheck")
    h.add_argument("snapshot")
    h.add_argument("--prompt", required=True)
    h.add_argument("--against", help="a `run` output whose logits rows to compare (default: recompute)")
    h.add_argument("--slow-dequant", action="store_true")
    f = sub.add_parser("facts")
    f.add_argument("path", help="snapshot directory or config.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    {"run": cmd_run, "hfcheck": cmd_hfcheck, "facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
