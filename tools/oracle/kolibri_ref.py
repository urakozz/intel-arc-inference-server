#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kolibri 1 (Aleph Alpha, `Aleph-Alpha/Kolibri-1-BF16`) reference - spec 20a.

The semantics are Aleph Alpha's vLLM plugin (`aleph_alpha_inference/kolibri1.py` @ 049a6a7b,
Apache-2.0) on top of vLLM's Qwen3-MoE; third_party/kolibri1/modeling_kolibri1.py is our transformers
port of the same rules (what AutoRound loads). This file is the layer-at-a-time reference: it reads
the bf16 checkpoint or an int4 AutoRound GPTQ export (dequantised by dequant.py's rule) by tensor
name, one decoder layer at a time, and dumps what spec 20 §5 needs: per-layer residuals, the routing
per layer per position (ids ascending, weights, the selection-score gap at the top-6 cut), every
logits row, the greedy ids, and the prompt's perplexity. `run_batch` runs many sequences
layer-major (each layer's weights read once for all of them) - the quantisation helpers use it for
the expert-coverage count and the KL evaluation.

    kolibri_ref.py run <snapshot> --prompt <ids-file> --out <file.safetensors> [--gen 32] [--mode bf16]
    kolibri_ref.py ppl <snapshot> --ids <ids-file> [<ids-file> ...] [--device cpu]
    kolibri_ref.py hfcheck <snapshot> --prompt <ids-file> [--against <file.safetensors>]
    kolibri_ref.py facts <snapshot-or-config.json>

Semantics (plugin = aleph_alpha_inference/kolibri1.py; Qwen3-MoE = vllm's qwen3_moe.py):

  layer      x += post_attn_norm(Attn(input_layernorm(x)))                   plugin Kolibri1DecoderLayer
             x += post_ffn_norm(MoE(post_attention_layernorm(x)))            (sandwich norms)
  RMSNorm    w * bf16(x * rsqrt(mean(x^2) + eps)), fp32 statistics, PLAIN w   vLLM rms_norm / HF Qwen3
  attention  q, k, v = W x; q/k RMSNorm per head over head_dim; sliding layers then RoPE (neox
             rotate_half, theta 1e4, all 128 dims), full layers NoPE; scale 128^-0.5; GQA head h reads
             kv head h // 12. Sliding window W = 513 keys INCLUDING the query: key j is visible to
             query i iff i - W < j <= i (vLLM's per_layer_sliding_window, transformers' sliding cache
             keeping W - 1 = 512 past keys; the model card: "512 preceding tokens plus the current")
  router     logits = fp32(x) @ fp32(W_gate)^T (GateLinear out_dtype fp32); select top 6 on
             logits + fp32(expert_bias); weights sigmoid(logits[sel]), NOT renormalised
             (norm_topk_prob false)                                           sigmoid_logit_add_routing
  experts    y_e = down(silu(gate x) * up x); shared expert the same, ungated (no sigmoid scalar gate,
             unlike Qwen2-MoE's shared_expert_gate), added to the routed sum    Qwen3MoeMLP, FusedMoE
  head       logits = fp32(h) @ fp32(lm_head)^T: config `head_dtype: float32` is vLLM's
             ModelConfig.head_dtype - LogitsProcessor runs the lm_head in fp32 (bf16 hidden state and
             weights widened exactly, fp32 accumulation, fp32 logits) instead of the model dtype

What the plugin leaves to the implementation, and this reference fixes (the engine follows it):
  * top-6 ties go to the LOWER expert id (stable descending sort); torch.topk leaves ties unspecified;
  * the combine: sum over the selected experts in ASCENDING id of fp32(w_e) * fp32(bf16 y_e), in
    fp32, then + fp32(bf16 shared), ONE rounding to bf16 (FusedMoE's order is unspecified);
  * rounding points follow transformers' Qwen3-MoE in bf16: every linear output, each elementwise
    op, the residual adds, norms once, eager attention (scores bf16, x scale bf16, softmax fp32 ->
    bf16, P V bf16). vLLM's fused add+RMSNorm normalises the fp32 sum before rounding the residual
    (ulp-level, not a semantic difference).

Modes: "bf16" (default) computes in bf16 tensors exactly where the port computes in bf16 and in
fp32 where the semantics say fp32 - on the same torch and device it is bit-identical to
third_party/kolibri1/modeling_kolibri1.py in bf16 with eager attention, prompt and cached decode
(test_kolibri_ref.py). "f32" never rounds (= the port in fp32, within fp32 tolerance).

The KV cache keeps every position for full layers and the last W - 1 = 512 for sliding layers (the
ring the engine keeps), so a cached decode reads exactly what a full forward reads.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

THIRD_PARTY = os.path.join(_HERE, "third_party", "kolibri1")
LAYER_TYPES = ("sliding_attention", "full_attention")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"kolibri_oracle_{name}", os.path.join(_HERE, f"{name}.py"))
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
class KConfig:
    """config.json (the fields the semantics read), with the real model's values as defaults."""
    vocab_size: int = 128000
    hidden_size: int = 2560
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int | None = 513
    layer_types: list = field(default_factory=list)
    num_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    tie_word_embeddings: bool = False
    head_dtype: str | None = "float32"
    bos_token_id: int | None = None
    eos_token_ids: list = field(default_factory=list)
    group_size: int | None = None      # GPTQ group size when the checkpoint is quantised

    @classmethod
    def from_dict(cls, d: dict) -> "KConfig":
        if d.get("model_type") not in (None, "kolibri1"):
            raise ValueError(f"model_type {d.get('model_type')!r} is not kolibri1")
        if d.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {d.get('hidden_act')!r}: Kolibri 1's experts are SiLU")
        if d.get("attention_bias"):
            raise ValueError("attention_bias: Kolibri 1 has none")
        if d.get("norm_topk_prob"):
            raise ValueError("norm_topk_prob: Kolibri 1 does not renormalise")
        rp = d.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rp.get('rope_type')!r}: only 'default' is implemented")
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in names}
        kw["rope_theta"] = float(rp.get("rope_theta", d.get("rope_theta", 10000.0)))
        if not d.get("use_sliding_window", True):
            kw["sliding_window"] = None
        eos = d.get("eos_token_id")
        kw["eos_token_ids"] = list(eos) if isinstance(eos, list) else ([] if eos is None else [eos])
        q = d.get("quantization_config")
        if q:
            if q.get("bits") != 4 or not q.get("sym") or q.get("desc_act"):
                raise ValueError(f"quantization_config: only int4 symmetric without desc_act, got {q}")
            if q.get("quant_method") not in ("gptq", "auto-round"):
                raise ValueError(f"quant_method {q.get('quant_method')!r}")
            kw["group_size"] = int(q["group_size"])
        c = cls(**kw)
        if not c.layer_types:
            raise ValueError("config.json has no layer_types")
        if len(c.layer_types) != c.num_hidden_layers or set(c.layer_types) - set(LAYER_TYPES):
            raise ValueError(f"layer_types: {c.layer_types}")
        if "sliding_attention" in c.layer_types and not (c.sliding_window and c.sliding_window > 0):
            raise ValueError("sliding layers need sliding_window > 0")
        if c.num_attention_heads % c.num_key_value_heads:
            raise ValueError("num_attention_heads % num_key_value_heads != 0")
        return c

    @classmethod
    def from_file(cls, path: str) -> "KConfig":
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"

    @property
    def head_fp32(self) -> bool:
        return self.head_dtype in ("float32", "fp32")


def expected_names(c: KConfig) -> list[str]:
    """Every tensor the bf16 checkpoint ships, by name."""
    out = ["model.embed_tokens.weight", "model.norm.weight"]
    if not c.tie_word_embeddings:
        out.append("lm_head.weight")
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        out += [p + n + ".weight" for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm",
                                            "post_ffn_norm")]
        out += [p + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        out += [p + "mlp.gate.weight", p + "moe.router.expert_bias"]
        out += [p + f"mlp.shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")]
        for e in range(c.num_experts):
            out += [p + f"mlp.experts.{e}.{n}_proj.weight" for n in ("gate", "up", "down")]
    return out


# --------------------------------------------------------------------------------------------
# arithmetic

class Ops:
    """Activation dtype and device. bf16 mode keeps activations in bf16 tensors and runs torch's bf16
    ops (the port's ops, so the results are the port's bits); f32 mode keeps everything in fp32."""

    def __init__(self, mode: str = "bf16", device: str = "cpu"):
        if mode not in ("bf16", "f32"):
            raise ValueError(f"mode {mode!r}: bf16 or f32")
        self.mode, self.device = mode, torch.device(device)
        self.dt = torch.bfloat16 if mode == "bf16" else torch.float32

    def w(self, t: torch.Tensor) -> torch.Tensor:
        """A weight in the activation dtype on the device (bf16 weights are exact in fp32)."""
        return t.to(device=self.device, dtype=self.dt)

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w(w))


def rms_norm(ops: Ops, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """w * act(x * rsqrt(mean(x^2) + eps)): statistics in fp32, x_hat rounded once, then the product
    in the activation dtype (vLLM `x.to(weight.dtype) * weight`, HF Qwen3 `weight * x.to(dt)`)."""
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return ops.w(w) * h.to(ops.dt)


def rope_cos_sin(ops: Ops, positions: torch.Tensor, head_dim: int, theta: float):
    """[T, head_dim] each in the activation dtype: f = pos * theta^(-2i/d) (fp32), cat(f, f)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=ops.device) / head_dim))
    freqs = (inv[None, :, None] @ positions.to(ops.device).float()[None, None, :]).transpose(1, 2)[0]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(ops.dt), emb.sin().to(ops.dt)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def visible(q_idx: torch.Tensor, k_idx: torch.Tensor, window: int | None) -> torch.Tensor:
    """[Tq, Tk] bool: causal, and inside the window when there is one: q - W < k <= q."""
    ok = k_idx[None, :] <= q_idx[:, None]
    if window is not None:
        ok = ok & ((q_idx[:, None] - k_idx[None, :]) < window)
    return ok


def attention(ops: Ops, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ok: torch.Tensor, n_rep: int):
    """Eager attention as transformers runs it: q [B, H, T, d], k/v [B, Hkv, L, d], ok [Tq, Tk] or
    [B, 1, Tq, Tk] bool -> [B, T, H * d]. scores = (q k^T) * d^-0.5 in the activation dtype, + 0 or
    finfo.min, softmax in fp32, rounded, P V."""
    B, H, T, d = q.shape
    kk = k.repeat_interleave(n_rep, dim=1)
    vv = v.repeat_interleave(n_rep, dim=1)
    s = torch.matmul(q, kk.transpose(2, 3)) * (d ** -0.5)
    s = s + torch.where(ok, 0.0, torch.finfo(s.dtype).min).to(s.dtype)
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(p, vv)
    return o.transpose(1, 2).reshape(B, T, H * d)


@dataclass
class Route:
    ids: torch.Tensor        # [N, k] int64, ascending expert id
    weights: torch.Tensor    # [N, k] fp32, sigmoid(logit) of each id (not renormalised)
    gap: torch.Tensor        # [N] fp32: selection score of the k-th minus the (k+1)-th (0 = a tie at the cut)
    sel_order: torch.Tensor  # [N, k] int64, the ids in selection order (descending score, ties lower id)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int) -> Route:
    """plugin sigmoid_logit_add_routing: select on fp32 logits + fp32 bias, weights sigmoid(logits) of
    the selected, no renormalisation. Ties at equal selection score go to the lower id (stable)."""
    logits = logits.float()
    sel = logits + bias.float().to(logits.device)
    vals, order = torch.sort(sel, dim=-1, descending=True, stable=True)
    top = order[:, :k]
    gap = (vals[:, k - 1] - vals[:, k]) if sel.shape[-1] > k else torch.full(sel.shape[:-1], float("inf"))
    w_sel = torch.sigmoid(logits.gather(1, top))
    ids, perm = torch.sort(top, dim=-1)
    return Route(ids=ids, weights=torch.gather(w_sel, -1, perm), gap=gap, sel_order=top)


def mlp(ops: Ops, x: torch.Tensor, gate, up, down) -> torch.Tensor:
    """down(silu(gate x) * up x) in the activation dtype (Qwen3MoeMLP / the port's Kolibri1MLP)."""
    return ops.linear(F.silu(ops.linear(x, gate)) * ops.linear(x, up), down)


def moe(ops: Ops, x: torch.Tensor, rt: Route, expert_fn, shared) -> torch.Tensor:
    """routed = sum over each row's selected experts in ASCENDING id of fp32(w_e) * fp32(y_e) in fp32,
    + fp32(shared(x)), one rounding to the activation dtype. x [N, D]."""
    acc = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for e in torch.unique(rt.ids).tolist():                 # ascending
        rows, slot = torch.where(rt.ids == e)
        y = expert_fn(e, x[rows])
        acc.index_add_(0, rows, y.float() * rt.weights[rows, slot, None].to(x.device))
    return (acc + shared(x).float()).to(ops.dt)


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
    """Tensors by name over model.safetensors.index.json's shards (bf16, or an int4 GPTQ export).

    `linear(base)` returns the [out, in] bf16 weight: `base.weight` as stored, or `base.qweight` +
    `.scales` dequantised by dequant.py's rule ((q - 8) * scale, fp32 product, one RNE cast). GPTQ
    preconditions are checked per tensor: sequential g_idx (k // group) when shipped and v1 qzeros
    of 0x77777777 (zero point 8). `fast` uses stream.py's C++ dequant_t (bit-identical)."""
    QZ = 0x77777777

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
        self.cfg = KConfig.from_file(snapshot)
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
    """One decoder layer: the dense part up front, experts read on first use (or all at once)."""

    def __init__(self, src, c: KConfig, i: int, dense: dict):
        self.src, self.c, self.i, self.d = src, c, i, dense
        self.cache: dict = {}

    def __getitem__(self, k: str) -> torch.Tensor:
        return self.d[k]

    def expert(self, e: int):
        if e not in self.cache:
            b = f"model.layers.{self.i}.mlp.experts.{e}."
            self.cache[e] = tuple(self.src.linear(b + n + "_proj") for n in ("gate", "up", "down"))
        return self.cache[e]


def load_dense_layer(src, c: KConfig, i: int) -> dict:
    """Layer i's non-expert tensors, keys relative to the layer."""
    p = f"model.layers.{i}."
    d = {n: src.tensor(p + n + ".weight") for n in ("input_layernorm", "post_attn_norm",
                                                     "post_attention_layernorm", "post_ffn_norm")}
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        d[n] = src.linear(p + "self_attn." + n)
    d["q_norm"] = src.tensor(p + "self_attn.q_norm.weight")
    d["k_norm"] = src.tensor(p + "self_attn.k_norm.weight")
    d["gate"] = src.tensor(p + "mlp.gate.weight")
    d["expert_bias"] = src.tensor(p + "moe.router.expert_bias")
    for n in ("gate", "up", "down"):
        d["shared." + n] = src.linear(p + f"mlp.shared_experts.{n}_proj")
    return d


# --------------------------------------------------------------------------------------------
# the model

class Recorder:
    """What a forward leaves behind, in the golden layout (k2_ref's names)."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len
        self.t: dict[str, list] = {}

    def add(self, name: str, x: torch.Tensor) -> None:
        self.t.setdefault(name, []).append(x.detach().cpu().clone())

    def tensors(self, ops: Ops) -> dict:
        out = {}
        for k, v in self.t.items():
            x = torch.cat(v, dim=0)
            if k.startswith(("resid.", "mixer.", "mlp.")):
                out[k] = x[: self.prompt_len].to(ops.dt).contiguous()
            elif ".ids." in k:
                out[k] = x.to(torch.int32).contiguous()
            elif k.startswith("rope."):
                out[k] = x.to(ops.dt).contiguous()
            else:
                out[k] = x.float().contiguous()
        return out


class KolibriRef:
    """The reference, layer at a time. `src` is a Checkpoint or a DictSource."""

    def __init__(self, c: KConfig, src, mode: str = "bf16", device: str = "cpu", prefetch: bool = True):
        self.c, self.src, self.ops = c, src, Ops(mode, device)
        self.embed = src.tensor("model.embed_tokens.weight")
        self.norm_w = src.tensor("model.norm.weight")
        self.lm_head = self.embed if c.tie_word_embeddings else src.tensor("lm_head.weight")
        self.pf = None
        if prefetch:
            self.pf = stream_mod()._Prefetch(lambda i: load_dense_layer(src, c, i), c.num_hidden_layers)
        self.load_seconds = 0.0

    def layer(self, i: int) -> LayerWeights:
        t = time.time()
        d = self.pf.get(i) if self.pf else load_dense_layer(self.src, self.c, i)
        self.load_seconds += time.time() - t
        return LayerWeights(self.src, self.c, i, d)

    def new_cache(self) -> list:
        return [None] * self.c.num_hidden_layers

    # one decoder layer, a batch of B rows of T tokens -----------------------------------------
    def attn(self, i: int, lw: LayerWeights, x: torch.Tensor, q_idx: torch.Tensor, cos, sin, cache,
             pad: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, D] (input_layernorm output); q_idx [T] absolute positions; cache: None (no cache)
        or the per-layer list (B == 1). pad [B, T] bool (True = a real token) masks padded keys."""
        c, ops = self.c, self.ops
        B, T, _ = x.shape
        d, H, Hkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        q = ops.linear(x, lw["q_proj"]).view(B, T, H, d)
        k = ops.linear(x, lw["k_proj"]).view(B, T, Hkv, d)
        v = ops.linear(x, lw["v_proj"]).view(B, T, Hkv, d)
        q = rms_norm(ops, q, lw["q_norm"], c.rms_norm_eps).transpose(1, 2)
        k = rms_norm(ops, k, lw["k_norm"], c.rms_norm_eps).transpose(1, 2)
        v = v.transpose(1, 2)
        window = c.sliding_window if c.is_sliding(i) else None
        if c.is_sliding(i):                                  # RoPE in sliding layers only (NoPE in full)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        k_idx = q_idx
        if cache is not None:
            if cache[i] is not None:
                pk, pv, pidx = cache[i]
                k, v, k_idx = torch.cat([pk, k], 2), torch.cat([pv, v], 2), torch.cat([pidx, q_idx])
            keep = k.shape[2] if window is None else window - 1     # the sliding layers' ring
            cache[i] = (k[:, :, -keep:], v[:, :, -keep:], k_idx[-keep:]) if keep > 0 else (k[:, :, :0], v[:, :, :0], k_idx[:0])
        ok = visible(q_idx.to(ops.device), k_idx.to(ops.device), window)
        if pad is not None:
            ok = ok[None, None] & pad.to(ops.device)[:, None, None, :]
        o = attention(ops, q, k, v, ok, c.n_rep)
        return ops.linear(o, lw["o_proj"])

    def ffn(self, i: int, lw: LayerWeights, x: torch.Tensor) -> tuple[torch.Tensor, Route]:
        """x [N, D] (post_attention_layernorm output) -> (MoE output [N, D], the routing)."""
        c, ops = self.c, self.ops
        logits = F.linear(x.float(), lw["gate"].to(device=ops.device, dtype=torch.float32))
        rt = route(logits, lw["expert_bias"].to(ops.device), c.num_experts_per_tok)
        y = moe(ops, x, rt, lambda e, xs: mlp(ops, xs, *lw.expert(e)),
                lambda xs: mlp(ops, xs, lw["shared.gate"], lw["shared.up"], lw["shared.down"]))
        return y, rt

    def decoder_layer(self, i, lw, h, q_idx, cos, sin, cache, rec=None, pad=None, on_route=None):
        c, ops = self.c, self.ops
        B, T, D = h.shape
        a = self.attn(i, lw, rms_norm(ops, h, lw["input_layernorm"], c.rms_norm_eps), q_idx, cos, sin, cache, pad)
        a = rms_norm(ops, a, lw["post_attn_norm"], c.rms_norm_eps)
        h = h + a
        x = rms_norm(ops, h, lw["post_attention_layernorm"], c.rms_norm_eps)
        m, rt = self.ffn(i, lw, x.reshape(B * T, D))
        m = rms_norm(ops, m.view(B, T, D), lw["post_ffn_norm"], c.rms_norm_eps)
        h = h + m
        if rec:
            rec.add(f"route.moe.ids.L{i}", rt.ids)
            rec.add(f"route.moe.w.L{i}", rt.weights)
            rec.add(f"route.moe.gap.L{i}", rt.gap)
            rec.add(f"mixer.L{i}", a[0])
            rec.add(f"mlp.L{i}", m[0])
            rec.add(f"resid.L{i}", h[0])
        if on_route is not None:
            on_route(i, rt)
        return h

    # the whole model --------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids, pos0: int, cache: list, rec: Recorder | None = None) -> torch.Tensor:
        """Tokens `ids` at positions pos0.. through every layer; returns fp32 logits [T, vocab]."""
        c, ops = self.c, self.ops
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
        T = ids.numel()
        pos = torch.arange(pos0, pos0 + T)
        cos, sin = rope_cos_sin(ops, pos, c.head_dim, c.rope_theta)
        if rec:
            rec.add("rope.cos", cos)
            rec.add("rope.sin", sin)
        h = ops.w(self.embed[ids])[None]
        for i in range(c.num_hidden_layers):
            lw = self.layer(i)
            h = self.decoder_layer(i, lw, h, pos, cos, sin, cache, rec)
            del lw
        return self.head(rms_norm(ops, h[0], self.norm_w, c.rms_norm_eps))

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """lm_head; fp32 operands and logits when head_dtype is float32 (the checkpoint's setting)."""
        if self.c.head_fp32:
            return F.linear(x.float(), self.lm_head.to(device=x.device, dtype=torch.float32))
        return F.linear(x, self.ops.w(self.lm_head)).float()

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

    # layer-major over many sequences ------------------------------------------------------------
    @torch.no_grad()
    def run_batch(self, seqs: torch.Tensor, rows: int = 1, on_route=None, log=None,
                  load_all_experts: bool = True) -> torch.Tensor:
        """seqs [N, T] int64 (no padding): every sequence through layer 0, then every sequence
        through layer 1, ... - each layer's weights are read (and, for an int4 checkpoint,
        dequantised) once for all N. `rows` sequences share one forward (attention per row, the MoE
        over all rows' tokens). on_route(layer, seq_index_tensor [rows*T], Route) sees every
        routing decision. Returns the final-norm hidden states [N, T, D] in the activation dtype on
        the CPU (the head is the caller's: evaluation needs fp32 logits a chunk at a time)."""
        c, ops = self.c, self.ops
        N, T = seqs.shape
        h = torch.empty(N, T, c.hidden_size, dtype=ops.dt)
        for a in range(0, N, 64):
            h[a:a + 64] = self.embed[seqs[a:a + 64]].to(ops.dt)
        pos = torch.arange(T)
        cos, sin = rope_cos_sin(ops, pos, c.head_dim, c.rope_theta)
        for i in range(c.num_hidden_layers):
            t0 = time.time()
            lw = self.layer(i)
            if load_all_experts:
                for e in range(c.num_experts):
                    lw.cache[e] = tuple(ops.w(t) for t in lw.expert(e))
            for a in range(0, N, rows):
                b = min(N, a + rows)
                cb = None
                if on_route is not None:
                    seq_ix = torch.arange(a, b).repeat_interleave(T)
                    cb = (lambda layer, rt, s=seq_ix: on_route(layer, s, rt))
                out = self.decoder_layer(i, lw, h[a:b].to(ops.device), pos, cos, sin, None, on_route=cb)
                h[a:b] = out.to("cpu")
            del lw
            if log:
                log(f"layer {i} ({c.layer_types[i]}): {N} x {T} tokens, {time.time() - t0:.1f}s")
        for a in range(0, N, rows):
            h[a:a + rows] = rms_norm(ops, h[a:a + rows].to(ops.device), self.norm_w, c.rms_norm_eps).to("cpu")
        return h


def nll(logits: torch.Tensor, ids) -> torch.Tensor:
    """Per-position negative log-likelihood of ids[t + 1] under logits[t] (fp32), [T - 1]."""
    ids = torch.as_tensor(ids, dtype=torch.long).flatten()
    lp = torch.log_softmax(logits[: ids.numel() - 1].float(), dim=-1)
    return -lp.gather(1, ids[1:, None])[:, 0]


# --------------------------------------------------------------------------------------------
# the transformers port (third_party/kolibri1), for cross-checks

def hf_module():
    """third_party/kolibri1/modeling_kolibri1.py, imported as a package (its relative import)."""
    name = "kolibri1_vendored"
    if name + ".modeling_kolibri1" in sys.modules:
        return sys.modules[name + ".modeling_kolibri1"]
    pkg = types.ModuleType(name)
    pkg.__path__ = [THIRD_PARTY]
    sys.modules[name] = pkg
    return importlib.import_module(name + ".modeling_kolibri1")


def hf_config(config: dict, dtype=torch.bfloat16, attn: str = "eager"):
    hf_module()
    conf_mod = sys.modules["kolibri1_vendored.configuration_kolibri1"]
    keep = {k: v for k, v in config.items() if k not in ("auto_map", "architectures", "quantization_config",
                                                           "transformers_version", "model_type", "torch_dtype")}
    keep["dtype"] = str(dtype).replace("torch.", "")
    cfg = conf_mod.Kolibri1Config(**keep)
    cfg._attn_implementation = attn
    return cfg


def build_hf(config: dict, device: str = "cpu", dtype=torch.bfloat16, attn: str = "eager"):
    """Kolibri1ForCausalLM from a config.json dict, in `dtype`."""
    m = hf_module()
    cfg = hf_config(config, dtype, attn)
    with torch.device(device):
        model = m.Kolibri1ForCausalLM(cfg).to(dtype)
    model.eval()
    return model, cfg


def hf_layer_sd(src, c: KConfig, i: int) -> dict:
    p = f"model.layers.{i}."
    out = {}
    for name in expected_names(c):
        if not name.startswith(p):
            continue
        rel = name[len(p):]
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if base and src.has(base + ".qweight"):
            out[rel] = src.linear(base)
        else:
            out[rel] = src.tensor(name)
    return out


def hf_streamed(src, config: dict, dtype=torch.bfloat16):
    """The port on meta, layers streamed by stream.attach (the spec 14 pattern)."""
    st = stream_mod()
    c = KConfig.from_dict(config)
    model, cfg = build_hf(config, device="meta", dtype=dtype)
    resident = {"model.embed_tokens.weight": src.tensor("model.embed_tokens.weight").to(dtype),
                "model.norm.weight": src.tensor("model.norm.weight").to(dtype)}
    if not c.tie_word_embeddings:
        resident["lm_head.weight"] = src.tensor("lm_head.weight").to(dtype)
    pf = st.attach(model, model.model.layers,
                   lambda i: {k: v.to(dtype) for k, v in hf_layer_sd(src, c, i).items()}, resident)
    return model, pf


# --------------------------------------------------------------------------------------------
# CLI

def _read_ids(path: str) -> list[int]:
    with open(path, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    if not ids:
        raise SystemExit(f"{path} is empty")
    return ids


def _check_names(src, c):
    missing = [n for n in expected_names(c) if not (src.has(n) or src.has(n[:-len('.weight')] + ".qweight"))]
    if missing:
        raise SystemExit(f"checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    print(f"torch {torch.__version__}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})", flush=True)
    src = Checkpoint(a.snapshot, fast=None if not a.slow_dequant else False)
    c = src.cfg
    _check_names(src, c)
    ids = _read_ids(a.prompt)
    if len(ids) > a.max_prompt:
        raise SystemExit(f"prompt is {len(ids)} ids, --max-prompt is {a.max_prompt}")
    print(f"{a.snapshot}: {c.num_hidden_layers} layers ({c.layer_types.count('full_attention')} full), "
          f"{'int4 g%d dequantised' % c.group_size if c.group_size else 'bf16'}; mode {a.mode}; "
          f"prompt {len(ids)} ids; gen {a.gen}", flush=True)
    ref = KolibriRef(c, src, mode=a.mode, device=a.device)
    rec = Recorder(len(ids))
    logits, toks = ref.generate(ids, a.gen, rec, log=lambda s: print(s, flush=True))
    out = rec.tensors(ref.ops)
    out["logits"] = logits.float().contiguous()
    out["tokens"] = torch.tensor(toks, dtype=torch.int32)
    out["nll"] = nll(logits[: len(ids)], ids)
    last = out[f"resid.L{c.num_hidden_layers - 1}"]
    if not torch.isfinite(last.float()).all():
        raise SystemExit("the last residual is not finite - the weights are wrong")
    ppl = float(torch.exp(out["nll"].mean())) if len(ids) > 1 else float("nan")
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt_ids": " ".join(map(str, ids)),
            "n_prompt": str(len(ids)), "gen": str(a.gen), "mode": a.mode, "ppl": f"{ppl:.6g}",
            "reference": "tools/oracle/kolibri_ref.py (Kolibri 1, layer at a time)",
            "weights": f"int4 g{c.group_size} dequant.py rule" if c.group_size else "bf16 as stored",
            "torch": torch.__version__, "threads": str(torch.get_num_threads()), "device": a.device}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(out, a.out, metadata=meta)
    gaps = torch.cat([v.flatten() for k, v in out.items() if k.startswith("route.moe.gap.")])
    print(f"MoE 6th/7th selection gap: min {float(gaps.min()):.3e}, "
          f"#<1e-4 {int((gaps < 1e-4).sum())} of {gaps.numel()}, #==0 {int((gaps == 0).sum())}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"prompt perplexity {ppl:.4g}; tokens {toks}")
    print(f"wrote {a.out}: {len(out)} tensors; layer-load wait {ref.load_seconds:.1f}s; "
          f"wall {time.time() - t0:.1f}s; peak RSS {rss:.1f} GiB")


def cmd_ppl(a) -> None:
    """Perplexity of each ids file (teacher-forced, all positions), layer-major."""
    src = Checkpoint(a.snapshot, fast=None if not a.slow_dequant else False)
    c = src.cfg
    _check_names(src, c)
    ref = KolibriRef(c, src, mode=a.mode, device=a.device)
    for path in a.ids:
        ids = _read_ids(path)[: a.max_len]
        h = ref.run_batch(torch.tensor([ids]), log=(lambda s: print(s, flush=True)) if a.verbose else None)
        logits = ref.head(h[0].to(ref.ops.device)).cpu()
        x = nll(logits, ids)
        print(f"{path}: {len(ids)} ids, perplexity {float(torch.exp(x.mean())):.4g} "
              f"(mean NLL {float(x.mean()):.4f})", flush=True)


def cmd_hfcheck(a) -> None:
    """Prompt forward through the port (layer-streamed, eager bf16) against the reference."""
    src = Checkpoint(a.snapshot, fast=None if not a.slow_dequant else False)
    with open(os.path.join(a.snapshot, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    ids = _read_ids(a.prompt)
    t = time.time()
    model, _ = hf_streamed(src, config)
    with torch.no_grad():
        hf = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    print(f"port (streamed, eager bf16) prompt forward: {time.time() - t:.1f}s", flush=True)
    if a.against:
        from safetensors import safe_open
        with safe_open(a.against, framework="pt") as h:
            ours = h.get_tensor("logits")[: len(ids)]
    else:
        ref = KolibriRef(src.cfg, src, mode="bf16")
        ours = ref.forward(ids, 0, ref.new_cache())
    diff = (hf - ours).abs()
    agree = (hf.argmax(-1) == ours.argmax(-1)).float().mean()
    cos = F.cosine_similarity(hf, ours, dim=-1)
    print(f"logits vs reference: max |diff| {float(diff.max()):.4g}, bitwise rows "
          f"{int((diff.amax(-1) == 0).sum())}/{len(ids)}, argmax agree {float(agree):.3f}, "
          f"min cosine {float(cos.min()):.6f}")


def cmd_facts(a) -> None:
    c = KConfig.from_file(a.path)
    L = c.num_hidden_layers
    full = [i for i in range(L) if not c.is_sliding(i)]
    print(f"layers {L}: full attention (NoPE) at {full}, the other {L - len(full)} sliding "
          f"(window {c.sliding_window}, RoPE theta {c.rope_theta:g})")
    print(f"hidden {c.hidden_size}, eps {c.rms_norm_eps}; heads {c.num_attention_heads}/{c.num_key_value_heads} "
          f"(GQA {c.n_rep}) x {c.head_dim}; MoE {c.num_experts} x {c.moe_intermediate_size} top "
          f"{c.num_experts_per_tok} + shared {c.shared_expert_intermediate_size}; head_dtype {c.head_dtype}")
    snap = a.path if os.path.isdir(a.path) else os.path.dirname(a.path)
    gen_eos = None
    if os.path.exists(os.path.join(snap, "generation_config.json")):
        with open(os.path.join(snap, "generation_config.json"), encoding="utf-8") as f:
            gen_eos = json.load(f).get("eos_token_id")
    print(f"vocab {c.vocab_size}, BOS {c.bos_token_id}, EOS {c.eos_token_ids} (config.json), {gen_eos} "
          f"(generation_config.json), tied {c.tie_word_embeddings}, "
          f"quant {'int4 g%d' % c.group_size if c.group_size else 'none'}")
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        import re
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        want = set(expected_names(c))
        have = {n[:-len(".qweight")] + ".weight" if n.endswith(".qweight") else n for n in wm
                if not n.endswith((".scales", ".qzeros", ".g_idx"))}
        print(f"index: {len(wm)} entries; expected {len(want)} tensors; missing {sorted(want - have)[:5]}, "
              f"unexpected {sorted(have - want)[:5]}")
        squash = (lambda s: re.sub(r"experts\.\d+", "experts.E", re.sub(r"layers\.\d+\.", "layers.L.", s)))
        quant = sorted({squash(n[:-len(".qweight")]) for n in wm if n.endswith(".qweight")})
        plain = sorted({squash(n) for n in wm if not n.endswith((".qweight", ".scales", ".qzeros", ".g_idx"))})
        print("int4: " + (", ".join(quant) if quant else "none"))
        print("plain: " + ", ".join(plain))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--prompt", required=True, help="file of whitespace-separated token ids")
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=4096)
    r.add_argument("--mode", choices=("bf16", "f32"), default="bf16")
    r.add_argument("--device", default="cpu")
    r.add_argument("--slow-dequant", action="store_true", help="dequant.py's torch path instead of the C++ one")
    p = sub.add_parser("ppl")
    p.add_argument("snapshot")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--mode", choices=("bf16", "f32"), default="bf16")
    p.add_argument("--device", default="cpu")
    p.add_argument("--slow-dequant", action="store_true")
    p.add_argument("--verbose", action="store_true")
    h = sub.add_parser("hfcheck")
    h.add_argument("snapshot")
    h.add_argument("--prompt", required=True)
    h.add_argument("--against", help="a `run` output whose logits rows to compare (default: recompute)")
    h.add_argument("--slow-dequant", action="store_true")
    f = sub.add_parser("facts")
    f.add_argument("path", help="snapshot directory or config.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    {"run": cmd_run, "ppl": cmd_ppl, "hfcheck": cmd_hfcheck, "facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
