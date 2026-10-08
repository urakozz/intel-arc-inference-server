#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# A CPU reference of the EAGLE-3 drafter `Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3` (vLLM
# Speculators 0.8.0's `Eagle3DraftModel`, Apache-2.0) for the EAGLE3 K2 P0
# (docs/probe-eagle3-k2-2026-10-08.md). A re-implementation in plain torch, not a copy. The
# semantics are pinned from source, file:line against:
#   speculators 0.8.0 (PyPI wheel)     speculators/models/eagle3/{core,model_definitions,data,
#                                      attention}.py, speculators/train/vocab_mapping.py,
#                                      speculators/model.py
#   vLLM v0.28.0 (the plugin's pin)    vllm/model_executor/models/{llama_eagle3,interfaces,
#                                      llama}.py, vllm/v1/spec_decode/llm_base_proposer.py,
#                                      vllm/v1/worker/gpu_model_runner.py
#   k2-horizon-vllm @ 13ee389          k2_horizon_vllm/model.py (the target's aux capture)
"""EAGLE-3 reference drafter: CPU, float32 compute, deterministic.

    d = load_drafter(snapshot)                          # or Drafter(cfg, tensors)
    ctx = d.context(ids, aux, aux_from)                 # step 0 over every row (target taps)
    drafts = d.propose(ctx, p, K)                       # K greedy target ids for anchor p
    drafts = d.propose_batch(ctx, anchors, K)           # the same, B anchors at once

    eagle3_ref.py facts <drafter snapshot> [--k2 <K2 snapshot>]

**S1 taps.** Aux id a = the target's residual stream at the INPUT of decoder layer a (= the
output of layer a - 1, pre-norm; a = 0 is the embedding): the plugin calls
`_maybe_add_hidden_state(aux, 0, h)` before layer 0 and `(aux, idx + 1, h)` after layer idx
(k2_horizon_vllm/model.py:340-345); interfaces.py:1499-1509 appends h at index a; vLLM adds 1 to
DFlash's output-of-layer ids to get these (gpu_model_runner.py:5629-5633). The aux states are
concatenated in layer order (gpu_model_runner.py:5326-5328), [L2 | L24 | L45] for this drafter.
The FINAL hidden state is a training target only (verifier_last_hidden_states, core.py:248-253);
inference never reads it.

**S2 fusion.** fused = fc(input_norm(cat aux)) - norm_before_fc: ONE RMSNorm over all 7680
(core.py:236-244, llama_eagle3.py:367-379); fc_norm (per-chunk norms) is false here.

**S3 the decoder layer** (model_definitions.py:71-110, llama_eagle3.py:99-119):
    e = input_layernorm(embed(token)); h = hidden_norm(fused or fed-back hidden)
    residual = h if norm_before_residual else the un-normed hidden          (true here)
    x = cat([e, h])  - the EMBEDDING FIRST - -> q / k / v (in 2 x 2560)
    h = residual + o_proj(attn(x)); h = h + mlp(post_attention_layernorm(h)); out = norm(h)
Llama pieces: plain RMSNorm w * x_hat (one group, eps 1e-6), RoPE rotate_half over all 128 dims
at theta 1e4, GQA 32 / 8, SiLU-GLU MLP, no biases.

**S4 fed-back hidden.** norm_output: the POST-norm `out` feeds the next step (core.py:302-306,
llama_eagle3.py:249-252); logits = lm_head(out) either way.

**S5 row alignment.** The row of token x_s carries the aux of position s - 1 (data.py:24-35's
shift; llm_base_proposer.py:854-864 rotates the ids by one and keeps the target positions).
RoPE position: training uses s (data.py:35), vLLM s - 1; attention only sees differences, so
the drafts agree (test_eagle3_ref.py checks). This file uses s; rows exist for s >= 1.

**S6 draft steps** (core.py:269-357, llm_base_proposer.py:690-770). Step 0 is the anchor's
row p (token x[p], aux of p - 1): d_0 = its greedy id, the drafter's guess for position p + 1.
Step j >= 1: token d_{j-1} (a TARGET id, embedded by embed_tokens), the previous step's
fed-back hidden, position p + j; d_j = greedy. Greedy = argmax of the 32768-row head, the first
maximum (torch / vLLM argmax), mapped to the target vocabulary (S8).

**S7 attention.** The context is step 0 run over every row with the TARGET's taps: vLLM re-runs
the drafter over each verify's rows (llm_base_proposer.py:510-603), so the K/V of every
committed position comes from target hidden states, never from a draft. Causal with a sliding
window of 2048 (layer_types sliding_attention; training attention.py:16-20 `kv > q - W`; vLLM
per_layer_sliding_window llama.py:183-217 -> flash_attn.py:850 / triton_attn.py:534, the window
(W - 1, 0)). Step j sees the step-0 keys of rows <= p and its own chain's keys
(attention.py:29-30 + 115-132: the diagonal blocks). The window for those step-0
keys: vLLM measures it from the query's own position p + j (`window="query"`, the default:
what a served engine does), training from the anchor row p (`window="anchor"`); they differ
only in the j oldest keys of a full window.

**S8 vocab maps.** target id = draft id + d2t[draft id] (vocab_mapping.py:88-91,
interfaces.py:1465-1469, llama_eagle3.py:347-348); t2d [250624] bool marks the selected target
ids, ascending in draft id (vocab_mapping.py:86, 93-94). `check_vocab_maps` holds the
checkpoint to that.

**S9 embedding.** The drafter ships embed_tokens [250624, 2560] (embed_requires_grad false: the
verifier's, loaded when NaN, model.py:165-170); `facts --k2` compares it with K2's bytes.

Arms (`quantize`): bf16 (as shipped, float32 compute); int8 = every body linear (fc, q / k / v /
o, gate / up / down) int8 per output row, symmetric RTN s = amax / 127, q = rne(w / s) in
[-127, 127] (dflash_accept.rtn_int8_rows, spec 9's gemv_i8w form), head bf16; int8h = int8 + the
32k head int8 the same way; int4h = the body int4 g64 symmetric RTN exactly as
src/loader/rtn.h (s = f16(2 amax / 15), q = clamp(rint(w / s) + 8, 0, 15), w = (q - 8) s) + the
head int8. Norms and the embedding stay bf16.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tokenize.py shadows the stdlib

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

try:
    import torch
except ImportError:          # the config / bytes helpers run without it (the drivers' dry runs)
    torch = None

ARMS = ("bf16", "int8", "int8h", "int4h")
BODY_LINEARS = ("fc.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
                "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight")
REPO_ID = "Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3"


# --------------------------------------------------------------------------------------------
# config


@dataclass
class Eagle3Config:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    eps: float
    rope_theta: float
    window: int | None            # sliding window (keys with q - k <= window - 1), None = full
    vocab: int                    # the target's (embed_tokens rows, t2d length)
    draft_vocab: int              # lm_head rows (d2t length)
    aux_ids: list
    target_hidden: int
    norm_before_fc: bool
    fc_norm: bool
    norm_before_residual: bool
    norm_output: bool
    speculative_tokens: int       # the checkpoint's greedy proposal default (vLLM serves 2)
    verifier: str
    raw: dict

    @property
    def gqa(self) -> int:
        return self.heads // self.kv_heads

    @property
    def n_aux(self) -> int:
        return len(self.aux_ids)


def config_from_dict(c: dict) -> Eagle3Config:
    """The speculators config.json, held to what this file implements (refused otherwise)."""
    if c.get("speculators_model_type") != "eagle3":
        raise ValueError(f"speculators_model_type {c.get('speculators_model_type')!r} is not eagle3")
    t = c["transformer_layer_config"]
    if t.get("model_type", "llama") != "llama":
        raise NotImplementedError(f"transformer_layer_config model_type {t.get('model_type')!r}: llama only "
                                  "(qwen3's q/k norms are not implemented)")
    if int(t.get("num_hidden_layers", 1)) != 1:
        raise NotImplementedError("num_hidden_layers != 1: vLLM's layers >= 1 take a hidden-only input "
                                  "(llama_eagle3.py:104-106), not implemented")
    if t.get("hidden_act", "silu") != "silu" or t.get("attention_bias") or t.get("mlp_bias"):
        raise NotImplementedError("hidden_act silu, no attention / mlp bias only")
    rp = t.get("rope_parameters") or {"rope_type": (t.get("rope_scaling") or {}).get("rope_type", "default"),
                                      "rope_theta": t.get("rope_theta", 10000.0)}
    if rp.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope_type {rp.get('rope_type')!r}")
    if c.get("fc_norm") and c.get("norm_before_fc"):
        raise ValueError("norm_before_fc and fc_norm are mutually exclusive (config.py _check_norm_flags)")
    types = t.get("layer_types") or ["full_attention"]
    window = int(t["sliding_window"]) if types[0] == "sliding_attention" and t.get("sliding_window") else None
    H = int(t["hidden_size"])
    aux = list(c.get("eagle_aux_hidden_state_layer_ids") or [])
    if not aux:
        raise ValueError("eagle_aux_hidden_state_layer_ids missing (resolve_target_layer_ids needs the verifier)")
    pm = (c.get("speculators_config") or {}).get("proposal_methods") or [{}]
    return Eagle3Config(
        hidden=H, heads=int(t["num_attention_heads"]), kv_heads=int(t["num_key_value_heads"]),
        head_dim=int(t.get("head_dim") or H // int(t["num_attention_heads"])),
        inter=int(t["intermediate_size"]), eps=float(t.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rp.get("rope_theta", 10000.0)), window=window, vocab=int(t["vocab_size"]),
        draft_vocab=int(c.get("draft_vocab_size") or t["vocab_size"]), aux_ids=aux,
        target_hidden=int(c.get("target_hidden_size") or H),
        norm_before_fc=bool(c.get("norm_before_fc", False)), fc_norm=bool(c.get("fc_norm", False)),
        norm_before_residual=bool(c.get("norm_before_residual", False)),
        norm_output=bool(c.get("norm_output", False)),
        speculative_tokens=int(pm[0].get("speculative_tokens", 0) or 0),
        verifier=str(((c.get("speculators_config") or {}).get("verifier") or {}).get("name_or_path", "")),
        raw=c)


def read_config(snapshot: str) -> Eagle3Config:
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def expected_shapes(c: Eagle3Config) -> dict:
    """Every tensor the drafter needs, canonical name -> shape (embed_tokens may come from K2)."""
    H, A = c.hidden, c.n_aux * c.target_hidden
    L = "layers.0."
    s = {"fc.weight": (H, A),
         L + "input_layernorm.weight": (H,), L + "hidden_norm.weight": (H,),
         L + "self_attn.q_proj.weight": (c.heads * c.head_dim, 2 * H),
         L + "self_attn.k_proj.weight": (c.kv_heads * c.head_dim, 2 * H),
         L + "self_attn.v_proj.weight": (c.kv_heads * c.head_dim, 2 * H),
         L + "self_attn.o_proj.weight": (H, c.heads * c.head_dim),
         L + "post_attention_layernorm.weight": (H,),
         L + "mlp.gate_proj.weight": (c.inter, H), L + "mlp.up_proj.weight": (c.inter, H),
         L + "mlp.down_proj.weight": (H, c.inter),
         "norm.weight": (H,), "lm_head.weight": (c.draft_vocab, H),
         "d2t": (c.draft_vocab,), "t2d": (c.vocab,)}
    if c.norm_before_fc:
        s["input_norm.weight"] = (A,)
    if c.fc_norm:
        for i in range(c.n_aux):
            s[f"fc_norm.{i}.weight"] = (c.target_hidden,)
    return s


# --------------------------------------------------------------------------------------------
# weights


def hf_hub_dir() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def find_snapshot(repo_id: str, need=("config.json",)) -> str | None:
    """The HF cache snapshot of `repo_id` whose files `need` all resolve; never downloads."""
    base = os.path.join(hf_hub_dir(), "models--" + repo_id.replace("/", "--"))
    for s in sorted(glob.glob(os.path.join(base, "snapshots", "*"))):
        if all(os.path.isfile(os.path.join(s, n)) for n in need):
            return s
    return None


def canon(name: str) -> str:
    """vLLM's renames (llama_eagle3.py:256-258 midlayer. -> layers.0.; 406 the model. prefix)."""
    if name.startswith("model."):
        name = name[len("model."):]
    return name.replace("midlayer.", "layers.0.")


def read_tensors(snapshot: str) -> dict:
    """Every tensor of the drafter checkpoint under canonical names (single file or index)."""
    from safetensors import safe_open
    idx = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    out = {}
    for fn in files:
        with safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu") as h:
            for k in h.keys():
                n = canon(k)
                if n in out:
                    raise ValueError(f"{k}: two tensors map to {n}")
                out[n] = h.get_tensor(k)
    return out


def check_vocab_maps(d2t: torch.Tensor, t2d: torch.Tensor) -> torch.Tensor:
    """S8: returns the target id of every draft id (draft + d2t[draft]), after holding the maps
    to vocab_mapping.py's construction: in range, strictly increasing in draft id, exactly
    t2d's set. Raises on anything else (e.g. a d2t that stores target ids directly)."""
    d2t, t2d = d2t.long(), t2d.bool()
    tgt = torch.arange(d2t.numel()) + d2t
    if int(tgt.min()) < 0 or int(tgt.max()) >= t2d.numel():
        raise ValueError(f"draft + d2t out of the target vocabulary [{int(tgt.min())}, {int(tgt.max())}]")
    if not bool((tgt[1:] > tgt[:-1]).all()):
        raise ValueError("draft + d2t is not strictly increasing (vocab_mapping.py sorts the selected ids)")
    sel = t2d.nonzero().flatten()
    if sel.numel() != tgt.numel() or not torch.equal(sel, tgt):
        raise ValueError(f"t2d marks {sel.numel()} ids, not the {tgt.numel()} of draft + d2t")
    return tgt


def rtn_int8_rows(w: torch.Tensor) -> torch.Tensor:
    """int8 per output row, symmetric RTN (s = amax / 127, q = rne(w / s), [-127, 127]),
    dequantised to float32 (dflash_accept.rtn_int8_rows' rule)."""
    wf = w.float()
    s = wf.abs().amax(dim=1) / 127.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round(wf / safe[:, None]), -127, 127)
    return q * s[:, None]


def rtn_int4_g64(w: torch.Tensor, group: int = 64) -> torch.Tensor:
    """src/loader/rtn.h's int4 g64 symmetric RTN of a [N][K] linear, dequantised to float32:
    per row n and group of 64 k, s16 = f16(2 amax / 15), q = clamp(rint(w / f32(s16)) + 8, 0, 15),
    w' = (q - 8) * s16; an all-zero group gives 0."""
    N, K = w.shape
    if K % group:
        raise ValueError(f"K {K} is not a multiple of {group}")
    wf = w.float().reshape(N, K // group, group)
    s = (2.0 * wf.abs().amax(-1) / 15.0).to(torch.float16).float()            # RNE to f16
    q = torch.where(s[..., None] > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))[..., None]) + 8,
                    torch.full_like(wf, 8.0))
    q = q.clamp(0, 15)
    return ((q - 8) * s[..., None]).reshape(N, K)


# --------------------------------------------------------------------------------------------
# math (float32)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """LlamaRMSNorm: w * (x * rsqrt(mean(x^2) + eps)), one group, the plain weight."""
    x = x.float()
    return w.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope_cos_sin(pos: torch.Tensor, head_dim: int, theta: float):
    """LlamaRotaryEmbedding (default): inv_freq in fp32, freqs = pos * inv_freq, cat(f, f)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    f = pos.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [N, heads, hd]; rotate_half (neox) over all dims."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos[:, None] + rot * sin[:, None]


def greedy_ids(logits: torch.Tensor) -> torch.Tensor:
    """argmax over the draft vocabulary, the first maximum on ties (torch.argmax / vLLM)."""
    best = logits.max(-1, keepdim=True).values
    idx = torch.arange(logits.shape[-1])
    return torch.where(logits == best, idx, logits.shape[-1]).min(-1).values


@dataclass
class Context:
    """Step 0 over rows s = row_from .. N-1 (row s: token ids[s], aux of s - 1, position s)."""
    pos: torch.Tensor      # [R] the rows' positions
    k: torch.Tensor        # [R, kv_heads, hd] post-RoPE
    v: torch.Tensor        # [R, kv_heads, hd]
    hid: torch.Tensor      # [R, H] the fed-back hidden of each row (S4)
    d0: torch.Tensor       # [R] each row's greedy TARGET id (the row's d_0)

    def row(self, p: int) -> int:
        i = p - int(self.pos[0])
        if not (0 <= i < self.pos.numel()) or int(self.pos[i]) != p:
            raise IndexError(f"position {p} is not a context row ({int(self.pos[0])}..{int(self.pos[-1])})")
        return i


class Drafter:
    def __init__(self, cfg: Eagle3Config, tensors: dict, embed: torch.Tensor | None = None,
                 pos_offset: int = 0):
        """`tensors` under canonical names (read_tensors); the embedding is the checkpoint's or
        `embed` (K2's). Linear weights are held in float32. `pos_offset` shifts every RoPE
        position (-1 = vLLM's convention, S5)."""
        self.cfg, self.pos_offset = cfg, pos_offset
        want = expected_shapes(cfg)
        e = tensors.get("embed_tokens.weight", embed)
        if e is None:
            raise ValueError("the drafter ships no embed_tokens: pass K2's embedding")
        if tuple(e.shape) != (cfg.vocab, cfg.target_hidden):
            raise ValueError(f"embed_tokens {tuple(e.shape)}, want {(cfg.vocab, cfg.target_hidden)}")
        missing = [n for n in want if n not in tensors]
        if missing:
            raise ValueError(f"the drafter lacks {missing}")
        bad = [(n, tuple(tensors[n].shape), s) for n, s in want.items() if tuple(tensors[n].shape) != s]
        if bad:
            raise ValueError(f"shape mismatch {bad}")
        self.extra = sorted(set(tensors) - set(want) - {"embed_tokens.weight"})
        self.embed = e                                      # bf16 as stored; rows upcast on use
        self.w = {n: tensors[n].float() for n in want if n not in ("d2t", "t2d")}
        self.d2t, self.t2d = tensors["d2t"].long(), tensors["t2d"].bool()
        self.d2t_target = check_vocab_maps(self.d2t, self.t2d)
        self.arm = "bf16"

    # --- arms ---------------------------------------------------------------------------
    def quantize(self, arm: str) -> int:
        """Apply an arm to the float32 weights (once, from bf16). Returns the linears changed."""
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r}, one of {ARMS}")
        if self.arm != "bf16":
            raise RuntimeError("quantize once, from the bf16 weights")
        n = 0
        if arm in ("int8", "int8h"):
            for name in BODY_LINEARS:
                self.w[name] = rtn_int8_rows(self.w[name])
                n += 1
        if arm == "int4h":
            for name in BODY_LINEARS:
                self.w[name] = rtn_int4_g64(self.w[name])
                n += 1
        if arm in ("int8h", "int4h"):
            self.w["lm_head.weight"] = rtn_int8_rows(self.w["lm_head.weight"])
            n += 1
        self.arm = arm
        return n

    # --- pieces -------------------------------------------------------------------------
    def fuse(self, aux: torch.Tensor) -> torch.Tensor:
        """S2: aux [N, n_aux * H_t] -> fused [N, H]."""
        c, x = self.cfg, aux.float()
        if c.norm_before_fc:
            x = rms_norm(x, self.w["input_norm.weight"], c.eps)
        if c.fc_norm:
            ch = x.chunk(c.n_aux, -1)
            x = torch.cat([rms_norm(t, self.w[f"fc_norm.{i}.weight"], c.eps) for i, t in enumerate(ch)], -1)
        return x @ self.w["fc.weight"].t()

    def embed_rows(self, ids) -> torch.Tensor:
        return self.embed[torch.as_tensor(ids, dtype=torch.long)].float()

    def qkv(self, emb: torch.Tensor, hid: torch.Tensor, pos: torch.Tensor):
        """S3's front half: (q [N, heads, hd], k, v [N, kv, hd], residual [N, H])."""
        c, L = self.cfg, "layers.0."
        e = rms_norm(emb, self.w[L + "input_layernorm.weight"], c.eps)
        h = rms_norm(hid, self.w[L + "hidden_norm.weight"], c.eps)
        residual = h if c.norm_before_residual else hid.float()
        x = torch.cat([e, h], -1)
        N = x.shape[0]
        q = (x @ self.w[L + "self_attn.q_proj.weight"].t()).view(N, c.heads, c.head_dim)
        k = (x @ self.w[L + "self_attn.k_proj.weight"].t()).view(N, c.kv_heads, c.head_dim)
        v = (x @ self.w[L + "self_attn.v_proj.weight"].t()).view(N, c.kv_heads, c.head_dim)
        cos, sin = rope_cos_sin(pos + self.pos_offset, c.head_dim, c.rope_theta)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, residual

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """q [N, heads, hd], k / v [P, kv, hd], vis [N, P] bool -> [N, heads * hd]; fp32 softmax."""
        c = self.cfg
        kk = k.repeat_interleave(c.gqa, dim=1)
        vv = v.repeat_interleave(c.gqa, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, kk) * (c.head_dim ** -0.5)
        s = s.masked_fill(~vis[None], float("-inf"))
        p = torch.softmax(s, -1)
        return torch.einsum("hqk,khd->qhd", p, vv).reshape(q.shape[0], c.heads * c.head_dim)

    def finish(self, o: torch.Tensor, residual: torch.Tensor):
        """S3's back half: (post-norm out [N, H], pre-norm h [N, H])."""
        c, L = self.cfg, "layers.0."
        h = residual + o @ self.w[L + "self_attn.o_proj.weight"].t()
        x = rms_norm(h, self.w[L + "post_attention_layernorm.weight"], c.eps)
        g = x @ self.w[L + "mlp.gate_proj.weight"].t()
        u = x @ self.w[L + "mlp.up_proj.weight"].t()
        h = h + (torch.nn.functional.silu(g) * u) @ self.w[L + "mlp.down_proj.weight"].t()
        return rms_norm(h, self.w["norm.weight"], c.eps), h

    def fed_back(self, out: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return out if self.cfg.norm_output else pre                      # S4

    def logits(self, out: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        return torch.cat([out[i:i + chunk] @ self.w["lm_head.weight"].t() for i in range(0, out.shape[0], chunk)])

    def to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]                           # S8

    def greedy_target(self, out: torch.Tensor) -> torch.Tensor:
        return self.to_target(greedy_ids(self.logits(out)))

    def window_ok(self, dist: torch.Tensor) -> torch.Tensor:
        w = self.cfg.window
        return torch.ones_like(dist, dtype=torch.bool) if w is None else dist <= w - 1

    # --- step 0 over the context ----------------------------------------------------------
    def context(self, ids, aux: torch.Tensor, aux_from: int = 0, row_from: int | None = None,
                chunk: int = 256, head_from: int = 0) -> Context:
        """Step 0 over rows s = row_from .. N-1 (default aux_from + 1, the first row whose aux
        exists): token ids[s], aux row s - 1 - aux_from (aux covers positions aux_from..N-1),
        position s; causal over the rows, sliding window (S7). The head runs on rows >= head_from
        only (d0 = -1 below: rows that are keys, never anchors)."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        N = ids.numel()
        r0 = aux_from + 1 if row_from is None else row_from
        if r0 < max(1, aux_from + 1):
            raise ValueError(f"row_from {r0}: rows need the aux of s - 1 >= aux_from {aux_from}")
        if aux.shape[0] != N - aux_from:
            raise ValueError(f"aux has {aux.shape[0]} rows for positions {aux_from}..{N - 1}")
        pos = torch.arange(r0, N)
        fused = self.fuse(aux[r0 - 1 - aux_from: N - 1 - aux_from])
        q, k, v, res = self.qkv(self.embed_rows(ids[r0:]), fused, pos)
        hid = torch.empty(N - r0, self.cfg.hidden)
        d0 = torch.full((N - r0,), -1, dtype=torch.long)
        W = self.cfg.window
        for a in range(0, N - r0, chunk):
            b = min(N - r0, a + chunk)
            k0 = 0 if W is None else max(0, a - (W - 1))
            dist = pos[a:b, None] - pos[None, k0:b]
            vis = (dist >= 0) & self.window_ok(dist)
            o = self.attend(q[a:b], k[k0:b], v[k0:b], vis)
            out, pre = self.finish(o, res[a:b])
            hid[a:b] = self.fed_back(out, pre)
            h0 = max(a, min(b, head_from - r0))
            if h0 < b:
                d0[h0:b] = self.greedy_target(out[h0 - a:])
        return Context(pos=pos, k=k, v=v, hid=hid, d0=d0)

    # --- the draft steps -------------------------------------------------------------------
    def propose(self, ctx: Context, p: int, K: int, window: str = "query") -> torch.Tensor:
        """K greedy drafts (target ids) for anchor p, one row at a time - the reference."""
        if window not in ("query", "anchor"):
            raise ValueError(window)
        i = ctx.row(p)
        if int(ctx.d0[i]) < 0:
            raise IndexError(f"anchor {p} is below the context's head_from (no d_0)")
        drafts = [int(ctx.d0[i])]
        hid = ctx.hid[i:i + 1]
        ck, cv = [], []
        for j in range(1, K):
            qp = torch.tensor([p + j])
            q, k, v, res = self.qkv(self.embed_rows([drafts[-1]]), hid, qp)
            ck.append(k)
            cv.append(v)
            ref_pos = p + j if window == "query" else p
            sel = (ctx.pos <= p) & self.window_ok(ref_pos - ctx.pos)
            keys = torch.cat([ctx.k[sel]] + ck)
            vals = torch.cat([ctx.v[sel]] + cv)
            o = self.attend(q, keys, vals, torch.ones(1, keys.shape[0], dtype=torch.bool))
            out, pre = self.finish(o, res)
            hid = self.fed_back(out, pre)
            drafts.append(int(self.greedy_target(out)[0]))
        return torch.tensor(drafts[:K], dtype=torch.long)

    def propose_batch(self, ctx: Context, anchors, K: int, window: str = "query") -> torch.Tensor:
        """propose() for B anchors at once -> [B, K]. Step j's rows (one per anchor) see the
        step-0 keys of the union window, masked per anchor, plus the chain keys of their own
        anchor; test_eagle3_ref.py holds it to propose()."""
        if window not in ("query", "anchor"):
            raise ValueError(window)
        ps = torch.as_tensor(anchors, dtype=torch.long)
        B = ps.numel()
        rows = ps - int(ctx.pos[0])
        if int(rows.min()) < 0 or int(ps.max()) > int(ctx.pos[-1]):
            raise IndexError("an anchor is not a context row")
        out_ids = torch.empty(B, K, dtype=torch.long)
        out_ids[:, 0] = ctx.d0[rows]
        if bool((out_ids[:, 0] < 0).any()):
            raise IndexError("an anchor below the context's head_from (no d_0)")
        if K == 1:
            return out_ids
        W = self.cfg.window
        lo = 0 if W is None else max(0, int(rows.min()) - (W - 1))
        hi = int(rows.max()) + 1
        cpos = ctx.pos[lo:hi]
        ck_ctx, cv_ctx = ctx.k[lo:hi], ctx.v[lo:hi]
        hid = ctx.hid[rows]
        chain_k, chain_v, owner = [], [], []
        for j in range(1, K):
            qp = ps + j
            q, k, v, res = self.qkv(self.embed_rows(out_ids[:, j - 1]), hid, qp)
            chain_k.append(k)
            chain_v.append(v)
            owner.append(torch.arange(B))
            ref_pos = qp if window == "query" else ps
            vis_ctx = (cpos[None, :] <= ps[:, None]) & self.window_ok(ref_pos[:, None] - cpos[None, :])
            own = torch.cat(owner)
            vis_chain = own[None, :] == torch.arange(B)[:, None]
            vis = torch.cat([vis_ctx, vis_chain], 1)
            o = self.attend(q, torch.cat([ck_ctx] + chain_k), torch.cat([cv_ctx] + chain_v), vis)
            out, pre = self.finish(o, res)
            hid = self.fed_back(out, pre)
            out_ids[:, j] = self.greedy_target(out)
        return out_ids


def load_drafter(snapshot: str, embed: torch.Tensor | None = None, pos_offset: int = 0) -> Drafter:
    return Drafter(read_config(snapshot), read_tensors(snapshot), embed=embed, pos_offset=pos_offset)


# --------------------------------------------------------------------------------------------
# bytes (derived; eagle3_cost.py uses these)


def linear_bytes(N: int, K: int, fmt: str) -> int:
    """One [N][K] linear as the engine would read it: bf16 2 B; int8 1 B + an fp32 scale per
    row (spec 9's gemv_i8w); int4 g64 0.5 B + an f16 scale per row and group of 64 (GPTQ)."""
    if fmt == "bf16":
        return 2 * N * K
    if fmt == "int8":
        return N * K + 4 * N
    if fmt == "int4":
        return N * K // 2 + 2 * N * (K // 64)
    raise ValueError(fmt)


def drafter_bytes(c: Eagle3Config, body: str, head: str, kv_positions: int = 0) -> dict:
    """Bytes one draft step reads (derived): the body linears, the 32k head, the norms (fp32),
    one embedding row (bf16), and the drafter's own K/V over `kv_positions` (bf16, <= window);
    fc once per round (step 0)."""
    H, L = c.hidden, "layers.0."
    sh = expected_shapes(c)
    body_names = [n for n in BODY_LINEARS if n != "fc.weight"]
    b = sum(linear_bytes(*sh[n], body) for n in body_names)
    norms = 4 * (sum(math.prod(sh[n]) for n in sh if n.endswith("norm.weight")))
    kv = 2 * c.kv_heads * c.head_dim * 2 * min(kv_positions, c.window or kv_positions)
    step = b + linear_bytes(*sh["lm_head.weight"], head) + norms + 2 * c.target_hidden + kv
    return {"step": step, "fc": linear_bytes(*sh["fc.weight"], body), "body": b,
            "head": linear_bytes(*sh["lm_head.weight"], head), "kv": kv, "norms": norms}


ARM_FORMATS = {"bf16": ("bf16", "bf16"), "int8": ("int8", "bf16"), "int8h": ("int8", "int8"),
               "int4h": ("int4", "int8")}


# --------------------------------------------------------------------------------------------
# facts


def tensor_digest(path: str, name: str) -> tuple:
    """(sha256 hex, dtype, shape) of one tensor's raw bytes in a safetensors file, read from its
    header's data_offsets - no torch, so a shard can be checked the moment it is complete."""
    import hashlib
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        e = hdr[name]
        a, b = e["data_offsets"]
        f.seek(8 + n + a)
        h = hashlib.sha256()
        left = b - a
        while left:
            chunk = f.read(min(left, 1 << 24))
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest(), e["dtype"], tuple(e["shape"])


def k2_embed_file(k2: str) -> str:
    with open(os.path.join(k2, "model.safetensors.index.json"), encoding="utf-8") as f:
        return os.path.join(k2, json.load(f)["weight_map"]["model.embed_tokens.weight"])


def vocab_coverage(t2d, k2: str) -> list:
    """Lines: which of K2's added tokens (tokenizer.json) and EOS ids the 32k draft vocabulary
    holds - a token outside it can never be drafted."""
    lines = []
    tp = os.path.join(k2, "tokenizer.json")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as f:
            added = json.load(f).get("added_tokens", [])
        inside = [t for t in added if bool(t2d[int(t["id"])])]
        out = [t for t in added if not bool(t2d[int(t["id"])])]
        lines.append(f"K2's added tokens in the draft vocabulary: {len(inside)} of {len(added)}; outside, e.g. "
                     + ", ".join(f"{t['id']} {t['content']!r}" for t in out[:40]))
    gp = os.path.join(k2, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            e = json.load(f).get("eos_token_id")
        eos = e if isinstance(e, list) else [e]
        lines.append("K2's EOS ids " + ", ".join(f"{i} {'in' if bool(t2d[int(i)]) else 'NOT in'}" for i in eos if i is not None)
                     + " the draft vocabulary")
    return lines


def cmd_facts(a) -> None:
    c = read_config(a.snapshot)
    print(f"{a.snapshot}: Eagle3DraftModel, {c.hidden} hidden, {c.heads}/{c.kv_heads} heads x {c.head_dim}, "
          f"inter {c.inter}, window {c.window}, rope theta {c.rope_theta:g}; aux ids {c.aux_ids} "
          f"(inputs of those K2 layers); norm_before_fc {c.norm_before_fc}, fc_norm {c.fc_norm}, "
          f"norm_before_residual {c.norm_before_residual}, norm_output {c.norm_output}; draft vocab "
          f"{c.draft_vocab} of {c.vocab}; proposal default {c.speculative_tokens} tokens; verifier {c.verifier}")
    for arm, (body, head) in ARM_FORMATS.items():
        b = drafter_bytes(c, body, head, kv_positions=c.window or 2048)
        print(f"  derived bytes per draft step, {arm} (body {body}, head {head}): {b['step'] / 1e6:.1f} MB "
              f"(body {b['body'] / 1e6:.1f}, head {b['head'] / 1e6:.1f}, KV at a full window {b['kv'] / 1e6:.1f}); "
              f"fc once per round {b['fc'] / 1e6:.1f} MB")
    if not os.path.isfile(os.path.join(a.snapshot, "model.safetensors")) and \
            not os.path.isfile(os.path.join(a.snapshot, "model.safetensors.index.json")):
        print("model.safetensors: not here yet (config only)")
        return
    t = read_tensors(a.snapshot)
    want = expected_shapes(c)
    print(f"tensors: {len(t)}; " + ", ".join(f"{n} {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
                                             for n, v in sorted(t.items())))
    missing = [n for n in want if n not in t]
    extra = sorted(set(t) - set(want) - {"embed_tokens.weight"})
    print(f"missing {missing}; extra {extra}; embed_tokens {'shipped' if 'embed_tokens.weight' in t else 'NOT shipped'}")
    tgt = check_vocab_maps(t["d2t"], t["t2d"])
    direct_ok = bool((t["d2t"].long() >= 0).all() and t["d2t"].long().max() < c.vocab)
    print(f"d2t: target = draft + d2t[draft] holds (strictly increasing, = t2d's {int(t['t2d'].sum())} ids; "
          f"draft 0 -> {int(tgt[0])}, last -> {int(tgt[-1])}); d2t read as target ids directly would "
          f"{'also be in range' if direct_ok else 'leave the vocabulary'} - the offset reading is the one "
          f"vocab_mapping.py:88-91 writes")
    dfile = os.path.join(a.snapshot, "model.safetensors")
    dd = None
    if os.path.isfile(dfile) and "embed_tokens.weight" in t:
        from safetensors import safe_open
        with safe_open(dfile, framework="pt", device="cpu") as h:
            dname = next(k for k in h.keys() if canon(k) == "embed_tokens.weight")
        dd = tensor_digest(dfile, dname)
        print(f"drafter embed_tokens: sha256 {dd[0]} ({dd[1]} {dd[2]})")
    if a.k2:
        for line in vocab_coverage(t["t2d"], a.k2):
            print(line)
        kfile = k2_embed_file(a.k2)
        de = t.get("embed_tokens.weight")
        if de is None:
            print("embedding: the drafter ships none - K2's is used")
        elif not os.path.isfile(kfile):
            print(f"embedding: {kfile} is not here yet (still downloading?) - not compared")
        else:
            kd = tensor_digest(kfile, "model.embed_tokens.weight")
            print(f"K2 model.embed_tokens.weight: sha256 {kd[0]} ({kd[1]} {kd[2]}, {os.path.basename(kfile)})")
            if dd is not None and kd[0] == dd[0] and kd[1:] == dd[1:]:
                print(f"embedding: BYTE-IDENTICAL to {a.k2}'s model.embed_tokens.weight")
            else:
                from safetensors import safe_open
                with safe_open(kfile, framework="pt", device="cpu") as h:
                    ke = h.get_tensor("model.embed_tokens.weight")
                rows = (de.float() == ke.float()).all(-1) if de.shape == ke.shape else None
                print(f"embedding: DIFFERS from K2's ({de.dtype} {tuple(de.shape)} vs {ke.dtype} {tuple(ke.shape)}"
                      + (f"; {int(rows.sum())} of {rows.numel()} rows equal, max |diff| "
                         f"{float((de.float() - ke.float()).abs().max()):.3g}" if rows is not None else "")
                      + ") - K2_EMBED=1 measures drafting with K2's table")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("snapshot")
    f.add_argument("--k2", help="a K2 snapshot: compare the embeddings byte for byte (sha256), the draft "
                                "vocabulary's coverage of K2's added tokens and EOS ids")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cmd_facts(a)


if __name__ == "__main__":
    sys.exit(main())
