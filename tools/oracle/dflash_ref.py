#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# A CPU reference of the DFlash and DFlash 2 drafters (spec 19 §2, plan 19a Task 1).
#
# The semantics are ported from vLLM (Apache-2.0, SPDX-FileCopyrightText: Copyright
# contributors to the vLLM project), the operator's checkout at
# /Users/urakozz/PycharmProjects/vllm (96b0efe6b0):
#   vllm/model_executor/models/qwen3_dflash.py      context K/V, decoder layer, attention modes,
#                                                   mask embedding, d2t, head
#   vllm/model_executor/models/qwen3_dflash2.py     grouped two-tap conv, candidate selector,
#                                                   input_embedding_scale, output_multiplier/softcap
#   vllm/v1/worker/gpu/spec_decode/dflash/speculator.py      block layout, sample positions
#   vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py     the selector walk
#   vllm/v1/worker/gpu/sample/gumbel.py             murmur3 uniform + Gumbel-max keying
#   vllm/model_executor/models/interfaces.py        _maybe_add_hidden_state (the tap offset)
# This file is a re-implementation in plain torch, not a copy; line references below are to
# those files at that commit.
"""DFlash / DFlash 2 reference drafter: CPU, float32 compute, deterministic.

    cfg = read_config(drafter_dir)
    d = load_drafter(drafter_dir, target_dir)          # or Drafter(cfg, tensors, embed, lm_head)
    ctx = d.context_kv(taps, positions)               # taps: {target_layer_id: [N, H_t]}
    out = d.draft_block(anchor_id, p, K, ctx, temperature=0.0, seed=None)

**Taps (the tap offset).** `taps[i]` is the target's residual stream AFTER decoder layer i
(0-based), i.e. the INPUT of layer i + 1: vLLM turns `target_layer_ids` i into aux index i + 1
(gpu_model_runner.py:5469) and captures aux index k as `hidden + residual` after layer k - 1
(interfaces.py:1600, qwen3_next.py:712-723). In our dumps that is `resid.L{i}` (dump.py), in
transformers' `output_hidden_states` it is `hidden_states[i + 1]` (index 0 is the embedding).
`context_kv` only takes a dict keyed by exactly `cfg.target_layer_ids`; build it with
`taps_from_resid` or `taps_from_hf_hidden_states`, which apply the convention.

**Block layout** (dflash/speculator.py:624-670). Committed positions < p hold context K/V.
The block is 1 + K query rows at positions p .. p + K: row 0 is the anchor (the newest
committed token, which has no target hidden state yet), rows 1..K are `mask_token_id`.
Mask row j predicts the token AT position p + j (not p + j + 1). Row 0 is never sampled.

**Attention** (qwen3_dflash.py:85-179, flash_attn.py:1131-1136). A query at position q sees a
key at position k iff the layer is full attention or |q - k| <= window - 1 (the symmetric
window vLLM uses for non-causal layers; for the context k < q, so q - k <= window - 1), and,
for a causal layer, k <= q. The context is the target-derived K/V of positions < p.

**DFlash 2** adds the per-layer grouped convolutions around attention and the MLP (one row of
look-back inside the block, `row % (1 + K) >= tap`) and the candidate selector: top-k ids per
mask row from the head, a rank-r bilinear edge score, and a sequential walk whose row-0
predecessor is the anchor token. DFlash (v1) samples each mask row independently from its own
full-vocabulary logits. `cfg.version` picks the path.

Run in the oracle container (tools/oracle/README.md, "DFlash reference").
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import torch

SLIDING = "sliding_attention"
DRAFT_NOISE_SALT = 1 << 30          # gumbel.py:18, keeps draft noise disjoint from the target's
_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------------------------
# config


@dataclass
class DFlashConfig:
    version: int                    # 1 = DFlash, 2 = DFlash 2
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    eps: float
    rope_theta: float
    neox: bool
    vocab: int                      # the target's vocabulary (embed / codebook rows)
    draft_vocab: int                # rows of the head the drafter samples from (d2t when != vocab)
    block_size: int                 # trained block: 1 anchor + (block_size - 1) mask rows
    mask_token_id: int
    target_layer_ids: list[int]
    windows: list[int | None]       # per layer: None = full attention
    causal: list[bool]              # per layer
    attention_bias: bool = False
    v_scale: float | None = None
    attention_sink_bias: bool = False
    logit_scale: float = 1.0        # v1 head scale (LogitsProcessor(scale=logit_scale))
    # DFlash 2
    conv_group_size: int = 0
    conv_kernel_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def max_k(self) -> int:
        return self.block_size - 1


def _layer_causal(c: dict, i: int) -> bool:
    """qwen3_dflash.py:85 `_dflash_layer_causal`: explicit is_causal, then dflash_config.causal,
    then 'sliding layers are causal' when layer_types exist, else non-causal."""
    if c.get("is_causal") is not None:
        return bool(c["is_causal"])
    dc = c.get("dflash_config") or {}
    if dc.get("causal") is not None:
        return bool(dc["causal"])
    lt = c.get("layer_types")
    if not lt:
        return False
    return lt[i] == SLIDING


def resolve_layer_attention(c: dict, i: int) -> tuple[int | None, bool]:
    """qwen3_dflash.py:119 `_resolve_layer_attention` -> (sliding_window or None, causal)."""
    dc = c.get("dflash_config") or {}
    lt = c.get("layer_types")
    use_swa = dc.get("use_swa", False)
    any_sliding = lt is not None and any(t == SLIDING for t in lt)
    if lt is None or (use_swa and not any_sliding):
        sliding = bool(use_swa)
    else:
        sliding = lt[i] == SLIDING
    window = None
    if sliding:
        window = dc.get("swa_window_size", c.get("sliding_window"))
        if window is None:
            raise ValueError("sliding attention needs dflash_config.swa_window_size or sliding_window")
        window = int(window)
    return window, _layer_causal(c, i)


def config_from_dict(c: dict) -> DFlashConfig:
    dc = dict(c.get("eagle_config") or {})
    dc.update(c.get("dflash_config") or {})
    arch = " ".join(c.get("architectures") or [])
    v2 = "DFlash2" in arch or "selector_rank" in dc or "conv_kernel_size" in dc
    if c.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {c.get('hidden_act')}")
    rp = c.get("rope_parameters") or {}
    if rp.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope_type {rp.get('rope_type')}")
    if rp.get("partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("partial rotary")
    theta = float(rp.get("rope_theta", c.get("rope_theta", 1_000_000)))  # set_default_rope_theta
    if dc.get("sample_from_anchor", False):
        raise ValueError("sample_from_anchor=True is not DFlash (dflash/speculator.py:79)")
    if dc.get("use_aux_hidden_state", c.get("use_aux_hidden_state", True)) is False:
        raise NotImplementedError("use_aux_hidden_state=False (no fc)")
    ids = [int(i) for i in dc["target_layer_ids"]]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"target_layer_ids must be strictly increasing (the aux capture order): {ids}")
    n = int(c["num_hidden_layers"])
    att = [resolve_layer_attention(c, i) for i in range(n)]
    hidden = int(c["hidden_size"])
    nh = int(c["num_attention_heads"])
    cfg = DFlashConfig(
        version=2 if v2 else 1,
        hidden=hidden,
        n_layers=n,
        n_heads=nh,
        n_kv_heads=int(c["num_key_value_heads"]),
        head_dim=int(c.get("head_dim") or hidden // nh),
        ffn=int(c["intermediate_size"]),
        eps=float(c.get("rms_norm_eps", 1e-6)),
        rope_theta=theta,
        neox=bool(c.get("is_neox_style", True)),
        vocab=int(c["vocab_size"]),
        draft_vocab=int(c.get("draft_vocab_size") or c["vocab_size"]),
        block_size=int(dc["block_size"]),
        mask_token_id=int(dc.get("mask_token_id", c.get("mask_token_id"))),
        target_layer_ids=ids,
        windows=[w for w, _ in att],
        causal=[k for _, k in att],
        attention_bias=bool(c.get("attention_bias", False)),
        v_scale=dc.get("attention_value_scale"),
        attention_sink_bias=bool(dc.get("attention_sink_bias", c.get("add_swa_attention_sink_bias", False))),
        logit_scale=float(c.get("logit_scale", 1.0)),
        raw=c,
    )
    if v2:
        cfg.conv_group_size = int(dc["conv_group_size"])
        cfg.conv_kernel_size = int(dc["conv_kernel_size"])
        cfg.selector_rank = int(dc["selector_rank"])
        cfg.selector_top_k = int(dc["selector_top_k"])
        cfg.input_embedding_scale = float(dc.get("input_embedding_scale", 1.0))
        cfg.output_multiplier = float(dc.get("output_multiplier", 1.0))
        cfg.final_logit_softcapping = float(dc.get("final_logit_softcapping") or 0.0)
        if hidden % cfg.conv_group_size:
            raise ValueError("conv_group_size must divide hidden_size")
        if cfg.draft_vocab != cfg.vocab:
            # vLLM's DFlash 2 candidate path never applies d2t, while the codebooks are indexed
            # by target ids (qwen3_dflash2.py:408): the combination is undefined there.
            raise NotImplementedError("DFlash 2 with a draft vocabulary (d2t) is undefined upstream")
    return cfg


def read_config(drafter_dir: str) -> DFlashConfig:
    with open(os.path.join(drafter_dir, "config.json")) as f:
        return config_from_dict(json.load(f))


# --------------------------------------------------------------------------------------------
# weights


def hf_hub_dir() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def find_snapshot(repo_id: str, need: tuple[str, ...] = ("config.json",)) -> str | None:
    """The snapshot directory of `repo_id` in the HF cache whose files `need` all resolve
    (a dangling symlink = still downloading), else None. Never downloads."""
    base = os.path.join(hf_hub_dir(), "models--" + repo_id.replace("/", "--"))
    snaps = sorted(glob.glob(os.path.join(base, "snapshots", "*")))
    ref = os.path.join(base, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            main = os.path.join(base, "snapshots", f.read().strip())
        snaps = [main] + [s for s in snaps if s != main]
    for s in snaps:
        if all(os.path.isfile(os.path.join(s, n)) for n in need):
            return s
    return None


def _weight_map(snapshot: str) -> dict[str, str]:
    """tensor name -> shard file, from the index or the single model.safetensors."""
    idx = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            return {k: os.path.join(snapshot, v) for k, v in json.load(f)["weight_map"].items()}
    from safetensors import safe_open
    one = os.path.join(snapshot, "model.safetensors")
    with safe_open(one, "pt") as f:
        return {k: one for k in f.keys()}


def _read(wm: dict[str, str], names) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(wm[n], []).append(n)
    out = {}
    for path, ns in by_file.items():
        with safe_open(path, "pt") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def dequant_pack_quantized(packed: torch.Tensor, scale: torch.Tensor, shape, group: int,
                           bits: int = 4) -> torch.Tensor:
    """compressed-tensors `pack-quantized`, symmetric int, group strategy (the syvai W4A16
    drafter): `packed` int32 [out, in / (32/bits)], each int32 holds 32/bits consecutive input
    columns, the first in the lowest bits, stored as q + 2^(bits-1); w = (u - 2^(bits-1)) * s.
    Returns float32 [out, in]."""
    out_f, in_f = int(shape[0]), int(shape[1])
    per = 32 // bits
    p = packed.to(torch.int64) & _M32
    shifts = torch.arange(per, dtype=torch.int64) * bits
    u = (p.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)            # [out, in/per, per]
    q = u.reshape(out_f, -1)[:, :in_f] - (1 << (bits - 1))
    s = scale.float().repeat_interleave(group, dim=1)[:, :in_f]
    return q.float() * s


def _canon(name: str) -> str:
    """qwen3_dflash.py:397-405's renames, plus a leading `model.`."""
    if name.startswith("model."):
        name = name[len("model."):]
    for a, b in (("midlayer.", "layers.0."), ("encoder.output_norm_enc.", "hidden_norm."),
                 ("encoder.fc.", "fc.")):
        name = name.replace(a, b)
    return name


def read_drafter_tensors(drafter_dir: str, cfg: DFlashConfig) -> dict[str, torch.Tensor]:
    """The drafter's own tensors under canonical names; W4A16 linears dequantised to float32."""
    wm = _weight_map(drafter_dir)
    raw = _read(wm, list(wm))
    qc = cfg.raw.get("quantization_config")
    group = bits = None
    if qc is not None:
        if qc.get("quant_method") != "compressed-tensors" or qc.get("format") != "pack-quantized":
            raise NotImplementedError(f"quantization {qc.get('quant_method')}/{qc.get('format')}")
        (g,) = qc["config_groups"].values()
        w = g["weights"]
        if not (w["symmetric"] and w["type"] == "int" and w["strategy"] == "group"):
            raise NotImplementedError(f"quant scheme {w}")
        group, bits = int(w["group_size"]), int(w["num_bits"])
    packed = {n[: -len(".weight_packed")] for n in raw if n.endswith(".weight_packed")}
    if packed and group is None:
        raise ValueError("packed weights without a quantization_config")
    aux = {b + s for b in packed for s in (".weight_scale", ".weight_shape", ".weight_zero_point")}
    t = {}
    for n, v in raw.items():
        if n.endswith(".weight_packed"):
            base = n[: -len(".weight_packed")]
            t[_canon(base + ".weight")] = dequant_pack_quantized(
                v, raw[base + ".weight_scale"], raw[base + ".weight_shape"].tolist(), group, bits)
        elif n in aux:
            continue
        elif "t2d" in n:
            continue                                    # qwen3_dflash.py:877
        elif "mask_hidden" in n:
            raise ValueError("DFlash should not ship mask_hidden (qwen3_dflash.py:872)")
        else:
            t[_canon(n)] = v
    return t


def read_target_embed_head(target_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's input embedding and lm_head (bf16 as stored). Names are found by suffix so
    `model.embed_tokens` and `model.language_model.embed_tokens` both work; mtp.* and the
    vision tower are skipped. A tied target returns the embedding twice."""
    wm = _weight_map(target_dir)
    emb = [n for n in wm if n.endswith("embed_tokens.weight")
           and not n.startswith(("mtp.", "model.visual", "visual"))]
    head = [n for n in wm if n.endswith("lm_head.weight") and not n.startswith("mtp.")]
    if len(emb) != 1 or len(head) > 1:
        raise ValueError(f"target embed/head names ambiguous: {emb} {head}")
    t = _read(wm, emb + head)
    e = t[emb[0]]
    return e, (t[head[0]] if head else e)


def read_mask_embedding(drafter_dir: str, mask_token_id: int) -> torch.Tensor | None:
    """qwen3_dflash.py:910 `_read_mask_embedding`: optional `mask_embedding.pt`, a tensor or
    {'embedding', 'mask_token_id'}; replaces the embedding row of mask_token_id."""
    path = os.path.join(drafter_dir, "mask_embedding.pt")
    if not os.path.isfile(path):
        return None
    state = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(state, dict):
        if state.get("mask_token_id", mask_token_id) != mask_token_id:
            raise ValueError("mask_embedding.pt mask_token_id does not match dflash_config")
        state = state["embedding"]
    return state.reshape(-1)


def load_drafter(drafter_dir: str, target_dir: str | None = None, *,
                 embed: torch.Tensor | None = None, lm_head: torch.Tensor | None = None,
                 upcast_head: bool = False) -> "Drafter":
    """Drafter from a checkpoint directory. The embedding and head are the drafter's own if it
    ships them, else the target's (pass `target_dir`, or the tensors directly to share them
    between several drafters)."""
    cfg = read_config(drafter_dir)
    t = read_drafter_tensors(drafter_dir, cfg)
    if "embed_tokens.weight" in t:
        embed = t.pop("embed_tokens.weight")
    if "lm_head.weight" in t:
        lm_head = t.pop("lm_head.weight")
    if embed is None or lm_head is None:
        if target_dir is None:
            raise ValueError("the drafter ships no embedding / head: pass target_dir")
        e, h = read_target_embed_head(target_dir)
        embed = e if embed is None else embed
        lm_head = h if lm_head is None else lm_head
    mask = read_mask_embedding(drafter_dir, cfg.mask_token_id)
    return Drafter(cfg, t, embed, lm_head, mask_embedding=mask, upcast_head=upcast_head)


# --------------------------------------------------------------------------------------------
# taps


def taps_from_resid(tensors: dict, target_layer_ids) -> dict[int, torch.Tensor]:
    """dump.py's `resid.L{i}` (decoder layer i's output) IS the tap for target_layer_id i."""
    return {i: tensors[f"resid.L{i}"] for i in target_layer_ids}


def taps_from_hf_hidden_states(hidden_states, target_layer_ids) -> dict[int, torch.Tensor]:
    """transformers' `output_hidden_states`: index 0 is the embedding, index i + 1 the output
    of layer i, so the tap for target_layer_id i is hidden_states[i + 1] (vLLM's i + 1)."""
    n = len(hidden_states) - 1
    if max(target_layer_ids) >= n - 1:
        # the last entry is post-final-norm in transformers, not a residual
        raise ValueError("a tap at the last layer would read the post-norm hidden state")
    return {i: hidden_states[i + 1] for i in target_layer_ids}


# --------------------------------------------------------------------------------------------
# math


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None,
           chunk: int = 32768) -> torch.Tensor:
    """x @ w.T in float32. A non-float32 weight is upcast in row chunks, so a bf16 lm_head never
    exists whole in float32."""
    x = x.float()
    if w.dtype == torch.float32:
        y = x @ w.t()
    else:
        y = torch.cat([x @ w[i:i + chunk].float().t() for i in range(0, w.shape[0], chunk)], -1)
    return y if b is None else y + b.float()


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float, neox: bool = True) -> torch.Tensor:
    """x [N, heads, hd] rotated at positions pos [N] (default rope, full rotary)."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    ang = pos.to(torch.float64)[:, None] * inv[None]                  # [N, hd/2]
    cos, sin = ang.cos().float()[:, None], ang.sin().float()[:, None]  # [N, 1, hd/2]
    if neox:
        a, b = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2)


def grouped_conv(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, block_len: int,
                 group: int) -> torch.Tensor:
    """qwen3_dflash2.py:144 `_grouped_conv`. x [R, H], delta [R, taps, H/group] (this row's
    predicted coefficient per group), base [taps, H]. Rows are blocks of `block_len` = 1 + K
    stacked; tap t reaches row - t only when row % block_len >= t:
        out[r] = sum_t (base[t] + delta[r, t]) * x[r - t]."""
    R, H = x.shape
    taps = base.shape[0]
    coef = base.float()[None] + delta.float().repeat_interleave(group, dim=-1)   # [R, taps, H]
    out = coef[:, 0] * x
    row_in_block = torch.arange(R) % block_len
    for t in range(1, taps):
        shifted = torch.cat([x.new_zeros(t, H), x[:-t]], 0)
        out = out + coef[:, t] * shifted * (row_in_block >= t).float()[:, None]
    return out


def _murmur_mul(a: torch.Tensor, b: int) -> torch.Tensor:
    lo = a * (b & 0xFFFF)
    hi = ((a * (b >> 16)) & 0xFFFF) << 16
    return (lo + hi) & _M32


def _rotl(v: torch.Tensor, s: int) -> torch.Tensor:
    return ((v << s) | (v >> (32 - s))) & _M32


def _murmur_mix(h: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    key = _murmur_mul(key & _M32, 0xCC9E2D51)
    key = _rotl(key, 15)
    key = _murmur_mul(key, 0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (_murmur_mul(h, 5) + 0xE6546B64) & _M32


def murmur3_uniform32(seed: int, pos: int, keys: torch.Tensor) -> torch.Tensor:
    """gumbel.py:96-120 `murmur3_hash32` / `murmur3_uniform32` (domain 0), keyed by token id:
    the same (seed, position, token) always draws the same uniform. float64 result holding the
    fp32 value vLLM computes."""
    keys = keys.to(torch.int64) & _M32
    h = torch.zeros_like(keys)
    for k in (seed & _M32, (seed >> 32) & _M32, pos & _M32):
        h = _murmur_mix(h, torch.full_like(keys, k))
    h = _murmur_mix(h, keys)
    h = h ^ 16
    h = h ^ (h >> 16)
    h = _murmur_mul(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = _murmur_mul(h, 0xC2B2AE35)
    h = h ^ (h >> 16)
    u = (h >> 16).double() * 2.0 ** -16 + ((h & 0xFFFF).double() + 0.5) * 2.0 ** -32
    return u.float().double()


def gumbel_argmax(logits: torch.Tensor, keys: torch.Tensor, temperature: float,
                  seed: int | None, pos: int) -> int:
    """gumbel.py:165 `gumbel_noised_argmax` with IS_DRAFTING: argmax at temperature 0, else
    argmax(logits / T + g), g = -log(-log(1 - u)), u = murmur3(seed, pos + salt, key).
    Ties go to the lowest index (Triton's tl.max(return_indices=True) tie-break-left)."""
    x = logits.double()
    if temperature != 0.0:
        if seed is None:
            raise ValueError("temperature > 0 needs a seed (the reference is deterministic)")
        u = murmur3_uniform32(seed, pos + DRAFT_NOISE_SALT, keys)
        x = x / temperature - torch.log(-torch.log1p(-u))
    best = x.max()
    return int(torch.nonzero(x == best)[0, 0])


def top_k_stable(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k per row, descending, equal values in increasing id order (deterministic)."""
    v, i = torch.sort(logits, dim=-1, descending=True, stable=True)
    return i[..., :k], v[..., :k]


def score_edges(pred: torch.Tensor, succ: torch.Tensor, cand: torch.Tensor, unary: torch.Tensor,
                hidden: torch.Tensor, anchor_id: int) -> torch.Tensor:
    """qwen3_dflash2.py:290 `_score_edges` for one block. cand/unary [K, k], hidden [K, r]
    (hidden_projection of the mask rows). Returns S [K, k(prev), k(cand)]:
        S[j, a, c] = unary[j, c] + sum_r pred[prev(j, a)][r] * hidden[j, r] * succ[cand[j, c]][r]
    with prev(0, a) = the anchor for every a and prev(j, a) = cand[j - 1, a] for j > 0."""
    K, k = cand.shape
    prev = torch.cat([torch.full((1, k), anchor_id, dtype=torch.long), cand[:-1]], 0)
    P = pred[prev].float() * hidden.float()[:, None, :]                 # [K, k, r]
    S = succ[cand].float()                                              # [K, k, r]
    return unary.float()[:, None, :] + torch.einsum("jar,jcr->jac", P, S)


def selector_walk(S: torch.Tensor, cand: torch.Tensor, p: int, temperature: float = 0.0,
                  seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dflash2/speculator.py:16 `_selector_walk_kernel` for one block. `previous` starts at 0:
    row 0's predecessor rows are all the anchor, so index 0 IS the anchor. Row j picks over
    S[j, previous, :] (argmax, or Gumbel-argmax keyed by candidate id at position
    P - 1 = p + j for mask row j + 1, which predicts position p + j + 1).
    Returns (chosen index [K], ids [K], realized scores [K, k])."""
    K = S.shape[0]
    prev = 0
    idx, real = [], []
    for j in range(K):
        row = S[j, prev]
        c = gumbel_argmax(row, cand[j], temperature, seed, p + j)
        idx.append(c)
        real.append(row)
        prev = c
    idx_t = torch.tensor(idx, dtype=torch.long)
    return idx_t, cand[torch.arange(K), idx_t], torch.stack(real)


# --------------------------------------------------------------------------------------------
# the drafter


@dataclass
class ContextKV:
    """Target-derived K/V of committed positions (spec 19 §2 'State'). k/v: per layer
    [N, n_kv, hd] after k_norm + RoPE (v after v_scale); fc: the hidden_norm(fc(taps)) plane."""
    pos: torch.Tensor
    k: list[torch.Tensor]
    v: list[torch.Tensor]
    fc: torch.Tensor


@dataclass
class DraftOut:
    ids: torch.Tensor           # [K] draft ids (target vocabulary)
    cand_ids: torch.Tensor      # [K, top_k] per mask row, descending unary (ties: lower id)
    unary: torch.Tensor         # [K, top_k] head logits of cand_ids (v2: output_multiplier, softcap)
    scores: torch.Tensor        # [K, top_k] v2: realized selector scores; v1: = unary
    edges: torch.Tensor | None  # [K, top_k, top_k] v2: every (predecessor, candidate) score
    chosen: torch.Tensor        # [K] index into cand_ids (v1: rank of ids in the top_k, or -1)
    q: torch.Tensor             # [K, vocab] draft distribution (see draft_block)
    hidden: torch.Tensor        # [1 + K, H] final-norm hidden rows
    positions: torch.Tensor     # [1 + K] p .. p + K


class Drafter:
    """`tensors` uses the checkpoint names (fc.weight, hidden_norm.weight, norm.weight,
    layers.{i}.self_attn.{q,k,v,o}_proj.weight, ..., layers.{i}.attention_conv.base_kernel,
    candidate_selector.*). Linear weights are upcast to float32 once; the codebooks, embedding
    and head keep their stored dtype and are upcast per use (`upcast_head` caches the head)."""

    V1_TOP_K = 16   # v1 has no selector; DraftOut still reports a top-16 for the D1 gate

    def __init__(self, cfg: DFlashConfig, tensors: dict[str, torch.Tensor], embed: torch.Tensor,
                 lm_head: torch.Tensor, mask_embedding: torch.Tensor | None = None,
                 upcast_head: bool = False):
        self.cfg = cfg
        keep = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
        self.w = {n: (v if n in keep else v.float()) for n, v in tensors.items()}
        self.embed = embed
        self.lm_head = lm_head.float() if upcast_head else lm_head
        self.mask_embedding = None if mask_embedding is None else mask_embedding.float()
        self.d2t = self.w.pop("d2t", None)
        if self.d2t is not None:
            self.d2t = self.d2t.long()
        self._check()

    # -- shapes -----------------------------------------------------------------------------

    def _check(self) -> None:
        c, w = self.cfg, self.w
        H, hd, nh, nkv = c.hidden, c.head_dim, c.n_heads, c.n_kv_heads
        exp = {"hidden_norm.weight": (H,), "norm.weight": (H,)}
        for i in range(c.n_layers):
            L = f"layers.{i}."
            exp.update({
                L + "input_layernorm.weight": (H,), L + "post_attention_layernorm.weight": (H,),
                L + "self_attn.q_proj.weight": (nh * hd, H), L + "self_attn.k_proj.weight": (nkv * hd, H),
                L + "self_attn.v_proj.weight": (nkv * hd, H), L + "self_attn.o_proj.weight": (H, nh * hd),
                L + "self_attn.q_norm.weight": (hd,), L + "self_attn.k_norm.weight": (hd,),
                L + "mlp.gate_proj.weight": (c.ffn, H), L + "mlp.up_proj.weight": (c.ffn, H),
                L + "mlp.down_proj.weight": (H, c.ffn)})
            if c.version == 2:
                for cv in ("attention_conv", "mlp_conv"):
                    exp[L + cv + ".base_kernel"] = (2, c.conv_kernel_size, H)
                    exp[L + cv + ".kernel_projection.weight"] = (2 * c.conv_kernel_size * H // c.conv_group_size, H)
        if c.version == 2:
            exp["candidate_selector.hidden_projection.weight"] = (c.selector_rank, H)
            exp["candidate_selector.predecessor_codebook"] = (c.vocab, c.selector_rank)
            exp["candidate_selector.successor_codebook"] = (c.vocab, c.selector_rank)
        for n, s in exp.items():
            if n not in w:
                raise KeyError(f"missing tensor {n}")
            if tuple(w[n].shape) != s:
                raise ValueError(f"{n}: shape {tuple(w[n].shape)} != {s}")
        fc = w["fc.weight"]
        if fc.shape[0] != H or fc.shape[1] % len(c.target_layer_ids):
            raise ValueError(f"fc.weight {tuple(fc.shape)} vs {len(c.target_layer_ids)} taps")
        if tuple(self.embed.shape) != (c.vocab, H):
            raise ValueError(f"embedding {tuple(self.embed.shape)} != {(c.vocab, H)}")
        if tuple(self.lm_head.shape) != (c.draft_vocab, H):
            raise ValueError(f"lm_head {tuple(self.lm_head.shape)} != {(c.draft_vocab, H)}")
        if (c.draft_vocab != c.vocab) != (self.d2t is not None):
            raise ValueError("d2t must be present exactly when draft_vocab_size != vocab_size")
        if c.attention_sink_bias:
            for i in range(c.n_layers):
                if f"layers.{i}.self_attn.attention_sink_bias" not in w:
                    raise KeyError(f"layers.{i}.self_attn.attention_sink_bias")

    @property
    def target_hidden(self) -> int:
        return self.w["fc.weight"].shape[1] // len(self.cfg.target_layer_ids)

    # -- context ----------------------------------------------------------------------------

    def context_kv(self, taps: dict[int, torch.Tensor], positions) -> ContextKV:
        """qwen3_dflash.py:632 `precompute_and_store_context_kv`: per committed position,
        c = hidden_norm(fc(cat(taps in target_layer_ids order))), then per layer
        k = RoPE(k_norm(k_proj(c)), pos), v = v_proj(c) (* attention_value_scale).
        `taps` maps each target_layer_id i to the residual AFTER target layer i, [N, H_t]."""
        c = self.cfg
        if not isinstance(taps, dict) or sorted(taps) != c.target_layer_ids:
            got = sorted(taps) if isinstance(taps, dict) else type(taps).__name__
            raise ValueError(f"taps must be {{target_layer_id: residual after that layer}} for "
                             f"exactly {c.target_layer_ids}; got {got} (taps_from_resid / "
                             f"taps_from_hf_hidden_states build it)")
        pos = torch.as_tensor(positions, dtype=torch.long).reshape(-1)
        xs = [taps[i].float() for i in c.target_layer_ids]
        for x in xs:
            if x.shape != (pos.shape[0], self.target_hidden):
                raise ValueError(f"tap shape {tuple(x.shape)} != {(pos.shape[0], self.target_hidden)}")
        ctx = rms_norm(linear(torch.cat(xs, -1), self.w["fc.weight"]), self.w["hidden_norm.weight"], c.eps)
        ks, vs = [], []
        for i in range(c.n_layers):
            k, v = self._kv(i, ctx, pos)
            ks.append(k)
            vs.append(v)
        return ContextKV(pos=pos, k=ks, v=vs, fc=ctx)

    def _kv(self, i: int, x: torch.Tensor, pos: torch.Tensor):
        c, L = self.cfg, f"layers.{i}.self_attn."
        N = x.shape[0]
        k = linear(x, self.w[L + "k_proj.weight"], self.w.get(L + "k_proj.bias")).view(N, c.n_kv_heads, c.head_dim)
        v = linear(x, self.w[L + "v_proj.weight"], self.w.get(L + "v_proj.bias")).view(N, c.n_kv_heads, c.head_dim)
        k = rope(rms_norm(k, self.w[L + "k_norm.weight"], c.eps), pos, c.rope_theta, c.neox)
        if c.v_scale is not None:
            v = v * c.v_scale
        return k, v

    # -- block forward ----------------------------------------------------------------------

    def embed_block(self, anchor_id: int, K: int) -> torch.Tensor:
        """[anchor, mask x K] through the target's embedding (qwen3_dflash.py:496; the mask row
        replaced by mask_embedding.pt when shipped), times input_embedding_scale (DFlash 2,
        qwen3_dflash2.py:391)."""
        c = self.cfg
        ids = torch.tensor([anchor_id] + [c.mask_token_id] * K)
        x = self.embed[ids].float()
        if self.mask_embedding is not None:
            x[ids == c.mask_token_id] = self.mask_embedding
        if c.version == 2:
            x = x * c.input_embedding_scale
        return x

    def attention_mask(self, i: int, qpos: torch.Tensor, kpos: torch.Tensor) -> torch.Tensor:
        """[Q, Kv] bool, True = visible (module docstring 'Attention')."""
        c = self.cfg
        d = qpos[:, None] - kpos[None, :]
        m = torch.ones_like(d, dtype=torch.bool)
        if c.windows[i] is not None:
            m &= d.abs() <= c.windows[i] - 1
        if c.causal[i]:
            m &= d >= 0
        return m

    def _attention(self, i: int, x: torch.Tensor, qpos: torch.Tensor, ctx: ContextKV) -> torch.Tensor:
        c, L = self.cfg, f"layers.{i}.self_attn."
        R = x.shape[0]
        q = linear(x, self.w[L + "q_proj.weight"], self.w.get(L + "q_proj.bias")).view(R, c.n_heads, c.head_dim)
        q = rope(rms_norm(q, self.w[L + "q_norm.weight"], c.eps), qpos, c.rope_theta, c.neox)
        kb, vb = self._kv(i, x, qpos)
        sel = ctx.pos < qpos[0]
        k = torch.cat([ctx.k[i][sel], kb], 0)
        v = torch.cat([ctx.v[i][sel], vb], 0)
        kpos = torch.cat([ctx.pos[sel], qpos])
        g = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(g, dim=1)                       # [Kv, nh, hd]; head h -> kv h // g
        v = v.repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) * c.head_dim ** -0.5
        s = s.masked_fill(~self.attention_mask(i, qpos, kpos)[None], float("-inf"))
        if c.attention_sink_bias:
            sink = self.w[L + "attention_sink_bias"].float()[:, None, None].expand(-1, R, 1)
            pr = torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]
        else:
            pr = torch.softmax(s, -1)
        o = torch.einsum("hqk,khd->qhd", pr, v).reshape(R, c.n_heads * c.head_dim)
        return linear(o, self.w[L + "o_proj.weight"], self.w.get(L + "o_proj.bias"))

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        L = f"layers.{i}.mlp."
        g = linear(x, self.w[L + "gate_proj.weight"])
        u = linear(x, self.w[L + "up_proj.weight"])
        return linear(torch.nn.functional.silu(g) * u, self.w[L + "down_proj.weight"])

    def _conv_prepare(self, i: int, which: str, x: torch.Tensor, block_len: int):
        """DFlashGroupedConv.prepare (qwen3_dflash2.py:217): coefficients from x, side 0
        applied to x now, side 1 kept for finish."""
        c, L = self.cfg, f"layers.{i}.{which}."
        G = c.hidden // c.conv_group_size
        co = linear(x, self.w[L + "kernel_projection.weight"]).view(x.shape[0], 2, c.conv_kernel_size, G)
        base = self.w[L + "base_kernel"]
        return grouped_conv(x, co[:, 0], base[0], block_len, c.conv_group_size), co[:, 1]

    def _conv_finish(self, i: int, which: str, x: torch.Tensor, co: torch.Tensor, block_len: int):
        c = self.cfg
        base = self.w[f"layers.{i}.{which}.base_kernel"]
        return grouped_conv(x, co, base[1], block_len, c.conv_group_size)

    def forward_block(self, x: torch.Tensor, p: int, ctx: ContextKV) -> torch.Tensor:
        """Block rows x [1 + K, H] (embeddings) at positions p .. p + K over the context's
        positions < p. Pre-norm Qwen3 layers (qwen3_dflash.py:371; DFlash 2 adds the convs,
        qwen3_dflash2.py:267). Returns the final-norm hidden [1 + K, H]."""
        c = self.cfg
        R = x.shape[0]
        qpos = torch.arange(p, p + R)
        resid = x.float()
        for i in range(c.n_layers):
            L = f"layers.{i}."
            h = rms_norm(resid, self.w[L + "input_layernorm.weight"], c.eps)
            if c.version == 2:
                h, co = self._conv_prepare(i, "attention_conv", h, R)
                h = self._attention(i, h, qpos, ctx)
                h = self._conv_finish(i, "attention_conv", h, co, R)
            else:
                h = self._attention(i, h, qpos, ctx)
            resid = resid + h
            h = rms_norm(resid, self.w[L + "post_attention_layernorm.weight"], c.eps)
            if c.version == 2:
                h, co = self._conv_prepare(i, "mlp_conv", h, R)
                h = self._mlp(i, h)
                h = self._conv_finish(i, "mlp_conv", h, co, R)
            else:
                h = self._mlp(i, h)
            resid = resid + h
        return rms_norm(resid, self.w["norm.weight"], c.eps)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Raw lm_head logits [rows, draft_vocab] in float32."""
        return linear(h, self.lm_head)

    # -- one draft --------------------------------------------------------------------------

    def draft_block(self, anchor_id: int, p: int, K: int, ctx: ContextKV, temperature: float = 0.0,
                    seed: int | None = None) -> DraftOut:
        """K drafts for positions p + 1 .. p + K, given the anchor (the token at p) and the
        context K/V (positions < p are used; later ones in `ctx` are ignored).

        `q` is the per-row draft distribution at `temperature` (T = 0 reports T = 1; greedy
        acceptance never reads q): v2: softmax(realized scores / T) on the row's top_k
        candidates, 0 elsewhere (the -inf draft logits of dflash2/speculator.py:200);
        v1: softmax(logits / T) over the head (0 outside d2t's image)."""
        c = self.cfg
        if not 1 <= K <= c.max_k:
            raise ValueError(f"K={K} outside 1..{c.max_k} (block_size {c.block_size})")
        h = self.forward_block(self.embed_block(anchor_id, K), p, ctx)
        hm = h[1:]
        T = temperature if temperature > 0 else 1.0
        if c.version == 2:
            cand, unary = top_k_stable(self.head_logits(hm), c.selector_top_k)
            unary = unary * c.output_multiplier                   # logits_processor.py:314-317
            if c.final_logit_softcapping > 0:
                cap = c.final_logit_softcapping
                unary = torch.tanh(unary / cap) * cap
            hid = linear(hm, self.w["candidate_selector.hidden_projection.weight"])
            S = score_edges(self.w["candidate_selector.predecessor_codebook"],
                            self.w["candidate_selector.successor_codebook"], cand, unary, hid, anchor_id)
            chosen, ids, real = selector_walk(S, cand, p, temperature, seed)
            q = torch.zeros(K, c.vocab)
            q.scatter_(1, cand, torch.softmax(real.double() / T, -1).float())
            return DraftOut(ids=ids, cand_ids=cand, unary=unary, scores=real, edges=S,
                            chosen=chosen, q=q, hidden=h, positions=torch.arange(p, p + K + 1))
        logits = self.head_logits(hm) * c.logit_scale
        if self.d2t is not None:                                  # qwen3_dflash.py:822-832
            full = torch.full((K, c.vocab), float("-inf"))
            full[:, torch.arange(c.draft_vocab) + self.d2t] = logits
            logits = full
        keys = torch.arange(c.vocab)
        ids = torch.tensor([gumbel_argmax(logits[j], keys, temperature, seed, p + j) for j in range(K)])
        cand, unary = top_k_stable(logits, self.V1_TOP_K)
        hit = cand == ids[:, None]
        chosen = torch.where(hit.any(1), hit.float().argmax(1), torch.full((K,), -1))
        q = torch.softmax(logits.double() / T, -1).float()
        return DraftOut(ids=ids, cand_ids=cand, unary=unary, scores=unary, edges=None, chosen=chosen,
                        q=q, hidden=h, positions=torch.arange(p, p + K + 1))
