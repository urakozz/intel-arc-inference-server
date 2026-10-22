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
#
# The LiLiCorr form (spec 26a) follows the same checkout at 3901d56800 (branch b70-xpu-stack), semantics only:
#   vllm/model_executor/models/lilicorr.py          LiLiCorrHead (the lattice), the backbone class choice (L:)
#   vllm/v1/worker/gpu/spec_decode/lilicorr/speculator.py   the anchor row (prepare_context_anchor)
#   vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py    the candidate walk (spec_step_seed keying)
#   vllm/model_executor/layers/logits_processor.py  get_top_k_tokens(return_log_probs): full-vocabulary log-probs
#   vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py   NVFP4's nibble order and scales
# (the head is "adapted from sgl-project/sglang PR #37462" upstream).
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

**The LiLiCorr form** (spec 26a; `architectures: [LiLiCorrDraftModel]`, nvidia/...-24GB-LiLiCorr): the DFlash 2
backbone above with attention sinks (a per-head logit in the softmax denominator, unscaled, no value) and every layer
sliding at |q - k| <= window - 1 (1023), non-causal, always the full block (anchor + 7 masks: K = 1..7 are prefixes of
one walk); the NVFP4 linears decoded exactly (`nvfp4_decode`: W' = s_b x e bf16-exact, g kept apart and applied to the
linear's output); candidates = the target head's top 8 per mask row with full-vocabulary log-probs
(`lilicorr_candidates`); the LiLiCorr head (`Drafter.lilicorr_lattice`) scores a lattice [7][8 previous][8 next] from
the candidates' embeddings, the mask rows' final-normed hidden, five log-prob features and the ANCHOR - hidden_norm(fc(
taps)) of the newest committed position, p - 1 (`ContextKV.anchor_row`, `LiLiCorrSession`); the walk reads the lattice
alone (`Drafter.lilicorr_walk`), q = softmax(realized / T) on the 8 candidates.

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
    # Spec 23f: the reference the semantics follow. "vllm" (DFlash / DFlash 2 above); "muse" =
    # transformers 5.19.0's MuseGlimmerAssistantModel + DFlashTokenCandidateGenerator (the
    # module docstring's "Muse-Glimmer form"): the window INCLUDES |q - k| == window
    # (MU:142-152), the RMSNorm rounds x-hat to the model dtype before x w (A:50-55), the block
    # is always block_size rows (rows past K computed and discarded), the drafts read the raw
    # head rows (G:1658-1660) and embed the raw rows (G:1563-1579).
    form: str = "vllm"
    window_inclusive: bool = False      # |q - k| <= window (muse) or <= window - 1 (vllm)
    norm_two_roundings: bool = False    # w x dtype(x-hat) (muse) or the fp32 product (vllm)
    embed_norm: bool = False            # the target's weight-less embedding norm on the rows: never (G:1563-1579)
    head_cap: bool = False              # the target's multiplier / softcap on the draft rows: never (G:1658)
    q_scale: float = 1.0                # q x this after q_norm: 1 for the assistant (no qk_scale_factor, A:181-192)
    # Spec 26a: the LiLiCorr form (form "lilicorr", version 2: the DFlash 2 backbone with sinks + the lattice head,
    # module docstring "The LiLiCorr form"). cand_topk = lilicorr_candidate_topk; lil_* the head's widths.
    cand_topk: int = 0
    lil_hidden: int = 0
    lil_layers: int = 0
    lil_heads: int = 0
    lil_mlp: int = 0
    factor_dim: int = 0
    vector_eps: float = 0.0
    lil_logit_scale: float = 1.0
    feature_eps: float = 1e-5           # nn.LayerNorm(5)'s default eps (the head's feature_norm)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def max_k(self) -> int:
        return self.block_size - 1

    @property
    def sinks(self) -> bool:
        return self.attention_sink_bias


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


MUSE_MODEL_TYPE = "muse_glimmer_assistant"
MUSE_ARCH = "MuseGlimmerAssistantModel"


def is_muse_config(c: dict) -> bool:
    return c.get("model_type") == MUSE_MODEL_TYPE or MUSE_ARCH in (c.get("architectures") or [])


def muse_config_from_dict(c: dict) -> DFlashConfig:
    """MuseGlimmerAssistantConfig (AC:62-89): block_size, mask_token_id and target_layer_ids at the top level,
    every layer sliding (layer_types) with `sliding_window`, non-causal (A:146), RoPE default at
    rope_parameters.rope_theta (A:330-346), eps rms_norm_eps. The checkpoint carries no vocab_size: the vocabulary is
    the target's (set by the Drafter from the shared embedding)."""
    if c.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {c.get('hidden_act')}")
    rp = c.get("rope_parameters") or {}
    if rp.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope_type {rp.get('rope_type')}")
    n = int(c["num_hidden_layers"])
    lt = c.get("layer_types") or ["sliding_attention"] * n
    if any(t != SLIDING for t in lt):
        raise NotImplementedError(f"a full-attention assistant layer ({lt}): the Muse form is all sliding")
    ids = [int(i) for i in (c.get("target_layer_ids") or [1, 13, 25, 37, 49])]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"target_layer_ids must be strictly increasing: {ids}")
    hidden, nh = int(c["hidden_size"]), int(c["num_attention_heads"])
    vocab = int(c.get("vocab_size") or 0)
    w = int(c["sliding_window"])
    return DFlashConfig(
        version=1, hidden=hidden, n_layers=n, n_heads=nh, n_kv_heads=int(c["num_key_value_heads"]),
        head_dim=int(c.get("head_dim") or hidden // nh), ffn=int(c["intermediate_size"]),
        eps=float(c.get("rms_norm_eps", 1e-5)), rope_theta=float(rp.get("rope_theta", 500000.0)), neox=True,
        vocab=vocab, draft_vocab=vocab, block_size=int(c.get("block_size", 16)),
        mask_token_id=int(c.get("mask_token_id", 201818)), target_layer_ids=ids, windows=[w] * n,
        causal=[False] * n, form="muse", window_inclusive=True, norm_two_roundings=True, raw=c)


LILICORR_ARCH = "LiLiCorrDraftModel"
NVFP4_QUANT = {"quant_method": "modelopt_fp4", "quant_algo": "W4A16_NVFP4", "group_size": 16}


def is_lilicorr_config(c: dict) -> bool:
    return LILICORR_ARCH in (c.get("architectures") or [])


def lilicorr_config_from_dict(c: dict) -> DFlashConfig:
    """LiLiCorrDraftModel (V/models/lilicorr.py:428-476): the DFlash 2 backbone (conv_kernel_size set -> vLLM's
    DFlash2Qwen3DecoderLayer) read by the vLLM rules (attention resolution, RoPE, taps), block_size at the top level
    when dflash_config has none (L:438), no candidate selector: the LiLiCorr head scores the candidates (lilicorr_*).
    Refused by name: lilicorr_enabled false, has_own_lm_head, a draft vocabulary, candidate_topk not a power of two
    <= 16, K >= block (nothing to refuse here: K is per call), a causal layer, conv taps without groups, any
    quantization_config but ModelOpt's W4A16 NVFP4 at group 16 (spec 26 §5.1)."""
    dc = dict(c.get("dflash_config") or {})
    if not dc.get("lilicorr_enabled", True):
        raise ValueError("LiLiCorr requires lilicorr_enabled=True (V/models/lilicorr.py:436)")
    if c.get("has_own_lm_head") or dc.get("has_own_lm_head"):
        raise NotImplementedError("has_own_lm_head: LiLiCorr here drafts with the target's lm_head (spec 26 §1)")
    if c.get("draft_vocab_size") and int(c["draft_vocab_size"]) != int(c["vocab_size"]):
        raise ValueError("LiLiCorr candidates require the full target vocabulary (V/models/lilicorr.py:472)")
    topk = int(dc["lilicorr_candidate_topk"])
    if topk <= 0 or topk > 16 or topk & (topk - 1):
        raise ValueError(f"LiLiCorr candidate_topk {topk} must be a power of two <= 16 (V/models/lilicorr.py:143)")
    taps, groups = int(dc.get("conv_kernel_size", 0)), int(dc.get("conv_group_size", 0))
    if taps <= 0 or groups <= 0:
        raise NotImplementedError(f"LiLiCorr without the DFlash 2 convolutions (taps {taps}, groups {groups})")
    qc = c.get("quantization_config")
    if qc is not None and any(qc.get(k) != v for k, v in NVFP4_QUANT.items()):
        raise NotImplementedError(f"quantization_config {({k: qc.get(k) for k in NVFP4_QUANT})}: only ModelOpt's "
                                  f"W4A16 NVFP4 at group 16 is read (decoded exactly, spec 26 §3)")
    block = int(dc.get("block_size", c.get("block_size", 0)))
    sel = dict(dc)
    sel.update({"block_size": block, "selector_rank": 0, "selector_top_k": topk})
    base = dict(c)
    base["dflash_config"] = sel
    base["architectures"] = []                  # read by the vLLM DFlash 2 rules below (not this function again)
    base.pop("quantization_config", None)
    cfg = config_from_dict(base)
    if any(cfg.causal):
        raise NotImplementedError("a causal LiLiCorr layer: the trained form is non-causal (causal: false)")
    cfg.form = "lilicorr"
    cfg.selector_rank, cfg.selector_top_k = 0, topk
    cfg.cand_topk = topk
    cfg.lil_hidden = int(dc.get("lilicorr_hidden_size") or cfg.hidden)
    cfg.lil_layers = int(dc["lilicorr_num_layers"])
    cfg.lil_heads = int(dc["lilicorr_num_heads"])
    cfg.lil_mlp = int(cfg.lil_hidden * float(dc["lilicorr_mlp_ratio"]))
    cfg.factor_dim = int(dc["lilicorr_factor_dim"])
    cfg.vector_eps = float(dc["lilicorr_vector_eps"])
    cfg.lil_logit_scale = float(dc["lilicorr_logit_scale"])
    cfg.raw = c
    if cfg.lil_hidden % cfg.lil_heads:
        raise ValueError(f"LiLiCorr hidden {cfg.lil_hidden} must be divisible by its {cfg.lil_heads} heads")
    return cfg


def config_from_dict(c: dict) -> DFlashConfig:
    if is_muse_config(c):
        return muse_config_from_dict(c)
    if is_lilicorr_config(c):
        return lilicorr_config_from_dict(c)
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


# --- NVFP4 (ModelOpt W4A16, spec 26 §3): W = g x s_b x e, decoded as W' = s_b x e (exact in bf16 and f16) and g -------

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)        # the code's low 3 bits; bit 3 is the sign
E4M3_NAN = (0x7F, 0xFF)                                 # e4m3fn: S.1111.111 is NaN, there is no infinity


def e4m3_value(code: int) -> float:
    """One float8 e4m3fn byte, exactly (bias 7; exponent 0 = subnormal m x 2^-9; S.1111.111 NaN - refused)."""
    code &= 0xFF
    if code in E4M3_NAN:
        raise ValueError(f"e4m3 code 0x{code:02x} is NaN")
    s = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 0xF, code & 0x7
    return s * (m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))


def e2m1_value(code: int) -> float:
    return (-1.0 if code & 0x8 else 1.0) * E2M1[code & 0x7]


_E4M3_TABLE = None


def _e4m3_table() -> torch.Tensor:
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        _E4M3_TABLE = torch.tensor([float("nan") if c in E4M3_NAN else e4m3_value(c) for c in range(256)],
                                   dtype=torch.float64)
    return _E4M3_TABLE


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.contiguous().view(torch.uint8)


def nvfp4_decode(weight_u8: torch.Tensor, scale_e4m3: torch.Tensor, g, group: int = 16,
                 name: str = "nvfp4") -> tuple[torch.Tensor, float]:
    """ModelOpt NVFP4 (V/layers/quantization/utils/nvfp4_emulation_utils.py: break_fp4_bytes / dequantize_to_dtype,
    the checkpoint's linear - unswizzled - scale layout): `weight` U8 [N][K/2], element 2i in the LOW nibble and 2i + 1
    in the high one; `weight_scale` F8_E4M3 [N][K/16] (one per 16 consecutive k); `weight_scale_2` the fp32 global g.
    Returns (W' = s_b x e as float32 holding bf16-exact values [N][K], g as a python float). W' is exact in bf16 and in
    f16 (at most 6 significant bits, |W'| in [2^-10, 2688]); g is NOT folded in (spec 26 §3: y = g x (W' x)). A NaN
    scale code (0x7F / 0xFF) is refused naming the tensor and the element."""
    w = _as_u8(weight_u8)
    s = _as_u8(scale_e4m3)
    N, K2 = w.shape
    K = 2 * K2
    if s.shape != (N, K // group):
        raise ValueError(f"{name}: weight_scale {tuple(s.shape)} != {(N, K // group)} (U8 {tuple(w.shape)}, group {group})")
    bad = (s == E4M3_NAN[0]) | (s == E4M3_NAN[1])
    if bool(bad.any()):
        r, c = (int(v) for v in bad.nonzero()[0])
        raise ValueError(f"{name}: weight_scale[{r}][{c}] is an e4m3 NaN code (0x{int(s[r, c]):02x})")
    gv = float(g.float().reshape(-1)[0]) if isinstance(g, torch.Tensor) else float(g)
    if isinstance(g, torch.Tensor) and g.numel() != 1:
        raise ValueError(f"{name}: weight_scale_2 has {g.numel()} values, want one (the per-tensor global scale)")
    if not math.isfinite(gv):
        raise ValueError(f"{name}: weight_scale_2 {gv} is not finite")
    codes = torch.stack([w & 0xF, w >> 4], -1).reshape(N, K).long()
    e = torch.tensor([e2m1_value(c) for c in range(16)], dtype=torch.float64)[codes]
    sb = _e4m3_table()[s.long()].repeat_interleave(group, dim=1)
    wp = e * sb                                                     # exact in fp64 (<= 6 significant bits)
    out = wp.float()
    if not torch.equal(out.double(), wp) or not torch.equal(out.to(torch.bfloat16).float(), out):
        raise AssertionError(f"{name}: W' is not exact in bf16 (impossible for e2m1 x e4m3)")
    return out, gv


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
    if cfg.form == "lilicorr":
        return _read_nvfp4_tensors(raw, qc)
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


G_SUFFIX = ".weight_global"     # an NVFP4 linear's g under its canonical name (W' stays at `.weight`)


def _read_nvfp4_tensors(raw: dict, qc: dict | None) -> dict[str, torch.Tensor]:
    """The LiLiCorr checkpoint's tensors under canonical names: every NVFP4 linear (`weight` U8 + `weight_scale`
    F8_E4M3 + `weight_scale_2` F32) as W' (float32, bf16-exact) at `<base>.weight` and its g as a 0-d float64 tensor at
    `<base>.weight_global` (Drafter keeps it apart and applies it to the linear's output); BF16 tensors as stored. An
    `input_scale` (W4A16's unused activation scale, V/models/lilicorr.py:502-506) is dropped; anything else quantised
    (a scale without its U8 weight, a U8 weight without the three) is refused."""
    group = int((qc or {}).get("group_size", 16))
    bases = {n[: -len(".weight_scale_2")] for n in raw if n.endswith(".weight_scale_2")}
    if bases and qc is None:
        raise ValueError("NVFP4 tensors without a quantization_config")
    t = {}
    for n, v in raw.items():
        base = next((b for b in (n[: -len(sfx)] for sfx in (".weight", ".weight_scale", ".weight_scale_2",
                                                              ".input_scale") if n.endswith(sfx)) if b in bases), None)
        if base is not None:
            if n == base + ".weight":
                for need in (".weight_scale", ".weight_scale_2"):
                    if base + need not in raw:
                        raise KeyError(f"{base}{need} (NVFP4 {base})")
                if v.dtype != torch.uint8:
                    raise ValueError(f"{n}: an NVFP4 weight is U8, got {v.dtype}")
                wp, g = nvfp4_decode(v, raw[base + ".weight_scale"], raw[base + ".weight_scale_2"], group, base)
                t[_canon(n)] = wp
                t[_canon(base) + G_SUFFIX] = torch.tensor(g, dtype=torch.float64)
            continue
        if n.endswith((".weight_scale", ".weight_scale_2", ".input_scale")) or v.dtype in (torch.uint8,):
            raise ValueError(f"{n}: a quantised tensor outside a complete NVFP4 linear")
        t[_canon(n)] = v
    return t


def read_target_embed_head(target_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's input embedding and lm_head (bf16 as stored). Names are found by suffix so
    `model.embed_tokens` and `model.language_model.embed_tokens` both work; mtp.* and the
    vision tower are skipped. A tied target returns the embedding twice."""
    wm = _weight_map(target_dir)
    emb = [n for n in wm if n.endswith("embed_tokens.weight")
           and not n.startswith(("mtp.", "model.visual", "visual"))]
    if not emb:     # Nemotron-H (Bolt, spec 26): language_model.{model,backbone}.embeddings.weight; vision_model.* skipped
        emb = [n for n in wm if n.endswith(("model.embeddings.weight", "backbone.embeddings.weight"))
               and not n.startswith("vision")]
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
                 upcast_head: bool = False, dtype: torch.dtype = torch.float32) -> "Drafter":
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
    if cfg.form == "muse" and mask is not None:
        raise ValueError("a Muse-Glimmer assistant ships no mask embedding (the mask row is the target's "
                         f"embedding row {cfg.mask_token_id}); refusing {drafter_dir}/mask_embedding.pt")
    return Drafter(cfg, t, embed, lm_head, mask_embedding=mask, upcast_head=upcast_head, dtype=dtype)


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


def lilicorr_head_shapes(c: DFlashConfig) -> dict[str, tuple]:
    """Every tensor of the LiLiCorr head (V/models/lilicorr.py:125-256, the checkpoint's `lilicorr.*` names) and its
    shape. Every linear carries a bias (ReplicatedLinear / NemotronHMLP with bias=True); the NVFP4 ones (token_proj,
    pass_hidden_proj, context_proj, layers.N.attn.out_proj, layers.N.mlp.{up,down}_proj) are W' + g by then."""
    H, Hl, F_, S, k, nh = c.hidden, c.lil_hidden, c.factor_dim, c.block_size - 1, c.cand_topk, c.lil_heads
    P = "lilicorr."
    s = {}

    def lin(name, o, i):
        s[P + name + ".weight"] = (o, i)
        s[P + name + ".bias"] = (o,)
    if Hl != H:
        lin("token_proj", Hl, H)
    lin("pass_hidden_proj", Hl, H)
    s[P + "feature_norm.weight"] = (5,)
    s[P + "feature_norm.bias"] = (5,)
    lin("feature_mlp.up_proj", Hl, 5)
    lin("feature_mlp.down_proj", Hl, Hl)
    s[P + "slot_embedding"] = (1, 1, S, 1, Hl)
    s[P + "rank_embedding"] = (1, 1, 1, k, Hl)
    s[P + "relative_slot_bias"] = (nh, 2 * c.block_size - 1)
    s[P + "same_slot_bias"] = (nh,)
    lin("context_proj", Hl, H)
    for i in range(c.lil_layers):
        L = f"layers.{i}."
        s[P + L + "attn_norm.weight"] = (Hl,)
        s[P + L + "attn.in_proj_weight"] = (3 * Hl, Hl)
        s[P + L + "attn.in_proj_bias"] = (3 * Hl,)
        lin(L + "attn.out_proj", Hl, Hl)
        s[P + L + "mlp_norm.weight"] = (Hl,)
        lin(L + "mlp.up_proj", c.lil_mlp, Hl)
        lin(L + "mlp.down_proj", Hl, c.lil_mlp)
    s[P + "output_norm.weight"] = (Hl,)
    s[P + "anchor_norm.weight"] = (Hl,)
    lin("factor_input_proj", Hl, 3 * Hl)
    lin("out_head", F_, Hl)
    lin("in_head", F_, Hl)
    lin("anchor_out_head", F_, Hl)
    return s


@dataclass
class ContextKV:
    """Target-derived K/V of committed positions (spec 19 §2 'State'). k/v: per layer
    [N, n_kv, hd] after k_norm + RoPE (v after v_scale); fc: the hidden_norm(fc(taps)) plane."""
    pos: torch.Tensor
    k: list[torch.Tensor]
    v: list[torch.Tensor]
    fc: torch.Tensor
    fc_raw: torch.Tensor | None = None      # fc(taps) before hidden_norm (the vLLM forms; a test's mutation reads it)

    def anchor_row(self, p: int) -> torch.Tensor | None:
        """LiLiCorr's anchor (spec 26 §2.3): the normed context row of the NEWEST committed position, p - 1 when the
        block's anchor token sits at p (V/spec_decode/lilicorr/speculator.py:71-87: hidden_norm(fc(taps)) of the last
        accepted verify row); None when p - 1 is not in the context (anchor_valid false)."""
        hit = (self.pos == p - 1).nonzero()
        return None if hit.numel() == 0 else self.fc[int(hit[0, 0])]


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
    # LiLiCorr (spec 26a): cand_ids / unary are all block_size - 1 slots' (unary = the candidates' log-probs), edges =
    # lattice [7][8 previous][8 next] (x logit_scale, fp32), scores = the realized rows lattice[s][previous] [K][8]
    log_probs: torch.Tensor | None = None
    lattice: torch.Tensor | None = None
    anchor: torch.Tensor | None = None    # the anchor row the head read (zeros when anchor_valid is false)


class Drafter:
    """`tensors` uses the checkpoint names (fc.weight, hidden_norm.weight, norm.weight,
    layers.{i}.self_attn.{q,k,v,o}_proj.weight, ..., layers.{i}.attention_conv.base_kernel,
    candidate_selector.*). Linear weights are upcast to float32 once; the codebooks, embedding
    and head keep their stored dtype and are upcast per use (`upcast_head` caches the head)."""

    V1_TOP_K = 16   # v1 has no selector; DraftOut still reports a top-16 for the D1 gate

    def __init__(self, cfg: DFlashConfig, tensors: dict[str, torch.Tensor], embed: torch.Tensor,
                 lm_head: torch.Tensor, mask_embedding: torch.Tensor | None = None,
                 upcast_head: bool = False, dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        # The Muse form computes in `dtype` exactly as transformers' modules do (bf16 = the checkpoint's own
        # arithmetic, float32 = the D1 reference); the vLLM forms are float32 compute (dtype unused).
        self.dtype = dtype if cfg.form == "muse" else torch.float32
        if cfg.form == "muse" and cfg.vocab == 0:
            cfg.vocab = cfg.draft_vocab = int(embed.shape[0])   # the assistant's config carries none: the target's
        keep = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
        # Spec 26: an NVFP4 linear's g rides beside its W' (G_SUFFIX) and is applied to the linear's output (_lin).
        self.g = {n[: -len(G_SUFFIX)] + ".weight": float(v) for n, v in tensors.items() if n.endswith(G_SUFFIX)}
        self.w = {n: (v if n in keep else v.to(self.dtype)) for n, v in tensors.items() if not n.endswith(G_SUFFIX)}
        # Test-only mutations of the LiLiCorr form (test_dflash_ref.py's Review Focus checks; empty = the reference).
        self.mut: set[str] = set()
        self.embed = embed
        self.lm_head = lm_head.to(self.dtype) if (upcast_head or cfg.form == "muse") else lm_head
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
        if c.form == "lilicorr":
            exp.update(lilicorr_head_shapes(c))
        elif c.version == 2:
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
        if c.form == "lilicorr":
            sinks = {f"layers.{i}.self_attn.attention_sink_bias" for i in range(c.n_layers)}
            for n in sinks:
                if tuple(w[n].shape) != (nh,):
                    raise ValueError(f"{n}: shape {tuple(w[n].shape)} != {(nh,)}")
            extra = sorted(set(w) - set(exp) - sinks - {"fc.weight"})
            if extra:
                raise ValueError(f"LiLiCorr: tensors this form does not read: {extra[:8]}")
            stray = sorted(set(self.g) - set(exp) - {"fc.weight"})
            if stray:
                raise ValueError(f"LiLiCorr: a global scale without a linear: {stray}")

    # -- linears ----------------------------------------------------------------------------

    def _lin(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        """x @ W.T (+ b) in float32 by checkpoint name. An NVFP4 linear (spec 26 §3) is g x (x @ W'.T) + b: W' exact,
        g one fp32 multiply on the output (the factorisation that IS exact); mutation `fold_g` folds g into W' instead
        (Review Focus 3's check). Every other weight: `linear` unchanged (the vLLM / Muse forms are bitwise as before)."""
        w = self.w[name]
        b = self.w.get(bias) if bias else None
        g = self.g.get(name)
        if g is None:
            return linear(x, w, b)
        if "fold_g" in self.mut:
            return linear(x, (w.float() * g).float(), b)
        y = linear(x, w) * torch.tensor(g, dtype=torch.float32)
        return y if b is None else y + b.float()

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
        if c.form == "muse":
            return self._m_context_kv([taps[i] for i in c.target_layer_ids], pos)
        raw = self._lin(torch.cat(xs, -1), "fc.weight")
        ctx = rms_norm(raw, self.w["hidden_norm.weight"], c.eps)
        ks, vs = [], []
        for i in range(c.n_layers):
            k, v = self._kv(i, ctx, pos)
            ks.append(k)
            vs.append(v)
        return ContextKV(pos=pos, k=ks, v=vs, fc=ctx, fc_raw=raw)

    def _kv(self, i: int, x: torch.Tensor, pos: torch.Tensor):
        c, L = self.cfg, f"layers.{i}.self_attn."
        N = x.shape[0]
        k = self._lin(x, L + "k_proj.weight", L + "k_proj.bias").view(N, c.n_kv_heads, c.head_dim)
        v = self._lin(x, L + "v_proj.weight", L + "v_proj.bias").view(N, c.n_kv_heads, c.head_dim)
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
        q = self._lin(x, L + "q_proj.weight", L + "q_proj.bias").view(R, c.n_heads, c.head_dim)
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
        pr = self._softmax_sink(i, s, R)
        o = torch.einsum("hqk,khd->qhd", pr, v).reshape(R, c.n_heads * c.head_dim)
        return self._lin(o, L + "o_proj.weight", L + "o_proj.bias")

    def _softmax_sink(self, i: int, s: torch.Tensor, R: int) -> torch.Tensor:
        """softmax over the keys of s [heads, R, Kv]; with sinks (attention_sink_bias) a per-head logit joins the
        denominator - not scaled by head_dim^-0.5, no value row (Review Focus 4). Mutations: `no_sink`, `sink_scaled`."""
        c = self.cfg
        if not c.attention_sink_bias or "no_sink" in self.mut:
            return torch.softmax(s, -1)
        sink = self.w[f"layers.{i}.self_attn.attention_sink_bias"].float()
        if "sink_scaled" in self.mut:
            sink = sink * c.head_dim ** -0.5
        sink = sink[:, None, None].expand(-1, R, 1)
        return torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        L = f"layers.{i}.mlp."
        g = self._lin(x, L + "gate_proj.weight")
        u = self._lin(x, L + "up_proj.weight")
        return self._lin(torch.nn.functional.silu(g) * u, L + "down_proj.weight")

    def _conv_prepare(self, i: int, which: str, x: torch.Tensor, block_len: int):
        """DFlashGroupedConv.prepare (qwen3_dflash2.py:217): coefficients from x, side 0
        applied to x now, side 1 kept for finish."""
        c, L = self.cfg, f"layers.{i}.{which}."
        G = c.hidden // c.conv_group_size
        co = self._lin(x, L + "kernel_projection.weight").view(x.shape[0], 2, c.conv_kernel_size, G)
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
        if c.form == "muse":
            return self._m_draft_block(anchor_id, p, K, ctx, temperature, seed)
        if c.form == "lilicorr":
            return self._l_draft_block(anchor_id, p, K, ctx, temperature, seed)
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

    # -- the Muse-Glimmer form (spec 23f): transformers 5.19.0's own ops, in self.dtype -------------
    #
    # Each line restates MuseGlimmerAssistantModel (A:) / DFlashTokenCandidateGenerator (G:) op for op, so in
    # bfloat16 the arithmetic is the checkpoint's own and in float32 it is the D1 reference. MuseSession below drives
    # these ops exactly as transformers' cache does (bitwise); draft_block is the stateless form over a ContextKV
    # (every context position projected once, the window applied in the mask) that dflash_accept batches.

    def _m_lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.w[name])

    def _m_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """MuseGlimmerAssistantRMSNorm (A:50-55): x-hat in fp32, rounded to the input dtype, THEN w x it (two
        roundings in bf16); norm_two_roundings False is the one-rounding form (tests: Review Focus 4)."""
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        if self.cfg.norm_two_roundings:
            return w * h.to(dt)
        return (w.float() * h).to(dt)

    def _m_cos_sin(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MuseGlimmerAssistantRotaryEmbedding (A:340-356): fp32 inv_freq = 1 / theta^(2i / hd), freqs = pos x
        inv_freq in fp32, emb = [freqs | freqs], cos / sin in fp32 then rounded to the dtype. [N, hd]."""
        hd = self.cfg.head_dim
        inv = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
        freqs = pos.to(torch.float32)[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * 1.0).to(self.dtype), (emb.sin() * 1.0).to(self.dtype)

    @staticmethod
    def _m_rot_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def _m_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """apply_rotary_pos_emb's lines (A:126-131): x [1, heads, L, hd] with cos / sin broadcast over the heads."""
        return (x * cos) + (self._m_rot_half(x) * sin)

    def _m_encode(self, taps: list) -> torch.Tensor:
        """MuseGlimmerAssistantContextProjection (A:302-309): output_norm_enc(fc(cat(taps)))."""
        x = torch.cat([t.to(self.dtype) for t in taps], dim=-1)
        return self._m_norm(self._m_lin(x, "fc.weight"), self.w["hidden_norm.weight"])

    def _m_heads(self, i: int, x: torch.Tensor, which: str) -> torch.Tensor:
        """x [1, L, H] -> the layer's q / k / v heads [1, heads, L, hd], q / k normed (A:181-187); q x q_scale."""
        c, L = self.cfg, f"layers.{i}.self_attn."
        y = self._m_lin(x, L + which + "_proj.weight")
        y = y.view(*x.shape[:-1], -1, c.head_dim).transpose(1, 2)
        if which == "v":
            return y
        y = self._m_norm(y, self.w[L + which + "_norm.weight"])
        if which == "q" and c.q_scale != 1.0:
            y = y * c.q_scale
        return y

    def _m_visible(self, qpos: torch.Tensor, kpos: torch.Tensor, i: int) -> torch.Tensor:
        """create_bidirectional_sliding_window_mask (MU:142-152): |q - k| <= window (window - 1 when not inclusive)."""
        w = self.cfg.windows[i]
        lim = w if self.cfg.window_inclusive else w - 1
        return (qpos[:, None] - kpos[None, :]).abs() <= lim

    def _m_attend(self, i: int, q, k, v, qpos, kpos) -> torch.Tensor:
        """eager_attention_forward (A:80-102) + o_proj: q [1, nh, R, hd], k / v [1, nkv, Kv, hd] (roped)."""
        c = self.cfg
        g = c.n_heads // c.n_kv_heads
        B, nkv, kv_len, hd = k.shape
        if g > 1:   # repeat_kv (A:68-77)
            k = k[:, :, None, :, :].expand(B, nkv, g, kv_len, hd).reshape(B, nkv * g, kv_len, hd)
            v = v[:, :, None, :, :].expand(B, nkv, g, kv_len, hd).reshape(B, nkv * g, kv_len, hd)
        s = torch.matmul(q, k.transpose(2, 3)) * (c.head_dim ** -0.5)
        dt = q.dtype
        mask = torch.where(self._m_visible(qpos, kpos, i), torch.tensor(0.0, dtype=dt), torch.finfo(dt).min)
        s = s + mask[None, None]
        p = torch.nn.functional.softmax(s, dim=-1, dtype=torch.float32).to(dt)
        o = torch.matmul(p, v).transpose(1, 2).contiguous()
        o = o.reshape(*o.shape[:2], -1)
        return self._m_lin(o, f"layers.{i}.self_attn.o_proj.weight")

    def _m_mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        L = f"layers.{i}.mlp."
        return self._m_lin(torch.nn.functional.silu(self._m_lin(x, L + "gate_proj.weight")) *
                           self._m_lin(x, L + "up_proj.weight"), L + "down_proj.weight")

    def _m_embed(self, anchor_id: int) -> torch.Tensor:
        """[anchor, mask x (block - 1)] as the target's RAW rows (G:1563-1579: the weight-less norm bypassed),
        [1, block, H]; embed_norm True applies the target's norm (Review Focus 4's mutation)."""
        c = self.cfg
        ids = torch.tensor([anchor_id] + [c.mask_token_id] * (c.block_size - 1))
        x = self.embed[ids].to(self.dtype)[None]
        if c.embed_norm:   # MuseGlimmerRMSNorm without scale: x * pow(mean(x^2) + eps, -0.5) in fp32, one cast
            xf = x.float()
            x = (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + c.eps, -0.5)).to(self.dtype)
        return x

    def _m_layers(self, x: torch.Tensor, qpos: torch.Tensor, kv) -> torch.Tensor:
        """The decoder layers (A:253-269) and the final norm (A:444) over the block x [1, R, H] at qpos; kv(i, h)
        returns this layer's (k, v [1, nkv, Kv, hd] roped, kpos [Kv]) given the block's normed rows h."""
        c = self.cfg
        cos, sin = self._m_cos_sin(qpos)
        resid = x
        for i in range(c.n_layers):
            L = f"layers.{i}."
            h = self._m_norm(resid, self.w[L + "input_layernorm.weight"])
            q = self._m_rope(self._m_heads(i, h, "q"), cos[None, None], sin[None, None])
            k, v, kpos = kv(i, h)
            resid = resid + self._m_attend(i, q, k, v, qpos, kpos)
            resid = resid + self._m_mlp(i, self._m_norm(resid, self.w[L + "post_attention_layernorm.weight"]))
        return self._m_norm(resid, self.w["norm.weight"])

    def _m_context_kv(self, taps: list, pos: torch.Tensor) -> ContextKV:
        """Every given position's context K / V: c = encoder(taps) (A:405), per layer k = rope(k_norm(k_proj(c))),
        v = v_proj(c) (A:177-192) - [N, nkv, hd] in the dtype; fc = c."""
        ctx = self._m_encode([t[None] for t in taps])            # [1, N, H]
        cos, sin = self._m_cos_sin(pos)
        ks, vs = [], []
        for i in range(self.cfg.n_layers):
            k = self._m_rope(self._m_heads(i, ctx, "k"), cos[None, None], sin[None, None])
            v = self._m_heads(i, ctx, "v")
            ks.append(k[0].transpose(0, 1).contiguous())
            vs.append(v[0].transpose(0, 1).contiguous())
        return ContextKV(pos=pos, k=ks, v=vs, fc=ctx[0])

    def _m_draft_block(self, anchor_id: int, p: int, K: int, ctx: ContextKV, temperature: float,
                       seed: int | None) -> "DraftOut":
        """The stateless block: 16 rows at p .. p + 15 (row 0 the anchor, rows 1..15 the mask), keys = the context
        positions in [p - window, p - 1] plus the block's own (bidirectional); drafts = the RAW head on rows 1..K
        (rows past K computed and discarded - the block is always block_size rows)."""
        c = self.cfg
        B = c.block_size
        qpos = torch.arange(p, p + B)
        lo = p - (c.windows[0] if c.window_inclusive else c.windows[0] - 1)
        sel = (ctx.pos < p) & (ctx.pos >= lo)
        cos, sin = self._m_cos_sin(qpos)

        def kv(i, h):
            kb = self._m_rope(self._m_heads(i, h, "k"), cos[None, None], sin[None, None])
            vb = self._m_heads(i, h, "v")
            kc = ctx.k[i][sel].transpose(0, 1)[None]
            vc = ctx.v[i][sel].transpose(0, 1)[None]
            return torch.cat([kc, kb], 2), torch.cat([vc, vb], 2), torch.cat([ctx.pos[sel], qpos])

        with torch.no_grad():
            h = self._m_layers(self._m_embed(anchor_id), qpos, kv)[0]
            logits = torch.nn.functional.linear(h[1:], self.lm_head).float()          # [15, V] raw (G:1658)
        return self._m_out(logits, h, p, K, temperature, seed)

    def _m_out(self, logits: torch.Tensor, h: torch.Tensor, p: int, K: int, temperature: float,
               seed: int | None) -> "DraftOut":
        """DraftOut from the raw head rows of mask rows 1..K: greedy = torch.argmax (the lowest id among equals,
        G:1685); sampled = gumbel_argmax over the head; q = softmax(raw / T) (decision 11's raw form)."""
        c = self.cfg
        lg = logits[:K]
        keys = torch.arange(c.vocab)
        if temperature == 0.0:
            ids = lg.argmax(-1)
        else:
            ids = torch.tensor([gumbel_argmax(lg[j], keys, temperature, seed, p + j) for j in range(K)])
        cand, unary = top_k_stable(lg, self.V1_TOP_K)
        hit = cand == ids[:, None]
        chosen = torch.where(hit.any(1), hit.float().argmax(1), torch.full((K,), -1))
        T = temperature if temperature > 0 else 1.0
        q = torch.softmax(lg.double() / T, -1).float()
        return DraftOut(ids=ids, cand_ids=cand, unary=unary, scores=unary, edges=None, chosen=chosen, q=q,
                        hidden=h, positions=torch.arange(p, p + c.block_size))

    # -- the LiLiCorr form (spec 26a): vLLM's LiLiCorrHead + the DFlash 2 walk, float32 -----------------------------
    #
    # The backbone is the vLLM DFlash 2 path above (sinks in _softmax_sink, the window |q - k| <= 1023, the NVFP4
    # linears through _lin). The block ALWAYS has block_size rows (anchor + 7 masks; spec 26 §5.3: rows past K computed
    # and discarded) and the head always scores all 7 slots, so K = 1..7 from one block are prefixes of one walk.

    def lilicorr_anchor(self, anchor_id: int, p: int, ctx: ContextKV) -> tuple[torch.Tensor, bool]:
        """(the anchor row [H], anchor_valid): the normed context row at p - 1 (ContextKV.anchor_row); zeros and False
        when it is not in the context. Mutations (Review Focus 1): `anchor_embed` (the anchor token's embedding row),
        `anchor_prenorm` (fc's output before hidden_norm), `anchor_p2` (the row at p - 2), `anchor_invalid`."""
        m = self.mut
        if "anchor_embed" in m:
            return self.embed[anchor_id].float(), True
        q = p - 1 if "anchor_p2" not in m else p - 2
        row = ctx.anchor_row(q + 1)
        if row is not None and "anchor_prenorm" in m:
            if ctx.fc_raw is None:
                raise ValueError("anchor_prenorm needs the context's pre-norm plane (context_kv keeps it)")
            row = ctx.fc_raw[int((ctx.pos == q).nonzero()[0, 0])]
        if row is None or "anchor_invalid" in m:
            return torch.zeros(self.cfg.hidden), False
        return row.float(), True

    def _hlin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """A head linear by its base name, as vLLM evaluates it: F.linear(x, W, b) - or, for an NVFP4 one,
        g x F.linear(x, W') + b (spec 26 §3; `fold_g` folds instead)."""
        w, b = self.w[name + ".weight"], self.w.get(name + ".bias")
        g = self.g.get(name + ".weight")
        F_ = torch.nn.functional
        if g is None:
            return F_.linear(x, w.float(), None if b is None else b.float())
        if "fold_g" in self.mut:
            return F_.linear(x, (w.float() * g).float(), None if b is None else b.float())
        y = F_.linear(x, w.float()) * torch.tensor(g, dtype=torch.float32)
        return y if b is None else y + b.float()

    def lilicorr_slot_bias(self, n_slots: int) -> torch.Tensor:
        """The lattice attention's additive bias [heads][S k][S k] (V/models/lilicorr.py:303-324): per head
        relative_slot_bias[slot(i) - slot(j) + block_size - 1] (the TRAINED offset, also for S < 7) + same_slot_bias
        where slot(i) == slot(j). Mutations: `rel_offset_s` (offset = n_slots: the short-draft misread V:315-317 warns
        about - the same at S = 7, wrong below), `rel_sign` (slot(j) - slot(i)), `no_same_slot`."""
        c, P, m = self.cfg, "lilicorr.", self.mut
        slot = torch.arange(n_slots).repeat_interleave(c.cand_topk)
        rel = slot[:, None] - slot[None, :]
        if "rel_sign" in m:
            rel = -rel
        off = n_slots if "rel_offset_s" in m else (c.block_size - 1)
        bias = self.w[P + "relative_slot_bias"].float()[:, rel + off]
        if "no_same_slot" not in m:
            same = (slot[:, None] == slot[None, :]).float()
            bias = bias + same[None] * self.w[P + "same_slot_bias"].float()[:, None, None]
        return bias

    def lilicorr_lattice(self, cand: torch.Tensor, lp: torch.Tensor, rows: torch.Tensor, anchor: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
        """vLLM's LiLiCorrHead.forward (V/models/lilicorr.py:326-425), float32, batched: cand [B][S][k] candidate ids,
        lp [B][S][k] their full-vocabulary log-probs, rows [B][S][H] the final-normed backbone rows of mask rows
        1..S ("pass_hidden"), anchor [B][H] the anchor rows, valid [B] bool -> the lattice [B][S][k (previous)][k (next)]
        x logit_scale: slot 0 = start broadcast over the previous index, slot s >= 1 = pair[s - 1] (Review Focus 2).
        The log-probs enter only as the five features. Mutations: `lp_topk` (log-probs normalised over the top k only),
        `start_transposed` (slot 0 = start[previous] broadcast over next), `pair_shift` (slot s scored against itself),
        `no_scale`, `cross_no_anchor` (W_cross applied to h, not h x a)."""
        c, P, m = self.cfg, "lilicorr.", self.mut
        F_ = torch.nn.functional
        B, S, k = cand.shape
        Hl, nh = c.lil_hidden, c.lil_heads
        hd = Hl // nh
        emb = self.embed[cand.reshape(-1)].float().view(B, S, k, -1)
        tok = self._hlin(emb, P + "token_proj") if Hl != c.hidden else emb
        pas = self._hlin(rows.float(), P + "pass_hidden_proj").unsqueeze(-2)
        lp = lp.float()
        if "lp_topk" in m:
            lp = lp - torch.logsumexp(lp, -1, keepdim=True)
        rank = (torch.arange(k, dtype=torch.float32) / float(k - 1)) if k > 1 else torch.zeros(k)
        top1 = torch.zeros(k)
        top1[0] = 1.0
        f = torch.stack([lp, lp.exp(), lp - lp.max(dim=-1, keepdim=True).values, rank.expand_as(lp),
                         top1.expand_as(lp)], dim=-1)
        f = F_.layer_norm(f, (5,), self.w[P + "feature_norm.weight"].float(), self.w[P + "feature_norm.bias"].float(),
                          c.feature_eps)
        fm = self._hlin(F_.silu(self._hlin(f, P + "feature_mlp.up_proj")), P + "feature_mlp.down_proj")
        h = tok + pas
        h = h + fm
        h = h + self.w[P + "slot_embedding"].float()[:, 0, :S]
        h = h + self.w[P + "rank_embedding"].float()[:, 0]
        h = h.reshape(B, S * k, Hl)
        a = self._hlin(anchor.float(), P + "context_proj")
        a = a * valid[:, None].to(a.dtype)
        bias = self.lilicorr_slot_bias(S)[None]
        for i in range(c.lil_layers):
            L = f"{P}layers.{i}."
            x = F_.rms_norm(h, (Hl,), self.w[L + "attn_norm.weight"].float(), c.eps)
            qkv = F_.linear(x, self.w[L + "attn.in_proj_weight"].float(), self.w[L + "attn.in_proj_bias"].float())
            q, kk, v = qkv.chunk(3, dim=-1)
            q, kk, v = (t.view(B, S * k, nh, hd).transpose(1, 2) for t in (q, kk, v))
            o = F_.scaled_dot_product_attention(q, kk, v, attn_mask=bias)
            h = h + self._hlin(o.transpose(1, 2).reshape(B, S * k, Hl), L + "attn.out_proj")
            x = F_.rms_norm(h, (Hl,), self.w[L + "mlp_norm.weight"].float(), c.eps)
            h = h + self._hlin(F_.silu(self._hlin(x, L + "mlp.up_proj")), L + "mlp.down_proj")
        h = F_.rms_norm(h, (Hl,), self.w[P + "output_norm.weight"].float(), c.eps).reshape(B, S, k, Hl)
        a = F_.rms_norm(a, (Hl,), self.w[P + "anchor_norm.weight"].float(), c.eps)
        if any(n in self.g for n in (P + "factor_input_proj.weight", P + "out_head.weight", P + "in_head.weight")):
            raise ValueError("factor_input_proj / out_head / in_head are BF16 in the checkpoint (no global scale)")
        W = self.w[P + "factor_input_proj.weight"].float()
        w_self, w_anchor, w_cross = W[:, :Hl].contiguous(), W[:, Hl:2 * Hl].contiguous(), W[:, 2 * Hl:].contiguous()
        arow = a[:, None, None, :]
        pre = F_.linear(h, w_self, self.w[P + "factor_input_proj.bias"].float())
        pre = pre + F_.linear(arow, w_anchor)
        pre = pre + F_.linear(h if "cross_no_anchor" in m else h * arow, w_cross)
        fh = F_.silu(pre)
        ew = torch.cat([self.w[P + "out_head.weight"], self.w[P + "in_head.weight"]], 0).float().contiguous()
        eb = torch.cat([self.w[P + "out_head.bias"], self.w[P + "in_head.bias"]], 0).float().contiguous()
        out_v, in_v = F_.normalize(F_.linear(fh, ew, eb).unflatten(-1, (2, c.factor_dim)), dim=-1,
                                   eps=c.vector_eps).unbind(-2)
        a_out = F_.normalize(self._hlin(a, P + "anchor_out_head"), dim=-1, eps=c.vector_eps)
        start = (a_out[:, None, :] * in_v[:, 0, :, :]).sum(dim=-1)                        # [B][k]
        if "pair_shift" in m:
            pairs = torch.matmul(out_v[:, 1:], in_v[:, 1:].transpose(-1, -2))
        else:
            pairs = torch.matmul(out_v[:, :-1], in_v[:, 1:].transpose(-1, -2))            # [B][S - 1][k][k]
        scale = 1.0 if "no_scale" in m else c.lil_logit_scale
        start = scale * start.float()
        pairs = scale * pairs.float()
        if "start_transposed" in m:
            first = start[:, None, :, None].expand(-1, 1, -1, k)
        else:
            first = start[:, None, None, :].expand(-1, 1, k, -1)
        return torch.cat((first, pairs), dim=1)

    def lilicorr_walk(self, lattice: torch.Tensor, cand: torch.Tensor, p: int, K: int, temperature: float = 0.0,
                      seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The shared DFlash2 / LiLiCorr walk (V/spec_decode/dflash2/speculator.py:15-75) over one block's lattice
        [S][k][k]: previous = 0; slot s scores lattice[s][previous] (the lattice ALONE - Review Focus 2), argmax (ties:
        the lower index) or Gumbel-argmax keyed by (seed ^ (s << 32) - spec_step_seed, vLLM 3901d56800 -, position
        p + s = the predicted position - 1, candidate id). Returns (chosen [K], ids [K], realized rows [K][k]).
        Mutation `lp_in_walk` needs the log-probs: see _l_draft_block."""
        prev = 0
        idx, real = [], []
        for s in range(K):
            row = lattice[s, prev]
            sd = None if seed is None else (seed if "no_step_seed" in self.mut else seed ^ (s << 32))
            j = gumbel_argmax(row, cand[s], temperature, sd, p + s)
            idx.append(j)
            real.append(row)
            prev = j
        it = torch.tensor(idx, dtype=torch.long)
        return it, cand[torch.arange(K), it], torch.stack(real)

    def _l_draft_block(self, anchor_id: int, p: int, K: int, ctx: ContextKV, temperature: float,
                       seed: int | None) -> DraftOut:
        c = self.cfg
        S = c.block_size - 1
        h = self.forward_block(self.embed_block(anchor_id, S), p, ctx)
        hm = h[1:]
        cand, lp = lilicorr_candidates(self.head_logits(hm), c.cand_topk)
        anchor, valid = self.lilicorr_anchor(anchor_id, p, ctx)
        lat = self.lilicorr_lattice(cand[None], lp[None], hm[None], anchor[None], torch.tensor([valid]))[0]
        walk_lat = lat + lp[:, None, :] if "lp_in_walk" in self.mut else lat
        chosen, ids, real = self.lilicorr_walk(walk_lat, cand, p, K, temperature, seed)
        T = temperature if temperature > 0 else 1.0
        q = torch.zeros(K, c.vocab)
        q.scatter_(1, cand[:K], torch.softmax(real.double() / T, -1).float())
        return DraftOut(ids=ids, cand_ids=cand, unary=lp, scores=real, edges=lat, chosen=chosen, q=q, hidden=h,
                        positions=torch.arange(p, p + c.block_size), log_probs=lp, lattice=lat, anchor=anchor)


def lilicorr_candidates(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """V/models/lilicorr.py:478-486 (LogitsProcessor.get_top_k_tokens, return_log_probs): per row the top k ids by
    logit (descending; exact ties to the lower id - the engine's comparator; torch's topk keeps either, recorded by the
    tests) and their log-probs = logit - logsumexp over the WHOLE vocabulary (fp32)."""
    lg = logits.float()
    cand, v = top_k_stable(lg, k)
    return cand, v - torch.logsumexp(lg, dim=-1, keepdim=True)


class LiLiCorrSession:
    """One sequence's drafting state as the engine keeps it (spec 26 §5.3-§5.4, 19c's loop): the committed context
    (every committed position's normed fc row and per-layer K / V - the engine's 1024-row ring holds the window of
    it) and the anchor = the newest committed row. `commit(taps, positions)` appends verified rows (the anchor token
    and the accepted drafts of a verify; rejected rows never arrive); `draft(anchor_id, K, ...)` drafts from the
    position after the last commit."""

    def __init__(self, d: Drafter):
        if d.cfg.form != "lilicorr":
            raise ValueError("LiLiCorrSession is the LiLiCorr form's")
        self.d = d
        self.ctx: ContextKV | None = None

    @property
    def next_pos(self) -> int:
        return 0 if self.ctx is None else int(self.ctx.pos[-1]) + 1

    def commit(self, taps: dict, positions) -> None:
        pos = torch.as_tensor(positions, dtype=torch.long).reshape(-1)
        if int(pos[0]) != self.next_pos or not torch.equal(pos, torch.arange(int(pos[0]), int(pos[0]) + len(pos))):
            raise ValueError(f"commit at {pos.tolist()[:3]}...: rows must continue at {self.next_pos}")
        new = self.d.context_kv(taps, pos)
        if self.ctx is None:
            self.ctx = new
            return
        o = self.ctx
        self.ctx = ContextKV(pos=torch.cat([o.pos, new.pos]), k=[torch.cat([a, b]) for a, b in zip(o.k, new.k)],
                             v=[torch.cat([a, b]) for a, b in zip(o.v, new.v)], fc=torch.cat([o.fc, new.fc]),
                             fc_raw=None if o.fc_raw is None or new.fc_raw is None else torch.cat([o.fc_raw, new.fc_raw]))

    def draft(self, anchor_id: int, K: int, temperature: float = 0.0, seed: int | None = None) -> DraftOut:
        return self.d.draft_block(anchor_id, self.next_pos, K, self.ctx, temperature, seed)


class MuseSession:
    """transformers 5.19.0's DFlashTokenCandidateGenerator + DFlashCache (G:1505-1688, CU:2192-2209 over
    DynamicSlidingWindowLayer CU:212-325) for ONE sequence, op for op on the Drafter's Muse ops - BITWISE
    transformers' candidate logits in bf16 and fp32 (test_dflash_ref.py::test_muse_equals_transformers).

        s = MuseSession(drafter)
        ids, logits = s.draft(taps_new, anchor_id)   # taps of the positions committed since the last draft

    Per call: the previous block's 16 rows are cropped from every layer's cache (G:1620-1621; past the window only
    the last window - 1 context rows are kept, CU:296-325), the new context rows are projected by the encoder, and
    every layer projects k / v of [new context | block] in ONE linear (A:177-183), ropes them at [c .. p + 15]
    (c = the context positions before this call, p = c + n the anchor's), appends them, and attends over the cache's
    last window - 1 + n + 16 rows (CU:269-274) under |q - k| <= window. Returns the drafts (argmax of the raw head on
    rows 1..15) and their raw logits [15, V] (float32)."""

    def __init__(self, d: Drafter):
        if d.cfg.form != "muse":
            raise ValueError("MuseSession is the Muse-Glimmer form's (transformers') cache")
        self.d = d
        self.k: list = [None] * d.cfg.n_layers
        self.v: list = [None] * d.cfg.n_layers
        self.c = 0                    # committed context positions in the cache (cumulative_length minus the block)
        self.block_in_cache = False

    def crop(self) -> None:
        """crop(-block): the block's rows go; at or past the window only the last window - 1 context rows stay."""
        B, W = self.d.cfg.block_size, self.d.cfg.windows[0]
        full = self.c + B >= W
        for i in range(len(self.k)):
            if full:
                self.k[i] = self.k[i][:, :, -W + 1 - B:-B]
                self.v[i] = self.v[i][:, :, -W + 1 - B:-B]
            else:
                self.k[i] = self.k[i][:, :, :-B]
                self.v[i] = self.v[i][:, :, :-B]
        self.block_in_cache = False

    @torch.no_grad()
    def draft(self, taps: dict, anchor_id: int):
        d, c = self.d, self.d.cfg
        if self.block_in_cache:
            self.crop()
        B, W = c.block_size, c.windows[0]
        tl = [taps[i] for i in c.target_layer_ids]
        n = tl[0].shape[0]
        ctx = d._m_encode([t[None] for t in tl])                  # [1, n, H]
        p = self.c + n
        qpos = torch.arange(p, p + B)
        allpos = torch.arange(self.c, p + B)
        cos, sin = d._m_cos_sin(allpos)

        def kv(i, h):
            x = torch.cat([ctx, h], dim=1)
            k = d._m_rope(d._m_heads(i, x, "k"), cos[None, None], sin[None, None])
            v = d._m_heads(i, x, "v")
            full_k = k if self.k[i] is None else torch.cat([self.k[i], k], dim=2)
            full_v = v if self.v[i] is None else torch.cat([self.v[i], v], dim=2)
            self.k[i], self.v[i] = full_k, full_v
            nvis = W - 1 + n + B
            ku, vu = full_k[:, :, -nvis:], full_v[:, :, -nvis:]
            return ku, vu, torch.arange(p + B - ku.shape[2], p + B)

        h = d._m_layers(d._m_embed(anchor_id), qpos, kv)
        logits = torch.nn.functional.linear(h[:, 1:], d.lm_head)[0]
        self.c = p
        self.block_in_cache = True
        return logits.argmax(-1), logits.float()
