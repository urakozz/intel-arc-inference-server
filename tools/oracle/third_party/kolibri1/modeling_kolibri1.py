# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the b70-inference-server authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the
# License.
#
# A transformers (5.x) port of Kolibri 1, written for spec 20a of b70-inference-server so that
# AutoRound (which loads models through transformers) can quantise `Aleph-Alpha/Kolibri-1-BF16`.
#
# Source of the semantics: Aleph Alpha's vLLM plugin `aleph-alpha-inference`
# (https://github.com/Aleph-Alpha/aleph-alpha-inference, Apache-2.0, Copyright 2026 Aleph Alpha GmbH),
# `aleph_alpha_inference/kolibri1.py` @ 049a6a7b, which subclasses vLLM's Qwen3-MoE
# (`vllm/model_executor/models/qwen3_moe.py`, Apache-2.0, Copyright the vLLM project). The module
# layout and the RMSNorm / RoPE / eager-attention code follow transformers' Qwen3-MoE
# (`transformers/models/qwen3_moe/modeling_qwen3_moe.py`, Apache-2.0, Copyright 2025 The Qwen team,
# Alibaba Group and the HuggingFace Inc. team). transformers has no Kolibri implementation
# (huggingface/transformers#49281 is an open request).
"""Kolibri 1 for transformers 5.x.

What the plugin defines, and this port implements (tools/oracle/kolibri_ref.py is the layer-streamed
reference on the same rules; docs/probe-kolibri-2026-10-05.md the facts):

  layer      x += post_attn_norm(Attn(input_layernorm(x)));
             x += post_ffn_norm(MoE(post_attention_layernorm(x)))          (sandwich norms)
  RMSNorm    w * bf16(x * rsqrt(mean(x^2) + eps)), statistics in fp32, plain w (not 1 + w)
  attention  48 q / 4 kv heads x 128, q/k RMSNorm per head (before RoPE), scale head_dim^-0.5,
             no bias. `layer_types[i]` "sliding_attention": causal window of `sliding_window` keys
             INCLUDING the query (513 = 512 preceding + itself) and RoPE (neox rotate_half, theta
             1e4, full head_dim). "full_attention": causal, NO positional encoding (NoPE).
  router     logits = fp32(x) @ fp32(W_gate)^T (bf16 operands, fp32 accumulate, fp32 logits);
             select top-k on logits + fp32(expert_bias) (ties: the lower expert id, a stable
             descending sort); weights sigmoid(logits) of the selected, NOT renormalised
  combine    routed = sum over the selected experts in ASCENDING id of fp32(w_e) * fp32(y_e),
             accumulated in fp32; + fp32(shared_expert(x)) (ungated: no sigmoid gate); one rounding
             to the activation dtype. y_e = down(silu(gate x) * up x), SwiGLU, intermediate 512
  head       logits = fp32(h) @ fp32(lm_head)^T when config.head_dtype == "float32" (vLLM's
             head_dtype: the bf16 hidden state and weights, fp32 accumulation, fp32 logits)

Names match the checkpoint: `mlp.gate.weight` (the router, deliberately NOT an nn.Linear so no
quantiser treats it as one), `moe.router.expert_bias`, `mlp.experts.{e}.{gate,up,down}_proj`
(per-expert nn.Linear, what AutoRound quantises), `mlp.shared_experts.{gate,up,down}_proj`,
`self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, and the four norms per layer.

Masks are built inside each attention layer from (a) the 2D padding mask or None and (b) the
layer's own cache length, never passed in pre-built: AutoRound captures the first block's inputs
once and replays them into every block, and layer 0 is a sliding layer, so a pre-built mask would
put the sliding window on the full layers.
"""

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_kolibri1 import Kolibri1Config


class Kolibri1RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * h.to(dt)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rope_inv_freq(config: Kolibri1Config, device=None) -> torch.Tensor:
    rp = config.rope_parameters or {}
    if rp.get("rope_type", "default") != "default":
        raise ValueError(f"rope_type {rp.get('rope_type')!r}: Kolibri 1 uses default RoPE")
    theta = float(rp.get("rope_theta", 10000.0))
    d = config.head_dim
    return 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))


class Kolibri1RotaryEmbedding(nn.Module):
    """cos / sin [B, T, head_dim] in the activation dtype. inv_freq is recomputed in fp32 on every
    call (64 values) rather than held as a buffer, so a model cast with .to(bf16) or built on meta
    cannot round it."""

    def __init__(self, config: Kolibri1Config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv = rope_inv_freq(self.config, device=x.device)
        inv_e = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos_e = position_ids[:, None, :].float()
        freqs = (inv_e @ pos_e).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(h: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kv, s, d = h.shape
    if n_rep == 1:
        return h
    return h[:, :, None, :, :].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


class Kolibri1Attention(nn.Module):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = True
        H, d = config.hidden_size, self.head_dim
        self.q_proj = nn.Linear(H, self.num_heads * d, bias=False)
        self.k_proj = nn.Linear(H, self.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(H, self.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.num_heads * d, H, bias=False)
        self.q_norm = Kolibri1RMSNorm(d, eps=config.rms_norm_eps)
        self.k_norm = Kolibri1RMSNorm(d, eps=config.rms_norm_eps)

    def allowed(self, attention_mask, q_len: int, k_len: int, past: int, device) -> torch.Tensor:
        """[B or 1, 1, q_len, k_len] bool. Query i has absolute index past + i; the k_len keys are the
        contiguous run ending at the last query (a full cache returns everything, a sliding cache the
        last window - 1 plus the new ones), so key j has index past + q_len - k_len + j."""
        q_idx = past + torch.arange(q_len, device=device)
        k_idx = past + q_len - k_len + torch.arange(k_len, device=device)
        ok = k_idx[None, :] <= q_idx[:, None]
        if self.sliding_window is not None:
            ok = ok & ((q_idx[:, None] - k_idx[None, :]) < self.sliding_window)
        ok = ok[None, None]
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("Kolibri1: attention_mask must be the 2D [batch, keys] padding mask "
                                 f"(got {attention_mask.dim()}D); causal and window masks are built per layer")
            if attention_mask.shape[1] < past + q_len:
                raise ValueError(f"attention_mask has {attention_mask.shape[1]} columns for {past + q_len} positions")
            am = attention_mask[:, attention_mask.shape[1] - (past + q_len):].to(device).bool()
            ok = ok & am[:, k_idx][:, None, None, :]
        return ok

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        B, T, _ = hidden_states.shape
        d = self.head_dim
        q = self.q_norm(self.q_proj(hidden_states).view(B, T, -1, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(B, T, -1, d)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, -1, d).transpose(1, 2)
        if self.is_sliding:                           # RoPE only in the sliding layers; full = NoPE
            cos, sin = position_embeddings
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        past = 0
        if past_key_values is not None:
            past = past_key_values.get_seq_length(self.layer_idx)
            k, v = past_key_values.update(k, v, self.layer_idx)
        ok = self.allowed(attention_mask, T, k.shape[2], past, hidden_states.device)
        kk = repeat_kv(k, self.num_key_value_groups)
        vv = repeat_kv(v, self.num_key_value_groups)
        if (getattr(self.config, "_attn_implementation", None) or "eager") == "eager":
            # transformers' eager_attention_forward, with this layer's mask
            w = torch.matmul(q, kk.transpose(2, 3)) * self.scaling
            w = w + torch.where(ok, 0.0, torch.finfo(w.dtype).min).to(w.dtype)
            w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(w, vv)
        else:
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=ok, scale=self.scaling)
        o = o.transpose(1, 2).reshape(B, T, -1).contiguous()
        return self.o_proj(o)


class Kolibri1MLP(nn.Module):
    """SwiGLU: down(silu(gate x) * up x). Routed experts and the (ungated) shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Kolibri1Router(nn.Module):
    """`mlp.gate.weight` [num_experts, hidden]. Not an nn.Linear: the router stays in bf16 and no
    quantiser that walks nn.Linear modules can pick it up. Logits in fp32."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(self, x):
        return F.linear(x.float(), self.weight.float())


class Kolibri1ExpertBias(nn.Module):
    """`moe.router.expert_bias` [num_experts] (the torchtitan name the checkpoint keeps; vLLM loads
    it into `mlp.gate.e_score_correction_bias`). Selection only, never a weight."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)


class Kolibri1MoeHolder(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.router = Kolibri1ExpertBias(num_experts)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int):
    """(ids [N, k] in selection order, weights [N, k] fp32). Selection on logits + bias with ties to
    the lower id (stable descending sort); weights sigmoid(logits), not renormalised."""
    sel = logits.float() + bias.float()
    ids = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, :k]
    return ids, torch.sigmoid(logits.float().gather(1, ids))


class Kolibri1SparseMoeBlock(nn.Module):
    def __init__(self, config: Kolibri1Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gate = Kolibri1Router(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList(
            [Kolibri1MLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.shared_experts = Kolibri1MLP(config.hidden_size, config.shared_expert_intermediate_size)

    def forward(self, hidden_states: torch.Tensor, expert_bias: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        ids, w = route(self.gate(x), expert_bias, self.top_k)
        acc = torch.zeros(x.shape[0], shape[-1], dtype=torch.float32, device=x.device)
        for e in torch.unique(ids).tolist():             # ascending expert id
            tok, slot = torch.where(ids == e)
            y = self.experts[e](x[tok])
            acc.index_add_(0, tok, y.float() * w[tok, slot, None])
        out = acc + self.shared_experts(x).float()
        return out.to(hidden_states.dtype).reshape(shape)


class Kolibri1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Kolibri1Attention(config, layer_idx)
        self.mlp = Kolibri1SparseMoeBlock(config)
        self.moe = Kolibri1MoeHolder(config.num_experts)
        H, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = Kolibri1RMSNorm(H, eps)
        self.post_attn_norm = Kolibri1RMSNorm(H, eps)
        self.post_attention_layernorm = Kolibri1RMSNorm(H, eps)
        self.post_ffn_norm = Kolibri1RMSNorm(H, eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        a = self.self_attn(self.input_layernorm(hidden_states), position_embeddings,
                           attention_mask=attention_mask, past_key_values=past_key_values)
        hidden_states = hidden_states + self.post_attn_norm(a)
        m = self.mlp(self.post_attention_layernorm(hidden_states), self.moe.router.expert_bias)
        return hidden_states + self.post_ffn_norm(m)


class Kolibri1PreTrainedModel(PreTrainedModel):
    config_class = Kolibri1Config
    config: Kolibri1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Kolibri1DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Kolibri1RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, Kolibri1Router):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, Kolibri1ExpertBias):
            init.zeros_(module.expert_bias)


class Kolibri1Model(Kolibri1PreTrainedModel):
    def __init__(self, config: Kolibri1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Kolibri1DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Kolibri1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Kolibri1RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past)[None]
        h = inputs_embeds
        pe = self.rotary_emb(h, position_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:
            h = layer(h, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_values=past_key_values, use_cache=use_cache, position_embeddings=pe)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(h),
                                       past_key_values=past_key_values if use_cache else None)


class Kolibri1ForCausalLM(Kolibri1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config: Kolibri1Config):
        super().__init__(config)
        self.model = Kolibri1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """lm_head with config.head_dtype ("float32": fp32 operands and logits; else the model dtype)."""
        if getattr(self.config, "head_dtype", None) in ("float32", "fp32", torch.float32):
            return F.linear(h.float(), self.lm_head.weight.float())
        return self.lm_head(h)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache)
        h = out.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(h[:, sl, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)


__all__ = ["Kolibri1ForCausalLM", "Kolibri1Model", "Kolibri1PreTrainedModel"]
