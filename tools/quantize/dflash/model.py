#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 drafter as torch modules AutoRound can tune: tools/oracle/dflash_ref.py's forward, batched.

    m = load(drafter_dir, dtype=torch.bfloat16)        # z-lab/Qwen3.8-27B-DFlash2's tensors, names kept
    sel = m(**sample)                                   # calib.py's samples; returns the selector projection

The same arithmetic as `dflash_ref.Drafter` (`context_kv`, `forward_block`, the selector's `hidden_projection`),
for B samples of A blocks each - the batched form of `dflash_accept.forward_blocks`: block a of a sample is the
anchor row and `block_size - 1` mask rows at positions p_a .. p_a + 7 (non-causal, block size 8), and sees the
context positions < p_a within the layer's window. The context plane is `hidden_norm(fc(taps))` of the target's
taps at the sample's context positions (`kpos`, -1 = a padding row nobody sees).

Three block lists for AutoRound (`BLOCKS`), each tuned on its own cached inputs:
  encoder              fc + hidden_norm: the taps [B, C, 5 x 5120] -> the context plane [B, C, 5120]
  layers.0 .. layers.4 the decoder layers: q/k/v/o, gate/up/down and both convs' kernel_projection; k_proj and
                       v_proj also project the context plane (`ctx`, an input of every layer)
  candidate_selector   hidden_projection of the final-norm mask rows [B, A x 7, 5120] -> [B, A x 7, 256]
So every drafter linear is tuned, `fc` and the selector projection included; norms, the conv bases and the
selector codebooks are parameters outside any linear and stay as loaded. `encoder` is the one module that is
not in the checkpoint: `ENCODER_RENAME` maps its tensors back to `fc.*` / `hidden_norm.*` (finalize.py).
RoPE comes in as cos | sin tables (calib.py computes them in float64, as dflash_ref.rope does: the card has no
float64), float32 carried as int32 bits (`q_rope`, `k_rope`): AutoRound casts every floating block input to bf16.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
from torch import nn

_ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oracle")


def _load_ref():
    if "dflash_ref" in sys.modules:
        return sys.modules["dflash_ref"]
    spec = importlib.util.spec_from_file_location("dflash_ref", os.path.join(_ORACLE, "dflash_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dflash_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load_ref()

ENCODER_RENAME = {"encoder.fc.": "fc.", "encoder.hidden_norm.": "hidden_norm."}


def blocks(n_layers: int) -> list[list[str]]:
    """AutoRound's `to_quant_block_names`: three block lists, tuned in this order."""
    return [["encoder"], [f"layers.{i}" for i in range(n_layers)], ["candidate_selector"]]


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return y.to(x.dtype)


def rope(x, table, neox: bool):
    """x [B, N, heads, hd] at the table's positions: cos | sin [B, N, hd] float32 as int32 bits (dflash_ref.rope)."""
    hd = x.shape[-1]
    c, s = table.view(torch.float32)[:, :, None].chunk(2, -1)
    y = x.float()
    if neox:
        a, b = y[..., : hd // 2], y[..., hd // 2:]
        out = torch.cat([a * c - b * s, b * c + a * s], -1)
    else:
        a, b = y[..., 0::2], y[..., 1::2]
        out = torch.stack([a * c - b * s, b * c + a * s], -1).flatten(-2)
    return out.to(x.dtype)


def grouped_conv(x, delta, base, block_len: int, group: int):
    """dflash_ref.grouped_conv for [B, R, H] rows: delta [B, R, taps, H / group], base [taps, H]."""
    B, R, H = x.shape
    coef = base.float()[None, None] + delta.float().repeat_interleave(group, dim=-1)    # [B, R, taps, H]
    xf = x.float()
    out = coef[:, :, 0] * xf
    row_in_block = (torch.arange(R, device=x.device) % block_len)[None, :, None]
    for t in range(1, base.shape[0]):
        shifted = torch.cat([xf.new_zeros(B, t, H), xf[:, :-t]], 1)
        out = out + coef[:, :, t] * shifted * (row_in_block >= t).float()
    return out.to(x.dtype)


class GroupedConv(nn.Module):
    """DFlashGroupedConv: coefficients from the input (kernel_projection), side 0 before, side 1 after."""

    def __init__(self, c):
        super().__init__()
        self.ksize, self.group = c.conv_kernel_size, c.conv_group_size
        self.G = c.hidden // c.conv_group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, c.conv_kernel_size, c.hidden))
        self.kernel_projection = nn.Linear(c.hidden, 2 * c.conv_kernel_size * self.G, bias=False)

    def prepare(self, x, block_len: int):
        B, R, _ = x.shape
        co = self.kernel_projection(x).view(B, R, 2, self.ksize, self.G)
        return grouped_conv(x, co[:, :, 0], self.base_kernel[0], block_len, self.group), co[:, :, 1]

    def finish(self, x, co, block_len: int):
        return grouped_conv(x, co, self.base_kernel[1], block_len, self.group)


class Attention(nn.Module):
    def __init__(self, c, i: int):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.window, self.causal, self.neox, self.v_scale = c.windows[i], c.causal[i], c.neox, c.v_scale
        self.block_len = c.block_size
        self.q_proj = nn.Linear(c.hidden, c.n_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.n_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.n_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_heads * c.head_dim, c.hidden, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)

    def _kv(self, x, table):
        B, N, _ = x.shape
        k = rope(self.k_norm(self.k_proj(x).view(B, N, self.nkv, self.hd)), table, self.neox)
        v = self.v_proj(x).view(B, N, self.nkv, self.hd)
        if self.v_scale is not None:
            v = v * self.v_scale
        return k, v

    def visible(self, qpos, kpos):
        """[B, R, C + R] bool, dflash_accept._attention's rule: a row of the block anchored at p sees the
        context positions < p and its own block's rows, within the window / causality of the layer."""
        R = qpos.shape[1]
        rows = torch.arange(R, device=qpos.device)
        anchor = qpos - (rows % self.block_len)[None]                       # p of each row's block
        ctx = (kpos[:, None, :] >= 0) & (kpos[:, None, :] < anchor[:, :, None])
        blk = ((rows // self.block_len)[:, None] == (rows // self.block_len)[None, :])[None].expand(qpos.shape[0], R, R)
        vis = torch.cat([ctx, blk], -1)
        d = qpos[:, :, None] - torch.cat([kpos, qpos], -1)[:, None, :]
        if self.window is not None:
            vis = vis & (d.abs() <= self.window - 1)
        if self.causal:
            vis = vis & (d >= 0)
        return vis

    def forward(self, x, ctx, qpos, kpos, q_rope, k_rope):
        B, R, _ = x.shape
        q = rope(self.q_norm(self.q_proj(x).view(B, R, self.nh, self.hd)), q_rope, self.neox)
        kb, vb = self._kv(x, q_rope)
        kc, vc = self._kv(ctx, k_rope)
        g = self.nh // self.nkv
        k = torch.cat([kc, kb], 1).repeat_interleave(g, dim=2).float()
        v = torch.cat([vc, vb], 1).repeat_interleave(g, dim=2).float()
        s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) * self.hd ** -0.5
        s = s.masked_fill(~self.visible(qpos, kpos)[:, None], float("-inf"))
        o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v).reshape(B, R, self.nh * self.hd)
        return self.o_proj(o.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.up_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.down_proj = nn.Linear(c.ffn, c.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    """dflash_ref.Drafter.forward_block's loop body (DFlash 2: convs around attention and the MLP)."""

    def __init__(self, c, i: int):
        super().__init__()
        self.block_len = c.block_size
        self.input_layernorm = RMSNorm(c.hidden, c.eps)
        self.post_attention_layernorm = RMSNorm(c.hidden, c.eps)
        self.self_attn = Attention(c, i)
        self.mlp = MLP(c)
        self.attention_conv = GroupedConv(c)
        self.mlp_conv = GroupedConv(c)

    def forward(self, hidden_states, ctx, qpos, kpos, q_rope, k_rope):
        x, n = hidden_states, self.block_len
        h, co = self.attention_conv.prepare(self.input_layernorm(x), n)
        h = self.self_attn(h, ctx, qpos, kpos, q_rope, k_rope)
        x = x + self.attention_conv.finish(h, co, n)
        h, co = self.mlp_conv.prepare(self.post_attention_layernorm(x), n)
        return x + self.mlp_conv.finish(self.mlp(h), co, n)


class Encoder(nn.Module):
    """dflash_ref.Drafter.context_kv's plane: hidden_norm(fc(cat(taps)))."""

    def __init__(self, c, fc_in: int):
        super().__init__()
        self.fc = nn.Linear(fc_in, c.hidden, bias=False)
        self.hidden_norm = RMSNorm(c.hidden, c.eps)

    def forward(self, hidden_states):
        return self.hidden_norm(self.fc(hidden_states))


class CandidateSelector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hidden_projection = nn.Linear(c.hidden, c.selector_rank, bias=False)
        self.predecessor_codebook = nn.Parameter(torch.zeros(c.vocab, c.selector_rank))
        self.successor_codebook = nn.Parameter(torch.zeros(c.vocab, c.selector_rank))

    def forward(self, hidden_states):
        return self.hidden_projection(hidden_states)


def _calib_classes():
    from transformers import PretrainedConfig, PreTrainedModel

    class CalibConfig(PretrainedConfig):
        model_type = "dflash2_calib"

    class DFlash2Calib(PreTrainedModel):
        config_class = CalibConfig
        base_model_prefix = "dflash2_calib"
        _no_split_modules = ["Layer"]

        def __init__(self, config, fc_in: int):
            super().__init__(config)
            c = self.dflash = ref.config_from_dict(config.dflash_source)
            if c.version != 2 or c.attention_bias or c.attention_sink_bias:
                raise NotImplementedError("the calibration model is DFlash 2 without attention biases / sinks")
            self.encoder = Encoder(c, fc_in)
            self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layers))
            self.norm = RMSNorm(c.hidden, c.eps)
            self.candidate_selector = CandidateSelector(c)

        def forward(self, input_ids=None, attention_mask=None, taps=None, embeds=None, qpos=None, kpos=None,
                    q_rope=None, k_rope=None, use_cache=None, **_):
            """`input_ids` / `attention_mask` are AutoRound's bookkeeping (the block ids, all valid)."""
            ctx = self.encoder(taps)
            h = embeds
            for layer in self.layers:
                h = layer(h, ctx=ctx, qpos=qpos, kpos=kpos, q_rope=q_rope, k_rope=k_rope)
            h = self.norm(h)
            B, R, H = h.shape
            n = self.dflash.block_size
            mask_rows = h.view(B, R // n, n, H)[:, :, 1:].reshape(B, -1, H)
            return self.candidate_selector(mask_rows)

    return CalibConfig, DFlash2Calib


def load(drafter_dir: str, dtype=torch.bfloat16):
    """The calibration model with the drafter's tensors as stored (strict: every checkpoint tensor has a home)."""
    CalibConfig, DFlash2Calib = _calib_classes()
    c = ref.read_config(drafter_dir)
    t = ref.read_drafter_tensors(drafter_dir, c)
    raw = c.raw
    config = CalibConfig(
        dflash_source=raw, hidden_size=c.hidden, intermediate_size=c.ffn, num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads, head_dim=c.head_dim,
        vocab_size=c.vocab, rms_norm_eps=c.eps, tie_word_embeddings=False)
    with torch.device("meta"):
        m = DFlash2Calib(config, fc_in=t["fc.weight"].shape[1])
    back = {v: k for k, v in ENCODER_RENAME.items()}
    sd = {}
    for n, v in t.items():
        for a, b in back.items():
            if n.startswith(a):
                n = b + n[len(a):]
        sd[n] = v.to(dtype)
    m.load_state_dict(sd, strict=True, assign=True)
    return m.eval()


def export_name(n: str) -> str:
    """The calibration model's tensor name -> the checkpoint's."""
    for a, b in ENCODER_RENAME.items():
        if n.startswith(a):
            return b + n[len(a):]
    return n
