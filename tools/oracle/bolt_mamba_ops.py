#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a: plain-torch implementations of the five training-kernel functions Bolt's remote code calls.

The checkpoint's own modelling file (R = modeling_nemotron_h.py, blob de290021) has two Mamba-2 paths: the kernel
path `cuda_kernels_forward` (R:384-532), which calls mamba_ssm's / causal_conv1d's GPU kernels, and a torch fallback
`torch_forward` (R:535-726) with other roundings and a dt clamp. Spec 25 Decision 1: the reference is the KERNEL path,
run on the CPU through the functions below - fp32 inside, no dt clamp (`dt_limit` (0, inf)), every reduction in one
fixed order that 25c's host twin copies. R itself is never edited; `stub_package()` makes R's imports resolve to these.

The orders (each a choice of this file - the GPU kernels' tiled orders are not defined to the bit; spec 25 §2.2):
  conv      acc = f32(b); acc = acc + f32(w_j) * f32(x_j) for the taps OLDEST FIRST (j = 0..K-1, x_{K-1} the new
            column); y = rne(silu(acc)) with silu(v) = v / (1 + exp(-v)) - the bias seeds the accumulator as
            causal_conv1d's and vLLM's kernels do; bf16 x bf16 products are exact in fp32, so fma or not is moot;
            ONE rounding. The conv state holds RAW xBC columns (R:487-490, R:568-572).
  dt        dt = softplus(f32(dt_raw) + f32(dt_bias)), softplus(v) = v <= 20 ? log1p(exp(v)) : v (mamba_ssm's
            threshold); NO clamp (`dt_clamp` is a test switch only: R's fallback clamps at time_step_min, R:595 /
            R:643, which Decision 1 does not copy).
  A         A = -exp(f32(A_log)) - torch's fp32 exp, exactly R:416 / R:448. Spec 25c's switch (A_EXP, env
            B70_BOLT_A_EXP, default "torch"): "engine" computes A with exp_engine - the loader's own function
            (src/common/exp_torch.h, Sleef's expf u10) in torch fp32 ops - so a Mamba twin held BITWISE against
            these ops takes the engine's A (25b measured torch.exp (MKL's vsExp) 1 ulp off it on 196 of 33,266 bf16
            values). R's kernel path computes A itself (R:416, R:448); under "engine" the stubs below recover A_log
            from R's A (exactly: A_log is a bf16 value, asserted) and use the engine's A instead.
  update    dA = exp(A * dt); dB[n] = f32(B[g][n]) * dt; S = S * dA + dB * f32(x)   (mamba_ssm selective_state_update's
            order: products rounded to fp32, then the add - no fma); y = rne(tree_n(S * f32(C[g][n])) + f32(x) * D)
            with tree_n the halving pairwise tree over the 128 (power of two) state entries:
            v[0:n/2] + v[n/2:n], then again, down to one value.
  chunk     the SSD form over sub-chunks of `chunk` rows (128 - the config's chunk_size), every quantity fp32: the
            sequential cumsum of A*dt inside a sub-chunk, exp(segsum) masked BEFORE the exp (no inf * 0 - the NaN
            class vLLM 783380af39 fixed), C B^T, the state passed sub-chunk to sub-chunk. Its sums are torch einsums
            (matmuls): the chunked form is held to the recurrence within a measured bound, never bitwise (spec 25
            §2.2: prefill is held to decode's fill by cosine bars).
  gnorm     rmsnorm_fn with norm_before_gate False (R:300-308): v = f32(x) * (f32(z) * sigmoid(f32(z))) with
            sigmoid(u) = 1 / (1 + exp(-u)) (the training kernels' gate form); per group of `group` (1024 on Bolt,
            d_inner / n_groups): ms = tree(v * v) / group; out = rne((v * (1 / sqrt(ms + eps))) * f32(w)).

    conv_update / conv_seq / ssm_update / ssm_chunk / rmsnorm_gated / softplus_mamba / A_from_log / tree_sum
    exp_engine / engine_A_of     spec 25c: the loader's exp in torch fp32 ops; R's A -> the engine's
    stub_package()  -> {module name: module}  the mamba_ssm / causal_conv1d names R imports (bolt_remote installs them)
"""
import importlib.machinery
import math
import os
import types

import torch

CALLS: dict[str, int] = {}          # stub entry -> calls (bolt_tiny_check prints it)
CHUNK = 128
DT_CLAMP_TEST: float | None = None  # a TEST switch: the stubs clamp dt at this floor (R's fallback's 0.001) when set
# Spec 25c: which exp computes A = -exp(A_log) - "torch" (R's torch.exp, the reference's default) or "engine" (the
# loader's src/common/exp_torch.h, exp_engine below). bolt_ref.py run --a-exp sets it; the env var is the default.
A_EXP: str = os.environ.get("B70_BOLT_A_EXP", "torch")
if A_EXP not in ("torch", "engine"):
    raise ValueError(f"B70_BOLT_A_EXP={A_EXP!r}: torch or engine")


def _count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


# ------------------------------------------------------------------------------------------------
# elementwise pieces

def tree_sum(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The halving pairwise tree over `dim` (length a power of two): v[:n/2] + v[n/2:], repeated. fp32 in, fp32 out."""
    v = v.movedim(dim, -1)
    n = v.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"tree_sum over {n} values: the twin's tree needs a power of two")
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = v[..., :h] + v[..., h:]
    return v[..., 0]


def softplus_mamba(v: torch.Tensor) -> torch.Tensor:
    """mamba_ssm's softplus: v <= 20 ? log1p(exp(v)) : v (fp32)."""
    return torch.where(v <= 20.0, torch.log1p(torch.exp(v)), v)


_F32 = torch.float32


def _c32(x: float) -> torch.Tensor:
    """A C++ `float` constant: the literal rounded once to fp32."""
    return torch.tensor(x, dtype=_F32)


_RLN2 = _c32(1.442695040888963407359924681001892137426645954152985934135449406931)
_L2U, _L2L = _c32(0.693145751953125), _c32(1.428606765330187045e-06)
_POLY = [_c32(c) for c in (0.000198527617612853646278381, 0.00139304355252534151077271, 0.00833336077630519866943359,
                           0.0416664853692054748535156, 0.166666671633720397949219, 0.5)]


def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fma(a, b, c) through float64: a x b is exact in float64 (24 + 24 bits); the add rounds once in float64 and
    once to fp32. In exp_engine's chain the float64 sum is exact (the Cody-Waite reduction; each polynomial step spans
    <= 51 bits for |s| <= ln2 / 2) except where s is tiny, where a double rounding needs a float64 result ON an fp32
    midpoint - checked against the C++ function over every bf16 value in [-7, 9] (bolt_exp_compare.py: 0 differ)."""
    return (a.double() * b.double() + c.double()).to(_F32)


def exp_engine(d: torch.Tensor) -> torch.Tensor:
    """src/common/exp_torch.h's exp_torch (Sleef_expf8_u10 at AVX2 + FMA) in torch fp32 ops, step for step: q =
    nearest-even(d x 1/ln2); s = fma(q, -L2U, d); s = fma(q, -L2L, s); the u10 polynomial by fma; u = 1 + fma(s^2, u,
    s); u x 2^(q >> 1) x 2^(q - (q >> 1)); 0 below -104, inf above 100. fp32 in, fp32 out; NaN is not handled."""
    d = d.to(_F32)
    q = torch.round(d * _RLN2)                         # an fp32 product, then round-half-even (cvtps2dq)
    qi = q.to(torch.int32)
    s = _fma32(q, -_L2U, d)
    s = _fma32(q, -_L2L, s)
    u = _POLY[0].expand_as(s)
    for c in _POLY[1:]:
        u = _fma32(u, s, c.expand_as(s))
    ss = s * s
    u = _fma32(ss, u, s) + _c32(1.0)
    e1 = qi >> 1
    e2 = qi - e1

    def pow2(e: torch.Tensor) -> torch.Tensor:
        return ((e.clamp(-126, 127) + 127) << 23).to(torch.int32).view(_F32)
    u = u * pow2(e1)
    u = u * pow2(e2)
    u = torch.where(d < -104.0, torch.zeros_like(u), u)
    return torch.where(d > 100.0, torch.full_like(u, math.inf), u)


def A_from_log(A_log: torch.Tensor, a_exp: str | None = None) -> torch.Tensor:
    """A = -exp(A_log) in fp32 (R:416, R:448: `-torch.exp(self.A_log.float())`); a_exp (default A_EXP) "engine":
    -exp_engine(A_log) - the loader's A (src/loader/bolt_repack.cc bolt_neg_exp)."""
    a_exp = a_exp or A_EXP
    if a_exp == "engine":
        return -exp_engine(A_log.float())
    if a_exp != "torch":
        raise ValueError(f"a_exp {a_exp!r}: torch or engine")
    return -torch.exp(A_log.float())


def engine_A_of(A: torch.Tensor) -> torch.Tensor:
    """R's A (= -torch.exp(A_log.float()), the kernel path's own, R:416 / R:448) -> the engine's: A_log is recovered as
    bf16(log(-A)) - exact, since A_log is a bf16 value and torch.exp is within an ulp; asserted by recomputing R's A
    bitwise - and A = -exp_engine(A_log)."""
    A_log = torch.log(-A.float()).to(torch.bfloat16).float()
    if not torch.equal(-torch.exp(A_log), A.float()):
        raise RuntimeError("engine_A_of: A is not -torch.exp of a bf16 A_log (B70_BOLT_A_EXP=engine needs R's A)")
    return -exp_engine(A_log)


def silu_conv(v: torch.Tensor) -> torch.Tensor:
    """The conv's SiLU: v / (1 + exp(-v)) (causal_conv1d's and vLLM's form)."""
    return v / (1.0 + torch.exp(-v))


def silu_gate(z: torch.Tensor) -> torch.Tensor:
    """The gated norm's SiLU: z * sigmoid(z), sigmoid(u) = 1 / (1 + exp(-u)) (layernorm_gated's form)."""
    return z * (1.0 / (1.0 + torch.exp(-z)))


def dt_of(dt_raw: torch.Tensor, dt_bias: torch.Tensor, dt_clamp: float | None = None) -> torch.Tensor:
    """dt = softplus(f32(dt_raw) + f32(dt_bias)); no clamp unless the test switch names a floor."""
    dt = softplus_mamba(dt_raw.float() + dt_bias.float())
    if dt_clamp is not None:
        dt = torch.clamp(dt, min=dt_clamp)
    return dt


# ------------------------------------------------------------------------------------------------
# the conv (causal_conv1d_update / causal_conv1d_fn)

def conv_update(x_new: torch.Tensor, ring_cols: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """One decode column. x_new [B][C] (the new raw xBC), ring_cols [B][C][K-1] (the previous K-1 raw columns,
    oldest first; zeros before position 0), w [C][K], b [C] -> y [B][C] in x_new's dtype (or out_dtype)."""
    K = w.shape[-1]
    wf = w.float()
    acc = b.float().expand(x_new.shape[0], -1).clone() if b is not None else torch.zeros(x_new.shape, dtype=torch.float32)
    for j in range(K - 1):
        acc = acc + wf[:, j] * ring_cols[..., j].float()
    acc = acc + wf[:, K - 1] * x_new.float()
    return silu_conv(acc).to(out_dtype or x_new.dtype)


def conv_seq(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, state: torch.Tensor | None = None,
             out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """A sequence. x [B][C][T] raw, state [B][C][K-1] the raw columns before x (zeros when None) -> y [B][C][T].
    Element for element the same operations as conv_update (chunked = whole, bitwise)."""
    Bt, C, T = x.shape
    K = w.shape[-1]
    if state is None:
        state = torch.zeros(Bt, C, K - 1, dtype=x.dtype)
    full = torch.cat([state.float(), x.float()], -1)                       # [B][C][K-1+T]
    wf = w.float()
    acc = b.float()[None, :, None].expand(Bt, C, T).clone() if b is not None else torch.zeros(Bt, C, T)
    for j in range(K):
        acc = acc + wf[None, :, j, None] * full[:, :, j:j + T]
    return silu_conv(acc).to(out_dtype or x.dtype)


# ------------------------------------------------------------------------------------------------
# the SSM (selective_state_update / mamba_chunk_scan_combined)

def _per_head(t: torch.Tensor, rep: int, dim: int) -> torch.Tensor:
    return t.repeat_interleave(rep, dim=dim)


def ssm_update(S: torch.Tensor, x: torch.Tensor, dt_raw: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor | None, dt_bias: torch.Tensor, dt_clamp: float | None = None,
               out_dtype: torch.dtype | None = None):
    """One decode row. S [Bt][H][P][N] fp32, x [Bt][H][P], dt_raw [Bt][H], A [H] fp32 (A_from_log), B / C [Bt][G][N],
    D [H], dt_bias [H] -> (y [Bt][H][P] in x's dtype, the new S fp32). The order of the module docstring."""
    H, G = x.shape[1], B.shape[1]
    dt = dt_of(dt_raw, dt_bias, dt_clamp)                                    # [Bt][H]
    dA = torch.exp(A.float()[None, :] * dt)                                  # [Bt][H]
    Bh = _per_head(B.float(), H // G, 1)                                     # [Bt][H][N]
    Ch = _per_head(C.float(), H // G, 1)
    dB = Bh * dt[..., None]                                                  # [Bt][H][N]
    xf = x.float()
    S = S.float() * dA[..., None, None] + dB[:, :, None, :] * xf[..., None]
    y = tree_sum(S * Ch[:, :, None, :])
    if D is not None:
        y = y + xf * D.float()[None, :, None]
    return y.to(out_dtype or x.dtype), S


def cumsum_seq(v: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Sequential fp32 prefix sum along `dim` (one add per step, in order)."""
    v = v.movedim(dim, 0)
    out = torch.empty_like(v)
    acc = v[0].clone()
    out[0] = acc
    for i in range(1, v.shape[0]):
        acc = acc + v[i]
        out[i] = acc
    return out.movedim(0, dim)


def ssm_chunk(x: torch.Tensor, dt_raw: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
              D: torch.Tensor | None, dt_bias: torch.Tensor, S0: torch.Tensor | None = None, chunk: int = CHUNK,
              dt_clamp: float | None = None, out_dtype: torch.dtype | None = None):
    """A sequence of one batch row in the SSD (chunked) form. x [T][H][P], dt_raw [T][H], A [H], B / C [T][G][N],
    D [H], dt_bias [H], S0 [H][P][N] fp32 (zeros when None) -> (y [T][H][P] in x's dtype, the final S fp32).
    Sub-chunks of `chunk` rows from the first row of x (a continuation's sub-chunks start where it starts)."""
    T, H, P = x.shape
    G, N = B.shape[1], B.shape[2]
    S = torch.zeros(H, P, N) if S0 is None else S0.float().clone()
    y = torch.empty(T, H, P, dtype=torch.float32)
    Af = A.float()
    Dv = D.float() if D is not None else None
    for a in range(0, T, chunk):
        L = min(chunk, T - a)
        xf = x[a:a + L].float()                                              # [L][H][P]
        dt = dt_of(dt_raw[a:a + L], dt_bias, dt_clamp)                       # [L][H]
        cs = cumsum_seq(Af[None, :] * dt, 0)                                 # [L][H]
        Bh = _per_head(B[a:a + L].float(), H // G, 1)                        # [L][H][N]
        Ch = _per_head(C[a:a + L].float(), H // G, 1)
        seg = cs[:, None, :] - cs[None, :, :]                                # [t][s][H]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[:, :, None]
        Lm = torch.exp(torch.where(causal, seg, torch.tensor(-math.inf)))    # masked BEFORE the exp
        CB = torch.einsum("thn,shn->tsh", Ch, Bh)
        Mw = CB * Lm * dt[None, :, :]                                        # [t][s][H]
        y_diag = torch.einsum("tsh,shp->thp", Mw, xf)
        y_off = torch.einsum("thn,hpn->thp", Ch, S) * torch.exp(cs)[..., None]
        yc = y_diag + y_off
        if Dv is not None:
            yc = yc + xf * Dv[None, :, None]
        y[a:a + L] = yc
        decay = torch.exp(cs[-1][None, :] - cs) * dt                         # [L][H]
        S = S * torch.exp(cs[-1])[:, None, None] + torch.einsum("sh,shp,shn->hpn", decay, xf, Bh)
    return y.to(out_dtype or x.dtype), S


# ------------------------------------------------------------------------------------------------
# the gated RMSNorm (rmsnorm_fn)

def rmsnorm_gated(x: torch.Tensor, z: torch.Tensor | None, w: torch.Tensor, eps: float, group: int,
                  norm_before_gate: bool = False) -> torch.Tensor:
    """mamba_ssm rmsnorm_fn (bias None) over groups of `group` channels; norm_before_gate False is Bolt's
    (R:300-308): v = f32(x) * silu_gate(f32(z)), per group v * (1 / sqrt(tree(v v) / group + eps)) * w, cast to x's
    dtype. norm_before_gate True (the mutation the tests reject) normalises x first and gates after."""
    D = x.shape[-1]
    if D % group:
        raise ValueError(f"group {group} does not divide {D}")
    v = x.float()
    if z is not None and not norm_before_gate:
        v = v * silu_gate(z.float())
    vg = v.reshape(*v.shape[:-1], D // group, group)
    ms = tree_sum(vg * vg) / group
    rstd = 1.0 / torch.sqrt(ms + eps)
    out = (vg * rstd[..., None]).reshape(v.shape) * w.float()
    if z is not None and norm_before_gate:
        out = out * silu_gate(z.float())
    return out.to(x.dtype)


# ------------------------------------------------------------------------------------------------
# the stub package: the names R imports, with the training kernels' signatures

def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    _count("rmsnorm_fn")
    if bias is not None:
        raise NotImplementedError("rmsnorm_fn with a bias (Bolt has none)")
    if not upcast:
        raise NotImplementedError("rmsnorm_fn upcast=False")
    return rmsnorm_gated(x, z, weight, eps, group_size or x.shape[-1], norm_before_gate)


def _tied(t: torch.Tensor, dims: tuple[int, ...], name: str) -> torch.Tensor:
    """The per-head value of an expanded tensor (R:418-421 expand per head over head_dim / state): the expanded dims
    must be broadcasts (stride 0), as the Triton kernel's TIE_HDIM case assumes."""
    for d in dims:
        if t.stride(d) != 0 and t.shape[d] != 1:
            raise NotImplementedError(f"selective_state_update: {name} varies along dim {d} (untied head_dim)")
    idx = [slice(None)] * t.dim()
    for d in dims:
        idx[d] = 0
    return t[tuple(idx)]


def _selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                            state_batch_indices=None):
    """mamba_ssm selective_state_update with R's arguments (R:425-436): state [B][H][P][N] (updated in place), x / dt
    [B][H][P], A [H][P][N], B / C [B][G][N], D / dt_bias [H][P], dt_softplus True -> out [B][H][P]."""
    _count("selective_state_update")
    if z is not None or state_batch_indices is not None or not dt_softplus or dt_bias is None:
        raise NotImplementedError("selective_state_update: only R's call (z None, dt_softplus, dt_bias)")
    A_h = _tied(A, (1, 2), "A")
    if A_EXP == "engine":
        A_h = engine_A_of(A_h)
    y, S = ssm_update(state, x, _tied(dt, (2,), "dt"), A_h, B, C,
                      None if D is None else _tied(D, (1,), "D"), _tied(dt_bias, (1,), "dt_bias"), DT_CLAMP_TEST)
    state.copy_(S)
    return y


def _limit_floor(dt_limit) -> float | None:
    lo, hi = float(dt_limit[0]), float(dt_limit[1])
    if hi != math.inf:
        raise NotImplementedError(f"dt_limit {dt_limit}: an upper clamp (Bolt's is (0, inf))")
    return None if lo <= 0.0 else lo


def _mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None,
                               seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                               return_final_states=False, return_varlen_states=False):
    """mamba_ssm mamba_chunk_scan_combined with R's arguments (R:511-525): x [B][T][H][P], dt [B][T][H], A [H],
    B / C [B][T][G][N], D [H], dt_bias [H], dt_softplus True, dt_limit (0, inf) -> (out [B][T][H][P], final [B][H][P][N])."""
    _count("mamba_chunk_scan_combined")
    if z is not None or seq_idx is not None or cu_seqlens is not None or return_varlen_states or not dt_softplus:
        raise NotImplementedError("mamba_chunk_scan_combined: only R's call")
    floor = _limit_floor(dt_limit)
    if DT_CLAMP_TEST is not None:
        floor = DT_CLAMP_TEST
    if A_EXP == "engine":
        A = engine_A_of(A)
    outs, finals = [], []
    for b in range(x.shape[0]):
        y, S = ssm_chunk(x[b], dt[b], A, B[b], C[b], D, dt_bias,
                         None if initial_states is None else initial_states[b], chunk_size, dt_clamp=floor)
        outs.append(y)
        finals.append(S)
    out = torch.stack(outs)
    return (out, torch.stack(finals)) if return_final_states else out


def _mamba_split_conv1d_scan_combined(*_a, **_k):
    raise NotImplementedError("mamba_split_conv1d_scan_combined is R's training path (R:455-475): never on inference")


def _causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, return_final_states=False,
                      final_states_out=None, activation=None):
    """causal_conv1d_fn with R's arguments (R:496-501): x [B][C][T], weight [C][K], bias [C], activation silu."""
    _count("causal_conv1d_fn")
    if activation not in ("silu", "swish") or seq_idx is not None or return_final_states:
        raise NotImplementedError("causal_conv1d_fn: only R's call (silu)")
    st = None
    if initial_states is not None:
        st = initial_states[..., -(weight.shape[-1] - 1):]
    return conv_seq(x, weight, bias, st)


def _causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """causal_conv1d_update with R's arguments (R:403-409): x [B][C] the new raw column, conv_state [B][C][S]
    (S >= K-1; R keeps conv_kernel = 4 columns) updated in place - shifted by one, x written last."""
    _count("causal_conv1d_update")
    if activation not in ("silu", "swish") or cache_seqlens is not None or x.dim() != 2:
        raise NotImplementedError("causal_conv1d_update: only R's call (one column, silu)")
    K = weight.shape[-1]
    y = conv_update(x, conv_state[..., -(K - 1):], weight, bias)
    conv_state.copy_(torch.cat([conv_state[..., 1:], x[..., None].to(conv_state.dtype)], -1))
    return y


def stub_package() -> dict[str, types.ModuleType]:
    """{name: module} for mamba_ssm (.ops.triton.{layernorm_gated, selective_state_update, ssd_combined}) and
    causal_conv1d, exposing the functions above under the training kernels' names."""
    mods = {}
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__bolt_stub__ = True
        mods[name] = m
    ln = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")
    ln.rmsnorm_fn = _rmsnorm_fn
    ssu = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
    ssu.selective_state_update = _selective_state_update
    ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd.mamba_chunk_scan_combined = _mamba_chunk_scan_combined
    ssd.mamba_split_conv1d_scan_combined = _mamba_split_conv1d_scan_combined
    cc = types.ModuleType("causal_conv1d")
    cc.causal_conv1d_fn = _causal_conv1d_fn
    cc.causal_conv1d_update = _causal_conv1d_update
    for m in (ln, ssu, ssd, cc):
        m.__bolt_stub__ = True
        mods[m.__name__] = m
    for name, m in mods.items():     # a spec (find_spec needs one), no __file__ / __version__: transformers'
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=hasattr(m, "__path__"))  # checks say absent
    mods["mamba_ssm"].ops = mods["mamba_ssm.ops"]
    mods["mamba_ssm.ops"].triton = mods["mamba_ssm.ops.triton"]
    mods["mamba_ssm.ops.triton"].layernorm_gated = ln
    mods["mamba_ssm.ops.triton"].selective_state_update = ssu
    mods["mamba_ssm.ops.triton"].ssd_combined = ssd
    return mods


KERNEL_FUNCTIONS = {
    "selective_state_update": _selective_state_update,
    "mamba_chunk_scan_combined": _mamba_chunk_scan_combined,
    "mamba_split_conv1d_scan_combined": _mamba_split_conv1d_scan_combined,
    "causal_conv1d_fn": _causal_conv1d_fn,
    "causal_conv1d_update": _causal_conv1d_update,
}
