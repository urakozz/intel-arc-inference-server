#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a Task 2: the tiny Bolt variants and the training-kernel ops (bolt_mamba_ops.py).

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_bolt_make_tiny.py [test_name ...]   (pytest collects it too)

The variants are written by bolt_make_tiny.py under $BOLT_TINY_DIR (default <repo>/oracle-out-bolt-tiny) from the
pinned snapshot in the HF cache; everything SKIPs (exit 77 from the runner) without it. Each variant loads through
bolt_remote.build_lm; the int4 dequant is (q - 8) x scale exactly and the g128 -> g64 expansion is exact; the ops are
held to the formulas they restate and each Mamba trap is a mutation that must change the result.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("bolt_remote")
ops = _load("bolt_mamba_ops")
mk = _load("bolt_make_tiny")
REPO = os.path.dirname(os.path.dirname(_HERE))
TINY_DIR = os.environ.get("BOLT_TINY_DIR", os.path.join(REPO, "oracle-out-bolt-tiny"))
BF, F32 = torch.bfloat16, torch.float32


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def snapshot() -> str:
    s = br.find_snapshot()
    if s is None:
        _skip("the pinned Bolt snapshot is not in the HF cache (BOLT_SNAPSHOT, /hf or ~/.cache/huggingface)")
    br.load_remote(s)
    return s


def variant(name: str, signed: bool = False) -> str:
    """The variant's directory (written on first use; int4 variants with their dequantised twin `<dir>-twin`)."""
    snapshot()
    sub = name + ("-signed" if signed else "")
    d = os.path.join(TINY_DIR, sub)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        mk.make(d, name, 0, signed=signed, twin=d + "-twin" if name.startswith("int4") else None)
    return d


def rand_ids(n, seed, hi=131072):
    return torch.randint(0, hi, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def unpack(qweight: torch.Tensor) -> torch.Tensor:
    """qweight I32 [K/8][N] -> q [K][N] (nibble k % 8 of word k / 8, low first), independently of dequant.py."""
    w = qweight.to(torch.int64) & 0xFFFFFFFF
    return torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], 1).reshape(-1, qweight.shape[1])


# ------------------------------------------------------------------------------------------------
# the variants

def test_variants_load():
    for name in mk.VARIANTS:
        d = variant(name)
        dtype = F32 if name == "pattern12-fp32" else BF
        model = mk.load_model(d, dtype)
        ids = torch.tensor([rand_ids(20, 1)])
        with torch.no_grad():
            lg = model(input_ids=ids, past_key_values=br.fp32_cache(model), use_cache=True).logits
        assert torch.isfinite(lg).all(), name
        cfg = json.load(open(os.path.join(d, "config.json")))
        n = sum(1 for _ in model.parameters())
        print(f"  {name}: loads strict ({n} parameters, {cfg['llm_config']['dtype']}), a 20-id forward finite")
        if name.startswith("int4"):
            twin = mk.load_model(d + "-twin", BF)
            for (ka, a), (kb, b) in zip(sorted(model.state_dict().items()), sorted(twin.state_dict().items())):
                assert ka == kb and torch.equal(a, b), ka
            print(f"  {name}: its dequantised twin is the same model tensor for tensor")


def test_pattern_is_real():
    s = snapshot()
    real = [b["block_type"] for b in br.llm_config(s)["block_configs"]]
    llm, _ = mk.read_state(variant("pattern12"))
    tiny = [b["block_type"] for b in llm["block_configs"]]
    assert tiny == real[:12], (tiny, real[:12])
    print(f"  pattern12's blocks are the real model's first 12: {''.join({'mamba': 'M', 'moe': 'E', 'attention': '*'}[t] for t in tiny)}")


def test_f32_tensors_hold_bf16_values():
    from safetensors import safe_open
    with safe_open(os.path.join(variant("pattern12"), "model.safetensors"), framework="pt") as h:
        names = list(h.keys())
        for k in names:
            t = h.get_tensor(k)
            if k.endswith(mk.F32_NAMES):
                assert t.dtype == F32, k
                if not k.endswith("e_score_correction_bias"):
                    assert torch.equal(t, t.to(BF).float()), k
            elif t.is_floating_point():
                assert t.dtype == BF, k
        b = h.get_tensor("language_model.model.layers.1.mixer.gate.e_score_correction_bias")
        dtb = torch.cat([h.get_tensor(k) for k in names if k.endswith(".dt_bias")])
    assert not torch.equal(b, b.to(BF).float())
    assert dtb.min() == -10.5 and dtb.max() == 8.5
    print(f"  F32 A_log / D / dt_bias hold bf16 values; the router bias is fp32 ({b.min():.4f} .. {b.max():.4f}); "
          f"dt_bias {dtb.min():.2f} .. {dtb.max():.2f}")


def test_ties_really_tie():
    model = mk.load_model(variant("ties"), BF)
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(3)).to(BF)
    for layer, (a, c) in mk.TIES.items():
        gate = model.model.layers[layer].mixer.gate
        s = F.linear(x.float(), gate.weight.float()).sigmoid()
        sel = s + gate.e_score_correction_bias
        top = sel.topk(10, dim=-1)
        assert torch.equal(top.values[:, 8], top.values[:, 9]), layer
        assert all(set(r[8:10]) == {a, c} for r in top.indices.tolist()), layer
    print(f"  ties: the planted pairs {mk.TIES} are exactly 9th / 10th on 64 of 64 rows")


def test_int4_dequant_exact():
    from safetensors import safe_open
    for name, g in (("int4-g64", 64), ("int4-g128-gptq", 128)):
        d = variant(name)
        with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as h:
            keys = [k for k in h.keys() if k.endswith(".qweight")]
            neg = tot = 0
            for k in keys:
                b = k[: -len(".qweight")]
                qw, sc, qz = h.get_tensor(k), h.get_tensor(b + ".scales"), h.get_tensor(b + ".qzeros")
                assert sc.dtype == torch.float16 and qz.eq(0x77777777).all(), b
                assert sc.shape[0] * g == qw.shape[0] * 8, b
                want = ((unpack(qw).float() - 8.0) * sc.float().repeat_interleave(g, 0)).to(BF).t()
                assert torch.equal(mk.dequant(qw, sc, g), want), b
                if name == "int4-g128-gptq":
                    gi = h.get_tensor(b + ".g_idx")
                    assert torch.equal(gi, torch.arange(qw.shape[0] * 8, dtype=torch.int32) // 128), b
                neg += int((sc < 0).sum())
                tot += sc.numel()
        cfg = json.load(open(os.path.join(d, "config.json")))["quantization_config"]
        print(f"  {name}: {len(keys)} linears, dequant = (q - 8) x scale exactly; {neg} / {tot} scales negative; "
              f"quant_method {cfg['quant_method']}, group {cfg['group_size']}"
              + (f", dynamic + rules {[k for k in cfg.get('dynamic', {}) if k.startswith('+')]}" if 'dynamic' in cfg else ""))
        if name == "int4-g128-gptq":
            assert cfg["quant_method"] == "gptq" and cfg["group_size"] == 128 and 0.3 < neg / tot < 0.7


def test_g128_expansion_exact():
    stream = _load("stream")
    from safetensors import safe_open
    with safe_open(os.path.join(variant("int4-g128-gptq"), "model.safetensors"), framework="pt") as h:
        n = 0
        for k in [k for k in h.keys() if k.endswith(".qweight")]:
            qw, sc = h.get_tensor(k), h.get_tensor(k[: -len(".qweight")] + ".scales")
            a = stream.dequant_t(qw, sc, 128)
            b = stream.dequant_t(qw, sc.repeat_interleave(2, 0).contiguous(), 64)
            assert torch.equal(a, b), k
            n += 1
    print(f"  the g128 -> g64 expansion (each scale row written to both g64 rows) dequantises bitwise equal: {n} linears")


def test_signed_scales_pack():
    w = (torch.randn(64, 256, generator=torch.Generator().manual_seed(5)) / 16).to(BF)
    qw, qz, sc = mk.pack_int4(w, 64, signed=True)
    q = unpack(qw).t().reshape(64, 4, 64)                                # [N][groups][64]
    wf = w.float().reshape(64, 4, 64)
    ext = wf.abs().argmax(-1, keepdim=True)
    assert torch.equal(torch.gather(q, -1, ext), torch.zeros_like(ext)), "the extreme maps to q = 0 (-8)"
    frac = float((sc < 0).float().mean())
    deq = mk.dequant(qw, sc, 64).float()
    err_s = (deq - w.float()).abs().max().item()
    qw2, _, sc2 = mk.pack_int4(w, 64, signed=False)
    err_u = (mk.dequant(qw2, sc2, 64).float() - w.float()).abs().max().item()
    assert 0.3 < frac < 0.7
    print(f"  signed scales: every group's extreme maps to q = 0; {frac:.2f} of scales negative; max |err| signed "
          f"{err_s:.4f}, unsigned (rtn.h) {err_u:.4f}")


# ------------------------------------------------------------------------------------------------
# the ops

def _g(seed):
    return torch.Generator().manual_seed(seed)


def test_rmsnorm_gated_formula():
    g = _g(11)
    x = torch.randn(5, 2048, generator=g).to(BF)
    z = torch.randn(5, 2048, generator=g).to(BF)
    w = (1 + 0.1 * torch.randn(2048, generator=g)).to(BF)
    got = ops.rmsnorm_gated(x, z, w, 1e-5, 1024)
    # NVIDIA's documented formula (mamba_ssm layernorm_gated.rms_norm_ref, norm_before_gate False), in fp32
    v = x.float() * F.silu(z.float())
    vg = v.reshape(5, 2, 1024)
    want = (vg * (1 / torch.sqrt(vg.square().mean(-1, keepdim=True) + 1e-5))).reshape(5, 2048) * w.float()
    rel = ((got.float() - want).abs() / want.abs().clamp_min(1e-6)).max().item()
    ulp = (got.float() - want.to(BF).float()).abs().max().item()
    assert rel < 1e-2 and int((got != want.to(BF)).sum()) <= 0.01 * got.numel(), (rel, ulp)
    before = ops.rmsnorm_gated(x, z, w, 1e-5, 1024, norm_before_gate=True)
    whole = ops.rmsnorm_gated(x, z, w, 1e-5, 2048)
    assert not torch.equal(got, before) and not torch.equal(got, whole)
    print(f"  rmsnorm_gated = rms_norm_ref (fp32, rounded once): {int((got != want.to(BF)).sum())} of {got.numel()} "
          f"elements an ulp apart; norm_before_gate True and one group of the whole width both differ")


def test_conv_mutations():
    g = _g(12)
    C, K, T = 64, 4, 9
    x = torch.randn(1, C, T, generator=g).to(BF)
    w = (0.5 * torch.randn(C, K, generator=g)).to(BF)
    b = (0.2 * torch.randn(C, generator=g)).to(BF)
    seq = ops.conv_seq(x, w, b)
    ring = torch.zeros(1, C, K - 1)
    cols = []
    for t in range(T):
        cols.append(ops.conv_update(x[:, :, t], ring, w, b))
        ring = torch.cat([ring[..., 1:], x[:, :, t, None].float()], -1)
    assert torch.equal(torch.stack(cols, -1), seq), "conv_update column by column = conv_seq, bitwise"
    nob = ops.conv_seq(x, w, None)
    full = torch.cat([torch.zeros(1, C, K - 1), x.float()], -1)
    acc = torch.zeros(1, C, T)
    for j in range(K):
        acc = acc + w.float()[None, :, j, None] * full[:, :, j:j + T]
    silu_first = (ops.silu_conv(acc) + b.float()[None, :, None]).to(BF)
    swapped = ops.conv_seq(x, w.flip(-1), b)
    assert not torch.equal(seq, nob) and not torch.equal(seq, silu_first) and not torch.equal(seq, swapped)
    ref = F.silu(F.conv1d(full, w.float()[:, None, :], b.float(), groups=C)).to(BF)
    print(f"  conv: update = seq bitwise; dropping the bias, SiLU before the bias, newest-first taps all differ; "
          f"vs torch's fp32 conv1d + SiLU: {int((seq != ref).sum())} of {seq.numel()} elements an ulp apart")


def test_ssm_update_clamp_and_chain():
    g = _g(13)
    H, P, N, G = 8, 32, 16, 2
    S = torch.randn(1, H, P, N, generator=g)
    x = torch.randn(1, H, P, generator=g).to(BF)
    dt_raw = torch.randn(1, H, generator=g).to(BF)
    A = ops.A_from_log((torch.rand(H, generator=g) * 9.1 - 4.7).to(BF))
    B = torch.randn(1, G, N, generator=g).to(BF)
    Cm = torch.randn(1, G, N, generator=g).to(BF)
    D = torch.randn(H, generator=g).to(BF)
    dt_bias = torch.full((H,), -9.0).to(BF)                              # softplus ~ 1e-4: below the clamp
    y0, S0 = ops.ssm_update(S, x, dt_raw, A, B, Cm, D, dt_bias)
    y1, S1 = ops.ssm_update(S, x, dt_raw, A, B, Cm, D, dt_bias, dt_clamp=1e-3)
    assert not torch.equal(S0, S1) and not torch.equal(y0, y1)
    # fp64 evaluation of the same recurrence
    dt = F.softplus(dt_raw.double() + dt_bias.double())
    Ad = A.double()
    Bh = B.double().repeat_interleave(H // G, 1)
    Ch = Cm.double().repeat_interleave(H // G, 1)
    S64 = S.double() * torch.exp(Ad * dt)[..., None, None] + (Bh * dt[..., None])[:, :, None, :] * x.double()[..., None]
    y64 = (S64 * Ch[:, :, None, :]).sum(-1) + x.double() * D.double()[None, :, None]
    rs = ((S0.double() - S64).abs().max() / S64.abs().max()).item()
    no_d = ops.ssm_update(S, x, dt_raw, A, B, Cm, None, dt_bias)[0]
    swapped_groups = ops.ssm_update(S, x, dt_raw, A, B.flip(1), Cm, D, dt_bias)[0]
    assert rs < 1e-6 and not torch.equal(no_d, y0) and not torch.equal(swapped_groups, y0)
    print(f"  ssm_update: the clamp (test switch) changes state and output at dt_bias -9; the state within {rs:.1e} "
          f"(relative) of fp64, y max |diff| vs fp64 {(y0.double() - y64).abs().max().item():.2e} (one bf16 rounding); "
          f"dropping D and swapping the groups both differ")


def test_stub_signatures():
    """The stub entries with R's argument shapes (R:403-436, R:496-525) equal the plain ops."""
    g = _g(14)
    Bt, H, P, N, G, K = 1, 8, 32, 16, 2, 4
    Cd = H * P + 2 * G * N
    st = torch.randn(Bt, H, P, N, generator=g)
    x = torch.randn(Bt, H, P, generator=g).to(BF)
    dt = torch.randn(Bt, H, generator=g).to(BF)
    A_log = torch.rand(H, generator=g).to(BF)
    D = torch.randn(H, generator=g).to(BF)
    dt_bias = torch.randn(H, generator=g).to(BF)
    B = torch.randn(Bt, G, N, generator=g).to(BF)
    Cm = torch.randn(Bt, G, N, generator=g).to(BF)
    A = -torch.exp(A_log.float())
    s1 = st.clone()
    y = ops._selective_state_update(s1, x, dt[:, :, None].expand(-1, -1, P), A[:, None, None].expand(-1, P, N), B, Cm,
                                    D[:, None].expand(-1, P), z=None, dt_bias=dt_bias[:, None].expand(-1, P),
                                    dt_softplus=True)
    y2, S2 = ops.ssm_update(st, x, dt, A, B, Cm, D, dt_bias)
    assert torch.equal(y, y2) and torch.equal(s1, S2)
    cs = torch.randn(Bt, Cd, K, generator=g)
    xc = torch.randn(Bt, Cd, generator=g).to(BF)
    w = torch.randn(Cd, K, generator=g).to(BF)
    b = torch.randn(Cd, generator=g).to(BF)
    c1 = cs.clone()
    yc = ops._causal_conv1d_update(xc, c1, w, b, "silu")
    assert torch.equal(yc, ops.conv_update(xc, cs[..., 1:], w, b))
    assert torch.equal(c1, torch.cat([cs[..., 1:], xc[..., None].float()], -1))
    T = 7
    xs = torch.randn(Bt, Cd, T, generator=g).to(BF)
    assert torch.equal(ops._causal_conv1d_fn(xs, w, b, activation="silu"), ops.conv_seq(xs, w, b))
    xx = torch.randn(Bt, T, H, P, generator=g).to(BF)
    dd = torch.randn(Bt, T, H, generator=g).to(BF)
    BB = torch.randn(Bt, T, G, N, generator=g).to(BF)
    CC = torch.randn(Bt, T, G, N, generator=g).to(BF)
    out, fin = ops._mamba_chunk_scan_combined(xx, dd, A, BB, CC, chunk_size=128, D=D, dt_bias=dt_bias, dt_softplus=True,
                                              dt_limit=[0.0, math.inf], return_final_states=True)
    o2, f2 = ops.ssm_chunk(xx[0], dd[0], A, BB[0], CC[0], D, dt_bias)
    assert torch.equal(out[0], o2) and torch.equal(fin[0], f2)
    print("  the stub's selective_state_update / causal_conv1d_update / causal_conv1d_fn / mamba_chunk_scan_combined "
          "with R's argument shapes = the plain ops, bitwise (state written in place)")


def test_chunk_scan_partial_and_finite():
    """783380af39's edge: 181 = 128 + 53 rows (a partial last sub-chunk), ngroups 8, dstate 128, headdim 64 - finite,
    and within a small bound of the recurrence (each row through ssm_update)."""
    g = _g(15)
    T, H, P, N, G = 181, 16, 64, 128, 8
    x = torch.randn(T, H, P, generator=g).to(BF)
    dt = torch.randn(T, H, generator=g).to(BF)
    A = ops.A_from_log((torch.rand(H, generator=g) * 9.1 - 4.7).to(BF))
    B = torch.randn(T, G, N, generator=g).to(BF)
    Cm = torch.randn(T, G, N, generator=g).to(BF)
    D = torch.randn(H, generator=g).to(BF)
    dt_bias = (torch.rand(H, generator=g) * 19 - 10.5).to(BF)
    y, S = ops.ssm_chunk(x, dt, A, B, Cm, D, dt_bias, out_dtype=F32)
    assert torch.isfinite(y).all() and torch.isfinite(S).all()
    Sr = torch.zeros(1, H, P, N)
    yr = []
    for t in range(T):
        yt, Sr = ops.ssm_update(Sr, x[t:t + 1], dt[t:t + 1], A, B[t:t + 1], Cm[t:t + 1], D, dt_bias, out_dtype=F32)
        yr.append(yt[0])
    yr = torch.stack(yr)
    ry = ((y - yr).abs().max() / yr.abs().max()).item()
    rS = ((S - Sr[0]).abs().max() / Sr.abs().max()).item()
    assert ry < 1e-4 and rS < 1e-4, (ry, rS)
    y2, S2 = ops.ssm_chunk(x, dt, A, B, Cm, D, dt_bias, out_dtype=F32)
    assert torch.equal(y, y2) and torch.equal(S, S2)
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        y1, S1 = ops.ssm_chunk(x, dt, A, B, Cm, D, dt_bias, out_dtype=F32)
    finally:
        torch.set_num_threads(n)
    same_threads = torch.equal(y, y1) and torch.equal(S, S1)
    print(f"  ssm_chunk at 181 rows (128 + 53): finite; vs the recurrence (fp32 out) max relative |diff| y {ry:.2e}, "
          f"state {rS:.2e}; repeat bitwise; 1 thread vs {n}: {'bitwise' if same_threads else 'NOT bitwise'}")


if __name__ == "__main__":
    want = sys.argv[1:]
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and (not want or n in want)]
    skipped = 0
    for n, f in tests:
        try:
            f()
            print(f"ok  {n}", flush=True)
        except Skip as e:
            skipped += 1
            print(f"SKIP {n}: {e}", flush=True)
    print(f"{len(tests) - skipped} tests passed, {skipped} skipped")
    sys.exit(77 if skipped and skipped == len(tests) else 0)
