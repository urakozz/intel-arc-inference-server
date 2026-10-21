#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B1 for bolt_ref.py (spec 25a Task 3): the streamed port against Bolt's unmodified remote code (R), un-streamed.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_bolt_ref.py [test_name ...]   (pytest collects it too)

The tiny variants are bolt_make_tiny.py's under $BOLT_TINY_DIR (default <repo>/oracle-out-bolt-tiny), written from the
pinned snapshot in the HF cache; every test SKIPs (the runner exits 77 when all do) without it. Review Focus 1: outside
the Mamba mixer the port is BITWISE R (fp32 and bf16, prompt and cached decode, chunked = whole) - shown by feeding
both the same Mamba outputs; inside it the port is bitwise its restated ops / the twin's order and within a MEASURED
fp32 bound of R's torch fallback (the bound printed and asserted as found); the clamp is exercised only on `dtclamp`.
One test per trap of spec 25 §2 (Review Focus 2-5).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
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
ref = _load("bolt_ref")
ops = _load("bolt_mamba_ops")
mk = _load("bolt_make_tiny")
REPO = os.path.dirname(os.path.dirname(_HERE))
TINY_DIR = os.environ.get("BOLT_TINY_DIR", os.path.join(REPO, "oracle-out-bolt-tiny"))
BF, F32 = torch.bfloat16, torch.float32
# The measured fp32 bounds (max |port - fallback| / max |fallback|), asserted as found (Review Focus 1); see the
# facts sheet's "B1 bounds" for the printed values.
FP32_BOUND = 1e-4


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


def variant(name: str) -> str:
    snapshot()
    d = os.path.join(TINY_DIR, name)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        mk.make(d, name, 0, twin=d + "-twin" if name.startswith("int4") else None)
    return d


def rand_ids(n, seed, hi=131072):
    return torch.randint(0, hi, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


_CACHE: dict = {}


def resident(name, dtype, path):
    key = ("res", name, dtype, path)
    if key not in _CACHE:
        d = variant(name)
        _CACHE[key] = ref.build_resident(d, ref.config(d), dtype, path)
    return _CACHE[key]


def streamed(name, dtype, path="kernel", weights=None, layers=None):
    key = ("str", name, dtype, path, weights, layers)
    if key not in _CACHE:
        d = variant(name)
        w = weights or ("int4" if name.startswith("int4") else "bf16")
        _CACHE[key] = ref.build_streamed(d, ref.config(d, layers), w, dtype, path)
    return _CACHE[key]


def mamba_layers(model):
    return [i for i, b in enumerate(model.model.layers) if b.block_type == "mamba"]


class MixerTape:
    """Forward hooks on every Mamba mixer: record its outputs (mode "rec") or replace them by a recorded tape (mode
    "play", consumed in call order) - the 'one shared source' of Review Focus 1."""

    def __init__(self, model, mode, tape=None, slicer=None):
        self.mode, self.tape = mode, tape if tape is not None else {}
        self.pos = {}
        self.hooks = []
        self.slicer = slicer
        for i in mamba_layers(model):
            self.hooks.append(model.model.layers[i].mixer.register_forward_hook(self._fn(i)))

    def _fn(self, i):
        def fn(_m, _a, out):
            if self.mode == "rec":
                self.tape.setdefault(i, []).append(out.detach().clone())
                return None
            j = self.pos.get(i, 0)
            self.pos[i] = j + 1
            if self.slicer is not None:
                return self.slicer(i, j, out)
            return self.tape[i][j]
        return fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def block_outputs(model):
    """Hooks capturing every block's output (rows concatenated in call order)."""
    got: dict = {}
    hooks = [b.register_forward_hook(lambda _m, _a, out, i=i: got.setdefault(i, []).append(out.detach()[0].clone()))
             for i, b in enumerate(model.model.layers)]
    return got, hooks


@torch.no_grad()
def run_seq(model, ids, gen=8, chunk=0, cache_kind="fp32"):
    """Prompt (whole or in chunks) + `gen` greedy cached steps through model(...); returns (logits rows, tokens)."""
    cache = br.fp32_cache(model, 1, F32 if cache_kind == "fp32" else BF)
    rows = []
    c = chunk if chunk > 0 else len(ids)
    for a in range(0, len(ids), c):
        rows.append(model(input_ids=torch.tensor([ids[a:a + c]]), past_key_values=cache, use_cache=True).logits[0])
    toks = []
    for _ in range(gen):
        nxt = int(rows[-1][-1].float().argmax())
        toks.append(nxt)
        rows.append(model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True).logits[0])
    return torch.cat(rows).float(), toks, cache


def cat_rows(got):
    return {i: torch.cat(v, 0) for i, v in got.items()}


# ------------------------------------------------------------------------------------------------
# Review Focus 1

def test_streamed_equals_resident():
    """The streaming (meta build, per-block materialisation, lazy experts, the dummy skip) changes no bit: the streamed
    port = R's model fully resident on the kernel path, logits and every block, prompt + cached decode."""
    for name, dt in (("pattern12", BF), ("pattern12-fp32", F32)):
        ids = rand_ids(40, 1)
        a, ha = streamed(name, dt), None
        ga, hk = block_outputs(a)
        la, ta, _ = run_seq(a, ids)
        [h.remove() for h in hk]
        b = resident(name, dt, "kernel")
        gb, hk = block_outputs(b)
        lb, tb, _ = run_seq(b, ids)
        [h.remove() for h in hk]
        ga, gb = cat_rows(ga), cat_rows(gb)
        assert ta == tb and torch.equal(la, lb), name
        assert all(torch.equal(ga[i], gb[i]) for i in ga), name
        del ha
        print(f"  {name}: streamed = resident, bitwise (logits, {len(ga)} blocks x {ga[0].shape[0]} rows, greedy {ta})")


def test_moe_dummy_noop():
    """R's moe() (with its dummy forward of unrouted experts, R:941-945) = moe_routed_only, bitwise; some expert is
    unrouted in the test (the branch runs)."""
    m = resident("pattern12", BF, "kernel")
    mix = m.model.layers[1].mixer
    x = torch.randn(1, 3, 256, generator=torch.Generator().manual_seed(2)).to(BF)
    ids, w = mix.gate(x)
    lat = mix.fc1_latent_proj(x.view(-1, 256))
    a = mix.moe(lat, ids, w)
    b = ref.moe_routed_only(mix, lat, ids, w)
    unrouted = len(mix.experts) - len(set(ids.reshape(-1).tolist()))
    assert unrouted > 0 and torch.equal(a, b)
    print(f"  R's moe (dummy forward of {unrouted} unrouted experts) = moe_routed_only, bitwise")


def test_non_mamba_bitwise():
    """With the Mamba mixers' outputs fed from one shared source (the port's, replayed into R), every block output and
    the logits of the unmodified R (its torch fallback dispatch) equal the streamed port's bitwise: prompt (40 and 300
    ids) + 8 cached steps, bf16 and fp32."""
    for name, dt in (("pattern12", BF), ("pattern12-fp32", F32)):
        for n in (40, 300):
            ids = rand_ids(n, 3)
            port = streamed(name, dt)
            tape = MixerTape(port, "rec")
            gp, hk = block_outputs(port)
            lp, tp, _ = run_seq(port, ids)
            [h.remove() for h in hk]
            tape.remove()
            r = resident(name, dt, "torch")
            play = MixerTape(r, "play", tape.tape)
            gr, hk = block_outputs(r)
            lr, tr, _ = run_seq(r, ids)
            [h.remove() for h in hk]
            play.remove()
            gp, gr = cat_rows(gp), cat_rows(gr)
            bad = [i for i in gp if not torch.equal(gp[i], gr[i])]
            assert tp == tr and torch.equal(lp, lr) and not bad, (name, n, bad)
            print(f"  {name} {n} ids + 8 steps: R (Mamba outputs replayed from the port) = the port, bitwise: logits and "
                  f"{len(gp)} blocks")


def dt_min(model, ids, gen=8, chunk=0):
    """The smallest softplus(dt_raw + dt_bias) any Mamba head sees over a run (R's fallback clamps below 0.001)."""
    low = [math.inf]
    hooks = []
    for i in mamba_layers(model):
        mix = model.model.layers[i].mixer

        def fn(_m, _a, out, mix=mix):
            d = out[..., -mix.num_heads:].float() + mix.dt_bias.float()
            low[0] = min(low[0], F.softplus(d).min().item())
        hooks.append(mix.in_proj.register_forward_hook(fn))
    run_seq(model, ids, gen, chunk)
    for h in hooks:
        h.remove()
    return low[0]


class FixedFallback:
    """R's fallback mixers replaced, for a run, by the restated fallback with the state-passing fix (fix_state_passing:
    the source-chunk sum of N:328-330) - every rounding point kept."""

    def __init__(self, model):
        import types
        self.mixers = [model.model.layers[i].mixer for i in mamba_layers(model)]
        for mx in self.mixers:
            mx.torch_forward = types.MethodType(
                lambda self, h, c=None, mask=None: ref.fallback_forward(self, h, c, fix_state_passing=True), mx)

    def remove(self):
        for mx in self.mixers:
            del mx.torch_forward


def port_on_fallback_stream(name, dtype, ids, gen=8, cache_kind="fp32", fix=False):
    """R's torch fallback drives the stream (its Mamba outputs feed the blocks); the port's mixers run on the SAME
    inputs (the fallback's outputs replayed into the port's stream while the port's own outputs are captured).
    Returns ({layer: fallback outputs}, {layer: port outputs}, fallback states, port states)."""
    fb = resident(name, dtype, "torch")
    fixed = FixedFallback(fb) if fix else None
    tape = MixerTape(fb, "rec")
    _, toks, cf = run_seq(fb, ids, gen, cache_kind=cache_kind)
    tape.remove()
    if fixed is not None:
        fixed.remove()
    port = resident(name, dtype, "kernel")
    own = {}

    def slicer(i, j, out):
        own.setdefault(i, []).append(out.detach()[0].clone())
        return tape.tape[i][j]
    play = MixerTape(port, "play", slicer=slicer)
    _, toks_p, cp = run_seq(port, ids, gen, cache_kind=cache_kind)
    play.remove()
    assert toks == toks_p
    out_f = {i: torch.cat([t[0] for t in v]) for i, v in tape.tape.items()}
    out_p = {i: torch.cat(v) for i, v in own.items()}
    st_f = {i: cf.ssm_states[i][0].float().clone() for i in mamba_layers(fb)}
    st_p = {i: cp.ssm_states[i][0].float().clone() for i in mamba_layers(port)}
    return out_f, out_p, st_f, st_p


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _rel_by_layer(out_f, out_p, st_f, st_p):
    line, worst = [], 0.0
    for i in out_f:
        ry, rs = rel(out_p[i], out_f[i]), rel(st_p[i], st_f[i])
        worst = max(worst, ry, rs)
        line.append(f"L{i} y {ry:.1e} S {rs:.1e}")
    return worst, "; ".join(line)


def test_mamba_vs_fallback_fp32():
    """pattern12-fp32 (every head stays above the fallback's clamp - measured): the port's Mamba mixers against R's
    torch fallback on the same inputs, a 40-id and a 400-id prompt + 8 cached steps; the max relative difference of
    every output and state printed. 40 ids (one SSD chunk): within the bound as found. 400 ids: R's fallback passes the
    SSD state between chunks wrongly (R:692-697 sums over the target chunk; transformers' native class sums over the
    source, N:328-330) and is far off; the restated fallback with only that index fixed is within the bound."""
    worst = 0.0
    for n in (40, 400):
        ids = rand_ids(n, 4)
        low = dt_min(resident("pattern12-fp32", F32, "torch"), ids)
        assert low > 1e-3, f"the clamp would engage (min dt {low})"
        w_r, line_r = _rel_by_layer(*port_on_fallback_stream("pattern12-fp32", F32, ids))
        print(f"  {n} ids + 8 steps (fp32): port vs R's fallback, max relative |diff|: {line_r} (min dt {low:.3g} > the "
              f"0.001 clamp)")
        if n <= 128:
            worst = max(worst, w_r)
            continue
        assert w_r > 100 * FP32_BOUND, w_r
        w_f, line_f = _rel_by_layer(*port_on_fallback_stream("pattern12-fp32", F32, ids, fix=True))
        print(f"  {n} ids + 8 steps (fp32): port vs R's fallback with the state-passing index fixed: {line_f}")
        worst = max(worst, w_f)
    assert worst < FP32_BOUND, worst
    print(f"  the fp32 bound as found: {worst:.2e} (asserted < {FP32_BOUND:g})")


def test_fallback_state_passing_bug():
    """The restated fallback (every point kept, no fix) is R's torch_forward bitwise on a 300-row prefill (3 chunks) -
    so the multi-chunk error is R's own; the port's state agrees with the recurrence (300 ssm_update rows) and R's
    fallback state does not."""
    m = resident("pattern12-fp32", F32, "torch")
    mix = m.model.layers[2].mixer
    x = torch.randn(1, 300, 256, generator=torch.Generator().manual_seed(26)).to(F32)

    def prefill(fn):
        c = br.fp32_cache(m)
        with torch.no_grad():
            y = fn(x, c)
        return y, c.ssm_states[2].clone()
    yr, sr = prefill(lambda h, c: mix.torch_forward(h, c))
    ya, sa = prefill(lambda h, c: ref.fallback_forward(mix, h, c))
    assert torch.equal(yr, ya) and torch.equal(sr, sa)
    yf, sf = prefill(lambda h, c: ref.fallback_forward(mix, h, c, fix_state_passing=True))
    yp, st = ref.mamba_seq(x, mix)
    c = br.fp32_cache(m)
    c.has_previous_state = True
    mix._bolt_path = "kernel"
    with torch.no_grad():
        for t in range(300):
            mix(x[:, t:t + 1], cache_params=c)
    mix._bolt_path = "torch"
    srec = c.ssm_states[2]
    print(f"  300-row prefill: restated fallback = R bitwise; state vs the recurrence: R {rel(sr, srec):.2e}, R with the "
          f"index fixed {rel(sf, srec):.2e}, the port {rel(st['ssm'], srec):.2e}")
    assert rel(sr, srec) > 1e-2 and rel(sf, srec) < FP32_BOUND and rel(st["ssm"], srec) < FP32_BOUND


def test_mamba_restated_bitwise():
    """mamba_seq / mamba_step (the restated kernel path) = R's cuda_kernels_forward via the stub, bitwise: a prompt,
    a continuation chunk and three decode rows, bf16 and fp32."""
    for name, dt in (("pattern12", BF), ("pattern12-fp32", F32)):
        m = resident(name, dt, "kernel")
        mix = m.model.layers[2].mixer
        cache = br.fp32_cache(m)
        x = torch.randn(1, 37, 256, generator=torch.Generator().manual_seed(6)).to(dt)
        outs_r, outs_p = [], []
        st = None
        with torch.no_grad():
            for part in (x[:, :30], x[:, 30:34]):
                outs_r.append(mix(part, cache_params=cache))
                cache.has_previous_state = True
                y, st = ref.mamba_seq(part, mix, st)
                outs_p.append(y)
            for t in range(34, 37):
                outs_r.append(mix(x[:, t:t + 1], cache_params=cache))
                y, st = ref.mamba_step(x[:, t:t + 1], mix, st)
                outs_p.append(y)
        assert all(torch.equal(a, b) for a, b in zip(outs_r, outs_p)), name
        assert torch.equal(st["ssm"], cache.ssm_states[2]) and torch.equal(st["conv"], cache.conv_states[2])
        print(f"  {name}: mamba_seq (prompt 30 + continuation 4) and mamba_step (3 rows) = R's kernel path, bitwise "
              f"(outputs, SSM and conv states)")


def _scalar_twin_update(S, x, dt, dA, B, C, D, rep):
    """The decode update in the twin's order, one scalar op at a time (0-d fp32): dB[n] = B[g][n] * dt;
    S = S * dA + dB * x; y = halving tree over n of S * C, then + x * D. Transcendentals (dt, dA) are given."""
    H, P, N = S.shape
    Sn = S.clone()
    y = torch.empty(H, P)
    for h in range(H):
        g = h // rep
        for p in range(P):
            v = []
            for n in range(N):
                dB = B[g, n] * dt[h]
                s = Sn[h, p, n] * dA[h] + dB * x[h, p]
                Sn[h, p, n] = s
                v.append(s * C[g, n])
            while len(v) > 1:
                half = len(v) // 2
                v = [v[i] + v[i + half] for i in range(half)]
            y[h, p] = v[0] + x[h, p] * D[h]
    return y, Sn


def test_mamba_twin_order():
    """bolt_mamba_ops' decode functions against the twin's order restated one scalar op at a time (bitwise) and
    against fp64 (bounded): ssm_update, conv_update, rmsnorm_gated."""
    g = torch.Generator().manual_seed(7)
    H, P, N, G = 4, 3, 8, 2
    S = torch.randn(H, P, N, generator=g)
    x = torch.randn(1, H, P, generator=g).to(BF)
    dt_raw = torch.randn(1, H, generator=g).to(BF)
    A = ops.A_from_log(torch.rand(H, generator=g).to(BF))
    B = torch.randn(1, G, N, generator=g).to(BF)
    C = torch.randn(1, G, N, generator=g).to(BF)
    D = torch.randn(H, generator=g).to(BF)
    dt_bias = torch.randn(H, generator=g).to(BF)
    y, Sn = ops.ssm_update(S[None], x, dt_raw, A, B, C, D, dt_bias, out_dtype=F32)
    dt = ops.dt_of(dt_raw, dt_bias)[0]
    dA = torch.exp(A * dt)
    yt, St = _scalar_twin_update(S, x[0].float(), dt, dA, B[0].float(), C[0].float(), D.float(), H // G)
    assert torch.equal(y[0], yt) and torch.equal(Sn[0], St)
    S64 = S.double() * torch.exp(A.double() * F.softplus(dt_raw.double()[0] + dt_bias.double()))[:, None, None] + \
        (B[0].double().repeat_interleave(2, 0) * dt.double()[:, None])[:, None, :] * x[0].double()[..., None]
    print(f"  ssm_update = the scalar twin bitwise (state and y); state vs fp64 (fp64 softplus/exp) relative "
          f"{rel(Sn[0], S64):.1e}")
    Cc, K = 16, 4
    ring = torch.randn(1, Cc, K - 1, generator=g).to(BF).float()
    xn = torch.randn(1, Cc, generator=g).to(BF)
    w = torch.randn(Cc, K, generator=g).to(BF)
    b = torch.randn(Cc, generator=g).to(BF)
    got = ops.conv_update(xn, ring, w, b, out_dtype=F32)
    want = torch.empty(Cc)
    for c in range(Cc):
        acc = b[c].float()
        for j in range(K - 1):
            acc = acc + w[c, j].float() * ring[0, c, j]
        acc = acc + w[c, K - 1].float() * xn[0, c].float()
        want[c] = acc
    assert torch.equal(got[0], ops.silu_conv(want))
    print("  conv_update = bias first, then the taps oldest first, one scalar add at a time - bitwise")
    v = torch.randn(1, 8, generator=g).to(BF)
    z = torch.randn(1, 8, generator=g).to(BF)
    wn = torch.randn(8, generator=g).to(BF)
    got = ops.rmsnorm_gated(v, z, wn, 1e-5, 4)
    vv = v.float() * ops.silu_gate(z.float())
    want = torch.empty(8)
    for g0 in (0, 4):
        sq = [vv[0, g0 + i] * vv[0, g0 + i] for i in range(4)]
        ms = ((sq[0] + sq[2]) + (sq[1] + sq[3])) / 4
        r = 1.0 / torch.sqrt(ms + 1e-5)
        for i in range(4):
            want[g0 + i] = (vv[0, g0 + i] * r) * wn[g0 + i].float()
    assert torch.equal(got[0], want.to(BF))
    print("  rmsnorm_gated = the halving tree over each group, 1 / sqrt(ms + eps), (v r) w - bitwise")


class RouterTape:
    """Forward hooks on every router: record its (ids, w) per call, or replay the whole forward's rows (sliced by the
    rows each call covers) while capturing the router's own output - torch's fp32 GEMM is row-count dependent."""

    def __init__(self, model, mode, whole=None):
        self.mode, self.whole, self.rec, self.own, self.off = mode, whole, {}, {}, {}
        self.hooks = [b.mixer.gate.register_forward_hook(self._fn(i)) for i, b in enumerate(model.model.layers)
                      if b.block_type == "moe"]

    def _fn(self, i):
        def fn(_m, _a, out):
            ids, w = out
            if self.mode == "rec":
                self.rec.setdefault(i, []).append((ids.clone(), w.clone()))
                return None
            a = self.off.get(i, 0)
            self.off[i] = a + ids.shape[0]
            self.own.setdefault(i, []).append((ids.clone(), w.clone()))
            wi, ww = self.whole[i]
            return wi[a:a + ids.shape[0]], ww[a:a + ids.shape[0]]
        return fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def test_chunked_equals_whole():
    """A 300-id prompt fed in chunks of 1, 2, 3, 64, 127, 128, 129 and 255 through the cache against one forward.
    bf16: with the whole forward's Mamba outputs AND router outputs replayed (torch's fp32 GEMM - the router's logits,
    R:1044 - is row-count dependent on this CPU while its bf16 GEMM is not: the facts sheet's measurement), every block
    output and logit is bitwise the whole forward's (attention KV and masks across chunk edges, the experts on other
    row subsets, the combine, norms); the conv states bitwise; the router's own (ids, w) differences counted. fp32:
    every linear is an fp32 GEMM, so everything is held to the bound. The chunked Mamba mixers' own outputs and the SSM
    states are within the chunk scan's bound."""
    ids = rand_ids(300, 8)
    worst = 0.0
    for name, dt in (("pattern12", BF), ("pattern12-fp32", F32)):
        m = resident(name, dt, "kernel")
        tape = MixerTape(m, "rec")
        rt = RouterTape(m, "rec")
        gw, hk = block_outputs(m)
        lw, _, cw = run_seq(m, ids, gen=0)
        [h.remove() for h in hk]
        tape.remove()
        rt.remove()
        whole = {i: v[0] for i, v in tape.tape.items()}
        rwhole = {i: v[0] for i, v in rt.rec.items()}
        gw = cat_rows(gw)
        line = []
        for c in (1, 2, 3, 64, 127, 128, 129, 255):
            own, off = {}, {}

            def slicer(i, j, out, own=own, off=off):
                a = off.get(i, 0)
                off[i] = a + out.shape[1]
                own.setdefault(i, []).append(out.detach().clone())
                return whole[i][:, a:a + out.shape[1]]
            play = MixerTape(m, "play", slicer=slicer)
            rplay = RouterTape(m, "play", rwhole) if dt == BF else None
            gc, hk = block_outputs(m)
            lc, _, cc = run_seq(m, ids, gen=0, chunk=c)
            [h.remove() for h in hk]
            play.remove()
            gc = cat_rows(gc)
            r = 0.0
            if dt == BF:
                rplay.remove()
                assert torch.equal(lc, lw) and all(torch.equal(gc[i], gw[i]) for i in gw), (name, c)
                flips = wdiff = 0
                for i, parts in rplay.own.items():
                    ci = torch.cat([p[0] for p in parts]).sort(-1).values
                    cw_ = torch.cat([p[1] for p in parts])
                    wi = rwhole[i][0].sort(-1).values
                    flips += int((ci != wi).any(-1).sum())
                    wdiff += int((cw_ != rwhole[i][1]).any(-1).sum())
                for i in whole:
                    assert torch.equal(cc.conv_states[i], cw.conv_states[i]), (name, c, i)
            else:
                r = max(rel(lc, lw), max(rel(gc[i], gw[i]) for i in gw))
            for i in whole:
                mine = torch.cat(own[i], 1)
                r = max(r, rel(mine, whole[i]), rel(cc.ssm_states[i], cw.ssm_states[i]))
            if dt == F32:
                worst = max(worst, r)
                line.append(f"{c}: {r:.1e}")
            else:
                line.append(f"{c}: Mamba {r:.1e}, router rows (set / w) differing {flips} / {wdiff}")
        what = ("bitwise outside the Mamba mixers with the routers replayed" if dt == BF else
                "every block and logit within the bound")
        print(f"  {name}: chunked vs whole - {what}; by chunk size (max relative |diff| of the Mamba outputs / states"
              f"{'' if dt == BF else ' and every block'}): {'; '.join(line)}")
    assert worst < FP32_BOUND, worst
    print(f"  the fp32 bound as found: {worst:.2e} (asserted < {FP32_BOUND:g})")


def test_bf16_differences_recorded():
    """bf16 (pattern12, an fp32 cache and a bf16 cache): the restated fallback with every rounding point kept is R's
    torch_forward bitwise; the port and R's fallback differ; with every point switched to the kernel path's semantics
    the difference drops to the order-level residue; each point switched off alone is printed (the bf16 table)."""
    m = resident("pattern12", BF, "torch")
    mix = m.model.layers[2].mixer
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 40, 256, generator=g).to(BF)
    steps = torch.randn(1, 6, 256, generator=g).to(BF)
    allp = frozenset(ref.FALLBACK_POINTS)

    def run(fn, cache_kind):
        cache = br.fp32_cache(m, 1, F32 if cache_kind == "fp32" else BF)
        outs = []
        with torch.no_grad():
            outs.append(fn(x, cache))
            cache.has_previous_state = True
            for t in range(steps.shape[1]):
                outs.append(fn(steps[:, t:t + 1], cache))
        return torch.cat([o.reshape(-1, 256) for o in outs]), cache.ssm_states[2].float().clone()
    for ck in ("fp32", "bf16"):
        r_out, r_st = run(lambda h, c: mix.torch_forward(h, c), ck)
        a_out, a_st = run(lambda h, c: ref.fallback_forward(mix, h, c, allp), ck)
        assert torch.equal(r_out, a_out) and torch.equal(r_st, a_st), ck
        mix._bolt_path = "kernel"
        p_out, p_st = run(lambda h, c: mix(h, cache_params=c), ck)
        mix._bolt_path = "torch"
        n_out, n_st = run(lambda h, c: ref.fallback_forward(mix, h, c, frozenset()), ck)
        full = (rel(r_out, p_out), rel(r_st, p_st))
        none = (rel(n_out, p_out), rel(n_st, p_st))
        assert full[0] > none[0], (ck, full, none)
        rows = []
        for pt in ref.FALLBACK_POINTS:
            o, s_ = run(lambda h, c, pt=pt: ref.fallback_forward(mix, h, c, frozenset({pt})), ck)
            rows.append(f"{pt} {rel(o, p_out):.2e}/{rel(s_, p_st):.2e}")
        print(f"  cache {ck}: restated fallback (all points) = R's torch_forward bitwise; port vs R: out {full[0]:.2e} "
              f"state {full[1]:.2e}; vs the fallback with no point kept: out {none[0]:.2e} state {none[1]:.2e}")
        print("    each point ALONE (the others at the kernel path's semantics), out / state vs the port: "
              + "; ".join(rows))


def test_dt_clamp():
    """dtclamp (half the heads at dt_bias < -6.9), fp32: the port (no clamp) and R's fallback (clamp) differ far beyond
    the bound; with the ops' test switch DT_CLAMP_TEST = 0.001 they agree within it."""
    ids = rand_ids(60, 10)
    low = dt_min(resident("dtclamp", F32, "torch"), ids)
    assert low < 1e-3
    out_f, out_p, st_f, st_p = port_on_fallback_stream("dtclamp", F32, ids)
    far = max(max(rel(out_p[i], out_f[i]), rel(st_p[i], st_f[i])) for i in out_f)
    ops.DT_CLAMP_TEST = 1e-3
    try:
        out_f2, out_p2, st_f2, st_p2 = port_on_fallback_stream("dtclamp", F32, ids)
    finally:
        ops.DT_CLAMP_TEST = None
    near = max(max(rel(out_p2[i], out_f2[i]), rel(st_p2[i], st_f2[i])) for i in out_f2)
    assert far > 100 * FP32_BOUND and near < FP32_BOUND, (far, near)
    print(f"  dtclamp (min dt {low:.2e}): port vs R's fallback {far:.2e} relative; with the clamp switched on in the "
          f"ops {near:.2e} (< {FP32_BOUND:g})")


def test_mamba_chain():
    """Review Focus 2: each mutation of the Mamba chain changes the mixer's output (prompt and step); the unmutated
    restatement is R's kernel path bitwise (test_mamba_restated_bitwise)."""
    m = resident("dtclamp", F32, "kernel")
    mix = m.model.layers[0].mixer
    x = torch.randn(1, 20, 256, generator=torch.Generator().manual_seed(11)).to(F32)
    base, st = ref.mamba_seq(x[:, :19], mix)
    step, _ = ref.mamba_step(x[:, 19:], mix, st)
    for mut in ("swap_zx", "no_conv_bias", "gate_before_norm", "group_whole", "clamp"):
        y, st2 = ref.mamba_seq(x[:, :19], mix, mutate=frozenset({mut}))
        s2, _ = ref.mamba_step(x[:, 19:], mix, st2, mutate=frozenset({mut}))
        assert not torch.equal(y, base) and not torch.equal(s2, step), mut
    print("  swap z / xBC, drop the conv bias, gate before the norm, one norm group of d_inner, the dt clamp: each "
          "changes the prompt and the step output")


def test_router_chain():
    """Review Focus 3: route() = NemotronHTopkRouter bitwise (ids in its order, w); the bias reorders (an expert with a
    lower sigmoid selected on > 0 rows); the + 1e-20 is live on tiny scores; w is fp32 and never rounded."""
    m = resident("pattern12", BF, "kernel")
    gate = m.model.layers[3].mixer.gate
    x = torch.randn(256, 256, generator=torch.Generator().manual_seed(12)).to(BF)
    ids_r, w_r = gate(x[None])
    ids, w, ids_a, w_a, sel10, gap, tot = ref.route(x, gate.weight, gate.e_score_correction_bias)
    assert torch.equal(ids, ids_r) and torch.equal(w, w_r) and w.dtype == F32
    s = F.linear(x.float(), gate.weight.float()).sigmoid()
    by_s = s.topk(9).indices.sort(-1).values
    reorder = int((by_s != ids_a).any(-1).sum())
    assert reorder > 0
    tiny = torch.full((1, gate.weight.shape[0]), -70.0)          # sigmoid ~ 4e-31: sum ~ 3.6e-30 << 1e-20
    s_t = tiny.sigmoid()
    sel_t = s_t + gate.e_score_correction_bias
    id_t = torch.topk(sel_t, 9, sorted=False)[1]
    w_t = s_t.gather(1, id_t)
    with_eps = w_t / (w_t.sum(-1, keepdim=True) + 1e-20) * 5.0
    no_eps = w_t / w_t.sum(-1, keepdim=True) * 5.0
    assert not torch.equal(with_eps, no_eps)
    print(f"  route = R's router bitwise on 256 rows (fp32 w); the bias changes the top-9 set on {reorder} / 256 rows; "
          f"at sigmoid ~4e-31 the + 1e-20 makes w {with_eps.sum().item():.2e} instead of {no_eps.sum().item():.2f}")


def test_route_ties():
    """`ties`: the planted pairs are 9th / 10th with an EXACT tie (gap 0) on every row; which of the pair torch's
    topk(sorted=False) keeps is recorded (the engine's rule is the lower id; exact ties are undetermined)."""
    m = resident("ties", BF, "kernel")
    x = torch.randn(128, 256, generator=torch.Generator().manual_seed(13)).to(BF)
    for layer, (a, c) in mk.TIES.items():
        gate = m.model.layers[layer].mixer.gate
        ids, w, ids_a, w_a, sel10, gap, tot = ref.route(x, gate.weight, gate.e_score_correction_bias)
        assert bool((gap == 0).all())
        low = int((ids_a == min(a, c)).any(-1).sum())
        high = int((ids_a == max(a, c)).any(-1).sum())
        assert low + high == 128
        print(f"  layer {layer} pair ({a}, {c}): gap 0 on 128 / 128 rows; torch keeps the lower id on {low}, the higher "
              f"on {high}")


def test_combine_order():
    """R's combine (ascending expert id, fp32 product then add, one rounding) = combine(order="ascending") bitwise;
    the rank order differs on > 0 rows."""
    m = resident("pattern12", BF, "kernel")
    mix = m.model.layers[5].mixer
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(14)).to(BF)
    ids, w = mix.gate(x[None])
    lat = mix.fc1_latent_proj(x)
    want = mix.moe(lat, ids, w)
    d = torch.zeros(64, ids.shape[1], lat.shape[-1], dtype=BF)
    for e in sorted(set(ids.reshape(-1).tolist())):
        tok, slot = torch.where(ids == e)
        d[tok, slot] = mix.experts[e](lat[tok])
    assert torch.equal(ref.combine(d, w, ids), want)
    diff = int((ref.combine(d, w, ids, order="rank") != want).any(-1).sum())
    assert diff > 0
    print(f"  combine ascending = R bitwise; rank order differs on {diff} / 64 rows")


def test_relu2_bf16():
    """relu² on bf16 (relu then torch.square, each rounded): R's expert = down(relu2(up(x))); the up projection kept in
    fp32 into the square (one rounding fewer) differs."""
    m = resident("pattern12", BF, "kernel")
    ex = m.model.layers[1].mixer.experts[0]
    x = torch.randn(32, 64, generator=torch.Generator().manual_seed(15)).to(BF)
    assert torch.equal(ex(x), F.linear(ref.relu2(F.linear(x, ex.up_proj.weight)), ex.down_proj.weight))
    u32 = F.linear(x.float(), ex.up_proj.weight.float())
    alt = F.linear(torch.square(F.relu(u32)).to(BF), ex.down_proj.weight)
    assert not torch.equal(ex(x), alt)
    print("  relu2 on bf16 = R's expert bitwise; an fp32 up projection into the square differs")


def test_shared_on_normed_hidden_and_latent():
    """moe_block (route, fc1 latent, experts in the latent, combine, fc2, + the shared expert on the NORMED x) = R's
    NemotronHMoE bitwise; the shared expert on the un-normed residual, and an fp32 combine fed to fc2, both differ."""
    m = resident("pattern12", BF, "kernel")
    blk = m.model.layers[3]
    h = (3 * torch.randn(1, 24, 256, generator=torch.Generator().manual_seed(16))).to(BF)
    xn = blk.norm(h)
    want = blk.mixer(xn)
    assert torch.equal(ref.moe_block(xn, blk.mixer), want)
    sh = blk.mixer.shared_experts
    alt = want - sh(xn) + sh(h)
    assert not torch.equal(alt, want)
    ids, w, *_ = ref.route(xn.view(-1, 256), blk.mixer.gate.weight, blk.mixer.gate.e_score_correction_bias)
    lat = F.linear(xn.view(-1, 256), blk.mixer.fc1_latent_proj.weight)
    d = torch.zeros(24, 9, lat.shape[-1], dtype=BF)
    for e in sorted(set(ids.reshape(-1).tolist())):
        tok, slot = torch.where(ids == e)
        d[tok, slot] = blk.mixer.experts[e](lat[tok])
    t = d.float() * w.unsqueeze(-1)
    o32 = F.linear(t.sum(1), blk.mixer.fc2_latent_proj.weight.float()).to(BF).view(1, 24, 256)
    assert not torch.equal(o32 + sh(xn), want)
    print("  moe_block = R's MoE bitwise; the shared expert on the residual, or the combine unrounded into fc2, differ")


def test_rms_norm():
    m = resident("pattern12", BF, "kernel")
    nrm = m.model.layers[4].norm
    h = (4 * torch.randn(16, 256, generator=torch.Generator().manual_seed(17))).to(BF)
    assert torch.equal(ref.rms_norm(h, nrm.weight, nrm.variance_epsilon), nrm(h))
    xf = h.float()
    two = nrm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + nrm.variance_epsilon)).to(BF)
    assert not torch.equal(two, nrm(h))
    print("  rms_norm = NemotronHRMSNorm bitwise (one rounding); the two-rounding (LFM2-style) variant differs")


def test_nope():
    """Review Focus 5: R's attention has no positional term and no q / k norm; a restated eager attention equals it
    bitwise and the same with a RoPE (theta 1e4) on q, k differs."""
    m = resident("pattern12", BF, "kernel")
    att = m.model.layers[7].mixer
    assert not any("rotary" in n or "norm" in n for n, _ in att.named_modules() if n)
    h = torch.randn(1, 12, 256, generator=torch.Generator().manual_seed(18)).to(BF)
    mask = m.model._update_causal_mask(None, h, torch.arange(12))

    def attn(rope):
        q = att.q_proj(h).view(1, 12, att.num_heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(h).view(1, 12, att.num_key_value_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(h).view(1, 12, att.num_key_value_heads, att.head_dim).transpose(1, 2)
        if rope:
            half = att.head_dim // 2
            inv = 1.0 / (10000 ** (torch.arange(half).float() / half))
            ang = torch.arange(12).float()[:, None] * inv[None]
            cos, sin = torch.cat([ang.cos()] * 2, -1).to(BF), torch.cat([ang.sin()] * 2, -1).to(BF)

            def rot(t):
                return t * cos + torch.cat([-t[..., half:], t[..., :half]], -1) * sin
            q, k = rot(q), rot(k)
        k = k.repeat_interleave(att.num_key_value_groups, 1)
        v = v.repeat_interleave(att.num_key_value_groups, 1)
        w = torch.matmul(q, k.transpose(2, 3)) * att.scaling + mask[:, :, :, :12]
        p = F.softmax(w, dim=-1, dtype=F32).to(BF)
        return att.o_proj(torch.matmul(p, v).transpose(1, 2).reshape(1, 12, -1))
    want = att(hidden_states=h, attention_mask=mask)[0]
    assert torch.equal(attn(False), want) and not torch.equal(attn(True), want)
    print("  R's attention = the restated NoPE eager attention bitwise; adding a RoPE changes it")


def test_untied_head():
    m = streamed("pattern12", BF)
    assert m.lm_head.weight.data_ptr() != m.model.embeddings.weight.data_ptr()
    assert not torch.equal(m.lm_head.weight, m.model.embeddings.weight)
    h = torch.randn(4, 256, generator=torch.Generator().manual_seed(19)).to(BF)
    assert not torch.equal(m.lm_head(h), F.linear(h, m.model.embeddings.weight))
    print("  lm_head is its own tensor (untied): the embedding as the head changes the logits")


def test_tap_convention():
    """Review Focus 4: output_hidden_states[i + 1] is block i's OUTPUT (the tuple holds block INPUTS, R:1481-1482),
    hs[0] the embeddings, the last entry norm_f(...) - checked at blocks 1 and 8; the Recorder's tap.L1 = block 1's
    output, every row."""
    m = resident("pattern12", BF, "kernel")
    ids = torch.tensor([rand_ids(30, 20)])
    got, hk = block_outputs(m)
    with torch.no_grad():
        out = m.model(input_ids=ids, past_key_values=br.fp32_cache(m), use_cache=True, output_hidden_states=True)
    [h.remove() for h in hk]
    hs = out.hidden_states
    assert len(hs) == 13 and torch.equal(hs[0][0], m.model.embeddings(ids)[0])
    for i in (1, 8):
        assert torch.equal(hs[i + 1][0], got[i][0]), i
    assert torch.equal(hs[12][0], m.model.norm_f(got[11][0])) and not torch.equal(hs[12][0], got[11][0])
    s = streamed("pattern12", BF)
    rec = ref.Recorder(s, ref.config(variant("pattern12")), acts=True, taps=True)
    ref.forward_chunks(s, ids[0].tolist(), 0, br.fp32_cache(s), rec)
    rec.remove()
    t = rec.tensors()
    assert rec.taps == [1] and torch.equal(t["tap.L1"], got[1][0]) and torch.equal(t["h.L8"], got[8][0])
    print("  hidden_states[i + 1] = block i's output at i = 1, 8; hs[0] the embeddings, hs[-1] norm_f'd; tap.L1 = "
          "block 1's output")


def test_int4_both_spellings():
    """build_streamed(weights="int4") on int4-g64 and int4-g128-gptq (g128 through the exact g64 expansion) = the port
    on the dequantised twin (the same weights as plain tensors), bitwise: logits and every block, prompt + steps."""
    for name in ("int4-g64", "int4-g128-gptq"):
        ids = rand_ids(40, 21)
        a = streamed(name, BF)
        ga, hk = block_outputs(a)
        la, ta, _ = run_seq(a, ids)
        [h.remove() for h in hk]
        d = variant(name) + "-twin"
        b = ref.build_streamed(d, ref.config(d), "bf16", BF)
        gb, hk = block_outputs(b)
        lb, tb, _ = run_seq(b, ids)
        [h.remove() for h in hk]
        ga, gb = cat_rows(ga), cat_rows(gb)
        assert ta == tb and torch.equal(la, lb) and all(torch.equal(ga[i], gb[i]) for i in ga), name
        print(f"  {name}: the int4 reference = the port on its dequantised twin, bitwise")


def test_layers_truncation():
    d = variant("pattern12")
    cfg = ref.config(d, 8)
    assert len(cfg["block_configs"]) == 8
    ids = rand_ids(30, 22)
    a = ref.build_streamed(d, cfg, "bf16", BF)
    la, ta, _ = run_seq(a, ids)
    full = ref.config(d)
    full["block_configs"] = full["block_configs"][:8]
    b = ref.build_resident(d, full, BF, "kernel")
    lb, tb, _ = run_seq(b, ids)
    assert ta == tb and torch.equal(la, lb)
    try:
        ref.config(d, 7)
        raise AssertionError("layers 7 accepted")
    except SystemExit:
        pass
    print("  config(layers=8) = R with blocks 0..7 bitwise; layers=7 refused (no attention block)")


def test_expected_names():
    """expected_names against the real indexes: SergiioB's (gptq-g128) and NVIDIA's BF16 original (when its small files
    are in the HF cache), the vision / projector names aside."""
    s = snapshot()
    cfg = ref.config(s)
    import json
    wm = json.load(open(os.path.join(s, "model.safetensors.index.json")))["weight_map"]
    lang = {k for k in wm if k.startswith("language_model.")}
    want = set(ref.expected_names(cfg, "gptq-g128"))
    assert lang == want, (sorted(lang - want)[:3], sorted(want - lang)[:3])
    print(f"  gptq-g128: {len(want)} names = SergiioB's index minus its {len(wm) - len(lang)} vision / projector names")
    hub = os.path.dirname(os.path.dirname(os.path.dirname(s)))
    bf = os.path.join(hub, "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16", "snapshots")
    snaps = [os.path.join(bf, x) for x in sorted(os.listdir(bf))] if os.path.isdir(bf) else []
    snaps = [x for x in snaps if os.path.exists(os.path.join(x, "model.safetensors.index.json"))]
    if snaps:
        wm2 = json.load(open(os.path.join(snaps[-1], "model.safetensors.index.json")))["weight_map"]
        lang2 = {k for k in wm2 if k.startswith("language_model.")}
        want2 = set(ref.expected_names(cfg, "bf16"))
        assert lang2 == want2, (sorted(lang2 - want2)[:3], sorted(want2 - lang2)[:3])
        print(f"  bf16: {len(want2)} names = NVIDIA's BF16 index minus its {len(wm2) - len(lang2)} vision / projector "
              f"names (the trunk spelled language_model.backbone.*)")
    ar = ref.expected_names(cfg, "ar-g64")
    assert len(ar) == len(want) - sum(1 for n in want if n.endswith(".g_idx"))


def test_layer_major():
    """layer_major's two-item pass = each item's own model(...) calls, bitwise (prompt chunk then one step)."""
    m = streamed("pattern12", BF)
    a_ids, b_ids = rand_ids(20, 23), rand_ids(33, 24)
    want = []
    for ids in (a_ids, b_ids):
        lg, toks, _ = run_seq(m, ids, gen=1)
        want.append((lg, toks))
    step = ref.layer_major(m)
    sa, sb = {}, {}
    out = step([(a_ids, 0, sa), (b_ids, 0, sb)])
    nxt = [int(out[0][-1].argmax()), int(out[1][-1].argmax())]
    out2 = step([([nxt[0]], 20, sa), ([nxt[1]], 33, sb)])
    for j in range(2):
        got = torch.cat([out[j], out2[j]])
        assert nxt[j] == want[j][1][0] and torch.equal(got, want[j][0]), j
    print("  layer_major (2 items, a prompt and a step) = sequential model calls, bitwise")


def test_gateq_tiny():
    """bolt_gateq.py on the tiny models: kl of int4-g64 against its own dequantised twin is exactly 0 with top-1 1.0
    (the same weights), against pattern12 (the un-quantised original it was packed from) > 0; the verdict reads the
    A4 outputs with the qwen-think reader."""
    import shutil
    import tempfile
    gq = _load("bolt_gateq")
    d = tempfile.mkdtemp(prefix="bolt_gq_")
    sdir, run = os.path.join(d, "set"), os.path.join(d, "run")
    os.makedirs(sdir)
    os.makedirs(run)
    import hashlib
    import json
    man = []
    for i, n in enumerate(("t1_a", "t2_b")):
        raw = (" ".join(map(str, rand_ids(20 + 5 * i, 30 + i))) + "\n").encode()
        open(os.path.join(sdir, f"{n}.ids"), "wb").write(raw)
        man.append({"name": n, "ids": 20 + 5 * i, "sha256": hashlib.sha256(raw).hexdigest()})
        open(os.path.join(run, f"{n}.bf16.ids"), "w").write(" ".join(map(str, rand_ids(6, 40 + i))) + "\n")
    json.dump(man, open(os.path.join(sdir, "manifest.json"), "w"))
    res = {}
    for label, bf in (("twin", variant("int4-g64") + "-twin"), ("original", variant("pattern12"))):
        for f in os.listdir(run):
            if f.endswith(".rows.safetensors"):
                os.remove(os.path.join(run, f))
        out = os.path.join(d, f"{label}.json")
        gq.main(["kl", run, variant("int4-g64"), "--set", sdir, "--out", out, "--bf16", bf])
        res[label] = json.load(open(out))["a4"]
    assert res["twin"]["kl_max"] == 0.0 and res["twin"]["top1"] == 1.0, res["twin"]
    assert res["original"]["kl_mean"] > 0.0, res["original"]
    a4 = os.path.join(d, "a4")
    os.makedirs(a4)
    call = "x</think>\n<tool_call>\n<function=read>\n<parameter=filePath>\n/w\n</parameter>\n</function>\n</tool_call>"
    for n in ("t1_a", "t2_b"):
        open(os.path.join(a4, f"{n}.bf16.txt"), "w").write(call)
        open(os.path.join(a4, f"{n}.int4.txt"), "w").write(call if n == "t1_a" else "still thinking")
    t = gq.a4_table(a4)
    assert t == {"n": 2, "ref_fail": 0, "cand_fail": 0, "agree": 1, "missing": 0}, t
    ok, lines = gq.verdict(a4, os.path.join(d, "twin.json"))
    assert not ok and lines[-1].startswith("gate Q: OUTSIDE")
    shutil.rmtree(d)
    print(f"  gateq: int4-g64 vs its twin KL {res['twin']['kl_max']} top-1 {res['twin']['top1']}; vs pattern12 KL mean "
          f"{res['original']['kl_mean']:.3g} top-1 {res['original']['top1']:.3f}; the verdict's A4 table {t}")


def test_cli_run():
    import subprocess
    import tempfile
    from safetensors import safe_open
    snap = variant("pattern12")
    d = tempfile.mkdtemp(prefix="bolt_cli_")
    p = os.path.join(d, "p.ids")
    with open(p, "w") as f:
        f.write(" ".join(map(str, rand_ids(30, 25))) + "\n")
    out = os.path.join(d, "p.golden.safetensors")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "bolt_ref.py"), "run", snap, "--prompt", p, "--out", out,
                        "--gen", "3", "--weights", "bf16", "--taps"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "MoE 9th/10th sel gap" in r.stdout and "SSM state after the prompt" in r.stdout
    with safe_open(out, framework="pt") as f:
        keys = set(f.keys())
        assert f.get_tensor("logits").shape == (33, 131072) and f.get_tensor("tokens").numel() == 3
        assert f.get_tensor("route.ids.L1").shape == (33, 9) and f.get_tensor("route.sel.L3").shape == (33, 10)
        assert f.get_tensor("ssm.L0").shape == (8, 32, 16) and f.get_tensor("conv.L2").shape == (3, 320)
        assert f.get_tensor("h.L11").shape == (33, 256) and f.get_tensor("nll").shape == (29,)
        assert f.get_tensor("tap.L1").shape == (33, 256) and "ssm.L1" not in keys and "route.ids.L0" not in keys
        ids_ok = bool((f.get_tensor("route.ids.L1").diff(dim=-1) > 0).all())
        meta = f.metadata()
    assert ids_ok and meta["mamba_path"] == "kernel" and meta["cache"] == "fp32"
    print(f"  run: {len(keys)} tensors, the golden layout (route ids ascending, ssm fp32, conv [3][C], taps)")


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
