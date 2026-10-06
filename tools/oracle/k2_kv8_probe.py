#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 18e Task 1, Review Focus 1: rotkv at head_dim 128 re-checked on K2-Horizon with 18a's
reference (tools/oracle/k2_ref.py), spec 12a's method - the golden decision rows, A4-style long
prompts, and the attention-only replay at depth - before int8 KV ships for K2.

The model is k2_ref.py's K2Ref layer at a time; every variant carries its OWN residual stream and
KV cache through the whole model (one layer's weights loaded once, the variants run on it in turn),
so a variant's KV error propagates as it would in the engine - through 45 sparse layers whose MoE
and MoVA routers are tie-sensitive (biases 8-28 in bf16 steps): the routing diagnostic below counts
where each variant's expert sets part from the reference's. Variants (attention only; everything
else is k2_ref's bf16 arithmetic):

  bf16     k2_ref.attention() itself - the reference (bitwise HF eager in bf16)
  rotkv    the ENGINE's scheme, bit for bit src/common/kv8.h hd128: K = rotate_kv(bf16 K),
           V = rotate_kv(bf16 V) (MoVA's routed mix, the cache's V), per (position, kv head) int8
           with an fp16 scale (amax / 127, RNE, divided by the ROUNDED scale, +-127), q =
           rotate_q(q) (rounded to bf16 in a prompt pass - the prefill flash's operand - fp32 in a
           one-token step - decode's); then the reference's rounding points on those operands
           (score rounded twice, fp32 softmax rounded to bf16, P.V in fp32: the engine's eager
           form), the output un-rotated in fp32 and rounded once - k2_attn_eager_*_kv8's chain
  pt       per token int8 + fp16 scale WITHOUT the rotation (the control rotkv must beat)
  f32attn  the attention in fp32 on the bf16 cache (no score / probability rounding): the
           envelope of what attention arithmetic alone moves

The rotation's scales are powers of two (rotate_kv /8 = sqrt(2) x R, rotate_q /16 = R / sqrt(2),
unrotate /16), so q.k and the un-rotated output are bf16 KV's up to the int8 error alone.

Subcommands (in the oracle container, agnes-ref-img: torch 2.14.1, safetensors):

  run      one prompt (+ teacher-forced continuation ids, or --gen N greedy steps through the
           cache on the reference's own choice) through every variant; per-position logits
           metrics against the bf16 element (cos, KL, argmax, the reference's gap), the routing
           diagnostic per variant (rows x layers whose MoVA / MoE expert sets differ), optional
           q / K / V capture of the bf16 element per layer
  replay   fp64 attention-only replay from a capture at real and tiled depths, per scheme
           (rotkv, pt, K:rot, V:rot) against bf16 KV - the 12a replay at head_dim 128
  summary  the tables over `run` outputs (kv_int8_probe.py's summary format)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

VARIANTS = ("bf16", "rotkv", "pt", "f32attn")
QMAX = 127
# The engine's signs (src/common/kv8.h kSignWords[0..3]): bit j of word j / 32 set when s[j] = -1;
# torch's hadamard(128, 0) - randint(0, 2, (128,), Generator().manual_seed(0)) * 2 - 1.
SIGN_WORDS = (0xA197D809, 0xA6850592, 0xB205A73E, 0x20AFA769)


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # k2_ref's dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the scheme (kv8.h hd128, torch)
def signs(n: int = 128) -> torch.Tensor:
    """+-1 per dim, fp32 (the engine's)."""
    if n != 128:
        raise ValueError("the engine's K2 signs are 128")
    neg = [(SIGN_WORDS[j >> 5] >> (j & 31)) & 1 for j in range(n)]
    return torch.tensor([-1.0 if b else 1.0 for b in neg], dtype=torch.float32)


def torch_signs(n: int = 128, seed: int = 0) -> torch.Tensor:
    """hadamard(n, seed)'s signs as kv_int8_probe.py derives them (for the self-check)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=g).float() * 2 - 1


def fwht(x: torch.Tensor) -> torch.Tensor:
    """The canonical in-place FWHT over the last dim, stages h = 1, 2, ... ascending, (a + c, a - c):
    fp32, one rounding per op in kv8.h's order (the same values as the host and the device)."""
    x = x.float().clone()
    n = x.shape[-1]
    h = 1
    while h < n:
        y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, c = y[..., 0, :].clone(), y[..., 1, :].clone()
        y[..., 0, :] = a + c
        y[..., 1, :] = a - c
        x = y.reshape(*x.shape[:-1], n)
        h *= 2
    return x


def rotate_kv(x):
    return fwht(x) * signs(x.shape[-1]) * 0.125


def rotate_q(x):
    return fwht(x) * signs(x.shape[-1]) * 0.0625


def unrotate(y):
    return fwht(y.float() * signs(y.shape[-1])) * 0.0625


def quant_dequant(y: torch.Tensor) -> torch.Tensor:
    """kv8.h quantise + dequant over the last dim: s16 = fp16(amax / 127), q = clamp(rint(y / f(s16)),
    +-127) (0 where the scale is 0), deq = q x f(s16) - exact in fp32."""
    y = y.float()
    s = (y.abs().amax(-1, keepdim=True) / QMAX).half().float()
    q = torch.where(s > 0, torch.clamp(torch.round(y / torch.where(s > 0, s, torch.ones_like(s))), -QMAX, QMAX),
                    torch.zeros_like(y))
    return q * s


# ---------------------------------------------------------------- the variants' attention
def make_attention(ref_mod, name: str, chunk: int = 256):
    """An attention(ar, q, k, v, q_pos, k_pos, n_rep) for k2_ref.K2Ref.attn: q [H, T, d], k / v
    [Hkv, P, d] (the cache: post-RoPE K, MoVA's V) -> [T, H, d]. Chunked over queries (the
    reference's eager math is row-local: chunking leaves bf16 bitwise)."""
    base = ref_mod.attention

    def bf16(ar, q, k, v, q_pos, k_pos, n_rep):
        outs = [base(ar, q[:, i:i + chunk], k, v, q_pos[i:i + chunk], k_pos, n_rep)
                for i in range(0, q.shape[1], chunk)]
        return torch.cat(outs, dim=0)

    def f32attn(ar, q, k, v, q_pos, k_pos, n_rep):
        d = q.shape[-1]
        kk, vv = k.repeat_interleave(n_rep, 0).float(), v.repeat_interleave(n_rep, 0).float()
        outs = []
        for i in range(0, q.shape[1], chunk):
            qc, qp = q[:, i:i + chunk].float(), q_pos[i:i + chunk]
            s = (qc @ kk.transpose(1, 2)) * (d ** -0.5)
            s = s.masked_fill((k_pos[None, :] > qp[:, None])[None], float("-inf"))
            outs.append(ar.r(torch.softmax(s, -1) @ vv).transpose(0, 1))
        return torch.cat(outs, dim=0)

    def int8(rot: bool):
        def fn(ar, q, k, v, q_pos, k_pos, n_rep):
            d = q.shape[-1]
            if rot:
                kq, vq = quant_dequant(rotate_kv(k)), quant_dequant(rotate_kv(v))
                qr = rotate_q(q)
                if q.shape[1] > 1:          # a prompt pass: the prefill flash's bf16 operand
                    qr = qr.to(torch.bfloat16).float()
            else:
                kq, vq, qr = quant_dequant(k), quant_dequant(v), q.float()
            kk, vv = kq.repeat_interleave(n_rep, 0), vq.repeat_interleave(n_rep, 0)
            outs = []
            for i in range(0, q.shape[1], chunk):
                qc, qp = qr[:, i:i + chunk], q_pos[i:i + chunk]
                s = ar.r(qc @ kk.transpose(1, 2))           # fp32 dot (exact operands), rounded
                s = ar.r(s * (d ** -0.5))
                s = s.masked_fill((k_pos[None, :] > qp[:, None])[None], float("-inf"))
                p = ar.r(torch.softmax(s, dim=-1))
                o = p @ vv                                    # fp32
                if rot:
                    o = unrotate(o)
                outs.append(ar.r(o).transpose(0, 1))
            return torch.cat(outs, dim=0)
        return fn

    return {"bf16": bf16, "f32attn": f32attn, "rotkv": int8(True), "pt": int8(False)}[name]


# ---------------------------------------------------------------- the multi-variant forward
class Probe:
    """K2Ref's forward with one residual stream and one KV cache per variant; each layer's weights
    loaded once. `capture`: layer -> (q, k, v) of the bf16 element's LAST forward (bf16 values)."""

    def __init__(self, ref_mod, ref, variants):
        self.m, self.ref, self.variants = ref_mod, ref, list(variants)
        if self.variants[0] != "bf16":
            raise ValueError("the first variant must be bf16 (the reference)")
        self.attn = {v: make_attention(ref_mod, v) for v in self.variants}
        self.caches = {v: ref.new_cache() for v in self.variants}
        self.recs = {v: ref_mod.Recorder(0) for v in self.variants}
        self.capture = None

    @torch.no_grad()
    def forward(self, ids, pos0: int):
        m, ref = self.m, self.ref
        c, ar = ref.c, ref.ar
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
        T = ids.numel()
        pos = torch.arange(pos0, pos0 + T)
        cos, sin = m.rope_cos_sin(ar, pos, c.rope_dim, c.rope_theta)
        G, eps = c.layernorm_num_groups, c.rms_norm_eps
        h = {v: ref.embed[ids].float() for v in self.variants}
        orig = m.attention
        try:
            for i in range(c.num_hidden_layers):
                lw = ref.layer(i)
                for v in self.variants:
                    if self.capture is not None and v == "bf16":
                        def cap(ar_, q, k, vv, qp, kp, n_rep, _i=i, _f=self.attn[v]):
                            self.capture[_i] = (q.to(torch.bfloat16), k.to(torch.bfloat16), vv.to(torch.bfloat16))
                            return _f(ar_, q, k, vv, qp, kp, n_rep)
                        m.attention = cap
                    else:
                        m.attention = self.attn[v]
                    a = ref.attn(i, lw, m.grouped_rms_norm(ar, h[v], lw["input_layernorm"], G, eps), pos, cos, sin,
                                 self.caches[v], self.recs[v])
                    h[v] = ar.r(h[v] + a)
                    f = ref.ffn(i, lw, m.grouped_rms_norm(ar, h[v], lw["post_attention_layernorm"], G, eps),
                                self.recs[v])
                    h[v] = ar.r(h[v] + f)
                del lw
        finally:
            m.attention = orig
        return {v: ref.head(m.grouped_rms_norm(ar, h[v], ref.norm_w, G, eps)) for v in self.variants}

    def routes(self):
        """variant -> {name: [rows, k] int tensor} for every route.*.ids.L* the forwards recorded."""
        return {v: {k: torch.cat(x, 0) for k, x in r.t.items() if ".ids." in k} for v, r in self.recs.items()}


def logits_metrics(lg: dict, names) -> dict:
    """lg: variant -> fp32 [T, V]. kv_int8_probe.py's metric dict (cos, KL, argmax, gap) per
    comparison (variants 1.. and ctl_bf16out = the reference's logits rounded to bf16)."""
    ref = lg[names[0]].double()
    lref = torch.log_softmax(ref, -1)
    am0 = ref.argmax(-1)
    top2 = ref.topk(2, -1).values
    res = {"ref": {"argmax": am0, "top2gap": (top2[:, 0] - top2[:, 1])}}
    comps = [(n, lg[n].double()) for n in names[1:]] + [("ctl_bf16out", lg[names[0]].to(torch.bfloat16).double())]
    for c, x in comps:
        am = x.argmax(-1)
        res[c] = {"cos": torch.nn.functional.cosine_similarity(x, ref, dim=-1),
                  "kl": (lref.exp() * (lref - torch.log_softmax(x, -1))).sum(-1),
                  "argmax": am,
                  "gap": (ref.gather(-1, am0[:, None]) - ref.gather(-1, am[:, None]))[:, 0]}
    return res


def routing_diag(routes: dict, names) -> dict:
    """Per variant: rows x layers whose MoVA / MoE expert set differs from the reference's, and the
    first (layer, row) where it does."""
    out = {}
    ref = routes[names[0]]
    for v in names[1:]:
        d = {"mova": 0, "moe": 0, "rows_x_layers": 0, "first": None}
        for k, ids in ref.items():
            got = routes[v][k]
            diff = (got != ids).any(-1)
            which = "mova" if ".mova." in k else "moe"
            d[which] += int(diff.sum())
            d["rows_x_layers"] += int(diff.numel())
            if diff.any():
                L = int(k.rsplit(".L", 1)[1])
                r = int(diff.nonzero()[0, 0])
                if d["first"] is None or (L, r) < tuple(d["first"]):
                    d["first"] = [L, r]
        out[v] = d
    return out


def _read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def run_probe(ref_mod, ref, ids, variants, cont=(), gen=0, capture=None, log=print):
    """The probe on an id list: one prompt forward over ids + cont (teacher-forced), then `gen`
    greedy steps through the caches (every variant fed the REFERENCE's argmax). Returns (logits
    dict variant -> [rows, V], tokens, routes, n_prompt)."""
    p = Probe(ref_mod, ref, variants)
    p.capture = capture
    seq = list(ids) + list(cont)
    t = time.time()
    rows = {v: [x] for v, x in p.forward(seq, 0).items()}
    log(f"prompt forward ({len(seq)} ids x {len(variants)} variants): {time.time() - t:.1f}s")
    p.capture = None
    toks = []
    for step in range(gen):
        nxt = int(torch.argmax(rows["bf16"][-1][-1]))
        toks.append(nxt)
        t = time.time()
        for v, x in p.forward([nxt], len(seq) + step).items():
            rows[v].append(x)
        log(f"  step {step}: token {nxt}, {time.time() - t:.1f}s")
    return {v: torch.cat(x, 0).float() for v, x in rows.items()}, toks, p.routes(), len(ids)


def cmd_run(a) -> None:
    import resource
    from safetensors.torch import save_file
    torch.manual_seed(0)
    t0 = time.time()
    m = _load("k2_ref")
    src = m.Checkpoint(a.snapshot)
    ref = m.K2Ref(src.cfg, src, mode="bf16")
    variants = tuple(a.variants.split(","))
    ids = _read_ids(a.ids)
    if a.n:
        ids = ids[:a.n]
    cont = _read_ids(a.cont_ids)[:a.gen_cont] if a.cont_ids else []
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}; {len(ids)} prompt ids + {len(cont)} "
          f"teacher-forced + {a.gen} greedy; variants {variants}", flush=True)
    capture = {} if a.capture else None
    lg, toks, routes, n_prompt = run_probe(m, ref, ids, variants, cont, a.gen, capture,
                                           log=lambda s: print(s, flush=True))
    n = lg["bf16"].shape[0]
    if capture is not None:
        tens = {}
        for i, (q, k, v) in capture.items():
            tens[f"q.L{i}"], tens[f"k.L{i}"], tens[f"v.L{i}"] = q.contiguous(), k.contiguous(), v.contiguous()
        save_file(tens, a.capture, metadata={"ids": a.ids, "n_prompt": str(len(ids) + len(cont)), "n": str(n),
                                              "fa_layers": ",".join(map(str, sorted(capture))),
                                              "head_dim": "128", "scale": repr(128 ** -0.5)})
        print(f"captured {len(capture)} layers -> {a.capture}", flush=True)
    out = {"meta": {"ids": a.ids, "n_prompt": n_prompt, "n": n, "variants": variants, "forward_s": time.time() - t0,
                    "wall_s": time.time() - t0, "tokens": toks, "model": "k2_horizon",
                    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20},
           "metrics": logits_metrics(lg, variants), "routing": routing_diag(routes, variants)}
    torch.save(out, a.out)
    print(f"wrote {a.out}; wall {time.time() - t0:.0f}s", flush=True)
    summarize([a.out])


def print_routing(d) -> None:
    print("| variant | MoVA rows x layers differing | MoE | of | first (layer, row) |")
    print("|---|---:|---:|---:|---|")
    for v, r in d["routing"].items():
        print(f"| {v} | {r['mova']} | {r['moe']} | {r['rows_x_layers'] // 2} | {r['first']} |")


def summarize(paths) -> None:
    kv = _load("kv_int8_probe")
    for p in paths:
        kv.summarize([p])
        d = torch.load(p, weights_only=False)
        if "routing" in d:
            print(f"\nrouting ({os.path.basename(p)}): expert sets differing from the bf16 element's")
            print_routing(d)


# ---------------------------------------------------------------- replay (fp64, 12a's at head_dim 128)
def replay_operands(q, k, v):
    """q [H, T, 128], k, v [Hkv, T, 128] bf16 -> {name: (q_op, k_op, v_op, out_fn)} fp64; out_fn
    maps the fp64 attention output back to the un-rotated basis (None: none needed)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    kr, vr = quant_dequant(rotate_kv(kf)).double(), quant_dequant(rotate_kv(vf)).double()
    qr = rotate_q(qf).to(torch.bfloat16).double()          # the prefill flash's operand

    def un(o):   # unrotate in fp64: y R^T / sqrt(2) = FWHT(s y) / 16
        x = o * signs().double()
        n, h = x.shape[-1], 1
        while h < n:
            y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
            a, c = y[..., 0, :].clone(), y[..., 1, :].clone()
            y[..., 0, :], y[..., 1, :] = a + c, a - c
            x = y.reshape(*x.shape[:-1], n)
            h *= 2
        return x / 16.0
    q64, k64, v64 = q.double(), k.double(), v.double()
    return {"rotkv": (qr, kr, vr, un),
            "pt": (q64, quant_dequant(kf).double(), quant_dequant(vf).double(), None),
            "K:rot": (qr, kr, (rotate_kv(vf)).double(), un),      # V exact (rotated), K int8
            "V:rot": (qr, rotate_kv(kf).double(), vr, un)}        # K exact (rotated), V int8


def cmd_replay(a) -> None:
    from safetensors import safe_open
    kv = _load("kv_int8_probe")
    t0 = time.time()
    f = safe_open(a.capture, framework="pt", device="cpu")
    md = f.metadata()
    layers = [int(x) for x in md["fa_layers"].split(",")]
    tcap = int(md["n_prompt"])
    scale = 128 ** -0.5
    g = torch.Generator().manual_seed(0)
    buckets = []
    for dpt in [int(x) for x in a.depths.split(",") if x]:
        if dpt <= tcap:
            lo = max(0, dpt - a.window)
            buckets.append((f"{dpt}", (lo + torch.randperm(dpt - lo, generator=g)[:a.nq]).sort().values, 1))
    for dpt in [int(x) for x in a.tiles.split(",") if x]:
        if dpt % tcap:
            raise SystemExit(f"tile depth {dpt} is not a multiple of the capture's {tcap}")
        lo = tcap - a.window
        buckets.append((f"{dpt} tiled", (lo + torch.randperm(a.window, generator=g)[:a.nq]).sort().values,
                        dpt // tcap))
    schemes = ("rotkv", "pt", "K:rot", "V:rot")
    comps = list(schemes) + ["ctl_bf16out"]
    acc = {(b[0], c): {"cos": [], "maxabs": [], "rel": []} for b in buckets for c in comps}
    for li in layers:
        q, k, v = (f.get_tensor(f"{x}.L{li}")[:, :tcap] for x in "qkv")
        ops = replay_operands(q, k, v)
        for lab, pos, mcp in buckets:
            lens = (mcp - 1) * tcap + pos + 1

            def tile(x):
                return x if mcp == 1 else x.repeat(1, mcp, 1)
            ref = kv.attend64(q[:, pos].double(), tile(k.double()), tile(v.double()), lens, scale)
            outs = {"ctl_bf16out": ref.to(torch.bfloat16).double()}
            for s in schemes:
                qo, ko, vo, un = ops[s]
                o = kv.attend64(qo[:, pos], tile(ko), tile(vo), lens, scale)
                outs[s] = o if un is None else un(o)
            for c, o in outs.items():
                mt = kv._metrics(o, ref)
                for key in mt:
                    acc[(lab, c)][key].append(mt[key])
        print(f"layer {li}: {time.time() - t0:.0f}s", flush=True)
    print(f"\n### replay: {os.path.basename(a.capture)} ({tcap} positions, {len(layers)} layers, {a.nq} queries per "
          f"bucket from the last {a.window}; tiled = the capture repeated), head_dim 128, scale 1/sqrt(128)")
    print("| depth | scheme | cos min | cos p0.1 | cos mean | max abs | rel L2 mean | rel L2 max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for lab, _, _ in buckets:
        for c in comps:
            cos = torch.cat(acc[(lab, c)]["cos"])
            mx = torch.cat(acc[(lab, c)]["maxabs"])
            rel = torch.cat(acc[(lab, c)]["rel"])
            row = {"cos_min": float(cos.min()), "cos_p001": float(torch.quantile(cos, 0.001)),
                   "cos_mean": float(cos.mean()), "maxabs": float(mx.max()), "rel_mean": float(rel.mean()),
                   "rel_max": float(rel.max())}
            summary[f"{lab}/{c}"] = row
            print(f"| {lab} | {c} | {row['cos_min']:.7f} | {row['cos_p001']:.7f} | {row['cos_mean']:.8f} | "
                  f"{row['maxabs']:.2e} | {row['rel_mean']:.2e} | {row['rel_max']:.2e} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fo:
            json.dump({"capture": a.capture, "rows": summary}, fo, indent=1)
    print(f"replay wall {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--ids", required=True, help="K2's tokenisation, BOS 0 first (tokenize.py --bos)")
    r.add_argument("--n", type=int, default=0, help="the first N ids only (0: all)")
    r.add_argument("--cont-ids", help="ids teacher-forced after the prompt (e.g. a golden set's tokens)")
    r.add_argument("--gen-cont", type=int, default=32, help="how many of --cont-ids")
    r.add_argument("--gen", type=int, default=0, help="greedy steps through the caches (decode's q path)")
    r.add_argument("--variants", default=",".join(VARIANTS))
    r.add_argument("--capture", help="write the bf16 element's q / K / V per layer here (prompt forward)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("capture")
    p.add_argument("--depths", default="1024,2048,4096")
    p.add_argument("--tiles", default="16384,32768,65536")
    p.add_argument("--nq", type=int, default=32)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--out")
    s = sub.add_parser("summary")
    s.add_argument("runs", nargs="+")
    a = ap.parse_args()
    {"run": cmd_run, "replay": cmd_replay, "summary": lambda x: summarize(x.runs)}[a.cmd](a)


if __name__ == "__main__":
    main()
