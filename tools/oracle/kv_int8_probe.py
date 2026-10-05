#!/usr/bin/env python3
"""Spec 12a: an int8 KV cache, the accuracy probe on the CPU (spec 12 §2, §4 Q1).

Three int8 schemes against bf16 KV, all symmetric (q = rne(x / s) clamped to
[-127, 127], s = amax / 127, scales fp32 unless named), K quantised AFTER the
k-norm and RoPE (what the cache stores), V as stored, q never quantised:

    pt    per token, per head: one scale per (position, kv head) for K and for V
    pt16  pt with the scales rounded to fp16 (the cheaper scale array)
    kivi  K per channel over groups of 64 positions (aligned at position 0); the last
          partial group stays bf16 until it fills (a decoder at length L has
          positions >= floor(L / 64) * 64 in bf16); V per token
    rot   K rotated per kv head by a 256 Hadamard with random signs (x -> x @ R),
          then per token; q gets the same rotation (it cancels in q.k); V per token

Dequantised K / V (and the rotated q) are rounded to bf16, as the engine's attention
builds bf16 DPAS operands from int8 x scale.

Subcommands (all in the oracle container; `run` layer-streams the model, stream.py):

  run      teacher-forced forward of one id sequence with the batch dimension carrying
           the variants (bf16 = the oracle's own eager math, pt, kivi, rot, f32attn =
           the attention in fp32 as a control); every FA layer's attention is replaced
           by `make_e2e_attention`, which quantises each element's K/V by its scheme.
           Per-position logits metrics against the bf16 element (fp32 lm_head), plus the
           bf16-output control; optionally captures q, K, V of the bf16 element.
  replay   attention-only replay from a capture: fp64 attention per scheme against bf16
           KV at sampled query positions per depth bucket, real depths and the capture
           tiled to 16k / 32k; controls: the bf16-rounded output and the eager bf16 math.
  summary  tables over `run` outputs.
  tokcheck the snapshot's tokenizer.json round trip on an ids file (+ combining marks).
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

QMAX = 127
GROUP = 64
E2E_VARIANTS = ("bf16", "pt", "kivi", "rot", "rotkv", "f32attn")
REPLAY_SCHEMES = ("pt", "pt16", "kivi", "rot")
# Diagnostics beside the three schemes: K alone (V bf16) and V alone (K bf16), to say which
# side carries the error; `V:rot` rotates V per token like K and un-rotates the output
# (o @ R^T, foldable into o_proj's per-head columns) - not a spec 12 scheme.
REPLAY_PARTS = ("K:pt", "K:kivi", "K:rot", "V:pt", "V:rot", "KV:rot")


# ---------------------------------------------------------------- the schemes
def _quant(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.clamp(torch.round(x / safe), -QMAX, QMAX) * s


def quant_per_token(x: torch.Tensor, scale_dtype=torch.float32) -> torch.Tensor:
    """Symmetric int8 over the last dim (one scale per head and position), dequantised, fp32."""
    x = x.float()
    s = (x.abs().amax(-1, keepdim=True) / QMAX).to(scale_dtype).float()
    return _quant(x, s)


def quant_kivi_k(x: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """K per channel over groups of `group` positions (axis -2, aligned at 0); the last
    partial group is returned unchanged (bf16 in the cache until it fills). fp32."""
    x = x.float()
    t = x.shape[-2]
    full = t // group * group
    out = x.clone()
    if full:
        xg = x[..., :full, :].reshape(*x.shape[:-2], full // group, group, x.shape[-1])
        s = xg.abs().amax(-2, keepdim=True) / QMAX
        out[..., :full, :] = _quant(xg, s).reshape(*x.shape[:-2], full, x.shape[-1])
    return out


def kivi_tail_start(p):
    """First bf16 position for a decoder holding positions 0..p (length p + 1)."""
    return (p + 1) // GROUP * GROUP


def hadamard(n: int = 256, seed: int = 0) -> torch.Tensor:
    """Orthogonal R = H_n diag(signs) / sqrt(n), Sylvester H, random +-1 signs; fp64."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != n:
        raise ValueError(f"{n} is not a power of two")
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
    return h * signs[None, :] / math.sqrt(n)


_R32 = None


def _rot32() -> torch.Tensor:
    global _R32
    if _R32 is None:
        _R32 = hadamard(256, 0).float()
    return _R32


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


# --fp32-matmul: a bf16 x bf16 matmul computed as fp32 sgemm on the same (exactly
# representable) values, then rounded once to bf16 - the same math as the bf16 kernel
# (exact products, fp32 accumulation, one RNE) in another summation order; ~5x faster on
# the Mac's AVX2 CPU (346 vs 64 GFLOPS measured). Off by default: the tests check the
# exact eager path.
FP32_MATMUL = False


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if FP32_MATMUL and a.dtype == torch.bfloat16:
        return torch.matmul(a.float(), b.float()).to(torch.bfloat16)
    return torch.matmul(a, b)


def fp32_linear() -> None:
    """nn.Linear on bf16 inputs as fp32 sgemm rounded to bf16 (see FP32_MATMUL)."""
    global FP32_MATMUL
    FP32_MATMUL = True
    import torch.nn.functional as F
    from torch import nn
    if getattr(nn.Linear.forward, "_fp32", False):
        return
    orig = nn.Linear.forward

    def forward(self, x):
        if x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
            b = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), b).to(torch.bfloat16)
        return orig(self, x)
    forward._fp32 = True
    nn.Linear.forward = forward


# ---------------------------------------------------------------- replay (fp64)
def attend64(q, k, v, lens, scale, k_tail=None):
    """fp64 softmax attention. q [H, n, D]; k, v [Hkv, Tk, D]; query i sees keys < lens[i].
    k_tail: KIVI's bf16 K, used for keys >= kivi_tail_start(lens[i] - 1)."""
    h, n, d = q.shape
    hkv, tk = k.shape[0], k.shape[1]
    g = h // hkv
    j = torch.arange(tk)
    lens = torch.as_tensor(lens)
    valid = j[None, :] < lens[:, None]
    use_tail = None if k_tail is None else j[None, :] >= kivi_tail_start(lens - 1)[:, None]
    out = torch.empty(h, n, d, dtype=torch.float64)
    for hk in range(hkv):
        qh = q[hk * g:(hk + 1) * g].double()
        s = qh @ k[hk].double().T * scale
        if use_tail is not None:
            s = torch.where(use_tail, qh @ k_tail[hk].double().T * scale, s)
        s = s.masked_fill(~valid, float("-inf"))
        out[hk * g:(hk + 1) * g] = torch.softmax(s, -1) @ v[hk].double()
    return out


def attend_eager_like(q, k, v, lens, scale):
    """The oracle's eager math emulated on the replay's queries: bf16 scores, fp32 softmax
    rounded to bf16, bf16 P.V (fp32 accumulate). q, k, v bf16-valued."""
    h, n, d = q.shape
    hkv, tk = k.shape[0], k.shape[1]
    g = h // hkv
    j = torch.arange(tk)
    valid = j[None, :] < torch.as_tensor(lens)[:, None]
    out = torch.empty(h, n, d, dtype=torch.float64)
    for hk in range(hkv):
        s = _bf(_bf(q[hk * g:(hk + 1) * g]) @ _bf(k[hk]).T) * scale
        s = s.float().masked_fill(~valid, float("-inf"))
        p = _bf(torch.softmax(s, -1))
        out[hk * g:(hk + 1) * g] = (p @ _bf(v[hk])).double()
    return out


def _metrics(x: torch.Tensor, ref: torch.Tensor) -> dict:
    """Per (head, query) over D: cosine, max abs error, relative L2."""
    cos = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    err = (x - ref)
    return {"cos": cos.flatten(), "maxabs": err.abs().amax(-1).flatten(),
            "rel": (err.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-300)).flatten()}


def replay_operands(q, k, v):
    """q [H, T, D], k, v [Hkv, T, D] (bf16) -> {name: (q_op, k_op, v_op, k_tail, out_rot)} as
    fp64; out_rot (or None) multiplies the attention output on the right (V:rot's R^T)."""
    kf, vf = k.float(), v.float()
    q64, k64, v64 = q.double(), k.double(), v.double()
    vpt = _bf(quant_per_token(vf)).double()
    r = _rot32()
    kpt = _bf(quant_per_token(kf)).double()
    kkv = _bf(quant_kivi_k(kf)).double()
    qr, kr = _bf(q.float() @ r).double(), _bf(quant_per_token(kf @ r)).double()
    ops = {
        "pt": (q64, kpt, vpt, None, None),
        "pt16": (q64, _bf(quant_per_token(kf, torch.float16)).double(),
                 _bf(quant_per_token(vf, torch.float16)).double(), None, None),
        "kivi": (q64, kkv, vpt, k64, None),
        "rot": (qr, kr, vpt, None, None),
        "K:pt": (q64, kpt, v64, None, None),
        "K:kivi": (q64, kkv, v64, k64, None),
        "K:rot": (qr, kr, v64, None, None),
        "V:pt": (q64, k64, vpt, None, None),
        "V:rot": (q64, k64, _bf(quant_per_token(vf @ r)).double(), None, hadamard(256, 0).T),
        "KV:rot": (qr, kr, _bf(quant_per_token(vf @ r)).double(), None, hadamard(256, 0).T),
    }
    return ops


def rotation_check(q, k, n: int = 64) -> float:
    """Review Focus 2 on real rows: max |(qR).(kR) - q.k| / (|q||k|) in fp64."""
    r = hadamard(256, 0)
    g = q.shape[0] // k.shape[0]
    qs = q[:, -n:].double()
    ks = k[:, -n:].double().repeat_interleave(g, 0)
    a = qs @ ks.transpose(1, 2)
    b = (qs @ r) @ (ks @ r).transpose(1, 2)
    den = qs.norm(dim=-1)[..., None] * ks.norm(dim=-1)[:, None, :]
    return float(((a - b).abs() / den).max())


# ---------------------------------------------------------------- end to end (patched FA attention)
def attend_variant(name, q, k, v, off, scaling, chunk: int = 256):
    """One batch element's attention, causal with query i at position off + i.
    q [H, Tq, D], k, v [Hkv, Tk, D] bf16 -> [Tq, H, D] bf16. `bf16` is
    eager_attention_forward's math op for op (bitwise; checked), chunked over queries."""
    h, tq, d = q.shape
    hkv, tk = k.shape[0], k.shape[1]
    g = h // hkv
    k_tail = None
    if name == "bf16":
        qo, ko, vo = q, k, v
    elif name == "pt":
        qo, ko, vo = q, _bf(quant_per_token(k)), _bf(quant_per_token(v))
    elif name == "kivi":
        qo, ko, vo, k_tail = q, _bf(quant_kivi_k(k)), _bf(quant_per_token(v)), k
    elif name == "rot":
        r = _rot32()
        qo, ko, vo = _bf(q.float() @ r), _bf(quant_per_token(k.float() @ r)), _bf(quant_per_token(v))
    elif name == "rotkv":   # beyond spec 12: V rotated too, the output un-rotated in fp32
        r = _rot32()
        qo, ko = _bf(q.float() @ r), _bf(quant_per_token(k.float() @ r))
        vo = _bf(quant_per_token(v.float() @ r))
    elif name == "f32attn":
        qo, ko, vo = q.float(), k.float(), v.float()
    else:
        raise ValueError(name)
    unrot = _rot32().T if name == "rotkv" else None

    def rep(x):   # repeat_kv
        return x[:, None].expand(hkv, g, tk, d).reshape(1, h, tk, d)
    kr, vr = rep(ko), rep(vo)
    kt = rep(k_tail) if k_tail is not None else None
    j = torch.arange(tk)
    out = torch.empty(tq, h, d, dtype=q.dtype)
    for i0 in range(0, tq, chunk):
        i1 = min(tq, i0 + chunk)
        qc = qo[None, :, i0:i1]
        s = _mm(qc, kr.transpose(2, 3)) * scaling
        pos = off + torch.arange(i0, i1)
        if kt is not None:
            st = _mm(qc, kt.transpose(2, 3)) * scaling
            s = torch.where(j[None, :] >= kivi_tail_start(pos)[:, None], st, s)
        s = s.masked_fill(j[None, :] > pos[:, None], float("-inf"))
        p = torch.nn.functional.softmax(s, dim=-1, dtype=torch.float32).to(qo.dtype)
        if unrot is None:
            o = _mm(p, vr).transpose(1, 2)
        else:
            o = (torch.matmul(p.float(), vr.float()) @ unrot).transpose(1, 2)
        out[i0:i1] = o[0].to(q.dtype)
    return out


def make_e2e_attention(variants=E2E_VARIANTS, capture=None):
    """An eager_attention_forward replacement: batch element b uses variants[b].
    capture: dict filled with layer_idx -> (q, k, v) of element 0 (bf16, [H|Hkv, T, D])."""
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        b, h, tq, d = query.shape
        if b != len(variants):
            raise RuntimeError(f"batch {b} != {len(variants)} variants")
        if capture is not None:
            capture[module.layer_idx] = tuple(x[0].detach().clone() for x in (query, key, value))
        off = key.shape[2] - tq
        out = torch.empty(b, tq, h, d, dtype=query.dtype)
        for e, name in enumerate(variants):
            out[e] = attend_variant(name, query[e], key[e], value[e], off, scaling)
        return out, None
    return fn


# ---------------------------------------------------------------- run
def _load(name):
    spec = importlib.util.spec_from_file_location("oracle_" + name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def logits_metrics(hs: torch.Tensor, w: torch.Tensor, names, chunk: int = 32) -> dict:
    """hs [B, T, hidden] (element 0 the reference), w [V, hidden] fp32. Per position and
    comparison (variants 1.. and `ctl_bf16out` = the reference's fp32 logits rounded to
    bf16): cosine, argmax, KL(p_ref || p_x) unfiltered at T = 1 (fp64), and the reference's
    own logit gap between its argmax and x's argmax."""
    b, t, _ = hs.shape
    comps = list(names[1:]) + ["ctl_bf16out"]
    res = {c: {k: torch.empty(t, dtype=torch.float64) for k in ("cos", "kl", "gap")} for c in comps}
    for c in comps:
        res[c]["argmax"] = torch.empty(t, dtype=torch.long)
    res["ref"] = {"argmax": torch.empty(t, dtype=torch.long), "top2gap": torch.empty(t, dtype=torch.float64)}
    for i0 in range(0, t, chunk):
        i1 = min(t, i0 + chunk)
        lg = hs[:, i0:i1].float() @ w.T
        ref = lg[0]
        lref = torch.log_softmax(ref.double(), -1)
        am0 = ref.argmax(-1)
        top2 = ref.topk(2, -1).values
        res["ref"]["argmax"][i0:i1] = am0
        res["ref"]["top2gap"][i0:i1] = (top2[:, 0] - top2[:, 1]).double()
        xs = [lg[e] for e in range(1, b)] + [ref.to(torch.bfloat16).float()]
        for c, x in zip(comps, xs):
            am = x.argmax(-1)
            r = res[c]
            r["cos"][i0:i1] = torch.nn.functional.cosine_similarity(x.double(), ref.double(), dim=-1)
            r["kl"][i0:i1] = (lref.exp() * (lref - torch.log_softmax(x.double(), -1))).sum(-1)
            r["argmax"][i0:i1] = am
            r["gap"][i0:i1] = (ref.gather(-1, am0[:, None]) - ref.gather(-1, am[:, None]))[:, 0].double()
    return res


def cmd_run(a) -> None:
    import resource
    t0 = time.time()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    D = _load("dump")
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    variants = tuple(a.variants.split(","))
    if a.fp32_matmul:
        fp32_linear()
    if variants[0] != "bf16":
        raise SystemExit("the first variant must be bf16 (the reference)")
    print(f"torch threads {torch.get_num_threads()}, variants {variants}, fp32 matmul {FP32_MATMUL}",
          flush=True)
    with open(os.path.join(a.snapshot, "config.json"), encoding="utf-8") as f:
        quantised = bool(json.load(f).get("quantization_config"))
    # Spec 12b's Qwen3.8 repeat runs on the bf16 base checkpoint (Qwen/Qwen3.8-27B): no
    # quantization_config, every tensor shipped bf16 - dump.convert copies those as they are
    # (the same streamed path; group_size is then never read).
    group_size = D.check_quant_config(a.snapshot) if quantised else 0
    if not quantised:
        print("unquantised checkpoint: bf16 tensors streamed as shipped", flush=True)
    ids = _read_ids(a.ids)
    if a.n:
        ids = ids[:a.n]
    n_prompt = len(ids)
    golden = None
    if a.cont and a.cont_ids:
        raise SystemExit("--cont and --cont-ids are exclusive")
    if a.cont:
        golden = safe_open(a.cont, framework="pt", device="cpu")
        ids += golden.get_tensor("tokens")[:a.gen].tolist()
    elif a.cont_ids:
        ids += _read_ids(a.cont_ids)[:a.gen]
    print(f"ids: {n_prompt} prompt + {len(ids) - n_prompt} teacher-forced from {a.ids}", flush=True)
    with open(os.path.join(a.snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    agnes = D._agnes.is_agnes(raw)
    if agnes:
        kwargs, parallel = D._agnes.translate_text_config(raw)
        tc = Qwen3_5TextConfig(**kwargs)
    else:
        from transformers import AutoConfig
        tc = AutoConfig.from_pretrained(a.snapshot).get_text_config()
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = m.Qwen3_5ForCausalLM(tc)
        if agnes:
            D._agnes.attach_parallel_ffn(model, tc, parallel)
    pf = D.stream_weights(model, a.snapshot, group_size, tc, 0)
    model.eval()
    fa = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
    capture = {} if a.capture else None
    m.eager_attention_forward = make_e2e_attention(variants, capture)
    times = {}

    def pre(i):
        def f(_m, _a, _k=None):
            times[i] = time.time()
        return f

    def post(i):
        def f(_m, _a, _o):
            dt = time.time() - times[i]
            print(f"layer {i:2d} {'FA ' if i in fa else 'GDN'} {dt:7.1f}s  wall {time.time() - t0:7.0f}s",
                  flush=True)
        return f
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(pre(i), with_kwargs=True)
        layer.register_forward_hook(post(i))
    x = torch.tensor([ids], dtype=torch.long).expand(len(variants), -1)
    t_f = time.time()
    with torch.no_grad():
        hs = model.model(input_ids=x, use_cache=False).last_hidden_state
    t_fwd = time.time() - t_f
    print(f"forward: {t_fwd:.1f}s ({len(ids)} ids x {len(variants)}), dequant wait {pf.seconds:.1f}s", flush=True)
    if capture is not None:
        miss = sorted(set(fa) - set(capture))
        if miss:
            raise SystemExit(f"FA layers not captured: {miss}")
        tens = {}
        for i in fa:
            q, k, v = capture[i]
            tens[f"q.L{i}"], tens[f"k.L{i}"], tens[f"v.L{i}"] = q.contiguous(), k.contiguous(), v.contiguous()
        save_file(tens, a.capture, metadata={"ids": a.ids, "n_prompt": str(n_prompt), "n": str(len(ids)),
                                              "fa_layers": ",".join(map(str, fa))})
        print(f"captured {len(fa)} FA layers -> {a.capture}", flush=True)
        del capture, tens
    vocab = a.vocab_used or (248089 if agnes else 248077)
    w = model.lm_head.weight[:vocab].float()
    t_l = time.time()
    res = logits_metrics(hs, w, variants)
    print(f"logits metrics: {time.time() - t_l:.1f}s (vocab_used {vocab})", flush=True)
    out = {"meta": {"ids": a.ids, "n_prompt": n_prompt, "n": len(ids), "variants": variants,
                    "agnes": agnes, "forward_s": t_fwd, "wall_s": time.time() - t0,
                    "vocab_used": vocab, "threads": torch.get_num_threads(), "fp32_matmul": FP32_MATMUL},
           "metrics": res}
    if golden is not None:
        gl = golden.get_tensor("logits")
        rows = min(gl.shape[0], len(ids))
        lg0 = hs[0, :rows].float() @ w.T
        out["meta"]["cos_vs_golden_min"] = float(torch.nn.functional.cosine_similarity(
            lg0.double(), gl[:rows, :vocab].double(), dim=-1).min())
        out["meta"]["argmax_vs_golden_equal"] = int((lg0.argmax(-1) == gl[:rows, :vocab].argmax(-1)).sum())
        out["meta"]["golden_rows"] = rows
        print(f"bf16 element vs golden logits: cos min {out['meta']['cos_vs_golden_min']:.9f}, "
              f"argmax equal {out['meta']['argmax_vs_golden_equal']}/{rows}", flush=True)
    out["meta"]["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    torch.save(out, a.out)
    print(f"wrote {a.out}; wall {time.time() - t0:.0f}s, peak RSS {out['meta']['peak_rss_gib']:.1f} GiB",
          flush=True)
    summarize([a.out])


# ---------------------------------------------------------------- summary
def _rows(meta, which):
    n, n_prompt = meta["n"], meta["n_prompt"]
    if which == "decision":
        return list(range(n_prompt - 1, n))
    if which == "last":
        return [n_prompt - 1]
    lo, hi = which
    return [p for p in range(lo, hi) if p < n]


def summarize(paths, buckets=None) -> None:
    for path in paths:
        d = torch.load(path, weights_only=False)
        meta, res = d["meta"], d["metrics"]
        n = meta["n"]
        bks = buckets or (["decision"] if meta["n"] > meta["n_prompt"] else [])
        if not bks:
            edges = [e for e in (0, 512, 2048, 4096, 8192) if e < n] + [n]
            bks = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)] + ["last"]
        print(f"\n### {os.path.basename(path)}: {meta['n_prompt']} prompt ids + {n - meta['n_prompt']}, "
              f"forward {meta['forward_s']:.0f}s")
        print("| rows | variant | cos min | cos mean | argmax diff (ref gap <= 0.05) | KL mean | KL max |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for bk in bks:
            rows = torch.tensor(_rows(meta, bk))
            lab = bk if isinstance(bk, str) else f"{bk[0]}-{bk[1] - 1}"
            for c, r in res.items():
                if c == "ref":
                    continue
                cos, kl = r["cos"][rows], r["kl"][rows]
                diff = r["argmax"][rows] != res["ref"]["argmax"][rows]
                near = diff & (r["gap"][rows] <= 0.05)
                print(f"| {lab} ({len(rows)}) | {c} | {float(cos.min()):.7f} | {float(cos.mean()):.7f} | "
                      f"{int(diff.sum())} ({int(near.sum())}) | {float(kl.mean()):.2e} | {float(kl.max()):.2e} |")


# ---------------------------------------------------------------- replay
def cmd_replay(a) -> None:
    from safetensors import safe_open
    t0 = time.time()
    f = safe_open(a.capture, framework="pt", device="cpu")
    md = f.metadata()
    fa = [int(x) for x in md["fa_layers"].split(",")]
    tcap = int(md["n_prompt"])
    g = torch.Generator().manual_seed(0)
    buckets = []   # (label, query positions in the capture, copies m)
    for dpt in [int(x) for x in a.depths.split(",") if x]:
        if dpt <= tcap:
            lo = max(0, dpt - a.window)
            buckets.append((f"{dpt}", (lo + torch.randperm(dpt - lo, generator=g)[:a.nq]).sort().values, 1))
    for dpt in [int(x) for x in a.tiles.split(",") if x]:
        if dpt % tcap or tcap % GROUP:
            raise SystemExit(f"tile depth {dpt} is not a multiple of the capture {tcap} (or {tcap} % 64)")
        lo = tcap - a.window
        buckets.append((f"{dpt} tiled", (lo + torch.randperm(a.window, generator=g)[:a.nq]).sort().values,
                        dpt // tcap))
    comps = list(REPLAY_SCHEMES) + ["ctl_bf16out", "ctl_eager"] + list(REPLAY_PARTS)
    acc = {(b[0], c): {"cos": [], "maxabs": [], "rel": []} for b in buckets for c in comps}
    worst = {}
    rot_err = 0.0
    for li in fa:
        q, k, v = (f.get_tensor(f"{x}.L{li}")[:, :tcap] for x in "qkv")
        rot_err = max(rot_err, rotation_check(q, k))
        ops = replay_operands(q, k, v)
        for lab, pos, mcp in buckets:
            lens = (mcp - 1) * tcap + pos + 1

            def tile(x):
                return x if mcp == 1 or x is None else x.repeat(1, mcp, 1)
            kk, vv = tile(k.double()), tile(v.double())
            ref = attend64(q[:, pos].double(), kk, vv, lens, a.scale)
            outs = {}
            for s in REPLAY_SCHEMES + REPLAY_PARTS:
                qo, ko, vo, kt, orot = ops[s]
                o = attend64(qo[:, pos], tile(ko), tile(vo), lens, a.scale, k_tail=tile(kt))
                outs[s] = o if orot is None else o @ orot
            outs["ctl_bf16out"] = _bf(ref).double()
            outs["ctl_eager"] = attend_eager_like(q[:, pos].double(), kk, vv, lens, a.scale)
            for c, o in outs.items():
                mt = _metrics(o, ref)
                for key in mt:
                    acc[(lab, c)][key].append(mt[key])
                cmin = float(mt["cos"].min())
                if (lab, c) not in worst or cmin < worst[(lab, c)][1]:
                    worst[(lab, c)] = (li, cmin)
        print(f"layer {li}: {time.time() - t0:.0f}s", flush=True)
    print(f"\nrotation check (Review Focus 2): max |(qR).(kR) - q.k| / (|q||k|) = {rot_err:.2e} over every FA layer")
    print(f"\n### replay: {os.path.basename(a.capture)} ({tcap} positions, {len(fa)} FA layers, "
          f"{a.nq} queries per bucket from the last {a.window} positions; tiled = the capture repeated)")
    print("| depth | scheme | cos min (layer) | cos p0.1 | cos mean | max abs | rel L2 mean | rel L2 max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for lab, _, _ in buckets:
        for c in comps:
            cos = torch.cat(acc[(lab, c)]["cos"])
            mx = torch.cat(acc[(lab, c)]["maxabs"])
            rel = torch.cat(acc[(lab, c)]["rel"])
            row = {"cos_min": float(cos.min()), "cos_p001": float(torch.quantile(cos, 0.001)),
                   "cos_mean": float(cos.mean()), "maxabs": float(mx.max()),
                   "rel_mean": float(rel.mean()), "rel_max": float(rel.max()), "worst_layer": worst[(lab, c)][0]}
            summary[f"{lab}/{c}"] = row
            print(f"| {lab} | {c} | {row['cos_min']:.7f} (L{row['worst_layer']}) | {row['cos_p001']:.7f} | "
                  f"{row['cos_mean']:.8f} | {row['maxabs']:.2e} | {row['rel_mean']:.2e} | {row['rel_max']:.2e} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fo:
            json.dump({"capture": a.capture, "rotation_check": rot_err, "rows": summary}, fo, indent=1)
    print(f"replay wall {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- tokcheck
def cmd_tokcheck(a) -> None:
    import unicodedata
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.snapshot, "tokenizer.json"))
    for path in a.ids:
        ids = _read_ids(path)
        text = tok.decode(ids, skip_special_tokens=False)
        back = tok.encode(text, add_special_tokens=False).ids
        marks = sum(1 for ch in text if unicodedata.category(ch).startswith("M"))
        special = sum(1 for i in ids if i >= 248077)
        first = next((i for i, (x, y) in enumerate(zip(ids, back)) if x != y), None)
        print(f"{path}: {len(ids)} ids, round trip {'IDENTICAL' if back == ids else 'DIFFERS'} "
              f"(re-encoded {len(back)}, first difference at {first}), combining marks {marks}, "
              f"ids >= 248077: {special}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--ids", required=True)
    r.add_argument("--n", type=int, default=0, help="use the first N ids (0: all)")
    r.add_argument("--cont", help="a golden .safetensors: append its `tokens`[:--gen] (teacher-forced)")
    r.add_argument("--cont-ids", help="a whitespace ids file to append [:--gen] instead (teacher-forced;"
                   " no golden logits to compare against)")
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--variants", default=",".join(E2E_VARIANTS))
    r.add_argument("--capture", help="write q/k/v of the bf16 element per FA layer here")
    r.add_argument("--vocab-used", type=int, default=0)
    r.add_argument("--fp32-matmul", action="store_true", help="bf16 matmuls as fp32 sgemm + one bf16 rounding")
    r.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("capture")
    p.add_argument("--depths", default="2048,4096,8192")
    p.add_argument("--tiles", default="16384,32768")
    p.add_argument("--nq", type=int, default=32)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--scale", type=float, default=1 / 16)
    p.add_argument("--out")
    s = sub.add_parser("summary")
    s.add_argument("runs", nargs="+")
    t = sub.add_parser("tokcheck")
    t.add_argument("snapshot")
    t.add_argument("ids", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "replay":
        cmd_replay(a)
    elif a.cmd == "summary":
        summarize(a.runs)
    else:
        cmd_tokcheck(a)


if __name__ == "__main__":
    main()
