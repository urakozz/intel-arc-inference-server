#!/usr/bin/env python3
"""Plan 2026-11-12 (Ornith MoE prefill) Task 1, numerics: the expert GEMM forms against f32, on REAL activations.

One f32 prompt forward of the Ornith reference (tools/oracle/ornith_ref.py, the f32 judge's model: the int4
checkpoint dequantised by the repo's rule, f32 math) captures, at the chosen layers, the MoE block's input rows
x [T][2048] (the post-attention norm's output), the reference router's top-8 ids and weights and the shared
expert gate's logit. Then, per layer, every routed (token, expert) pair and the shared expert are recomputed in
each form below, and three tensors are compared with the f32 form ("ref"):

  h      gate||up -> SiLU(gate) * up, the down projection's input            [pairs][512]
  down   the down projection alone, on ref's h                                [pairs][2048]
  block  the MoE block output, sum_k w_k down_k + sigmoid(sg) shared          [T][2048]

Forms (x is the bf16 row the engine reads; W_b the checkpoint dequant bf16(rne((q - 8) s)), the reference's):
  ref     x @ W_b in f32, h = silu(g) u in f32
  l0      the engine's bf16 path: f32 accumulation of bf16 x bf16, then h = rne(rne(silu(rne g)) rne u), y = rne(.)
  i8      today's l0-int8: gate||up through int8_sim.fq_x / fq_w (1024-block Hadamard, per-token x, per-column W
          int8), down as l0
  w4a8t   W4A8, unrotated: x per-token int8 (s = max|x| / 127, rint), W the exact int4 (q - 8) s; down's input
          h per-row int8 the same way
  w4a8g   as w4a8t with one activation scale per token and 64-k group
  w4a8r   the rotation folded into the weights OFFLINE (numerics only, no engine form): W' = RTN int4 g64 of
          (R^T W_b) per output column (loader::rtn_int4_g64's rule), R = D blockdiag(H_b) / sqrt(b) (b = 1024 for
          gate||up's 2048 inputs, 512 for down's 512, D the engine's mt19937(K) signs), x' = x R per-token int8
  w4a8rg  as w4a8r with per-token-and-group activation scales
Each W4A8 form is also reported as "<form>+bf16dn": its gate||up with l0's down (down kept W4A16).

Metrics per tensor: relative L2 (sqrt(sum err^2 / sum ref^2)), the worst and the mean row cosine, and the ratio of
the form's relative L2 to l0's (l0 = the engine's distance to f32 today). Bar of the tensor study (this project's
tensor bar, tests/golden/golden_common.h): worst row cosine >= 0.999. The engine gate (the f32 judge, Task 5) is
separate.

    python3 tools/probe/moe_w4a8_accuracy.py <snapshot> --prompt tests/golden/prompts/prose.ids
        [--layers 1,8,15,22,29,36,39] [--out <json>]
Run in the reference container (tools/oracle/run_in_container.sh), under /tmp/b70-cpu-model.lock.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE = os.path.join(os.path.dirname(_HERE), "oracle")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oref = _load(os.path.join(_ORACLE, "ornith_ref.py"), "ornith_ref_mod")
i8 = oref._int8
LP = oref.LP


def rne_bf16(t):
    return t.to(torch.bfloat16).float()


def silu(t):
    return t / (1.0 + torch.exp(-t))


def raw_int4(where, base):
    """(q int [K][N] in 0..15, s f32 [K/64][N]) of a GPTQ linear (k = 8 r + i in nibble i)."""
    qw = where[base + ".qweight"].get_tensor(base + ".qweight").to(torch.int64) & 0xFFFFFFFF
    sc = where[base + ".scales"].get_tensor(base + ".scales").float()
    K = qw.shape[0] * 8
    q = torch.empty(K, qw.shape[1], dtype=torch.int64)
    for i in range(8):
        q[i::8] = (qw >> (4 * i)) & 0xF
    return q, sc


def exact_w(q, s):
    """[N][K] f32 (q - 8) s, the exact int4 value (not bf16-rounded)."""
    return ((q.float() - 8.0) * s.repeat_interleave(64, 0)).t().contiguous()


def bf16_w(q, s):
    return rne_bf16(exact_w(q, s))


_R = {}


def rot(K):
    """R = D blockdiag(H_b) / sqrt(b) [K][K], b = min(1024, K), D the engine's int8 signs for K."""
    if K not in _R:
        b = min(1024, K)
        H = i8.hadamard(b).float() / (b ** 0.5)
        Rm = torch.block_diag(*([H] * (K // b)))
        d = i8.int8_signs(K).float()
        _R[K] = d.unsqueeze(1) * Rm
    return _R[K]


def rtn_g64(w):
    """loader::rtn_int4_g64's rule on an f32 [N][K]: per column n and 64-k group s16 = f16(2 amax / 15),
    q = clamp(rint(w / s) + 8, 0, 15); returns the dequantised (q - 8) s, f32."""
    N, K = w.shape
    wf = w.reshape(N, K // 64, 64)
    amax = wf.abs().amax(-1, keepdim=True)
    s = (2.0 * amax / 15.0).to(torch.float16).float()
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))) + 8.0,
                    torch.full_like(wf, 8.0)).clamp(0.0, 15.0)
    return ((q - 8.0) * s).reshape(N, K)


def q8(x, group):
    """Per-row (group=None) or per-row-and-64-group symmetric int8 of x [M][K], de-quantised."""
    M, K = x.shape
    g = K if group is None else group
    v = x.reshape(M, K // g, g)
    s = v.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.round(v / s).clamp(-127, 127) * s).reshape(M, K)


class Acc:
    def __init__(self):
        self.err = 0.0
        self.ref = 0.0
        self.worst = 1.0
        self.cos_sum = 0.0
        self.rows = 0

    def add(self, ref, got):
        e = (got - ref).double()
        r = ref.double()
        self.err += float((e * e).sum())
        self.ref += float((r * r).sum())
        num = (got.double() * r).sum(-1)
        den = got.double().norm(dim=-1) * r.norm(dim=-1)
        cos = torch.where(den > 0, num / den.clamp_min(1e-300), torch.ones_like(num))
        self.worst = min(self.worst, float(cos.min()))
        self.cos_sum += float(cos.sum())
        self.rows += cos.numel()

    def out(self):
        return {"rel_l2": (self.err / self.ref) ** 0.5 if self.ref else 0.0, "worst_cos": self.worst,
                "mean_cos": self.cos_sum / max(self.rows, 1), "rows": self.rows}


FORMS = ["l0", "i8", "w4a8t", "w4a8g", "w4a8r", "w4a8rg"]


def gate_up(form, x, Wgu_b, Wgu_x, Wgu_r, I):
    """-> (g, u) f32 [M][I] of `form` (x [M][2048] bf16 values in f32)."""
    if form in ("ref", "l0"):
        gu = x @ Wgu_b.t()
    elif form == "i8":
        gu = i8.fq_x(x, True) @ i8.fq_w(Wgu_b, True).t()
    elif form in ("w4a8t", "w4a8g"):
        gu = q8(x, None if form == "w4a8t" else 64) @ Wgu_x.t()
    else:
        R = rot(x.shape[1])
        gu = q8(x @ R, None if form == "w4a8r" else 64) @ Wgu_r.t()
    return gu[:, :I], gu[:, I:]


def h_of(form, g, u):
    if form == "ref":
        return silu(g) * u
    return rne_bf16(rne_bf16(silu(rne_bf16(g))) * rne_bf16(u))


def down(form, h, Wd_b, Wd_x, Wd_r):
    if form == "ref":
        return h @ Wd_b.t()
    if form in ("l0", "i8"):
        return rne_bf16(h @ Wd_b.t())
    if form in ("w4a8t", "w4a8g"):
        return rne_bf16(q8(h, None if form == "w4a8t" else 64) @ Wd_x.t())
    R = rot(h.shape[1])
    return rne_bf16(q8(h @ R, None if form == "w4a8r" else 64) @ Wd_r.t())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", default="1,8,15,22,29,36,39")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(v) for v in a.layers.split(",")]
    tc = oref.text_config(a.snapshot, "grouped_mm", torch.float32)
    ids = [int(v) for v in open(a.prompt).read().split()]
    model, pf, lazy = oref.build_streamed(a.snapshot, tc)
    where = lazy.where
    E, I, H = tc.num_experts, tc.moe_intermediate_size, tc.hidden_size
    cap = {}

    def pre_mlp(i):
        def fn(_m, args):
            cap.setdefault(i, {})["x"] = args[0].detach()[0].float().clone()
        return fn

    def pre_experts(i):
        def fn(_m, args):
            cap[i]["ids"] = args[1].detach().reshape(-1, args[1].shape[-1]).clone()
            cap[i]["w"] = args[2].detach().reshape(-1, args[2].shape[-1]).float().clone()
        return fn

    def post_sg(i):
        def fn(_m, _a, out):
            cap[i]["sg"] = out.detach().reshape(-1).float().clone()
        return fn
    hs = []
    for i in layers:
        mlp = model.model.layers[i].mlp
        hs += [mlp.register_forward_pre_hook(pre_mlp(i)), mlp.experts.register_forward_pre_hook(pre_experts(i)),
               mlp.shared_expert_gate.register_forward_hook(post_sg(i))]
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1)
    for h in hs:
        h.remove()
    print(f"forward: {len(ids)} ids, {time.time() - t0:.0f}s; layers {layers}", flush=True)

    res = {}
    allf = ["ref"] + FORMS
    for i in layers:
        t1 = time.time()
        c = cap[i]
        x = rne_bf16(c["x"])                       # the engine's bf16 rows
        T = x.shape[0]
        acc_h = {f: Acc() for f in FORMS}
        acc_d = {f: Acc() for f in FORMS}
        blocks = {f: torch.zeros(T, H) for f in allf + [f + "+bf16dn" for f in FORMS if f.startswith("w4a8")]}
        experts = [(e, f"{LP}{i}.mlp.experts.{e}.") for e in range(E)] + [(E, f"{LP}{i}.mlp.shared_expert.")]
        for e, base in experts:
            if e < E:
                tok, slot = (c["ids"] == e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                wt = c["w"][tok, slot]
            else:
                tok = torch.arange(T)
                wt = torch.sigmoid(c["sg"])
            qg, sg_ = raw_int4(where, base + "gate_proj")
            qu, su = raw_int4(where, base + "up_proj")
            qd, sd = raw_int4(where, base + "down_proj")
            Wgu_x = torch.cat([exact_w(qg, sg_), exact_w(qu, su)], 0)       # [2I][H]
            Wgu_b = rne_bf16(Wgu_x)
            Wd_x = exact_w(qd, sd)                                            # [H][I]
            Wd_b = rne_bf16(Wd_x)
            Wgu_r = rtn_g64(Wgu_b @ rot(H))
            Wd_r = rtn_g64(Wd_b @ rot(I))
            xe = x[tok]
            g, u = gate_up("ref", xe, Wgu_b, Wgu_x, Wgu_r, I)
            h_ref = h_of("ref", g, u)
            y_ref = down("ref", h_ref, Wd_b, Wd_x, Wd_r)
            blocks["ref"].index_add_(0, tok, y_ref * wt.unsqueeze(1))
            for f in FORMS:
                g, u = gate_up(f, xe, Wgu_b, Wgu_x, Wgu_r, I)
                hf = h_of(f, g, u)
                acc_h[f].add(h_ref, hf)
                acc_d[f].add(y_ref, down(f, h_ref, Wd_b, Wd_x, Wd_r))
                blocks[f].index_add_(0, tok, down(f, hf, Wd_b, Wd_x, Wd_r) * wt.unsqueeze(1))
                if f.startswith("w4a8"):
                    blocks[f + "+bf16dn"].index_add_(0, tok, down("l0", hf, Wd_b, Wd_x, Wd_r) * wt.unsqueeze(1))
        out = {}
        for f in blocks:
            if f == "ref":
                continue
            ab = Acc()
            ab.add(blocks["ref"], blocks[f])
            out[f] = {"block": ab.out()}
            if f in acc_h:
                out[f]["h"] = acc_h[f].out()
                out[f]["down"] = acc_d[f].out()
        for f in out:
            for t in out[f]:
                base = out["l0"][t]["rel_l2"] if t in out["l0"] else None
                if base:
                    out[f][t]["x_l0"] = out[f][t]["rel_l2"] / base
        res[i] = out
        print(f"layer {i} ({time.time() - t1:.0f}s):", flush=True)
        for f in out:
            print("  %-14s " % f + "  ".join(f"{t} relL2 {v['rel_l2'] * 100:.3f}% worst {v['worst_cos']:.6f} "
                                            f"x{v.get('x_l0', 1):.1f}" for t, v in out[f].items()), flush=True)
    # summary over layers: worst of worst-cos, mean of rel L2
    print("\nsummary (mean relL2 over layers / worst row cos over layers / mean x l0):")
    for f in res[layers[0]]:
        for t in res[layers[0]][f]:
            rl = [res[i][f][t]["rel_l2"] for i in layers]
            wc = [res[i][f][t]["worst_cos"] for i in layers]
            xl = [res[i][f][t].get("x_l0", 1.0) for i in layers]
            print(f"  {f:14s} {t:5s} relL2 {sum(rl) / len(rl) * 100:.3f}%  worst {min(wc):.6f}  "
                  f"x l0 {sum(xl) / len(xl):.1f}  {'PASS' if min(wc) >= 0.999 else 'fail'} (0.999)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"prompt": a.prompt, "n": len(ids), "layers": layers, "results": res}, f, indent=1)
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
