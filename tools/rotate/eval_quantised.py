#!/usr/bin/env python3
"""Evaluate an int4 GPTQ checkpoint of the ROTATED model (tools/rotate), any group size.

    eval_quantised.py logits   <bf16 snapshot> <quantised snapshot> --prompt <ids> [--layers 0]
    eval_quantised.py actquant <quantised snapshot> --prompt <ids> [--tokens 512] [--layers 0]
    eval_quantised.py rtn      <bf16 snapshot> <bf16 snapshot to quantise> --prompt <ids> --group <-1|64|128>
    eval_quantised.py sim      <bf16 snapshot> <quantised snapshot> --prompt <ids> [--variants none,a8,a8w8,h8]
                               [--group 256] [--per-class h8,h4,h4p2] [--vocab-used N] [--out f.json]

logits    the quantised model against the ORIGINAL bf16 model, the metrics of
          check_rotation.py. The yardstick on the 42-id prose prompt is the
          unrotated int4 g64 gate checkpoint: rel L2 7.77 %, worst cos 0.992587,
          argmax 38/42, top-5 0.905 (commit 046f289).
actquant  one CPU forward of the quantised model with a hook on every linear of
          every decoder layer. For each, against y = x W^T with the model's own
          weights, it measures int8 activations, per-token symmetric:
            A8           Q8(x) W^T. The whole error of the W4A8 path for a
                         residual-stream reader, whose x arrives rotated.
            H+A8+W8      Q8(x R) Q8pc(W R)^T with R the 1024-block Hadamard
                         over K: the in-engine rotation, for the linears whose
                         input the checkpoint cannot rotate (down_proj,
                         o_proj, out_proj).
          Bar: worst-row cosine 0.999 (tests/golden/golden_common.h).
rtn       a control with no tuning: round-to-nearest symmetric int4 (the GPTQ
          grid, q in [-8, 7], scale = max|w| / 7.5) at --group, applied in memory
          to every decoder linear AutoRound quantises (all but in_proj_a/b) of
          the second snapshot, logits against the first. Separates what the
          group size costs from what the rotation or the tuning does.
sim       the end-to-end cost of the int8 paths, simulated: every decoder
          linear AutoRound quantised (all but in_proj_a/b) gets its forward
          replaced, fp32 inside, output back in the model dtype:
            none   x W^T                      the fp32-matmul control
            a8     Q8(x) W^T                  per-token int8 activations
            a8w8   Q8(x) Q8pc(W)^T            + per-channel int8 weights (the
                                              unrotated fast path, 1.70x / 1.85x)
            h8     Q8(xR) Q8pc(WR)^T          runtime 1024-block Hadamard on
                                              both sides (1.37x / 1.52x)
          and the W4A4 probe's (plan 2026-10-09-w4a4-probe, docs/07 §3):
            h4     Q4g(xR) Q4g(WR)^T          h8's rotation exactly (same signs,
                                              R.rot on both sides), then int4
                                              symmetric on BOTH operands, q in
                                              [-8, 7], s = max|v| / 7.5 per group
                                              of --group (default 256) along K,
                                              groups at K offsets 0, g, 2g, ...
                                              for x and W alike; fp32 product
            h4p2   the same, each scale rounded up to 2^ceil(log2 s) (the
                   shift-rescale kernel's numerics)
            w4a16  the module's own forward: the checkpoint's int4 g64 weights
                   (dequantised to bf16) times bf16 activations in bf16 - the
                   shipped model, bitwise the `logits` mode's
          W in every variant is the quantised checkpoint's dequantised weight
          (so h8 / h4 re-quantise the g64 grid's values, as the engine would).
          The model is loaded once; each variant is one forward, logits
          against one bf16 reference: CR.compare's table (rel L2, worst
          position cosine, argmax, top-5, max |log-softmax| difference) plus
          the unfiltered KL(p_bf16 || p_variant) at T = 1 over ids < --vocab-used
          (mean / p99 over positions; lm_head_probe.py's klfull). Per linear,
          in each full pass, the replaced linear's own error against x W^T on
          the same input (rel L2, worst-row cosine), summarised per class:
          attn_qkv (q/k/v_proj), attn_o (o_proj), gdn_in (in_proj_qkv/z),
          gdn_out (out_proj), gate_up, down. --per-class V,...: for each
          variant V and each class, one more forward with V on that class
          only (every other linear `none`), the same end-to-end metrics - an
          end-to-end pass can hide one bad class. A summary table closes.

Per-channel checkpoints (group_size -1) store scales [1, N]; the group is read
from each tensor's scale shape, so g64 checkpoints work too. The dequant is the
oracle's own (tools/oracle/dequant.py). Run in the reference container.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE = os.path.join(os.path.dirname(_HERE), "oracle")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) not in (_HERE, _ORACLE)]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("b70_rotation", os.path.join(_HERE, "rotation.py"))
CR = _load("b70_check_rotation", os.path.join(_HERE, "check_rotation.py"))
DQ = _load("b70_dequant", os.path.join(_ORACLE, "dequant.py"))

STREAM_READERS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "q_proj", "k_proj",
                  "v_proj", "gate_proj", "up_proj")
OTHER_INPUTS = ("down_proj", "o_proj", "out_proj")
BAR = 0.999


def load_quantised(snapshot: str, layers: int) -> dict[str, torch.Tensor]:
    """Checkpoint-named bf16 state dict, int4 tensors dequantised at their own group size."""
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    q = cfg.get("quantization_config") or {}
    if q.get("bits") != 4 or not q.get("sym") or q.get("desc_act"):
        sys.exit(f"FATAL: need int4 symmetric without desc_act, config says {q}")
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        wmap = json.load(f)["weight_map"]
    if any(k.endswith(".g_idx") for k in wmap) and "desc_act" not in q:
        sys.exit("FATAL: g_idx present without an explicit desc_act: false")
    handles, sd, groups = {}, {}, {}

    def get(k):
        p = os.path.join(snapshot, wmap[k])
        if p not in handles:
            handles[p] = safe_open(p, framework="pt", device="cpu")
        return handles[p].get_tensor(k)

    for k in sorted(wmap):
        if k.startswith(("model.visual.", "mtp.")) or k.endswith((".scales", ".qzeros", ".g_idx")):
            continue
        if layers and k.startswith(f"{R.PREFIX}layers."):
            if int(k[len(f"{R.PREFIX}layers."):].split(".")[0]) >= layers:
                continue
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            qw, sc = get(k), get(base + ".scales")
            kdim = qw.shape[0] * 8
            gs = kdim // sc.shape[0]
            if gs * sc.shape[0] != kdim:
                sys.exit(f"FATAL: {base}: K {kdim} not a multiple of {sc.shape[0]} scale rows")
            if base + ".g_idx" in wmap:   # GPTQ ships a trivial one with desc_act false: prove it
                gi = get(base + ".g_idx")
                if not bool((gi == torch.arange(kdim, dtype=gi.dtype) // gs).all()):
                    sys.exit(f"FATAL: {base}: g_idx is not k // {gs}; the checkpoint is permuted")
            zeros = get(base + ".qzeros")
            if not bool((zeros == 0x77777777).all()):
                sys.exit(f"FATAL: {base}: qzeros are not the symmetric v1 0x77777777")
            groups[gs] = groups.get(gs, 0) + 1
            sd[base + ".weight"] = DQ.dequant_gptq(qw, sc, gs, 8192 if qw.shape[1] > 65536 else 0).t().contiguous()
        else:
            sd[k] = get(k).to(torch.bfloat16)
    print(f"dequantised: {groups} (group size: tensors); {len(sd)} tensors")
    return sd


def q8_token(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(dim=1, keepdim=True) / 127.0
    s[s == 0] = 1.0
    return torch.clamp(torch.round(x / s), -127, 127) * s


def q8_channel(w: torch.Tensor) -> torch.Tensor:
    """w [N, K]: one scale per output row."""
    s = w.abs().amax(dim=1, keepdim=True) / 127.0
    s[s == 0] = 1.0
    return torch.clamp(torch.round(w / s), -128, 127) * s


def rtn_int4(w: torch.Tensor, group: int) -> torch.Tensor:
    """w [N, K] -> symmetric int4 RTN along K in groups of `group` (-1 = all of K)."""
    n, k = w.shape
    g = k if group <= 0 else group
    wf = w.to(torch.float32).reshape(n, k // g, g)
    s = wf.abs().amax(dim=2, keepdim=True) / 7.5
    s[s == 0] = 1.0
    return (torch.clamp(torch.round(wf / s), -8, 7) * s).reshape(n, k).to(w.dtype)


# ---- the W4A4 probe (plan 2026-10-09-w4a4-probe) ---------------------------------------------
LINEAR_CLASSES = (   # (class, layer-relative module names); o / out_proj / down apart (Review Focus 5)
    ("attn_qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("attn_o", ("self_attn.o_proj",)),
    ("gdn_in", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")),
    ("gdn_out", ("linear_attn.out_proj",)),
    ("gate_up", ("mlp.gate_proj", "mlp.up_proj")),
    ("down", ("mlp.down_proj",)),
)
CLASS_NAMES = tuple(c for c, _ in LINEAR_CLASSES)
FP32_VARIANTS = ("none", "a8", "a8w8", "h8", "h4", "h4p2")
VARIANTS = FP32_VARIANTS + ("w4a16",)
ROTATED = ("h8", "h4", "h4p2")
INT4_ACT = ("h4", "h4p2")


def linear_class(name: str) -> str:
    """'L3.mlp.down_proj' -> 'down'; 'other' for a linear in no class."""
    for c, mods in LINEAR_CLASSES:
        if name.endswith(tuple("." + m for m in mods)) or name in mods:
            return c
    return "other"


def check_group(k: int, group: int, name: str) -> int:
    """The group along K (group <= 0: all of K), or ValueError naming the linear."""
    if group <= 0:
        return k
    if k % group:
        raise ValueError(f"{name}: K {k} is not a multiple of the int4 group {group} - the groups must "
                         f"tile K from offset 0 on both operands; refused, not padded")
    return group


def pow2_ceil(s: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2 s) per element, exactly (frexp, not log2: log2 of a value one ulp above a
    power of two can round down to the integer). s = m 2^e with m in [0.5, 1): m == 0.5 is
    2^(e-1) itself, anything else rounds up to 2^e. Zeros stay zero."""
    m, e = torch.frexp(s)
    p = torch.ldexp(torch.ones_like(s), e - (m == 0.5).to(e.dtype))
    return torch.where(s > 0, p, s)


def quant_int4_groups(x: torch.Tensor, g: int = 256, pow2: bool = False, name: str = "tensor"):
    """x [R, K] -> (q int8 [R, K] in [-8, 7], s fp32 [R, K/g]).

    One symmetric scale per row per g elements along K, the groups at K offsets 0, g, 2g, ...
    s = max|v| / 7.5 (rtn_int4's RTN rule), q = clamp(round(v / s), -8, 7); pow2 rounds each s
    up to a power of two first. An all-zero group gets s = 1, q = 0. g <= 0: one group of K."""
    r, k = x.shape
    g = check_group(k, g, name)
    xg = x.to(torch.float32).reshape(r, k // g, g)
    s = xg.abs().amax(dim=2, keepdim=True) / 7.5
    if pow2:
        s = pow2_ceil(s)
    s[s == 0] = 1.0
    q = torch.clamp(torch.round(xg / s), -8, 7)
    return q.to(torch.int8).reshape(r, k), s.reshape(r, k // g)


def fake_int4_groups(x: torch.Tensor, g: int = 256, pow2: bool = False, name: str = "tensor") -> torch.Tensor:
    """quant_int4_groups dequantised: q * s in fp32, [R, K]."""
    q, s = quant_int4_groups(x, g, pow2, name)
    r, k = q.shape
    return (q.to(torch.float32).reshape(r, s.shape[1], -1) * s.unsqueeze(2)).reshape(r, k)


def w4a4_matmul(x: torch.Tensor, w: torch.Tensor, g: int = 256, pow2: bool = False, rot=None,
                name: str = "tensor") -> torch.Tensor:
    """Q4g(rot(x)) Q4g(rot(w))^T in fp32: x [T, K], w [N, K], the same rotation and the same K
    groups on both operands, so every group's partial sum has one x scale and one w scale - what
    an int4 x int4 kernel with a rescale every g along K computes, up to float association."""
    if rot is not None:
        x, w = rot(x), rot(w)
    return fake_int4_groups(x, g, pow2, name) @ fake_int4_groups(w, g, pow2, name).t()


def rotation_signs(k: int) -> torch.Tensor:
    """The runtime rotation's signs for depth K: h8's (R.signs(K, R.SEED + K))."""
    return R.signs(k, R.SEED + k)


def variant_matmul(v: str, x32: torch.Tensor, w32: torch.Tensor, d, group: int, name: str) -> torch.Tensor:
    """One fp32 variant's x W^T (no bias). d: the rotation signs for the rotated variants."""
    if v == "none":
        return x32 @ w32.t()
    if v == "a8":
        return q8_token(x32) @ w32.t()
    if v == "a8w8":
        return q8_token(x32) @ q8_channel(w32).t()
    if v == "h8":
        return q8_token(R.rot(x32, d)) @ q8_channel(R.rot(w32, d)).t()
    if v in INT4_ACT:
        return w4a4_matmul(x32, w32, group, v == "h4p2", lambda t: R.rot(t, d), name)
    raise ValueError(v)


def kl_unfiltered(ref: torch.Tensor, q: torch.Tensor, vocab_used: int, rows: int = 32) -> torch.Tensor:
    """KL(p_ref || p_q) per position at T = 1 over ids < vocab_used, float64 (lm_head_probe's klfull)."""
    out = []
    for a in range(0, ref.shape[0], rows):
        lp = torch.log_softmax(ref[a:a + rows, :vocab_used].double(), -1)
        lq = torch.log_softmax(q[a:a + rows, :vocab_used].double(), -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out)


def e2e_metrics(a: torch.Tensor, b: torch.Tensor, vocab_used: int) -> dict:
    """a against the bf16 reference b: CR.compare's numbers (full width) plus the unfiltered KL."""
    a64, b64 = a.double(), b.double()
    ta, tb = a.topk(5, dim=1).indices, b.topk(5, dim=1).indices
    kl = kl_unfiltered(b, a, vocab_used)
    return {"n": a.shape[0], "rel": ((a64 - b64).norm() / b64.norm()).item(),
            "cos_min": torch.nn.functional.cosine_similarity(a64, b64, dim=1).min().item(),
            "top1": int((a.argmax(1) == b.argmax(1)).sum().item()),
            "top5": sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)) / (5 * a.shape[0]),
            "kl_mean": kl.mean().item(), "kl_p99": torch.quantile(kl, 0.99).item(), "kl_max": kl.max().item()}


def class_table(stats: list) -> None:
    """Per-linear errors of one full pass, summarised per class."""
    print("\n| class | linears | median rel L2 | max rel L2 | worst-row cos | rows < 0.999 |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in CLASS_NAMES + ("other",):
        rs = [r for r in stats if r["class"] == c]
        if not rs:
            continue
        rels = sorted(r["rel"] for r in rs)
        print(f"| {c} | {len(rs)} | {rels[len(rels) // 2] * 100:.3f} % | {rels[-1] * 100:.3f} % | "
              f"{min(r['cos'] for r in rs):.6f} | {sum(1 for r in rs if r['cos'] < BAR)} |")
    worst = sorted(stats, key=lambda r: r["cos"])[:4]
    print("worst 4: " + "; ".join(f"{r['name']} cos {r['cos']:.6f} rel {r['rel'] * 100:.3f} %" for r in worst),
          flush=True)


def metrics(y: torch.Tensor, ref: torch.Tensor):
    rel = ((y - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(y.double(), ref.double(), dim=1).min().item()
    return rel, cos


def actquant(args) -> None:
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()][: args.tokens]
    tc = CR.text_config(args.snapshot, args.layers)
    with open(os.path.join(args.snapshot, "config.json"), encoding="utf-8") as f:
        rotated = "b70_rotation" in json.load(f)
    print(f"checkpoint stream: {'ROTATED (b70_rotation present)' if rotated else 'NOT rotated'}")
    sd = CR.to_model_names(load_quantised(args.snapshot, args.layers))
    with torch.device("meta"):
        model = CR.Qwen3_5ForCausalLM(tc)
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    del sd
    rows = []
    signs = {}

    def hook(name, lin):
        kind = name.rsplit(".", 1)[-1]

        def fn(_mod, inp):
            x = inp[0][0].to(torch.float32)                    # [T, K]
            w = lin.weight.to(torch.float32)                   # [N, K]
            ref = x @ w.t()
            r = {"name": name, "kind": kind, "K": x.shape[1]}
            r["A8"] = metrics(q8_token(x) @ w.t(), ref)
            if kind in OTHER_INPUTS and x.shape[1] % R.BLOCK == 0:
                d = signs.setdefault(x.shape[1], R.signs(x.shape[1], R.SEED + x.shape[1]))
                xr, wr = R.rot(x, d), R.rot(w, d)
                r["H+A8+W8"] = metrics(q8_token(xr) @ q8_channel(wr).t(), ref)
            rows.append(r)
        return fn

    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                m.register_forward_pre_hook(hook(f"L{i}.{n}", m))
    t = time.time()
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    print(f"forward + {len(rows)} hooked linears: {len(ids)} tokens, {time.time() - t:.0f}s\n")

    print("| linear | input | layers | A8 worst cos | A8 median rel L2 | A8 fails | "
          "H+A8+W8 worst cos | H+A8+W8 median rel L2 | H+A8+W8 fails |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for kind in STREAM_READERS + OTHER_INPUTS:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue

        def col(key):
            v = [r[key] for r in rs if key in r]
            if not v:
                return "-", "-", "-"
            rels = sorted(x[0] for x in v)
            return (f"{min(x[1] for x in v):.6f}", f"{rels[len(rels) // 2] * 100:.3f} %",
                    f"{sum(1 for x in v if x[1] < BAR)}")
        a, h = col("A8"), col("H+A8+W8")
        src = ("rotated stream" if rotated else "stream, NOT rotated") if kind in STREAM_READERS \
            else "not rotatable offline"
        print(f"| {kind} | {src} | {len(rs)} | {a[0]} | {a[1]} | {a[2]} | {h[0]} | {h[1]} | {h[2]} |")
    worst = sorted(rows, key=lambda r: r.get("H+A8+W8", r["A8"])[1])[:8]
    print("\nworst 8 (by the column the engine would use):")
    for r in worst:
        best = r.get("H+A8+W8", r["A8"])
        print(f"  {r['name']}: cos {best[1]:.6f}, rel L2 {best[0] * 100:.3f} %"
              f"{'  (A8 ' + format(r['A8'][1], '.6f') + ')' if 'H+A8+W8' in r else ''}")


def sim(args, keep_logits: bool = False) -> dict:
    """The `sim` mode. Returns {"base": bf16 logits, "passes": [{variant, only, metrics,
    linear_stats, logits (keep_logits only)}]}."""
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    variants = [v for v in args.variants.split(",") if v]
    per_class = [v for v in (getattr(args, "per_class", "") or "").split(",") if v]
    for v in variants + per_class:
        if v not in VARIANTS:
            sys.exit(f"FATAL: unknown variant {v!r}; known: {','.join(VARIANTS)}")
    group = 256 if getattr(args, "group", None) is None else args.group
    vocab_used = getattr(args, "vocab_used", 0) or 0
    linear_stats = getattr(args, "linear_stats", True)
    tc = CR.text_config(args.snapshot, args.layers)
    base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
    vocab_used = vocab_used or base.shape[1]
    sd = CR.to_model_names(load_quantised(args.quantised, args.layers))
    with torch.device("meta"):
        model = CR.Qwen3_5ForCausalLM(tc)
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    del sd
    state = {"v": "none", "only": None, "stats": None}
    signs = {}
    targets = []
    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) and not n.endswith(("in_proj_a", "in_proj_b")):
                targets.append((f"L{i}.{n}", m))
    if any(v in INT4_ACT for v in variants + per_class):
        for name, m in targets:          # refuse before any forward, by name (Review Focus 1)
            k = m.weight.shape[1]
            try:
                check_group(k, group, name)
            except ValueError as e:
                sys.exit(f"FATAL: {e}")
            if k % R.BLOCK:
                sys.exit(f"FATAL: {name}: K {k} is not a multiple of the rotation's {R.BLOCK}-block")

    def make_forward(name, m):
        cls = linear_class(name)
        orig = m.forward

        def fwd(x):
            v = state["v"]
            if state["only"] is not None and cls != state["only"]:
                v = "none"
            shp = x.shape
            if v == "w4a16":                 # the module's own bf16 forward: the shipped model
                y = orig(x)
                if state["stats"] is not None:
                    x32 = x.reshape(-1, shp[-1]).to(torch.float32)
                    ref = x32 @ m.weight.to(torch.float32).t()
                    if m.bias is not None:
                        ref = ref + m.bias.to(torch.float32)
                    rel, cos = metrics(y.reshape(-1, y.shape[-1]).to(torch.float32), ref)
                    state["stats"].append({"name": name, "class": cls, "rel": rel, "cos": cos})
                return y
            x32 = x.reshape(-1, shp[-1]).to(torch.float32)
            w32 = m.weight.to(torch.float32)
            k = shp[-1]
            d = signs.setdefault(k, rotation_signs(k)) if v in ROTATED else None
            y = variant_matmul(v, x32, w32, d, group, name)
            if m.bias is not None:
                y = y + m.bias.to(torch.float32)
            if state["stats"] is not None and v != "none":
                ref = x32 @ w32.t()
                if m.bias is not None:
                    ref = ref + m.bias.to(torch.float32)
                rel, cos = metrics(y, ref)
                state["stats"].append({"name": name, "class": cls, "rel": rel, "cos": cos})
            return y.to(x.dtype).reshape(*shp[:-1], -1)
        return fwd

    for name, m in targets:
        m.forward = make_forward(name, m)
    present = [c for c in CLASS_NAMES if any(linear_class(n) == c for n, _ in targets)]
    print(f"sim: {len(targets)} linears replaced; {len(ids)} ids; group {group} (h4 / h4p2); "
          f"KL over {vocab_used} ids; classes {present}")
    passes = []
    qname = os.path.basename(args.quantised.rstrip('/'))

    def one(v, only):
        state["v"], state["only"] = v, only
        state["stats"] = [] if (only is None and linear_stats and v != "none") else None
        t = time.time()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0].to(torch.float32)
        tag = v if only is None else f"{v} on {only} only (others none)"
        print(f"  {tag}: forward {time.time() - t:.0f}s", flush=True)
        CR.compare(logits, base, f"sim {tag}: {qname} vs original bf16")
        mt = e2e_metrics(logits, base, vocab_used)
        print(f"| unfiltered KL(bf16 || {v}) at T 1, mean / p99 / max | {mt['kl_mean']:.3e} / "
              f"{mt['kl_p99']:.3e} / {mt['kl_max']:.3e} |", flush=True)
        if state["stats"]:
            class_table(state["stats"])
        p = {"variant": v, "only": only, "metrics": mt, "linear_stats": state["stats"]}
        if keep_logits:
            p["logits"] = logits.clone()
        passes.append(p)

    for v in variants:
        one(v, None)
    for v in per_class:
        for c in present:
            one(v, c)
    print(f"\n## summary: {qname} vs original bf16, {len(ids)} ids\n")
    print("| variant | scope | rel L2 | worst cos | argmax | top-5 | KL mean | KL p99 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in passes:
        mt = p["metrics"]
        print(f"| {p['variant']} | {p['only'] or 'all'} | {mt['rel'] * 100:.2f} % | {mt['cos_min']:.6f} | "
              f"{mt['top1']}/{mt['n']} | {mt['top5']:.3f} | {mt['kl_mean']:.3e} | {mt['kl_p99']:.3e} |")
    out = getattr(args, "out", None)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"snapshot": args.snapshot, "quantised": args.quantised, "prompt": args.prompt,
                       "ids": len(ids), "group": group, "vocab_used": vocab_used,
                       "passes": [{k: v for k, v in p.items() if k != "logits"} for p in passes]}, f, indent=1)
        print(f"wrote {out}")
    return {"base": base, "passes": passes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("logits", "actquant", "rtn", "sim"))
    ap.add_argument("snapshot")
    ap.add_argument("quantised", nargs="?")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--group", type=int, default=None,
                    help="rtn: the int4 group (default -1, per-channel); sim: h4 / h4p2's group along K "
                         "on both operands (default 256; -1 = all of K)")
    ap.add_argument("--variants", default="none,a8,a8w8,h8",
                    help=f"sim: comma list of {','.join(VARIANTS)}")
    ap.add_argument("--per-class", default="",
                    help="sim: variants to run once per linear class, that class only, the rest none")
    ap.add_argument("--vocab-used", type=int, default=0,
                    help="sim: the KL's ids (< N; 0 = all logits; Qwen3.8 248077, src/model/qwen35.h)")
    ap.add_argument("--no-linear-stats", dest="linear_stats", action="store_false",
                    help="sim: skip the per-linear error (one extra x W^T per replaced linear)")
    ap.add_argument("--out", help="sim: the metrics as JSON")
    args = ap.parse_args()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    if args.mode == "actquant":
        actquant(args)
        return
    if not args.quantised:
        sys.exit(f"{args.mode} mode needs a second snapshot")
    if args.mode == "sim":
        sim(args)
        return
    if args.mode == "rtn":
        with open(args.prompt, encoding="utf-8") as f:
            ids = [int(x) for x in f.read().split()]
        tc = CR.text_config(args.snapshot, args.layers)
        base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
        sd = CR.load_sd(args.quantised, args.layers, torch.bfloat16)
        if args.group is None:
            args.group = -1
        nq = 0
        for k in list(sd):
            if (k.startswith(f"{R.PREFIX}layers.") and k.endswith("_proj.weight") or
                    k.endswith(("in_proj_qkv.weight", "in_proj_z.weight"))) and sd[k].dim() == 2:
                if "in_proj_a" in k or "in_proj_b" in k:
                    continue
                sd[k] = rtn_int4(sd[k], args.group)
                nq += 1
        print(f"RTN int4 group {args.group}: {nq} linears")
        CR.compare(CR.run(tc, sd, ids), base,
                   f"RTN int4 g{args.group} of {os.path.basename(args.quantised.rstrip('/'))} vs original bf16")
        return
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    tc = CR.text_config(args.snapshot, args.layers)
    base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
    quant = CR.run(tc, load_quantised(args.quantised, args.layers), ids)
    CR.compare(quant, base, f"quantised {os.path.basename(args.quantised.rstrip('/'))} vs original bf16, "
                            f"{tc.num_hidden_layers} layers")


if __name__ == "__main__":
    main()
