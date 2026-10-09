#!/usr/bin/env python3
"""Checks for the W4A4 variants of eval_quantised.py sim (plan 2026-10-09-w4a4-probe, Task 1).

    python3 tools/rotate/test_w4a4.py [--before <tree>] [test_name ...]

CPU, synthetic tensors and a tiny random Qwen3.5 model, ~4 min on 4 cores. In the oracle image:

    docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -w /ws --entrypoint python3 \\
        agnes-ref-img:latest tools/rotate/test_w4a4.py

1. quant_int4_groups round-trips within s / 2 of every element, s = max|v| / 7.5 per group
   (and per power-of-two scale); an all-zero group survives as zeros;
2. the groups sit at K offsets 0, g, 2g, ...: each scale is its own slice's max, and a change
   inside one group moves that group's scale and codes only (Review Focus 1);
3. a K that is not a multiple of the group is refused, naming the linear; every Qwen3.8 linear
   the sim replaces has K a multiple of 256 (and of the rotation's 1024-block), read from the
   engine's descriptor (src/model/model_desc.cc set_qwen38_shapes / make_qwen38, src/model/qwen35.h);
4. the power-of-two scale is a power of two, >= the fp32 scale and < twice it - also for scales
   one ulp above a power of two, where log2 would round (Review Focus 3);
5. w4a4_matmul with the identity rotation equals the int4 x int4 product computed group by group
   in float64 from explicit K slices [g j, g (j + 1)) of both operands, within 1e-6 relative;
   with h8's rotation it equals the same product of the rotated operands (Review Focus 1);
6. the sim's h4 / h4p2 use h8's rotation exactly: the same signs (R.signs(K, R.SEED + K)) and
   R.rot on both sides (Review Focus 2); h8's own product is unchanged;
7. end to end on a tiny random Qwen3.5 model (GPTQ g64 checkpoint written here): every variant
   runs; w4a16 is the checkpoint as shipped (bitwise the `logits` mode's model); a per-class pass
   touches only its class; the per-linear statistics cover every class. With --before <tree>
   (a copy of tools/rotate + tools/oracle/dequant.py from before this change), `none` and `h8`
   logits are bitwise the old sim's.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("b70_rotation", os.path.join(_HERE, "rotation.py"))
EQ = _load("b70_eval_quantised", os.path.join(_HERE, "eval_quantised.py"))

FAILS = []
BEFORE = None


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def outliers(r: int, k: int, seed: int) -> torch.Tensor:
    """Activation-like rows: gaussian plus a few large channels, as real x has."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(r, k, generator=g)
    for c in (3, 77, k // 2 + 5, k - 1):
        x[:, c] = x[:, c] * 30 + 10
    return x


# ---- 1. the round trip --------------------------------------------------------------------
def test_round_trip() -> None:
    x = outliers(6, 1024, 1)
    x[2, 256:512] = 0.0                                        # an all-zero group
    for pow2 in (False, True):
        q, s = EQ.quant_int4_groups(x, 256, pow2=pow2)
        y = EQ.fake_int4_groups(x, 256, pow2=pow2)
        sb = s.repeat_interleave(256, dim=1)
        err = (y - x).abs()
        ok = bool((err <= sb / 2 * (1 + 1e-6)).all())
        check(f"round trip within s/2 (pow2={pow2})", ok, f"max err/s {float((err / sb).max()):.4f}")
        check(f"codes in [-8, 7], int8 (pow2={pow2})",
              q.dtype == torch.int8 and int(q.min()) >= -8 and int(q.max()) <= 7)
        check(f"zero group stays zero with s = 1 (pow2={pow2})",
              bool((y[2, 256:512] == 0).all()) and float(s[2, 1]) == 1.0)
        if not pow2:
            ref = x.reshape(6, 4, 256).abs().amax(2) / 7.5
            ref[ref == 0] = 1.0
            check("fp32 scale = max|v| / 7.5 (the RTN rule rtn_int4 uses)", torch.equal(s, ref))
    # the existing g64 RTN control is the same grid: fake_int4_groups at g = 64 == rtn_int4
    w = outliers(16, 512, 2)
    check("fake_int4_groups(g) == rtn_int4(g) bitwise (one grid in the tool)",
          torch.equal(EQ.fake_int4_groups(w, 64), EQ.rtn_int4(w, 64)))


# ---- 2. group alignment -------------------------------------------------------------------
def test_alignment() -> None:
    k, g = 2048, 256
    x = outliers(5, k, 3)
    q0, s0 = EQ.quant_int4_groups(x, g)
    ok = True
    for j in range(k // g):
        own = x[:, j * g:(j + 1) * g].abs().amax(1) / 7.5
        ok &= torch.equal(s0[:, j], own)
    check("scale j = max|x[:, 256 j : 256 (j + 1)]| / 7.5 for every j (offsets 0, 256, ...)", ok)
    for j in (0, 3, 7):
        x2 = x.clone()
        x2[:, j * g + 17] = 1e4                                 # one element inside group j
        q2, s2 = EQ.quant_int4_groups(x2, g)
        other = [i for i in range(k // g) if i != j]
        same_s = torch.equal(s2[:, other], s0[:, other])
        moved_s = bool((s2[:, j] != s0[:, j]).all())
        outside = torch.ones(k, dtype=torch.bool)
        outside[j * g:(j + 1) * g] = False
        same_q = torch.equal(q2[:, outside], q0[:, outside])
        check(f"a change in group {j} moves only group {j}'s scale and codes", same_s and moved_s and same_q)


# ---- 3. K refused by name; Qwen3.8's real K --------------------------------------------------
def qwen38_linear_k() -> dict:
    """The K of every linear the sim replaces, from the engine's descriptor (no checkpoint on the
    Mac): src/model/model_desc.cc set_qwen38_shapes() / make_qwen38(), src/model/qwen35.h."""
    with open(os.path.join(_ROOT, "src/model/model_desc.cc"), encoding="utf-8") as f:
        desc = f.read()
    with open(os.path.join(_ROOT, "src/model/qwen35.h"), encoding="utf-8") as f:
        hdr = f.read()
    shapes = re.search(r"void set_qwen38_shapes\(ModelDesc& d\) \{(.*?)\n\}", desc, re.S).group(1)
    make = re.search(r"ModelDesc make_qwen38\(\) \{(.*?)\n\}", desc, re.S).group(1)

    def field(src, name):
        return int(re.search(rf"d\.{name} = (\d+);", src).group(1))

    def const(name):
        return int(re.search(rf"{name} = (\d+)", hdr).group(1))
    hidden = field(shapes, "hidden")
    fa_value = field(shapes, "fa_q_heads") * const("kFaHeadDim")           # o_proj K
    gdn_value = field(shapes, "gdn_v_heads") * const("kGdnHeadDim")        # out_proj K
    inter = field(make, "intermediate")                                    # down_proj K
    return {"self_attn.q_proj": hidden, "self_attn.k_proj": hidden, "self_attn.v_proj": hidden,
            "self_attn.o_proj": fa_value, "linear_attn.in_proj_qkv": hidden,
            "linear_attn.in_proj_z": hidden, "linear_attn.out_proj": gdn_value,
            "mlp.gate_proj": hidden, "mlp.up_proj": hidden, "mlp.down_proj": inter}


def test_k_refused() -> None:
    try:
        EQ.quant_int4_groups(torch.zeros(4, 300), 256, name="L7.mlp.down_proj")
        check("K 300 at group 256 refused", False, "no exception")
    except ValueError as e:
        msg = str(e)
        check("K 300 at group 256 refused, naming the linear", "L7.mlp.down_proj" in msg
              and "300" in msg and "256" in msg, msg)
    ks = qwen38_linear_k()
    print(f"      Qwen3.8 linear K (src/model/model_desc.cc): {ks}")
    check("the descriptor gives hidden 5120, o_proj 6144, out_proj 6144, down 17408",
          (ks["mlp.gate_proj"], ks["self_attn.o_proj"], ks["linear_attn.out_proj"], ks["mlp.down_proj"])
          == (5120, 6144, 6144, 17408))
    check("every Qwen3.8 linear's K is a multiple of 256", all(k % 256 == 0 for k in ks.values()))
    check("... and of the rotation's 1024-block", all(k % R.BLOCK == 0 for k in ks.values()))
    check("every replaced linear name maps to one of the six classes",
          all(EQ.linear_class("L0." + n) in EQ.CLASS_NAMES for n in ks))


# ---- 4. power-of-two scales ---------------------------------------------------------------
def test_pow2() -> None:
    x = outliers(8, 1024, 4) * torch.logspace(-6, 3, 8).unsqueeze(1)
    # a group whose fp32 scale is exactly a power of two, and ones one ulp above / below one
    x[0, :256] = 0.0
    x[0, 0] = 7.5 * 2.0 ** -5
    x[1, :256] = 0.0
    x[1, 0] = 7.5 * float(torch.nextafter(torch.tensor(2.0 ** -5), torch.tensor(1.0)))
    x[2, :256] = 0.0
    x[2, 0] = 7.5 * float(torch.nextafter(torch.tensor(2.0 ** 3), torch.tensor(0.0)))
    _, s = EQ.quant_int4_groups(x, 256)
    _, s2 = EQ.quant_int4_groups(x, 256, pow2=True)
    m, _ = torch.frexp(s2)
    check("pow2 scale is a power of two", bool((m == 0.5).all()))
    check("pow2 scale >= the fp32 scale", bool((s2 >= s).all()),
          f"min ratio {float((s2 / s).min()):.9f}")
    check("pow2 scale < 2 x the fp32 scale", bool((s2 < 2 * s).all()))
    check("an exact power-of-two scale is kept", float(s2[0, 0]) == float(s[0, 0]) == 2.0 ** -5)
    check("one ulp above 2^-5 rounds up to 2^-4", float(s2[1, 0]) == 2.0 ** -4, f"{float(s2[1, 0])!r}")
    check("one ulp below 2^3 rounds up to 2^3", float(s2[2, 0]) == 2.0 ** 3, f"{float(s2[2, 0])!r}")


# ---- 5. the int4 x int4 product, group by group -------------------------------------------
def int4_product_f64(x: torch.Tensor, w: torch.Tensor, g: int, pow2: bool) -> torch.Tensor:
    """sum_j s_x[j] s_w[j] (q_x[:, j] . q_w[:, j]) in float64, every group cut by hand."""
    k = x.shape[1]
    out = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float64)
    for j in range(k // g):
        xs, ws = x[:, j * g:(j + 1) * g].float(), w[:, j * g:(j + 1) * g].float()
        sx = xs.abs().amax(1, keepdim=True) / 7.5
        sw = ws.abs().amax(1, keepdim=True) / 7.5
        if pow2:
            sx, sw = (torch.where(t > 0, torch.ldexp(torch.ones_like(t), torch.frexp(t)[1]
                                                     - (torch.frexp(t)[0] == 0.5).to(torch.int32)), t)
                      for t in (sx, sw))
        sx[sx == 0] = 1.0
        sw[sw == 0] = 1.0
        qx = torch.clamp(torch.round(xs / sx), -8, 7).double()
        qw = torch.clamp(torch.round(ws / sw), -8, 7).double()
        out += (qx @ qw.t()) * (sx.double() @ sw.double().t())
    return out


def test_identity_product() -> None:
    for k in (1024, 5120, 17408):
        x = outliers(9, k, 5)
        w = torch.randn(33, k, generator=torch.Generator().manual_seed(6)) * 0.02
        for pow2 in (False, True):
            y = EQ.w4a4_matmul(x, w, 256, pow2=pow2, rot=None)
            ref = int4_product_f64(x, w, 256, pow2)
            rel = float((y.double() - ref).norm() / ref.norm())
            check(f"identity rotation, K {k}, pow2={pow2}: = float64 int4 x int4 within 1e-6", rel < 1e-6,
                  f"rel {rel:.2e}")
    k = 6144
    x, w = outliers(7, k, 7), torch.randn(19, k, generator=torch.Generator().manual_seed(8)) * 0.02
    d = R.signs(k, R.SEED + k)
    y = EQ.w4a4_matmul(x, w, 256, rot=lambda t: R.rot(t, d))
    ref = int4_product_f64(R.rot(x, d), R.rot(w, d), 256, False)
    rel = float((y.double() - ref).norm() / ref.norm())
    check("h8's rotation then groups of 256 on BOTH rotated operands = float64 reference", rel < 1e-6,
          f"rel {rel:.2e}")
    # the rotation is orthogonal, so without quantisation it changes nothing
    ex = (R.rot(x.double(), d.double()) @ R.rot(w.double(), d.double()).t() - x.double() @ w.double().t())
    check("the rotation itself preserves x W^T in float64",
          float(ex.norm() / (x.double() @ w.double().t()).norm()) < 1e-12)


# ---- 6. h4 is h8's rotation, h8 unchanged --------------------------------------------------
def test_variant_rotation() -> None:
    k = 5120
    x, w = outliers(11, k, 9), torch.randn(40, k, generator=torch.Generator().manual_seed(10)) * 0.02
    d = EQ.rotation_signs(k)
    check("rotation_signs(K) = R.signs(K, R.SEED + K), h8's", torch.equal(d, R.signs(k, R.SEED + k)))
    for v, pow2 in (("h4", False), ("h4p2", True)):
        got = EQ.variant_matmul(v, x, w, d, 256, "t")
        want = EQ.w4a4_matmul(x, w, 256, pow2=pow2, rot=lambda t: R.rot(t, d))
        check(f"{v} = w4a4_matmul over R.rot(., signs) on both sides, bitwise", torch.equal(got, want))
    h8 = EQ.variant_matmul("h8", x, w, d, 256, "t")
    check("h8 = q8_token(xR) q8_channel(WR)^T, bitwise (unchanged)",
          torch.equal(h8, EQ.q8_token(R.rot(x, d)) @ EQ.q8_channel(R.rot(w, d)).t()))
    check("none = x W^T, bitwise", torch.equal(EQ.variant_matmul("none", x, w, None, 256, "t"), x @ w.t()))
    g_all = EQ.variant_matmul("h4", x, w, d, -1, "t")
    check("--group -1 is one group over all of K", torch.equal(
        g_all, EQ.w4a4_matmul(x, w, k, rot=lambda t: R.rot(t, d))))


# ---- 7. the tiny model end to end ---------------------------------------------------------
VOCAB = 512
TEXT = dict(hidden_size=1024, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=8,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
            full_attention_interval=4, vocab_size=VOCAB, rms_norm_eps=1e-6, attn_output_gate=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            partial_rotary_factor=0.25, tie_word_embeddings=False, model_type="qwen3_5_text",
            layer_types=["linear_attention"] * 3 + ["full_attention"])
_STATE = {}


def pack_gptq(w: torch.Tensor, gs: int = 64):
    """w [N, K] -> GPTQ v1 sym int4 (qweight [K/8, N] int32, scales [K/gs, N] f16, qzeros, g_idx)."""
    n, k = w.shape
    wt = w.float().t().contiguous()                                        # [K, N]
    s = (wt.reshape(k // gs, gs, n).abs().amax(1) / 7.5).clamp_min(1e-8).to(torch.float16)
    q = torch.clamp(torch.round(wt / s.float().repeat_interleave(gs, 0)), -8, 7).to(torch.int64) + 8
    words = torch.zeros(k // 8, n, dtype=torch.int64)
    for i in range(8):
        words |= q[i::8] << (4 * i)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    zeros = torch.full((k // gs, n // 8), 0x77777777, dtype=torch.int32)
    return words, s, zeros, (torch.arange(k, dtype=torch.int32) // gs)


def tiny_checkpoints():
    """(bf16 snapshot, quantised snapshot, prompt file) - the bf16 model with outlier channels, and
    its GPTQ g64 RTN pack of every linear AutoRound quantises (in_proj_a/b, lm_head stay bf16)."""
    if "snap" in _STATE:
        return _STATE["snap"]
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    tc = Qwen3_5TextConfig(**TEXT)
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tc).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 2 and "embed" not in n:
                p.copy_(torch.randn_like(p.float()) / p.shape[1] ** 0.5)
        emb = model.model.embed_tokens.weight
        emb[:, [5, 300, 777]] *= 25                             # residual-stream outlier channels
    root = tempfile.mkdtemp(prefix="w4a4_test_")
    bf, qd = os.path.join(root, "bf16"), os.path.join(root, "quant")
    os.makedirs(bf)
    os.makedirs(qd)
    sd_b, sd_q = {}, {}
    for n, t in model.state_dict().items():
        name = "lm_head.weight" if n == "lm_head.weight" else "model.language_model." + n[len("model."):]
        sd_b[name] = t.contiguous()
        quant = (".layers." in name and name.endswith("_proj.weight") or
                 name.endswith(("in_proj_qkv.weight", "in_proj_z.weight"))) and t.dim() == 2 \
            and "in_proj_a" not in name and "in_proj_b" not in name
        if quant:
            base = name[: -len(".weight")]
            qw, sc, qz, gi = pack_gptq(t)
            sd_q.update({base + ".qweight": qw, base + ".scales": sc, base + ".qzeros": qz, base + ".g_idx": gi})
        else:
            sd_q[name] = t.contiguous()
    for d, sd, extra in ((bf, sd_b, {}), (qd, sd_q, {"quantization_config": {
            "bits": 4, "group_size": 64, "sym": True, "desc_act": False, "quant_method": "gptq"}})):
        save_file(sd, os.path.join(d, "model-00001-of-00001.safetensors"))
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}}, f)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                       "text_config": TEXT, **extra}, f)
    prompt = os.path.join(root, "p.ids")
    ids = torch.randint(0, VOCAB, (48,), generator=torch.Generator().manual_seed(11)).tolist()
    with open(prompt, "w") as f:
        f.write(" ".join(map(str, ids)) + "\n")
    _STATE["snap"] = (bf, qd, prompt)
    return _STATE["snap"]


def sim_args(variants: str, per_class: str = "", group=None):
    bf, qd, prompt = tiny_checkpoints()
    return argparse.Namespace(mode="sim", snapshot=bf, quantised=qd, prompt=prompt, layers=0, tokens=512,
                              group=group, variants=variants, per_class=per_class, vocab_used=0,
                              linear_stats=True, out=None)


def test_tiny_model() -> None:
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  7. transformers' Qwen3_5 is not importable here ({type(e).__name__}); "
              "run in the oracle image")
        return
    allv = "none,a8,a8w8,h8,w4a16,h4,h4p2"
    res = EQ.sim(sim_args(allv, per_class="h4"), keep_logits=True)
    passes = {(p["variant"], p["only"]): p for p in res["passes"]}
    check("every variant ran, finite logits",
          all(bool(torch.isfinite(passes[(v, None)]["logits"]).all()) for v in allv.split(",")))
    # w4a16 = the checkpoint as shipped = eval_quantised.py logits' model, bitwise
    bf, qd, prompt = tiny_checkpoints()
    with open(prompt, encoding="utf-8") as f:
        ids = [int(t) for t in f.read().split()]
    tc = EQ.CR.text_config(bf, 0)
    shipped = EQ.CR.run(tc, EQ.load_quantised(qd, 0), ids)
    check("w4a16 = the shipped model's forward (logits mode), bitwise",
          torch.equal(passes[("w4a16", None)]["logits"], shipped))
    base = res["base"]
    m = {v: passes[(v, None)]["metrics"] for v in allv.split(",")}
    for v in allv.split(","):
        print(f"      {v:6s} rel L2 {m[v]['rel']:.4%}  worst cos {m[v]['cos_min']:.6f}  "
              f"argmax {m[v]['top1']}/{m[v]['n']}  top-5 {m[v]['top5']:.3f}  "
              f"KL mean {m[v]['kl_mean']:.2e} p99 {m[v]['kl_p99']:.2e}")
    check("none's metrics are the CR.compare ones (rel L2 of the logits)",
          abs(m["none"]["rel"] - float((passes[("none", None)]["logits"].double() - base.double()).norm()
                                       / base.double().norm())) < 1e-12)
    check("h4 costs more than h8 on the tiny model (int4 activations are coarser)", m["h4"]["rel"] > m["h8"]["rel"])
    check("KL >= 0, p99 >= mean", all(m[v]["kl_mean"] >= 0 and m[v]["kl_p99"] >= m[v]["kl_mean"] * 0.999
                                      for v in m))
    classes = [c for c in EQ.CLASS_NAMES if (("h4", c) in passes)]
    check("the per-class passes cover the six classes", classes == list(EQ.CLASS_NAMES), str(classes))
    none_logits = passes[("none", None)]["logits"]
    check("each per-class h4 pass differs from none (its class was quantised)",
          all(not torch.equal(passes[("h4", c)]["logits"], none_logits) for c in classes))
    stats = passes[("h4", None)]["linear_stats"]
    got = sorted({s["class"] for s in stats})
    check("per-linear stats cover every class in the full h4 pass", got == sorted(EQ.CLASS_NAMES), str(got))
    check("per-linear stats: one row per replaced linear (3 GDN x 6 + 1 FA x 7 = 25)", len(stats) == 25,
          str(len(stats)))
    check("no per-linear stats on a per-class pass", passes[("h4", "down")]["linear_stats"] is None)
    try:
        EQ.sim(sim_args("h4", group=384))
        check("group 384 refused before any forward", False, "ran")
    except SystemExit as e:
        check("group 384 refused before any forward, naming a linear", "L0." in str(e) and "384" in str(e),
              str(e)[:160])
    # the command line, as the box runs it: argparse, --out JSON, --vocab-used, no per-linear stats
    out = os.path.join(os.path.dirname(prompt), "cli.json")
    argv = sys.argv
    sys.argv = ["eval_quantised.py", "sim", bf, qd, "--prompt", prompt, "--variants", "none,h4p2", "--group", "256",
                "--per-class", "h8", "--vocab-used", "500", "--no-linear-stats", "--out", out]
    try:
        EQ.main()
    finally:
        sys.argv = argv
    with open(out, encoding="utf-8") as f:
        js = json.load(f)
    scopes = [(p["variant"], p["only"]) for p in js["passes"]]
    check("the CLI: --out JSON with every pass, group 256, vocab_used 500, no per-linear stats",
          js["group"] == 256 and js["vocab_used"] == 500 and scopes[:2] == [("none", None), ("h4p2", None)]
          and len(scopes) == 2 + len(EQ.CLASS_NAMES) and all(p["linear_stats"] is None for p in js["passes"])
          and abs(js["passes"][1]["metrics"]["rel"] - m["h4p2"]["rel"]) < 1e-12, str(scopes))
    if BEFORE:
        test_before(passes)


def test_before(passes) -> None:
    """`none` and `h8` bitwise the sim before this change (BEFORE: a tree with the old tools/rotate)."""
    old = _load("b70_eval_quantised_before", os.path.join(BEFORE, "tools/rotate/eval_quantised.py"))
    got = []
    old.CR.compare = lambda a, b, label: got.append(a.clone())
    args = sim_args("none,h8")
    old.sim(argparse.Namespace(**{k: v for k, v in vars(args).items()
                                  if k in ("snapshot", "quantised", "prompt", "layers", "variants")}))
    check("before/after: none bitwise the old sim's", torch.equal(got[0], passes[("none", None)]["logits"]))
    check("before/after: h8 bitwise the old sim's", torch.equal(got[1], passes[("h8", None)]["logits"]))


TESTS = [test_round_trip, test_alignment, test_k_refused, test_pow2, test_identity_product,
         test_variant_rotation, test_tiny_model]


def main() -> None:
    global BEFORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", help="a tree holding the old tools/rotate and tools/oracle/dequant.py")
    ap.add_argument("tests", nargs="*")
    a = ap.parse_args()
    BEFORE = a.before
    torch.set_grad_enabled(False)
    for t in TESTS:
        if not a.tests or t.__name__ in a.tests:
            t()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
