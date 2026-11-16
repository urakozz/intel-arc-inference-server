#!/usr/bin/env python3
"""Plan 30a Task 2 Step 2: the facts of Laguna S 2.1's reference that the engine's kernels and plans 30b-30f rely
on, ASSERTED on a model the vendor's code runs (the tiny model of laguna_make_tiny.py carries the original's rope
parameters, window, head size, gate kind and routing rule unchanged).

    laguna_facts.py <tiny snapshot>          prints the facts; exits non-zero if one does not hold

1. the sliding window: which keys a query sees on a sliding layer (measured: perturb one input row, see which
   output rows move) on a 600-token prompt, and that a full layer sees every earlier key;
2. the rotary tables: full layers yarn over the first 64 of 128 dimensions (the rest pass through), cos / sin
   scaled by attention_factor, the inverse frequencies restated here from the yarn formula and equal bitwise to
   the reference's; sliding layers the plain table, theta 10,000, all 128 dimensions, scale 1;
3. the per-head gate: softplus(g_proj(x)) with x the layer's normed input, computed in f32, rounded to the
   activation dtype, one value per query head broadcast over its 128 outputs, multiplied in BEFORE o_proj
   (checked bitwise in bf16, where the rounding points show);
4. the router: logits in the activation dtype widened to f32, sigmoid scores, the selection bias added only for the ranking, the weights the UNBIASED
   scores of the chosen k divided by their sum, cast to the activation dtype; the routed sum times 2.5, then the
   shared expert's output added with no gate.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import copy  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

WINDOW_PROMPT = 600


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"laguna_facts_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fail(msg: str):
    raise AssertionError(msg)


def yarn_inv_freq(rp: dict, head_dim: int) -> torch.Tensor:
    """The yarn inverse frequencies from the config's rope parameters, restated (YaRN, arXiv 2309.00071, with the
    correction range floored / ceiled): the engine's host-side table follows this."""
    base, factor, orig = rp["rope_theta"], rp["factor"], rp["original_max_position_embeddings"]
    dim = int(head_dim * rp.get("partial_rotary_factor", 1.0))

    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(corr(rp.get("beta_fast") or 32)), 0)
    high = min(math.ceil(corr(rp.get("beta_slow") or 1)), dim - 1)
    pos = base ** (torch.arange(0, dim, 2).float() / dim)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extra = 1 - ramp
    return (1.0 / (factor * pos)) * (1 - extra) + (1.0 / pos) * extra


@torch.no_grad()
def window_facts(model) -> list[str]:
    tm = model.model
    cfg = tm.config
    ref = _load("laguna_ref")
    T = WINDOW_PROMPT
    g = torch.Generator().manual_seed(3)
    ids = [2] + torch.randint(100, 100000, (T - 1,), generator=g).tolist()
    p = ref.prologue(model, ids)
    out = []
    for kind in ("sliding_attention", "full_attention"):
        i = cfg.layer_types.index(kind)
        layer = tm.layers[i]

        def run(h):      # the attention alone: the MoE after it groups rows by expert, so a perturbed row moves
            # other rows' sums by rounding (the expert GEMMs' shapes change) - not a visibility
            return layer.self_attn(layer.input_layernorm(h), attention_mask=p["masks"][kind],
                                   position_embeddings=p["pe"][kind])[0][0]
        base = run(p["h"])
        q = T - 1
        seen = []
        for j in range(T):
            h = p["h"].clone()
            h[0, j] += 1.0
            if not torch.equal(run(h)[q], base[q]):
                seen.append(j)
        if kind == "sliding_attention":
            want = list(range(q - cfg.sliding_window + 1, q + 1))
            if seen != want:
                _fail(f"sliding layer {i}: query {q} sees keys {seen[0]}..{seen[-1]} ({len(seen)}), expected "
                      f"{want[0]}..{want[-1]}")
            out.append(f"sliding window: query p sees keys p-{cfg.sliding_window - 1} .. p ({cfg.sliding_window} keys "
                       f"including itself; measured on layer {i}, query {q} of a {T}-token prompt)")
        else:
            if seen != list(range(T)):
                _fail(f"full layer {i}: query {q} sees {len(seen)} keys, expected {T}")
            out.append(f"full attention: query p sees keys 0 .. p (measured on layer {i}, query {q})")
    return out


@torch.no_grad()
def rotary_facts(model) -> list[str]:
    tm = model.model
    cfg = tm.config
    full_rp = cfg.rope_parameters["full_attention"]
    swa_rp = cfg.rope_parameters["sliding_attention"]
    D = cfg.head_dim
    out = []
    fr = tm.rotary_emb
    if fr.rope_type != "yarn" or fr.inv_freq.shape[0] * 2 != int(D * full_rp["partial_rotary_factor"]):
        _fail(f"full layers: rope {fr.rope_type}, {fr.inv_freq.shape[0] * 2} rotating dimensions")
    want = yarn_inv_freq(full_rp, D)
    if not torch.equal(fr.inv_freq.float(), want):
        _fail(f"full layers: the yarn inverse frequencies differ from the restated formula "
              f"(max |d| {(fr.inv_freq.float() - want).abs().max():.3g})")
    if fr.attention_scaling != full_rp["attention_factor"]:
        _fail(f"full layers: cos / sin scale {fr.attention_scaling}, config {full_rp['attention_factor']}")
    sr = tm.swa_rotary_emb
    if sr is None or sr.rope_type != "default":
        _fail("sliding layers do not have their own plain rotary table (they would reuse the yarn one)")
    want_s = 1.0 / (swa_rp["rope_theta"] ** (torch.arange(0, D, 2, dtype=torch.int64).float() / D))
    if not torch.equal(sr.inv_freq.float(), want_s) or sr.attention_scaling != 1.0:
        _fail("sliding layers: the inverse frequencies are not theta^(-2i/128) or the scale is not 1")
    pos = torch.arange(8).unsqueeze(0)
    x = torch.zeros(1, 8, cfg.hidden_size)
    cos, sin = fr(x, pos)
    if cos.shape[-1] != want.shape[0] * 2 or not torch.allclose(cos[0, 0], torch.full_like(cos[0, 0], fr.attention_scaling)):
        _fail(f"full layers: cos table width {cos.shape[-1]}, row 0 not attention_factor")
    m = sys.modules[type(model).__module__]
    q = torch.randn(1, 2, 8, D)
    qe, _ = m.apply_rotary_pos_emb(q, q, cos, sin)
    if not torch.equal(qe[..., cos.shape[-1]:], q[..., cos.shape[-1]:]):
        _fail("full layers: dimensions past the rotating part are changed")
    out.append(f"full-layer rope: yarn, theta {full_rp['rope_theta']:g}, factor {full_rp['factor']:g}, original "
               f"{full_rp['original_max_position_embeddings']}, beta fast/slow {full_rp['beta_fast']:g}/"
               f"{full_rp['beta_slow']:g}; rotates dims 0..{cos.shape[-1] - 1} of {D} as rotate_half pairs (i, "
               f"i + {cos.shape[-1] // 2}), dims {cos.shape[-1]}..{D - 1} pass through; cos and sin multiplied by "
               f"{fr.attention_scaling!r}; inv_freq[0] {want[0].item():.9g}, inv_freq[-1] {want[-1].item():.9g}")
    out.append(f"sliding-layer rope: plain, theta {swa_rp['rope_theta']:g}, all {D} dims (pairs i, i + {D // 2}), "
               f"scale 1")
    out.append("rope order: q_norm / k_norm (RMSNorm over each head's 128 values, f32 inside, weight times the "
               "result cast back) BEFORE the rotary embedding; cos / sin computed in f32 and cast to the activation dtype")
    return out


@torch.no_grad()
def gate_facts(model) -> list[str]:
    tm = model.model
    cfg = tm.config
    src = inspect.getsource(type(tm.layers[0].self_attn).forward)
    if "F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)" not in src:
        _fail("the gate expression in the vendor's attention changed")
    i = cfg.layer_types.index("sliding_attention")
    attn = copy.deepcopy(tm.layers[i].self_attn).to(torch.bfloat16)
    if not attn.gate_per_head or attn.g_proj.out_features != attn.num_heads:
        _fail(f"gate: per-head {attn.gate_per_head}, g_proj out {attn.g_proj.out_features}, heads {attn.num_heads}")
    T = 16
    x = torch.randn(1, T, cfg.hidden_size, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = tm.swa_rotary_emb(x, pos)
    seen = {}
    h1 = attn.o_proj.register_forward_pre_hook(lambda _m, a: seen.__setitem__("o_in", a[0].clone()))
    m = sys.modules[type(model).__module__]
    sdpa = m.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture(module, q, k, v, mask, **kw):
        o, w = sdpa(module, q, k, v, mask, **kw)
        seen["attn"] = o.clone()
        return o, w
    m.ALL_ATTENTION_FUNCTIONS["sdpa"] = capture
    try:
        attn(x, position_embeddings=(cos, sin), attention_mask=None)
    finally:
        m.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa
        h1.remove()
    a = seen["attn"].reshape(1, T, attn.num_heads, attn.head_dim)
    gate = F.softplus(attn.g_proj(x).float()).to(torch.bfloat16)
    want = (a * gate.unsqueeze(-1)).reshape(1, T, -1)
    if not torch.equal(seen["o_in"], want):
        _fail("gate: o_proj's input is not attn_out x bf16(softplus_f32(g_proj(x))) per head")
    wrong = (a.float() * F.softplus(attn.g_proj(x).float()).unsqueeze(-1)).to(torch.bfloat16).reshape(1, T, -1)
    differs = not torch.equal(wrong, want)
    return [f"gate: g_proj [{attn.num_heads}][{cfg.hidden_size}] (no bias) on the layer's normed input (q/k/v's "
            f"input); softplus(beta 1, threshold 20) in f32, rounded to the activation dtype, then one bf16 multiply "
            f"of each head's {attn.head_dim} attention outputs, before o_proj (bitwise in bf16"
            + ("; an f32 multiply rounded once differs" if differs else "") + ")"]


@torch.no_grad()
def router_facts(model) -> list[str]:
    tm = model.model
    cfg = tm.config
    ref = _load("laguna_ref")
    i = ref.moe_layers(cfg)[0]
    mlp = tm.layers[i].mlp
    T = 256
    x = torch.randn(1, T, cfg.hidden_size, generator=torch.Generator().manual_seed(7))
    seen = {}
    hook = mlp.gate.register_forward_hook(lambda _m, _a, out: seen.__setitem__("gate", out))
    y = mlp(x)
    hook.remove()
    logits, w, sel = seen["gate"]
    k = cfg.num_experts_per_tok
    xf = x.reshape(-1, cfg.hidden_size)
    want_logits = F.linear(xf, mlp.gate.weight).float()
    if not torch.equal(logits, want_logits):
        _fail("router: logits are not F.linear(x, W) widened to f32")
    s = torch.sigmoid(want_logits)
    bias = mlp.gate.e_score_correction_bias
    want_sel = torch.topk(s + bias, k, dim=-1).indices
    if not torch.equal(sel, want_sel):
        _fail("router: the selection is not topk(sigmoid(logits) + bias)")
    unbiased_pick = torch.topk(s, k, dim=-1).indices
    moved = int((unbiased_pick.sort(-1).values != want_sel.sort(-1).values).any(-1).sum())
    g = s.gather(-1, sel)
    if not torch.equal(w, (g / g.sum(-1, keepdim=True)).to(x.dtype)):
        _fail("router: weights are not the unbiased chosen scores over their sum")
    if not cfg.norm_topk_prob or cfg.moe_router_logit_softcapping != 0.0:
        _fail(f"router: norm_topk_prob {cfg.norm_topk_prob}, soft-cap {cfg.moe_router_logit_softcapping}")
    ex = mlp.experts
    routed = torch.zeros_like(xf)
    for r in range(xf.shape[0]):
        for j in range(k):
            e = int(sel[r, j])
            gt, up = F.linear(xf[r], ex.gate_up_proj[e]).chunk(2, dim=-1)
            routed[r] += F.linear(F.silu(gt) * up, ex.down_proj[e]) * w[r, j]
    want_y = routed * cfg.moe_routed_scaling_factor + mlp.shared_expert(xf)
    err = (y.reshape(-1, cfg.hidden_size) - want_y).abs().max().item()
    if err > 1e-5 * max(1.0, want_y.abs().max().item()):
        _fail(f"router: the block output is not 2.5 x sum_k w_k f_k(x) + shared(x) (max |d| {err:.3g})")
    if hasattr(mlp, "shared_expert_gate"):
        _fail("router: a shared-expert gate exists")
    return [f"router: logits = x W^T in the ACTIVATION dtype (W [{cfg.num_experts}][{cfg.hidden_size}]; a bf16 run "
            f"rounds them to bf16), then widened to f32; scores = sigmoid(logits) in f32; "
            f"the top {k} by scores + e_score_correction_bias (the bias only ranks: it moved the chosen set on "
            f"{moved} of {T} random rows here); weights = the chosen UNBIASED scores / their sum (norm_topk_prob), "
            f"cast to the activation dtype; no soft-cap",
            f"moe block: out = {cfg.moe_routed_scaling_factor:g} x sum_k w_k expert_k(x) + shared_expert(x); the "
            f"scale on the routed sum (after the weighted sum, before the shared add), the shared expert ungated; "
            f"expert_k = down(silu(gate x) * up x), gate_up fused gate rows first"]


def check(snapshot: str, model=None) -> list[str]:
    """Every fact, asserted; returns the printed lines. `model`: the vendor's model loaded plainly (f32)."""
    if model is None:
        ref = _load("laguna_ref")
        ref.require_version()
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32,
                                                     attn_implementation="sdpa").eval()
    cfg = model.config
    lines = [f"model: {type(model).__module__}.{type(model).__name__} (the snapshot's own code), "
             f"{cfg.num_hidden_layers} layers {''.join('F' if t == 'full_attention' else 's' for t in cfg.layer_types)}, "
             f"dense layers {cfg.mlp_only_layers}, head size {cfg.head_dim}"]
    lines += window_facts(model) + rotary_facts(model) + gate_facts(model) + router_facts(model)
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    for line in check(sys.argv[1]):
        print(line)
    print("FACTS OK")


if __name__ == "__main__":
    main()
