#!/usr/bin/env python3
"""Spec 24a: the CPU reference of LiquidAI LFM2.5-24B-A2B (`lfm2_moe`), F1.

The model is transformers 5.19.0's own `Lfm2MoeForCausalLM` (refused on any other version), built on `meta` from
the checkpoint's config and LAYER-STREAMED (stream.py: the embedding - which is also the tied head -, the final
`embedding_norm` and the rotary table resident; each decoder layer materialised by a forward pre-hook and dropped
after). Routed experts are read on demand (LazyExperts, a pre-hook on `feed_forward.experts` that sees the routed
ids): bf16 per expert from the original (`experts.E.{w1,w3,w2}.weight`, w1 / w3 concatenated gate rows first, as
conversion_mapping.py:1160-1174 fuses them), or dequantised from our int4 g64 export (`qweight` / `scales`,
stream.dequant_t: bf16(float(q - 8) x float(scale)); either expert spelling, w1 / w3 / w2 or gate_proj / up_proj
/ down_proj) - the ENGINE-FORMAT reference. Neither experts implementation reads an expert no token routes to, so
the result is the materialised model's. Attention is eager (spec 24 §2.6, decision 13).

LFM2 adds no mechanism to spec 21a's arrangement; its traps are rounding points, each restated below as a plain
function (rms_norm, route, combine, short_conv, dense_mlp) and tested bitwise against the module it restates
(test_lfm2moe_ref.py); 24c's host twin is generated from them. `port_forward` chains the restated ops (with
transformers' attention module) into a whole forward - hfcheck holds it bitwise against transformers' own model.

    lfm2moe_ref.py run <snapshot> --prompt <ids> --out <p>.golden.safetensors [--gen 32] [--layers N]
                       [--weights bf16|int4] [--chunk C] [--logits-tail L] [--act-tail A]
    lfm2moe_ref.py ppl <snapshot> [--text NAME=FILE ...] [--ids NAME=FILE ...] [--layers N] [--max-tokens T]
    lfm2moe_ref.py hfcheck <snapshot> --layers N [--prompt <ids>] [--n 2048] [--gen 4]
    lfm2moe_ref.py facts <snapshot>

`run`'s golden layout (rows are absolute positions; `act_row0` / `logits_row0` in the metadata):
    h.L<l>          bf16 [A][H]     the residual after layer l: every row for T <= 4096, else the last --act-tail
                                    (256) prompt rows; plus every generated row
    mixer.L<l>      bf16 [A'][H]    the conv / attention output (before the residual add), last --act-tail rows +
    ffn.L<l>        bf16 [A'][H]    the FFN output (dense or MoE)                       every generated row
    conv.L<l>       bf16 [2][H]     conv layers: the conv state after the prompt - Bx of positions T-2, T-1, oldest
                                    first (the engine's two ring slots; transformers keeps 3, the oldest unread)
    route.ids.L<l>  i32  [T+gen][4] MoE layers: the selection in ITS order (descending sel = the combine's slots)
    route.w.L<l>    f32  [T+gen][4] the renormalised weights as the experts get them (bf16 values, widened)
    route.sel.L<l>  f32  [T+gen][5] the top 5 sel = f32(bf16(sigmoid(logit))) + expert_bias, descending
    route.gap.L<l>  f32  [T+gen]    sel(4th) - sel(5th) (0 = an exact tie at the cut: undetermined)
    logits          f32  [Lr][V]    rows logits_row0 .. T+gen-1 (the last --logits-tail 1024 prompt rows)
    tokens          i32  [gen]      greedy;   nll f32 [T-1] the prompt's teacher-forced NLL
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
# 5.19.0's default for Lfm2MoeExperts on torch 2.14.1 CPU (Task 1, lfm2moe_tiny_check.py: grouped_mm dispatches)
DEFAULT_EXPERTS = "grouped_mm"
EXPERT_SPELLINGS = {"w": ("w1", "w3", "w2"), "proj": ("gate_proj", "up_proj", "down_proj")}
INT4 = (".qweight", ".qzeros", ".scales")
LONG_PROMPT = 4096                      # rows past this keep only the activation tail


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"lfm_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stream = _load("stream")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def require_version() -> None:
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(f"lfm2moe_ref restates transformers {PINNED_TRANSFORMERS}'s lfm2_moe; this is "
                           f"{transformers.__version__} ({os.path.dirname(transformers.__file__)}) - put the 5.19.0 "
                           f"site first on PYTHONPATH (tools/oracle/qwen4exp_env.sh)")


def mm():
    require_version()
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    return m


# ------------------------------------------------------------------------------------------------
# the config

def read_config(snapshot: str) -> dict:
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def config(snapshot: str, layers: int | None = None, experts: str = DEFAULT_EXPERTS,
           dtype: torch.dtype = torch.bfloat16, raw: dict | None = None):
    """config.json -> the Lfm2MoeConfig the reference builds (eager attention, `experts`, `dtype`).

    layers=N truncates num_hidden_layers and layer_types[:N] (the engine's --layers N: layers [0, N), then
    embedding_norm and the head). N must keep an attention layer: transformers 5.19.0's DynamicCache takes the
    sequence length from one (cache_utils.py:1549-1566; spec 21 §12 finding 4) and the first is layer 2."""
    require_version()
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    d = dict(raw if raw is not None else read_config(snapshot))
    d.pop("quantization_config", None)
    if d.get("model_type") != "lfm2_moe":
        die(f"{snapshot}: model_type {d.get('model_type')!r}, expected lfm2_moe")
    if layers is not None:
        n = d["num_hidden_layers"]
        if not 1 <= layers <= n:
            die(f"--layers {layers}: the checkpoint has {n}")
        types = list(d["layer_types"][:layers])
        if "full_attention" not in types:
            first = d["layer_types"].index("full_attention") if "full_attention" in d["layer_types"] else None
            die(f"--layers {layers} keeps no attention layer (the first is layer {first}): transformers' DynamicCache "
                f"takes the sequence length from an attention layer, so N >= {None if first is None else first + 1}")
        d["num_hidden_layers"] = layers
        d["layer_types"] = types
    cfg = Lfm2MoeConfig(**d)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = experts
    cfg.dtype = dtype
    return cfg


def expected_names(cfg, form: str, expert_names: str = "w") -> list[str]:
    """Every tensor of a checkpoint of `cfg` in `form`:
      "bf16"  the original: every weight `.weight` (per-expert experts.E.{w1,w3,w2}), `expert_bias` (F32),
              no `lm_head.weight` (tied)
      "int4"  our export (spec 24 §5): every nn.Linear packed (`.qweight` / `.qzeros` / `.scales`) - routed
              experts, dense w1 / w2 / w3, conv in_proj / out_proj, attention q / k / v / out_proj; the router
              `feed_forward.gate.weight`, `expert_bias`, `conv.conv.weight`, the norms and the embedding as shipped
    expert_names "w" (experts.E.w1 / w3 / w2) or "proj" (experts.E.gate_proj / up_proj / down_proj)."""
    if form not in ("bf16", "int4"):
        raise ValueError(form)
    if expert_names not in EXPERT_SPELLINGS:
        raise ValueError(expert_names)
    q = form == "int4"

    def lin(base: str) -> list[str]:
        return [base + s for s in INT4] if q else [base + ".weight"]
    names = ["model.embed_tokens.weight", "model.embedding_norm.weight"]
    for i, t in enumerate(cfg.layer_types[: cfg.num_hidden_layers]):
        b = f"model.layers.{i}."
        names += [b + "operator_norm.weight", b + "ffn_norm.weight"]
        if t == "full_attention":
            names += [b + "self_attn.q_layernorm.weight", b + "self_attn.k_layernorm.weight"]
            for s in ("q_proj", "k_proj", "v_proj", "out_proj"):
                names += lin(f"{b}self_attn.{s}")
        else:
            names += [b + "conv.conv.weight"] + lin(b + "conv.in_proj") + lin(b + "conv.out_proj")
        f = b + "feed_forward."
        if i < cfg.num_dense_layers:
            for s in ("w1", "w2", "w3"):
                names += lin(f + s)
            continue
        names += [f + "gate.weight", f + "expert_bias"]
        spell = EXPERT_SPELLINGS[expert_names] if q else EXPERT_SPELLINGS["w"]
        for e in range(cfg.num_experts):
            for s in spell:
                names += lin(f"{f}experts.{e}.{s}")
    return names


# ------------------------------------------------------------------------------------------------
# the checkpoint

class Ckpt:
    """Name -> tensor over a snapshot's shards (the index, or a single model.safetensors)."""

    def __init__(self, snapshot: str):
        from safetensors import safe_open
        self.dir = snapshot
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                self.wm = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(snapshot, "model.safetensors"), framework="pt") as h:
                self.wm = {k: "model.safetensors" for k in h.keys()}
        missing = sorted({fn for fn in self.wm.values() if not os.path.exists(os.path.join(snapshot, fn))})
        if missing:
            die(f"{snapshot}: {len(missing)} shard(s) of the index are absent ({missing[:2]}): the download is "
                f"incomplete")
        self.h = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
                  for fn in sorted(set(self.wm.values()))}

    def __contains__(self, k: str) -> bool:
        return k in self.wm

    def keys(self):
        return self.wm.keys()

    def get(self, k: str) -> torch.Tensor:
        return self.h[self.wm[k]].get_tensor(k)

    def form(self) -> str:
        return "int4" if any(k.endswith(".qweight") for k in self.wm) else "bf16"


def _cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype) if t.is_floating_point() and t.dtype != dtype else t


def _dequant(ck: Ckpt, base: str, checked: set) -> torch.Tensor:
    """`base`.{qweight, scales, qzeros} -> the nn.Linear weight [N][K] bf16: bf16(float(q - 8) x float(scale)),
    the group size from the scales' shape. qzeros must be the symmetric constant."""
    qw, sc = ck.get(base + ".qweight"), ck.get(base + ".scales")
    g = qw.shape[0] * 8 // sc.shape[0]
    if base not in checked:
        qz = ck.get(base + ".qzeros")
        if not bool((qz == 0x77777777).all()):
            die(f"{base}.qzeros is not 0x77777777: not symmetric int4 (the engine's rule is (q - 8) x scale)")
    out = _stream.dequant_t(qw, sc, g, check=base not in checked)
    checked.add(base)
    return out


def layer_state(ck: Ckpt, keys, dtype: torch.dtype, checked: set) -> dict[str, torch.Tensor]:
    """Checkpoint names -> model tensors (int4 dequantised, floats cast to the model's dtype; `expert_bias` stays
    fp32 - the module's buffer is fp32 whatever the model dtype, M:205, and from_pretrained keeps it so)."""
    sd = {}
    for k in keys:
        if k.endswith((".scales", ".qzeros")):
            continue
        if k.endswith(".qweight"):
            b = k[: -len(".qweight")]
            sd[b + ".weight"] = _cast(_dequant(ck, b, checked), dtype)
        elif k.endswith("expert_bias"):
            sd[k] = ck.get(k).float()
        else:
            sd[k] = _cast(ck.get(k), dtype)
    return sd


class LazyExperts:
    """One shared [E][2I][H] / [E][H][I] pair; a layer's routed experts filled on demand by a pre-hook on
    `feed_forward.experts` (its args[1] is top_k_index). Forms per expert: bf16 `.weight` (w1 / w3 / w2), int4
    (either spelling, dequant_t). `keep`: layers that get their own buffers (resident once read)."""

    def __init__(self, ck: Ckpt, cfg, keep=(), dtype=None):
        self.ck, self.cfg = ck, cfg
        self.dtype = dtype or cfg.dtype
        E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.I = I
        self.shape = ((E, 2 * I, H), (E, H, I))
        self.keep = set(keep)
        self.gate_up = torch.empty(*self.shape[0], dtype=self.dtype)
        self.down = torch.empty(*self.shape[1], dtype=self.dtype)
        self.own: dict[int, tuple] = {}
        self.lock = threading.Lock()
        self.filled = 0
        self.checked: set[str] = set()
        self.layer: int | None = None
        self.have: set[int] = set()

    def tensors(self, layer: int):
        if layer not in self.keep:
            return self.gate_up, self.down
        with self.lock:
            if layer not in self.own:
                self.own[layer] = (torch.empty(*self.shape[0], dtype=self.dtype),
                                   torch.empty(*self.shape[1], dtype=self.dtype), set())
            return self.own[layer][:2]

    def buffers(self, layer: int):
        if layer in self.keep:
            self.tensors(layer)
            return self.own[layer]
        if layer != self.layer:
            self.layer, self.have = layer, set()
        return self.gate_up, self.down, self.have

    def expert(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [2I][H] - w1 rows then w3 rows -, down [H][I]) of expert e in the model's dtype."""
        b = f"model.layers.{layer}.feed_forward.experts.{e}."
        ck = self.ck
        for g, u, d in (EXPERT_SPELLINGS["w"], EXPERT_SPELLINGS["proj"]):
            if f"{b}{g}.qweight" in ck:
                gw, uw, dw = (_dequant(ck, f"{b}{s}", self.checked) for s in (g, u, d))
                break
            if f"{b}{g}.weight" in ck:
                gw, uw, dw = (ck.get(f"{b}{s}.weight") for s in (g, u, d))
                break
        else:
            die(f"{b}w1 / gate_proj: neither bf16 nor int4 in the checkpoint")
        if gw.shape[0] != self.I:
            die(f"{b}: gate has {gw.shape[0]} rows, expected {self.I}")
        return _cast(torch.cat([gw, uw], 0), self.dtype), _cast(dw, self.dtype)

    def fill(self, layer: int, ids) -> None:
        gate_up, down, have = self.buffers(layer)
        for e in ids:
            if e in have:
                continue
            have.add(e)
            gate_up[e], down[e] = self.expert(layer, e)
            self.filled += 1

    def hook(self, layer: int):
        def fn(_mod, args):
            self.fill(layer, sorted(set(int(x) for x in args[1].reshape(-1).tolist())))
        return fn


# ------------------------------------------------------------------------------------------------
# the streamed model

def build_streamed(snapshot: str, cfg, weights: str = "bf16", keep_dense=(), keep_experts=()):
    """Lfm2MoeForCausalLM on meta; embed (= the tied head), embedding_norm and the rotary table resident; decoder
    layers streamed; routed experts lazy. `weights` names the checkpoint's form - "bf16" (the original) or "int4"
    (an export, dequantised: the engine-format reference); a snapshot of the other form is refused. The model
    carries `lfm_pf` (the prefetcher: `.seconds` is the dequant wait), `lfm_lazy`, `lfm_ckpt`, `lfm_layer_sd`."""
    m = mm()
    ck = Ckpt(snapshot)
    if weights not in ("bf16", "int4"):
        die(f"weights {weights!r}: bf16 or int4")
    if ck.form() != weights:
        die(f"{snapshot} is a {ck.form()} checkpoint, --weights {weights} asked for")
    with torch.device("meta"):
        model = m.Lfm2MoeForCausalLM(cfg)
    n = cfg.num_hidden_layers
    per: dict[int, list[str]] = {i: [] for i in range(n)}
    rest = []
    for key in sorted(ck.keys()):
        mt = re.match(r"model\.layers\.(\d+)\.", key)
        if mt:
            i = int(mt.group(1))
            if i < n and ".feed_forward.experts." not in key:
                per[i].append(key)
        else:
            rest.append(key)
    checked: set = set()
    resident = layer_state(ck, rest, cfg.dtype, checked)
    if "lm_head.weight" in resident:
        die(f"{snapshot} stores lm_head.weight: LFM2.5 ties the head to the embedding (M:632)")
    resident["lm_head.weight"] = resident["model.embed_tokens.weight"]
    lazy = LazyExperts(ck, cfg, keep_experts)

    def layer_sd(i: int) -> dict[str, torch.Tensor]:
        sd = layer_state(ck, per[i], cfg.dtype, checked)
        lp = f"model.layers.{i}."
        out = {k[len(lp):]: v for k, v in sd.items()}
        if i >= cfg.num_dense_layers:
            out["feed_forward.experts.gate_up_proj"], out["feed_forward.experts.down_proj"] = lazy.tensors(i)
        return out

    layers = list(model.model.layers)
    pf = _stream.attach(model, layers, layer_sd, resident,
                        rebuild=[(model.model, "pos_emb", lambda: type(model.model.pos_emb)(cfg))], keep=keep_dense)
    model.lm_head.weight = model.model.embed_tokens.weight          # one tensor: the tie (M:632)
    for i, layer in enumerate(layers):
        if i >= cfg.num_dense_layers:
            layer.feed_forward.experts.register_forward_pre_hook(lazy.hook(i))
    model.lfm_pf, model.lfm_lazy, model.lfm_ckpt, model.lfm_layer_sd = pf, lazy, ck, layer_sd
    return model.eval()


def hf_reference(snapshot: str, dtype: torch.dtype, num_layers: int | None = None, experts: str = DEFAULT_EXPERTS):
    """transformers' own from_pretrained model, un-streamed (tiny / test sizes only)."""
    m = mm()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot)
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers
        cfg.layer_types = cfg.layer_types[:num_layers]
    model = m.Lfm2MoeForCausalLM.from_pretrained(snapshot, config=cfg, dtype=dtype, attn_implementation="eager")
    model.config._experts_implementation = experts
    return model.eval()


# ------------------------------------------------------------------------------------------------
# the restated ops (spec 24 §2's rounding chain; 24c's host twin is generated from these)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """M:60-65: v = mean(f32(x)^2) (torch's fp32 mean over the last dim); x^ = bf16(f32(x) * rsqrt(v + eps)) - the
    FIRST rounding -; y = w * x^ in the parameter's dtype - the SECOND. The q / k norms are this over each head's 64."""
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return w * (xf * torch.rsqrt(v + eps)).to(x.dtype)


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, k: int = 4, scale: float = 1.0):
    """M:183-194 on rows x [T][H] (bf16): l = bf16(x Wg^T); s = bf16(sigmoid(l)) (computed in fp32, rounded once);
    sel = f32(s) + bias (fp32: the bias moves SELECTION only); ids = topk(sel, k) - descending, torch's CPU order on
    exact ties (Task 4) -; w = s[ids] (bf16, never sel); d = bf16(bf16(sum w) + 1e-6); w = bf16(w / d); w = bf16(w x
    scale). Returns (ids [T][k] in rank order, w bf16 [T][k], sel f32 [T][E], gap f32 [T] = 4th - 5th sel)."""
    logits = F.linear(x, gate_w)
    s = logits.sigmoid()
    sel = s + bias
    _, ids = torch.topk(sel, k=k, dim=-1)
    w = torch.gather(s, dim=1, index=ids).type_as(logits)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
    w = w * scale
    top = sel.topk(min(k + 1, sel.shape[-1]), dim=-1).values
    gap = top[:, k - 1] - top[:, k] if top.shape[-1] > k else torch.full((sel.shape[0],), math.inf)
    return ids, w, sel, gap


def dense_mlp(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """M:127-128: w2(rne(rne(silu(rne(w1 x))) * rne(w3 x))) - every op in the dtype."""
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def expert_outputs(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, ids: torch.Tensor,
                   impl: str = DEFAULT_EXPERTS) -> torch.Tensor:
    """Each (token, slot)'s expert output BEFORE its weight, [T][k][H], computed exactly as `impl` computes it:
    "grouped_mm" - integrations/moe.py:388-480 (pairs sorted by expert, the two grouped linears, a = rne(rne(silu(g))
    * u)), un-permuted; "eager" - M:156-164 (per expert ascending, F.linear over its rows)."""
    T, k = ids.shape
    if impl == "grouped_mm":
        from transformers.integrations.moe import _grouped_linear
        E = gate_up.shape[0]
        flat = ids.reshape(-1)
        eg, perm = torch.sort(flat)
        xs = x[perm // k]
        tpe = torch.histc(eg.float(), bins=E, min=0, max=E - 1)
        offs = torch.cumsum(tpe, dim=0, dtype=torch.int32)
        p = _grouped_linear(xs, gate_up, offs)
        g, u = p.chunk(2, dim=-1)
        p = _grouped_linear(F.silu(g) * u, down, offs)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        return p[inv].view(T, k, -1)
    if impl == "eager":
        out = torch.empty(T, k, x.shape[-1], dtype=x.dtype)
        for e in sorted(set(ids.reshape(-1).tolist())):
            tok, slot = torch.where(ids == e)
            g, u = F.linear(x[tok], gate_up[e]).chunk(2, dim=-1)
            out[tok, slot] = F.linear(F.silu(g) * u, down[e])
        return out
    raise ValueError(impl)


def combine(proj: torch.Tensor, w: torch.Tensor, order: str = DEFAULT_EXPERTS, ids: torch.Tensor | None = None):
    """The MoE sum over slots of proj [T][k][H] (each expert's output, bf16) and w [T][k] (bf16).
    "grouped_mm" (integrations/moe.py:484-495): t_k = rne(proj_k * w_k), then ONE fp32 sum of the k products in
    SLOT order (descending sel: the router's rank) and one rounding - restated as the sequential fp32 chain
    ((t_0 + t_1) + t_2) + t_3 (test_combine_order holds it bitwise to torch's bf16 .sum(dim=1)).
    "eager" (M:150-166): the same t_k, accumulated with index_add_ into a bf16 zero row in ASCENDING expert id,
    every add rounded (needs `ids`)."""
    t = proj * w.unsqueeze(-1)
    if order == "grouped_mm":
        acc = t[:, 0].float()
        for j in range(1, t.shape[1]):
            acc = acc + t[:, j].float()
        return acc.to(proj.dtype)
    if order == "eager":
        if ids is None:
            raise ValueError("combine(order='eager') needs the ids: its order is the expert id")
        out = torch.zeros(t.shape[0], t.shape[2], dtype=proj.dtype)
        order_k = ids.argsort(dim=-1)                  # ascending expert id per row (ids of a row are distinct)
        for j in range(t.shape[1]):
            out = out + t.gather(1, order_k[:, j, None, None].expand(-1, 1, t.shape[2]))[:, 0]
        return out
    raise ValueError(order)


def conv_taps(bx: torch.Tensor, conv_w: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """The depthwise causal conv over Bx rows [T][H] (bf16) with `history` [2][H] (Bx of the two positions
    before, oldest first; zeros at the start): c[t] = rne(w0 * Bx[t-2] + w1 * Bx[t-1] + w2 * Bx[t]) with conv_w
    [H][1][3] (or [H][3]) - F.conv1d's cross-correlation, weight[..., -1] against the newest. Computed by the SAME
    torch call transformers' prefill path makes (causal_conv1d_fn, M:388-394: F.conv1d over [history, Bx] at
    padding 2, the last T outputs), whose accumulation the facts sheet measures against fp32 fma taps-ascending."""
    w = conv_w.reshape(conv_w.shape[0], -1)
    full = torch.cat([history, bx], 0).t().unsqueeze(0)           # [1][H][2 + T]
    out = F.conv1d(full.to(w.dtype), w.unsqueeze(1), None, padding=w.shape[-1] - 1, groups=w.shape[0])
    return out[0, :, 2: 2 + bx.shape[0]].t()


def short_conv(x: torch.Tensor, in_w: torch.Tensor, conv_w: torch.Tensor, out_w: torch.Tensor,
               state: torch.Tensor | None = None):
    """Lfm2MoeShortConv (M:437-465) on normed rows x [T][H]: B, C, xx = chunk3(rne(x in_w^T)) (rows [0, H) B, [H, 2H)
    C, [2H, 3H) x); Bx = rne(B * xx); c = conv_taps(Bx, conv_w, state) - `state` [2][H] the previous two Bx (zeros
    when None: positions < 0 convolve against zeros); y = rne(C * c); out = rne(y out_w^T). Returns (out [T][H],
    the new state [2][H] = Bx of the last two positions, oldest first)."""
    H = x.shape[-1]
    if state is None:
        state = torch.zeros(2, H, dtype=x.dtype)
    bcx = F.linear(x, in_w)
    B, C, xx = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
    bx = B * xx
    c = conv_taps(bx, conv_w, state)
    y = C * c
    new = torch.cat([state, bx], 0)[-2:].clone()
    return F.linear(y, out_w), new


# ------------------------------------------------------------------------------------------------
# recording

class Recorder:
    """Per layer and row (the golden layout, module docstring): h after the layer, the mixer output, the FFN
    output, the routes; `conv_states(cache)` the conv state after a prompt. `h_from` / `act_from` = the first
    absolute row kept for h / mixer + ffn. Rows are bucketed by `key` (layer_major's item) - 0 for a sequential run."""

    def __init__(self, model, cfg, h_from: int = 0, act_from: int = 0, acts: bool = True):
        self.cfg, self.k, self.n = cfg, cfg.num_experts_per_tok, cfg.num_hidden_layers
        self.h_from, self.act_from, self.acts = h_from, act_from, acts
        self.key = 0
        self.base: dict = {}
        self.rows: dict = {}
        self.hooks = []
        for i, layer in enumerate(model.model.layers[: self.n]):
            mixer = layer.self_attn if layer.is_attention_layer else layer.conv
            if acts:
                self.hooks += [layer.register_forward_hook(self._act(f"h.L{i}", "h")),
                               mixer.register_forward_hook(self._act(f"mixer.L{i}", "act")),
                               layer.feed_forward.register_forward_hook(self._act(f"ffn.L{i}", "act"))]
            if i >= cfg.num_dense_layers:
                self.hooks.append(layer.feed_forward.gate.register_forward_hook(self._router(i)))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _put(self, name, t):
        self.rows.setdefault(self.key, {}).setdefault(name, []).append(t)

    def _act(self, name, which):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach()[0]
            frm = self.h_from if which == "h" else self.act_from
            skip = max(0, frm - self.base.get(self.key, 0))
            if skip < t.shape[0]:
                self._put(name, t[skip:].to(torch.bfloat16).clone())
        return fn

    def _router(self, i):
        def fn(_m, args, out):
            logits, w, ids = out
            sel = logits.detach().sigmoid() + args[1]
            top = sel.topk(min(self.k + 1, sel.shape[-1]), dim=-1).values.float()
            self._put(f"route.ids.L{i}", ids.detach().to(torch.int32).clone())
            self._put(f"route.w.L{i}", w.detach().float().clone())
            self._put(f"route.sel.L{i}", top.clone())
            self._put(f"route.gap.L{i}", (top[:, self.k - 1] - top[:, self.k]).clone())
        return fn

    def tensors(self, key=0) -> dict[str, torch.Tensor]:
        return {k: torch.cat(v, 0).contiguous() for k, v in self.rows.get(key, {}).items()}


def conv_states(cache, cfg) -> dict[str, torch.Tensor]:
    """`conv.L<l>` bf16 [2][H]: the newest two columns of transformers' 3-column conv state (Bx of positions T-2,
    T-1, oldest first; zeros where the prompt was shorter) - what a cached decode step reads (M:368-371)."""
    out = {}
    for i, t in enumerate(cfg.layer_types[: cfg.num_hidden_layers]):
        if t == "conv":
            st = cache.layers[i].conv_states[0]                       # [1][H][3]
            out[f"conv.L{i}"] = st[0, :, -2:].t().contiguous().to(torch.bfloat16)
    return out


def gap_line(g: torch.Tensor, what: str) -> str:
    g = g[torch.isfinite(g)].double()
    if g.numel() == 0:
        return f"{what}: no rows"
    qs = torch.quantile(g, torch.tensor([0.0, 0.0001, 0.001, 0.01, 0.5], dtype=torch.float64)).tolist()
    return (f"{what} over {g.numel()} (row, layer): ==0 {int((g == 0).sum())}, <1e-4 {int((g < 1e-4).sum())}, "
            f"<1e-3 {int((g < 1e-3).sum())}, <1e-2 {int((g < 1e-2).sum())}; min/p0.01%/p0.1%/p1%/median "
            + " / ".join(f"{x:.3g}" for x in qs))


def gap_summary(t: dict, cfg) -> list[str]:
    k = cfg.num_experts_per_tok
    moe = [i for i in range(cfg.num_dense_layers, cfg.num_hidden_layers) if f"route.gap.L{i}" in t]
    if not moe:
        return ["no MoE layer"]
    lines = [gap_line(torch.cat([t[f"route.gap.L{i}"] for i in moe]), f"MoE {k}th/{k + 1}th sel gap (all layers)")]
    for i in moe:
        lines.append("  " + gap_line(t[f"route.gap.L{i}"], f"MoE {k}th/{k + 1}th sel gap L{i}"))
    return lines


# ------------------------------------------------------------------------------------------------
# layer-major batches (oracle_generate --batch's shape)

def layer_major(model, pf=None, rec: "Recorder | None" = None):
    """-> step_many(items, keys=None) for build_streamed's model: items = [(ids, pos, state)], state a dict whose
    "cache" is None before the first chunk. Returns per item the residual after the last layer [T][H] (before
    embedding_norm) - logits via final_logits. BITWISE what model(input_ids=[ids], past_key_values=cache,
    use_cache=True) computes: Lfm2MoeModel.forward's prologue (M:584-609) with the module's own functions, each
    decoder layer called with the arguments its loop passes (M:612-620). Layer-major: every item through layer 0,
    then layer 1, ... with stream.py's hold mode (each layer's weights read once per pass)."""
    m = mm()
    tm = model.model
    cfg = tm.config
    pf = pf or model.lfm_pf
    layers = list(tm.layers[: cfg.num_hidden_layers])

    def prologue(ids, pos, st):
        h = tm.embed_tokens(torch.tensor([list(ids)]))
        if st.get("cache") is None:
            st["cache"] = m.DynamicCache(config=cfg)
        cache = st["cache"]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a cache of {past} tokens")
        position_ids = (torch.arange(h.shape[1]) + past).unsqueeze(0)
        mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": cache,
              "position_ids": position_ids}
        masks = {"full_attention": m.create_causal_mask(**mk), "conv": m.create_recurrent_attention_mask(**mk)}
        return {"h": h, "pe": tm.pos_emb(h, position_ids=position_ids), "masks": masks, "cache": cache,
                "past": past, "pid": position_ids}

    def select(key, past):
        if rec is not None:
            rec.key = key
            rec.base[key] = past

    @torch.no_grad()
    def step_many(items, keys=None):
        require_version()
        keys = list(range(len(items))) if keys is None else list(keys)
        ps = []
        for j, (ids, pos, st) in enumerate(items):
            select(keys[j], pos)
            ps.append(prologue(ids, pos, st))
        pf.hold = True
        try:
            for i, layer in enumerate(layers):
                for j, p in enumerate(ps):
                    select(keys[j], p["past"])
                    p["h"] = layer(p["h"], attention_mask=p["masks"][cfg.layer_types[i]], position_ids=p["pid"],
                                   past_key_values=p["cache"], position_embeddings=p["pe"])
                pf.release()
        finally:
            pf.hold = False
            pf.release()
        out = []
        for p in ps:
            out.append(p["h"][0])
            p.clear()
        return out
    return step_many


@torch.no_grad()
def final_logits(model, R: torch.Tensor) -> torch.Tensor:
    """embedding_norm then the tied head over residual rows R [T][H] (M:622, M:690)."""
    return model.lm_head(model.model.embedding_norm(R[None]))[0]


# ------------------------------------------------------------------------------------------------
# the port: the restated ops chained (hfcheck's other arm)

@torch.no_grad()
def port_forward(model, ids: list[int], st: dict, impl: str = DEFAULT_EXPERTS, logits_to_keep: int = 0,
                 rec: dict | None = None):
    """One forward of `ids` at st["pos"] (0 when absent) through the RESTATED ops - rms_norm, short_conv (state
    st["conv"][l] [2][H]), route + expert_outputs + combine, dense_mlp, embedding_norm and the tied head - with
    transformers' attention module (and its DynamicCache, st["cache"]) for the attention layers. Each layer's
    weights are materialised from build_streamed's layer_sd and dropped after, as the hooks would. Returns the
    logits of the last `logits_to_keep` rows (0 = all); `rec` (a dict) gets h.L / mixer.L / ffn.L / route.* rows
    (the Recorder's names) appended."""
    m = mm()
    tm = model.model
    cfg = tm.config
    eps = cfg.norm_eps
    pos = st.setdefault("pos", 0)
    if st.get("cache") is None:
        st["cache"] = m.DynamicCache(config=cfg)
    conv = st.setdefault("conv", {})
    cache = st["cache"]
    x = tm.embed_tokens(torch.tensor([list(ids)]))
    pid = (torch.arange(x.shape[1]) + pos).unsqueeze(0)
    mask = m.create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache,
                                position_ids=pid)
    pe = tm.pos_emb(x, position_ids=pid)
    h = x[0]

    def put(name, t):
        if rec is not None:
            rec.setdefault(name, []).append(t.detach().clone())
    for i in range(cfg.num_hidden_layers):
        layer = tm.layers[i]
        sd = model.lfm_layer_sd(i)
        g = lambda n: sd[n]                                   # noqa: E731
        r = h
        xn = rms_norm(h, g("operator_norm.weight"), eps)
        if cfg.layer_types[i] == "full_attention":
            att = layer.self_attn
            att.load_state_dict({k[len("self_attn."):]: v for k, v in sd.items() if k.startswith("self_attn.")},
                                strict=True, assign=True)
            y = att(hidden_states=xn[None], position_embeddings=pe, attention_mask=mask, past_key_values=cache)[0][0]
            att.to_empty(device="meta")
        else:
            y, conv[i] = short_conv(xn, g("conv.in_proj.weight"), g("conv.conv.weight"), g("conv.out_proj.weight"),
                                    conv.get(i))
        put(f"mixer.L{i}", y)
        h = y + r
        xn = rms_norm(h, g("ffn_norm.weight"), eps)
        if i < cfg.num_dense_layers:
            f = dense_mlp(xn, g("feed_forward.w1.weight"), g("feed_forward.w3.weight"), g("feed_forward.w2.weight"))
        else:
            rids, w, sel, gap = route(xn, g("feed_forward.gate.weight"), g("feed_forward.expert_bias"),
                                      cfg.num_experts_per_tok, cfg.routed_scaling_factor)
            model.lfm_lazy.fill(i, sorted(set(rids.reshape(-1).tolist())))
            gu, dn = model.lfm_lazy.tensors(i)
            f = combine(expert_outputs(xn, gu, dn, rids, impl), w, impl, rids)
            k = cfg.num_experts_per_tok
            put(f"route.ids.L{i}", rids.to(torch.int32))
            put(f"route.w.L{i}", w.float())
            top = sel.topk(min(k + 1, sel.shape[-1]), dim=-1).values.float()
            put(f"route.sel.L{i}", top)
            put(f"route.gap.L{i}", gap.float())
        put(f"ffn.L{i}", f)
        h = h + f
        put(f"h.L{i}", h)
    st["pos"] = pos + len(ids)
    hn = rms_norm(h, tm.embedding_norm.weight, eps)
    if logits_to_keep:
        hn = hn[-logits_to_keep:]
    return F.linear(hn, tm.embed_tokens.weight)


# ------------------------------------------------------------------------------------------------
# a sequential run (prompt in chunks, then greedy decode)

def read_ids(path: str) -> list[int]:
    return [int(x) for x in open(path, encoding="utf-8").read().split()]


@torch.no_grad()
def forward_chunks(model, ids: list[int], chunk: int, cache=None, rec: "Recorder | None" = None,
                   on_logits=None, logits_to_keep: int = 0):
    """model(input_ids=chunk, past_key_values=cache) over `ids` in chunks (whole when chunk <= 0): the reference's
    prefill. on_logits(row0, logits bf16 [c][V]) sees every chunk's logits. Returns the cache."""
    pos = 0 if cache is None else cache.get_seq_length()
    c = chunk if chunk > 0 else len(ids)
    for a in range(0, len(ids), c):
        part = ids[a:a + c]
        if rec is not None:
            rec.base[rec.key] = pos + a
        out = model(input_ids=torch.tensor([part]), past_key_values=cache, use_cache=True,
                    logits_to_keep=logits_to_keep)
        cache = out.past_key_values
        if on_logits is not None:
            on_logits(pos + a + len(part) - out.logits.shape[1], out.logits[0])
    return cache


def auto_chunk(n: int, chunk: int) -> int:
    if chunk >= 0:
        return chunk
    return 0 if n <= LONG_PROMPT else 2048


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, transformers {PINNED_TRANSFORMERS}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    cfg = config(a.snapshot, a.layers, a.experts)
    ids = read_ids(a.prompt)
    T = len(ids)
    if not 1 <= T <= a.max_prompt:
        die(f"{a.prompt}: {T} ids (1..{a.max_prompt})")
    model = build_streamed(a.snapshot, cfg, a.weights)
    chunk = auto_chunk(T, a.chunk)
    act_from = max(0, T - a.act_tail) if a.act_tail >= 0 else 0
    h_from = 0 if T <= LONG_PROMPT else act_from
    log_from = max(0, T - max(1, a.logits_tail)) if a.logits_tail >= 0 else 0
    print(f"config: {cfg.num_hidden_layers} layers ({cfg.layer_types[:cfg.num_hidden_layers].count('full_attention')} "
          f"attention), {cfg.num_experts} experts top-{cfg.num_experts_per_tok}, hidden {cfg.hidden_size}, attn eager, "
          f"experts {cfg._experts_implementation}, weights {a.weights}; prompt {T} ids, chunk {chunk or 'whole'}; h "
          f"from row {h_from}, mixer / ffn from row {act_from}, logits from row {log_from}; built {time.time() - t0:.1f}s",
          flush=True)
    rec = Recorder(model, cfg, h_from=h_from, act_from=act_from)
    rows, nll = [], torch.empty(max(T - 1, 0))
    ids_t = torch.tensor(ids)

    def on_logits(r0, lg):
        for b0 in range(0, lg.shape[0], 256):
            lf = lg[b0:b0 + 256].float()
            ra = r0 + b0
            r1 = ra + lf.shape[0]
            a_, b_ = ra, min(r1, T - 1)
            if b_ > a_:
                nll[a_:b_] = -torch.log_softmax(lf[: b_ - a_], -1).gather(-1, ids_t[a_ + 1:b_ + 1, None])[:, 0]
            if r1 > log_from:
                rows.append(lf[max(0, log_from - ra):].clone())
    t1 = time.time()
    cache = forward_chunks(model, ids, chunk, rec=rec, on_logits=on_logits)
    t_prompt = time.time() - t1
    print(f"prompt forward {t_prompt:.1f}s ({model.lfm_lazy.filled} expert reads)", flush=True)
    convs = conv_states(cache, cfg)
    tokens: list[int] = []
    t2 = time.time()
    last = rows[-1][-1]
    for step in range(a.gen):
        nxt = int(torch.argmax(last).item())
        tokens.append(nxt)
        rec.base[0] = T + step
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last = out.logits[0, -1].float()
        rows.append(out.logits[0].float())
        if step == 0:
            print(f"  first decode step {time.time() - t2:.1f}s", flush=True)
    print(f"greedy {a.gen}: {time.time() - t2:.1f}s -> {tokens}", flush=True)
    rec.remove()
    t = rec.tensors()
    t.update(convs)
    t["logits"] = torch.cat(rows, 0).contiguous()
    t["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    t["nll"] = nll.float().contiguous()
    n_rows = T + a.gen
    for i in range(cfg.num_dense_layers, cfg.num_hidden_layers):
        for k in ("route.ids", "route.w", "route.sel", "route.gap"):
            if t[f"{k}.L{i}"].shape[0] != n_rows:
                die(f"{k}.L{i} has {t[f'{k}.L{i}'].shape[0]} rows, expected {n_rows}")
    if not torch.isfinite(t[f"h.L{cfg.num_hidden_layers - 1}"].float()).all():
        die("the last layer's output has NaN/Inf - the weights are wrong")
    for line in gap_summary(t, cfg):
        print(line)
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt": os.path.abspath(a.prompt), "n_prompt": str(T),
            "gen": str(a.gen), "layers": str(cfg.num_hidden_layers), "chunk": str(chunk), "weights": a.weights,
            "h_row0": str(h_from), "act_row0": str(act_from), "logits_row0": str(log_from),
            "attn_implementation": "eager", "experts_implementation": cfg._experts_implementation,
            "transformers": PINNED_TRANSFORMERS, "torch": torch.__version__,
            "ppl": f"{math.exp(float(nll.double().mean())) if T > 1 else float('nan'):.6g}",
            "model": "LFM2.5-24B-A2B (Lfm2MoeForCausalLM, tools/oracle/lfm2moe_ref.py, layer-streamed)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(t, a.out, metadata=meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    print(f"wrote {a.out}: {len(t)} tensors; prompt ppl {meta['ppl']}")
    print(f"wall {time.time() - t0:.1f}s (prompt {t_prompt:.1f}s, {a.gen} steps {time.time() - t2:.1f}s), peak RSS "
          f"{rss_gib:.1f} GiB, prefetch wait {model.lfm_pf.seconds:.1f}s")


# ------------------------------------------------------------------------------------------------
# ppl

def tokenize_text(snapshot: str, path: str, bos: int | None) -> list[int]:
    """golden-prompt rule (tools/oracle/tokenize.py): the text without its trailing newlines, no template; LFM2's
    BOS first (the model was trained with it at position 0)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    text = open(path, encoding="utf-8").read().rstrip("\n")
    return ([bos] if bos is not None else []) + tok.encode(text, add_special_tokens=False).ids


def cmd_ppl(a) -> None:
    t0 = time.time()
    cfg = config(a.snapshot, a.layers, a.experts)
    model = build_streamed(a.snapshot, cfg, a.weights)
    srcs = [(n, tokenize_text(a.snapshot, p, cfg.bos_token_id)) for n, p in (x.split("=", 1) for x in a.text)]
    srcs += [(n, read_ids(p)) for n, p in (x.split("=", 1) for x in a.ids)]
    if not srcs:
        die("ppl: no --text / --ids")
    for name, ids in srcs:
        ids = ids[: a.max_tokens]
        tot, n = 0.0, 0
        ids_t = torch.tensor(ids)

        def on_logits(r0, lg):
            nonlocal tot, n
            for b0 in range(0, lg.shape[0], 256):
                ra = r0 + b0
                b_ = min(ra + min(256, lg.shape[0] - b0), len(ids) - 1)
                if b_ > ra:
                    lp = torch.log_softmax(lg[b0:b0 + b_ - ra].float(), -1).gather(-1, ids_t[ra + 1:b_ + 1, None])[:, 0]
                    tot -= float(lp.double().sum())
                    n += b_ - ra
        t1 = time.time()
        forward_chunks(model, ids, auto_chunk(len(ids), a.chunk), on_logits=on_logits)
        ppl = math.exp(tot / max(n, 1))
        print(f"ppl {name}: {ppl:.4f} over {n} predicted tokens ({len(ids)} ids, {cfg.num_hidden_layers} layers, "
              f"{time.time() - t1:.0f}s)" + ("" if math.isfinite(ppl) else "  NOT SANE (not finite)"), flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"wall {time.time() - t0:.0f}s, peak RSS {rss / 2**30 if sys.platform == 'darwin' else rss / 2**20:.1f} GiB")


# ------------------------------------------------------------------------------------------------
# hfcheck: transformers' own model against the port (the restated ops), on any weights

@torch.no_grad()
def hfcheck_arms(snapshot: str, cfg, ids: list[int], gen: int, weights: str = "bf16", chunk: int = 0):
    """Both arms on the same streamed checkpoint: transformers' model (forward_chunks + cached greedy steps, the
    Recorder) and port_forward (the same chunks and steps). Returns ((tensors, tokens), (tensors, tokens))."""
    model = build_streamed(snapshot, cfg, weights)
    res = []
    rec = Recorder(model, cfg)
    rows = []
    cache = forward_chunks(model, ids, chunk, rec=rec, on_logits=lambda r0, lg: rows.append(lg))
    toks = []
    for step in range(gen):
        nxt = int(rows[-1][-1].float().argmax())
        toks.append(nxt)
        rec.base[0] = len(ids) + step
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    rec.remove()
    t = rec.tensors()
    t.update(conv_states(cache, cfg))
    t["logits"] = torch.cat(rows)
    res.append((t, toks))
    st, prec, prows = {}, {}, []
    c = chunk if chunk > 0 else len(ids)
    for a0 in range(0, len(ids), c):
        prows.append(port_forward(model, ids[a0:a0 + c], st, rec=prec))
    ptoks = []
    for step in range(gen):
        nxt = int(prows[-1][-1].float().argmax())
        ptoks.append(nxt)
        prows.append(port_forward(model, [nxt], st, rec=prec))
    pt = {k: torch.cat(v, 0) for k, v in prec.items()}
    for i, s in st["conv"].items():
        pt[f"conv.L{i}"] = s.contiguous()
    pt["logits"] = torch.cat(prows)
    res.append((pt, ptoks))
    return res[0], res[1]


def cmd_hfcheck(a) -> None:
    t0 = time.time()
    cfg = config(a.snapshot, a.layers, a.experts)
    if a.prompt:
        ids = read_ids(a.prompt)[: a.n]
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, 124000, (a.n,), generator=g).tolist()
    (ht, htok), (pt, ptok) = hfcheck_arms(a.snapshot, cfg, ids, a.gen, a.weights, auto_chunk(len(ids), a.chunk))
    keys = sorted(set(ht) | set(pt))
    bad = [k for k in keys if k not in ht or k not in pt or ht[k].shape != pt[k].shape or not torch.equal(ht[k], pt[k])]
    print(f"hfcheck: {len(keys)} tensors ({cfg.num_hidden_layers} layers, {len(ids)} ids + {a.gen} cached steps, "
          f"weights {a.weights}), greedy {htok} vs {ptok}; differing: {bad[:8] if bad else 'none'}; "
          f"{time.time() - t0:.0f}s")
    if bad or ptok != htok:
        for k in bad[:8]:
            if k in ht and k in pt and ht[k].shape == pt[k].shape:
                print(f"  {k}: max |diff| {(ht[k].float() - pt[k].float()).abs().max().item():.3e}")
        print("hfcheck: FAIL (the restated ops are not transformers' arithmetic on these weights)")
        sys.exit(1)
    print(f"hfcheck: BITWISE ({cfg.num_hidden_layers} layers, {len(ids)} ids + {a.gen} cached steps: logits, every "
          f"h / mixer / ffn row, the routes, the conv states)")


# ------------------------------------------------------------------------------------------------
# facts (the facts sheet's tables: lfm2moe_facts.py)

def topk_ties(n: int = 64, k: int = 4, trials: int = 50) -> list[str]:
    """Which index torch.topk returns on planted exact ties at the k-th / (k+1)-th of n fp32 values (the router's
    sel, M:187): count of trials where the lower id wins, and whether repeated calls agree."""
    g = torch.Generator().manual_seed(0)
    lower = higher = stable = 0
    for _ in range(trials):
        v = torch.rand(n, generator=g)
        order = v.argsort(descending=True)
        a_, b_ = sorted(order[k - 1:k + 1].tolist())
        v[b_] = v[a_]
        idx = v.topk(k).indices.tolist()
        lower += (a_ in idx) and (b_ not in idx)
        higher += (b_ in idx) and (a_ not in idx)
        stable += idx == v.topk(k).indices.tolist()
    return [f"torch {torch.__version__} CPU topk(k={k}) of {n} fp32 with an exact tie at the {k}th / {k + 1}th: the "
            f"lower id wins {lower}/{trials}, the higher id {higher}/{trials}; repeat calls identical {stable}/{trials}"]


def cmd_facts(a) -> None:
    facts = _load("lfm2moe_facts")
    facts.main([a.snapshot] + (["--tiny", a.tiny] if a.tiny else []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("snapshot")
        p.add_argument("--layers", type=int)
        p.add_argument("--experts", default=DEFAULT_EXPERTS, choices=["grouped_mm", "eager"])
        p.add_argument("--weights", default="bf16", choices=["bf16", "int4"],
                       help="the checkpoint's form: bf16 (the original) or int4 (an export, dequantised)")
        p.add_argument("--chunk", type=int, default=-1, help="prefill chunk (-1: whole up to 4096 ids, else 2048)")
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=131072)
    r.add_argument("--logits-tail", type=int, default=1024, help="keep logits from row T - L (-1: all)")
    r.add_argument("--act-tail", type=int, default=256, help="keep mixer / ffn (and h past 4096 ids) from row T - A")
    p = sub.add_parser("ppl")
    common(p)
    p.add_argument("--text", action="append", default=[], help="NAME=FILE (the snapshot's tokenizer.json, BOS first)")
    p.add_argument("--ids", action="append", default=[], help="NAME=FILE")
    p.add_argument("--max-tokens", type=int, default=4096)
    h = sub.add_parser("hfcheck")
    common(h)
    h.add_argument("--prompt")
    h.add_argument("--n", type=int, default=2048)
    h.add_argument("--gen", type=int, default=4)
    f = sub.add_parser("facts")
    f.add_argument("snapshot", help="the LFM2.5 snapshot (its small files; shard headers by mmap where present)")
    f.add_argument("--tiny", help="a tiny snapshot for the measured rows (sdpa, conv accumulation)")
    a = ap.parse_args()
    {"run": cmd_run, "ppl": cmd_ppl, "hfcheck": cmd_hfcheck, "facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
