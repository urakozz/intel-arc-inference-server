#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a: the CPU reference of NVIDIA Nemotron 3.5 Bolt 74B-A9B (`nemotron_h_omni` / `nemotron_h_puzzle`), B1.

The model is the checkpoint's OWN remote code (R = modeling_nemotron_h.py, blob de290021, imported from the pinned
snapshot by bolt_remote.py - never vendored) run through transformers 5.19.0 (refused on any other version), built on
`meta` and LAYER-STREAMED (stream.py: the embedding, norm_f and the untied lm_head resident; each block materialised by
a forward pre-hook and dropped after). Routed experts are read on demand - only the ones a token routes to, from a
forward hook on the router - bf16 from the original or dequantised from int4 (SergiioB's g128 via the exact
g128 -> g64 expansion, or our g64) by stream.dequant_t: the ENGINE-FORMAT reference.

The Mamba mixers run R's kernel path (`cuda_kernels_forward`) on the CPU through bolt_mamba_ops' plain-torch training
kernels (spec 25 Decision 1: fp32 inside, no dt clamp, the twin's orders), with an fp32 SSM / conv cache passed in
(R:1664-1667 would build it in the model's dtype); `--path torch` runs R's torch fallback instead (comparisons only).
Two departures from R, each tested: the streamed MoE skips R's no-op "dummy" forward of experts no token routes to
(R:941-945: it adds exactly +0 - test_moe_dummy_noop), and a prompt chunk after a cached prefix runs
bolt_remote.continue_forward (R's kernel path has no such branch).

    bolt_ref.py run <snapshot> --prompt <ids> --out <p>.golden.safetensors [--gen 32] [--layers N] [--weights int4|bf16]
                    [--chunk C] [--taps] [--cache fp32|bf16] [--path kernel|torch] [--logits-tail L] [--act-tail A]
    bolt_ref.py ppl <snapshot> [--text NAME=FILE ...] [--ids NAME=FILE ...] [--layers N] [--max-tokens T]
    bolt_ref.py hfcheck <snapshot> --layers N [--prompt <ids>] [--n 2048] [--gen 4] [--dtype bf16|fp32]
    bolt_ref.py facts ...      (bolt_facts.py)
    bolt_ref.py taps <snapshot> --source NAME:IDS:N_PROMPT [...] --out DIR     (dump_taps' format, spec 26a's input)
    bolt_ref.py gateq --ref DIR --cand DIR [--a4-ref DIR --a4-cand DIR]        (spec 25 B5's table)

`run`'s golden layout (rows are absolute positions; `h_row0` / `logits_row0` in the metadata):
    h.L<l>          bf16 [A][4096]  the residual after block l: every row for T <= 4096, else the last --act-tail
                                    (256) prompt rows; plus every generated row
    route.ids.L<l>  i32  [T+gen][9] MoE blocks: the selected ids ASCENDING (the set; torch's sorted=False order is
                                    not kept)
    route.w.L<l>    f32  [T+gen][9] the weights at those ids (s / (sum + 1e-20) x 5.0, fp32, never rounded)
    route.sel.L<l>  f32  [T+gen][10] the top 10 sel = sigmoid(fp32 logit) + e_score_correction_bias, descending
    route.gap.L<l>  f32  [T+gen]    sel(9th) - sel(10th) (0 = an exact tie at the cut: undetermined)
    route.sum.L<l>  f32  [T+gen]    the sum of the nine s in R's order (before the + 1e-20)
    ssm.L<l>        f32  [H][P][N]  Mamba blocks: the SSM state after the prompt ([128][64][128] on Bolt)
    conv.L<l>       bf16 [3][C]     the last three RAW xBC columns before the first generated row, oldest first
    tap.L<i>        bf16 [T+gen][4096]  (--taps) the residual after block i for i in 1, 18, 35, 51, 68, 85 (those
                                    that exist), every row: dump_taps' names, = output_hidden_states[i + 1]
    logits          f32  [Lr][V]    rows logits_row0 .. T+gen-1 (the last --logits-tail 1024 prompt rows)
    tokens          i32  [gen]      greedy (the lower id on an exact tie);  nll f32 [T-1] the prompt's NLL
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
REMOTE_SHA = "de290021078450c4d5042f66dd74b8eecf1d3eb3"        # modeling_nemotron_h.py's HF blob (git sha1)
TAP_LAYERS = (1, 18, 35, 51, 68, 85)                            # LiLiCorr's target layers (spec 26)
FIRST_ATTENTION = 7
LONG_PROMPT = 4096
INT4 = (".qweight", ".qzeros", ".scales")
TOP_K = 9


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("bolt_remote")
ops = _load("bolt_mamba_ops")
_stream = _load("stream")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def require_version() -> None:
    br.require_version()


def remote_for(snapshot: str) -> str:
    """Where R comes from: the snapshot itself when it ships the remote code (a real checkpoint), else the pinned
    snapshot in the HF cache (the tiny variants)."""
    if os.path.exists(os.path.join(snapshot, "modeling_nemotron_h.py")):
        return snapshot
    s = br.find_snapshot()
    if s is None:
        die(f"{snapshot} has no remote code and the pinned Bolt snapshot is not in the HF cache")
    return s


def ns_for(snapshot: str):
    return br.load_remote(remote_for(snapshot))


# ------------------------------------------------------------------------------------------------
# the config and the names

def read_omni(snapshot: str) -> dict:
    require_version()
    from transformers.configuration_utils import PreTrainedConfig
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        c = PreTrainedConfig._decode_special_floats(json.load(f))
    if c.get("model_type") != "nemotron_h_omni":
        die(f"{snapshot}: model_type {c.get('model_type')!r}, expected nemotron_h_omni")
    return c


def config(snapshot: str, layers: int | None = None) -> dict:
    """config.json's llm_config (a dict); layers=N truncates block_configs[:N] (the engine's --layers N: blocks
    [0, N), then norm_f and the head). N >= 8: R's cache takes the sequence length from an attention block
    (R:247-253, `transformer_layers[0]`), and the first is block 7."""
    d = dict(read_omni(snapshot)["llm_config"])
    if d.get("model_type") != "nemotron_h_puzzle":
        die(f"{snapshot}: llm_config.model_type {d.get('model_type')!r}, expected nemotron_h_puzzle")
    if d.get("mtp_block_configs"):
        die(f"{snapshot}: MTP blocks in the config (Bolt has none)")
    d["block_configs"] = [dict(b) for b in d["block_configs"]]
    if layers is not None:
        n = len(d["block_configs"])
        if not 1 <= layers <= n:
            die(f"--layers {layers}: the checkpoint has {n}")
        kinds = [b["block_type"] for b in d["block_configs"]]
        if "attention" not in kinds[:layers]:
            first = kinds.index("attention") if "attention" in kinds else None
            die(f"--layers {layers} keeps no attention block (the first is block {first}): R's cache takes the "
                f"sequence length from an attention block, so N >= {None if first is None else first + 1}")
        d["block_configs"] = d["block_configs"][:layers]
    return d


def kinds(cfg: dict) -> list[str]:
    return [b["block_type"] for b in cfg["block_configs"]]


def expected_names(cfg: dict, form: str) -> list[str]:
    """Every language-model tensor of a checkpoint of `cfg` in `form` (the vision / projector tensors excluded):
      "bf16"       NVIDIA's BF16 original: `language_model.backbone.*` (the index's spelling), every linear `.weight`
      "gptq-g128"  SergiioB's: `language_model.model.*`, every linear `.qweight` / `.qzeros` / `.scales` / `.g_idx`
      "ar-g64"     ours (25q): `language_model.model.*`, `.qweight` / `.qzeros` / `.scales` (no g_idx)
    The kept tensors (router, its F32 bias, A_log / D / dt_bias, conv, norms, embeddings, lm_head) as shipped."""
    if form not in ("bf16", "gptq-g128", "ar-g64"):
        raise ValueError(form)
    trunk = "backbone" if form == "bf16" else "model"
    sfx = [".weight"] if form == "bf16" else list(INT4) + ([".g_idx"] if form == "gptq-g128" else [])

    def lin(base):
        return [base + s for s in sfx]
    p = f"language_model.{trunk}."
    names = [p + "embeddings.weight", p + "norm_f.weight", "language_model.lm_head.weight"]
    for i, b in enumerate(cfg["block_configs"]):
        m = f"{p}layers.{i}.mixer."
        names.append(f"{p}layers.{i}.norm.weight")
        t = b["block_type"]
        if t == "mamba":
            names += [m + s for s in ("A_log", "D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight")]
            names += lin(m + "in_proj") + lin(m + "out_proj")
        elif t == "attention":
            for s in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names += lin(m + s)
        elif t == "moe":
            names += [m + "gate.weight", m + "gate.e_score_correction_bias"]
            for s in ("fc1_latent_proj", "fc2_latent_proj", "shared_experts.up_proj", "shared_experts.down_proj"):
                names += lin(m + s)
            for e in range(b["n_routed_experts"]):
                names += lin(f"{m}experts.{e}.up_proj") + lin(f"{m}experts.{e}.down_proj")
        else:
            raise ValueError(t)
    return names


# ------------------------------------------------------------------------------------------------
# the checkpoint

class Ckpt:
    """Name -> tensor over a snapshot's shards (the index, or a single model.safetensors), R's names in and out:
    `language_model.` + R's name, `model.` read as `backbone.` when the checkpoint spells the trunk so (NVIDIA)."""

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
        self.trunk = "backbone" if any(k.startswith("language_model.backbone.") for k in self.wm) else "model"
        self.checked: set[str] = set()
        self.lock = threading.Lock()

    def c(self, rname: str) -> str:
        if self.trunk == "backbone" and rname.startswith("model."):
            rname = "backbone." + rname[len("model."):]
        return "language_model." + rname

    def has(self, rname: str, suffix: str = "") -> bool:
        return self.c(rname) + suffix in self.wm

    def raw(self, cname: str) -> torch.Tensor:
        return self.h[self.wm[cname]].get_tensor(cname)

    def form(self) -> str:
        return "int4" if any(k.endswith(".qweight") for k in self.wm) else "bf16"

    def weight(self, rname: str) -> torch.Tensor:
        """R's parameter `rname` (a `.weight` of a linear may be int4: dequantised; a g128 linear through the exact
        g128 -> g64 expansion - each scale row written to the two g64 rows it covers -, then (q - 8) x scale)."""
        if rname.endswith(".weight") and self.has(rname[: -len(".weight")], ".qweight"):
            return self.dequant(rname[: -len(".weight")])
        c = self.c(rname)
        if c not in self.wm:
            die(f"{self.dir}: {c} absent")
        return self.raw(c)

    def dequant(self, rbase: str) -> torch.Tensor:
        b = self.c(rbase)
        qw, sc = self.raw(b + ".qweight"), self.raw(b + ".scales")
        g = qw.shape[0] * 8 // sc.shape[0]
        first = False
        with self.lock:
            if b not in self.checked:
                self.checked.add(b)
                first = True
        if first:
            qz = self.raw(b + ".qzeros")
            if not bool((qz == 0x77777777).all()):
                die(f"{b}.qzeros is not 0x77777777: not symmetric int4 (the engine's rule is (q - 8) x scale)")
            if b + ".g_idx" in self.wm:
                gi = self.raw(b + ".g_idx")
                if not torch.equal(gi, torch.arange(gi.numel(), dtype=gi.dtype) // g):
                    die(f"{b}.g_idx is not i // {g} (act-order is not the engine's format)")
            if sc.dtype != torch.float16:
                die(f"{b}.scales is {sc.dtype}, expected F16")
        if g == 128:
            sc, g = sc.repeat_interleave(2, 0).contiguous(), 64                  # the exact expansion (docs/13)
        elif g != 64:
            die(f"{b}: group {g} (the engine's format is g64, or g128 through its exact expansion)")
        return _stream.dequant_t(qw, sc, g, check=first)


def _cast(name: str, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The model's dtype for every float, except the router bias - kept fp32 (R:997, R:1054-1063)."""
    if name.endswith("e_score_correction_bias"):
        return t.float()
    return t.to(dtype) if t.is_floating_point() and t.dtype != dtype else t


class LazyExperts:
    """The routed experts of a streamed MoE block, materialised from the router's forward hook (its output's
    topk_indices) - only the experts some row routes to; R's dummy forward of the others is skipped (moe_routed_only).
    `cache_bytes` > 0 keeps dequantised experts in an LRU across forwards (decode-heavy runs)."""

    def __init__(self, ck: Ckpt, model, dtype: torch.dtype, cache_bytes: int = 0):
        self.ck, self.model, self.dtype = ck, model, dtype
        self.filled = 0
        self.cache_bytes = cache_bytes
        self.lru: collections.OrderedDict = collections.OrderedDict()
        self.bytes = 0

    def tensors(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, e)
        if key in self.lru:
            self.lru.move_to_end(key)
            return self.lru[key]
        p = f"model.layers.{layer}.mixer.experts.{e}."
        up = _cast("", self.ck.weight(p + "up_proj.weight"), self.dtype)
        dn = _cast("", self.ck.weight(p + "down_proj.weight"), self.dtype)
        self.filled += 1
        if self.cache_bytes > 0:
            self.lru[key] = (up, dn)
            self.bytes += (up.numel() + dn.numel()) * up.element_size()
            while self.bytes > self.cache_bytes and self.lru:
                _, (a, b) = self.lru.popitem(last=False)
                self.bytes -= (a.numel() + b.numel()) * a.element_size()
        return up, dn

    def fill(self, layer: int, ids) -> None:
        experts = self.model.model.layers[layer].mixer.experts
        for e in ids:
            ex = experts[e]
            if not ex.up_proj.weight.is_meta:
                continue
            up, dn = self.tensors(layer, e)
            ex.up_proj.weight = torch.nn.Parameter(up, requires_grad=False)
            ex.down_proj.weight = torch.nn.Parameter(dn, requires_grad=False)

    def hook(self, layer: int):
        def fn(_mod, _args, out):
            self.fill(layer, sorted(set(int(x) for x in out[0].reshape(-1).tolist())))
        return fn


def moe_routed_only(self, hidden_states, topk_indices, topk_weights):
    """R's NemotronHMoE.moe (R:925-950) without the else branch (R:941-945): an expert no token routes to adds
    `expert(0)` = down(relu2(up(0))) = exactly 0 (no biases) to a buffer that cannot hold -0 (it starts at +0 and
    only receives adds), so skipping it changes no bit - test_moe_dummy_noop holds this against R's own moe."""
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
    expert_mask = expert_mask.permute(2, 0, 1)
    for expert_idx in range(len(self.experts)):
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)
        if token_indices.numel() > 0:
            expert = self.experts[expert_idx]
            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    return final_hidden_states.type(hidden_states.dtype)


def build_streamed(snapshot: str, cfg: dict, weights: str = "int4", dtype: torch.dtype = torch.bfloat16,
                   path: str = "kernel", keep=(), expert_cache_gb: float = 0.0):
    """R's NemotronHPuzzleForCausalLM on meta; embeddings, norm_f and lm_head resident; blocks streamed; routed experts
    lazy. `weights` names the checkpoint's form - "int4" (SergiioB's g128 or our g64, dequantised: the engine-format
    reference) or "bf16" (the original); a snapshot of the other form is refused. The model carries `bolt_pf` (the
    prefetcher), `bolt_lazy`, `bolt_ckpt`, `bolt_layer_sd`, `bolt_cfg`."""
    ns = ns_for(snapshot)
    ck = Ckpt(snapshot)
    if weights not in ("int4", "bf16"):
        die(f"weights {weights!r}: int4 or bf16")
    if ck.form() != weights:
        die(f"{snapshot} is a {ck.form()} checkpoint, --weights {weights} asked for")
    mcfg = br.make_config(cfg, dtype)
    with torch.device("meta"):
        model = ns.LM(mcfg)
    br.set_path(model, path)
    sd_meta = model.state_dict()
    n = len(cfg["block_configs"])
    per: dict[int, list[str]] = {i: [] for i in range(n)}
    resident = {}
    for k, v in sd_meta.items():
        mt = re.match(r"model\.layers\.(\d+)\.", k)
        if mt:
            per[int(mt.group(1))].append(k)
        else:
            resident[k] = _cast(k, ck.weight(k), dtype)
    lazy = LazyExperts(ck, model, dtype, int(expert_cache_gb * 2**30))

    def layer_sd(i: int) -> dict[str, torch.Tensor]:
        lp = f"model.layers.{i}."
        out = {}
        for k in per[i]:
            if ".mixer.experts." in k:
                out[k[len(lp):]] = torch.empty(sd_meta[k].shape, dtype=dtype, device="meta")
            else:
                out[k[len(lp):]] = _cast(k, ck.weight(k), dtype)
        return out

    layers = list(model.model.layers)
    pf = _stream.attach(model, layers, layer_sd, resident, keep=keep)
    for i, blk in enumerate(layers):
        if blk.block_type == "moe":
            blk.mixer.gate.register_forward_hook(lazy.hook(i))
            blk.mixer.moe = types.MethodType(moe_routed_only, blk.mixer)
    model.bolt_pf, model.bolt_lazy, model.bolt_ckpt, model.bolt_layer_sd, model.bolt_cfg = pf, lazy, ck, layer_sd, cfg
    return model.eval()


def build_resident(snapshot: str, cfg: dict, dtype: torch.dtype = torch.bfloat16, path: str = "kernel"):
    """R's model of a (tiny) checkpoint, un-streamed: build_lm + a strict load of every tensor (int4 dequantised)."""
    ns_for(snapshot)
    ck = Ckpt(snapshot)
    model = br.build_lm(cfg, dtype, path=path)
    sd = {k: _cast(k, ck.weight(k), dtype) for k in model.state_dict()}
    model.load_state_dict(sd, strict=True)
    return model.eval()


# ------------------------------------------------------------------------------------------------
# the restated ops (spec 25 §2's chain; 25c's host twin is generated from these, each tested against R's module)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """R:760-765: fp32 variance (torch's mean), x * rsqrt(var + eps), w (widened) times that in fp32, ONE rounding."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (w.to(torch.float32) * xf).to(x.dtype)


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, k: int = TOP_K, scale: float = 5.0):
    """R:1040-1052 on rows x [T][H] (n_group 1: R:1000-1017's group mask is all ones, so it is left out):
    l = F.linear(f32(x), f32(W)); s = sigmoid(l); sel = s + bias (fp32); ids = topk(sel, k, sorted=False) - the SET;
    w = s[ids] / (sum(s[ids]) + 1e-20) x scale, fp32. Returns (ids [T][k] in torch's order, w in that order, ids
    ascending, w ascending, sel10 [T][k+1] descending, gap [T] = k-th - (k+1)-th sel, sum [T])."""
    logits = F.linear(x.type(torch.float32), gate_w.type(torch.float32))
    s = logits.sigmoid()
    sel = s.view(-1, s.shape[-1]) + bias.unsqueeze(0)
    ids = torch.topk(sel, k=k, dim=-1, sorted=False)[1]
    w = s.gather(1, ids)
    tot = w.sum(dim=-1, keepdim=True)
    w = w / (tot + 1e-20)
    w = w * scale
    o = ids.argsort(-1)
    top = sel.topk(min(k + 1, sel.shape[-1]), dim=-1).values
    gap = top[:, k - 1] - top[:, k] if top.shape[-1] > k else torch.full((sel.shape[0],), math.inf)
    return ids, w, ids.gather(1, o), w.gather(1, o), top, gap, tot[:, 0]


def relu2(x: torch.Tensor) -> torch.Tensor:
    """ACT2FN["relu2"] (transformers activations.py ReLUSquaredActivation): relu then torch.square, both in x's dtype."""
    return torch.square(F.relu(x))


def combine(d: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, order: str = "ascending") -> torch.Tensor:
    """R:925-950's sum over the k slots of d [T][k][L] (each expert's output, bf16) with w [T][k] (fp32) and ids
    [T][k]: t_j = f32(d_j) * w_j rounded to fp32 (bf16 x fp32 promotes), added into an fp32 zero row in ASCENDING
    expert id (R's outer loop + index_add_), one rounding to d's dtype. order="rank" adds in the given slot order
    (the alternative test_combine_order rejects)."""
    t = d.float() * w.unsqueeze(-1)
    if order == "ascending":
        o = ids.argsort(-1)
        t = t.gather(1, o[..., None].expand(-1, -1, t.shape[-1]))
    elif order != "rank":
        raise ValueError(order)
    acc = torch.zeros(t.shape[0], t.shape[2], dtype=torch.float32)
    for j in range(t.shape[1]):
        acc = acc + t[:, j]
    return acc.to(d.dtype)


def moe_block(x: torch.Tensor, mixer) -> torch.Tensor:
    """R:952-965 end to end on the normed rows x [1][T][H], from the module's weights: route; fc1 latent; each routed
    expert in ascending id over exactly the rows routed to it (R's grouping - a bf16 linear's bits may depend on its
    row count); combine; fc2 latent; + the shared expert on the NORMED x (R:964), one bf16 add."""
    T, H = x.shape[1], x.shape[2]
    g = mixer.gate
    ids, w, *_ = route(x.view(-1, H), g.weight, g.e_score_correction_bias, g.top_k, g.routed_scaling_factor)
    lat = F.linear(x.view(-1, H), mixer.fc1_latent_proj.weight)
    d = torch.zeros(T, ids.shape[1], lat.shape[-1], dtype=lat.dtype)
    for e in sorted(set(ids.reshape(-1).tolist())):
        tok, slot = torch.where(ids == e)
        ex = mixer.experts[e]
        d[tok, slot] = F.linear(relu2(F.linear(lat[tok], ex.up_proj.weight)), ex.down_proj.weight)
    r = combine(d, w, ids)
    o = F.linear(r, mixer.fc2_latent_proj.weight).view(1, T, H)
    sh = mixer.shared_experts
    return o + F.linear(relu2(F.linear(x, sh.up_proj.weight)), sh.down_proj.weight)


def mamba_split(mixer, projected: torch.Tensor, mutate: frozenset = frozenset()):
    """R:478-482: z [d_inner] ‖ xBC [conv_dim] ‖ dt [heads]; "swap_zx" (a mutation) reads z from xBC's place."""
    di, cd = mixer.intermediate_size, mixer.conv_dim
    z, xBC, dt = torch.split(projected, [di, cd, mixer.num_heads], dim=-1)
    if "swap_zx" in mutate:
        z, xBC = xBC[..., :di], torch.cat([z, xBC[..., di:]], -1)
    return z, xBC, dt


def mamba_seq(x: torch.Tensor, mixer, state: dict | None = None, mutate: frozenset = frozenset()):
    """R's kernel path (R:446-531, + bolt_remote.continue_forward's seeding) restated on rows x [1][T][H] through
    bolt_mamba_ops: in_proj; the conv over the raw xBC (seeded from state["conv"] [1][C][K] raw columns, zeros
    first); the SSD scan from state["ssm"] [1][H][P][N] fp32 (zeros first); the gated norm; out_proj. Returns (out,
    the new state). Mutations (test_mamba_chain): swap_zx, no_conv_bias, gate_before_norm, group_whole, clamp."""
    T = x.shape[1]
    K = mixer.conv_kernel_size
    proj = F.linear(x, mixer.in_proj.weight)
    z, xBC, dt = mamba_split(mixer, proj, mutate)
    st_conv = None if state is None else state["conv"]
    xt = xBC.transpose(1, 2)
    prev = torch.zeros(1, xt.shape[1], K, dtype=torch.float32) if st_conv is None else st_conv.float()
    bias = None if "no_conv_bias" in mutate else mixer.conv1d.bias
    conv = ops.conv_seq(xt, mixer.conv1d.weight.squeeze(1), bias, prev[..., -(K - 1):]).transpose(1, 2)
    new_conv = torch.cat([prev, xt.float()], -1)[..., -K:]
    gts = mixer.n_groups * mixer.ssm_state_size
    xs, B, C = torch.split(conv, [mixer.intermediate_size, gts, gts], dim=-1)
    A = ops.A_from_log(mixer.A_log)
    S0 = None if state is None else state["ssm"][0]
    y, S = ops.ssm_chunk(xs.view(T, -1, mixer.head_dim), dt[0], A, B.view(T, mixer.n_groups, -1),
                         C.view(T, mixer.n_groups, -1), mixer.D, mixer.dt_bias, S0, mixer.chunk_size,
                         dt_clamp=1e-3 if "clamp" in mutate else None)
    group = mixer.intermediate_size if "group_whole" in mutate else mixer.norm.group_size
    out = ops.rmsnorm_gated(y.view(1, T, -1), z, mixer.norm.weight, mixer.norm.variance_epsilon, group,
                            norm_before_gate="gate_before_norm" in mutate)
    return F.linear(out, mixer.out_proj.weight), {"conv": new_conv, "ssm": S[None]}


def mamba_step(x: torch.Tensor, mixer, state: dict, mutate: frozenset = frozenset()):
    """R's cached kernel path (R:397-439) restated on ONE row x [1][1][H]: in_proj; conv_update over the state's last
    K-1 raw columns and the new one; ssm_update; the gated norm; out_proj. Returns (out [1][1][H], the new state)."""
    K = mixer.conv_kernel_size
    proj = F.linear(x[:, 0], mixer.in_proj.weight)
    z, xBC, dt = mamba_split(mixer, proj, mutate)
    prev = state["conv"].float()
    bias = None if "no_conv_bias" in mutate else mixer.conv1d.bias
    conv = ops.conv_update(xBC, prev[..., 1:], mixer.conv1d.weight.squeeze(1), bias)
    new_conv = torch.cat([prev[..., 1:], xBC[..., None].float()], -1)
    gts = mixer.n_groups * mixer.ssm_state_size
    xs, B, C = torch.split(conv, [mixer.intermediate_size, gts, gts], dim=-1)
    A = ops.A_from_log(mixer.A_log)
    y, S = ops.ssm_update(state["ssm"], xs.view(1, -1, mixer.head_dim), dt, A, B.view(1, mixer.n_groups, -1),
                          C.view(1, mixer.n_groups, -1), mixer.D, mixer.dt_bias,
                          dt_clamp=1e-3 if "clamp" in mutate else None)
    group = mixer.intermediate_size if "group_whole" in mutate else mixer.norm.group_size
    out = ops.rmsnorm_gated(y.view(1, -1), z, mixer.norm.weight, mixer.norm.variance_epsilon, group,
                            norm_before_gate="gate_before_norm" in mutate)
    return F.linear(out, mixer.out_proj.weight)[:, None], {"conv": new_conv, "ssm": S}


# ------------------------------------------------------------------------------------------------
# R's torch fallback restated with its rounding points as switches (the bf16 difference table; spec 25 §2.0)

FALLBACK_POINTS = {
    "conv_dec_cache_dtype": "decode conv: products, sum, bias and SiLU in the CACHE's dtype (R:557-565; bf16 cache only)",
    "conv_prefill_bf16": "prefill conv: a bf16 conv1d, rounded before the SiLU, the SiLU rounded again (R:573)",
    "dt_bf16": "dt + dt_bias and softplus in bf16 (R:594, R:642)",
    "clamp": "dt clamped at time_step_min 0.001 (R:595, R:643)",
    "dBx_bf16": "decode dB = dt B and dB x in bf16 (R:607, R:612)",
    "S_bf16": "decode: the fp32 state cast to bf16 before C S, a bf16 bmm (R:626-630)",
    "Dx_bf16": "decode: x D in bf16, y + x D in bf16 (two roundings, R:636)",
    "y_prefill_fp32": "prefill: y kept fp32 into the gated norm (R:720; the kernel path rounds y once)",
}


def fallback_forward(self, input_states, cache_params, keep=frozenset(FALLBACK_POINTS), fix_state_passing: bool = False):
    """R's NemotronHMamba2Mixer.torch_forward (R:535-726) for a CACHED call (a prompt with an empty cache, or one decode
    row), restated with each of FALLBACK_POINTS as a switch: with every point in `keep` it is R's code (test: bitwise R);
    a point left out computes that step as the kernel path does (fp32, one rounding, no clamp). `fix_state_passing`
    sums the SSD state passing over the SOURCE chunk (N:328-330, transformers' native fix): R:692-697 sums over the
    TARGET chunk, which is wrong for a prompt longer than one chunk (128 rows) - the facts sheet's finding."""
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    f32 = torch.float32
    if cache_params.has_previous_state:
        projected_states = self.in_proj(input_states.squeeze(1))
    else:
        projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size
             - self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    if cache_params.has_previous_state:
        gate = gate.unsqueeze(1)
        conv_state = cache_params.conv_states[self.layer_idx]
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
        cache_params.conv_states[self.layer_idx].copy_(conv_state)
        if "conv_dec_cache_dtype" in keep:
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]
        else:
            hs = torch.sum(conv_state.float() * self.conv1d.weight[:, 0, :].float(), dim=-1)
            if self.use_conv_bias:
                hs = hs + self.conv1d.bias.float()
            hidden_states = self.act(hs).to(dtype)[:, None, ...]
    else:
        hidden_states = hidden_states.transpose(1, 2)
        conv_state = F.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
        cache_params.conv_states[self.layer_idx].copy_(conv_state)
        if "conv_prefill_bf16" in keep:
            hidden_states = self.act(self.conv1d(hidden_states).transpose(1, 2))[:, :seq_len, :]
        else:
            c = F.conv1d(hidden_states.float(), self.conv1d.weight.float(), self.conv1d.bias.float(),
                         padding=self.conv_kernel_size - 1, groups=self.conv_dim)
            hidden_states = self.act(c.transpose(1, 2))[:, :seq_len, :].to(dtype)
    hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size,
                                                      self.n_groups * self.ssm_state_size], dim=-1)
    A = -torch.exp(self.A_log.float())
    if cache_params.has_previous_state:
        dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
        if "dt_bf16" in keep:
            dt = F.softplus(dt + dt_bias.to(dt.dtype))
        else:
            dt = F.softplus(dt.float() + dt_bias.float())
        if "clamp" in keep:
            dt = torch.clamp(dt, self.time_step_min)
        A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=f32)
        dA = torch.exp(dt[..., None] * A)
        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])
        hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
        if "dBx_bf16" in keep:
            dB = dt[..., None] * B[..., None, :]
            dBx = dB * hidden_states[..., None]
        else:
            dB = dt.float()[..., None] * B.float()[..., None, :]
            dBx = dB * hidden_states.float()[..., None]
        cache_params.ssm_states[self.layer_idx].copy_(cache_params.ssm_states[self.layer_idx] * dA + dBx)
        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])
        if "S_bf16" in keep:
            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)
            Cx = C
        else:
            ssm_states = cache_params.ssm_states[self.layer_idx].float()
            Cx = C.float()
        ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
        C_reshaped = Cx.view(batch_size * self.num_heads, self.ssm_state_size, 1)
        y = torch.bmm(ssm_states_reshaped, C_reshaped)
        y = y.view(batch_size, self.num_heads, self.head_dim)
        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        if "Dx_bf16" in keep:
            y = (y + hidden_states * D).to(y.dtype)
        else:
            y = (y.float() + hidden_states.float() * D.float()).to(dtype)
        y = y.reshape(batch_size, -1)[:, None, ...]
    else:
        pad_tensor_by_size, reshape_into_chunks, segment_sum = (ns_r().pad_tensor_by_size, ns_r().reshape_into_chunks,
                                                                ns_r().segment_sum)
        if "dt_bf16" in keep:
            dt = F.softplus(dt + self.dt_bias)
        else:
            dt = F.softplus(dt.float() + self.dt_bias.float())
        if "clamp" in keep:
            dt = torch.clamp(dt, self.time_step_min)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt
        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(segment_sum(A))
        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
                  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        states_permuted = states.permute(0, 2, 1, 3, 4)
        result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=3 if fix_state_passing else 2)
        new_states = result.permute(0, 2, 1, 3, 4)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = (C[..., None, :] * states[:, :, None, ...])
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
        if "y_prefill_fp32" not in keep:
            y = y.to(dtype)
    scan_output = self.norm(y, gate)
    return self.out_proj(scan_output.to(dtype))


def ns_r():
    return br._require().R


def set_fallback_fix(model, on: bool) -> None:
    """Every Mamba mixer's torch_forward -> fallback_forward(fix_state_passing=True) (on) or back to R's (off)."""
    for m in model.modules():
        if isinstance(m, ns_r().NemotronHMamba2Mixer):
            if on:
                m.torch_forward = types.MethodType(
                    lambda self, h, c=None, mask=None: fallback_forward(self, h, c, fix_state_passing=True), m)
            elif "torch_forward" in m.__dict__:
                del m.torch_forward


# ------------------------------------------------------------------------------------------------
# recording

class Recorder:
    """Per block and row: h after the block (`h_from` = the first absolute row kept); per MoE block the route (ids
    ascending, w, sel top 10, the 9th-10th gap, the sum); taps after TAP_LAYERS (every row, with `taps`). The SSM /
    conv states after the prompt come from the cache (mamba_states). Rows are bucketed by `key` (layer_major's item)."""

    def __init__(self, model, cfg: dict, h_from: int = 0, acts: bool = True, taps: bool = False):
        self.cfg, self.h_from, self.acts = cfg, h_from, acts
        self.n = len(cfg["block_configs"])
        self.taps = [i for i in TAP_LAYERS if i < self.n] if taps else []
        self.key = 0
        self.base: dict = {}
        self.rows: dict = {}
        self.hooks = []
        for i, blk in enumerate(model.model.layers[: self.n]):
            if acts or i in self.taps:
                self.hooks.append(blk.register_forward_hook(self._block(i)))
            if blk.block_type == "moe":
                self.hooks.append(blk.mixer.gate.register_forward_hook(self._router(i)))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _put(self, name, t):
        self.rows.setdefault(self.key, {}).setdefault(name, []).append(t)

    def _block(self, i):
        def fn(_m, _a, out):
            t = (out[0] if isinstance(out, tuple) else out).detach()[0]
            if i in self.taps:
                self._put(f"tap.L{i}", t.to(torch.bfloat16).clone())
            if self.acts:
                skip = max(0, self.h_from - self.base.get(self.key, 0))
                if skip < t.shape[0]:
                    self._put(f"h.L{i}", t[skip:].clone())
        return fn

    def _router(self, i):
        def fn(mod, args, out):
            ids, w = out
            x = args[0].reshape(-1, mod.config.hidden_size)
            s = F.linear(x.type(torch.float32), mod.weight.type(torch.float32)).sigmoid()
            sel = s + mod.e_score_correction_bias.unsqueeze(0)
            k = ids.shape[1]
            top = sel.topk(min(k + 1, sel.shape[-1]), dim=-1).values
            o = ids.argsort(-1)
            self._put(f"route.ids.L{i}", ids.gather(1, o).to(torch.int32).clone())
            self._put(f"route.w.L{i}", w.gather(1, o).float().clone())
            self._put(f"route.sel.L{i}", top.float().clone())
            self._put(f"route.gap.L{i}", (top[:, k - 1] - top[:, k]).float().clone())
            self._put(f"route.sum.L{i}", s.gather(1, ids).sum(-1).float().clone())
        return fn

    def tensors(self, key=0) -> dict[str, torch.Tensor]:
        return {k: torch.cat(v, 0).contiguous() for k, v in self.rows.get(key, {}).items()}


def mamba_states(cache, cfg: dict) -> dict[str, torch.Tensor]:
    """ssm.L<l> fp32 [H][P][N] and conv.L<l> bf16 [3][C] (the last three raw columns, oldest first; R's cache keeps
    four, the oldest unread by the next step) of every Mamba block."""
    out = {}
    for i, t in enumerate(kinds(cfg)):
        if t == "mamba":
            out[f"ssm.L{i}"] = cache.ssm_states[i][0].float().clone()
            out[f"conv.L{i}"] = cache.conv_states[i][0, :, 1:].t().to(torch.bfloat16).contiguous()
    return out


def gap_line(g: torch.Tensor, what: str) -> str:
    g = g[torch.isfinite(g)].double()
    if g.numel() == 0:
        return f"{what}: no rows"
    qs = torch.quantile(g, torch.tensor([0.0, 0.0001, 0.001, 0.01, 0.5], dtype=torch.float64)).tolist()
    return (f"{what} over {g.numel()} (row, layer): ==0 {int((g == 0).sum())}, <1e-4 {int((g < 1e-4).sum())}, "
            f"<1e-3 {int((g < 1e-3).sum())}, <1e-2 {int((g < 1e-2).sum())}; min/p0.01%/p0.1%/p1%/median "
            + " / ".join(f"{x:.3g}" for x in qs))


def gap_summary(t: dict, cfg: dict) -> list[str]:
    moe = [i for i, k in enumerate(kinds(cfg)) if k == "moe" and f"route.gap.L{i}" in t]
    if not moe:
        return ["no MoE block"]
    lines = [gap_line(torch.cat([t[f"route.gap.L{i}"] for i in moe]), "MoE 9th/10th sel gap (all blocks)")]
    for i in moe:
        lines.append("  " + gap_line(t[f"route.gap.L{i}"], f"MoE 9th/10th sel gap L{i}"))
    return lines


def state_summary(t: dict, cfg: dict) -> list[str]:
    """Per Mamba block the SSM state's magnitude after the prompt (max |S|, rms) - the Decision 2 evidence."""
    out = []
    for i, k in enumerate(kinds(cfg)):
        if k == "mamba" and f"ssm.L{i}" in t:
            s = t[f"ssm.L{i}"].double()
            out.append(f"L{i} max {s.abs().max().item():.3g} rms {s.pow(2).mean().sqrt().item():.3g}")
    return ["SSM state after the prompt: " + "; ".join(out)] if out else []


# ------------------------------------------------------------------------------------------------
# a sequential run (prompt in chunks, then greedy decode)

def read_ids(path: str) -> list[int]:
    return [int(x) for x in open(path, encoding="utf-8").read().split()]


def new_cache(model, kind: str = "fp32"):
    return br.fp32_cache(model, 1, torch.float32 if kind == "fp32" else torch.bfloat16)


@torch.no_grad()
def forward_chunks(model, ids: list[int], chunk: int, cache, rec: "Recorder | None" = None, on_logits=None):
    """model(input_ids=chunk, past_key_values=cache) over `ids` in chunks (whole when chunk <= 0). on_logits(row0,
    logits fp32 [c][V]) sees every chunk's logits (R computes the head over every row, R:1731). Returns the cache."""
    pos = cache.get_seq_length()
    c = chunk if chunk > 0 else len(ids)
    for a in range(0, len(ids), c):
        part = ids[a:a + c]
        if rec is not None:
            rec.base[rec.key] = pos + a
        out = model(input_ids=torch.tensor([part]), past_key_values=cache, use_cache=True)
        if on_logits is not None:
            on_logits(pos + a, out.logits[0])
    return cache


def auto_chunk(n: int, chunk: int) -> int:
    if chunk >= 0:
        return chunk
    return 0 if n <= LONG_PROMPT else 2048


def greedy_id(row: torch.Tensor) -> int:
    """argmax with the lower id on an exact tie (torch.argmax returns the first maximal index on CPU)."""
    return int(torch.argmax(row.float()).item())


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, transformers {PINNED_TRANSFORMERS}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    cfg = config(a.snapshot, a.layers)
    ids = read_ids(a.prompt)
    T = len(ids)
    if not 1 <= T <= a.max_prompt:
        die(f"{a.prompt}: {T} ids (1..{a.max_prompt})")
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    model = build_streamed(a.snapshot, cfg, a.weights, dtype, a.path, keep=parse_keep(a.keep, cfg),
                           expert_cache_gb=a.expert_cache_gb)
    chunk = auto_chunk(T, a.chunk)
    act_from = max(0, T - a.act_tail) if a.act_tail >= 0 else 0
    h_from = 0 if T <= LONG_PROMPT else act_from
    log_from = max(0, T - max(1, a.logits_tail)) if a.logits_tail >= 0 else 0
    ks = kinds(cfg)
    print(f"config: {len(ks)} blocks ({ks.count('mamba')} mamba, {ks.count('attention')} attention, {ks.count('moe')} "
          f"moe), hidden {cfg['hidden_size']}, weights {a.weights} ({model.bolt_ckpt.trunk} trunk), {a.dtype}, mamba path "
          f"{a.path}, cache {a.cache}; prompt {T} ids, chunk {chunk or 'whole'}; h from row {h_from}, logits from row "
          f"{log_from}; built {time.time() - t0:.1f}s", flush=True)
    rec = Recorder(model, cfg, h_from=h_from, taps=a.taps)
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
    cache = forward_chunks(model, ids, chunk, new_cache(model, a.cache), rec=rec, on_logits=on_logits)
    t_prompt = time.time() - t1
    print(f"prompt forward {t_prompt:.1f}s ({model.bolt_lazy.filled} expert reads)", flush=True)
    states = mamba_states(cache, cfg)
    tokens: list[int] = []
    t2 = time.time()
    last = rows[-1][-1]
    for step in range(a.gen):
        nxt = greedy_id(last)
        tokens.append(nxt)
        rec.base[0] = T + step
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        last = out.logits[0, -1].float()
        rows.append(out.logits[0].float())
        if step == 0:
            print(f"  first decode step {time.time() - t2:.1f}s", flush=True)
    print(f"greedy {a.gen}: {time.time() - t2:.1f}s -> {tokens}", flush=True)
    rec.remove()
    t = rec.tensors()
    t.update(states)
    t["logits"] = torch.cat(rows, 0).contiguous()
    t["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    t["nll"] = nll.float().contiguous()
    n_rows = T + a.gen
    for i, k in enumerate(ks):
        if k == "moe":
            for nm in ("route.ids", "route.w", "route.sel", "route.gap", "route.sum"):
                if t[f"{nm}.L{i}"].shape[0] != n_rows:
                    die(f"{nm}.L{i} has {t[f'{nm}.L{i}'].shape[0]} rows, expected {n_rows}")
    if not torch.isfinite(t[f"h.L{len(ks) - 1}"].float()).all():
        die("the last block's output has NaN/Inf - the weights are wrong")
    for line in gap_summary(t, cfg) + state_summary(t, cfg):
        print(line)
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt": os.path.abspath(a.prompt), "n_prompt": str(T),
            "gen": str(a.gen), "layers": str(len(ks)), "chunk": str(chunk), "weights": a.weights,
            "trunk": model.bolt_ckpt.trunk, "dtype": a.dtype, "mamba_path": a.path, "cache": a.cache,
            "h_row0": str(h_from), "logits_row0": str(log_from), "attn_implementation": "eager",
            "tap_layers": json.dumps(rec.taps), "transformers": PINNED_TRANSFORMERS, "torch": torch.__version__,
            "remote_blob": REMOTE_SHA,
            "ppl": f"{math.exp(float(nll.double().mean())) if T > 1 else float('nan'):.6g}",
            "model": "Nemotron 3.5 Bolt (R's NemotronHPuzzleForCausalLM, tools/oracle/bolt_ref.py, layer-streamed)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(t, a.out, metadata=meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    print(f"wrote {a.out}: {len(t)} tensors; prompt ppl {meta['ppl']}")
    print(f"wall {time.time() - t0:.1f}s (prompt {t_prompt:.1f}s, {a.gen} steps {time.time() - t2:.1f}s), peak RSS "
          f"{rss_gib:.1f} GiB, prefetch wait {model.bolt_pf.seconds:.1f}s")


def parse_keep(spec: str, cfg: dict) -> list[int]:
    """--keep: "" (stream every block), "all", or a comma list of block ids kept resident once loaded."""
    if not spec:
        return []
    if spec == "all":
        return list(range(len(cfg["block_configs"])))
    return [int(x) for x in spec.split(",")]


# ------------------------------------------------------------------------------------------------
# ppl

def tokenize_text(snapshot: str, path: str) -> list[int]:
    """golden-prompt rule (tools/oracle/tokenize.py): the text without its trailing newlines, no template, NO BOS
    (Bolt's template writes none and the post-processor adds none - spec 25 §7)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    text = open(path, encoding="utf-8").read().rstrip("\n")
    return tok.encode(text, add_special_tokens=False).ids


def cmd_ppl(a) -> None:
    t0 = time.time()
    cfg = config(a.snapshot, a.layers)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    model = build_streamed(a.snapshot, cfg, a.weights, dtype, a.path, keep=parse_keep(a.keep, cfg))
    tok_dir = a.tokenizer or a.snapshot
    srcs = [(n, tokenize_text(tok_dir, p)) for n, p in (x.split("=", 1) for x in a.text)]
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
        forward_chunks(model, ids, auto_chunk(len(ids), a.chunk), new_cache(model), on_logits=on_logits)
        ppl = math.exp(tot / max(n, 1))
        print(f"ppl {name}: {ppl:.4f} over {n} predicted tokens ({len(ids)} ids, {len(cfg['block_configs'])} blocks, "
              f"{time.time() - t1:.0f}s)" + ("" if math.isfinite(ppl) else "  NOT SANE (not finite)"), flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"wall {time.time() - t0:.0f}s, peak RSS {rss / 2**30 if sys.platform == 'darwin' else rss / 2**20:.1f} GiB")


# ------------------------------------------------------------------------------------------------
# the port's restated chain (hfcheck's bitwise arm) and layer-major batches

@torch.no_grad()
def port_forward(model, ids: list[int], st: dict, rec: dict | None = None):
    """One forward of `ids` through the RESTATED ops - rms_norm, mamba_seq / mamba_step (bolt_mamba_ops), R's
    attention module (eager, its own KV cache st["cache"]), moe_block, norm_f, lm_head - on build_streamed's weights
    (each block materialised from bolt_layer_sd, experts from bolt_lazy, dropped after). st carries "pos", the R cache
    (attention KV) and "mamba" {layer: state}. Returns the logits (fp32 [T][V]); `rec` gets h.L / route rows."""
    tm = model.model
    cfg = model.bolt_cfg
    eps = model.config.layer_norm_epsilon
    if st.get("cache") is None:
        st["cache"] = new_cache(model)
    cache = st["cache"]
    pos = st.setdefault("pos", 0)
    mam = st.setdefault("mamba", {})
    T = len(ids)
    x = tm.embeddings(torch.tensor([list(ids)]))
    cp = torch.arange(pos, pos + T)
    mask = tm._update_causal_mask(None, x, cp)
    h = x

    def put(name, t):
        if rec is not None:
            rec.setdefault(name, []).append(t.detach().clone())
    for i, kind in enumerate(kinds(cfg)):
        blk = tm.layers[i]
        sd = model.bolt_layer_sd(i)
        mixer_sd = {k[len("mixer."):]: v for k, v in sd.items() if k.startswith("mixer.")}
        blk.mixer.load_state_dict(mixer_sd, strict=True, assign=True)
        xn = rms_norm(h.to(sd["norm.weight"].dtype), sd["norm.weight"], eps)
        if kind == "mamba":
            if i in mam and T == 1:
                y, mam[i] = mamba_step(xn, blk.mixer, mam[i])
            else:
                y, mam[i] = mamba_seq(xn, blk.mixer, mam.get(i))
        elif kind == "attention":
            y = blk.mixer(hidden_states=xn, past_key_values=cache, attention_mask=mask)[0]
        else:
            g = blk.mixer.gate
            out = route(xn.view(-1, xn.shape[-1]), g.weight, g.e_score_correction_bias, g.top_k, g.routed_scaling_factor)
            model.bolt_lazy.fill(i, sorted(set(out[0].reshape(-1).tolist())))
            y = moe_block(xn, blk.mixer)
            put(f"route.ids.L{i}", out[2].to(torch.int32))
            put(f"route.w.L{i}", out[3].float())
            put(f"route.sel.L{i}", out[4].float())
            put(f"route.gap.L{i}", out[5].float())
            put(f"route.sum.L{i}", out[6].float())
        blk.mixer.to_empty(device="meta")
        h = h + y
        put(f"h.L{i}", h[0])
    st["pos"] = pos + T
    hn = rms_norm(h, tm.norm_f.weight, eps)
    return model.lm_head(hn.to(model.lm_head.weight.dtype)).float()[0]


def layer_major(model, pf=None, rec: "Recorder | None" = None):
    """-> step_many(items, keys=None) for build_streamed's model: items = [(ids, pos, state)], state a dict whose "cache"
    is None before the first chunk. Returns per item the logits fp32 [T][V]. BITWISE what model(input_ids=[ids],
    past_key_values=cache, use_cache=True) computes: NemotronHModel.forward's prologue (R:1449-1464) with the module's
    own functions, each block called with the arguments its loop passes (R:1489-1495), norm_f, has_previous_state set
    (R:1502-1503), the head (R:1731). Layer-major: every item through block 0, then block 1, ... with stream.py's hold
    mode (each block's weights read once per pass)."""
    tm = model.model
    cfg = model.bolt_cfg
    pf = pf or model.bolt_pf
    layers = list(tm.layers[: len(cfg["block_configs"])])

    def prologue(ids, pos, st):
        h = tm.embeddings(torch.tensor([list(ids)]))
        if st.get("cache") is None:
            st["cache"] = new_cache(model)
        cache = st["cache"]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a cache of {past} tokens")
        cp = torch.arange(past, past + h.shape[1])
        return {"h": h, "cache": cache, "cp": cp, "past": past,
                "masks": {"attention": tm._update_causal_mask(None, h, cp), "mamba": tm._update_mamba_mask(None, cp),
                          "moe": None}}

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
            for blk in layers:
                for j, p in enumerate(ps):
                    select(keys[j], p["past"])
                    p["h"] = blk(p["h"], past_key_values=p["cache"], cache_position=p["cp"],
                                 attention_mask=p["masks"][blk.block_type], output_attentions=False)
                pf.release()
        finally:
            pf.hold = False
            pf.release()
        out = []
        for p in ps:
            h = tm.norm_f(p["h"])
            if not p["cache"].has_previous_state:
                p["cache"].has_previous_state = True
            out.append(model.lm_head(h.to(model.lm_head.weight.dtype)).float()[0])
            p.clear()
        return out
    return step_many


# ------------------------------------------------------------------------------------------------
# hfcheck: the port against its restated chain (bitwise) and against R's torch fallback (recorded), any weights

def run_arm(model, cfg: dict, ids: list[int], gen: int, chunk: int, restated: bool = False, tail: int = 64):
    """Prompt (in chunks) + `gen` greedy cached steps; returns (tensors, tokens): h.L / route rows and the logits of
    the last `tail` prompt rows and every generated row."""
    rows, toks = [], []
    T = len(ids)

    def keep(r0, lg):
        k = max(0, T - tail - r0)
        if k < lg.shape[0]:
            rows.append(lg[k:].float())
    if restated:
        st, rec = {}, {}
        c = chunk if chunk > 0 else len(ids)
        for a0 in range(0, len(ids), c):
            keep(a0, port_forward(model, ids[a0:a0 + c], st, rec))
        for _ in range(gen):
            nxt = greedy_id(rows[-1][-1])
            toks.append(nxt)
            rows.append(port_forward(model, [nxt], st, rec))
        t = {k: torch.cat(v, 0) for k, v in rec.items()}
    else:
        rec = Recorder(model, cfg)
        cache = forward_chunks(model, ids, chunk, new_cache(model), rec=rec, on_logits=keep)
        for step in range(gen):
            nxt = greedy_id(rows[-1][-1])
            toks.append(nxt)
            rec.base[0] = len(ids) + step
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            rows.append(out.logits[0].float())
        rec.remove()
        t = rec.tensors()
    t["logits"] = torch.cat(rows)
    return t, toks


def compare(a: dict, b: dict) -> tuple[list[str], dict]:
    """Keys whose tensors differ, and per key (max |diff|, that over the key's own max |value|)."""
    keys = sorted(set(a) | set(b))
    bad, d = [], {}
    for k in keys:
        if k not in a or k not in b or a[k].shape != b[k].shape:
            bad.append(k)
            d[k] = (math.inf, math.inf)
            continue
        if not torch.equal(a[k], b[k]):
            bad.append(k)
            x, y = a[k].double(), b[k].double()
            m = (x - y).abs().max().item()
            d[k] = (m, m / max(y.abs().max().item(), 1e-30))
    return bad, d


def cmd_hfcheck(a) -> None:
    t0 = time.time()
    cfg = config(a.snapshot, a.layers)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    if a.prompt:
        ids = read_ids(a.prompt)[: a.n]
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, cfg["vocab_size"], (a.n,), generator=g).tolist()
    chunk = auto_chunk(len(ids), a.chunk)
    model = build_streamed(a.snapshot, cfg, a.weights, dtype, "kernel", keep=parse_keep(a.keep, cfg))
    port, ptok = run_arm(model, cfg, ids, a.gen, chunk)
    rest, rtok = run_arm(model, cfg, ids, a.gen, chunk, restated=True)
    bad, _ = compare(port, rest)
    print(f"hfcheck: {len(cfg['block_configs'])} blocks, {len(ids)} ids + {a.gen} cached steps, weights {a.weights}, "
          f"{a.dtype}; the port (R's kernel path) vs the restated chain: {len(port)} tensors, differing "
          f"{bad[:8] if bad else 'none'}; greedy {ptok} vs {rtok}", flush=True)
    ok = not bad and ptok == rtok
    # R's fallback has no prompt continuation (its cached branch reads one row, R:555-565): the prompt runs whole
    whole, wtok = (port, ptok) if chunk == 0 else run_arm(model, cfg, ids, a.gen, 0)
    br.set_path(model, "torch")
    arms = [("R's torch fallback", False)]
    if len(ids) > model.config.chunk_size:
        arms.append(("R's torch fallback with the SSD state-passing index fixed (N:328-330)", True))
    for what, fix in arms:
        set_fallback_fix(model, fix)
        fb, ftok = run_arm(model, cfg, ids, a.gen, 0)
        fbad, fd = compare(fb, whole)
        worst = sorted(((v[1], k) for k, v in fd.items() if math.isfinite(v[1])), reverse=True)[:6]
        print(f"hfcheck: {what} vs the port ({a.dtype}, the prompt whole): {len(fbad)} of {len(fb)} tensors differ "
              f"(recorded, Decision 1: other orders{', bf16 roundings' if a.dtype == 'bf16' else ''}, the dt clamp"
              f"{'' if fix else ', and past 128 rows its state passing'}); largest relative: "
              + ", ".join(f"{k} {r:.2e}" for r, k in worst) + f"; greedy {ftok} vs {wtok}", flush=True)
    set_fallback_fix(model, False)
    br.set_path(model, "kernel")
    print(f"hfcheck: {'BITWISE' if ok else 'FAIL'} (the port = its restated chain: logits, every h row, the routes); "
          f"{time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------------------------------------
# taps: dump_taps' format over [prompt + continuation] (spec 26a's input)

TAP_WINDOW = 1024            # LiLiCorr's context window (spec 26): rows older than n_prompt - 1024 are never read


def tap_from_of(n_prompt: int, window: int = TAP_WINDOW) -> int:
    return max(0, n_prompt - window)


def parse_tap_sources(specs) -> list[dict]:
    out = []
    for s in specs or []:
        parts = s.split(":")
        if len(parts) != 3:
            die(f"--source {s}: want NAME:IDS:N_PROMPT")
        name, path, n = parts
        if "/" not in name:
            die(f"--source {s}: the name must be corpus/label")
        ids = read_ids(path)
        if not 1 <= int(n) < len(ids):
            die(f"--source {s}: N_PROMPT {n} of {len(ids)} ids")
        out.append({"name": name, "ids": ids, "n_prompt": int(n)})
    return out


def cmd_taps(a) -> None:
    mt = _load("muse_taps")
    dt = _load("dump_taps")
    sources = parse_tap_sources(a.source)
    if not sources:
        die("no --source")
    os.makedirs(a.out, exist_ok=True)
    todo = [s for s in sources if not os.path.exists(os.path.join(a.out, dt.file_name(s["name"])))]
    toks = sum(len(s["ids"]) for s in todo)
    print(f"{len(sources)} sources, {len(sources) - len(todo)} already dumped, {toks} ids to teacher-force; taps after "
          f"blocks {list(TAP_LAYERS)}, tap_from = n_prompt - {a.window}", flush=True)
    for s in todo:
        print(f"  {s['name']}: {len(s['ids'])} ids ({s['n_prompt']} prompt), tap_from "
              f"{tap_from_of(s['n_prompt'], a.window)}", flush=True)
    if a.dry_run or not todo:
        return
    t0 = time.time()
    cfg = config(a.snapshot, a.layers)
    model = build_streamed(a.snapshot, cfg, a.weights, keep=parse_keep(a.keep, cfg))
    for s in todo:
        ts = time.time()
        f = tap_from_of(s["n_prompt"], a.window)
        rec = Recorder(model, cfg, acts=False, taps=True)
        parts = []

        def on_logits(r0, lg):
            for b0 in range(0, lg.shape[0], 256):
                ra = r0 + b0
                rb = lg[b0:b0 + 256]
                k = max(0, f - ra)
                if k < rb.shape[0]:
                    parts.append(mt.row_stats(rb[k:], rb[k:], a.topk))
        try:
            forward_chunks(model, s["ids"], auto_chunk(len(s["ids"]), a.chunk), new_cache(model), rec, on_logits)
            t = rec.tensors()
        finally:
            rec.remove()
        res = {k: torch.cat([p[k] for p in parts]) for k in ("greedy", "top_ids", "top_logits", "lse")}
        res["taps"] = {L: t[f"tap.L{L}"][f:].contiguous() for L in rec.taps}
        meta = {"snapshot": os.path.abspath(a.snapshot), "weights": a.weights, "transformers": PINNED_TRANSFORMERS,
                "source": "teacher-forced (bolt_ref.py taps)", "vocab_used": str(cfg["vocab_size"]),
                "chunk": str(auto_chunk(len(s["ids"]), a.chunk)), "window": str(a.window)}
        dt.save_dump(os.path.join(a.out, dt.file_name(s["name"])), s["name"], s["ids"], s["n_prompt"], f, res, meta)
        print(f"  {s['name']}: {time.time() - ts:.0f} s", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"wall {time.time() - t0:.0f} s, peak RSS {rss / 2**30 if sys.platform == 'darwin' else rss / 2**20:.1f} GiB")


# ------------------------------------------------------------------------------------------------

def cmd_facts(a) -> None:
    _load("bolt_facts").main(a.rest)


def cmd_gateq(a) -> None:
    _load("bolt_gateq").main(a.rest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, weights="int4"):
        p.add_argument("snapshot")
        p.add_argument("--layers", type=int)
        p.add_argument("--weights", default=weights, choices=["int4", "bf16"],
                       help="the checkpoint's form: int4 (SergiioB's g128 / our g64, dequantised) or bf16 (the original)")
        p.add_argument("--chunk", type=int, default=-1, help="prefill chunk (-1: whole up to 4096 ids, else 2048)")
        p.add_argument("--keep", default="", help="blocks kept resident once loaded: '', 'all' or a comma list")
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=262144)
    r.add_argument("--logits-tail", type=int, default=1024, help="keep logits from row T - L (-1: all)")
    r.add_argument("--act-tail", type=int, default=256, help="keep h past 4096 ids from row T - A")
    r.add_argument("--taps", action="store_true", help="tap.L{1,18,35,51,68,85} for every row")
    r.add_argument("--cache", default="fp32", choices=["fp32", "bf16"], help="the SSM / conv cache dtype (Decision 2)")
    r.add_argument("--path", default="kernel", choices=["kernel", "torch"], help="the Mamba path (torch: R's fallback)")
    r.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    r.add_argument("--expert-cache-gb", type=float, default=0.0, help="an LRU of dequantised experts across steps")
    p = sub.add_parser("ppl")
    common(p)
    p.add_argument("--text", action="append", default=[], help="NAME=FILE (the snapshot's tokenizer.json, no BOS)")
    p.add_argument("--ids", action="append", default=[], help="NAME=FILE")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--tokenizer", help="a directory with tokenizer.json (default: the snapshot)")
    p.add_argument("--path", default="kernel", choices=["kernel", "torch"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    h = sub.add_parser("hfcheck")
    common(h)
    h.add_argument("--prompt")
    h.add_argument("--n", type=int, default=2048)
    h.add_argument("--gen", type=int, default=4)
    h.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    t = sub.add_parser("taps")
    common(t)
    t.add_argument("--source", action="append", default=[], help="NAME:IDS:N_PROMPT (NAME corpus/label)")
    t.add_argument("--out", required=True)
    t.add_argument("--topk", type=int, default=64)
    t.add_argument("--window", type=int, default=TAP_WINDOW)
    t.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("facts", add_help=False)
    f.add_argument("rest", nargs=argparse.REMAINDER)
    q = sub.add_parser("gateq", add_help=False)
    q.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    {"run": cmd_run, "ppl": cmd_ppl, "hfcheck": cmd_hfcheck, "taps": cmd_taps, "facts": cmd_facts,
     "gateq": cmd_gateq}[a.cmd](a)


if __name__ == "__main__":
    main()
