#!/usr/bin/env python3
"""Spec 23a: the CPU reference of Muse-Glimmer-30B (text path), MG1.

The model is transformers 5.19.0's own `MuseGlimmerForConditionalGeneration` (refused on any other version)
built on `meta` from the snapshot's config with the vision tower, adapter and projection REMOVED (never
materialised; the text path never calls them), and LAYER-STREAMED (stream.py: `embed_tokens`, the final
`norm` and `lm_head` resident, each decoder layer materialised by a forward pre-hook and dropped after).
Muse needs no new mechanism: dense SwiGLU, no experts, no PLE. Two checkpoint forms are read by name:
  "bf16"  the original's `<linear>.weight` (and the tiny's)
  "gptq"  the int4 checkpoint: `<linear>.{qweight, scales, qzeros[, g_idx]}` - either Vishva007 repo, the same
          export in two spellings (-AutoRound: auto_round:auto_gptq, no g_idx; -AutoRound-GPTQ: gptq with g_idx) -
          dequantised on demand to bf16(float(q - 8) x float(scale)) (stream.dequant_t, g64); every linear's qzeros
          must be 0x77777777 and its g_idx, when shipped, i // 64 (checked the first time it is read)
The forward is the module's own, eager attention (`attn_implementation="eager"`); the reference dtype is bf16
(fp32 for the tests). A Recorder (forward hooks) keeps per layer the residual after the layer (`H.L<l>`), the
attention and MLP outputs, the drafter taps (the OUTPUT of layers TAP_LAYERS, OC:112-131), and per row the raw
`lm_head` logits and the capped logits (M:1172-1181) with the capped-tie flag.

    muse_ref.py run <snapshot> --prompt <ids> --out <p>.golden.safetensors [--gen 32] [--layers N] [--taps]
                    [--act-tail 256] [--logits-tail 1024] [--chunk C] [--sublayers]
    muse_ref.py ppl <snapshot> [--text NAME=FILE ...] [--ids NAME=FILE ...] [--layers N] [--max-tokens T]
    muse_ref.py hfcheck <snapshot> --layers N [--prompt <ids>] [--n 2100] [--gen 4]
    muse_ref.py facts <snapshot> [--gptq <snapshot>] [--assistant <snapshot>] [--scan-quant]

`run`'s golden layout (activation rows from `act_row0`, logits rows from `logits_row0`, both metadata):
    H.L<l>        bf16 [A][H]        the residual after layer l (prompt tail + every generated row)
    attn.L<l> / mlp.L<l>  bf16 [A][H]  with --sublayers: the attention (after o_proj) / MLP output, before the post norm
    logits.raw    f32  [Lr][V]       lm_head's bf16 row widened (no multiplier, no cap)
    logits.cap    f32  [Lr][V]       the model's logits: the four-rounding softcap of the raw row, widened
    cap_tie       u8   [T+gen]       1 where the capped row's maximum is shared by >= 2 ids while the raw maximum is unique
    gap.raw       f32  [T+gen]       raw top-1 minus top-2
    taps          bf16 [T+gen][n][H] with --taps: the residual after each layer of `tap_layers` (metadata)
    tokens        i32  [gen]         greedy by argmax over the CAPPED row (generate(do_sample=False)'s rule)
    tokens.raw    i32  [gen]         the raw row's argmax at the same steps
    nll           f32  [T-1]         the prompt's teacher-forced NLL under the capped distribution (T 1.0)
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
import time  # noqa: E402

import torch  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
TAP_LAYERS = (1, 13, 25, 37, 49)            # the assistant's target_layer_ids (AC:89)
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
LM = "model.language_model."
SYM_ZERO = 0x77777777
GROUP = 64


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"muse_{name}", os.path.join(_HERE, f"{name}.py"))
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
        raise RuntimeError(f"muse_ref restates transformers {PINNED_TRANSFORMERS}'s muse_glimmer; this is "
                           f"{transformers.__version__} ({os.path.dirname(transformers.__file__)}) - put the 5.19.0 "
                           f"site first on PYTHONPATH (tools/oracle/qwen4exp_env.sh)")


def mm():
    require_version()
    from transformers.models.muse_glimmer import modeling_muse_glimmer as m
    return m


# ------------------------------------------------------------------------------------------------
# the config

def read_config(snapshot: str) -> dict:
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def text_config(snapshot: str, layers: int | None = None, dtype: torch.dtype = torch.bfloat16, raw: dict | None = None):
    """config.json's text_config -> the MuseGlimmerTextConfig the reference builds. layers=N truncates
    num_hidden_layers, layer_types[:N] and layer_rope_theta[:N] (--layers N: layers [0, N), then the final norm
    and the head). Refuses a non-zero layer_rope_theta that differs from rope_parameters.rope_theta (5.19.0 builds
    ONE rotary table from the global theta, M:186-191; a layer's value is only a NoPE switch, M:512)."""
    require_version()
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    raw = raw if raw is not None else read_config(snapshot)
    d = dict(raw["text_config"] if "text_config" in raw else raw)
    d.pop("quantization_config", None)
    n = d["num_hidden_layers"]
    if layers is not None:
        if not 1 <= layers <= n:
            die(f"--layers {layers}: the checkpoint has {n}")
        d["num_hidden_layers"] = layers
        for k in ("layer_types", "layer_rope_theta"):
            if d.get(k) is not None:
                d[k] = list(d[k][:layers])
    tc = MuseGlimmerTextConfig(**d)
    if tc.model_type != "muse_glimmer_text":
        die(f"text config model_type {tc.model_type!r}, expected muse_glimmer_text")
    theta = tc.rope_parameters["rope_theta"]
    for i, x in enumerate(tc.layer_rope_theta):
        if x and x != theta:
            die(f"layer {i}: layer_rope_theta {x} differs from rope_parameters.rope_theta {theta} - transformers "
                f"5.19.0 builds one table from the global theta (M:186-191), so the per-layer value would be ignored")
    tc._attn_implementation = "eager"
    tc.dtype = dtype
    return tc


def full_config(snapshot: str, tc, raw: dict | None = None):
    """The MuseGlimmerConfig around `tc` (vision config as shipped; it is built on meta and then removed)."""
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    raw = dict(raw if raw is not None else read_config(snapshot))
    raw.pop("quantization_config", None)
    for k in ("text_config", "architectures", "model_type", "transformers_version", "dtype", "torch_dtype"):
        raw.pop(k, None)
    cfg = MuseGlimmerConfig(text_config=tc.to_dict(), **raw)
    cfg.text_config = tc
    cfg._attn_implementation = "eager"
    tc._attn_implementation = "eager"
    return cfg


# ------------------------------------------------------------------------------------------------
# names

def expected_names(tc, form: str) -> list[str]:
    """Every text-model tensor of a checkpoint in `form` - "gptq" (`qweight` / `scales` / `qzeros` / `g_idx` per
    linear: Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ), "autoround" (the same without `g_idx`: the same export's
    auto_round:auto_gptq spelling, Vishva007/Muse-Glimmer-30B-W4A16-AutoRound) or "bf16" (`.weight`) -; norms,
    embed_tokens, the final norm and lm_head (omitted when tied) in all."""
    if form not in ("gptq", "autoround", "bf16"):
        raise ValueError(form)
    sfx = {"gptq": GPTQ_SUFFIXES, "autoround": GPTQ_SUFFIXES[:3], "bf16": (".weight",)}[form]
    names = [LM + "embed_tokens.weight", LM + "norm.weight"]
    if not tc.tie_word_embeddings:
        names.append("lm_head.weight")
    for i in range(tc.num_hidden_layers):
        b = f"{LM}layers.{i}."
        names += [b + n + ".weight" for n in NORMS]
        for lin in LINEARS:
            names += [b + lin + s for s in sfx]
    return sorted(names)


def is_text(name: str) -> bool:
    return name.startswith(LM) or name == "lm_head.weight"


def vision_names(index: dict) -> list[str]:
    """Every name of an index (`weight_map` or the index json) the text path skips (809 on the real one)."""
    wm = index.get("weight_map", index)
    return sorted(k for k in wm if not is_text(k))


# ------------------------------------------------------------------------------------------------
# the checkpoint

class Ckpt:
    """Name -> tensor over a snapshot's shards (index or a single model.safetensors)."""

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
        missing = [fn for fn in sorted(set(self.wm.values())) if not os.path.exists(os.path.join(snapshot, fn))]
        if missing:
            raise FileNotFoundError(f"{snapshot}: shard(s) {missing[:3]} absent (only the small files are here?)")
        self.h = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
                  for fn in sorted(set(self.wm.values()))}

    def __contains__(self, k: str) -> bool:
        return k in self.wm

    def keys(self):
        return self.wm.keys()

    def get(self, k: str) -> torch.Tensor:
        return self.h[self.wm[k]].get_tensor(k)


def form_of(ck: Ckpt) -> str:
    return "gptq" if any(k.endswith(".qweight") for k in ck.keys()) else "bf16"


def check_gptq(ck: Ckpt, base: str) -> None:
    """docs/13's data asserts for one linear: qzeros all 0x77777777 (sym), g_idx == i // 64 (no act-order)."""
    qz = ck.get(base + ".qzeros")
    if not bool(((qz.to(torch.int64) & 0xFFFFFFFF) == SYM_ZERO).all()):
        die(f"{base}.qzeros is not 0x77777777: not symmetric int4 (the engine's rule is (q - 8) x scale)")
    if base + ".g_idx" in ck:
        g = ck.get(base + ".g_idx")
        if not torch.equal(g.to(torch.int64), torch.arange(g.numel(), dtype=torch.int64) // GROUP):
            die(f"{base}.g_idx is not i // {GROUP}: an act-order permutation the kernels do not implement")


def dequant(ck: Ckpt, base: str, checked: set) -> torch.Tensor:
    """`base`.{qweight, scales} -> the nn.Linear weight [N][K] bf16 = bf16(float(q - 8) x float(scale)), g64."""
    qw, sc = ck.get(base + ".qweight"), ck.get(base + ".scales")
    g = qw.shape[0] * 8 // sc.shape[0]
    if g != GROUP:
        die(f"{base}: group size {g}, the reference (and the checkpoint's quantization_config) is {GROUP}")
    first = base not in checked
    if first:
        check_gptq(ck, base)
    out = _stream.dequant_t(qw, sc, g, check=first)
    checked.add(base)
    return out


def _cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype) if t.is_floating_point() and t.dtype != dtype else t


def layer_state(ck: Ckpt, i: int, dtype: torch.dtype, checked: set) -> dict[str, torch.Tensor]:
    """Layer i's tensors, names relative to the decoder layer (int4 dequantised, floats cast to `dtype`)."""
    b = f"{LM}layers.{i}."
    sd = {}
    for n in NORMS:
        sd[n + ".weight"] = _cast(ck.get(b + n + ".weight"), dtype)
    for lin in LINEARS:
        if b + lin + ".qweight" in ck:
            sd[lin + ".weight"] = _cast(dequant(ck, b + lin, checked), dtype)
        else:
            sd[lin + ".weight"] = _cast(ck.get(b + lin + ".weight"), dtype)
    return sd


def resident_state(ck: Ckpt, tc, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    emb = _cast(ck.get(LM + "embed_tokens.weight"), dtype)
    sd = {LM + "embed_tokens.weight": emb, LM + "norm.weight": _cast(ck.get(LM + "norm.weight"), dtype)}
    if "lm_head.weight" in ck:
        sd["lm_head.weight"] = _cast(ck.get("lm_head.weight"), dtype)
    elif tc.tie_word_embeddings:
        sd["lm_head.weight"] = emb            # tied (the downloaded tiny): the same tensor, as from_pretrained ties it
    else:
        die("lm_head.weight is absent and tie_word_embeddings is false")
    return sd


# ------------------------------------------------------------------------------------------------
# the model: streamed, or resident (hfcheck's other arm)

def _meta_model(snapshot: str, tc, untie: bool):
    m = mm()
    cfg = full_config(snapshot, tc)
    if untie:
        cfg.tie_word_embeddings = False
    with torch.device("meta"):
        model = m.MuseGlimmerForConditionalGeneration(cfg)
    inner = model.model
    inner.vision_tower = None          # never materialised: the text path calls none of the three
    inner.vision_adapter = None
    inner.vision_projection = None
    return model


def build_streamed(snapshot: str, tc, form: str = "auto", keep=()):
    """MuseGlimmerForConditionalGeneration on meta, vision removed; embed_tokens / norm / lm_head resident; the
    decoder layers streamed (stream.attach). form "auto" reads the checkpoint's names. Returns (model, the
    prefetcher - `.seconds` is the dequant wait)."""
    ck = Ckpt(snapshot)
    got = form_of(ck)
    if form not in ("auto", got):
        die(f"{snapshot}: the checkpoint is the {got} form, not {form}")
    tied = tc.tie_word_embeddings and "lm_head.weight" not in ck
    model = _meta_model(snapshot, tc, untie=not tied)
    lm = model.model.language_model
    checked: set = set()
    resident = resident_state(ck, tc, tc.dtype)
    layers = list(lm.layers)
    pf = _stream.attach(model, layers, lambda i: layer_state(ck, i, tc.dtype, checked), resident,
                        rebuild=[(lm, "rotary_emb", lambda: type(lm.rotary_emb)(lm.config))], keep=keep)
    if tied and model.lm_head.weight.data_ptr() != lm.embed_tokens.weight.data_ptr():
        raise RuntimeError("tied checkpoint: lm_head is not embed_tokens after loading")
    model.muse_ckpt, model.muse_form = ck, got
    return model.eval(), pf


def build_resident(snapshot: str, tc):
    """The same class with every layer loaded (no hooks): the un-streamed arm on real weights (hfcheck)."""
    ck = Ckpt(snapshot)
    tied = tc.tie_word_embeddings and "lm_head.weight" not in ck
    model = _meta_model(snapshot, tc, untie=not tied)
    lm = model.model.language_model
    sd = resident_state(ck, tc, tc.dtype)
    checked: set = set()
    for i in range(tc.num_hidden_layers):
        sd.update({f"{LM}layers.{i}.{k}": v for k, v in layer_state(ck, i, tc.dtype, checked).items()})
    model.load_state_dict(sd, strict=True, assign=True)
    lm.rotary_emb = type(lm.rotary_emb)(lm.config)
    return model.eval()


def hf_reference(snapshot: str, dtype: torch.dtype):
    """transformers' own from_pretrained model, un-streamed (the tiny / test sizes only)."""
    m = mm()
    return m.MuseGlimmerForConditionalGeneration.from_pretrained(snapshot, dtype=dtype,
                                                                 attn_implementation="eager").eval()


# ------------------------------------------------------------------------------------------------
# the head

def softcap(z: torch.Tensor, mult: float, cap: float) -> torch.Tensor:
    """M:1178-1181 op for op: z * mult, / cap, tanh, * cap - each in z's dtype (four bf16 roundings in bf16)."""
    z = z * mult
    z = z / cap
    z = torch.tanh(z)
    return z * cap


def cap_tie(raw: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """Per row: the capped row's maximum is shared by >= 2 ids while the raw maximum is unique - the rows where
    the softcap's bf16 roundings make torch.argmax (the lowest id among equals) differ in kind from the raw rule.
    raw / cap [R][V] in the model's dtype. -> bool [R]."""
    rmax = raw.max(-1, keepdim=True).values
    cmax = cap.max(-1, keepdim=True).values
    return ((cap == cmax).sum(-1) >= 2) & ((raw == rmax).sum(-1) == 1)


def raw_gap(raw: torch.Tensor) -> torch.Tensor:
    v = raw.float().topk(2, dim=-1).values
    return v[:, 0] - v[:, 1]


# ------------------------------------------------------------------------------------------------
# recording

class Recorder:
    """Forward hooks on the decoder layers (H.L<l>), their self_attn / mlp (with `sublayers`), the tap layers
    and lm_head (the raw rows of the latest forward, `.raw`). `act_from` = the first absolute row whose
    activations are kept; rows are bucketed by `key` (layer_major's item) - 0 for a sequential run."""

    def __init__(self, model, tc, act_from: int = 0, acts: bool = True, sublayers: bool = False, taps=()):
        lm = model.model.language_model
        self.tc, self.act_from = tc, act_from
        self.taps = tuple(t for t in taps if t < tc.num_hidden_layers)
        self.key = 0
        self.base: dict = {}
        self.rows: dict = {}
        self.raw = None
        self.hooks = [model.lm_head.register_forward_hook(self._head)]
        for i, layer in enumerate(lm.layers):
            if acts:
                self.hooks.append(layer.register_forward_hook(self._act(f"H.L{i}")))
                if sublayers:
                    self.hooks += [layer.self_attn.register_forward_hook(self._act(f"attn.L{i}")),
                                   layer.mlp.register_forward_hook(self._act(f"mlp.L{i}"))]
            if i in self.taps:
                self.hooks.append(layer.register_forward_hook(self._tap(self.taps.index(i))))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _bucket(self):
        return self.rows.setdefault(self.key, {})

    def _put(self, name, t):
        self._bucket().setdefault(name, []).append(t)

    def _head(self, _m, _a, out):
        self.raw = out.detach()

    def _act(self, name):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach()[0]
            skip = max(0, self.act_from - self.base.get(self.key, 0))
            if skip < t.shape[0]:
                self._put(name, t[skip:].to(torch.bfloat16).clone())
        return fn

    def _tap(self, j):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            self._put(f"tap{j}", t.detach()[0].to(torch.bfloat16).clone())
        return fn

    def tensors(self, key=0) -> dict[str, torch.Tensor]:
        b = self.rows.get(key, {})
        out = {k: torch.cat(v, 0).contiguous() for k, v in b.items() if not k.startswith("tap")}
        if self.taps:
            out["taps"] = torch.stack([torch.cat(b[f"tap{j}"], 0) for j in range(len(self.taps))], 1).contiguous()
        return out


# ------------------------------------------------------------------------------------------------
# layer-major batches (oracle_generate --batch's shape)

def layer_major(model, pf, rec: "Recorder | None" = None):
    """-> step_many(items, keys=None) for build_streamed's model: items = [(ids, pos, state)], state a dict whose
    "cache" is None before the first chunk. Returns per item (raw, capped) logits [T][V] in the model's dtype.
    BITWISE what `model(input_ids=[ids], past_key_values=cache, use_cache=True)` computes: MuseGlimmerModel /
    MuseGlimmerTextModel.forward's prologue (M:1009-1013, M:480-505) restated with the modules' own functions, each
    decoder layer called with the arguments its loop passes (M:507-516), then the final norm, lm_head and the
    softcap (M:518, M:1172-1181). Layer-major: every item through layer 0, then layer 1, ... with stream.py's hold
    mode, so one pass reads each layer's weights once for all items."""
    m = mm()
    mdl = model.model
    tm = mdl.language_model
    cfg = tm.config
    tcfg = model.config.text_config
    layers = list(tm.layers[: cfg.num_hidden_layers])

    def prologue(ids, pos, st):
        x = torch.tensor([list(ids)])
        mmask = (x == model.config.image_token_id) | (x == model.config.video_token_id)
        x = x.clone()
        x[mmask] = 0
        h = mdl.get_input_embeddings()(x)
        if st.get("cache") is None:
            st["cache"] = m.DynamicCache(config=cfg)
        cache = st["cache"]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a cache of {past} tokens")
        position_ids = (torch.arange(h.shape[1], device=h.device) + past).unsqueeze(0)
        mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": cache,
              "position_ids": position_ids}
        masks = {"full_attention": m.create_causal_mask(**mk),
                 "sliding_attention": m.create_sliding_window_causal_mask(**mk)}
        pe = tm.rotary_emb(h, position_ids)
        return {"h": h, "pe": pe, "masks": masks, "pos_ids": position_ids, "cache": cache, "past": past}

    def select(key, past):
        if rec is not None:
            rec.key = key
            rec.base[key] = past

    @torch.no_grad()
    def step_many(items, keys=None, last_only: bool = False, tail: int = 0):
        """last_only (= tail 1): the head over each item's LAST row only - bitwise what model(..., logits_to_keep=1)
        computes (a 1-row lm_head; a T-row one may accumulate differently), oracle_generate.py's greedy step. tail=N:
        the head over the last N rows (logits_to_keep=N's shape; hfcheck)."""
        tail = 1 if last_only else tail
        import transformers
        if transformers.__version__ != PINNED_TRANSFORMERS:
            raise RuntimeError(f"layer_major restates transformers {PINNED_TRANSFORMERS}'s MuseGlimmerTextModel."
                               f"forward; this is {transformers.__version__}")
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
                    p["h"] = layer(p["h"], attention_mask=p["masks"][cfg.layer_types[i]],
                                   position_embeddings=p["pe"] if cfg.layer_rope_theta[i] else None,
                                   position_ids=p["pos_ids"], past_key_values=p["cache"])
                pf.release()
        finally:
            pf.hold = False
            pf.release()
        out = []
        for j, p in enumerate(ps):
            select(keys[j], p["past"])
            raw = model.lm_head(tm.norm(p["h"][:, -tail:] if tail > 0 else p["h"]))[0]
            out.append((raw, softcap(raw, tcfg.output_multiplier, tcfg.final_logit_softcapping)))
            p.clear()
        return out
    return step_many


# ------------------------------------------------------------------------------------------------
# a sequential run (prompt in chunks, then greedy decode)

def read_ids(path: str) -> list[int]:
    return [int(x) for x in open(path, encoding="utf-8").read().split()]


def auto_chunk(n: int, chunk: int) -> int:
    """Whole up to 2048 ids, else chunks of 2048 (the engine's prefill chunk): a 2048-row forward's head is 0.83 GB of
    bf16 logits and the softcap holds ~4 such tensors; at 4096 rows the peak doubles (the Mac's capped container)."""
    if chunk >= 0:
        return chunk
    return 0 if n <= 2048 else 2048


@torch.no_grad()
def forward_chunks(model, ids: list[int], chunk: int, rec: Recorder, cache=None, on_rows=None, logits_to_keep: int = 0):
    """model(input_ids=chunk, past_key_values=cache) over `ids` in chunks (whole when chunk <= 0): the reference's
    prefill. on_rows(row0, raw [c][V], capped [c][V]) sees every chunk's rows (the model's dtype). -> the cache."""
    pos = 0 if cache is None else cache.get_seq_length()
    c = chunk if chunk > 0 else len(ids)
    for a in range(0, len(ids), c):
        part = ids[a:a + c]
        rec.base[rec.key] = pos + a
        out = model(input_ids=torch.tensor([part]), past_key_values=cache, use_cache=True, logits_to_keep=logits_to_keep)
        cache = out.past_key_values
        if on_rows is not None:
            on_rows(pos + a + len(part) - out.logits.shape[1], rec.raw[0], out.logits[0])
    return cache


def gap_line(g: torch.Tensor, what: str) -> str:
    g = g[torch.isfinite(g)].double()
    if g.numel() == 0:
        return f"{what}: no rows"
    qs = torch.quantile(g, torch.tensor([0.0, 0.01, 0.5], dtype=torch.float64)).tolist()
    return (f"{what} over {g.numel()} rows: ==0 {int((g == 0).sum())}; min / p01 / median "
            + " / ".join(f"{x:.4g}" for x in qs))


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, transformers {PINNED_TRANSFORMERS}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    tc = text_config(a.snapshot, a.layers)
    ids = read_ids(a.prompt)
    T = len(ids)
    if not 1 <= T <= a.max_prompt:
        die(f"{a.prompt}: {T} ids (1..{a.max_prompt})")
    model, pf = build_streamed(a.snapshot, tc, a.form)
    chunk = auto_chunk(T, a.chunk)
    act_from = max(0, T - a.act_tail) if a.act_tail >= 0 else 0
    log_from = max(0, T - max(1, a.logits_tail)) if a.logits_tail >= 0 else 0
    taps = TAP_LAYERS if a.taps else ()
    rec = Recorder(model, tc, act_from=act_from, sublayers=a.sublayers, taps=taps)
    print(f"config: {tc.num_hidden_layers} layers ({sum(t == 'full_attention' for t in tc.layer_types)} full / NoPE), "
          f"hidden {tc.hidden_size}, window {tc.sliding_window}, attn eager, {tc.dtype}; form {model.muse_form}; prompt "
          f"{T} ids, chunk {chunk or 'whole'}; activations from row {act_from}, logits from row {log_from}; taps "
          f"{list(rec.taps)}; built {time.time() - t0:.1f}s", flush=True)
    raws, caps, ties, gaps = [], [], [], []
    nll = torch.empty(max(T - 1, 0))
    ids_t = torch.tensor(ids)
    mult, capv = tc.output_multiplier, tc.final_logit_softcapping

    def on_rows(r0, raw, cap):
        for b0 in range(0, cap.shape[0], 256):             # 256 rows at a time: a 2048-row f32 block is 1.6 GB
            rb, cb = raw[b0:b0 + 256], cap[b0:b0 + 256]
            ra = r0 + b0
            r1 = ra + cb.shape[0]
            ties.append(cap_tie(rb, cb))
            gaps.append(raw_gap(rb))
            a_, b_ = ra, min(r1, T - 1)
            if b_ > a_:
                lf = cb[: b_ - a_].float()
                nll[a_:b_] = -torch.log_softmax(lf, -1).gather(-1, ids_t[a_ + 1:b_ + 1, None])[:, 0]
            if r1 > log_from:
                k = max(0, log_from - ra)
                raws.append(rb[k:].float().clone())
                caps.append(cb[k:].float().clone())
        if not torch.equal(softcap(raw, mult, capv), cap):
            raise RuntimeError("the model's logits are not softcap(lm_head's rows): the head changed")
    t1 = time.time()
    cache = forward_chunks(model, ids, chunk, rec, on_rows=on_rows)
    print(f"prompt forward {time.time() - t1:.1f}s", flush=True)
    tokens, tokens_raw = [], []
    t2 = time.time()
    for step in range(a.gen):
        nxt = int(torch.argmax(caps[-1][-1]).item())
        tokens.append(nxt)
        tokens_raw.append(int(torch.argmax(raws[-1][-1]).item()))
        rec.base[0] = T + step
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        on_rows(T + step, rec.raw[0], out.logits[0])
        if step == 0:
            print(f"  first decode step {time.time() - t2:.1f}s", flush=True)
    print(f"greedy {a.gen} (argmax of the capped row): {time.time() - t2:.1f}s -> {tokens}", flush=True)
    if tokens != tokens_raw:
        print(f"  the raw rows' argmax differs at steps {[i for i, (x, y) in enumerate(zip(tokens, tokens_raw)) if x != y]}")
    rec.remove()
    t = rec.tensors()
    t["logits.raw"] = torch.cat(raws, 0).contiguous()
    t["logits.cap"] = torch.cat(caps, 0).contiguous()
    t["cap_tie"] = torch.cat(ties).to(torch.uint8).contiguous()
    t["gap.raw"] = torch.cat(gaps).float().contiguous()
    t["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    t["tokens.raw"] = torch.tensor(tokens_raw, dtype=torch.int32)
    t["nll"] = nll.float().contiguous()
    n_rows = T + a.gen
    if t["cap_tie"].shape[0] != n_rows or ("taps" in t and t["taps"].shape[0] != n_rows):
        die(f"rows: cap_tie {t['cap_tie'].shape[0]}, taps {t.get('taps', torch.empty(n_rows)).shape[0]}, expected {n_rows}")
    last = t.get(f"H.L{tc.num_hidden_layers - 1}", t["logits.raw"])     # --act-tail 0 keeps no activations
    if not torch.isfinite(last.float()).all():
        die("the last layer's output has NaN/Inf - the weights are wrong")
    print(f"cap_tie: {int(t['cap_tie'].sum())} of {n_rows} rows (capped maximum shared, raw maximum unique)")
    print(gap_line(t["gap.raw"], "raw top-1 - top-2 gap"))
    ppl = math.exp(float(nll.double().mean())) if T > 1 else float("nan")
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt": os.path.abspath(a.prompt), "n_prompt": str(T),
            "gen": str(a.gen), "layers": str(tc.num_hidden_layers), "chunk": str(chunk), "form": model.muse_form,
            "act_row0": str(act_from), "logits_row0": str(log_from), "tap_layers": json.dumps(list(rec.taps)),
            "attn_implementation": "eager", "dtype": str(tc.dtype), "transformers": PINNED_TRANSFORMERS,
            "torch": torch.__version__, "ppl": f"{ppl:.6g}", "cap_tie_rows": str(int(t["cap_tie"].sum())),
            "model": "Muse-Glimmer-30B (MuseGlimmerForConditionalGeneration text path, tools/oracle/muse_ref.py, "
                     "layer-streamed)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(t, a.out, metadata=meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wrote {a.out}: {len(t)} tensors; prompt ppl {meta['ppl']}")
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB, prefetch wait {pf.seconds:.1f}s")


# ------------------------------------------------------------------------------------------------
# ppl

def tokenize_text(snapshot: str, path: str) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    text = open(path, encoding="utf-8").read().rstrip("\n")
    return tok.encode(text, add_special_tokens=False).ids


def cmd_ppl(a) -> None:
    t0 = time.time()
    tc = text_config(a.snapshot, a.layers)
    model, pf = build_streamed(a.snapshot, tc, a.form)
    rec = Recorder(model, tc, acts=False)
    srcs = [(n, tokenize_text(a.snapshot, p)) for n, p in (x.split("=", 1) for x in a.text)]
    srcs += [(n, read_ids(p)) for n, p in (x.split("=", 1) for x in a.ids)]
    if not srcs:
        die("ppl: no --text / --ids")
    for name, ids in srcs:
        ids = ids[: a.max_tokens]
        tot, n = 0.0, 0
        ids_t = torch.tensor(ids)

        def on_rows(r0, _raw, cap):
            nonlocal tot, n
            for b0 in range(0, cap.shape[0], 256):
                ra = r0 + b0
                b_ = min(ra + min(256, cap.shape[0] - b0), len(ids) - 1)
                if b_ > ra:
                    lp = torch.log_softmax(cap[b0:b0 + b_ - ra].float(), -1).gather(-1, ids_t[ra + 1:b_ + 1, None])[:, 0]
                    tot -= float(lp.double().sum())
                    n += b_ - ra
        t1 = time.time()
        forward_chunks(model, ids, auto_chunk(len(ids), a.chunk), rec, on_rows=on_rows)
        print(f"ppl {name}: {math.exp(tot / max(n, 1)):.4f} over {n} predicted tokens ({len(ids)} ids, "
              f"{tc.num_hidden_layers} layers, form {model.muse_form}, {time.time() - t1:.0f}s)", flush=True)
    rec.remove()
    print(f"wall {time.time() - t0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB")


# ------------------------------------------------------------------------------------------------
# hfcheck: streamed vs resident vs layer-major on any weights (--layers N)

@torch.no_grad()
def _drive(model, rec, ids, gen, keep):
    rows, cache = [], None
    rec.base[0] = 0
    out = model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=keep)
    cache = out.past_key_values
    rows.append((rec.raw[0].clone(), out.logits[0].clone()))
    toks = []
    for step in range(gen):
        nxt = int(rows[-1][1][-1].float().argmax())
        toks.append(nxt)
        rec.base[0] = len(ids) + step
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append((rec.raw[0].clone(), out.logits[0].clone()))
    return rows, toks


def cmd_hfcheck(a) -> None:
    t0 = time.time()
    tc = text_config(a.snapshot, a.layers)
    if a.prompt:
        ids = read_ids(a.prompt)[: a.n]
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, 200000, (a.n,), generator=g).tolist()
    res = {}
    for arm in ("streamed", "resident"):
        if arm == "streamed":
            model, pf = build_streamed(a.snapshot, tc, a.form)
        else:
            model = build_resident(a.snapshot, tc)
        rec = Recorder(model, tc)
        rows, toks = _drive(model, rec, ids, a.gen, 64)
        rec.remove()
        t = rec.tensors()
        t["raw"] = torch.cat([r for r, _ in rows])
        t["cap"] = torch.cat([c for _, c in rows])
        res[arm] = (t, toks)
        print(f"hfcheck {arm}: {len(ids)} ids + {a.gen} steps, {time.time() - t0:.0f}s", flush=True)
        if arm == "streamed":
            step = layer_major(model, pf)
            st = {}
            lm_rows = [step([(ids, 0, st)], tail=64)[0]]       # the streamed arm's logits_to_keep=64: the same head shape
            for i, tk in enumerate(toks):
                lm_rows.append(step([([tk], len(ids) + i, st)])[0])
            res["layer_major"] = ({"raw": torch.cat([r for r, _ in lm_rows]),
                                   "cap": torch.cat([c for _, c in lm_rows])}, toks)
            print(f"hfcheck layer_major: {time.time() - t0:.0f}s", flush=True)
        del model
    (st_, stok), (rt, rtok) = res["streamed"], res["resident"]
    bad = [k for k in rt if not torch.equal(st_[k], rt[k])]
    lt = res["layer_major"][0]
    bad += [f"layer_major {k}" for k in ("raw", "cap") if not torch.equal(lt[k], st_[k])]
    print(f"hfcheck: {len(rt)} tensors + layer-major logits, greedy {stok} vs {rtok}; differing: {bad[:8] if bad else 'none'}")
    if bad or stok != rtok:
        print("hfcheck: FAIL")
        sys.exit(1)
    print(f"hfcheck: BITWISE ({tc.num_hidden_layers} layers, {len(ids)} ids, window {tc.sliding_window}; form "
          f"{form_of(Ckpt(a.snapshot))}: streamed = resident = layer-major)")


# ------------------------------------------------------------------------------------------------
# facts

def cmd_facts(a) -> None:
    facts = _load("muse_facts")
    if a.gptq:
        argv = [a.snapshot, a.gptq] + (["--assistant", a.assistant] if a.assistant else []) + \
               (["--scan-quant"] if a.scan_quant else [])
        facts.main(argv)
    tc = text_config(a.snapshot)
    idx = os.path.join(a.snapshot, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx, encoding="utf-8"))["weight_map"]
        names = {k for k in wm if is_text(k)}
        hit = [f for f in ("gptq", "autoround", "bf16") if set(expected_names(tc, f)) == names]
        print(f"{a.snapshot}: {len(names)} text names " + (f"== expected_names(tc, {hit[0]!r})" if hit else
              "match NO form") + f"; vision_names {len(vision_names(wm))}")
        if not hit:
            sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("snapshot")
        p.add_argument("--layers", type=int)
        p.add_argument("--form", default="auto", choices=["auto", "gptq", "bf16"])
        p.add_argument("--chunk", type=int, default=-1, help="prefill chunk (-1: whole up to 2048 ids, else 2048)")
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=131072)
    r.add_argument("--logits-tail", type=int, default=1024, help="keep logits from row T - L (-1: all)")
    r.add_argument("--act-tail", type=int, default=256, help="keep H.L* from row T - A (-1: all)")
    r.add_argument("--taps", action="store_true", help=f"record the residual after layers {list(TAP_LAYERS)}")
    r.add_argument("--sublayers", action="store_true", help="also attn.L* / mlp.L*")
    p = sub.add_parser("ppl")
    common(p)
    p.add_argument("--text", action="append", default=[], help="NAME=FILE (tokenized by the snapshot's tokenizer.json)")
    p.add_argument("--ids", action="append", default=[], help="NAME=FILE")
    p.add_argument("--max-tokens", type=int, default=4096)
    h = sub.add_parser("hfcheck")
    common(h)
    h.add_argument("--prompt")
    h.add_argument("--n", type=int, default=2100)
    h.add_argument("--gen", type=int, default=4)
    f = sub.add_parser("facts")
    f.add_argument("snapshot")
    f.add_argument("--gptq", help="the GPTQ snapshot: also print muse_facts.py's tables (snapshot = the bf16 one)")
    f.add_argument("--assistant")
    f.add_argument("--scan-quant", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "ppl": cmd_ppl, "hfcheck": cmd_hfcheck, "facts": cmd_facts}[a.cmd](a)


if __name__ == "__main__":
    main()
