#!/usr/bin/env python3
"""Spec 15a: the CPU reference of Ornith 1.5 35B-A3B FROM ITS INT4 CHECKPOINT (R1, R2, M1).

The engine runs `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` (int4 g64 sym, GPTQ packing,
per-expert `mlp.experts.E.{gate,up,down}_proj`, a||b int4 too - spec 15 §13). A reference made
from the SAME int4 weights - dequantised by the repo's one rule (dequant.py, stream.py's
bit-identical C++ dequant_t: bf16_rne(float(q - 8) x float(scale))) - is one the engine can match
token for token, as spec 18a did for K2. The model is transformers' own Qwen3_5MoeForCausalLM
(transformers 5.15, the box image's and agnes-ref-img's version), built on `meta` from the
checkpoint's text config and LAYER-STREAMED (stream.py): embed / final norm / lm_head resident,
each decoder layer's weights made by a forward pre-hook and dropped after.

    ornith_ref.py facts <snapshot>             the checkpoint facts spec 15 §13 rests on (headers only)
    ornith_ref.py run <snapshot> --prompt <ids> --out <golden.safetensors> [--gen 32]
                     [--mtp-out <dir>] [--mtp-rows 32] [--mtp-experts rtn|bf16]
                     [--experts grouped_mm|eager]

**The routed experts are materialised on demand.** transformers' Qwen3_5MoeExperts holds every
expert of a layer as two 3D parameters (gate_up_proj [E][2I][H], down_proj [E][H][I]: 1.61 GB
bf16 per layer). Here both are ONE pair of buffers shared by all layers, and a forward pre-hook
on `mlp.experts` - which runs after the router, so it receives top_k_index - dequantises into
them exactly the experts this forward routes to (gate = rows [0, I), up = rows [I, 2I): the
fused layout AutoRound unfused, spec 15 §10). Neither experts implementation reads an expert no
token routes to (`grouped_mm` multiplies only the groups its offsets name; `eager` loops over
the hit experts), so the result is the fully materialised model's, and a decode step dequantises
9 experts per layer instead of 257 (test_ornith_ref.py checks the streamed, lazily filled model
against a resident one, bitwise).

Output (`run`): dump.py's layout - resid / mixer / mlp.L{i} bf16 [T, 2048], gdn_state / conv_state
.L{i}, logits f32 [T + gen, V], tokens i32 [gen] - plus, for EVERY layer and every forward (the
prompt's rows, then one per generated step, aligned with `logits`):
    router_logits.L{i}       bf16 [T + gen, 256]   the router's F.linear output (ornith_decode_test R2)
    shared_gate_logits.L{i}  bf16 [T + gen, 1]     shared_expert_gate's output, pre-sigmoid
    route.ids.L{i}           i32  [T + gen, 8]     the reference's top-8, in its topk order
    route.w.L{i}             bf16 [T + gen, 8]     their renormalised weights
    route.gap.L{i}           f32  [T + gen]        p(8th) - p(9th) of the fp32 softmax (0 = a tie)
The log prints the gap distribution (R2's near-tie tolerance, 15a Task 3).

`--mtp-out DIR`: the MoE MTP head's M1 reference for mtp_head_test (mtp_head_ornith_test):
DIR/m1/<name>.mtp.safetensors (`logits` f32 [R][V], `pos`, `next`: the head's depth-1 logits on
the main model's post-norm hidden of rows n-1 .. n-2+R, as mtp_ref.dump writes them) and
DIR/<name>.cont256.ids - the continuation the engine teacher-forces: this run's 32 greedy tokens
plus the last row's argmax (33 ids; the file name is mtp_head_test's). The hidden rows come from
this same run (prompt forward + the greedy steps), the head is mtp_ref.MtpHead's wiring with a
Qwen3_5MoeDecoderLayer. `--mtp-experts rtn` (default) quantises the head's 257 bf16 SwiGLU
experts exactly as the engine's loader does (loader/rtn.h: int4 g64 sym, round to nearest on the
stored f16 scale) and dequantises them by dequant.py, so the comparison is like for like; `bf16`
keeps the shipped weights (the quantisation's own effect on the draft).

dtype: the checkpoint's top-level config says float16 (AutoRound's export); its text config,
every non-quantised tensor and the engine are bf16, and so is this reference.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ornith_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dump = _load("dump")
_stream = _dump._stream
_dequant = _dump._dequant

LP = "model.language_model.layers."
EXPERT_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.qweight$")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def mqm():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    return m


def text_config(snapshot: str, experts: str = "grouped_mm"):
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return text_config_from_dict(json.load(f), experts)


def text_config_from_dict(d: dict, experts: str = "grouped_mm"):
    """config.json's dict -> the Qwen3_5MoeTextConfig to build the model from (the
    quantization_config stays out: the weights are dequantised here, not by transformers)."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    tc = Qwen3_5MoeConfig(**{k: v for k, v in d.items() if k != "quantization_config"}).get_text_config()
    if tc.model_type != "qwen3_5_moe_text":
        die(f"text config model_type {tc.model_type!r}, expected qwen3_5_moe_text (Ornith)")
    if tc.tie_word_embeddings:
        die("tie_word_embeddings is true; Ornith ships a separate lm_head")
    tc._attn_implementation = "eager"          # fp32 softmax, the contract (doc 03)
    tc._experts_implementation = experts       # recorded in the output (spec 15 §10)
    tc.dtype = torch.bfloat16
    return tc


# --------------------------------------------------------------------------------------------
# the RTN of loader/rtn.h, for the MTP head's bf16 experts

def gptq_pack(q: torch.Tensor) -> torch.Tensor:
    """q [K, N] ints 0..15 -> qweight int32 [K/8, N], k = 8r + i in nibble i (low first)."""
    K, N = q.shape
    r = q.to(torch.int64).reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= r[:, i, :] << (4 * i)
    return (packed - ((packed >> 31) << 32)).to(torch.int32)


def rtn_int4_g64(w: torch.Tensor):
    """loader::rtn_int4_g64 in torch: w bf16 [N][K] -> (qweight int32 [K/8][N], scales f16 [K/64][N]).

    Per output column n and group of 64 k: amax = max |w| (bf16 values, exact in fp32),
    s16 = f16(2 amax / 15) round to nearest even, q = clamp(rint(w / f32(s16)) + 8, 0, 15) -
    the stored scale divides; an all-zero group has s16 = 0 and q = 8."""
    N, K = w.shape
    wf = w.float().reshape(N, K // 64, 64)
    amax = wf.abs().amax(-1)                                   # [N, K/64]
    s16 = (2.0 * amax / 15.0).to(torch.float16)
    s = s16.float().unsqueeze(-1)
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))) + 8.0,
                    torch.full_like(wf, 8.0)).clamp(0.0, 15.0)
    q = q.reshape(N, K).t().contiguous()                       # [K, N]
    return gptq_pack(q), s16.t().contiguous()                  # scales [K/64, N]


def rtn_dequant(w: torch.Tensor) -> torch.Tensor:
    """bf16 [N][K] -> its RTN int4 g64 form dequantised by dequant.py ([N][K] bf16)."""
    qw, sc = rtn_int4_g64(w)
    return _dequant.dequant_gptq(qw, sc, 64).t().contiguous()


# --------------------------------------------------------------------------------------------
# the streamed main model

class LazyExperts:
    """One shared [E][2I][H] / [E][H][I] buffer pair; a layer's routed experts dequantised into
    it on demand (module docstring). `where` maps checkpoint names to shard handles.

    `keep`: layers that get buffers of their OWN instead, filled on demand like the shared pair
    but never overwritten - so each of their experts is dequantised once per run, not once per
    forward (1.61 GB per layer when every expert has been used; oracle_generate.py sizes it to the
    container's cap). The experts module reads the same bf16 values either way."""

    def __init__(self, where, tc, keep=()):
        self.where, self.tc = where, tc
        self.keep = set(keep)
        E, I, H = tc.num_experts, tc.moe_intermediate_size, tc.hidden_size
        self.shape = ((E, 2 * I, H), (E, H, I))
        shared = len(self.keep) < tc.num_hidden_layers
        self.gate_up = torch.empty(*self.shape[0], dtype=torch.bfloat16) if shared else None
        self.down = torch.empty(*self.shape[1], dtype=torch.bfloat16) if shared else None
        self.own: dict[int, tuple[torch.Tensor, torch.Tensor, set]] = {}
        self.lock = threading.Lock()
        self.filled = 0
        self.checked: set[str] = set()
        # The experts of the layer the shared pair holds now: a layer-major batch (oracle_generate.py
        # --batch) calls the same layer once per sequence in a row, and an expert already filled
        # for this layer is not dequantised again (the bytes would be the same).
        self.layer: int | None = None
        self.have: set[int] = set()

    def tensors(self, layer: int):
        """(gate_up, down) for `layer`'s experts module: its own pair if kept, else the shared one.
        No other side effect - stream.py's prefetch thread calls it (layer_sd)."""
        if layer not in self.keep:
            return self.gate_up, self.down
        with self.lock:
            if layer not in self.own:
                self.own[layer] = (torch.empty(*self.shape[0], dtype=torch.bfloat16),
                                   torch.empty(*self.shape[1], dtype=torch.bfloat16), set())
            return self.own[layer][:2]

    def buffers(self, layer: int):
        """(gate_up, down, the set of experts they hold for `layer`) - the forward thread only
        (fill): the shared pair's set is reset whenever another layer is filled."""
        if layer in self.keep:
            self.tensors(layer)
            return self.own[layer]
        if layer != self.layer:
            self.layer, self.have = layer, set()
        return self.gate_up, self.down, self.have

    def _linear(self, base: str) -> torch.Tensor:
        qk, sk = base + ".qweight", base + ".scales"
        if qk not in self.where:
            die(f"{qk} is not in the checkpoint (per-expert GPTQ naming, spec 15 §10)")
        out = _stream.dequant_t(self.where[qk].get_tensor(qk), self.where[sk].get_tensor(sk), 64,
                                check=base not in self.checked)
        self.checked.add(base)
        return out

    def fill(self, layer: int, ids) -> None:
        I = self.tc.moe_intermediate_size
        gate_up, down, have = self.buffers(layer)
        for e in ids:
            if e in have:
                continue
            have.add(e)
            b = f"{LP}{layer}.mlp.experts.{e}."
            gate_up[e, :I] = self._linear(b + "gate_proj")
            gate_up[e, I:] = self._linear(b + "up_proj")
            down[e] = self._linear(b + "down_proj")
            self.filled += 1

    def hook(self, layer: int):
        def fn(_mod, args):
            self.fill(layer, sorted(set(int(x) for x in args[1].reshape(-1).tolist())))
        return fn


def build_streamed(snapshot: str, tc, keep_dense=(), keep_experts=()):
    """Qwen3_5MoeForCausalLM on meta, resident tensors loaded, layers streamed, experts lazy.
    keep_dense: layers whose dequantised non-expert weights stay materialised after their first
    forward (stream.attach's keep, ~74 MB a layer); keep_experts: layers with their own expert
    buffers (LazyExperts' keep). Neither changes a value any forward computes."""
    m = mqm()
    with torch.device("meta"):
        model = m.Qwen3_5MoeForCausalLM(tc)
    where = _dump.open_shards(snapshot)
    n = tc.num_hidden_layers
    per: dict[int, list[str]] = {i: [] for i in range(n)}
    rest = []
    for key in sorted(where):
        mm = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
        if mm and ".mlp.experts." in key:
            continue                                   # LazyExperts reads these
        (per[int(mm.group(1))] if mm else rest).append(key)
    resident: dict[str, torch.Tensor] = {}
    _, lm_kind = _dump.convert(where, rest, 64, resident)
    print(f"lm_head: {lm_kind}")
    lazy = LazyExperts(where, tc, keep_experts)

    def layer_sd(i: int) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        _dump.convert(where, per[i], 64, sd, fast=True)
        pre = f"model.layers.{i}."
        bad = [k for k in sd if not k.startswith(pre)]
        if bad:
            raise RuntimeError(f"layer {i}: {bad[:3]} outside {pre}")
        out = {k[len(pre):]: v for k, v in sd.items()}
        out["mlp.experts.gate_up_proj"], out["mlp.experts.down_proj"] = lazy.tensors(i)
        return out

    layers = list(model.model.layers)
    pf = _stream.attach(model, layers, layer_sd, resident,
                        rebuild=[(model.model, "rotary_emb", lambda: type(model.model.rotary_emb)(tc))],
                        keep=keep_dense)
    for i, layer in enumerate(layers):
        layer.mlp.experts.register_forward_pre_hook(lazy.hook(i))
    return model.eval(), pf, lazy


LAYER_MAJOR_TRANSFORMERS = "5.15."     # the version layer_major's prologue was checked against


def layer_major(model, pf):
    """-> step_many(items) for the streamed model of build_streamed: items = [(ids, pos, state)],
    state a dict whose "cache" is None before the prompt; returns each item's last logits row
    (fp32 [V]), as `model(input_ids=[ids], past_key_values=cache, use_cache=True)
    .logits[0, -1].float()` would, BITWISE.

    Layer-major: every item through decoder layer 0, then every item through layer 1, ... with
    stream.py's hold mode keeping a layer materialised until every item has run it, and
    LazyExperts dequantising each routed expert once per layer pass - so a pass reads each layer's
    weights once for all items. Each item runs alone, on its own [1, T] tensors and its own
    DynamicCache: what Qwen3_5MoeTextModel.forward does before its layer loop (embed, positions
    from the cache length, the two masks, rotary - transformers 5.15's code, restated here with
    the module's own functions), each decoder layer called with the arguments that loop passes,
    then the final norm and Qwen3_5MoeForCausalLM's lm_head over every row. No op sees two items,
    so nothing depends on what else is in the batch (tools/toolcall/test_oracle_batched.py checks ids and logits
    against the sequential path, bitwise)."""
    m = mqm()
    tm = model.model
    cfg = tm.config
    layers = list(tm.layers[: cfg.num_hidden_layers])

    def prologue(ids, pos, st):
        x = torch.tensor([list(ids)])
        h = tm.embed_tokens(x)
        if st.get("cache") is None:
            st["cache"] = m.DynamicCache(config=cfg)
        cache = st["cache"]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a cache of {past} tokens")
        position_ids = torch.arange(h.shape[1], device=h.device) + past
        position_ids = position_ids.view(1, 1, -1).expand(4, h.shape[0], -1)
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
        mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": cache,
              "position_ids": text_position_ids}
        masks = {"full_attention": m.create_causal_mask(**mk),
                 "linear_attention": m.create_recurrent_attention_mask(**mk)}
        return {"h": h, "pe": tm.rotary_emb(h, position_ids), "masks": masks, "tpos": text_position_ids,
                "cache": cache}

    @torch.no_grad()
    def step_many(items):
        import transformers
        if not transformers.__version__.startswith(LAYER_MAJOR_TRANSFORMERS):
            raise RuntimeError(f"layer_major restates transformers {LAYER_MAJOR_TRANSFORMERS}x's "
                               f"Qwen3_5MoeTextModel.forward; this is {transformers.__version__} - re-check it "
                               f"(tools/toolcall/test_oracle_batched.py) or run --batch 1")
        ps = [prologue(ids, pos, st) for ids, pos, st in items]
        pf.hold = True
        try:
            for i, layer in enumerate(layers):
                lt = cfg.layer_types[i]
                for p in ps:
                    p["h"] = layer(p["h"], position_embeddings=p["pe"], attention_mask=p["masks"][lt],
                                   position_ids=p["tpos"], past_key_values=p["cache"], use_cache=True,
                                   output_router_logits=False)
                pf.release()
        finally:
            pf.hold = False
            pf.release()
        rows = []
        for p in ps:
            h = tm.norm(p["h"])
            rows.append(model.lm_head(h[:, slice(0, None), :])[0, -1].float())
            p.clear()
        return rows
    return step_many


class Recorder:
    """Every layer's router / shared-gate outputs, per forward, and the final norm's output."""

    def __init__(self, model, tc):
        self.k = tc.num_experts_per_tok
        self.n = tc.num_hidden_layers
        self.rows: dict[str, list[torch.Tensor]] = {}
        self.post: list[torch.Tensor] = []
        for i, layer in enumerate(model.model.layers):
            layer.mlp.gate.register_forward_hook(self._router(i))
            layer.mlp.shared_expert_gate.register_forward_hook(self._put(f"shared_gate_logits.L{i}"))
        model.model.norm.register_forward_hook(lambda _m, _a, out: self.post.append(out.detach()[0].clone()))

    def _put(self, name):
        def fn(_m, _a, out):
            self.rows.setdefault(name, []).append(out.detach().reshape(-1, out.shape[-1]).clone())
        return fn

    def _router(self, i):
        def fn(_m, _a, out):
            logits, scores, idx = out
            self.rows.setdefault(f"router_logits.L{i}", []).append(logits.detach().clone())
            self.rows.setdefault(f"route.w.L{i}", []).append(scores.detach().clone())
            self.rows.setdefault(f"route.ids.L{i}", []).append(idx.detach().to(torch.int32).clone())
            p = torch.softmax(logits.detach().float(), -1).sort(-1, descending=True).values
            self.rows.setdefault(f"route.gap.L{i}", []).append((p[:, self.k - 1] - p[:, self.k]).clone())
        return fn

    def tensors(self) -> dict[str, torch.Tensor]:
        return {k: torch.cat(v, 0).contiguous() for k, v in self.rows.items()}


def gap_summary(t: dict[str, torch.Tensor], n_layers: int) -> str:
    g = torch.cat([t[f"route.gap.L{i}"] for i in range(n_layers)])
    qs = torch.quantile(g.double(), torch.tensor([0.0, 0.001, 0.01, 0.1, 0.5], dtype=torch.float64))
    return (f"route gap p8 - p9 over {g.numel()} (row, layer): ==0 {int((g == 0).sum())}, "
            f"<1e-4 {int((g < 1e-4).sum())}, <1e-3 {int((g < 1e-3).sum())}; quantiles "
            f"min/0.1%/1%/10%/50% " + " / ".join(f"{x:.3g}" for x in qs.tolist()))


# --------------------------------------------------------------------------------------------
# the MoE MTP head

def mtp_head(snapshot: str, tc, model, experts: str):
    """mtp_ref.MtpHead's wiring with Ornith's MoE layer (spec 15 §12): fc, the two pre-fc norms,
    one full-attention Qwen3_5MoeDecoderLayer, mtp.norm; embed and lm_head shared."""
    mtp_ref = _load("mtp_ref")
    m = mqm()
    keys, get = _stream.checkpoint_reader(snapshot)
    names = [k for k in keys if k.startswith("mtp.")]
    E, I = tc.num_experts, tc.moe_intermediate_size
    if len(names) != 14 + 3 * (E + 1):          # Ornith: 785 (spec 15 §12)
        die(f"{len(names)} mtp.* tensors, expected {14 + 3 * (E + 1)}")
    L = "mtp.layers.0."
    sd: dict[str, torch.Tensor] = {}
    gate_up = torch.empty(E, 2 * I, tc.hidden_size, dtype=torch.bfloat16)
    down = torch.empty(E, tc.hidden_size, I, dtype=torch.bfloat16)
    conv = rtn_dequant if experts == "rtn" else (lambda w: w)
    n_rtn = 0
    for k in names:
        t = get(k)
        if t.dtype != torch.bfloat16:
            die(f"{k} is {t.dtype}; the published head is all bf16")
        mm = re.match(r"mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", k)
        if mm:
            e, which = int(mm.group(1)), mm.group(2)
            w = conv(t)
            n_rtn += 1
            if which == "gate":
                gate_up[e, :I] = w
            elif which == "up":
                gate_up[e, I:] = w
            else:
                down[e] = w
        elif ".mlp.shared_expert." in k and k.endswith("_proj.weight"):
            sd[k] = conv(t)                      # the engine quantises the shared expert too
            n_rtn += 1
        else:
            sd[k] = t
    fa = tc.layer_types.index("full_attention")
    with torch.device("meta"):
        layer = m.Qwen3_5MoeDecoderLayer(tc, fa)
    lsd = {k[len(L):]: v for k, v in sd.items() if k.startswith(L)}
    lsd["mlp.experts.gate_up_proj"] = gate_up
    lsd["mlp.experts.down_proj"] = down
    layer.load_state_dict(lsd, strict=True, assign=True)

    class _Flat(torch.nn.Module):         # the MoE block takes [B, S, H]; MtpHead passes [T, H]
        def __init__(self, moe):
            super().__init__()
            self.moe = moe

        def forward(self, x):
            return self.moe(x[None])[0]

    layer.mlp = _Flat(layer.mlp)

    class OrnithHead(mtp_ref.MtpHead):
        def __init__(self):     # MtpHead.__init__ builds a dense Qwen3_5DecoderLayer: not this
            self.tc, self.layer, self.attn = tc, layer.eval(), layer.self_attn

            def norm(key):
                nn = m.Qwen3_5MoeRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
                nn.weight = torch.nn.Parameter(sd[key], requires_grad=False)
                return nn.eval()
            self.pre_e = norm("mtp.pre_fc_norm_embedding.weight")
            self.pre_h = norm("mtp.pre_fc_norm_hidden.weight")
            self.norm = norm("mtp.norm.weight")
            self.fc = sd["mtp.fc.weight"]
            self.embed = model.model.embed_tokens.weight
            self.lm_head = model.lm_head.weight
            self.rotary = m.Qwen3_5MoeTextRotaryEmbedding(tc)

        def _qkv(self, x, pos):     # MtpHead._qkv with this module's rotary helper
            a, T, D = self.attn, x.shape[0], self.attn.head_dim
            q, gate = torch.chunk(a.q_proj(x).view(T, -1, D * 2), 2, dim=-1)
            gate = gate.reshape(T, -1)
            q = a.q_norm(q).transpose(0, 1)
            k = a.k_norm(a.k_proj(x).view(T, -1, D)).transpose(0, 1)
            v = a.v_proj(x).view(T, -1, D).transpose(0, 1)
            pid = pos.view(1, 1, T).expand(3, 1, T)
            cos, sin = self.rotary(x, pid)
            q, k = m.apply_rotary_pos_emb(q[None], k[None], cos, sin)
            return q[0], gate, k[0], v

    print(f"MTP head: {len(names)} tensors, {n_rtn} expert linears "
          f"{'int4 g64 RTN (loader/rtn.h) + dequant.py' if experts == 'rtn' else 'bf16 as shipped'}")
    return OrnithHead()


# --------------------------------------------------------------------------------------------
# commands

def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, intra-op threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    gs = _dump.check_quant_config(a.snapshot)
    tc = text_config(a.snapshot, a.experts)
    ids = [int(x) for x in open(a.prompt, encoding="utf-8").read().split()]
    if not ids or len(ids) > a.max_prompt:
        die(f"{a.prompt}: {len(ids)} ids (1..{a.max_prompt})")
    model, pf, lazy = build_streamed(a.snapshot, tc)
    impl = getattr(model.config, "_experts_implementation", None)
    print(f"config: {tc.num_hidden_layers} layers, {tc.num_experts} experts top-{tc.num_experts_per_tok}, "
          f"hidden {tc.hidden_size}, attn {tc._attn_implementation}, experts {impl}; "
          f"prompt {len(ids)} ids; loaded {time.time() - t0:.1f}s")
    rec = Recorder(model, tc)
    caps: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t.detach()[0].to(torch.bfloat16).clone()
        return fn
    hs = []
    gdn = [i for i, t in enumerate(tc.layer_types) if t == "linear_attention"]
    for i, layer in enumerate(model.model.layers):
        mixer = layer.linear_attn if i in gdn else layer.self_attn
        hs += [layer.register_forward_hook(hook(f"resid.L{i}")),
               mixer.register_forward_hook(hook(f"mixer.L{i}")),
               layer.mlp.register_forward_hook(hook(f"mlp.L{i}"))]
    t1 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
    for h in hs:
        h.remove()
    print(f"prompt forward {time.time() - t1:.1f}s ({lazy.filled} expert dequants)", flush=True)
    cache = out.past_key_values
    states = {}
    for i in gdn:
        states[f"gdn_state.L{i}"] = _dump.read_state(cache.layers[i], "recurrent_states",
                                                     f"gdn_state.L{i}").detach()[0].float().clone()
        states[f"conv_state.L{i}"] = _dump.read_state(cache.layers[i], "conv_states",
                                                      f"conv_state.L{i}").detach()[0].clone()
    rows = [out.logits[0].float()]
    tokens: list[int] = []
    t2 = time.time()
    for step in range(a.gen):
        nxt = int(torch.argmax(rows[-1][-1]).item())
        tokens.append(nxt)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0].float())
        if step == 0:
            print(f"  first decode step {time.time() - t2:.1f}s", flush=True)
    print(f"greedy {a.gen}: {time.time() - t2:.1f}s -> {tokens}")
    tensors = dict(caps)
    tensors.update(states)
    tensors.update(rec.tensors())
    tensors["logits"] = torch.cat(rows, 0).contiguous()
    tensors["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    n_rows = len(ids) + a.gen
    for i in range(tc.num_hidden_layers):
        for k in ("router_logits", "shared_gate_logits", "route.ids", "route.w", "route.gap"):
            if tensors[f"{k}.L{i}"].shape[0] != n_rows:
                die(f"{k}.L{i} has {tensors[f'{k}.L{i}'].shape[0]} rows, expected {n_rows}")
    if len(caps) != 3 * tc.num_hidden_layers:
        die(f"{len(caps)} activations captured")
    if not torch.isfinite(tensors[f"resid.L{tc.num_hidden_layers - 1}"].float()).all():
        die("the last layer's output has NaN/Inf - the dequantised weights are wrong")
    print(gap_summary(tensors, tc.num_hidden_layers))
    meta = {"snapshot": os.path.abspath(a.snapshot), "prompt_ids": " ".join(map(str, ids)),
            "n_prompt": str(len(ids)), "gen": str(a.gen), "attn_implementation": "eager",
            "experts_implementation": str(impl), "group_size": str(gs),
            "model": "ornith (Qwen3_5MoeForCausalLM, int4 checkpoint dequantised, tools/oracle/ornith_ref.py)",
            "weights": "layer-streamed, routed experts on demand (tools/oracle/ornith_ref.py)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(tensors, a.out, metadata=meta)
    print(f"wrote {a.out}: {len(tensors)} tensors")

    if a.mtp_out:
        name = os.path.basename(a.prompt).split(".")[0]
        n, R = len(ids), a.mtp_rows
        if R > a.gen:
            die(f"--mtp-rows {R} needs --gen >= {R}")
        cont = tokens + [int(torch.argmax(rows[-1][-1]).item())]      # 33 ids at --gen 32
        post = torch.cat(rec.post, 0)                                   # [n + gen, H]
        head = mtp_head(a.snapshot, tc, model, a.mtp_experts)
        all_ids = torch.tensor(ids + cont[: R + 1])
        T = all_ids.numel()
        with torch.no_grad():
            (lg, _), = head.chain(post[: T - 1], all_ids[1:T], torch.arange(T - 1), 1)
        sel = torch.arange(n - 1, n - 1 + R)
        os.makedirs(os.path.join(a.mtp_out, "m1"), exist_ok=True)
        save_file({"logits": lg[sel].float().contiguous(), "pos": sel.to(torch.int32).contiguous(),
                   "next": all_ids[sel + 1].to(torch.int32).contiguous()},
                  os.path.join(a.mtp_out, "m1", f"{name}.mtp.safetensors"),
                  metadata={"head_experts": a.mtp_experts, "snapshot": os.path.abspath(a.snapshot)})
        with open(os.path.join(a.mtp_out, f"{name}.cont256.ids"), "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, cont)) + "\n")
        print(f"MTP M1: {name} rows {n - 1}..{n - 2 + R}, {len(cont)} continuation ids -> {a.mtp_out}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB, prefetch wait {pf.seconds:.1f}s")


def cmd_facts(a) -> None:
    """What spec 15 §13 read from the index and headers, re-checked on the downloaded snapshot
    (headers and small files only): run it first; a mismatch stops the reference before it
    spends an hour."""
    import struct
    s = a.snapshot
    cfg = json.load(open(os.path.join(s, "config.json"), encoding="utf-8"))
    q = cfg["quantization_config"]
    want = {"bits": 4, "group_size": 64, "sym": True, "desc_act": False, "quant_method": "gptq"}
    bad = {k: q.get(k) for k, v in want.items() if q.get(k) != v}
    if bad:
        die(f"quantization_config differs: {bad}")
    print(f"quant: {want}, provider {q.get('provider')}, autoround {q.get('autoround_version')}, "
          f"{len(q.get('dynamic', {}))} dynamic rules (all '-:' {all(r.startswith('-:') for r in q.get('dynamic', {}))})")
    wm = json.load(open(os.path.join(s, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    dt: dict[str, str] = {}
    for fn in sorted(set(wm.values())):          # the headers only: 8-byte length, then JSON
        with open(os.path.join(s, fn), "rb") as h:
            hdr = json.loads(h.read(struct.unpack("<Q", h.read(8))[0]))
        dt.update({k: v["dtype"] for k, v in hdr.items() if k != "__metadata__"})
    if set(dt) != set(wm):
        die("the index and the shard headers disagree")
    ab = [k for k in dt if re.search(r"linear_attn\.in_proj_[ab]\.qweight$", k)]
    experts = [k for k in dt if EXPERT_RE.match(k)]
    mtp = [k for k in dt if k.startswith("mtp.")]
    checks = {
        "in_proj_a/b int4 (30 layers x 2)": len(ab) == 60,
        "per-expert GPTQ experts (40 x 256 x 3)": len(experts) == 30720,
        "scales F16": all(dt[k] == "F16" for k in dt if k.endswith(".scales")),
        "routers / shared gates BF16": all(dt[k] == "BF16" for k in dt
                                           if k.endswith(("mlp.gate.weight", "mlp.shared_expert_gate.weight"))),
        "MTP head 785 BF16": len(mtp) == 785 and all(dt[k] == "BF16" for k in mtp),
        "lm_head BF16": dt.get("lm_head.weight") == "BF16",
    }
    for k, ok in checks.items():
        print(f"  {'ok ' if ok else 'BAD'} {k}")
    tj = json.load(open(os.path.join(s, "tokenizer.json"), encoding="utf-8"))
    n_tok = len(tj["model"]["vocab"]) + len(tj["added_tokens"])
    tmpl = hashlib.sha256(open(os.path.join(s, "chat_template.jinja"), "rb").read()).hexdigest()
    gen = json.load(open(os.path.join(s, "generation_config.json"), encoding="utf-8"))
    print(f"  tokenizer.json ids {n_tok} (descriptor vocab_used 248077), chat_template sha256 "
          f"{tmpl[:8]} (vendored 182e77dd), eos {gen.get('eos_token_id')}")
    if not all(checks.values()) or n_tok != 248077 or not tmpl.startswith("182e77dd"):
        die("the checkpoint is not the one spec 15 §13 read")
    print("facts OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("snapshot")
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--max-prompt", type=int, default=4096)
    r.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "eager"])
    r.add_argument("--mtp-out")
    r.add_argument("--mtp-rows", type=int, default=32)
    r.add_argument("--mtp-experts", default="rtn", choices=["rtn", "bf16"])
    a = ap.parse_args()
    {"facts": cmd_facts, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
