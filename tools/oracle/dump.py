#!/usr/bin/env python3
"""The oracle: layer-by-layer golden tensors for one prompt, from transformers on CPU.

This is the reference the C++ engine is tested against (doc 04 §testing, doc 10).
The model is built from the config and loaded from a state dict we dequantise
ourselves with tools/oracle/dequant.py - the SAME function the C++ loader is
bit-compared against - so the oracle and the engine start from identical bf16
weights. No `fla` is installed in the reference image on purpose: the pure-torch
gated-delta-rule in modeling_qwen3_5.py IS the contract (doc 03).

    dump.py <snapshot> --prompt <ids-file> --out <out.safetensors> [--gen 32] [--max-prompt 64]
            [--mlp-in <out.safetensors> --mlp-in-layers 0,3,35,71]

Spec 14: an Agnes 3.0 Flash checkpoint (config.json `model_type: agnes`) is
detected and built as tools/oracle/agnes.py's reference - the same
Qwen3_5ForCausalLM, its names mapped (`delta_attn.`/`global_attn.`), the
parallel FFN summed into each MLP, unfolded. `--mlp-in` also writes each listed
layer's MLP INPUT (`mlp_in.L{i}`, bf16 [T, hidden]) for agnes_fold.py --inputs.

Output tensors (batch dimension squeezed; batch is always 1):
    resid.L{i}      bf16 [T, hidden]   decoder layer i output, all prompt positions
    mixer.L{i}      bf16 [T, hidden]   linear_attn / self_attn submodule output
    mlp.L{i}        bf16 [T, hidden]   mlp submodule output
    gdn_state.L{i}  f32  [Hv, Dk, Dv]  recurrent state after the prompt, GDN layers only
    conv_state.L{i} bf16 [conv_dim, K] conv window after the prompt, GDN layers only
    logits          f32  [T + gen, V]  every prompt position, then one row per generated step
    tokens          i32  [gen]         greedy continuation

    logits[t]      for t < T is prompt position t;  tokens[0]   = argmax(logits[T-1])
    logits[T + j]  is the forward that consumed tokens[j];      tokens[j+1] = argmax(logits[T+j])
    the last row's argmax is deliberately not in `tokens` - it is a free extra check.

Hard aborts (a golden file is worthless if any of these is fudged): strict load,
no meta tensors left, `tokens` length == --gen, every GDN layer contributed a
state, resid of the last layer finite.
"""
import os
import sys

# `tokenize.py` next to this script shadows the stdlib `tokenize` that `inspect`
# imports, and torch/transformers import `inspect`. Drop this directory from
# sys.path BEFORE importing anything third-party (python -P does the same, but
# this works however the script is invoked).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

# The one meaning of the int4 bits lives in dequant.py; load it by path rather
# than by putting _HERE back on sys.path (see the guard above).
_spec = importlib.util.spec_from_file_location("oracle_dequant", os.path.join(_HERE, "dequant.py"))
_dequant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dequant)
dequant_gptq = _dequant.dequant_gptq
_spec = importlib.util.spec_from_file_location("oracle_agnes", os.path.join(_HERE, "agnes.py"))
_agnes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agnes)
_spec = importlib.util.spec_from_file_location("oracle_stream", os.path.join(_HERE, "stream.py"))
_stream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stream)

SKIP_PREFIXES = ("model.visual.", "mtp.")
QUANT_SUFFIXES = (".scales", ".qzeros", ".g_idx")

# Column-chunk width for the dequant of very wide tensors. The only one that
# needs it is an int4 `lm_head` at [5120, 248320]: unchunked it materialises
# several intermediates of that shape (int32 nibbles, fp32 q, fp32 scales,
# fp32 product) for ~25 GB of transient RSS on top of a ~47 GB state dict.
# dequant.py's chunking is bit-identical by construction and asserted in its
# fixture writer, so this changes peak memory and nothing else.
DEQUANT_CHUNK = 8192
DEQUANT_CHUNK_ABOVE = 65536   # columns; below this the unchunked path is fine


def die(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def map_name(name: str) -> str:
    """Checkpoint name -> Qwen3_5ForCausalLM name.

    Agnes's `delta_attn.`/`global_attn.` infixes are renamed too (agnes.map_name,
    vLLM PR #57003's mapper). No Qwen3.8 name contains either infix, so on Qwen3.8
    this is exactly the old prefix strip.
    """
    name = _agnes.map_name(name)
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    return name


def shard_paths(snapshot: str) -> list[str]:
    """The shards this checkpoint's index names - NOT every *.safetensors present.

    A snapshot may carry more than one complete shard set. `urakozz/
    Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ@84575a1` (the gate checkpoint since
    2026-09-09) holds a stale 4-shard set of 1,986 tensors beside the live
    7-shard set of 2,384, overlapping in 1,983 names; its index names the
    7-shard set plus `model_extra_tensors.safetensors`, 2,399 tensors in 8
    files. Globbing opened both, which the duplicate check below caught -- but
    the danger the check cannot see is the other order: had the sets not
    overlapped, the oracle would have dequantised weights the ENGINE never
    loads, since src/loader reads the index. The index is the authority for
    both sides or the golden set means nothing.

    No index (a locally produced checkpoint, e.g. tools/quantize_qwen38_rtn.sh's
    output) falls back to the glob, which is what every set before 2026-09-09
    was dumped with.
    """
    index = os.path.join(snapshot, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    with open(index, encoding="utf-8") as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    paths = [os.path.join(snapshot, f) for f in files]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        die(f"index names {len(files)} shards; {len(missing)} are missing or are "
            f"dangling symlinks, first {missing[0]!r}. An HF-cache snapshot stores "
            f"its shards as symlinks into ../../blobs/, so mounting the snapshot "
            f"directory alone breaks them -- use ORACLE_MODEL (the whole cache is "
            f"mounted) rather than ORACLE_SNAP for a downloaded checkpoint.")
    print(f"shards: {len(files)} from model.safetensors.index.json "
          f"({len(glob.glob(os.path.join(snapshot, '*.safetensors')))} .safetensors present)",
          file=sys.stderr)
    return paths


def open_shards(snapshot: str) -> dict[str, object]:
    """Checkpoint name -> its shard's safe_open handle, duplicates refused."""
    handles = {}
    where: dict[str, object] = {}
    for path in shard_paths(snapshot):
        h = safe_open(path, framework="pt", device="cpu")
        handles[path] = h
        for key in h.keys():
            if key in where:
                die(f"duplicate tensor name across shards: {key}")
            where[key] = h
    return where


_FAST_CHECKED: set[str] = set()


def convert(where: dict, keys, group_size: int, sd: dict, fast: bool = False) -> tuple[int, str]:
    """Dequantise / copy `keys` (checkpoint names) into `sd` under model names.

    fast: stream.dequant_t (C++, bit-identical to dequant_gptq + transpose, spot-checked
    per call) - the streamed path re-dequantises every layer per forward.
    Returns (int4 tensors dequantised, the lm_head's kind if it was among them)."""
    n_quant = 0
    lm_head_kind = "absent"
    for key in keys:
        # `mtp.*` lives in its own shard (`model_extra_tensors.safetensors`) on
        # both checkpoints and is skipped by NAME, so the shard is opened, its
        # header read, and nothing in it is ever materialised. The duplicate
        # check above still covers it, which is the point of doing it before
        # the skip: an extra-tensors shard that overlapped a numbered one is
        # the failure this whole loop is shaped around (doc 03, the 9B).
        if key.startswith(SKIP_PREFIXES) or key.endswith(QUANT_SUFFIXES):
            continue
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            scales_key = base + ".scales"
            if scales_key not in where:
                die(f"{key} has no {scales_key}")
            qw = where[key].get_tensor(key)
            if fast:
                sd[map_name(base) + ".weight"] = _stream.dequant_t(
                    qw, where[scales_key].get_tensor(scales_key), group_size,
                    check=key not in _FAST_CHECKED)
                _FAST_CHECKED.add(key)
                del qw
            else:
                chunk = DEQUANT_CHUNK if qw.shape[1] > DEQUANT_CHUNK_ABOVE else 0
                # dequant_gptq returns [K, N] = [in, out]; nn.Linear wants [out, in].
                w = dequant_gptq(qw, where[scales_key].get_tensor(scales_key), group_size, chunk)
                del qw
                sd[map_name(base) + ".weight"] = w.t().contiguous()
                del w
            n_quant += 1
            if base == "lm_head":
                lm_head_kind = "int4 (dequantised here)"
        else:
            t = where[key].get_tensor(key)
            if t.dtype != torch.bfloat16:
                die(f"unquantised tensor {key} is {t.dtype}, expected bfloat16")
            sd[map_name(key)] = t
            if key == "lm_head.weight":
                lm_head_kind = "bf16 (as shipped)"
    return n_quant, lm_head_kind


def build_state_dict(snapshot: str, group_size: int) -> dict[str, torch.Tensor]:
    """Dequantise the checkpoint into a plain bf16 state dict for Qwen3_5ForCausalLM."""
    where = open_shards(snapshot)
    sd: dict[str, torch.Tensor] = {}
    n_quant, lm_head_kind = convert(where, sorted(where), group_size, sd)
    if lm_head_kind == "absent":
        die("no lm_head.weight and no lm_head.qweight - this checkpoint has no head")
    # Printed because it is the one tensor whose FORMAT differs between the two
    # checkpoints this oracle is run against, and a golden set is only
    # comparable to an engine that made the same choice.
    print(f"lm_head: {lm_head_kind}")
    print(f"state dict: {len(sd)} tensors, {n_quant} dequantised from int4, "
          f"{sum(t.numel() * t.element_size() for t in sd.values()) / 2**30:.2f} GiB")
    return sd


def stream_weights(model, snapshot: str, group_size: int, tc, keep: int = 0):
    """--stream: the layers' weights materialised per forward (stream.py), the rest resident.

    The same `convert` (dequant_gptq, map_name) as the resident path, called per layer."""
    import re
    where = open_shards(snapshot)
    n = tc.num_hidden_layers
    per: dict[int, list[str]] = {i: [] for i in range(n)}
    rest = []
    for key in sorted(where):
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
        (per[int(m.group(1))] if m else rest).append(key)
    resident: dict[str, torch.Tensor] = {}
    _, lm_head_kind = convert(where, rest, group_size, resident)
    if lm_head_kind == "absent":
        die("no lm_head.weight and no lm_head.qweight - this checkpoint has no head")
    print(f"lm_head: {lm_head_kind}")

    def layer_sd(i: int) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        convert(where, per[i], group_size, sd, fast=True)
        pre = f"model.layers.{i}."
        bad = [k for k in sd if not k.startswith(pre)]
        if bad:
            raise RuntimeError(f"layer {i}: {bad[:3]} outside {pre}")
        return {k[len(pre):]: v for k, v in sd.items()}

    pf = _stream.attach(model, list(model.model.layers), layer_sd, resident,
                        rebuild=[(model.model, "rotary_emb", lambda: type(model.model.rotary_emb)(tc))],
                        keep=range(min(keep, n)))
    print(f"streamed: {n - min(keep, n)} layers per forward, {min(keep, n)} kept once loaded, "
          f"{len(resident)} resident tensors "
          f"{sum(t.numel() * t.element_size() for t in resident.values()) / 2**30:.2f} GiB")
    return pf


def check_quant_config(snapshot: str) -> int:
    """The same two config vocabularies loader::QuantConfig::parse accepts.

    GPTQ writes `quant_method: gptq` and an explicit `desc_act`; auto-round's
    `auto_round:auto_gptq` writer emits `quant_method: auto-round`,
    `packing_format`, and NO `desc_act` key at all. Absence is that writer's
    spelling of false - and it is proven here the way the C++ loader proves it,
    from the shipped tensors: a permutation needs a `g_idx` to carry it, so no
    `g_idx` anywhere means there is nothing an unstated `desc_act: true` could
    have meant. Keep this check in step with src/loader/quant.cc; the two are
    graded against each other by the golden gate and by nothing else.

    It is deliberately NOT a full mirror. `extra_config`'s per-module rules are
    not read here: the C++ loader validates them (a module claiming int4 at
    anything but g64 symmetric throws) and it runs first on any checkpoint this
    oracle is pointed at, so a config that would fool this function has already
    failed the load. What this checks is what `dequant_gptq` itself assumes.
    """
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        q = json.load(f).get("quantization_config") or {}
    if q.get("bits") != 4 or not q.get("sym"):
        die(f"dequant_gptq assumes int4 symmetric; config says "
            f"bits={q.get('bits')} sym={q.get('sym')}")
    method, packing = q.get("quant_method"), q.get("packing_format")
    if method not in (None, "gptq", "auto-round"):
        die(f"quant_method {method!r} is neither 'gptq' nor 'auto-round'; dequant.py implements "
            f"the GPTQ v1 packing those two share")
    if packing not in (None, "auto_round:auto_gptq"):
        die(f"packing_format {packing!r} is not 'auto_round:auto_gptq'")
    if "desc_act" in q:
        if q["desc_act"]:
            die("desc_act is true; dequant_gptq assumes no activation-order permutation")
    else:
        n_gidx = 0
        for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as h:
                n_gidx += sum(1 for k in h.keys() if k.endswith(".g_idx"))
        if n_gidx:
            die(f"config declares no desc_act (auto-round's false) but the checkpoint ships "
                f"{n_gidx} g_idx tensors - refusing to infer that nothing is permuted")
        print(f"desc_act: absent, inferred false ({method}); 0 g_idx tensors shipped")
    gs = q.get("group_size")
    if gs != 64:
        die(f"group_size {gs} != 64 (dequant.py's convention, doc 02)")
    return gs


def dump_cache_layout(cache) -> None:
    """Print enough of the cache API to fix this script when transformers moves."""
    print(f"cache type: {type(cache)}", file=sys.stderr)
    print("cache attrs: " + ", ".join(a for a in dir(cache) if not a.startswith("__")), file=sys.stderr)
    layers = getattr(cache, "layers", None)
    if layers is None:
        print("cache has no .layers", file=sys.stderr)
        return
    for i in (0, 1):
        if i < len(layers):
            attrs = ", ".join(a for a in dir(layers[i]) if not a.startswith("__"))
            print(f"cache.layers[{i}] type: {type(layers[i])}\n  attrs: {attrs}", file=sys.stderr)


def read_state(layer, attr: str, name: str):
    """cache.layers[i].{conv,recurrent}_states is dict[int, Tensor] in transformers 5.15."""
    states = getattr(layer, attr, None)
    if states is None:
        raise AttributeError(f"{name}: cache layer has no `{attr}`")
    try:
        t = states[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise AttributeError(f"{name}: cannot index `{attr}` ({type(states)}): {exc}") from exc
    if t is None:
        raise AttributeError(f"{name}: `{attr}`[0] is None")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt", required=True, help="file of whitespace-separated token ids (tokenize.py encode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--mlp-in", help="also write the listed layers' MLP inputs here (spec 14 G1)")
    ap.add_argument("--mlp-in-layers", default="0,3,35,71")
    ap.add_argument("--stream", action="store_true",
                    help="layer-streamed weights (stream.py): one layer resident at a time, for "
                         "hosts with less RAM than the bf16 model (Agnes on the Mac)")
    ap.add_argument("--stream-keep", type=int, default=0,
                    help="with --stream: layers 0..N-1 stay resident once loaded (~0.83 GB each on Agnes)")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)  # nothing samples, but pin it anyway
    # The CPU pool this run actually got. torch reads OMP_NUM_THREADS at import,
    # so this is the honest number after any cap the wrapper applied - printed
    # because a dump's wall time is only interpretable next to it, and the box
    # is shared (tools/oracle/run_in_container.sh, ORACLE_THREADS).
    print(f"torch intra-op threads: {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    group_size = check_quant_config(args.snapshot)

    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    if not ids:
        die(f"{args.prompt} is empty")
    if len(ids) > args.max_prompt:
        die(f"prompt is {len(ids)} ids, --max-prompt is {args.max_prompt}")
    print(f"prompt: {len(ids)} ids from {args.prompt}")

    # 1. the model, from the config alone, on meta so we never hold two copies
    with open(os.path.join(args.snapshot, "config.json"), encoding="utf-8") as f:
        raw_cfg = json.load(f)
    agnes = _agnes.is_agnes(raw_cfg)
    if agnes:
        # Spec 14 §3.4: Agnes as Qwen3.5 + the parallel FFN (agnes.py). Not
        # AutoConfig: the checkpoint's config class is remote code.
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        kwargs, parallel = _agnes.translate_text_config(raw_cfg)
        tc = Qwen3_5TextConfig(**kwargs)
    else:
        cfg = AutoConfig.from_pretrained(args.snapshot)
        tc = cfg.get_text_config()
    if tc.tie_word_embeddings:
        die("tie_word_embeddings is true; this checkpoint ships a separate lm_head (doc 03)")
    tc._attn_implementation = "eager"  # fp32 softmax, the contract in doc 03
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
        if agnes:
            n_par = _agnes.attach_parallel_ffn(model, tc, parallel)
            print(f"agnes: parallel FFN ({parallel}) attached to {n_par} layers, unfolded")
    n_layers = tc.num_hidden_layers
    layer_types = list(tc.layer_types)
    gdn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(f"config: {n_layers} layers, {len(gdn_layers)} linear_attention, "
          f"{n_layers - len(gdn_layers)} full_attention, attn_impl={tc._attn_implementation}, "
          f"hidden {tc.hidden_size}, vocab {tc.vocab_size}")

    # 2. the weights
    pf = None
    if args.stream:
        # The strict key check up front, by name only (each layer's own load is strict too).
        where = open_shards(args.snapshot)
        names = {map_name(k[: -len(".qweight")] + ".weight" if k.endswith(".qweight") else k)
                 for k in where if not k.startswith(SKIP_PREFIXES) and not k.endswith(QUANT_SUFFIXES)}
        del where
        sd = dict.fromkeys(names)
    else:
        sd = build_state_dict(args.snapshot, group_size)
    want = set(model.state_dict().keys())
    missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
    if missing or unexpected:
        print(f"FATAL: strict load would fail: {len(missing)} missing, {len(unexpected)} unexpected",
              file=sys.stderr)
        for k in missing:
            print(f"  missing:    {k}", file=sys.stderr)
        for k in unexpected:
            print(f"  unexpected: {k}", file=sys.stderr)
        sys.exit(2)
    if args.stream:
        del sd
        pf = stream_weights(model, args.snapshot, group_size, tc, args.stream_keep)
    else:
        model.load_state_dict(sd, strict=True, assign=True)  # assign: no second copy
        del sd
        # meta construction left non-persistent buffers (rope inv_freq) empty; rebuild on CPU.
        model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                  if t.is_meta and not (args.stream and n.startswith("model.layers."))]
    if still_meta:
        die("meta tensors survived the load: " + ", ".join(still_meta[:20]))
    model.eval()
    print(f"loaded strict, {time.time() - t0:.1f}s elapsed, dtype {model.dtype}")

    # 3. prompt forward with per-layer hooks
    caps: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def fn(_mod, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t.detach()[0].to(torch.bfloat16).clone()
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "linear_attn"):
            mixer = layer.linear_attn
        elif hasattr(layer, "self_attn"):
            mixer = layer.self_attn
        else:
            die(f"layer {i} has neither linear_attn nor self_attn: {type(layer)}")
        handles += [layer.register_forward_hook(hook(f"resid.L{i}")),
                    mixer.register_forward_hook(hook(f"mixer.L{i}")),
                    layer.mlp.register_forward_hook(hook(f"mlp.L{i}"))]
    # Spec 14 G1: the MLP inputs of the listed layers (post_attention_layernorm's
    # output), for agnes_fold.py's proof on real activations. Prompt forward only.
    mlp_in: dict[str, torch.Tensor] = {}
    if args.mlp_in:
        def pre_hook(name: str):
            def fn(_mod, a):
                if name not in mlp_in:
                    mlp_in[name] = a[0].detach()[0].to(torch.bfloat16).clone()
            return fn
        for i in [int(x) for x in args.mlp_in_layers.split(",")]:
            handles.append(model.model.layers[i].mlp.register_forward_pre_hook(pre_hook(f"mlp_in.L{i}")))

    input_ids = torch.tensor([ids], dtype=torch.long)
    t_fwd = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    for h in handles:
        h.remove()
    print(f"prompt forward: {time.time() - t_fwd:.1f}s"
          + (f" (dequant wait {pf.seconds:.1f}s)" if pf else ""))

    # 4. GDN cache state, read straight after the prompt (decode mutates it in place)
    cache = out.past_key_values
    states: dict[str, torch.Tensor] = {}
    try:
        layers = cache.layers
        for i in gdn_layers:
            rec = read_state(layers[i], "recurrent_states", f"gdn_state.L{i}")
            conv = read_state(layers[i], "conv_states", f"conv_state.L{i}")
            states[f"gdn_state.L{i}"] = rec.detach()[0].to(torch.float32).clone()
            states[f"conv_state.L{i}"] = conv.detach()[0].clone()
    except (AttributeError, TypeError, IndexError) as exc:
        print(f"FATAL: cache layout is not what transformers 5.15 had: {exc}", file=sys.stderr)
        dump_cache_layout(cache)
        sys.exit(3)

    # 5. greedy decode through the same cache
    rows = [out.logits[0].to(torch.float32)]
    cur = rows[-1][-1]
    tokens: list[int] = []
    t_gen = time.time()
    for step in range(args.gen):
        nxt = int(torch.argmax(cur).item())
        tokens.append(nxt)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long),
                        past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0].to(torch.float32))
        cur = rows[-1][-1]
        if pf and step == 0:
            print(f"  first decode step {time.time() - t_gen:.1f}s", flush=True)
    print(f"greedy {args.gen} tokens: {time.time() - t_gen:.1f}s -> {tokens}"
          + (f" (dequant wait total {pf.seconds:.1f}s)" if pf else ""))

    tensors = dict(caps)
    tensors.update(states)
    tensors["logits"] = torch.cat(rows, dim=0).contiguous()
    tensors["tokens"] = torch.tensor(tokens, dtype=torch.int32)

    # self-checks
    if len(caps) != 3 * n_layers:
        die(f"hooks captured {len(caps)} activations, expected {3 * n_layers} (resid/mixer/mlp per layer)")
    if len(tokens) != args.gen:
        die(f"generated {len(tokens)} tokens, asked for {args.gen}")
    got_gdn = sorted(int(k.split(".L")[1]) for k in states if k.startswith("gdn_state."))
    if got_gdn != gdn_layers:
        die(f"gdn_state missing for layers {sorted(set(gdn_layers) - set(got_gdn))}")
    last = tensors[f"resid.L{n_layers - 1}"]
    if not torch.isfinite(last.to(torch.float32)).all():
        die(f"resid.L{n_layers - 1} has NaN/Inf - the dequantised weights are wrong")
    if len(set(tokens)) == 1:
        print(f"WARNING: degenerate continuation, every token is {tokens[0]}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.mlp_in:
        save_file(mlp_in, args.mlp_in, metadata={"snapshot": os.path.abspath(args.snapshot)})
        print(f"wrote {args.mlp_in}: {sorted(mlp_in)}")
    meta = {
        "snapshot": os.path.abspath(args.snapshot),
        "prompt_ids": " ".join(str(i) for i in ids),
        "n_prompt": str(len(ids)),
        "gen": str(args.gen),
        "attn_implementation": tc._attn_implementation,
        "group_size": str(group_size),
    }
    if agnes:   # Qwen3.8's metadata stays exactly as before
        meta["model"] = "agnes (parallel FFN unfolded, tools/oracle/agnes.py)"
    if args.stream:
        meta["weights"] = "layer-streamed (tools/oracle/stream.py)"
    save_file(tensors, args.out, metadata=meta)

    total = 0
    for name in sorted(tensors, key=lambda n: (n.split(".L")[0], int(n.split(".L")[1]) if ".L" in n else 0)):
        t = tensors[name]
        total += t.numel() * t.element_size()
        print(f"  {name:<18} {str(t.dtype).replace('torch.', ''):<8} {tuple(t.shape)}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux
    print(f"wrote {args.out}: {len(tensors)} tensors, {total / 2**20:.1f} MiB")
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB")


if __name__ == "__main__":
    main()
